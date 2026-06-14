from fastapi import FastAPI, Query, BackgroundTasks, File, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
import os
import uuid
import urllib.parse
import requests
import numpy as np
from pydub import AudioSegment
import imageio_ffmpeg
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("REMO-Backend")

app = FastAPI()

BASE_URL = "https://symphonyai-server.onrender.com"
HF_WORKER_URL = "https://wers600-symphonyai-audio-worker.hf.space"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/static", StaticFiles(directory="static"), name="static")
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

jobs = {}


@app.get("/")
def read_root():
    return {
        "status": "ok",
        "message": "REMO / SymphonyAI Render Backend Running"
    }


def hf_headers():
    if HF_TOKEN:
        return {"Authorization": f"Bearer {HF_TOKEN}"}
    return {}


def normalize_hf_url(url: str):
    if not url:
        return ""
    if url.startswith("/"):
        return HF_WORKER_URL + url
    return url


def make_waveform_peaks(file_path: str, target_peaks: int = 12000):
    try:
        audio = AudioSegment.from_file(file_path).set_channels(1)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)

        if len(samples) == 0:
            return []

        max_value = np.max(np.abs(samples))
        if max_value == 0:
            return [0.0] * target_peaks

        samples = samples / max_value
        chunk_size = max(1, len(samples) // target_peaks)

        peaks = []
        for i in range(0, len(samples), chunk_size):
            chunk = samples[i:i + chunk_size]
            if len(chunk) == 0:
                continue
            peaks.append(round(float(np.max(np.abs(chunk))), 4))

        if len(peaks) < target_peaks:
            peaks.extend([0.0] * (target_peaks - len(peaks)))

        return peaks[:target_peaks]
    except Exception as e:
        logger.error(f"파형 추출 실패: {type(e).__name__} / {str(e)}")
        return [0.05] * target_peaks


def fake_bar_lines_ms(duration_ms: int, bpm: float = 120.0, beats_per_bar: int = 4):
    bpm = bpm if bpm and bpm > 0 else 120.0
    bar_ms = (60000.0 / bpm) * beats_per_bar
    lines = []
    current = 0.0
    while current <= duration_ms:
        lines.append(int(current))
        current += bar_ms
    return lines


def call_hf_extract_midi(file_path: str):
    """
    HF /extract-midi 호출.
    이제 HF는 full.mid, melody.mid, midi_accompaniment.mid를 같이 반환합니다.
    """
    try:
        midi_url = f"{HF_WORKER_URL}/extract-midi"
        logger.info(f"HF 분리 MIDI 추출 요청: {midi_url}")

        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, "audio/mpeg")}
            response = requests.post(
                midi_url,
                headers=hf_headers(),
                files=files,
                timeout=1200
            )

        if response.status_code != 200:
            return {
                "status": "failed",
                "message": f"HF MIDI 오류 {response.status_code}: {response.text[:1000]}",
                "midi_url": "",
                "melody_midi_url": "",
                "accompaniment_midi_url": "",
                "vocal_url": "",
                "accompaniment_url": "",
                "bar_lines_ms": [],
                "bpm": 0.0,
                "bar_offset_ms": 0
            }

        data = response.json()
        return {
            "status": data.get("status", "failed"),
            "message": data.get("message", ""),
            "midi_url": normalize_hf_url(data.get("midi_url", "")),
            "melody_midi_url": normalize_hf_url(data.get("melody_midi_url", "")),
            "accompaniment_midi_url": normalize_hf_url(data.get("accompaniment_midi_url", "")),
            "vocal_url": normalize_hf_url(data.get("vocal_url", "")),
            "accompaniment_url": normalize_hf_url(data.get("accompaniment_url", "")),
            "bar_lines_ms": data.get("bar_lines_ms", []),
            "bpm": float(data.get("bpm", 0.0) or 0.0),
            "bar_offset_ms": int(data.get("bar_offset_ms", 0) or 0)
        }
    except Exception as e:
        return {
            "status": "failed",
            "message": f"MIDI 호출 예외: {type(e).__name__} / {str(e)}",
            "midi_url": "",
            "melody_midi_url": "",
            "accompaniment_midi_url": "",
            "vocal_url": "",
            "accompaniment_url": "",
            "bar_lines_ms": [],
            "bpm": 0.0,
            "bar_offset_ms": 0
        }


def analyze_job(job_id: str):
    try:
        job = jobs[job_id]
        job["status"] = "processing"
        job["message"] = "오디오 파형 분석 중입니다."
        job["debug_step"] = "waveform"

        file_path = job["file_path"]
        audio = AudioSegment.from_file(file_path)
        duration_ms = len(audio)
        waveform_peaks = make_waveform_peaks(file_path, target_peaks=12000)

        job["duration_ms"] = duration_ms
        job["waveform_peaks"] = waveform_peaks

        job["message"] = "Stem 기반 원곡/주멜로디/반주 MIDI 추출 중입니다."
        job["debug_step"] = "split_midi"
        midi_result = call_hf_extract_midi(file_path)

        job["midi_status"] = midi_result["status"]
        job["midi_message"] = midi_result["message"]
        job["bar_offset_ms"] = midi_result.get("bar_offset_ms", 0)

        if midi_result["status"] == "success" and midi_result.get("midi_url"):
            bpm = float(midi_result.get("bpm", 0.0) or 0.0)
            job["midi_url"] = midi_result.get("midi_url", "")
            job["melody_midi_url"] = midi_result.get("melody_midi_url", "")
            job["accompaniment_midi_url"] = midi_result.get("accompaniment_midi_url", "")
            job["vocal_url"] = midi_result.get("vocal_url", "")
            job["accompaniment_url"] = midi_result.get("accompaniment_url", "")
            job["bpm"] = bpm if bpm > 0 else 120.0
            job["bar_lines_ms"] = midi_result.get("bar_lines_ms", []) or fake_bar_lines_ms(duration_ms, job["bpm"])
        else:
            job["midi_url"] = ""
            job["melody_midi_url"] = ""
            job["accompaniment_midi_url"] = ""
            job["vocal_url"] = ""
            job["accompaniment_url"] = ""
            job["bpm"] = 120.0
            job["bar_lines_ms"] = fake_bar_lines_ms(duration_ms, bpm=120.0)

        # HF /extract-midi에서 stem wav도 같이 반환하므로 별도 /separate-stems 재호출은 하지 않습니다.
        if job["vocal_url"] and job["accompaniment_url"]:
            job["stem_status"] = "success"
            job["stem_message"] = "MIDI 추출 단계에서 Stem URL 함께 생성됨"
        else:
            job["stem_status"] = "failed"
            job["stem_message"] = "Stem URL 없음"

        job["status"] = "done"
        job["debug_step"] = "done"
        job["message"] = "분석 완료 / 원곡·주멜로디·반주 MIDI 준비 완료"

    except Exception as e:
        logger.error(f"비동기 태스크 내부 치명적 에러: {type(e).__name__} / {str(e)}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["message"] = f"분석 오류: {type(e).__name__} / {str(e)}"
        jobs[job_id]["debug_step"] = "error"


@app.post("/api/file/upload")
async def upload_audio_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    original_filename = urllib.parse.unquote(file.filename)
    clean_filename = original_filename.replace(" ", "_")

    job_id = str(uuid.uuid4())
    safe_filename = f"{job_id}_{clean_filename}"
    file_path = f"static/{safe_filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    safe_url_filename = urllib.parse.quote(safe_filename)
    full_audio_url = f"{BASE_URL}/static/{safe_url_filename}"

    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "message": "작업 대기 중입니다.",
        "debug_step": "queued",
        "file_name": clean_filename,
        "stored_file_name": safe_filename,
        "file_path": file_path,
        "audio_url": full_audio_url,
        "duration_ms": 0,
        "waveform_peaks": [],
        "bar_lines_ms": [],
        "bar_offset_ms": 0,
        "bpm": 0.0,
        "vocal_url": "",
        "accompaniment_url": "",
        "midi_url": "",
        "melody_midi_url": "",
        "accompaniment_midi_url": "",
        "musicxml_url": "",
        "midi_status": "",
        "midi_message": "",
        "stem_status": "",
        "stem_message": ""
    }

    background_tasks.add_task(analyze_job, job_id)

    return {
        "status": "success",
        "job_id": job_id,
        "message": "업로드 완료. 분석을 시작합니다.",
        "audio_url": full_audio_url,
        "file_name": clean_filename
    }


@app.get("/api/job/status")
def job_status(job_id: str = Query(...)):
    if job_id not in jobs:
        return {
            "status": "failed",
            "message": "서버 휴면 전환으로 작업 메모리가 만료되었습니다. 앱에서 파일을 다시 올려주세요.",
            "job_id": job_id
        }

    job = jobs[job_id]
    return {
        "status": job["status"],
        "message": job["message"],
        "debug_step": job.get("debug_step", ""),
        "job_id": job_id,
        "file_name": job["file_name"],
        "audio_url": job["audio_url"],
        "duration_ms": job["duration_ms"],
        "waveform_peaks": job["waveform_peaks"],
        "bar_lines_ms": job["bar_lines_ms"],
        "bar_offset_ms": job.get("bar_offset_ms", 0),
        "bpm": job["bpm"],
        "vocal_url": job["vocal_url"],
        "accompaniment_url": job["accompaniment_url"],
        "midi_url": job["midi_url"],
        "melody_midi_url": job.get("melody_midi_url", ""),
        "accompaniment_midi_url": job.get("accompaniment_midi_url", ""),
        "musicxml_url": job["musicxml_url"],
        "midi_status": job.get("midi_status", ""),
        "midi_message": job.get("midi_message", ""),
        "stem_status": job.get("stem_status", ""),
        "stem_message": job.get("stem_message", "")
    }


@app.post("/api/convert")
def convert_music(
    song: str = Query(...),
    prompt: str = Query(...),
    start_ms: int = Query(0),
    end_ms: int = Query(0),
    key_change: int = Query(0),
    remove_vocal: str = Query("N"),
    mute_melody: str = Query("N")
):
    return {
        "status": "processing",
        "message": "AI 재편집 작업이 큐에 등록되었습니다.",
        "song": song,
        "prompt": prompt,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "key_change": key_change,
        "remove_vocal": remove_vocal,
        "mute_melody": mute_melody
    }


@app.get("/api/download")
def download_artifact(song: str, file_type: str = Query(...)):
    return {
        "status": "success",
        "file_type": file_type,
        "download_url": f"https://symphony-ai-storage.com/exports/{song}.{file_type if 'score' not in file_type else 'pdf'}"
    }


@app.get("/api/login")
def social_login(provider: str = Query(...)):
    if provider == "google":
        google_oauth_url = (
            "https://accounts.google.com/o/oauth2/v2/auth"
            "?client_id=YOUR_GOOGLE_CLIENT_ID"
            "&redirect_uri=https://symphonyai-server.onrender.com/api/login/callback/google"
            "&response_type=code"
            "&scope=email%20profile"
        )
        return RedirectResponse(url=google_oauth_url)

    elif provider == "kakao":
        kakao_oauth_url = (
            "https://kauth.kakao.com/oauth/authorize"
            "?client_id=YOUR_KAKAO_REST_KEY"
            "&redirect_uri=https://symphonyai-server.onrender.com/api/login/callback/kakao"
            "&response_type=code"
        )
        return RedirectResponse(url=kakao_oauth_url)

    return {
        "status": "failed",
        "message": "지원하지 않는 로그인 제공업체입니다."
    }


@app.get("/api/login/callback/{provider}")
def oauth_callback(provider: str, code: str):
    test_user_id = "symphony_user_777"
    test_email = "symphony_user@gmail.com" if provider == "google" else "kakao_user@kakao.com"
    app_deep_link_url = f"symphonyai://login_success?user_id={test_user_id}&email={test_email}"
    return RedirectResponse(url=app_deep_link_url)
