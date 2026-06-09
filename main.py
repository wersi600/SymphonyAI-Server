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

# 로그 세팅 (Render 로그 모니터링용)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SymphonyAI")

app = FastAPI()

BASE_URL = "https://symphonyai-server.onrender.com"
HF_WORKER_URL = "https://wers600-symphonyai-audio-worker.hf.space"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/static", StaticFiles(directory="static"), name="static")

AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

# 주의: 대규모 서비스시 Redis나 DB로 교체 권장 (우선 유지하되 메모리 관리 강화)
jobs = {}

@app.get("/")
def read_root():
    return {
        "status": "ok",
        "message": "REMO / SymphonyAI Render 서버 정상 가동 중입니다."
    }

def make_waveform_peaks(file_path: str, target_peaks: int = 1200):
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

        return peaks[:target_peaks]
    except Exception as e:
        logger.error(f"파형 추출 실패: {str(e)}")
        return [0.05] * target_peaks

def hf_headers():
    return {}

def call_hf_extract_midi(file_path: str):
    try:
        midi_url = f"{HF_WORKER_URL}/extract-midi"
        logger.info(f"MIDI 추출 시작: {midi_url}")

        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, "audio/mpeg")}
            response = requests.post(
                midi_url,
                headers=hf_headers(),
                files=files,
                timeout=600
            )

        if response.status_code != 200:
            return {
                "status": "failed",
                "message": f"HF MIDI 오류 {response.status_code}: {response.text[:500]}",
                "midi_url": "",
                "bar_lines_ms": [],
                "bpm": 0
            }

        data = response.json()
        result_midi_url = data.get("midi_url", "")
        if result_midi_url.startswith("/"):
            result_midi_url = HF_WORKER_URL + result_midi_url

        return {
            "status": data.get("status", "failed"),
            "message": data.get("message", ""),
            "midi_url": result_midi_url,
            "bar_lines_ms": data.get("bar_lines_ms", []),
            "bpm": data.get("bpm", 0)
        }
    except Exception as e:
        return {
            "status": "failed",
            "message": f"MIDI 호출 예외: {type(e).__name__} / {str(e)}",
            "midi_url": "",
            "bar_lines_ms": [],
            "bpm": 0
        }

def call_hf_separate_stems(file_path: str):
    try:
        stem_url = f"{HF_WORKER_URL}/separate-stems"
        logger.info(f"Stem 분리 시작: {stem_url}")

        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, "audio/mpeg")}
            response = requests.post(
                stem_url,
                headers=hf_headers(),
                files=files,
                timeout=900
            )

        if response.status_code != 200:
            return {
                "status": "failed",
                "message": f"HF Stem 오류 {response.status_code}: {response.text[:1000]}",
                "vocal_url": "",
                "accompaniment_url": ""
            }

        data = response.json()
        vocal_url = data.get("vocal_url", "")
        if vocal_url.startswith("/"):
            vocal_url = HF_WORKER_URL + vocal_url

        accompaniment_url = data.get("accompaniment_url", "")
        if accompaniment_url.startswith("/"):
            accompaniment_url = HF_WORKER_URL + accompaniment_url

        return {
            "status": data.get("status", "failed"),
            "message": data.get("message", ""),
            "vocal_url": vocal_url,
            "accompaniment_url": accompaniment_url
        }
    except Exception as e:
        return {
            "status": "failed",
            "message": f"Stem 호출 예외: {type(e).__name__} / {str(e)}",
            "vocal_url": "",
            "accompaniment_url": ""
        }

def fake_bar_lines_ms(duration_ms: int, bpm: float = 120.0, beats_per_bar: int = 4):
    # 정밀한 부동소수점 계산 후 정수ms 변환
    bar_ms = (60000.0 / bpm) * beats_per_bar
    lines = []
    current = 0.0
    while current <= duration_ms:
        lines.append(int(current))
        current += bar_ms
    return lines

def analyze_job(job_id: str):
    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["message"] = "오디오 파형 분석 중입니다."
        jobs[job_id]["debug_step"] = "waveform"

        file_path = jobs[job_id]["file_path"]

        audio = AudioSegment.from_file(file_path)
        duration_ms = len(audio)
        waveform_peaks = make_waveform_peaks(file_path, target_peaks=1200)

        jobs[job_id]["duration_ms"] = duration_ms
        jobs[job_id]["waveform_peaks"] = waveform_peaks

        # 1. MIDI 추출 단계
        jobs[job_id]["message"] = "MIDI 추출 중입니다."
        jobs[job_id]["debug_step"] = "midi"
        midi_result = call_hf_extract_midi(file_path)

        jobs[job_id]["midi_status"] = midi_result["status"]
        jobs[job_id]["midi_message"] = midi_result["message"]

        if midi_result["status"] == "success":
            jobs[job_id]["midi_url"] = midi_result["midi_url"]
            jobs[job_id]["bar_lines_ms"] = midi_result["bar_lines_ms"]
            jobs[job_id]["bpm"] = midi_result["bpm"]
        else:
            jobs[job_id]["midi_url"] = ""
            jobs[job_id]["bar_lines_ms"] = fake_bar_lines_ms(duration_ms, bpm=120.0)
            jobs[job_id]["bpm"] = 120.0

        # 2. Stem 분리 단계
        jobs[job_id]["message"] = "보컬/반주 Stem 분리 중입니다."
        jobs[job_id]["debug_step"] = "stem"
        stem_result = call_hf_separate_stems(file_path)

        jobs[job_id]["stem_status"] = stem_result["status"]
        jobs[job_id]["stem_message"] = stem_result["message"]

        if stem_result["status"] == "success":
            jobs[job_id]["vocal_url"] = stem_result["vocal_url"]
            jobs[job_id]["accompaniment_url"] = stem_result["accompaniment_url"]
        else:
            jobs[job_id]["vocal_url"] = ""
            jobs[job_id]["accompaniment_url"] = ""

        # 3. 최종 완료 처리
        jobs[job_id]["status"] = "done"
        jobs[job_id]["debug_step"] = "done"
        if jobs[job_id]["vocal_url"] and jobs[job_id]["accompaniment_url"]:
            jobs[job_id]["message"] = "분석 완료 / Stem 플레이 준비 완료"
        else:
            jobs[job_id]["message"] = "분석 완료 / 일부 고음질 음원 유실 가능성 있음"

    except Exception as e:
        logger.error(f"비동기 태스크 내부 치명적 에러: {str(e)}")
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

    # 버퍼를 이용한 청크 단위 안전 저장 (Render 서버 OOM 방지)
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
        "bpm": 0.0,
        "vocal_url": "",
        "accompaniment_url": "",
        "midi_url": "",
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
            "message": "서버 휴면 전환으로 인해 작업 메모리가 만료되었습니다. 앱에서 파일을 다시 올려주세요.",
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
        "bpm": job["bpm"],
        "vocal_url": job["vocal_url"],
        "accompaniment_url": job["accompaniment_url"],
        "midi_url": job["midi_url"],
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
