from fastapi import FastAPI, Query, BackgroundTasks, File, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
import os
import uuid
import urllib.parse
import time
import requests
import numpy as np
from pydub import AudioSegment
import imageio_ffmpeg

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
    return {"message": "SymphonyAI 서버 정상 가동 중입니다!"}


def make_waveform_peaks(file_path: str, target_peaks: int = 1200):
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_channels(1)

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
        peak = float(np.max(np.abs(chunk)))
        peaks.append(round(peak, 4))

    return peaks[:target_peaks]


def call_hf_extract_midi(file_path: str):
    headers = {}

    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    with open(file_path, "rb") as f:
        files = {
            "file": (
                os.path.basename(file_path),
                f,
                "audio/mpeg"
            )
        }

        response = requests.post(
            f"{HF_WORKER_URL}/extract-midi",
            headers=headers,
            files=files,
            timeout=300
        )

    if response.status_code != 200:
        return {
            "status": "failed",
            "message": f"HF 오류 {response.status_code}: {response.text}",
            "midi_url": "",
            "bar_lines_ms": [],
            "bpm": 0
        }

    data = response.json()

    midi_url = data.get("midi_url", "")
    if midi_url.startswith("/"):
        midi_url = HF_WORKER_URL + midi_url

    return {
        "status": data.get("status", "failed"),
        "message": data.get("message", ""),
        "midi_url": midi_url,
        "bar_lines_ms": data.get("bar_lines_ms", []),
        "bpm": data.get("bpm", 0)
    }


def fake_bar_lines_ms(duration_ms: int, bpm: int = 120, beats_per_bar: int = 4):
    bar_ms = int((60000 / bpm) * beats_per_bar)
    return list(range(0, duration_ms + bar_ms, bar_ms))


def analyze_job(job_id: str):
    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["message"] = "오디오 파형 분석 중입니다."

        file_path = jobs[job_id]["file_path"]

        audio = AudioSegment.from_file(file_path)
        duration_ms = len(audio)

        waveform_peaks = make_waveform_peaks(file_path, target_peaks=1200)

        jobs[job_id]["duration_ms"] = duration_ms
        jobs[job_id]["waveform_peaks"] = waveform_peaks

        jobs[job_id]["message"] = "MIDI 추출 중입니다."

        midi_result = call_hf_extract_midi(file_path)

        if midi_result["status"] == "success":
            jobs[job_id]["midi_url"] = midi_result["midi_url"]
            jobs[job_id]["bar_lines_ms"] = midi_result["bar_lines_ms"]
            jobs[job_id]["bpm"] = midi_result["bpm"]
            jobs[job_id]["message"] = "분석 완료"
        else:
            jobs[job_id]["midi_url"] = ""
            jobs[job_id]["bar_lines_ms"] = fake_bar_lines_ms(duration_ms)
            jobs[job_id]["bpm"] = 120
            jobs[job_id]["message"] = "파형 완료 / MIDI 실패: " + midi_result["message"]

        jobs[job_id]["status"] = "done"

        jobs[job_id]["vocal_url"] = ""
        jobs[job_id]["accompaniment_url"] = ""
        jobs[job_id]["musicxml_url"] = ""

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["message"] = str(e)


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

    with open(file_path, "wb+") as file_object:
        file_object.write(file.file.read())

    safe_url_filename = urllib.parse.quote(safe_filename)
    full_audio_url = f"{BASE_URL}/static/{safe_url_filename}"

    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "message": "작업 대기 중입니다.",
        "file_name": clean_filename,
        "stored_file_name": safe_filename,
        "file_path": file_path,
        "audio_url": full_audio_url,
        "duration_ms": 0,
        "waveform_peaks": [],
        "bar_lines_ms": [],
        "bpm": 0,
        "vocal_url": "",
        "accompaniment_url": "",
        "midi_url": "",
        "musicxml_url": ""
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
            "message": "job_id를 찾을 수 없습니다."
        }

    job = jobs[job_id]

    return {
        "status": job["status"],
        "message": job["message"],
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
        "musicxml_url": job["musicxml_url"]
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
def download_artifact(
    song: str,
    file_type: str = Query(...)
):
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

    return {"status": "failed", "message": "지원하지 않는 로그인 제공업체입니다."}


@app.get("/api/login/callback/{provider}")
def oauth_callback(provider: str, code: str):
    test_user_id = "symphony_user_777"
    test_email = "symphony_user@gmail.com" if provider == "google" else "kakao_user@kakao.com"

    app_deep_link_url = f"symphonyai://login_success?user_id={test_user_id}&email={test_email}"

    return RedirectResponse(url=app_deep_link_url)
