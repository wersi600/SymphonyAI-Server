from fastapi import FastAPI, Query, BackgroundTasks, File, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
import os
import urllib.parse
import numpy as np
from pydub import AudioSegment
import imageio_ffmpeg

app = FastAPI()

if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/static", StaticFiles(directory="static"), name="static")

# imageio-ffmpeg가 제공하는 ffmpeg 실행파일을 pydub에 연결
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()


@app.get("/")
def read_root():
    return {"message": "SymphonyAI 서버 정상 가동 중입니다!"}


def make_waveform_peaks(file_path: str, target_peaks: int = 1200):
    """
    실제 오디오를 디코딩해서 사운드포지/캡컷식 파형용 피크 데이터 생성.
    반환값: 0.0 ~ 1.0 사이의 float 리스트
    """
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


@app.post("/api/file/upload")
async def upload_audio_file(file: UploadFile = File(...)):
    original_filename = urllib.parse.unquote(file.filename)
    clean_filename = original_filename.replace(" ", "_")

    file_path = f"static/{clean_filename}"

    with open(file_path, "wb+") as file_object:
        file_object.write(file.file.read())

    safe_filename = urllib.parse.quote(clean_filename)
    full_audio_url = f"https://symphonyai-server.onrender.com/static/{safe_filename}"

    waveform_peaks = []
    waveform_status = "success"

    try:
        waveform_peaks = make_waveform_peaks(file_path, target_peaks=1200)
    except Exception as e:
        waveform_status = f"failed: {str(e)}"

    return {
        "status": "success",
        "audio_url": full_audio_url,
        "file_name": clean_filename,
        "waveform_status": waveform_status,
        "waveform_peaks": waveform_peaks
    }


@app.get("/api/waveform")
def get_waveform(file_name: str = Query(...)):
    clean_filename = urllib.parse.unquote(file_name)
    file_path = f"static/{clean_filename}"

    if not os.path.exists(file_path):
        return {
            "status": "failed",
            "message": "파일을 찾을 수 없습니다.",
            "waveform_peaks": []
        }

    try:
        peaks = make_waveform_peaks(file_path, target_peaks=1200)
        return {
            "status": "success",
            "file_name": clean_filename,
            "waveform_peaks": peaks
        }
    except Exception as e:
        return {
            "status": "failed",
            "message": str(e),
            "waveform_peaks": []
        }


@app.post("/api/convert")
def convert_music(
    song: str = Query(...),
    prompt: str = Query(...),
    start_ms: int = Query(0),
    end_ms: int = Query(0),
    key_change: int = Query(0),
    remove_vocal: str = Query("N"),
    mute_melody: str = Query("N"),
    background_tasks: BackgroundTasks = BackgroundTasks()
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


@app.post("/api/vocal-remove")
def vocal_remove(
    song: str = Query(...),
    remove_vocal: str = Query("Y"),
    mute_melody: str = Query("N")
):
    return {
        "status": "success",
        "message": f"보컬 처리 완료 (remove_vocal={remove_vocal}, mute_melody={mute_melody})"
    }


@app.get("/api/sync")
def sync_player(song: str, current_time: float):
    return {
        "song": song,
        "server_sync_time": current_time,
        "status": "synchronized"
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
