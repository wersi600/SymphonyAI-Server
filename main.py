from fastapi import FastAPI, Query, BackgroundTasks, File, UploadFile, Body
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
from ace_step_routes import router as ace_step_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("REMO-Backend")

app = FastAPI()
app.include_router(ace_step_router)

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


def safe_str(value, default=""):
    if value is None:
        return default
    text = str(value)
    if text.lower() in ("none", "null", "nan"):
        return default
    return text


def safe_float(value, default=0.0):
    try:
        v = float(value)
        if not np.isfinite(v):
            return default
        return v
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        return int(float(value))
    except Exception:
        return default


def safe_bar_lines(value):
    if not isinstance(value, list):
        return []
    bars = []
    for v in value:
        try:
            bars.append(int(float(v)))
        except Exception:
            pass
    return bars


def normalize_hf_url(url: str):
    url = safe_str(url, "")
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


def call_hf_extract_midi(file_path: str, cached_stems: dict = None):
    """
    HF /extract-midi 호출.
    BasicPitch와 YourMT3 성공 응답을 같은 구조로 정규화합니다.
    """
    try:
        midi_url = f"{HF_WORKER_URL}/extract-midi"
        logger.info(f"HF 분리 MIDI 추출 요청: {midi_url}")

        cached_stems = cached_stems or {}
        form_data = {
            "vocal_url": safe_str(cached_stems.get("vocal_url", ""), ""),
            "accompaniment_url": safe_str(cached_stems.get("accompaniment_url", ""), ""),
            "bass_url": safe_str(cached_stems.get("bass_url", ""), ""),
            "drums_url": safe_str(cached_stems.get("drums_url", ""), ""),
            "other_url": safe_str(cached_stems.get("other_url", ""), ""),
        }
        cache_ready = all(form_data.get(key) for key in ("vocal_url", "bass_url", "drums_url", "other_url"))
        logger.info(f"HF MIDI 요청 / stem_cache={'hit' if cache_ready else 'miss'}")

        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, "audio/mpeg")}
            response = requests.post(
                midi_url,
                headers=hf_headers(),
                files=files,
                data=form_data,
                timeout=1800
            )

        if response.status_code != 200:
            return {
                "status": "failed",
                "message": f"HF MIDI 오류 {response.status_code}: {response.text[:1000]}",
                "job_id": "",
                "midi_url": "",
                "melody_midi_url": "",
                "accompaniment_midi_url": "",
                "vocal_url": "",
                "accompaniment_url": "",
                "bass_url": "",
                "drums_url": "",
                "other_url": "",
                "stem_engine": "failed",
                "mp3_url": "",
                "bar_lines_ms": [],
                "bpm": 0.0,
                "midi_bpm": 0.0,
                "bar_offset_ms": 0,
                "midi_engine": "failed"
            }

        try:
            data = response.json()
        except Exception as e:
            return {
                "status": "failed",
                "message": f"HF JSON 파싱 오류: {type(e).__name__} / {response.text[:1000]}",
                "job_id": "",
                "midi_url": "",
                "melody_midi_url": "",
                "accompaniment_midi_url": "",
                "vocal_url": "",
                "accompaniment_url": "",
                "bass_url": "",
                "drums_url": "",
                "other_url": "",
                "stem_engine": "failed",
                "mp3_url": "",
                "bar_lines_ms": [],
                "bpm": 0.0,
                "midi_bpm": 0.0,
                "bar_offset_ms": 0,
                "midi_engine": "json_error"
            }

        # YourMT3/BasicPitch 모두 root 또는 data/result 중 하나에 결과가 올 수 있게 허용.
        result = data.get("result") or data.get("data") or data

        status = safe_str(data.get("status") or result.get("status"), "failed").lower()
        message = safe_str(data.get("message") or result.get("message"), "")

        full_url = normalize_hf_url(result.get("midi_url", ""))
        melody_url = normalize_hf_url(result.get("melody_midi_url", ""))
        acc_url = normalize_hf_url(result.get("accompaniment_midi_url", ""))

        # YourMT3에서 full.mid만 확실히 나오고 melody/accompaniment가 비는 경우도 앱이 정상 처리하게 보정.
        if not full_url:
            full_url = melody_url or acc_url
        # 주멜로디는 vocals.wav -> BasicPitch -> melody.mid 결과만 허용합니다.
        # 생성 실패 시 full MIDI로 대체하지 않습니다.
        if not acc_url:
            acc_url = full_url

        success_like = status in ("success", "done", "completed", "complete") and bool(full_url)

        return {
            "status": "success" if success_like else status,
            "message": message or ("YourMT3/BasicPitch MIDI 생성 완료" if success_like else "MIDI 생성 실패"),
            "job_id": safe_str(result.get("job_id", ""), ""),
            "midi_url": full_url,
            "raw_yourmt3_midi_url": normalize_hf_url(result.get("raw_yourmt3_midi_url", "")),
            "melody_midi_url": melody_url,
            "accompaniment_midi_url": acc_url,
            "vocal_url": normalize_hf_url(result.get("vocal_url", "")),
            "accompaniment_url": normalize_hf_url(result.get("accompaniment_url", "")),
            "bass_url": normalize_hf_url(result.get("bass_url", "")),
            "drums_url": normalize_hf_url(result.get("drums_url", "")),
            "other_url": normalize_hf_url(result.get("other_url", "")),
            "stem_engine": safe_str(result.get("stem_engine", ""), ""),
            "mp3_url": normalize_hf_url(result.get("mp3_url", "")),
            "bar_lines_ms": safe_bar_lines(result.get("bar_lines_ms", [])),
            "bpm": safe_float(result.get("bpm", 0.0), 0.0),
            "midi_bpm": safe_float(result.get("midi_bpm", 0.0), 0.0),
            "bar_offset_ms": safe_int(result.get("bar_offset_ms", 0), 0),
            "midi_engine": safe_str(result.get("midi_engine", data.get("midi_engine", "")), "")
        }
    except Exception as e:
        return {
            "status": "failed",
            "message": f"MIDI 호출 예외: {type(e).__name__} / {str(e)}",
            "job_id": "",
            "midi_url": "",
            "melody_midi_url": "",
            "accompaniment_midi_url": "",
            "vocal_url": "",
            "accompaniment_url": "",
            "bass_url": "",
            "drums_url": "",
            "other_url": "",
            "stem_engine": "failed",
            "mp3_url": "",
            "bar_lines_ms": [],
            "bpm": 0.0,
            "midi_bpm": 0.0,
            "bar_offset_ms": 0,
            "midi_engine": "exception"
        }




def call_hf_separate_stems(file_path: str):
    """HF Demucs-only endpoint. No MIDI work is started."""
    try:
        url = f"{HF_WORKER_URL}/separate-stems"
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, "audio/mpeg")}
            response = requests.post(url, headers=hf_headers(), files=files, timeout=1800)
        if response.status_code != 200:
            return {"status": "failed", "message": f"HF Stem 오류 {response.status_code}: {response.text[:800]}"}
        data = response.json()
        result = data.get("result") or data.get("data") or data
        status = safe_str(data.get("status") or result.get("status"), "failed").lower()
        return {
            "status": "success" if status in ("success", "done", "completed", "complete") else status,
            "message": safe_str(data.get("message") or result.get("message"), ""),
            "vocal_url": normalize_hf_url(result.get("vocal_url", "")),
            "accompaniment_url": normalize_hf_url(result.get("accompaniment_url", "")),
            "bass_url": normalize_hf_url(result.get("bass_url", "")),
            "drums_url": normalize_hf_url(result.get("drums_url", "")),
            "other_url": normalize_hf_url(result.get("other_url", "")),
            "stem_engine": safe_str(result.get("stem_engine", ""), ""),
        }
    except Exception as e:
        return {"status": "failed", "message": f"Stem 호출 예외: {type(e).__name__} / {e}"}


def prepare_uploaded_job(job_id: str):
    """Upload only: prepare waveform and duration, without Demucs or MIDI."""
    try:
        job = jobs[job_id]
        job["status"] = "processing"
        job["message"] = "원곡을 준비하고 있습니다."
        job["debug_step"] = "waveform"
        audio = AudioSegment.from_file(job["file_path"])
        job["duration_ms"] = len(audio)
        job["waveform_peaks"] = make_waveform_peaks(job["file_path"], target_peaks=12000)
        job["bpm"] = 120.0
        job["bar_lines_ms"] = []
        job["status"] = "done"
        job["message"] = "원곡 준비 완료. 필요한 분석을 선택하세요."
        job["debug_step"] = "source_ready"
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["message"] = f"원곡 준비 오류: {type(e).__name__} / {e}"
        jobs[job_id]["debug_step"] = "error"


def run_stem_job(job_id: str):
    job = jobs[job_id]
    try:
        job["status"] = "processing"
        job["message"] = "Demucs 4Stem 분리 중입니다."
        job["debug_step"] = "stems"
        result = call_hf_separate_stems(job["file_path"])
        if result.get("status") != "success":
            raise RuntimeError(result.get("message") or "Stem 분리 실패")
        for key in ("vocal_url", "accompaniment_url", "bass_url", "drums_url", "other_url", "stem_engine"):
            job[key] = result.get(key, "")
        job["stem_status"] = "success"
        job["stem_message"] = result.get("message") or "Demucs 4Stem 분리 완료"
        job["status"] = "done"
        job["message"] = "스템 분리 완료"
        job["debug_step"] = "stems_done"
    except Exception as e:
        job["stem_status"] = "failed"
        job["stem_message"] = str(e)
        job["status"] = "failed"
        job["message"] = f"스템 분리 실패: {e}"
        job["debug_step"] = "stems_error"


def run_midi_job(job_id: str):
    job = jobs[job_id]
    try:
        job["status"] = "processing"
        job["message"] = "Official YourMT3+ MIDI 변환 중입니다."
        job["debug_step"] = "midi"
        cached_stems = {
            "vocal_url": job.get("vocal_url", ""),
            "accompaniment_url": job.get("accompaniment_url", ""),
            "bass_url": job.get("bass_url", ""),
            "drums_url": job.get("drums_url", ""),
            "other_url": job.get("other_url", ""),
        }
        result = call_hf_extract_midi(job["file_path"], cached_stems=cached_stems)
        if result.get("status") != "success" or not result.get("midi_url"):
            raise RuntimeError(result.get("message") or "MIDI 변환 실패")
        for key in (
            "midi_url", "melody_midi_url", "accompaniment_midi_url", "vocal_url",
            "accompaniment_url", "bass_url", "drums_url", "other_url", "stem_engine",
            "mp3_url", "bar_lines_ms", "bpm", "midi_engine", "bar_offset_ms",
            "raw_yourmt3_midi_url"
        ):
            if key in result and result.get(key) not in (None, ""):
                job[key] = result.get(key)
        job["midi_status"] = "success"
        job["midi_message"] = result.get("message") or "MIDI 변환 완료"
        if job.get("vocal_url") and job.get("accompaniment_url"):
            job["stem_status"] = "success"
            job["stem_message"] = "MIDI 변환 과정에서 4Stem도 준비되었습니다."
        job["status"] = "done"
        job["message"] = "MIDI 변환 완료"
        job["debug_step"] = "midi_done"
    except Exception as e:
        job["midi_status"] = "failed"
        job["midi_message"] = str(e)
        job["status"] = "failed"
        job["message"] = f"MIDI 변환 실패: {e}"
        job["debug_step"] = "midi_error"


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

        job["midi_status"] = midi_result.get("status", "failed")
        job["midi_message"] = midi_result.get("message", "")
        job["midi_engine"] = midi_result.get("midi_engine", "")
        job["bar_offset_ms"] = midi_result.get("bar_offset_ms", 0)

        if midi_result.get("status") == "success" and midi_result.get("midi_url"):

            bpm = float(midi_result.get("bpm", 0.0) or 0.0)
            job["midi_url"] = midi_result.get("midi_url", "")
            job["melody_midi_url"] = midi_result.get("melody_midi_url", "")
            job["accompaniment_midi_url"] = midi_result.get("accompaniment_midi_url", "") or job["midi_url"]
            # melody_midi_url이 비어 있으면 그대로 유지합니다.
            # full MIDI를 주멜로디로 대체하지 않습니다.
            job["vocal_url"] = midi_result.get("vocal_url", "")
            job["accompaniment_url"] = midi_result.get("accompaniment_url", "")
            job["bass_url"] = midi_result.get("bass_url", "")
            job["drums_url"] = midi_result.get("drums_url", "")
            job["other_url"] = midi_result.get("other_url", "")
            job["stem_engine"] = midi_result.get("stem_engine", "")
            job["mp3_url"] = midi_result.get("mp3_url", "")
            job["bpm"] = bpm if bpm > 0 else 120.0
            job["bar_lines_ms"] = midi_result.get("bar_lines_ms", []) or fake_bar_lines_ms(duration_ms, job["bpm"])
        else:
            job["midi_url"] = ""
            job["melody_midi_url"] = ""
            job["accompaniment_midi_url"] = ""
            job["vocal_url"] = ""
            job["accompaniment_url"] = ""
            job["bass_url"] = ""
            job["drums_url"] = ""
            job["other_url"] = ""
            job["stem_engine"] = "failed"
            job["mp3_url"] = ""
            job["bpm"] = 120.0
            job["bar_lines_ms"] = fake_bar_lines_ms(duration_ms, bpm=120.0)

        # HF /extract-midi에서 stem wav도 같이 반환하므로 별도 /separate-stems 재호출은 하지 않습니다.
        if job["vocal_url"] and job["accompaniment_url"]:
            job["stem_status"] = "success"
            engine = job.get("stem_engine", "unknown")
            job["stem_message"] = f"Stem 생성 완료 ({engine})"
        else:
            job["stem_status"] = "failed"
            job["stem_message"] = "Stem URL 없음"

        job["status"] = "done"
        job["debug_step"] = "done"
        job["message"] = "분석 완료 / Demucs 4Stem·원곡·주멜로디·반주 MIDI 준비 완료"

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
        "bass_url": "",
        "drums_url": "",
        "other_url": "",
        "stem_engine": "",
        "midi_url": "",
        "raw_yourmt3_midi_url": "",
        "mp3_url": "",
        "melody_midi_url": "",
        "accompaniment_midi_url": "",
        "musicxml_url": "",
        "midi_status": "",
        "midi_message": "",
        "midi_engine": "",
        "stem_status": "",
        "stem_message": ""
    }

    background_tasks.add_task(prepare_uploaded_job, job_id)

    return {
        "status": "success",
        "job_id": job_id,
        "message": "업로드 완료. 원곡을 준비합니다.",
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
    midi_url = job.get("midi_url", "") or job.get("melody_midi_url", "") or job.get("accompaniment_midi_url", "")
    melody_midi_url = job.get("melody_midi_url", "")
    accompaniment_midi_url = job.get("accompaniment_midi_url", "") or midi_url
    bpm = safe_float(job.get("bpm", 0.0), 120.0)
    if bpm <= 0:
        bpm = 120.0

    return {
        "status": job.get("status", "failed"),
        "message": job.get("message", ""),
        "debug_step": job.get("debug_step", ""),
        "job_id": job_id,
        "file_name": job.get("file_name", ""),
        "audio_url": job.get("audio_url", ""),
        "duration_ms": safe_int(job.get("duration_ms", 0), 0),
        "waveform_peaks": job.get("waveform_peaks", []) or [],
        "bar_lines_ms": job.get("bar_lines_ms", []) or fake_bar_lines_ms(safe_int(job.get("duration_ms", 0), 0), bpm),
        "bar_offset_ms": safe_int(job.get("bar_offset_ms", 0), 0),
        "bpm": bpm,
        "vocal_url": job.get("vocal_url", "") or "",
        "accompaniment_url": job.get("accompaniment_url", "") or "",
        "bass_url": job.get("bass_url", "") or "",
        "drums_url": job.get("drums_url", "") or "",
        "other_url": job.get("other_url", "") or "",
        "stem_engine": job.get("stem_engine", "") or "",
        "mp3_url": job.get("mp3_url", "") or "",
        "midi_url": midi_url or "",
        "raw_yourmt3_midi_url": job.get("raw_yourmt3_midi_url", "") or "",
        "melody_midi_url": melody_midi_url or "",
        "accompaniment_midi_url": accompaniment_midi_url or "",
        "musicxml_url": job.get("musicxml_url", "") or "",
        "midi_status": job.get("midi_status", "") or "",
        "midi_message": job.get("midi_message", "") or "",
        "midi_engine": job.get("midi_engine", "") or "",
        "stem_status": job.get("stem_status", "") or "",
        "stem_message": job.get("stem_message", "") or "",
        "stem_cache_ready": bool(job.get("vocal_url") and job.get("bass_url") and job.get("drums_url") and job.get("other_url"))
    }




@app.post("/api/job/stems")
async def start_stem_analysis(background_tasks: BackgroundTasks, payload: dict = Body(...)):
    job_id = safe_str(payload.get("job_id"), "")
    if not job_id or job_id not in jobs:
        return {"status": "failed", "message": "유효한 작업을 찾지 못했습니다."}
    if jobs[job_id].get("status") == "processing":
        return {"status": "failed", "message": "다른 작업이 진행 중입니다."}
    jobs[job_id]["status"] = "processing"
    jobs[job_id]["message"] = "스템 분리 요청을 받았습니다."
    background_tasks.add_task(run_stem_job, job_id)
    return {"status": "accepted", "job_id": job_id, "message": "Demucs 4Stem 분리를 시작합니다."}


@app.post("/api/job/midi")
async def start_midi_analysis(background_tasks: BackgroundTasks, payload: dict = Body(...)):
    job_id = safe_str(payload.get("job_id"), "")
    if not job_id or job_id not in jobs:
        return {"status": "failed", "message": "유효한 작업을 찾지 못했습니다."}
    if jobs[job_id].get("status") == "processing":
        return {"status": "failed", "message": "다른 작업이 진행 중입니다."}
    jobs[job_id]["status"] = "processing"
    jobs[job_id]["message"] = "MIDI 변환 요청을 받았습니다."
    background_tasks.add_task(run_midi_job, job_id)
    return {"status": "accepted", "job_id": job_id, "message": "Official YourMT3+ MIDI 변환을 시작합니다."}


@app.post("/api/project/render")
def render_project_artifact(payload: dict = Body(...)):
    """Android 라이브러리/다운로드 요청을 HF Worker로 안전하게 프록시합니다."""
    try:
        logger.info(
            "Project render request / "
            f"stem_mixer={bool(payload.get('stem_mixer_enabled', False))} / "
            f"vocal_vol={payload.get('vocal_stem_volume', 1.0)} / "
            f"bass_vol={payload.get('bass_stem_volume', 1.0)} / "
            f"drums_vol={payload.get('drums_stem_volume', 1.0)} / "
            f"other_vol={payload.get('other_stem_volume', 1.0)} / "
            f"key_shift={payload.get('key_shift', 0)} / "
            "audio_drums_key=0 / midi_drums_key=0"
        )
        response = requests.post(
            f"{HF_WORKER_URL}/render-project",
            headers={**hf_headers(), "Content-Type": "application/json"},
            json=payload,
            timeout=1200
        )
        if response.status_code != 200:
            logger.error(f"HF /render-project failed: {response.status_code} / {response.text[:1000]}")
            return {
                "status": "failed",
                "message": f"HF 렌더링 오류 {response.status_code}: {response.text[:500]}",
                "download_url": ""
            }
        try:
            data = response.json()
        except Exception as e:
            logger.error(f"HF render JSON parse failed: {type(e).__name__} / {response.text[:1000]}")
            return {
                "status": "failed",
                "message": f"HF 렌더링 응답 해석 실패: {type(e).__name__}",
                "download_url": ""
            }
        raw_url = safe_str(data.get("download_url", ""), "")
        download_url = normalize_hf_url(raw_url)
        status = safe_str(data.get("status", "failed"), "failed").lower()
        if status == "success" and download_url:
            return {
                "status": "success",
                "message": safe_str(data.get("message", "프로젝트 렌더링 완료"), "프로젝트 렌더링 완료"),
                "download_url": download_url
            }
        return {
            "status": "failed",
            "message": safe_str(data.get("message", "HF 렌더링 결과 URL이 없습니다."), "HF 렌더링 결과 URL이 없습니다."),
            "download_url": ""
        }
    except Exception as e:
        logger.error(f"project render proxy exception: {type(e).__name__} / {str(e)}")
        return {
            "status": "failed",
            "message": f"프로젝트 렌더링 호출 오류: {type(e).__name__} / {str(e)}",
            "download_url": ""
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
