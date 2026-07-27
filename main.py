from fastapi import FastAPI, Query, BackgroundTasks, File, UploadFile, Body
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from ace_step_routes import router as ace_step_router
import os
import uuid
import urllib.parse
import requests
import numpy as np
from pydub import AudioSegment
import imageio_ffmpeg
import shutil
import logging
import json
import tempfile
import hashlib
from datetime import datetime, timezone

import boto3
from botocore.config import Config
import psycopg
from psycopg.rows import dict_row

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("REMO-Backend")

app = FastAPI()
app.include_router(ace_step_router)

BASE_URL = os.environ.get(
    "BASE_URL",
    "https://symphonyai-server.onrender.com"
).rstrip("/")

HF_WORKER_URL = os.environ.get(
    "HF_WORKER_URL",
    "https://wers600-symphonyai-audio-worker.hf.space"
).rstrip("/")

HF_TOKEN = os.environ.get("HF_TOKEN", "")

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID", "").strip()
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID", "").strip()
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY", "").strip()
R2_BUCKET_NAME = os.environ.get("R2_BUCKET_NAME", "").strip()
R2_ENDPOINT_URL = os.environ.get(
    "R2_ENDPOINT_URL",
    f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com" if R2_ACCOUNT_ID else ""
).strip()

LOCAL_WORK_DIR = os.environ.get("LOCAL_WORK_DIR", "static")

if not os.path.exists(LOCAL_WORK_DIR):
    os.makedirs(LOCAL_WORK_DIR)

# 기존 /static 경로는 호환용으로 유지한다.
# 영구 저장은 Cloudflare R2를 사용한다.
app.mount("/static", StaticFiles(directory=LOCAL_WORK_DIR), name="static")
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def db_enabled():
    return bool(DATABASE_URL)


def r2_enabled():
    return bool(
        R2_ENDPOINT_URL
        and R2_ACCESS_KEY_ID
        and R2_SECRET_ACCESS_KEY
        and R2_BUCKET_NAME
    )


def hf_headers():
    if HF_TOKEN:
        return {"Authorization": f"Bearer {HF_TOKEN}"}
    return {}


def get_db():
    if not db_enabled():
        raise RuntimeError("DATABASE_URL 환경변수가 없습니다.")
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


def init_database():
    if not db_enabled():
        logger.warning("DATABASE_URL 없음: DB 영구저장 비활성")
        return

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS remo_jobs (
                    job_id TEXT PRIMARY KEY,
                    payload JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_remo_jobs_updated_at
                ON remo_jobs(updated_at DESC)
                """
            )
        conn.commit()

    logger.info("Neon PostgreSQL ready")


def save_job(job: dict):
    if not db_enabled():
        return

    job["updated_at"] = utc_now_iso()

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO remo_jobs(job_id, payload, created_at, updated_at)
                VALUES (%s, %s::jsonb, NOW(), NOW())
                ON CONFLICT (job_id)
                DO UPDATE SET
                    payload = EXCLUDED.payload,
                    updated_at = NOW()
                """,
                (
                    job["job_id"],
                    json.dumps(job, ensure_ascii=False, default=str),
                ),
            )
        conn.commit()


def load_job(job_id: str):
    if not db_enabled():
        return None

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT payload FROM remo_jobs WHERE job_id = %s",
                (job_id,),
            )
            row = cur.fetchone()

    if not row:
        return None

    payload = row["payload"]
    if isinstance(payload, str):
        payload = json.loads(payload)
    return payload


def get_r2_client():
    if not r2_enabled():
        raise RuntimeError("R2 환경변수가 완성되지 않았습니다.")

    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )


def sha256_file(path: str):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def upload_file_to_r2(
    local_path: str,
    storage_key: str,
    content_type: str = "application/octet-stream",
):
    client = get_r2_client()
    client.upload_file(
        local_path,
        R2_BUCKET_NAME,
        storage_key,
        ExtraArgs={"ContentType": content_type},
    )
    return storage_key


def download_r2_to_file(storage_key: str, local_path: str):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    get_r2_client().download_file(
        R2_BUCKET_NAME,
        storage_key,
        local_path,
    )
    return local_path


def signed_r2_url(storage_key: str, expires_seconds: int = 3600):
    if not storage_key or not r2_enabled():
        return ""

    return get_r2_client().generate_presigned_url(
        "get_object",
        Params={
            "Bucket": R2_BUCKET_NAME,
            "Key": storage_key,
        },
        ExpiresIn=expires_seconds,
    )


def normalize_job_urls(job: dict):
    url_fields = (
        "audio_url",
        "vocal_url",
        "accompaniment_url",
        "bass_url",
        "drums_url",
        "other_url",
        "midi_url",
        "raw_yourmt3_midi_url",
        "melody_midi_url",
        "accompaniment_midi_url",
        "mp3_url",
        "musicxml_url",
    )

    for field in url_fields:
        key = job.get(f"{field}_storage_key", "")
        if key:
            job[field] = signed_r2_url(key)

    return job


def ensure_local_source(job: dict):
    local_path = job.get("file_path", "")
    if local_path and os.path.exists(local_path):
        return local_path

    storage_key = job.get("audio_url_storage_key", "")
    if not storage_key:
        raise RuntimeError("원곡 R2 저장 키가 없습니다.")

    ext = os.path.splitext(job.get("file_name", ""))[1] or ".audio"
    local_path = os.path.join(
        LOCAL_WORK_DIR,
        f"{job['job_id']}_restored{ext}",
    )

    download_r2_to_file(storage_key, local_path)
    job["file_path"] = local_path
    save_job(job)
    return local_path


def remote_extension(url: str, default_ext: str):
    parsed = urllib.parse.urlparse(url)
    ext = os.path.splitext(parsed.path)[1]
    return ext or default_ext


def persist_remote_file(
    job_id: str,
    field_name: str,
    remote_url: str,
    default_ext: str,
):
    if not remote_url:
        return "", ""

    if not r2_enabled():
        return remote_url, ""

    ext = remote_extension(remote_url, default_ext)
    storage_key = f"projects/{job_id}/results/{field_name}{ext}"

    fd, temp_path = tempfile.mkstemp(
        prefix=f"remo_{field_name}_",
        suffix=ext,
    )
    os.close(fd)

    try:
        with requests.get(
            remote_url,
            headers=hf_headers(),
            stream=True,
            timeout=1800,
        ) as response:
            response.raise_for_status()
            content_type = response.headers.get(
                "Content-Type",
                "application/octet-stream",
            )

            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if chunk:
                        f.write(chunk)

        upload_file_to_r2(
            temp_path,
            storage_key,
            content_type,
        )

        return signed_r2_url(storage_key), storage_key
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def persist_result_fields(job: dict, result: dict, field_map: dict):
    for field_name, default_ext in field_map.items():
        remote_url = result.get(field_name, "")
        if not remote_url:
            continue

        signed_url, storage_key = persist_remote_file(
            job["job_id"],
            field_name,
            remote_url,
            default_ext,
        )

        job[field_name] = signed_url or remote_url
        if storage_key:
            job[f"{field_name}_storage_key"] = storage_key


@app.on_event("startup")
def startup_event():
    init_database()


@app.get("/")
def read_root():
    return {
        "status": "ok",
        "message": "REMO / SymphonyAI Render Backend Running",
        "database": "ready" if db_enabled() else "disabled",
        "r2": "ready" if r2_enabled() else "disabled",
    }


@app.get("/api/health/storage")
def storage_health():
    result = {
        "status": "success",
        "database": {
            "enabled": db_enabled(),
            "ok": False,
        },
        "r2": {
            "enabled": r2_enabled(),
            "ok": False,
        },
    }

    if db_enabled():
        try:
            with get_db() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1 AS ok")
                    result["database"]["ok"] = (
                        cur.fetchone()["ok"] == 1
                    )
        except Exception as e:
            result["database"]["error"] = str(e)

    if r2_enabled():
        try:
            get_r2_client().head_bucket(
                Bucket=R2_BUCKET_NAME
            )
            result["r2"]["ok"] = True
        except Exception as e:
            result["r2"]["error"] = str(e)

    if not result["database"]["ok"] or not result["r2"]["ok"]:
        result["status"] = "degraded"

    return result


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


def make_waveform_peaks(
    file_path: str,
    target_peaks: int = 12000
):
    try:
        audio = AudioSegment.from_file(
            file_path
        ).set_channels(1)

        samples = np.array(
            audio.get_array_of_samples()
        ).astype(np.float32)

        if len(samples) == 0:
            return []

        max_value = np.max(np.abs(samples))
        if max_value == 0:
            return [0.0] * target_peaks

        samples = samples / max_value
        chunk_size = max(
            1,
            len(samples) // target_peaks
        )

        peaks = []
        for i in range(0, len(samples), chunk_size):
            chunk = samples[i:i + chunk_size]
            if len(chunk) == 0:
                continue
            peaks.append(
                round(
                    float(np.max(np.abs(chunk))),
                    4,
                )
            )

        if len(peaks) < target_peaks:
            peaks.extend(
                [0.0] * (target_peaks - len(peaks))
            )

        return peaks[:target_peaks]
    except Exception as e:
        logger.error(
            "파형 추출 실패: %s / %s",
            type(e).__name__,
            e,
        )
        return [0.05] * target_peaks


def fake_bar_lines_ms(
    duration_ms: int,
    bpm: float = 120.0,
    beats_per_bar: int = 4,
):
    bpm = bpm if bpm and bpm > 0 else 120.0
    bar_ms = (60000.0 / bpm) * beats_per_bar

    lines = []
    current = 0.0
    while current <= duration_ms:
        lines.append(int(current))
        current += bar_ms

    return lines


def call_hf_extract_midi(
    file_path: str,
    cached_stems: dict = None,
):
    try:
        midi_url = f"{HF_WORKER_URL}/extract-midi"
        logger.info("HF 분리 MIDI 추출 요청: %s", midi_url)

        cached_stems = cached_stems or {}
        form_data = {
            "vocal_url": safe_str(
                cached_stems.get("vocal_url", ""),
                "",
            ),
            "accompaniment_url": safe_str(
                cached_stems.get("accompaniment_url", ""),
                "",
            ),
            "bass_url": safe_str(
                cached_stems.get("bass_url", ""),
                "",
            ),
            "drums_url": safe_str(
                cached_stems.get("drums_url", ""),
                "",
            ),
            "other_url": safe_str(
                cached_stems.get("other_url", ""),
                "",
            ),
        }

        cache_ready = all(
            form_data.get(key)
            for key in (
                "vocal_url",
                "bass_url",
                "drums_url",
                "other_url",
            )
        )

        logger.info(
            "HF MIDI 요청 / stem_cache=%s",
            "hit" if cache_ready else "miss",
        )

        with open(file_path, "rb") as f:
            response = requests.post(
                midi_url,
                headers=hf_headers(),
                files={
                    "file": (
                        os.path.basename(file_path),
                        f,
                        "audio/mpeg",
                    )
                },
                data=form_data,
                timeout=1800,
            )

        if response.status_code != 200:
            return {
                "status": "failed",
                "message": (
                    f"HF MIDI 오류 {response.status_code}: "
                    f"{response.text[:1000]}"
                ),
            }

        data = response.json()
        result = data.get("result") or data.get("data") or data

        status = safe_str(
            data.get("status") or result.get("status"),
            "failed",
        ).lower()

        message = safe_str(
            data.get("message") or result.get("message"),
            "",
        )

        full_url = normalize_hf_url(
            result.get("midi_url", "")
        )
        melody_url = normalize_hf_url(
            result.get("melody_midi_url", "")
        )
        acc_url = normalize_hf_url(
            result.get("accompaniment_midi_url", "")
        )

        if not full_url:
            full_url = melody_url or acc_url

        if not acc_url:
            acc_url = full_url

        success_like = (
            status in (
                "success",
                "done",
                "completed",
                "complete",
            )
            and bool(full_url)
        )

        return {
            "status": (
                "success" if success_like else status
            ),
            "message": message or (
                "YourMT3/BasicPitch MIDI 생성 완료"
                if success_like
                else "MIDI 생성 실패"
            ),
            "job_id": safe_str(
                result.get("job_id", ""),
                "",
            ),
            "midi_url": full_url,
            "raw_yourmt3_midi_url": normalize_hf_url(
                result.get(
                    "raw_yourmt3_midi_url",
                    "",
                )
            ),
            "melody_midi_url": melody_url,
            "accompaniment_midi_url": acc_url,
            "vocal_url": normalize_hf_url(
                result.get("vocal_url", "")
            ),
            "accompaniment_url": normalize_hf_url(
                result.get("accompaniment_url", "")
            ),
            "bass_url": normalize_hf_url(
                result.get("bass_url", "")
            ),
            "drums_url": normalize_hf_url(
                result.get("drums_url", "")
            ),
            "other_url": normalize_hf_url(
                result.get("other_url", "")
            ),
            "stem_engine": safe_str(
                result.get("stem_engine", ""),
                "",
            ),
            "mp3_url": normalize_hf_url(
                result.get("mp3_url", "")
            ),
            "bar_lines_ms": safe_bar_lines(
                result.get("bar_lines_ms", [])
            ),
            "bpm": safe_float(
                result.get("bpm", 0.0),
                0.0,
            ),
            "midi_bpm": safe_float(
                result.get("midi_bpm", 0.0),
                0.0,
            ),
            "bar_offset_ms": safe_int(
                result.get("bar_offset_ms", 0),
                0,
            ),
            "midi_engine": safe_str(
                result.get(
                    "midi_engine",
                    data.get("midi_engine", ""),
                ),
                "",
            ),
        }
    except Exception as e:
        return {
            "status": "failed",
            "message": (
                f"MIDI 호출 예외: "
                f"{type(e).__name__} / {e}"
            ),
        }


def call_hf_separate_stems(file_path: str):
    try:
        url = f"{HF_WORKER_URL}/separate-stems"

        with open(file_path, "rb") as f:
            response = requests.post(
                url,
                headers=hf_headers(),
                files={
                    "file": (
                        os.path.basename(file_path),
                        f,
                        "audio/mpeg",
                    )
                },
                timeout=1800,
            )

        if response.status_code != 200:
            return {
                "status": "failed",
                "message": (
                    f"HF Stem 오류 "
                    f"{response.status_code}: "
                    f"{response.text[:800]}"
                ),
            }

        data = response.json()
        result = data.get("result") or data.get("data") or data

        status = safe_str(
            data.get("status") or result.get("status"),
            "failed",
        ).lower()

        return {
            "status": (
                "success"
                if status in (
                    "success",
                    "done",
                    "completed",
                    "complete",
                )
                else status
            ),
            "message": safe_str(
                data.get("message")
                or result.get("message"),
                "",
            ),
            "vocal_url": normalize_hf_url(
                result.get("vocal_url", "")
            ),
            "accompaniment_url": normalize_hf_url(
                result.get(
                    "accompaniment_url",
                    "",
                )
            ),
            "bass_url": normalize_hf_url(
                result.get("bass_url", "")
            ),
            "drums_url": normalize_hf_url(
                result.get("drums_url", "")
            ),
            "other_url": normalize_hf_url(
                result.get("other_url", "")
            ),
            "stem_engine": safe_str(
                result.get("stem_engine", ""),
                "",
            ),
        }
    except Exception as e:
        return {
            "status": "failed",
            "message": (
                f"Stem 호출 예외: "
                f"{type(e).__name__} / {e}"
            ),
        }


def prepare_uploaded_job(job_id: str):
    job = load_job(job_id)
    if not job:
        logger.error("prepare job not found: %s", job_id)
        return

    try:
        job["status"] = "processing"
        job["message"] = "원곡을 준비하고 있습니다."
        job["debug_step"] = "waveform"
        save_job(job)

        file_path = ensure_local_source(job)
        audio = AudioSegment.from_file(file_path)

        job["duration_ms"] = len(audio)
        job["waveform_peaks"] = make_waveform_peaks(
            file_path,
            target_peaks=12000,
        )
        job["bpm"] = 120.0
        job["bar_lines_ms"] = []
        job["status"] = "done"
        job["message"] = (
            "원곡 준비 완료. 필요한 분석을 선택하세요."
        )
        job["debug_step"] = "source_ready"
        save_job(job)
    except Exception as e:
        job["status"] = "failed"
        job["message"] = (
            f"원곡 준비 오류: "
            f"{type(e).__name__} / {e}"
        )
        job["debug_step"] = "error"
        save_job(job)


def run_stem_job(job_id: str):
    job = load_job(job_id)
    if not job:
        logger.error("stem job not found: %s", job_id)
        return

    try:
        job["status"] = "processing"
        job["message"] = "Demucs 4Stem 분리 중입니다."
        job["debug_step"] = "stems"
        save_job(job)

        result = call_hf_separate_stems(
            ensure_local_source(job)
        )

        if result.get("status") != "success":
            raise RuntimeError(
                result.get("message")
                or "Stem 분리 실패"
            )

        persist_result_fields(
            job,
            result,
            {
                "vocal_url": ".wav",
                "accompaniment_url": ".wav",
                "bass_url": ".wav",
                "drums_url": ".wav",
                "other_url": ".wav",
            },
        )

        job["stem_engine"] = result.get(
            "stem_engine",
            "",
        )
        job["stem_status"] = "success"
        job["stem_message"] = (
            result.get("message")
            or "Demucs 4Stem 분리 완료"
        )
        job["status"] = "done"
        job["message"] = "스템 분리 완료"
        job["debug_step"] = "stems_done"
        save_job(job)
    except Exception as e:
        job["stem_status"] = "failed"
        job["stem_message"] = str(e)
        job["status"] = "failed"
        job["message"] = f"스템 분리 실패: {e}"
        job["debug_step"] = "stems_error"
        save_job(job)


def run_midi_job(job_id: str):
    job = load_job(job_id)
    if not job:
        logger.error("midi job not found: %s", job_id)
        return

    try:
        job["status"] = "processing"
        job["message"] = (
            "Official YourMT3+ MIDI 변환 중입니다."
        )
        job["debug_step"] = "midi"
        save_job(job)

        normalize_job_urls(job)

        cached_stems = {
            "vocal_url": job.get("vocal_url", ""),
            "accompaniment_url": job.get(
                "accompaniment_url",
                "",
            ),
            "bass_url": job.get("bass_url", ""),
            "drums_url": job.get("drums_url", ""),
            "other_url": job.get("other_url", ""),
        }

        result = call_hf_extract_midi(
            ensure_local_source(job),
            cached_stems=cached_stems,
        )

        if (
            result.get("status") != "success"
            or not result.get("midi_url")
        ):
            raise RuntimeError(
                result.get("message")
                or "MIDI 변환 실패"
            )

        persist_result_fields(
            job,
            result,
            {
                "midi_url": ".mid",
                "raw_yourmt3_midi_url": ".mid",
                "melody_midi_url": ".mid",
                "accompaniment_midi_url": ".mid",
                "vocal_url": ".wav",
                "accompaniment_url": ".wav",
                "bass_url": ".wav",
                "drums_url": ".wav",
                "other_url": ".wav",
                "mp3_url": ".mp3",
            },
        )

        for key in (
            "stem_engine",
            "bar_lines_ms",
            "bpm",
            "midi_engine",
            "bar_offset_ms",
        ):
            if result.get(key) not in (None, ""):
                job[key] = result.get(key)

        job["midi_status"] = "success"
        job["midi_message"] = (
            result.get("message")
            or "MIDI 변환 완료"
        )

        if (
            job.get("vocal_url")
            and job.get("accompaniment_url")
        ):
            job["stem_status"] = "success"
            job["stem_message"] = (
                "MIDI 변환 과정에서 4Stem도 준비되었습니다."
            )

        job["status"] = "done"
        job["message"] = "MIDI 변환 완료"
        job["debug_step"] = "midi_done"
        save_job(job)
    except Exception as e:
        job["midi_status"] = "failed"
        job["midi_message"] = str(e)
        job["status"] = "failed"
        job["message"] = f"MIDI 변환 실패: {e}"
        job["debug_step"] = "midi_error"
        save_job(job)


@app.post("/api/file/upload")
async def upload_audio_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    original_filename = urllib.parse.unquote(
        file.filename or "audio"
    )
    clean_filename = original_filename.replace(
        " ",
        "_",
    )

    job_id = str(uuid.uuid4())
    safe_filename = f"{job_id}_{clean_filename}"
    file_path = os.path.join(
        LOCAL_WORK_DIR,
        safe_filename,
    )

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    storage_key = ""
    audio_url = ""

    if r2_enabled():
        storage_key = (
            f"projects/{job_id}/source/{clean_filename}"
        )
        upload_file_to_r2(
            file_path,
            storage_key,
            file.content_type
            or "application/octet-stream",
        )
        audio_url = signed_r2_url(storage_key)
    else:
        safe_url_filename = urllib.parse.quote(
            safe_filename
        )
        audio_url = (
            f"{BASE_URL}/static/{safe_url_filename}"
        )

    job = {
        "job_id": job_id,
        "status": "queued",
        "message": "작업 대기 중입니다.",
        "debug_step": "queued",
        "file_name": clean_filename,
        "stored_file_name": safe_filename,
        "file_path": file_path,
        "audio_url": audio_url,
        "audio_url_storage_key": storage_key,
        "source_sha256": sha256_file(file_path),
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
        "stem_message": "",
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
    }

    save_job(job)
    background_tasks.add_task(
        prepare_uploaded_job,
        job_id,
    )

    return {
        "status": "success",
        "job_id": job_id,
        "message": (
            "업로드 완료. 원곡을 준비합니다."
        ),
        "audio_url": audio_url,
        "file_name": clean_filename,
    }


@app.get("/api/job/status")
def job_status(job_id: str = Query(...)):
    job = load_job(job_id)

    if not job:
        return {
            "status": "failed",
            "message": "작업을 찾지 못했습니다.",
            "job_id": job_id,
        }

    normalize_job_urls(job)

    midi_url = (
        job.get("midi_url", "")
        or job.get("melody_midi_url", "")
        or job.get(
            "accompaniment_midi_url",
            "",
        )
    )

    melody_midi_url = job.get(
        "melody_midi_url",
        "",
    )

    accompaniment_midi_url = (
        job.get("accompaniment_midi_url", "")
        or midi_url
    )

    bpm = safe_float(
        job.get("bpm", 0.0),
        120.0,
    )

    if bpm <= 0:
        bpm = 120.0

    return {
        "status": job.get("status", "failed"),
        "message": job.get("message", ""),
        "debug_step": job.get("debug_step", ""),
        "job_id": job_id,
        "file_name": job.get("file_name", ""),
        "audio_url": job.get("audio_url", ""),
        "duration_ms": safe_int(
            job.get("duration_ms", 0),
            0,
        ),
        "waveform_peaks": (
            job.get("waveform_peaks", [])
            or []
        ),
        "bar_lines_ms": (
            job.get("bar_lines_ms", [])
            or fake_bar_lines_ms(
                safe_int(
                    job.get("duration_ms", 0),
                    0,
                ),
                bpm,
            )
        ),
        "bar_offset_ms": safe_int(
            job.get("bar_offset_ms", 0),
            0,
        ),
        "bpm": bpm,
        "vocal_url": job.get("vocal_url", "") or "",
        "accompaniment_url": (
            job.get("accompaniment_url", "") or ""
        ),
        "bass_url": job.get("bass_url", "") or "",
        "drums_url": job.get("drums_url", "") or "",
        "other_url": job.get("other_url", "") or "",
        "stem_engine": (
            job.get("stem_engine", "") or ""
        ),
        "mp3_url": job.get("mp3_url", "") or "",
        "midi_url": midi_url or "",
        "raw_yourmt3_midi_url": (
            job.get(
                "raw_yourmt3_midi_url",
                "",
            )
            or ""
        ),
        "melody_midi_url": melody_midi_url or "",
        "accompaniment_midi_url": (
            accompaniment_midi_url or ""
        ),
        "musicxml_url": (
            job.get("musicxml_url", "") or ""
        ),
        "midi_status": (
            job.get("midi_status", "") or ""
        ),
        "midi_message": (
            job.get("midi_message", "") or ""
        ),
        "midi_engine": (
            job.get("midi_engine", "") or ""
        ),
        "stem_status": (
            job.get("stem_status", "") or ""
        ),
        "stem_message": (
            job.get("stem_message", "") or ""
        ),
        "stem_cache_ready": bool(
            job.get("vocal_url")
            and job.get("bass_url")
            and job.get("drums_url")
            and job.get("other_url")
        ),
        "storage_persistent": bool(
            job.get("audio_url_storage_key")
        ),
    }


@app.post("/api/job/stems")
async def start_stem_analysis(
    background_tasks: BackgroundTasks,
    payload: dict = Body(...),
):
    job_id = safe_str(
        payload.get("job_id"),
        "",
    )

    job = load_job(job_id) if job_id else None

    if not job:
        return {
            "status": "failed",
            "message": "유효한 작업을 찾지 못했습니다.",
        }

    if job.get("status") == "processing":
        return {
            "status": "failed",
            "message": "다른 작업이 진행 중입니다.",
        }

    job["status"] = "processing"
    job["message"] = "스템 분리 요청을 받았습니다."
    save_job(job)

    background_tasks.add_task(
        run_stem_job,
        job_id,
    )

    return {
        "status": "accepted",
        "job_id": job_id,
        "message": "Demucs 4Stem 분리를 시작합니다.",
    }


@app.post("/api/job/midi")
async def start_midi_analysis(
    background_tasks: BackgroundTasks,
    payload: dict = Body(...),
):
    job_id = safe_str(
        payload.get("job_id"),
        "",
    )

    job = load_job(job_id) if job_id else None

    if not job:
        return {
            "status": "failed",
            "message": "유효한 작업을 찾지 못했습니다.",
        }

    if job.get("status") == "processing":
        return {
            "status": "failed",
            "message": "다른 작업이 진행 중입니다.",
        }

    job["status"] = "processing"
    job["message"] = "MIDI 변환 요청을 받았습니다."
    save_job(job)

    background_tasks.add_task(
        run_midi_job,
        job_id,
    )

    return {
        "status": "accepted",
        "job_id": job_id,
        "message": (
            "Official YourMT3+ MIDI 변환을 시작합니다."
        ),
    }


@app.post("/api/project/render")
def render_project_artifact(payload: dict = Body(...)):
    try:
        logger.info(
            "Project render request / "
            "stem_mixer=%s / "
            "vocal_vol=%s / "
            "bass_vol=%s / "
            "drums_vol=%s / "
            "other_vol=%s / "
            "key_shift=%s",
            bool(
                payload.get(
                    "stem_mixer_enabled",
                    False,
                )
            ),
            payload.get("vocal_stem_volume", 1.0),
            payload.get("bass_stem_volume", 1.0),
            payload.get("drums_stem_volume", 1.0),
            payload.get("other_stem_volume", 1.0),
            payload.get("key_shift", 0),
        )

        response = requests.post(
            f"{HF_WORKER_URL}/render-project",
            headers={
                **hf_headers(),
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=1200,
        )

        if response.status_code != 200:
            return {
                "status": "failed",
                "message": (
                    f"HF 렌더링 오류 "
                    f"{response.status_code}: "
                    f"{response.text[:500]}"
                ),
                "download_url": "",
            }

        data = response.json()
        raw_url = normalize_hf_url(
            safe_str(
                data.get("download_url", ""),
                "",
            )
        )

        status = safe_str(
            data.get("status", "failed"),
            "failed",
        ).lower()

        if status == "success" and raw_url:
            job_id = safe_str(
                payload.get("job_id")
                or payload.get("project_id"),
                "",
            )

            if job_id and r2_enabled():
                signed_url, storage_key = (
                    persist_remote_file(
                        job_id,
                        "render_latest",
                        raw_url,
                        ".mp3",
                    )
                )

                job = load_job(job_id)
                if job:
                    job["mp3_url"] = (
                        signed_url or raw_url
                    )
                    job["mp3_url_storage_key"] = (
                        storage_key
                    )
                    save_job(job)

                raw_url = signed_url or raw_url

            return {
                "status": "success",
                "message": safe_str(
                    data.get(
                        "message",
                        "프로젝트 렌더링 완료",
                    ),
                    "프로젝트 렌더링 완료",
                ),
                "download_url": raw_url,
            }

        return {
            "status": "failed",
            "message": safe_str(
                data.get(
                    "message",
                    "HF 렌더링 결과 URL이 없습니다.",
                ),
                "HF 렌더링 결과 URL이 없습니다.",
            ),
            "download_url": "",
        }
    except Exception as e:
        return {
            "status": "failed",
            "message": (
                f"프로젝트 렌더링 호출 오류: "
                f"{type(e).__name__} / {e}"
            ),
            "download_url": "",
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
        "mute_melody": mute_melody,
    }


@app.get("/api/download")
def download_artifact(
    song: str,
    file_type: str = Query(...),
):
    return {
        "status": "success",
        "file_type": file_type,
        "download_url": (
            "https://symphony-ai-storage.com/"
            f"exports/{song}."
            f"{file_type if 'score' not in file_type else 'pdf'}"
        ),
    }


@app.get("/api/login")
def social_login(provider: str = Query(...)):
    if provider == "google":
        return RedirectResponse(
            url=(
                "https://accounts.google.com/o/oauth2/v2/auth"
                "?client_id=YOUR_GOOGLE_CLIENT_ID"
                "&redirect_uri="
                "https://symphonyai-server.onrender.com/"
                "api/login/callback/google"
                "&response_type=code"
                "&scope=email%20profile"
            )
        )

    if provider == "kakao":
        return RedirectResponse(
            url=(
                "https://kauth.kakao.com/oauth/authorize"
                "?client_id=YOUR_KAKAO_REST_KEY"
                "&redirect_uri="
                "https://symphonyai-server.onrender.com/"
                "api/login/callback/kakao"
                "&response_type=code"
            )
        )

    return {
        "status": "failed",
        "message": "지원하지 않는 로그인 제공업체입니다.",
    }


@app.get("/api/login/callback/{provider}")
def oauth_callback(provider: str, code: str):
    test_user_id = "symphony_user_777"
    test_email = (
        "symphony_user@gmail.com"
        if provider == "google"
        else "kakao_user@kakao.com"
    )

    app_deep_link_url = (
        "symphonyai://login_success"
        f"?user_id={test_user_id}"
        f"&email={test_email}"
    )

    return RedirectResponse(url=app_deep_link_url)
