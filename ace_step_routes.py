import json
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import boto3
import psycopg
import requests
from botocore.config import Config
from fastapi import APIRouter, BackgroundTasks, Body, Query
from psycopg.rows import dict_row

router = APIRouter(prefix="/api/ace-step", tags=["ACE-Step"])

ACE_STEP_WORKER_URL = os.environ.get("ACE_STEP_WORKER_URL", "").rstrip("/")
ACE_STEP_API_KEY = os.environ.get("ACE_STEP_API_KEY", "").strip()
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID", "").strip()
R2_ENDPOINT_URL = os.environ.get(
    "R2_ENDPOINT_URL",
    f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com" if R2_ACCOUNT_ID else "",
).strip()
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID", "").strip()
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY", "").strip()
R2_BUCKET_NAME = os.environ.get("R2_BUCKET_NAME", "").strip()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def db_connect():
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


def save_job(job: dict) -> None:
    job["updated_at"] = utc_now_iso()
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO remo_jobs(job_id, payload, created_at, updated_at)
                VALUES (%s, %s::jsonb, NOW(), NOW())
                ON CONFLICT (job_id)
                DO UPDATE SET payload = EXCLUDED.payload, updated_at = NOW()
                """,
                (job["job_id"], json.dumps(job, ensure_ascii=False, default=str)),
            )
        conn.commit()


def load_job(job_id: str):
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT payload FROM remo_jobs WHERE job_id = %s", (job_id,))
            row = cur.fetchone()
    if not row:
        return None
    payload = row["payload"]
    return json.loads(payload) if isinstance(payload, str) else payload


def r2_client():
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )


def signed_url(storage_key: str, expires_seconds: int = 3600) -> str:
    return r2_client().generate_presigned_url(
        "get_object",
        Params={"Bucket": R2_BUCKET_NAME, "Key": storage_key},
        ExpiresIn=expires_seconds,
    )


def worker_headers() -> dict:
    return {"Authorization": f"Bearer {ACE_STEP_API_KEY}"} if ACE_STEP_API_KEY else {}


def persist_audio(job_id: str, remote_url: str) -> tuple[str, str]:
    suffix = Path(remote_url.split("?", 1)[0]).suffix or ".wav"
    storage_key = f"projects/{job_id}/ace-step/master{suffix}"
    fd, temp_path = tempfile.mkstemp(prefix="remo_ace_", suffix=suffix)
    os.close(fd)
    try:
        with requests.get(remote_url, headers=worker_headers(), stream=True, timeout=1800) as response:
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "audio/wav")
            with open(temp_path, "wb") as target:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        target.write(chunk)
        r2_client().upload_file(
            temp_path,
            R2_BUCKET_NAME,
            storage_key,
            ExtraArgs={"ContentType": content_type},
        )
        return signed_url(storage_key), storage_key
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def run_generation(job_id: str) -> None:
    job = load_job(job_id)
    if not job:
        return
    try:
        if not ACE_STEP_WORKER_URL:
            raise RuntimeError("ACE_STEP_WORKER_URL 환경변수가 없습니다.")
        job.update(status="processing", message="ACE-Step에서 음악을 생성하고 있습니다.", debug_step="ace_generate")
        save_job(job)

        response = requests.post(
            f"{ACE_STEP_WORKER_URL}/generate",
            headers={**worker_headers(), "Content-Type": "application/json"},
            json={
                "job_id": job_id,
                "title": job["title"],
                "lyrics": job["lyrics"],
                "prompt": job["prompt"],
            },
            timeout=3600,
        )
        response.raise_for_status()
        data = response.json()
        result = data.get("result") or data.get("data") or data
        audio_url = result.get("audio_url") or result.get("wav_url") or result.get("mp3_url") or result.get("download_url")
        if not audio_url:
            raise RuntimeError(result.get("message") or "Worker 응답에 생성 음원 URL이 없습니다.")

        permanent_url, storage_key = persist_audio(job_id, audio_url)
        job.update(
            status="done",
            message="ACE-Step 생성 완료 / R2 저장 완료",
            debug_step="ace_done",
            audio_url=permanent_url,
            mp3_url=permanent_url,
            audio_url_storage_key=storage_key,
            duration_ms=int(result.get("duration_ms") or 0),
            ace_worker_result=result,
        )
        save_job(job)
    except Exception as exc:
        job.update(status="failed", message=f"ACE-Step 생성 실패: {type(exc).__name__} / {exc}", debug_step="ace_error")
        save_job(job)


@router.post("/generate")
def create_song(background_tasks: BackgroundTasks, payload: dict = Body(...)):
    title = str(payload.get("title") or "").strip()
    lyrics = str(payload.get("lyrics") or "").strip()
    prompt = str(payload.get("prompt") or "").strip()
    if not title or not lyrics or not prompt:
        return {"status": "failed", "message": "제목, 가사, 프롬프트를 모두 입력하세요."}
    if not DATABASE_URL:
        return {"status": "failed", "message": "DATABASE_URL 환경변수가 없습니다."}
    if not all([R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME]):
        return {"status": "failed", "message": "R2 환경변수가 완성되지 않았습니다."}

    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id,
        "job_type": "ace_step_create",
        "status": "queued",
        "message": "ACE-Step 생성 대기 중입니다.",
        "debug_step": "ace_queued",
        "title": title,
        "lyrics": lyrics,
        "prompt": prompt,
        "file_name": f"{title}.wav",
        "audio_url": "",
        "mp3_url": "",
        "duration_ms": 0,
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
    }
    save_job(job)
    background_tasks.add_task(run_generation, job_id)
    return {"status": "accepted", "job_id": job_id, "message": "ACE-Step 생성을 시작합니다."}


@router.get("/status")
def generation_status(job_id: str = Query(...)):
    job = load_job(job_id)
    if not job or job.get("job_type") != "ace_step_create":
        return {"status": "failed", "job_id": job_id, "message": "ACE-Step 작업을 찾지 못했습니다."}
    storage_key = job.get("audio_url_storage_key", "")
    if storage_key and job.get("status") == "done":
        job["audio_url"] = signed_url(storage_key)
        job["mp3_url"] = job["audio_url"]
    return {
        "status": job.get("status", "failed"),
        "job_id": job_id,
        "message": job.get("message", ""),
        "debug_step": job.get("debug_step", ""),
        "title": job.get("title", ""),
        "lyrics": job.get("lyrics", ""),
        "prompt": job.get("prompt", ""),
        "audio_url": job.get("audio_url", ""),
        "mp3_url": job.get("mp3_url", ""),
        "duration_ms": int(job.get("duration_ms") or 0),
    }
