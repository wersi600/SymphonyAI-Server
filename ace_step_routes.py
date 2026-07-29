import json
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

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

# ACE-Step 1.5 official API defaults. Render environment variables can override them.
ACE_STEP_MODEL = os.environ.get("ACE_STEP_MODEL", "acestep-v15-turbo").strip()
ACE_STEP_LM_MODEL = os.environ.get("ACE_STEP_LM_MODEL", "acestep-5Hz-lm-0.6B").strip()
ACE_STEP_POLL_SECONDS = max(2, int(os.environ.get("ACE_STEP_POLL_SECONDS", "5")))
ACE_STEP_TIMEOUT_SECONDS = max(120, int(os.environ.get("ACE_STEP_TIMEOUT_SECONDS", "3600")))


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


def worker_headers(include_json: bool = False) -> dict:
    headers = {"User-Agent": "REMO-AceStep/1.0"}
    if ACE_STEP_API_KEY:
        headers["Authorization"] = f"Bearer {ACE_STEP_API_KEY}"
    if include_json:
        headers["Content-Type"] = "application/json"
    return headers


def checked_json(response: requests.Response) -> dict:
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict) and data.get("code") not in (None, 200):
        raise RuntimeError(data.get("error") or f"ACE-Step API code={data.get('code')}")
    return data


def absolute_worker_url(path_or_url: str) -> str:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        return path_or_url
    return urljoin(f"{ACE_STEP_WORKER_URL}/", path_or_url.lstrip("/"))


def persist_audio(job_id: str, remote_url: str) -> tuple[str, str, str]:
    parsed_path = urlparse(remote_url).path
    suffix = Path(parsed_path).suffix.lower() or ".mp3"
    if suffix not in {".mp3", ".wav", ".flac", ".aac", ".opus", ".m4a"}:
        suffix = ".mp3"

    storage_key = f"projects/{job_id}/ace-step/master{suffix}"
    fd, temp_path = tempfile.mkstemp(prefix="remo_ace_", suffix=suffix)
    os.close(fd)
    content_type = "audio/mpeg" if suffix == ".mp3" else "audio/wav"

    try:
        with requests.get(
            remote_url,
            headers=worker_headers(),
            stream=True,
            timeout=(30, 1800),
        ) as response:
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", content_type).split(";", 1)[0]
            with open(temp_path, "wb") as target:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        target.write(chunk)

        if os.path.getsize(temp_path) < 1024:
            raise RuntimeError("생성된 음원 파일이 비정상적으로 작습니다.")

        r2_client().upload_file(
            temp_path,
            R2_BUCKET_NAME,
            storage_key,
            ExtraArgs={"ContentType": content_type},
        )
        return signed_url(storage_key), storage_key, content_type
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def submit_official_task(job: dict) -> str:
    response = requests.post(
        f"{ACE_STEP_WORKER_URL}/release_task",
        headers=worker_headers(include_json=True),
        json={
            "prompt": job["prompt"],
            "lyrics": job["lyrics"],
            "thinking": True,
            "use_format": False,
            "model": ACE_STEP_MODEL,
            "lm_model_path": ACE_STEP_LM_MODEL,
            "lm_backend": "pt",
            "vocal_language": "ko",
            "inference_steps": 8,
            "batch_size": 1,
            "audio_format": "mp3",
            "task_type": "text2music",
            "use_random_seed": True,
        },
        timeout=(30, 180),
    )
    payload = checked_json(response)
    task = payload.get("data") or {}
    task_id = str(task.get("task_id") or "").strip()
    if not task_id:
        raise RuntimeError("ACE-Step /release_task 응답에 task_id가 없습니다.")
    return task_id


def query_official_task(worker_task_id: str) -> dict:
    response = requests.post(
        f"{ACE_STEP_WORKER_URL}/query_result",
        headers=worker_headers(include_json=True),
        json={"task_id_list": [worker_task_id]},
        timeout=(30, 180),
    )
    payload = checked_json(response)
    rows = payload.get("data") or []
    if not rows:
        return {"status": 0}
    return rows[0]


def parse_official_result(row: dict) -> dict:
    raw_result = row.get("result")
    if not raw_result:
        return {}
    if isinstance(raw_result, str):
        parsed = json.loads(raw_result)
    else:
        parsed = raw_result
    if isinstance(parsed, list):
        return parsed[0] if parsed else {}
    return parsed if isinstance(parsed, dict) else {}


def run_generation(job_id: str) -> None:
    job = load_job(job_id)
    if not job:
        return

    try:
        if not ACE_STEP_WORKER_URL:
            raise RuntimeError("ACE_STEP_WORKER_URL 환경변수가 없습니다.")

        job.update(
            status="processing",
            message="ACE-Step 작업을 전송하고 있습니다.",
            debug_step="ace_submit",
        )
        save_job(job)

        worker_task_id = submit_official_task(job)
        job.update(
            ace_worker_task_id=worker_task_id,
            message="ACE-Step에서 음악을 생성하고 있습니다.",
            debug_step="ace_generating",
        )
        save_job(job)

        deadline = time.monotonic() + ACE_STEP_TIMEOUT_SECONDS
        result = {}
        while time.monotonic() < deadline:
            row = query_official_task(worker_task_id)
            worker_status = int(row.get("status") or 0)

            if worker_status == 1:
                result = parse_official_result(row)
                break
            if worker_status == 2:
                result = parse_official_result(row)
                error_text = result.get("error") or result.get("message") or row.get("error")
                raise RuntimeError(error_text or "ACE-Step Worker가 생성 실패를 반환했습니다.")

            time.sleep(ACE_STEP_POLL_SECONDS)
        else:
            raise TimeoutError(f"ACE-Step 생성 제한시간 {ACE_STEP_TIMEOUT_SECONDS}초를 초과했습니다.")

        file_value = str(result.get("file") or "").strip()
        if not file_value:
            raise RuntimeError("ACE-Step 결과에 음원 file 경로가 없습니다.")

        remote_audio_url = absolute_worker_url(file_value)
        permanent_url, storage_key, content_type = persist_audio(job_id, remote_audio_url)

        metas = result.get("metas") if isinstance(result.get("metas"), dict) else {}
        duration_seconds = float(metas.get("duration") or 0)
        job.update(
            status="done",
            message="ACE-Step 생성 및 R2 저장이 완료되었습니다.",
            debug_step="ace_done",
            audio_url=permanent_url,
            mp3_url=permanent_url,
            audio_url_storage_key=storage_key,
            audio_content_type=content_type,
            duration_ms=max(0, int(duration_seconds * 1000)),
            ace_worker_result=result,
        )
        save_job(job)

    except Exception as exc:
        job.update(
            status="failed",
            message=f"ACE-Step 생성 실패: {type(exc).__name__} / {exc}",
            debug_step="ace_error",
        )
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
    if not ACE_STEP_WORKER_URL:
        return {"status": "failed", "message": "ACE_STEP_WORKER_URL 환경변수가 없습니다."}

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
        "file_name": f"{title}.mp3",
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
        fresh_url = signed_url(storage_key)
        job["audio_url"] = fresh_url
        job["mp3_url"] = fresh_url

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


@router.get("/health")
def ace_step_health():
    if not ACE_STEP_WORKER_URL:
        return {"status": "failed", "message": "ACE_STEP_WORKER_URL 환경변수가 없습니다."}
    try:
        response = requests.get(
            f"{ACE_STEP_WORKER_URL}/health",
            headers=worker_headers(),
            timeout=(15, 60),
        )
        payload = checked_json(response)
        return {"status": "success", "worker": payload}
    except Exception as exc:
        return {"status": "failed", "message": f"{type(exc).__name__}: {exc}"}
