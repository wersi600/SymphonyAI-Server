import os
import threading
import time
import uuid
from urllib.parse import urljoin

import requests
from fastapi import APIRouter, BackgroundTasks, Body, Query

from DatabaseService import DatabaseService
from StorageService import StorageService

router = APIRouter(prefix="/api/ace-step", tags=["ACE-Step"])

ACE_STEP_WORKER_URL = os.environ.get("ACE_STEP_WORKER_URL", "").rstrip("/")
ACE_STEP_API_KEY = os.environ.get("ACE_STEP_API_KEY", "").strip()
ACE_STEP_POLL_SECONDS = max(2, int(os.environ.get("ACE_STEP_POLL_SECONDS", "5")))
ACE_STEP_TIMEOUT_SECONDS = max(120, int(os.environ.get("ACE_STEP_TIMEOUT_SECONDS", "3600")))

_db = DatabaseService()
_storage = StorageService()
_db_init_lock = threading.Lock()
_db_initialized = False


def _ensure_database() -> None:
    global _db_initialized
    if _db_initialized:
        return
    with _db_init_lock:
        if not _db_initialized:
            _db.initialize()
            _db_initialized = True


def _worker_headers() -> dict:
    headers = {"User-Agent": "REMO-AceStep/2.0"}
    if ACE_STEP_API_KEY:
        headers["Authorization"] = f"Bearer {ACE_STEP_API_KEY}"
    return headers


def _checked_json(response: requests.Response) -> dict:
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("ACE-Step Worker 응답이 JSON 객체가 아닙니다.")
    return payload


def _optional_duration(value):
    if value in (None, "", -1, "-1"):
        return None
    try:
        duration = float(value)
    except Exception:
        return None
    return max(10.0, min(600.0, duration))


def _absolute_worker_url(path_or_url: str) -> str:
    text = str(path_or_url or "").strip()
    if text.startswith("http://") or text.startswith("https://"):
        return text
    return urljoin(f"{ACE_STEP_WORKER_URL}/", text.lstrip("/"))


def _save(job: dict) -> dict:
    _ensure_database()
    _db.save_job(job)
    return job


def _load(job_id: str):
    _ensure_database()
    return _db.load_job(job_id)


def _submit_worker(job: dict) -> dict:
    response = requests.post(
        f"{ACE_STEP_WORKER_URL}/ace-step/start",
        headers=_worker_headers(),
        json={
            "job_id": job["job_id"],
            "title": job["title"],
            "lyrics": job["lyrics"],
            "prompt": job["prompt"],
            "duration": job["duration_seconds"],
            "seed": job.get("seed", -1),
            "bpm": job.get("bpm", 0),
        },
        timeout=(30, 180),
    )
    return _checked_json(response)


def _query_worker(worker_job_id: str) -> dict:
    response = requests.get(
        f"{ACE_STEP_WORKER_URL}/ace-step/status",
        headers=_worker_headers(),
        params={"job_id": worker_job_id},
        timeout=(30, 180),
    )
    return _checked_json(response)


def _finish_from_worker(job: dict, worker: dict) -> dict:
    wav_storage_key = str(worker.get("audio_storage_key") or "").strip()
    if wav_storage_key:
        wav_url = _storage.signed_url(wav_storage_key)
    else:
        remote_url = _absolute_worker_url(worker.get("audio_url") or worker.get("mp3_url") or "")
        if not remote_url:
            raise RuntimeError("ACE-Step Worker 성공 응답에 음원 위치가 없습니다.")
        wav_url, wav_storage_key = _storage.persist_remote_file(
            job_id=job["job_id"],
            field_name="ace_step_master",
            remote_url=remote_url,
            default_ext=".wav",
            headers=_worker_headers(),
        )

    mp3_storage_key = f"projects/{job['job_id']}/results/ace_step_master.mp3"
    _storage.transcode_storage_audio_to_mp3(
        wav_storage_key,
        mp3_storage_key,
        title=str(job.get("title") or "").strip(),
    )
    mp3_url = _storage.signed_url(mp3_storage_key)

    job.update(
        status="done",
        message="ACE-Step 생성 및 MP3 영구 저장이 완료되었습니다.",
        debug_step="ace_done",
        audio_url=mp3_url,
        wav_url=wav_url,
        mp3_url=mp3_url,
        audio_url_storage_key=mp3_storage_key,
        wav_url_storage_key=wav_storage_key,
        mp3_url_storage_key=mp3_storage_key,
        duration_ms=int(worker.get("duration_ms") or float(worker.get("duration") or 0) * 1000),
        sample_rate=int(worker.get("sample_rate") or 0),
        generation_seconds=float(worker.get("generation_seconds") or 0),
        ace_worker_result=worker,
    )
    return _save(job)


def _sync_once(job: dict) -> dict:
    worker_job_id = str(job.get("ace_worker_job_id") or job["job_id"])
    worker = _query_worker(worker_job_id)
    worker_status = str(worker.get("status") or "").lower()

    if worker_status in {"queued", "pending"}:
        job.update(status="queued", message=worker.get("message") or "ACE-Step 생성 대기 중입니다.", debug_step="ace_queued")
        return _save(job)
    if worker_status in {"running", "processing"}:
        job.update(status="processing", message=worker.get("message") or "ACE-Step에서 음악을 생성하고 있습니다.", debug_step="ace_generating")
        return _save(job)
    if worker_status in {"success", "done", "completed"}:
        return _finish_from_worker(job, worker)
    if worker_status in {"failed", "error"}:
        job.update(status="failed", message=worker.get("message") or "ACE-Step Worker가 생성 실패를 반환했습니다.", debug_step="ace_error", ace_worker_result=worker)
        return _save(job)

    job.update(message=f"ACE-Step Worker 상태 확인 중: {worker_status or 'unknown'}", debug_step="ace_status_unknown")
    return _save(job)


def _run_generation(job_id: str) -> None:
    job = _load(job_id)
    if not job:
        return
    try:
        worker = _submit_worker(job)
        worker_status = str(worker.get("status") or "").lower()
        if worker_status == "failed":
            raise RuntimeError(worker.get("message") or "ACE-Step Worker 요청이 거절되었습니다.")

        worker_job_id = str(worker.get("job_id") or job_id)
        job.update(
            ace_worker_job_id=worker_job_id,
            status="queued",
            message=worker.get("message") or "ACE-Step 생성 요청을 접수했습니다.",
            debug_step="ace_submitted",
        )
        _save(job)

        deadline = time.monotonic() + ACE_STEP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            job = _load(job_id) or job
            job = _sync_once(job)
            if job.get("status") in {"done", "failed"}:
                return
            time.sleep(ACE_STEP_POLL_SECONDS)

        job.update(status="failed", message=f"ACE-Step 생성 제한시간 {ACE_STEP_TIMEOUT_SECONDS}초를 초과했습니다.", debug_step="ace_timeout")
        _save(job)
    except Exception as exc:
        job = _load(job_id) or job
        job.update(status="failed", message=f"ACE-Step 생성 실패: {type(exc).__name__} / {exc}", debug_step="ace_error")
        _save(job)


@router.post("/generate")
def create_song(background_tasks: BackgroundTasks, payload: dict = Body(...)):
    title = str(payload.get("title") or "")
    lyrics = str(payload.get("lyrics") or "")
    prompt = str(payload.get("prompt") or "")

    if not title.strip() or not lyrics.strip() or not prompt.strip():
        return {"status": "failed", "message": "제목, 가사, 프롬프트를 모두 입력하세요."}
    if not _db.enabled:
        return {"status": "failed", "message": "DATABASE_URL 환경변수가 없습니다."}
    if not _storage.enabled:
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
        "duration_seconds": _optional_duration(payload.get("duration")),
        "seed": int(payload.get("seed", -1) or -1),
        "bpm": int(payload.get("bpm", 0) or 0),
        "audio_url": "",
        "wav_url": "",
        "mp3_url": "",
        "duration_ms": 0,
        "created_at": DatabaseService.utc_now_iso(),
    }
    _save(job)
    background_tasks.add_task(_run_generation, job_id)
    return {"status": "accepted", "job_id": job_id, "message": "ACE-Step 생성을 시작합니다."}


@router.get("/status")
def generation_status(job_id: str = Query(...)):
    job = _load(job_id)
    if not job or job.get("job_type") != "ace_step_create":
        return {"status": "failed", "job_id": job_id, "message": "ACE-Step 작업을 찾지 못했습니다."}

    if job.get("status") in {"queued", "processing"} and job.get("ace_worker_job_id"):
        try:
            job = _sync_once(job)
        except Exception as exc:
            job["message"] = f"상태 확인 재시도 예정: {type(exc).__name__} / {exc}"
            _save(job)

    if job.get("status") == "done":
        job = _storage.refresh_job_urls(job)

    return {
        "status": job.get("status", "failed"),
        "job_id": job_id,
        "message": job.get("message", ""),
        "debug_step": job.get("debug_step", ""),
        "title": job.get("title", ""),
        "lyrics": job.get("lyrics", ""),
        "prompt": job.get("prompt", ""),
        "audio_url": job.get("audio_url", ""),
        "wav_url": job.get("wav_url", ""),
        "mp3_url": job.get("mp3_url", ""),
        "duration_ms": int(job.get("duration_ms") or 0),
        "sample_rate": int(job.get("sample_rate") or 0),
    }


@router.get("/health")
def ace_step_health():
    result = {
        "status": "success",
        "database": "configured" if _db.enabled else "missing",
        "storage": "configured" if _storage.enabled else "missing",
        "worker_url": ACE_STEP_WORKER_URL,
    }
    if not ACE_STEP_WORKER_URL:
        result.update(status="failed", message="ACE_STEP_WORKER_URL 환경변수가 없습니다.")
        return result
    try:
        response = requests.get(ACE_STEP_WORKER_URL, headers=_worker_headers(), timeout=(15, 60))
        result["worker"] = _checked_json(response)
        if _db.enabled:
            _ensure_database()
            result["database_health"] = _db.health_check()
        if _storage.enabled:
            result["storage_health"] = _storage.health_check()
        return result
    except Exception as exc:
        result.update(status="failed", message=f"{type(exc).__name__}: {exc}")
        return result
