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
ACE_STEP_DEFAULT_DURATION = max(210.0, float(os.environ.get("ACE_STEP_DEFAULT_DURATION", "210")))
ACE_STEP_MAX_DURATION = max(300.0, ACE_STEP_DEFAULT_DURATION, float(os.environ.get("ACE_STEP_MAX_DURATION", "300")))

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
        raise RuntimeError("ACE-Step Worker �묐떟�� JSON 媛앹껜媛� �꾨떃�덈떎.")
    return payload


def _safe_duration(value) -> float:
    try:
        duration = float(value)
    except Exception:
        duration = ACE_STEP_DEFAULT_DURATION
    return max(180.0, min(ACE_STEP_MAX_DURATION, duration))


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
    storage_key = str(worker.get("audio_storage_key") or "").strip()
    if storage_key:
        permanent_url = _storage.signed_url(storage_key)
    else:
        remote_url = _absolute_worker_url(worker.get("audio_url") or worker.get("mp3_url") or "")
        if not remote_url:
            raise RuntimeError("ACE-Step Worker �깃났 �묐떟�� �뚯썝 �꾩튂媛� �놁뒿�덈떎.")
        permanent_url, storage_key = _storage.persist_remote_file(
            job_id=job["job_id"],
            field_name="ace_step_master",
            remote_url=remote_url,
            default_ext=".mp3",
            headers=_worker_headers(),
        )

    job.update(
        status="done",
        message="ACE-Step �앹꽦 諛� MP3 �곴뎄 ���μ씠 �꾨즺�섏뿀�듬땲��.",
        debug_step="ace_done",
        audio_url=permanent_url,
        wav_url="",
        mp3_url=permanent_url,
        audio_url_storage_key=storage_key,
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
        job.update(status="queued", message=worker.get("message") or "ACE-Step �앹꽦 ��湲� 以묒엯�덈떎.", debug_step="ace_queued")
        return _save(job)
    if worker_status in {"running", "processing"}:
        job.update(status="processing", message=worker.get("message") or "ACE-Step�먯꽌 �뚯븙�� �앹꽦�섍퀬 �덉뒿�덈떎.", debug_step="ace_generating")
        return _save(job)
    if worker_status in {"success", "done", "completed"}:
        return _finish_from_worker(job, worker)
    if worker_status in {"failed", "error"}:
        job.update(status="failed", message=worker.get("message") or "ACE-Step Worker媛� �앹꽦 �ㅽ뙣瑜� 諛섑솚�덉뒿�덈떎.", debug_step="ace_error", ace_worker_result=worker)
        return _save(job)

    job.update(message=f"ACE-Step Worker �곹깭 �뺤씤 以�: {worker_status or 'unknown'}", debug_step="ace_status_unknown")
    return _save(job)


def _run_generation(job_id: str) -> None:
    job = _load(job_id)
    if not job:
        return
    try:
        worker = _submit_worker(job)
        worker_status = str(worker.get("status") or "").lower()
        if worker_status == "failed":
            raise RuntimeError(worker.get("message") or "ACE-Step Worker �붿껌�� 嫄곗젅�섏뿀�듬땲��.")

        worker_job_id = str(worker.get("job_id") or job_id)
        job.update(
            ace_worker_job_id=worker_job_id,
            status="queued",
            message=worker.get("message") or "ACE-Step �앹꽦 �붿껌�� �묒닔�덉뒿�덈떎.",
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

        job.update(status="failed", message=f"ACE-Step �앹꽦 �쒗븳�쒓컙 {ACE_STEP_TIMEOUT_SECONDS}珥덈� 珥덇낵�덉뒿�덈떎.", debug_step="ace_timeout")
        _save(job)
    except Exception as exc:
        job = _load(job_id) or job
        job.update(status="failed", message=f"ACE-Step �앹꽦 �ㅽ뙣: {type(exc).__name__} / {exc}", debug_step="ace_error")
        _save(job)


@router.post("/generate")
def create_song(background_tasks: BackgroundTasks, payload: dict = Body(...)):
    title = str(payload.get("title") or "").strip()
    lyrics = str(payload.get("lyrics") or "").strip()
    prompt = str(payload.get("prompt") or "").strip()

    if not title or not lyrics or not prompt:
        return {"status": "failed", "message": "�쒕ぉ, 媛���, �꾨＼�꾪듃瑜� 紐⑤몢 �낅젰�섏꽭��."}
    if not _db.enabled:
        return {"status": "failed", "message": "DATABASE_URL �섍꼍蹂��섍� �놁뒿�덈떎."}
    if not _storage.enabled:
        return {"status": "failed", "message": "R2 �섍꼍蹂��섍� �꾩꽦�섏� �딆븯�듬땲��."}
    if not ACE_STEP_WORKER_URL:
        return {"status": "failed", "message": "ACE_STEP_WORKER_URL �섍꼍蹂��섍� �놁뒿�덈떎."}

    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id,
        "job_type": "ace_step_create",
        "status": "queued",
        "message": "ACE-Step �앹꽦 ��湲� 以묒엯�덈떎.",
        "debug_step": "ace_queued",
        "title": title,
        "lyrics": lyrics,
        "prompt": prompt,
        "duration_seconds": _safe_duration(payload.get("duration", ACE_STEP_DEFAULT_DURATION)),
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
    return {"status": "accepted", "job_id": job_id, "message": "ACE-Step �앹꽦�� �쒖옉�⑸땲��."}


@router.get("/status")
def generation_status(job_id: str = Query(...)):
    job = _load(job_id)
    if not job or job.get("job_type") != "ace_step_create":
        return {"status": "failed", "job_id": job_id, "message": "ACE-Step �묒뾽�� 李얠� 紐삵뻽�듬땲��."}

    if job.get("status") in {"queued", "processing"} and job.get("ace_worker_job_id"):
        try:
            job = _sync_once(job)
        except Exception as exc:
            job["message"] = f"�곹깭 �뺤씤 �ъ떆�� �덉젙: {type(exc).__name__} / {exc}"
            _save(job)

    storage_key = job.get("audio_url_storage_key", "")
    if storage_key and job.get("status") == "done":
        fresh_url = _storage.signed_url(storage_key)
        job["audio_url"] = fresh_url
        job["wav_url"] = ""
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
        result.update(status="failed", message="ACE_STEP_WORKER_URL �섍꼍蹂��섍� �놁뒿�덈떎.")
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
