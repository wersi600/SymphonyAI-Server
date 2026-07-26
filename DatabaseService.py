import json
import os
from datetime import datetime, timezone
from typing import Optional

import psycopg
from psycopg.rows import dict_row


class DatabaseService:
    def __init__(self) -> None:
        self.database_url = os.environ.get("DATABASE_URL", "").strip()

    @property
    def enabled(self) -> bool:
        return bool(self.database_url)

    def _connect(self):
        if not self.enabled:
            raise RuntimeError("DATABASE_URL 환경변수가 없습니다.")
        return psycopg.connect(self.database_url, row_factory=dict_row)

    def initialize(self) -> None:
        if not self.enabled:
            return
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS remo_jobs (
                        job_id TEXT PRIMARY KEY,
                        payload JSONB NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_remo_jobs_updated_at
                    ON remo_jobs(updated_at DESC)
                """)
            conn.commit()

    def health_check(self) -> bool:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 AS ok")
                row = cur.fetchone()
        return bool(row and row["ok"] == 1)

    @staticmethod
    def utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def save_job(self, job: dict) -> None:
        if not self.enabled:
            return
        job["updated_at"] = self.utc_now_iso()
        payload = json.dumps(job, ensure_ascii=False, default=str)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO remo_jobs(job_id, payload, created_at, updated_at)
                    VALUES (%s, %s::jsonb, NOW(), NOW())
                    ON CONFLICT (job_id)
                    DO UPDATE SET payload = EXCLUDED.payload, updated_at = NOW()
                """, (job["job_id"], payload))
            conn.commit()

    def load_job(self, job_id: str) -> Optional[dict]:
        if not self.enabled:
            return None
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT payload FROM remo_jobs WHERE job_id = %s", (job_id,))
                row = cur.fetchone()
        if not row:
            return None
        payload = row["payload"]
        return json.loads(payload) if isinstance(payload, str) else payload
