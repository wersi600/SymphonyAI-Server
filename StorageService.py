import os
import tempfile
import urllib.parse
from pathlib import Path
from typing import Tuple

import boto3
import imageio_ffmpeg
import requests
from botocore.config import Config
from pydub import AudioSegment

AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()


class StorageService:
    def __init__(self) -> None:
        account_id = os.environ.get("R2_ACCOUNT_ID", "").strip()
        self.endpoint_url = os.environ.get(
            "R2_ENDPOINT_URL",
            f"https://{account_id}.r2.cloudflarestorage.com" if account_id else "",
        ).strip()
        self.access_key_id = os.environ.get("R2_ACCESS_KEY_ID", "").strip()
        self.secret_access_key = os.environ.get("R2_SECRET_ACCESS_KEY", "").strip()
        self.bucket_name = os.environ.get("R2_BUCKET_NAME", "").strip()

    @property
    def enabled(self) -> bool:
        return bool(self.endpoint_url and self.access_key_id and self.secret_access_key and self.bucket_name)

    def _client(self):
        if not self.enabled:
            raise RuntimeError("R2 환경변수가 완성되지 않았습니다.")
        return boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name="auto",
            config=Config(signature_version="s3v4"),
        )

    def health_check(self) -> bool:
        self._client().head_bucket(Bucket=self.bucket_name)
        return True

    def upload_file(self, local_path: str, storage_key: str, content_type: str = "application/octet-stream") -> str:
        self._client().upload_file(
            local_path,
            self.bucket_name,
            storage_key,
            ExtraArgs={"ContentType": content_type},
        )
        return storage_key

    def download_file(self, storage_key: str, local_path: str) -> str:
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        self._client().download_file(self.bucket_name, storage_key, local_path)
        return local_path

    def signed_url(self, storage_key: str, expires_seconds: int = 3600) -> str:
        if not storage_key:
            return ""
        return self._client().generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket_name, "Key": storage_key},
            ExpiresIn=expires_seconds,
        )

    def persist_remote_file(
        self,
        job_id: str,
        field_name: str,
        remote_url: str,
        default_ext: str,
        headers: dict | None = None,
    ) -> Tuple[str, str]:
        if not remote_url:
            return "", ""
        if not self.enabled:
            return remote_url, ""

        ext = os.path.splitext(urllib.parse.urlparse(remote_url).path)[1] or default_ext
        storage_key = f"projects/{job_id}/results/{field_name}{ext}"
        fd, temp_path = tempfile.mkstemp(prefix=f"remo_{field_name}_", suffix=ext)
        os.close(fd)

        try:
            with requests.get(remote_url, headers=headers or {}, stream=True, timeout=1800) as response:
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "application/octet-stream")
                with open(temp_path, "wb") as f:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            f.write(chunk)

            self.upload_file(temp_path, storage_key, content_type)
            return self.signed_url(storage_key), storage_key
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

    def transcode_storage_audio_to_mp3(
        self,
        source_storage_key: str,
        target_storage_key: str,
        bitrate: str = "320k",
        title: str = "",
        artist: str = "REMO AI Music Studio",
    ) -> str:
        """Download an R2 audio object, encode a real MP3, and upload it."""
        if not source_storage_key or not target_storage_key:
            raise ValueError("원본과 대상 R2 저장 키가 필요합니다.")

        source_fd, source_path = tempfile.mkstemp(prefix="remo_source_", suffix=Path(source_storage_key).suffix or ".wav")
        mp3_fd, mp3_path = tempfile.mkstemp(prefix="remo_mp3_", suffix=".mp3")
        os.close(source_fd)
        os.close(mp3_fd)
        try:
            self.download_file(source_storage_key, source_path)
            tags = {"artist": artist}
            if title.strip():
                tags["title"] = title.strip()
            AudioSegment.from_file(source_path).export(
                mp3_path,
                format="mp3",
                bitrate=bitrate,
                tags=tags,
                id3v2_version="3",
            )
            if os.path.getsize(mp3_path) < 1024:
                raise RuntimeError("MP3 변환 결과가 비정상적으로 작습니다.")
            self.upload_file(mp3_path, target_storage_key, "audio/mpeg")
            return target_storage_key
        finally:
            for path in (source_path, mp3_path):
                try:
                    os.remove(path)
                except OSError:
                    pass

    def refresh_job_urls(self, job: dict) -> dict:
        fields = (
            "audio_url", "vocal_url", "accompaniment_url", "bass_url", "drums_url",
            "other_url", "midi_url", "raw_yourmt3_midi_url", "melody_midi_url",
            "accompaniment_midi_url", "mp3_url", "musicxml_url",
        )
        for field in fields:
            key = job.get(f"{field}_storage_key", "")
            if key:
                job[field] = self.signed_url(key)
        return job
