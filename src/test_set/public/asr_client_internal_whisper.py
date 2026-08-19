from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

from asr_client_base import BaseASRClient, TranscriptionResult

DEFAULT_BASE_URL = "https://voice-staging.cyberbot.vn"


class InternalWhisperClient(BaseASRClient):
    """Client for the team's internal Whisper-based devboard API.

    Only calls POST /v1/asr/transcribe/file (same-language transcription).
    /v1/asr/translate/file is a different task (translation) and must not
    be voted alongside plain transcriptions in ROVER.
    """

    name = "internal_whisper"

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout_sec: int = 180,
        max_retries: int = 3,
        verify_ssl: bool = True,
        model_id: Optional[str] = None,
    ):
        self.transcribe_url = f"{base_url.rstrip('/')}/v1/asr/transcribe/file"
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.verify_ssl = verify_ssl
        self.model_id = model_id
        self.session = requests.Session()

    def transcribe(self, wav_path: Path, *, language: Optional[str] = None) -> TranscriptionResult:
        data = {}
        if language is not None:
            data["language"] = language
        if self.model_id is not None:
            data["model_id"] = self.model_id

        wav_bytes = wav_path.read_bytes()
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                start = time.perf_counter()
                r = self.session.post(
                    self.transcribe_url,
                    files={"file": (wav_path.name, wav_bytes, "audio/wav")},
                    data=data,
                    timeout=self.timeout_sec,
                    verify=self.verify_ssl,
                )
                latency = time.perf_counter() - start

                if r.status_code == 200:
                    payload = r.json()
                    return TranscriptionResult(
                        model_name=self.name,
                        text=payload.get("text", ""),
                        confidence=payload.get("mean_confidence_score"),
                        detected_language=payload.get("detected_text_language"),
                        latency_sec=latency,
                        error=None,
                        called_at=datetime.now(timezone.utc).isoformat(),
                        extra={"model_id": payload.get("model_id")},
                    )

                if r.status_code >= 500 or r.status_code == 429:
                    last_error = RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
                else:
                    last_error = RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
                    break  # client error, retrying won't help

            except Exception as e:  # network error, timeout, etc.
                last_error = e

            if attempt < self.max_retries - 1:
                time.sleep(2**attempt)

        return TranscriptionResult(
            model_name=self.name,
            text=None,
            confidence=None,
            detected_language=None,
            latency_sec=None,
            error=repr(last_error),
            called_at=datetime.now(timezone.utc).isoformat(),
        )
