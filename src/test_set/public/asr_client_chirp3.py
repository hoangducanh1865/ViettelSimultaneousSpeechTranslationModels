from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from asr_client_base import BaseASRClient, TranscriptionResult

# ISO 639-1 (as used by the internal API / this repo's --language flag) ->
# BCP-47 locale expected by Cloud Speech-to-Text v2.
_LANGUAGE_CODE_MAP = {"vi": "vi-VN", "en": "en-US"}


class Chirp3Client(BaseASRClient):
    """Client for Google Cloud Speech-to-Text v2, Chirp 3 model.

    Credentials are read from an in-memory service-account JSON string (env
    var, not a file path) since Colab Secrets store strings -- this avoids
    writing a temp key file from the notebook. Building the client is lazy:
    a missing/invalid credential only surfaces as a per-file error from
    transcribe(), it never raises during __init__/instantiation, so the
    ensemble degrades gracefully instead of crashing when this voter isn't
    configured.

    Chirp 3 (GA) is only available in the "us" and "eu" multi-region
    locations (not zonal regions like "us-central1"). Google's own docs
    state the confidence value Chirp 3 returns "isn't truly a confidence
    score" -- so this client reports confidence=None (untrusted, falls back
    to the ensemble's default vote weight) and keeps the raw value in
    extra["raw_confidence"] for inspection only.
    """

    name = "chirp3"

    def __init__(
        self,
        project_id_env: str = "GCP_PROJECT_ID",
        credentials_json_env: str = "GCP_SERVICE_ACCOUNT_JSON",
        location: str = "us",
        model: str = "chirp_3",
    ):
        self.project_id_env = project_id_env
        self.credentials_json_env = credentials_json_env
        self.location = location
        self.model = model
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client

        from google.cloud.speech_v2 import SpeechClient
        from google.api_core.client_options import ClientOptions
        from google.oauth2 import service_account

        creds_raw = os.environ.get(self.credentials_json_env)
        if not creds_raw:
            raise RuntimeError(f"Missing GCP credentials: {self.credentials_json_env} not set")

        info = json.loads(creds_raw)
        credentials = service_account.Credentials.from_service_account_info(info)

        client_options = None
        if self.location != "global":
            client_options = ClientOptions(api_endpoint=f"{self.location}-speech.googleapis.com")

        self._client = SpeechClient(credentials=credentials, client_options=client_options)
        return self._client

    def transcribe(self, wav_path: Path, *, language: Optional[str] = None) -> TranscriptionResult:
        from google.cloud.speech_v2.types import cloud_speech

        try:
            client = self._ensure_client()

            project_id = os.environ.get(self.project_id_env)
            if not project_id:
                raise RuntimeError(f"Missing GCP project id: {self.project_id_env} not set")

            language_code = _LANGUAGE_CODE_MAP.get(language or "vi", language or "vi-VN")

            config = cloud_speech.RecognitionConfig(
                auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
                language_codes=[language_code],
                model=self.model,
            )
            request = cloud_speech.RecognizeRequest(
                recognizer=f"projects/{project_id}/locations/{self.location}/recognizers/_",
                config=config,
                content=wav_path.read_bytes(),
            )

            start = time.perf_counter()
            response = client.recognize(request=request)
            latency = time.perf_counter() - start

            text_parts = []
            raw_confidence = None
            for result in response.results:
                if not result.alternatives:
                    continue
                alt = result.alternatives[0]
                text_parts.append(alt.transcript)
                if raw_confidence is None:
                    raw_confidence = getattr(alt, "confidence", None)

            return TranscriptionResult(
                model_name=self.name,
                text=" ".join(text_parts).strip(),
                confidence=None,  # Chirp 3's confidence isn't a reliable score; don't vote-weight by it
                detected_language=language_code,
                latency_sec=latency,
                error=None,
                called_at=datetime.now(timezone.utc).isoformat(),
                extra={"raw_confidence": raw_confidence, "model": self.model, "location": self.location},
            )

        except Exception as e:
            return TranscriptionResult(
                model_name=self.name,
                text=None,
                confidence=None,
                detected_language=None,
                latency_sec=None,
                error=repr(e),
                called_at=datetime.now(timezone.utc).isoformat(),
            )
