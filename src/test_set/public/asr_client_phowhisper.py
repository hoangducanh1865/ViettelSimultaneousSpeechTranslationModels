from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from asr_client_base import BaseASRClient, TranscriptionResult

# vinai/PhoWhisper-{tiny,base,small,medium,large} all exist on the HF Hub.
# Medium is the default for Colab-friendly load time / VRAM; bump to
# -large if the runtime has enough GPU memory.
PHOWHISPER_MODEL_ID = "vinai/PhoWhisper-medium"


class PhoWhisperClient(BaseASRClient):
    """Client for VinAI's PhoWhisper, run locally via transformers.

    Public HF model, no credentials needed. The pipeline is built lazily on
    first transcribe() call so simply instantiating this client (e.g. while
    building the full ensemble's client list) doesn't force a multi-GB
    model download/GPU allocation when this voter is excluded via
    --models.

    Vanilla ASR pipeline output has no reliable per-utterance confidence
    score, so confidence is always None here (falls back to the ensemble's
    default vote weight).
    """

    name = "phowhisper"

    def __init__(self, model_id: str = PHOWHISPER_MODEL_ID, device: Optional[int] = None):
        self.model_id = model_id
        self.device = device
        self._pipeline = None

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        import torch
        from transformers import pipeline

        device = self.device
        if device is None:
            device = 0 if torch.cuda.is_available() else -1

        self._pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model_id,
            device=device,
        )
        return self._pipeline

    def transcribe(self, wav_path: Path, *, language: Optional[str] = None) -> TranscriptionResult:
        try:
            asr_pipeline = self._ensure_pipeline()

            start = time.perf_counter()
            output = asr_pipeline(str(wav_path))
            latency = time.perf_counter() - start

            text = output.get("text", "") if isinstance(output, dict) else str(output)

            return TranscriptionResult(
                model_name=self.name,
                text=text.strip(),
                confidence=None,
                detected_language="vi",
                latency_sec=latency,
                error=None,
                called_at=datetime.now(timezone.utc).isoformat(),
                extra={"model_id": self.model_id},
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
