from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TranscriptionResult:
    model_name: str
    text: Optional[str]
    confidence: Optional[float]  # 0..1, None if unavailable/untrustworthy for voting
    detected_language: Optional[str]
    latency_sec: Optional[float] = None
    error: Optional[str] = None  # non-None => this call failed
    called_at: Optional[str] = None  # ISO8601
    extra: dict = field(default_factory=dict)

    def to_json_dict(self, audio_relpath: str) -> dict:
        return {
            "schema_version": 1,
            "model_name": self.model_name,
            "audio_relpath": audio_relpath,
            "text": self.text,
            "confidence": self.confidence,
            "detected_language": self.detected_language,
            "latency_sec": self.latency_sec,
            "error": self.error,
            "called_at": self.called_at,
            "extra": self.extra,
        }

    @classmethod
    def from_json_dict(cls, d: dict) -> "TranscriptionResult":
        return cls(
            model_name=d["model_name"],
            text=d.get("text"),
            confidence=d.get("confidence"),
            detected_language=d.get("detected_language"),
            latency_sec=d.get("latency_sec"),
            error=d.get("error"),
            called_at=d.get("called_at"),
            extra=d.get("extra") or {},
        )


class BaseASRClient(abc.ABC):
    """Common interface every ASR voter in the ROVER ensemble implements."""

    name: str  # stable id -> cache subfolder name + output JSON key

    @abc.abstractmethod
    def transcribe(self, wav_path: Path, *, language: Optional[str] = None) -> TranscriptionResult:
        """Transcribe a single wav file.

        Must never raise for expected failure modes (missing credentials,
        HTTP error, timeout, network issue) -- capture those into
        TranscriptionResult(error=...) instead, so that one voter failing
        degrades the ensemble rather than crashing the whole run.
        """
        raise NotImplementedError
