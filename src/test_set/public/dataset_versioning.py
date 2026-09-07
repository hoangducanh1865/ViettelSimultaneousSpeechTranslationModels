"""Normalize + validate + version the final `text_vi`/`text_en` pair of a
test-set sample, and manage immutable `versions/ver_N.json` snapshots of
`tmp.json` (see `manifest_io.py`).

Introduced to fix a real asymmetry bug in the old pipeline: only ONE side
of the pair was ever run through `text_normalize.normalize_text()`
(`text_en` for the vi_en direction, `text_vi` for en_vi/PhoST) -- the other
side kept natural casing/punctuation, and some rows even ended up
*accidentally* lowercase as a leftover of the ROVER-fusion stage rather
than a deliberate normalization step. `normalize_samples()` below applies
`normalize_text()` to BOTH fields, always, and `validate_samples()` enforces
it as a checked invariant instead of a convention.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from text_normalize import normalize_text, repetition_ratio

REQUIRED_FIELDS = ("id", "source_dataset", "audio_path", "duration_sec", "text_vi", "text_en")
REPETITION_THRESHOLD = 0.3
DURATION_TOLERANCE_SEC = 0.5


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# Normalize
# ---------------------------------------------------------------------------

def normalize_samples(samples: dict[str, dict]) -> None:
    """In place, for every sample: back up the natural-casing text into
    `text_vi_cased`/`text_en_cased` (if not already backed up -- idempotent,
    safe to re-run), then overwrite `text_vi`/`text_en` with
    `normalize_text()`. Applies to BOTH fields, unlike the old pipeline.
    """
    for sample in samples.values():
        if sample.get("text_vi_cased") is None and sample.get("text_vi"):
            sample["text_vi_cased"] = sample["text_vi"]
        if sample.get("text_en_cased") is None and sample.get("text_en"):
            sample["text_en_cased"] = sample["text_en"]

        if sample.get("text_vi_cased"):
            sample["text_vi"] = normalize_text(sample["text_vi_cased"])
        if sample.get("text_en_cased"):
            sample["text_en"] = normalize_text(sample["text_en_cased"])


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

@dataclass
class ValidationIssue:
    sample_id: str
    check: str
    detail: str


@dataclass
class ValidationReport:
    n_samples: int
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.issues) == 0

    def summary(self) -> str:
        by_check: dict[str, int] = {}
        for issue in self.issues:
            by_check[issue.check] = by_check.get(issue.check, 0) + 1
        lines = [f"{self.n_samples} sample kiểm tra, {len(self.issues)} vấn đề."]
        for check, count in sorted(by_check.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {check}: {count}")
        return "\n".join(lines)

    def raise_if_failed(self, *, max_examples: int = 20) -> None:
        if self.passed:
            return
        examples = "\n".join(
            f"  [{i.check}] {i.sample_id}: {i.detail}" for i in self.issues[:max_examples]
        )
        more = f"\n  ... và {len(self.issues) - max_examples} vấn đề khác" if len(self.issues) > max_examples else ""
        raise RuntimeError(
            f"Validate thất bại -- {self.summary()}\n\nVí dụ:\n{examples}{more}"
        )


def validate_samples(
    samples: dict[str, dict],
    *,
    audio_language: str,
    check_audio_language: bool = True,
    check_duration: bool = True,
) -> ValidationReport:
    """Run every check below on `samples` (keyed by id, tmp.json's shape)
    and return a `ValidationReport` -- never raises itself, call
    `.raise_if_failed()` for the old GATE-style hard-stop behavior.

    Checks: completeness, id/key match, audio existence, duration sanity
    (soundfile, `check_duration=False` to skip -- slow on Drive FUSE for a
    full re-run), language-ID sanity (`lang_id.py`, `check_audio_language=
    False` to skip/soft-skip e.g. for PhoST's already-official ground
    truth), degenerate/repetition, and the normalization invariant.
    """
    issues: list[ValidationIssue] = []

    def add(sample_id: str, check: str, detail: str) -> None:
        issues.append(ValidationIssue(sample_id=sample_id, check=check, detail=detail))

    seen_ids: set[str] = set()
    for sample_id, sample in samples.items():
        if sample_id in seen_ids:
            add(sample_id, "duplicate_id", "id trùng lặp trong samples")
        seen_ids.add(sample_id)

        if sample.get("id") != sample_id:
            add(sample_id, "id_mismatch", f"row['id']={sample.get('id')!r} khác key {sample_id!r}")

        missing = [f for f in REQUIRED_FIELDS if not sample.get(f) and sample.get(f) != 0]
        if missing:
            add(sample_id, "missing_field", f"thiếu/rỗng: {missing}")
            continue  # các check sau cần các field này, bỏ qua nếu đã thiếu

        if not str(sample["text_vi"]).strip() or not str(sample["text_en"]).strip():
            add(sample_id, "empty_text", "text_vi/text_en rỗng sau khi strip()")

        audio_path = Path(sample["audio_path"])
        if not audio_path.exists():
            add(sample_id, "audio_missing", f"không tồn tại: {audio_path}")
            continue  # các check audio sau cần file thật

        if check_duration:
            try:
                import soundfile as sf

                info = sf.info(str(audio_path))
                actual = info.frames / info.samplerate
                stored = float(sample["duration_sec"])
                if abs(actual - stored) > DURATION_TOLERANCE_SEC:
                    add(sample_id, "duration_mismatch", f"lưu={stored:.2f}s, thực tế={actual:.2f}s")
            except Exception as e:
                add(sample_id, "duration_check_error", repr(e))

        if check_audio_language:
            try:
                import lang_id

                verdict = lang_id.audio_language_verdict(audio_path, expected_language=audio_language)
                if verdict["error"] is None and not verdict["keep"]:
                    add(sample_id, "language_id_fail", f"không phát hiện đoạn '{audio_language}' nào trong audio")
            except Exception as e:
                add(sample_id, "language_id_error", repr(e))

        if repetition_ratio(sample["text_vi"]) >= REPETITION_THRESHOLD:
            add(sample_id, "degenerate_text_vi", "lặp từ >= 30% trong text_vi")
        if repetition_ratio(sample["text_en"]) >= REPETITION_THRESHOLD:
            add(sample_id, "degenerate_text_en", "lặp từ >= 30% trong text_en")

        text_vi_cased = sample.get("text_vi_cased")
        if text_vi_cased and normalize_text(text_vi_cased) != sample["text_vi"]:
            add(sample_id, "normalization_invariant_vi", "text_vi không khớp normalize_text(text_vi_cased)")
        text_en_cased = sample.get("text_en_cased")
        if text_en_cased and normalize_text(text_en_cased) != sample["text_en"]:
            add(sample_id, "normalization_invariant_en", "text_en không khớp normalize_text(text_en_cased)")

    return ValidationReport(n_samples=len(samples), issues=issues)


# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------

def next_version_number(versions_dir: Path) -> int:
    existing = sorted(versions_dir.glob("ver_*.json")) if versions_dir.is_dir() else []
    numbers = []
    for p in existing:
        try:
            numbers.append(int(p.stem.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return (max(numbers) + 1) if numbers else 1


def save_version(versions_dir: Path, tmp_envelope: dict, *, validation_passed: bool) -> Path:
    """Freeze a verbatim, same-structure copy of `tmp_envelope` into
    `versions/ver_N.json` -- immutable once written (never edited in place;
    a later fix goes into `ver_(N+1).json`). Also updates `index.json`."""
    version = next_version_number(versions_dir)
    version_path = versions_dir / f"ver_{version}.json"
    _atomic_write_json(version_path, tmp_envelope)

    index_path = versions_dir / "index.json"
    index = []
    if index_path.exists():
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
    index.append({
        "version": version,
        "file": version_path.name,
        "direction": tmp_envelope.get("direction"),
        "generation_mode": tmp_envelope.get("generation_mode"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(tmp_envelope.get("samples", {})),
        "validation_passed": validation_passed,
    })
    _atomic_write_json(index_path, index)
    return version_path


def list_versions(versions_dir: Path) -> list[dict]:
    index_path = versions_dir / "index.json"
    if not index_path.exists():
        return []
    with open(index_path, encoding="utf-8") as f:
        return json.load(f)


def load_version(versions_dir: Path, version: int) -> dict:
    version_path = versions_dir / f"ver_{version}.json"
    if not version_path.exists():
        raise FileNotFoundError(f"Không tìm thấy version {version}: {version_path}")
    with open(version_path, encoding="utf-8") as f:
        return json.load(f)
