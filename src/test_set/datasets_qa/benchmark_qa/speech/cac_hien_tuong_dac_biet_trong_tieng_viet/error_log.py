"""Log các SAMPLE bị lỗi trong quá trình chạy benchmark_qa -- lưu trong `data/logs/benchmark_qa/`
(đã gitignore, chỉ xem local). Dùng để chạy lại CHỈ các sample lỗi (`--rerun-mode failed`).

Mỗi dòng trong `failures.jsonl`: {timestamp, task, stage, sample_id, error, model, extra}.
`sample_id` là id gốc (base id, phần trước `__` nếu là id ghép).
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

_LOCK = threading.Lock()


def log_root() -> Path:
    """Gốc log: ưu tiên env BENCHMARK_QA_LOG_DIR (bash set), rồi QA_DATASETS_DIR, cuối cùng
    <repo>/data."""
    env = os.environ.get("BENCHMARK_QA_LOG_DIR")
    if env:
        return Path(env)
    qa = os.environ.get("QA_DATASETS_DIR")
    base = Path(qa) if qa else (Path(__file__).resolve().parents[6] / "data")
    return base / "logs" / "benchmark_qa"


def base_id(sample_id) -> str:
    return str(sample_id).split("__")[0]


def log_failure(task: str, stage: str, sample_id, error, *, model: Optional[str] = None,
                extra: Optional[dict] = None) -> None:
    """Ghi 1 dòng lỗi (thread-safe). Không bao giờ raise (log lỗi không được làm chết luồng)."""
    try:
        d = log_root() / task
        d.mkdir(parents=True, exist_ok=True)
        rec = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task": task, "stage": stage, "sample_id": base_id(sample_id),
            "error": str(error)[:1000], "model": model,
        }
        if extra:
            rec["extra"] = extra
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        with _LOCK:
            with open(d / "failures.jsonl", "a", encoding="utf-8") as f:
                f.write(line)
    except Exception:  # noqa: BLE001
        pass


def log_failures(task: str, stage: str, sample_ids: Iterable, error, *, model: Optional[str] = None) -> None:
    for sid in sample_ids:
        log_failure(task, stage, sid, error, model=model)


def clear_failures(task: str) -> None:
    p = log_root() / task / "failures.jsonl"
    if p.exists():
        p.unlink()


def failed_ids(task: str, *, stage: Optional[str] = None) -> set[str]:
    """Đọc các base sample_id đã lỗi (lọc theo stage nếu truyền)."""
    p = log_root() / task / "failures.jsonl"
    ids: set[str] = set()
    if not p.exists():
        return ids
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if stage is not None and rec.get("stage") != stage:
                continue
            if rec.get("sample_id"):
                ids.add(str(rec["sample_id"]))
    return ids
