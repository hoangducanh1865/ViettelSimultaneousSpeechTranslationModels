#!/usr/bin/env python3
"""Xuất JSON list base sample_id đã lỗi của 1 task (từ failures.jsonl) -- phục vụ `--rerun-mode failed`.

    python tools/failed_ids.py --task code_switching --output /path/only_ids.json [--stage generate-questions]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _root() -> Path:
    env = os.environ.get("BENCHMARK_QA_LOG_DIR")
    if env:
        return Path(env)
    qa = os.environ.get("QA_DATASETS_DIR")
    base = Path(qa) if qa else (Path(__file__).resolve().parents[4] / "data")
    return base / "logs" / "benchmark_qa"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--stage", default=None)
    args = p.parse_args(argv)

    src = _root() / args.task / "failures.jsonl"
    ids: list[str] = []
    seen: set[str] = set()
    if src.exists():
        for line in src.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if args.stage is not None and rec.get("stage") != args.stage:
                continue
            sid = rec.get("sample_id")
            if sid and sid not in seen:
                seen.add(sid)
                ids.append(str(sid))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
    print(f"[rerun] {args.task}: {len(ids)} sample lỗi -> {out}")


if __name__ == "__main__":
    main()
