#!/usr/bin/env python3
"""Tải dữ liệu HF cần cho benchmark QA -- thay các cell `snapshot_download` trong notebook.

Không hard-code token: nhận qua --token hoặc biến môi trường HF_TOKEN.

Ví dụ:
    python3 tools/fetch_datasets.py speech \
        --repo-id anhnbd2005/Vietnamese-Speech-QA \
        --local-dir /content/drive/.../qa_datasets/Vietnamese-Speech-QA \
        --sino-audio --dialect-audio

    python3 tools/fetch_datasets.py speech-zip \
        --repo-id anhnbd2005/Vietnamese-Speech-QA \
        --local-dir /content/drive/.../qa_datasets/Vietnamese-Speech-QA
"""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path


def _snapshot(repo_id: str, allow_patterns, local_dir: str, token: str | None) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=allow_patterns,
        local_dir=local_dir,
        token=token,
    )


def cmd_speech(args: argparse.Namespace) -> None:
    token = args.token or os.environ.get("HF_TOKEN")
    local_dir = args.local_dir
    print(f"Tải MAPPING_REPORT.json + test_speech.jsonl từ {args.repo_id} -> {local_dir}")
    resolved = _snapshot(args.repo_id, ["MAPPING_REPORT.json", "test_speech.jsonl"], local_dir, token)
    print(f"Đã tải về {resolved}")

    test_speech = Path(resolved) / "test_speech.jsonl"
    with open(test_speech, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    print(f"Đọc {len(records)} sample từ {test_speech}")

    patterns: list[str] = []
    if args.sino_audio:
        patterns += [r["audio"] for r in records if r.get("task") == "sino_vietnamese_lexical_semantics"]
        print(f"Cần tải {len(patterns)} file audio subset Hán-Việt.")
    if args.dialect_audio:
        dialect = [r["audio_id"] for r in records if r.get("sub-sub-category") == "Dialectal Form Normalization"]
        patterns += dialect
        print(f"Cần tải {len(dialect)} file audio subset Phương ngữ.")

    if patterns:
        _snapshot(args.repo_id, patterns, local_dir, token)
        print(f"Đã tải xong audio vào {local_dir}")
    else:
        print("Không yêu cầu tải audio subset nào (dùng --sino-audio / --dialect-audio nếu cần).")


def cmd_speech_zip(args: argparse.Namespace) -> None:
    token = args.token or os.environ.get("HF_TOKEN")
    _snapshot(args.repo_id, ["speech.zip"], args.local_dir, token)
    zip_path = Path(args.local_dir) / "speech.zip"
    print(f"Đã tải {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(args.local_dir)
    print(f"Đã giải nén vào {args.local_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--token", default=None, help="HF token (mặc định lấy từ env HF_TOKEN).")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("speech", help="Tải manifest test_speech + (tùy chọn) audio subset.")
    p.add_argument("--repo-id", required=True)
    p.add_argument("--local-dir", required=True)
    p.add_argument("--sino-audio", action="store_true", help="Tải audio subset Hán-Việt.")
    p.add_argument("--dialect-audio", action="store_true", help="Tải audio subset Phương ngữ.")
    p.set_defaults(func=cmd_speech)

    p = sub.add_parser("speech-zip", help="Tải + giải nén speech.zip đã publish.")
    p.add_argument("--repo-id", required=True)
    p.add_argument("--local-dir", required=True)
    p.set_defaults(func=cmd_speech_zip)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
