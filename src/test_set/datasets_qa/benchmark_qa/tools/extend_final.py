#!/usr/bin/env python3
"""Tạo bản "extended-final": file final (schema publish) + thêm field debug hữu ích (transcript,
question_type, difficulty, target words, level/max_level...) bằng cách JOIN với các file pre-final.

Ví dụ:
    python tools/extend_final.py \\
        --final code_switching_openai_kept_final.jsonl \\
        --output code_switching_openai_kept_final_extended.jsonl \\
        --join "code_switching_openai_kept.jsonl::id::question_type,target_terms,dataset_source" \\
        --join "code_switching_classified.jsonl::base_id::transcript=text"

Mỗi `--join` có dạng:  <path>::<by>::<field1,field2,...>
  - by = "id"        -> join theo id đầy đủ
  - by = "base_id"   -> join theo id.split("__")[0]
  - field có dạng "out" (giữ nguyên tên) hoặc "out=src" (đổi tên field).
  - Join sau chỉ ĐIỀN field nếu field đang thiếu/None (không ghi đè giá trị đã có).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _parse_join(spec: str) -> dict:
    parts = spec.split("::")
    if len(parts) != 3:
        raise ValueError(f"--join phải có dạng <path>::<by>::<fields>, nhận: {spec!r}")
    path, by, fields = parts
    if by not in ("id", "base_id"):
        raise ValueError(f"--join by phải là 'id' hoặc 'base_id', nhận: {by!r}")
    field_map = {}
    for token in fields.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" in token:
            out, src = token.split("=", 1)
        else:
            out = src = token
        field_map[out.strip()] = src.strip()
    return {"path": Path(path), "by": by, "field_map": field_map}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--final", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--join", action="append", default=[], help="<path>::<id|base_id>::<fields> (lặp lại được)")
    args = p.parse_args(argv)

    joins = [_parse_join(s) for s in args.join]
    indexes: list[dict] = []
    for j in joins:
        idx = {}
        for r in _load_jsonl(j["path"]):
            key = r.get("id")
            if key is None:
                continue
            if j["by"] == "base_id":
                key = str(key).split("__")[0]
            idx[key] = r
        indexes.append(idx)

    final_records = _load_jsonl(Path(args.final))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_out = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for r in final_records:
            key = r.get("id")
            base_key = str(key).split("__")[0] if key is not None else None
            for j, idx in zip(joins, indexes):
                lookup = base_key if j["by"] == "base_id" else key
                src = idx.get(lookup)
                if not src:
                    continue
                for out_field, src_field in j["field_map"].items():
                    if r.get(out_field) in (None, "", [], {}):
                        r[out_field] = src.get(src_field)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"Đã ghi {out_path} ({n_out} dòng) từ {args.final} + {len(joins)} join.")


if __name__ == "__main__":
    main()
