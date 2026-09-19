"""Bootstraps an initial Hán-Việt candidate-word CSV from the existing han_viet_qa.jsonl.

Hán Việt currently has NO hand-curated candidate CSV at all (unlike từ láy/từ mượn, which each
start from a curated word-list CSV) -- it only had 95 pre-existing single-fact Q&A records
already tied to a "Sino-Vietnamese Lexical Semantics" subset published on HF. This script gives
Hán Việt a starting-point CSV (word + meaning, deduped) so it can go through the SAME
filter-csv -> build-samples -> word-coverage-report -> manual_model_relay.py flow every other
task already uses -- the 3-model debate then reviews/corrects this seed AND actively proposes
brand-new Hán Việt words found via web-research + a speech_sources.jsonl corpus scan (Hán Việt
is an open lexical class, so this bootstrap is a starting point, not a ceiling).

Usage:
    python han_viet_seed_csv.py --input han_viet/han_viet_qa.jsonl --out-csv han_viet_seed.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional

from han_viet_pipeline import extract_target_word

SEED_CSV_FIELDS = ["STT", "Từ Hán Việt", "Nghĩa", "Ghi Chú"]


def _ensure_parent(path):
    """Tạo thư mục cha trước khi ghi file (local/thư mục tạm có thể chưa có sẵn như trên Drive)."""
    from pathlib import Path as _P
    p = _P(path)
    if str(p.parent) and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    return p

def build_seed_csv(han_viet_qa_path: Path, out_csv: Path) -> dict:
    with open(han_viet_qa_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    seen: dict[str, dict] = {}
    n_extracted = 0
    for r in records:
        word = extract_target_word(r)
        if word is None:
            continue
        n_extracted += 1
        # Giữ record ĐẦU TIÊN trích được cho mỗi từ -- đủ để có 1 nghĩa tham khảo ban đầu,
        # debate sẽ sửa lại nếu sai/chưa đủ.
        seen.setdefault(word, {"Từ Hán Việt": word, "Nghĩa": r["answer"], "Ghi Chú": f"source_id={r['id']}"})

    _ensure_parent(out_csv)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SEED_CSV_FIELDS)
        writer.writeheader()
        for i, row in enumerate(seen.values(), start=1):
            writer.writerow({"STT": i, **row})

    stats = {"n_records": len(records), "n_extracted": n_extracted, "n_unique_words": len(seen)}
    print(f"Đã đọc {stats['n_records']} record, trích được target_word cho {stats['n_extracted']} "
          f"record ({stats['n_unique_words']} từ duy nhất) -- lưu {out_csv}.")
    return stats


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="han_viet/han_viet_qa.jsonl")
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args(argv)
    build_seed_csv(Path(args.input), Path(args.out_csv))
