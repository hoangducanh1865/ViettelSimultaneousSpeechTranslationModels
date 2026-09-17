"""Materializes a debate's knowledge-graph JSON (see knowledge_graph.py) back onto the concrete
candidate-word CSV format every pipeline's filter-csv/build-samples/generate-questions already
expects -- ONE generic tool reused across all 4 tasks/6 variants, instead of 4 near-duplicate
patchers.

Usage:
    python apply_knowledge_graph.py \\
        --knowledge-json knowledge_han_viet.json \\
        --word-col "Từ Hán Việt" \\
        --field-map '{"meaning": "Nghĩa", "category_hint": "Phạm Trù Gợi Ý", "historical_fact": "Sự Kiện Lịch Sử"}' \\
        --out-csv han_viet_tu_dien_final.csv
    # --base-csv optional -- omit it entirely for a from-scratch task like Hán Việt's first CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional

from knowledge_graph import load_knowledge_graph


def apply_to_csv(
    knowledge: dict, base_csv: Optional[Path], word_col: str, field_to_column: dict[str, str], out_csv: Path,
) -> dict:
    """status=="confirmed"/"corrected": dòng được giữ/sửa (giá trị mới đè lên); status=="added":
    thêm dòng mới (word chưa có trong base_csv, hoặc base_csv=None -- trường hợp Hán Việt từ
    đầu); status=="rejected": xóa dòng dù đang có trong base_csv. Trả về thống kê để in ra."""
    fieldnames = [word_col] + list(field_to_column.values())
    rows_by_word: dict[str, dict] = {}

    if base_csv is not None:
        with open(base_csv, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or fieldnames
            for row in reader:
                w = row[word_col].strip()
                if w:
                    rows_by_word[w] = row

    stats = {"confirmed": 0, "corrected": 0, "added": 0, "rejected": 0}
    for entry in knowledge.get("words", []):
        word = entry["word"]
        status = entry.get("status")
        if status == "rejected":
            rows_by_word.pop(word, None)
            stats["rejected"] += 1
            continue

        if status not in ("confirmed", "corrected", "added"):
            continue

        new_row = dict(rows_by_word.get(word, {}))
        new_row[word_col] = word
        for field_key, column_name in field_to_column.items():
            value = entry.get("fields", {}).get(field_key)
            if value is not None:
                new_row[column_name] = value
                if column_name not in fieldnames:
                    fieldnames.append(column_name)
        rows_by_word[word] = new_row
        stats[status] += 1

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_by_word.values())

    print(f"Đã ghi {out_csv}: {len(rows_by_word)} dòng "
          f"(confirmed={stats['confirmed']}, corrected={stats['corrected']}, "
          f"added={stats['added']}, rejected={stats['rejected']}).")
    return stats


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--knowledge-json", required=True)
    parser.add_argument("--base-csv", default=None, help="Bỏ trống nếu chưa có CSV nào (ví dụ Hán Việt lần đầu).")
    parser.add_argument("--word-col", required=True)
    parser.add_argument("--field-map", required=True, help='JSON object {"knowledge_field": "csv_column_name", ...}')
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args(argv)

    knowledge = load_knowledge_graph(Path(args.knowledge_json))
    field_map = json.loads(args.field_map)
    apply_to_csv(knowledge, Path(args.base_csv) if args.base_csv else None, args.word_col, field_map, Path(args.out_csv))


if __name__ == "__main__":
    main()
