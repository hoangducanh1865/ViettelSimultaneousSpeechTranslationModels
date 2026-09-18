#!/usr/bin/env python3
"""Tiện ích THỐNG KÊ/IN manifest QA -- thay cho các cell Python "inspect" rải rác trong
notebook benchmark_qa_constructer (đọc manifest, in cây category, nghe/thử sample, head file...).

Chỉ dùng thư viện chuẩn, không gọi API, không ghi file.

Ví dụ:
    python3 tools/inspect_qa.py tree  /path/test_sound.jsonl
    python3 tools/inspect_qa.py stats /path/test_speech.jsonl
    python3 tools/inspect_qa.py sample /path/test_speech.jsonl --start 0 --n 3
    python3 tools/inspect_qa.py head   /path/MAPPING_REPORT.json --n 100
    python3 tools/inspect_qa.py knowledge-status /path/knowledge
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_flexible(path: Path) -> list[dict]:
    """Đọc .jsonl, hoặc .json (list) -- dùng cho output có thể là 1 trong 2."""
    with open(path, encoding="utf-8") as f:
        first = f.read(1)
    if first == "[":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    return load_jsonl(path)


def _total(node) -> int:
    if isinstance(node, dict):
        return sum(_total(v) for v in node.values())
    return int(node)


# ---------------------------------------------------------------------------------------------
# tree
# ---------------------------------------------------------------------------------------------
def cmd_tree(args: argparse.Namespace) -> None:
    records = load_jsonl(Path(args.path))
    print(f"Tổng số sample: {len(records)}\n")
    if not records:
        return
    sample = records[0]

    if "subsubcategory" in sample:  # sound: category/subcategory/subsubcategory/task
        tree: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))
        for r in records:
            tree[r["category"]][r["subcategory"]][r["subsubcategory"]][r["task"]] += 1
        for category, subs in tree.items():
            print(f"{category} ({_total(subs)})")
            for sub, subsubs in subs.items():
                print(f"  └─ {sub} ({_total(subsubs)})")
                for subsub, tasks in subsubs.items():
                    print(f"      └─ {subsub} ({_total(tasks)})")
                    for task, count in tasks.items():
                        print(f"          └─ {task}: {count}")
            print()
    elif "sub-category" in sample:  # speech: category/sub-category/sub-sub-category
        tree = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        for r in records:
            tree[r["category"]][r["sub-category"]][r["sub-sub-category"]] += 1
        for category, subs in tree.items():
            print(f"{category} ({_total(subs)})")
            for sub, subsubs in subs.items():
                print(f"  └─ {sub} ({_total(subsubs)})")
                for subsub, count in subsubs.items():
                    print(f"      └─ {subsub}: {count}")
            print()
    else:  # fallback: phân bố theo dataset / task / category nếu có
        key = next((k for k in ("dataset", "task", "category", "dataset_source") if k in sample), None)
        if key is None:
            print("Không nhận diện được schema để in cây (thiếu subsubcategory / sub-category / dataset).")
            return
        for value, count in Counter(r.get(key) for r in records).most_common():
            print(f"{value}: {count}")


# ---------------------------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------------------------
def cmd_stats(args: argparse.Namespace) -> None:
    records = load_jsonl(Path(args.path))
    n = len(records)
    print(f"Tổng số sample: {n}")
    if not records:
        return
    sample = records[0]

    if "answer" in sample and "choices" not in sample:
        yes_no = {"yes", "no"}
        yes = [r for r in records if str(r.get("answer", "")).strip().lower() in yes_no]
        non = [r for r in records if str(r.get("answer", "")).strip().lower() not in yes_no]
        if n:
            print(f"Yes/No: {len(yes)} ({100 * len(yes) / n:.1f}%)")
            print(f"Không phải Yes/No: {len(non)} ({100 * len(non) / n:.1f}%)")
        counter = Counter(str(r.get("answer", "")).strip().lower() for r in non)
        print(f"10 đáp án phổ biến nhất (ngoài yes/no): {counter.most_common(10)}")

    if "dataset_source" in sample:
        print(f"Phân bố dataset_source: {dict(Counter(r['dataset_source'] for r in records))}")
    if "cs_terms" in sample:
        with_terms = sum(1 for r in records if r.get("cs_terms"))
        print(f"Số sample có >=1 cs_terms: {with_terms}")
    if "task" in sample:
        print(f"Phân bố task: {dict(Counter(r['task'] for r in records))}")
    if "dataset" in sample:
        print(f"Phân bố dataset: {dict(Counter(r['dataset'] for r in records))}")
    if "max_level" in sample:
        print(f"Phân bố max_level: {dict(Counter(r['max_level'] for r in records))}")


# ---------------------------------------------------------------------------------------------
# sample / head
# ---------------------------------------------------------------------------------------------
def cmd_sample(args: argparse.Namespace) -> None:
    records = load_jsonl(Path(args.path))
    end = args.start + args.n
    for r in records[args.start:end]:
        print(json.dumps(r, ensure_ascii=False, indent=2))
        if args.audio:
            for key in ("audio", "audio_path", "audio_id"):
                if r.get(key):
                    print(f"audio ({key}): {r[key]}")
                    break
        print("---")
    print(f"(hiển thị {min(args.n, max(0, len(records) - args.start))} sample, bắt đầu từ index {args.start})")


def cmd_head(args: argparse.Namespace) -> None:
    with open(args.path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.n:
                break
            sys.stdout.write(line)


# ---------------------------------------------------------------------------------------------
# knowledge-status
# ---------------------------------------------------------------------------------------------
EXPECTED_KG_FILES = [
    "knowledge_han_viet.json",
    "knowledge_phuong_ngu.json",
    "knowledge_tu_muon.json",
    "knowledge_tu_lay_toan_bo.json",
    "knowledge_tu_lay_van.json",
    "knowledge_tu_lay_chung.json",
]


def cmd_knowledge_status(args: argparse.Namespace) -> None:
    base = Path(args.path)
    print(f"Knowledge dir: {base}")
    for name in EXPECTED_KG_FILES:
        p = base / name
        mark = "✓" if p.exists() else "✗ (chưa có -- cell dùng --knowledge-json sẽ fallback về hành vi cũ)"
        print(f"  {mark} {p}")


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("tree", help="In cây category/subcategory/... của manifest.")
    p.add_argument("path")
    p.set_defaults(func=cmd_tree)

    p = sub.add_parser("stats", help="Thống kê tổng quát + phân bố đáp án/dataset/task/level.")
    p.add_argument("path")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("sample", help="In vài sample đầy đủ.")
    p.add_argument("path")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--audio", action="store_true", help="In thêm đường dẫn audio của sample.")
    p.set_defaults(func=cmd_sample)

    p = sub.add_parser("head", help="In n dòng đầu của file (raw, không parse).")
    p.add_argument("path")
    p.add_argument("--n", type=int, default=3)
    p.set_defaults(func=cmd_head)

    p = sub.add_parser("knowledge-status", help="Kiểm tra 6 file knowledge_<task>.json đã có chưa.")
    p.add_argument("path")
    p.set_defaults(func=cmd_knowledge_status)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
