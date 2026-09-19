"""Strips a task's rich "pre-final" JSONL (transcript, question_type, target_word, level,
historical_fact/cultural_fact/region_or_ethnic_group, tone_register, ...) down to the clean
"final" schema published to HF -- ONE shared trim implementation reused by all 4 tasks/6
variants, instead of 4 near-duplicate patchers with 4 chances to drift from the target schema.

Pre-final files stay on Drive for debugging/traceability; ONLY the final file produced here is
ever pushed to HF via hf_pr_push.py.

Usage:
    python finalize_qa.py --pre-final tu_lay_toan_bo_multihop_qa.jsonl \\
        --final tu_lay_toan_bo_qa_final.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

FINAL_FIELDS = [
    "id", "audio_id", "question", "choices", "answer", "dataset", "task", "split",
    "category", "sub-category", "difficulty",
]


def finalize_record(record: dict, *, sub_category_override: Optional[str] = None) -> dict:
    """Chiếu record giàu field xuống ĐÚNG FINAL_FIELDS. "sub-category" ưu tiên field có gạch
    nối (tu_muon/tu_lay/phuong_ngu) trước, rồi mới rơi về "subcategory" không gạch nối (han_viet
    fill-fields, join từ test_speech.jsonl) -- không có "sub-sub-category" ở bản final, đúng
    khớp schema mẫu user cung cấp (chỉ có 1 field "sub-category" duy nhất).

    Validate CỨNG (raise ValueError, không âm thầm bỏ qua): đúng 4 choices, 4 giá trị PHÂN BIỆT,
    answer thuộc choices, không field FINAL nào null/thiếu -- đây là điểm kiểm tra tính hợp lệ
    publish-time ĐẦU TIÊN trong toàn luồng, độc lập với các validate mềm hơn ở bước sinh."""
    sub_category = sub_category_override or record.get("sub-category") or record.get("subcategory")
    out = {
        "id": record.get("id"), "audio_id": record.get("audio_id"), "question": record.get("question"),
        "choices": record.get("choices"), "answer": record.get("answer"), "dataset": record.get("dataset"),
        "task": record.get("task"), "split": record.get("split"), "category": record.get("category"),
        "sub-category": sub_category, "difficulty": record.get("difficulty"),
    }

    missing = [k for k in FINAL_FIELDS if out.get(k) is None]
    if missing:
        raise ValueError(f"Record {record.get('id')!r} thiếu field: {missing}")
    choices = out["choices"]
    if not isinstance(choices, list) or len(choices) != 4:
        raise ValueError(f"Record {record.get('id')!r} không có đúng 4 choices: {choices!r}")
    if len(set(choices)) != 4:
        raise ValueError(f"Record {record.get('id')!r} có choices trùng nhau: {choices!r}")
    if out["answer"] not in choices:
        raise ValueError(f"Record {record.get('id')!r} có answer {out['answer']!r} không thuộc choices {choices!r}")

    return out


def finalize_file(
    pre_final_path: Path, out_path: Path, *, on_missing: str = "skip", sub_category_override: Optional[str] = None,
) -> dict:
    """Luôn GHI ĐÈ TOÀN BỘ out_path (không append/resumable như bước sinh -- finalize là 1 phép
    biến đổi cục bộ rẻ, an toàn chạy lại bất cứ lúc nào khi pre-final có cập nhật mới)."""
    with open(pre_final_path, encoding="utf-8") as f:
        pre_final_records = [json.loads(line) for line in f if line.strip()]

    n_in = len(pre_final_records)
    n_skipped = 0
    finalized = []
    for r in pre_final_records:
        try:
            finalized.append(finalize_record(r, sub_category_override=sub_category_override))
        except ValueError as e:
            if on_missing == "raise":
                raise
            n_skipped += 1
            print(f"[BỎ QUA] {e}")

    with open(out_path, "w", encoding="utf-8") as f:
        for r in finalized:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {"n_in": n_in, "n_out": len(finalized), "n_skipped": n_skipped}
    print(f"Đã ghi {out_path}: {stats['n_out']}/{stats['n_in']} record hợp lệ ({stats['n_skipped']} bị bỏ qua).")
    return stats


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pre-final", required=True)
    parser.add_argument("--final", required=True)
    parser.add_argument("--on-missing", choices=["skip", "raise"], default="skip")
    parser.add_argument("--sub-category-override", default=None)
    args = parser.parse_args(argv)

    finalize_file(
        Path(args.pre_final), Path(args.final),
        on_missing=args.on_missing, sub_category_override=args.sub_category_override,
    )
