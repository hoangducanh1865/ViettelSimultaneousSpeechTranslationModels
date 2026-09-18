"""MMSU (ddwang2000/MMSU) -> Code-switching subset pipeline.

MMSU không có config/subset riêng cho từng hiện tượng -- toàn bộ 5000 sample nằm trong 1 split
"train" duy nhất, phân biệt qua cột "task_name" (47 giá trị). Subset "Code-switching" tương ứng
task_name == "code_switch_question_answering" (111/5000 sample, category=Reasoning,
sub-category=Linguistics, sub-sub-category=Semantics).

1 subcommand (thêm subcommand khác khi có bước tiếp theo, ví dụ dịch sang tiếng Việt):
  1. build-manifest -- tải 3 shard parquet của MMSU (dataset không hỗ trợ lọc theo cột khi tải),
                        giữ lại CHỈ sample thuộc subset Code-switching, ghi audio .wav + manifest
                        JSONL (question/choices/answer gốc tiếng Anh).

Usage:
    python code_switching_pipeline.py build-manifest \\
        --output-dir /path/to/code_switching
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from tqdm.auto import tqdm

TASK_NAME = "code_switch_question_answering"


# ============================================================================================
# Step 1: build-manifest
# ============================================================================================

def build_manifest(output_dir: Path) -> None:
    """Tải toàn bộ 3 shard parquet của ddwang2000/MMSU, lọc lại CHỈ sample có
    task_name == TASK_NAME (subset Code-switching), ghi audio .wav + manifest JSONL."""
    import soundfile as sf
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "code_switching_test.jsonl"

    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        print(f"{manifest_path} đã tồn tại -- đọc lại {len(records)} sample, không tải lại.")
        return

    local_dir = snapshot_download(
        repo_id="ddwang2000/MMSU",
        repo_type="dataset",
        allow_patterns=["data/train-*.parquet"],
        max_workers=8,
    )
    ds = load_dataset("parquet", data_files=f"{local_dir}/data/train-*.parquet", split="train")
    print(f"Đã load {len(ds)} sample (toàn bộ MMSU) từ {local_dir}.")

    ds = ds.filter(lambda r: r["task_name"] == TASK_NAME)
    print(f"Giữ lại {len(ds)} sample thuộc subset Code-switching (task_name=\"{TASK_NAME}\").")

    def _write_one(idx_row):
        _, row = idx_row
        sid = row["id"]
        out_wav = audio_dir / f"{sid}.wav"
        if not out_wav.exists():
            audio = row["audio"]
            sf.write(out_wav, audio["array"], audio["sampling_rate"])
        return {
            "id": sid,
            "audio_path": f"audio/{sid}.wav",
            "question": row["question"],
            "choices": [row["choice_a"], row["choice_b"], row["choice_c"], row["choice_d"]],
            "answer": row["answer_gt"],
            "category": row["category"],
            "sub_category": row["sub-category"],
            "sub_sub_category": row["sub-sub-category"],
        }

    # KHÔNG pre-list toàn bộ dataset đã lọc vào RAM trước khi submit (giữ nguyên thói quen lazy
    # iterate như clotho_aqa_pipeline.py, dù ở đây dataset đã lọc nhỏ nên ít rủi ro OOM hơn).
    records = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(_write_one, idx_row)
            for idx_row in tqdm(enumerate(ds), total=len(ds), desc="submit task")
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="ghi audio Code-switching"):
            records.append(future.result())

    records.sort(key=lambda r: r["id"])
    with open(manifest_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Đã lưu {len(records)} sample (audio .wav + manifest) vào {output_dir}")


# ============================================================================================
# CLI
# ============================================================================================

def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("build-manifest", help="Tải + xây manifest audio subset Code-switching (MMSU).")
    p1.add_argument("--output-dir", required=True)

    args = parser.parse_args(argv)

    if args.command == "build-manifest":
        build_manifest(Path(args.output_dir))


if __name__ == "__main__":
    main()
