"""Merge MMAU test-mini (sound domain) + ClothoAQA (translated) into 1 unified test_sound.jsonl.

Extracted from noteboooks/benchmark_qa/Benchmarck_QA_contructer copy 2.ipynb.

Copies each sample's audio into a single "sound/" directory under a NEW, deterministic id
(hash of the original sample id) -- deterministic so re-running the merge never duplicates
audio under a different name and always reproduces byte-identical output for unchanged input.

Usage:
    python sound_dataset_merge.py \\
        --mmau-manifest /path/to/mmau_sound/sound-test-mini.jsonl \\
        --mmau-audio-dir /path/to/mmau_sound/audio \\
        --clotho-manifest /path/to/clotho_aqa/clotho_aqa_test_vi_qa.jsonl \\
        --clotho-audio-dir /path/to/clotho_aqa/audio \\
        --sound-out-dir /path/to/sound \\
        --output /path/to/test_sound.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Optional


def _snake_case(text: str) -> str:
    text = re.sub(r"[^\w\s]", "", text)  # bỏ dấu câu (vd "/" trong "Region / Accent")
    text = re.sub(r"\s+", "_", text.strip())
    return text.lower()


def _new_id(prefix: str, seed: str) -> str:
    digest = hashlib.md5(seed.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}_{digest}"


def merge(
    mmau_manifest_path: Path, mmau_audio_dir: Path,
    clotho_manifest_path: Path, clotho_audio_dir: Path,
    sound_out_dir: Path, output_path: Path,
) -> list[dict]:
    sound_out_dir.mkdir(parents=True, exist_ok=True)
    merged_records: list[dict] = []

    # --- 1. MMAU test-mini (sound domain, đã dịch VN) --------------------------------------
    with open(mmau_manifest_path, encoding="utf-8") as f:
        mmau_records = [json.loads(line) for line in f if line.strip()]

    n_missing_audio_mmau = 0
    for r in mmau_records:
        src_wav = mmau_audio_dir / Path(r["audio_path"]).name
        if not src_wav.exists():
            n_missing_audio_mmau += 1
            continue

        audio_id = _new_id("audio", r["id"])
        dst_wav = sound_out_dir / f"{audio_id}.wav"
        if not dst_wav.exists():
            shutil.copyfile(src_wav, dst_wav)

        subsubcategory = r["sub_category"]  # vd "Acoustic Source Inference", "Temporal Event Reasoning"
        merged_records.append({
            "id": _new_id("qa", r["id"]),
            "audio_id": audio_id,
            "audio": f"sound/{audio_id}.wav",
            "question": r["question_vi"],
            "choices": json.loads(r["choices_vi"]) if isinstance(r["choices_vi"], str) else r["choices_vi"],
            "answer": r["answer_vi"],
            "dataset": r["dataset"],
            "task": _snake_case(subsubcategory),
            "split": "test",
            "category": "Reasoning",
            "subcategory": "Environmental Sound",
            "subsubcategory": subsubcategory,
            "difficulty": r.get("difficulty", "medium"),
        })

    if n_missing_audio_mmau:
        print(f"CẢNH BÁO: {n_missing_audio_mmau} sample MMAU thiếu file audio local -- đã bỏ qua.")
    print(f"MMAU test-mini: {len(merged_records)} sample đã merge.")

    # --- 2. ClothoAQA (đã dịch + sinh nhiễu, chỉ phần non-yes/no + lọc VN-relevance) --------
    n_before = len(merged_records)
    n_missing_audio_clotho = 0
    with open(clotho_manifest_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)

            src_wav = clotho_audio_dir / Path(r["audio_id"]).name
            if not src_wav.exists():
                n_missing_audio_clotho += 1
                continue

            audio_id = _new_id("audio", r["id"])
            dst_wav = sound_out_dir / f"{audio_id}.wav"
            if not dst_wav.exists():
                shutil.copyfile(src_wav, dst_wav)

            subsubcategory = "Sound Event Recognition"
            merged_records.append({
                "id": _new_id("qa", r["id"]),
                "audio_id": audio_id,
                "audio": f"sound/{audio_id}.wav",
                "question": r["question"],
                "choices": r["choices"],
                "answer": r["answer"],
                "dataset": r["dataset"],
                "task": _snake_case(subsubcategory),
                "split": "test",
                "category": "Perception",
                "subcategory": "Environmental Sound",
                "subsubcategory": subsubcategory,
                "difficulty": r.get("difficulty", "medium"),
            })

    if n_missing_audio_clotho:
        print(f"CẢNH BÁO: {n_missing_audio_clotho} sample ClothoAQA thiếu file audio local -- đã bỏ qua.")
    print(f"ClothoAQA: {len(merged_records) - n_before} sample đã merge.")

    with open(output_path, "w", encoding="utf-8") as f:
        for r in merged_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nTổng cộng {len(merged_records)} sample -- đã lưu {output_path}")
    print(f"Audio gộp tại {sound_out_dir} ({len(list(sound_out_dir.glob('*.wav')))} file .wav)")
    return merged_records


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mmau-manifest", required=True)
    parser.add_argument("--mmau-audio-dir", required=True)
    parser.add_argument("--clotho-manifest", required=True)
    parser.add_argument("--clotho-audio-dir", required=True)
    parser.add_argument("--sound-out-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    merge(
        Path(args.mmau_manifest), Path(args.mmau_audio_dir),
        Path(args.clotho_manifest), Path(args.clotho_audio_dir),
        Path(args.sound_out_dir), Path(args.output),
    )


if __name__ == "__main__":
    main()
