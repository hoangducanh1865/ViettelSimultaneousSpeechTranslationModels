"""One-time migration of the OLD per-direction pipeline outputs into the
NEW unified `tmp.json` working file (see `manifest_io.py`). Run once per
direction against real Drive data; from then on `tmp.json` is the sole
source of truth going forward -- the legacy files are read-only inputs
here and are never modified.

vi_en (`manifest.jsonl` -> `tmp.json`, generation_mode="heavy_pipeline"):
    `manifest.jsonl`'s `final_asr_text`/`final_mt_text` were NEVER run
    through `text_normalize.normalize_text()` (confirmed: only the
    *derived* `test_set_final.jsonl`'s `text_en` was, as a one-time
    transform never written back) -- so `manifest.jsonl` is already the
    correct natural-casing source for BOTH fields, no extra file needed.
    Every raw field (asr_google/asr_internal/asr_phowhisper/asr_rover/
    mt_en/final_asr_text/final_mt_text) is carried over verbatim; rows that
    haven't finished human review yet (empty final_asr_text/final_mt_text)
    correctly end up with `text_vi`/`text_en` still null, exactly
    reflecting where the old pipeline left them.

en_vi (`phost_test_final.jsonl` + `phost_test_final_cased.jsonl` ->
`tmp.json`, generation_mode="phost"):
    Opposite asymmetry: `phost_test_final.jsonl`'s `text_vi` IS already
    normalized (PhoST.ipynb's own mục 4), `text_en` is natural/untouched.
    The pre-normalization backup `phost_test_final_cased.jsonl` is joined
    by `id` to recover the natural-casing `text_vi_cased`; if a row is
    missing from the backup (shouldn't happen, but don't silently trust
    it), fall back to the already-normalized `text_vi` with a printed
    warning instead of crashing.

Usage:
    python migrate_legacy_to_tmp.py --direction vi_en \\
        --testset-dir /content/drive/MyDrive/.../datasets_ast/test_set_construction
    python migrate_legacy_to_tmp.py --direction en_vi \\
        --testset-dir /content/drive/MyDrive/.../datasets_ast/PhoST
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from dataset_versioning import normalize_samples
from manifest_io import derive_final_fields, new_envelope, save_tmp


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def migrate_vi_en(testset_dir: Path) -> dict:
    manifest_path = testset_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise SystemExit(f"Không tìm thấy {manifest_path}")

    envelope = new_envelope(direction="vi_en", generation_mode="heavy_pipeline")
    n_final = 0
    for row in _load_jsonl(manifest_path):
        sample = dict(row)  # copy toàn bộ field gốc (asr_google/.../final_mt_text) nguyên vẹn
        derive_final_fields(sample, direction="vi_en", generation_mode="heavy_pipeline")
        if sample.get("text_vi") and sample.get("text_en"):
            n_final += 1
        envelope["samples"][sample["id"]] = sample

    normalize_samples(envelope["samples"])
    print(f"vi_en: {len(envelope['samples'])} sample, {n_final} đã có đủ text_vi/text_en (đã review xong).")
    return envelope


def migrate_en_vi(testset_dir: Path) -> dict:
    final_path = testset_dir / "phost_test_final.jsonl"
    cased_path = testset_dir / "phost_test_final_cased.jsonl"
    if not final_path.exists():
        raise SystemExit(f"Không tìm thấy {final_path}")

    cased_by_id: dict[str, dict] = {}
    if cased_path.exists():
        cased_by_id = {row["id"]: row for row in _load_jsonl(cased_path)}
    else:
        print(f"CẢNH BÁO: không tìm thấy {cased_path} -- sẽ không phục hồi được text_vi có dấu câu gốc.")

    envelope = new_envelope(direction="en_vi", generation_mode="phost")
    n_missing_cased = 0
    for row in _load_jsonl(final_path):
        sample_id = row["id"]
        cased_row = cased_by_id.get(sample_id)
        if cased_row is not None:
            text_vi_cased = cased_row["text_vi"]
        else:
            n_missing_cased += 1
            text_vi_cased = row["text_vi"]  # đã lowercase/mất dấu câu -- chấp nhận tạm, không có bản gốc

        sample = {
            "id": sample_id,
            "source_dataset": row["source_dataset"],
            "speaker_id": row.get("speaker_id"),
            "audio_path": row["audio_path"],
            "duration_sec": row.get("duration_sec"),
            "text_vi_cased": text_vi_cased,
            "text_en_cased": row["text_en"],  # text_en chưa từng bị chuẩn hoá, giữ nguyên là bản gốc
        }
        envelope["samples"][sample_id] = sample

    normalize_samples(envelope["samples"])
    if n_missing_cased:
        print(f"CẢNH BÁO: {n_missing_cased}/{len(envelope['samples'])} sample thiếu bản text_vi có dấu câu gốc "
              f"trong {cased_path.name} -- dùng tạm bản đã chuẩn hoá sẵn làm text_vi_cased.")
    print(f"en_vi: {len(envelope['samples'])} sample migrated từ PhoST.")
    return envelope


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", required=True, choices=["vi_en", "en_vi"])
    parser.add_argument("--testset-dir", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None, help="Mặc định: <testset-dir>/tmp.json")
    args = parser.parse_args(argv)

    out_path = args.out or (args.testset_dir / "tmp.json")
    if out_path.exists():
        raise SystemExit(
            f"{out_path} đã tồn tại -- migrate chỉ chạy 1 lần trên 1 thư mục trống, "
            "xoá/di chuyển file cũ đi nếu thực sự muốn chạy lại (sẽ ghi đè)."
        )

    if args.direction == "vi_en":
        envelope = migrate_vi_en(args.testset_dir)
    else:
        envelope = migrate_en_vi(args.testset_dir)

    save_tmp(out_path, envelope)
    print(f"Đã ghi {out_path}")


if __name__ == "__main__":
    main()
