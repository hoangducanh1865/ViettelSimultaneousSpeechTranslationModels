"""Chuẩn bị dữ liệu cho web labeling từ manifest.jsonl của TestSet_construction.ipynb.

Chạy script này Ở NƠI CÓ THỂ ĐỌC ĐƯỢC audio (trong Colab, sau khi đã có
manifest.jsonl + 3 bản ASR; hoặc trên máy local nếu Google Drive Desktop đã
sync path datasets/public/... xuống máy). Nó copy từng file audio (đổi tên
theo id để tránh trùng) và xuất seed_samples.json cho backend/seed.py nạp vào DB.

Sau khi chạy trong Colab: zip 2 thứ này lại rồi tải về, giải nén đè vào
`src/labeling_app/backend/audio_data/` và `src/labeling_app/backend/seed_samples.json`
trên máy local, rồi git add/commit/push như bình thường.

Usage:
    python prepare_data.py \\
        --manifest /content/drive/MyDrive/.../test_set_construction/manifest.jsonl \\
        --audio-out ./audio_data \\
        --seed-out ./seed_samples.json
"""
import argparse
import json
import re
import shutil
from pathlib import Path


def sanitize_filename(sample_id: str) -> str:
    # "VieSpeaker::id00000/0007.wav" -> "VieSpeaker__id00000_0007.wav"
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", sample_id)
    return safe if safe.lower().endswith(".wav") else f"{safe}.wav"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Đường dẫn tới manifest.jsonl")
    parser.add_argument("--audio-out", default="./audio_data", help="Thư mục đích chứa audio đã đổi tên")
    parser.add_argument("--seed-out", default="./seed_samples.json", help="Đường dẫn file seed_samples.json xuất ra")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    audio_out_dir = Path(args.audio_out)
    seed_out_path = Path(args.seed_out)
    audio_out_dir.mkdir(parents=True, exist_ok=True)

    assert manifest_path.exists(), f"Không tìm thấy manifest: {manifest_path}"

    seed_rows = []
    n_missing_audio = 0

    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            src_audio = Path(row["audio_path"])
            if not src_audio.exists():
                n_missing_audio += 1
                print(f"[BỎ QUA] không tìm thấy audio: {src_audio}")
                continue

            audio_filename = sanitize_filename(row["id"])
            shutil.copyfile(src_audio, audio_out_dir / audio_filename)

            seed_rows.append({
                "id": row["id"],
                "source_dataset": row["source_dataset"],
                "speaker_id": row.get("speaker_id"),
                "duration_sec": row.get("duration_sec"),
                "audio_filename": audio_filename,
                "original_audio_path": row["audio_path"],
                "asr_google": (row.get("asr_google") or {}).get("text"),
                "asr_internal": (row.get("asr_internal") or {}).get("text"),
                "asr_phowhisper": (row.get("asr_phowhisper") or {}).get("text"),
                "asr_rover": row.get("asr_rover"),
            })

    with open(seed_out_path, "w", encoding="utf-8") as f:
        json.dump(seed_rows, f, ensure_ascii=False, indent=2)

    print(f"\nĐã copy {len(seed_rows)} file audio vào {audio_out_dir}")
    print(f"Đã ghi {seed_out_path}")
    if n_missing_audio:
        print(f"CẢNH BÁO: {n_missing_audio} sample bị bỏ qua vì không tìm thấy file audio gốc.")


if __name__ == "__main__":
    main()
