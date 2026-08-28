"""Load seed_samples.json (produced by prepare_data.py) into the database.

Idempotent: re-running updates metadata/ASR candidates for existing rows but
never overwrites final_asr_text/status, so it's safe to re-seed after adding
more samples without losing work already done by reviewers.

Usage: python seed.py [path/to/seed_samples.json]
"""
import json
import sys
from pathlib import Path

from database import Base, SessionLocal, engine
from models import Sample

DEFAULT_SEED_PATH = Path(__file__).parent / "seed_samples.json"


def run(seed_path: Path):
    if not seed_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {seed_path} -- chạy prepare_data.py trước để tạo file này."
        )

    Base.metadata.create_all(bind=engine)

    with open(seed_path, encoding="utf-8") as f:
        rows = json.load(f)

    db = SessionLocal()
    try:
        n_new, n_updated = 0, 0
        for i, row in enumerate(rows):
            existing = db.get(Sample, row["id"])
            if existing is None:
                db.add(Sample(
                    id=row["id"],
                    order_index=i,
                    source_dataset=row["source_dataset"],
                    speaker_id=row.get("speaker_id"),
                    duration_sec=row.get("duration_sec"),
                    audio_filename=row["audio_filename"],
                    original_audio_path=row.get("original_audio_path"),
                    asr_google=row.get("asr_google"),
                    asr_internal=row.get("asr_internal"),
                    asr_phowhisper=row.get("asr_phowhisper"),
                    asr_rover=row.get("asr_rover"),
                ))
                n_new += 1
            else:
                existing.order_index = i
                existing.source_dataset = row["source_dataset"]
                existing.speaker_id = row.get("speaker_id")
                existing.duration_sec = row.get("duration_sec")
                existing.audio_filename = row["audio_filename"]
                existing.original_audio_path = row.get("original_audio_path")
                existing.asr_google = row.get("asr_google")
                existing.asr_internal = row.get("asr_internal")
                existing.asr_phowhisper = row.get("asr_phowhisper")
                existing.asr_rover = row.get("asr_rover")
                n_updated += 1

        db.commit()
        print(f"Seed xong: {n_new} sample mới, {n_updated} sample đã cập nhật metadata "
              f"(final_asr_text/status của sample cũ được giữ nguyên).")
    finally:
        db.close()


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SEED_PATH
    run(path)
