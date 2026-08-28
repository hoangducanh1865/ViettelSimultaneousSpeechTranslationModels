"""Load seed_samples.json (produced by prepare_data.py) into the database.

Idempotent + safe to re-run: uses a bulk INSERT ... ON CONFLICT DO UPDATE per
batch instead of one SELECT+INSERT/UPDATE per row, so re-seeding thousands of
rows against a remote DB (e.g. Render Postgres in a different region) takes
seconds instead of tens of minutes -- the old row-by-row loop paid one full
network round trip per row just to check existence. final_asr_text/
final_mt_text/status are NEVER included in the update set, so work already
done by reviewers through the web app is preserved no matter how many times
this is re-run.

Usage: python seed.py [path/to/seed_samples.json]
"""
import json
import sys
from pathlib import Path

from tqdm import tqdm

from database import Base, SessionLocal, engine
from models import Sample

DEFAULT_SEED_PATH = Path(__file__).parent / "seed_samples.json"
BATCH_SIZE = 300

# Cột được phép seed ghi đè khi sample đã tồn tại -- KHÔNG bao giờ đụng tới
# final_asr_text/final_mt_text/status, đó là dữ liệu do người review tạo ra
# qua web, seed.py không được phép làm mất dù chạy lại bao nhiêu lần.
UPSERT_COLUMNS = [
    "order_index", "source_dataset", "speaker_id", "duration_sec",
    "audio_filename", "original_audio_path",
    "asr_google", "asr_internal", "asr_phowhisper", "asr_rover", "mt_en",
]


def _insert_builder():
    # Postgres và SQLite (dev local) dùng 2 module dialect khác nhau nhưng cùng
    # API on_conflict_do_update -- chọn đúng cái khớp với DATABASE_URL đang dùng.
    if engine.dialect.name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    else:
        from sqlalchemy.dialects.sqlite import insert
    return insert


def run(seed_path: Path):
    if not seed_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {seed_path} -- chạy prepare_data.py trước để tạo file này."
        )

    Base.metadata.create_all(bind=engine)

    with open(seed_path, encoding="utf-8") as f:
        rows = json.load(f)

    insert = _insert_builder()
    n_total = len(rows)

    db = SessionLocal()
    try:
        for batch_start in tqdm(range(0, n_total, BATCH_SIZE), desc="Seeding", unit="batch"):
            batch = rows[batch_start : batch_start + BATCH_SIZE]
            values = [
                {
                    "id": row["id"],
                    "order_index": i,
                    "source_dataset": row["source_dataset"],
                    "speaker_id": row.get("speaker_id"),
                    "duration_sec": row.get("duration_sec"),
                    "audio_filename": row["audio_filename"],
                    "original_audio_path": row.get("original_audio_path"),
                    "asr_google": row.get("asr_google"),
                    "asr_internal": row.get("asr_internal"),
                    "asr_phowhisper": row.get("asr_phowhisper"),
                    "asr_rover": row.get("asr_rover"),
                    "mt_en": row.get("mt_en"),
                    "status": "pending",  # chỉ áp dụng khi INSERT mới -- bị bỏ qua khi conflict, xem set_ bên dưới
                }
                for i, row in enumerate(batch, start=batch_start)
            ]

            stmt = insert(Sample).values(values)
            stmt = stmt.on_conflict_do_update(
                index_elements=["id"],
                set_={col: getattr(stmt.excluded, col) for col in UPSERT_COLUMNS},
            )
            db.execute(stmt)
            db.commit()  # commit theo từng batch -- lỗi/rớt mạng giữa chừng không mất phần đã xong

        print(f"Seed xong: {n_total} sample (upsert theo batch {BATCH_SIZE}). "
              "final_asr_text/final_mt_text/status của sample cũ được giữ nguyên.")
    finally:
        db.close()


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SEED_PATH
    run(path)
