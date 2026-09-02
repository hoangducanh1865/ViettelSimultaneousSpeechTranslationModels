"""Load seed_samples.json (produced by prepare_data.py) into the database.

Idempotent + safe to re-run: uses a bulk INSERT ... ON CONFLICT DO UPDATE per
batch instead of one SELECT+INSERT/UPDATE per row, so re-seeding thousands of
rows against a remote DB (e.g. Render Postgres in a different region) takes
seconds instead of tens of minutes -- the old row-by-row loop paid one full
network round trip per row just to check existence. final_asr_text/
final_mt_text/status are NEVER included in the update set, so work already
done by reviewers through the web app is preserved no matter how many times
this is re-run.

Sau khi upsert, XOÁ khỏi DB mọi sample không còn xuất hiện trong seed_samples.json
(vd bị loại bởi bộ lọc tiếng Anh/hallucination mục 2.5, hoặc audio gốc không tìm
thấy) -- nếu không, những sample rác đã seed từ 1 lần chạy trước sẽ tồn tại vĩnh
viễn trong DB dù bạn lọc lại manifest.jsonl sạch đến đâu và seed lại bao nhiêu lần,
vì upsert chỉ update/insert, không bao giờ tự xoá row thừa.

Usage: python seed.py [path/to/seed_samples.json]
"""
import json
import sys
from pathlib import Path

from tqdm import tqdm
from sqlalchemy import delete, select

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
    seed_ids = {row["id"] for row in rows}

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

        # Xoá sample thừa (không còn trong seed_samples.json) -- vd bị lọc nhiễu ở
        # mục 2.5 sau lần seed trước, hoặc audio gốc không tìm thấy khi chạy
        # prepare_data.py. Không xoá được bằng upsert, phải làm riêng.
        existing_ids = set(db.execute(select(Sample.id)).scalars().all())
        orphan_ids = sorted(existing_ids - seed_ids)
        if orphan_ids:
            n_reviewed = db.execute(
                select(Sample.id).where(
                    Sample.id.in_(orphan_ids), Sample.final_asr_text.is_not(None)
                )
            ).scalars().all()
            if n_reviewed:
                print(f"CẢNH BÁO: {len(n_reviewed)}/{len(orphan_ids)} sample bị xoá ĐÃ ĐƯỢC REVIEW "
                      f"tay trước đó (final_asr_text có giá trị) -- công review đó sẽ mất theo: "
                      f"{n_reviewed[:10]}{'...' if len(n_reviewed) > 10 else ''}")

            for i in range(0, len(orphan_ids), BATCH_SIZE):
                batch_ids = orphan_ids[i : i + BATCH_SIZE]
                db.execute(delete(Sample).where(Sample.id.in_(batch_ids)))
                db.commit()
            print(f"Đã xoá {len(orphan_ids)} sample thừa (không còn trong seed_samples.json).")
        else:
            print("Không có sample thừa cần xoá.")
    finally:
        db.close()


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SEED_PATH
    run(path)
