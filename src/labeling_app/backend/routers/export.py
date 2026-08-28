import csv
import io

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy import asc
from sqlalchemy.orm import Session

from database import get_db
from models import Sample

router = APIRouter(prefix="/api/export", tags=["export"])

# Cột khớp CHÍNH XÁC với mt_manual_check.csv trong TestSet_construction.ipynb
# (mục 4, sau khi gộp check ASR + dịch thành 1 bước), để tải file này về rồi thả
# thẳng vào Drive là GATE ở mục 5 của notebook đọc được luôn.
CSV_COLUMNS = ["id", "audio_path", "asr_rover", "final_asr_text", "mt_en", "final_mt_text"]


@router.get("/mt-check.csv")
def export_mt_check_csv(db: Session = Depends(get_db)):
    samples = db.query(Sample).order_by(asc(Sample.order_index)).all()

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    for s in samples:
        writer.writerow({
            "id": s.id,
            "audio_path": s.original_audio_path or "",
            "asr_rover": s.asr_rover or "",
            "final_asr_text": s.final_asr_text or "",
            "mt_en": s.mt_en or "",
            "final_mt_text": s.final_mt_text or "",
        })

    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=mt_manual_check.csv"},
    )
