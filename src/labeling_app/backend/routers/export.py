import csv
import io

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy import asc
from sqlalchemy.orm import Session

from database import get_db
from models import Sample

router = APIRouter(prefix="/api/export", tags=["export"])

# Cột khớp CHÍNH XÁC với asr_manual_check.csv trong TestSet_construction.ipynb,
# để tải file này về rồi thả thẳng vào Drive là GATE 1 của notebook đọc được luôn.
CSV_COLUMNS = [
    "id", "source_dataset", "audio_path", "duration_sec",
    "asr_google", "asr_internal", "asr_phowhisper", "asr_rover", "final_asr_text",
]


@router.get("/asr-check.csv")
def export_asr_check_csv(db: Session = Depends(get_db)):
    samples = db.query(Sample).order_by(asc(Sample.order_index)).all()

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    for s in samples:
        writer.writerow({
            "id": s.id,
            "source_dataset": s.source_dataset,
            "audio_path": s.original_audio_path or "",
            "duration_sec": round(s.duration_sec, 2) if s.duration_sec is not None else "",
            "asr_google": s.asr_google or "",
            "asr_internal": s.asr_internal or "",
            "asr_phowhisper": s.asr_phowhisper or "",
            "asr_rover": s.asr_rover or "",
            "final_asr_text": s.final_asr_text or "",
        })

    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=asr_manual_check.csv"},
    )
