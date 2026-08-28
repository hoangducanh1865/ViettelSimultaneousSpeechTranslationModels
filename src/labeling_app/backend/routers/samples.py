import os

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import asc
from sqlalchemy.orm import Session

from database import get_db
from models import Sample
from schemas import SampleDetail, SampleListItem, SubmitPayload

router = APIRouter(prefix="/api/samples", tags=["samples"])

AUDIO_ACCESS_KEY = os.environ.get("AUDIO_ACCESS_KEY", "")


def _audio_url(sample: Sample) -> str:
    suffix = f"?key={AUDIO_ACCESS_KEY}" if AUDIO_ACCESS_KEY else ""
    return f"/audio/{sample.audio_filename}{suffix}"


def _to_detail(sample: Sample) -> SampleDetail:
    return SampleDetail(
        id=sample.id,
        order_index=sample.order_index,
        source_dataset=sample.source_dataset,
        speaker_id=sample.speaker_id,
        duration_sec=sample.duration_sec,
        audio_url=_audio_url(sample),
        asr_google=sample.asr_google,
        asr_internal=sample.asr_internal,
        asr_phowhisper=sample.asr_phowhisper,
        asr_rover=sample.asr_rover,
        final_asr_text=sample.final_asr_text,
        mt_en=sample.mt_en,
        final_mt_text=sample.final_mt_text,
        status=sample.status,
        updated_at=sample.updated_at,
    )


@router.get("", response_model=list[SampleListItem])
def list_samples(db: Session = Depends(get_db)):
    return db.query(Sample).order_by(asc(Sample.order_index)).all()


@router.get("/{sample_id:path}", response_model=SampleDetail)
def get_sample(sample_id: str, db: Session = Depends(get_db)):
    sample = db.get(Sample, sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy sample")
    return _to_detail(sample)


@router.post("/{sample_id:path}/submit", response_model=SampleDetail)
def submit_sample(sample_id: str, payload: SubmitPayload, db: Session = Depends(get_db)):
    sample = db.get(Sample, sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy sample")

    asr_text = payload.final_asr_text.strip()
    mt_text = payload.final_mt_text.strip()
    if not asr_text:
        raise HTTPException(status_code=400, detail="final_asr_text không được để trống")
    if not mt_text:
        raise HTTPException(status_code=400, detail="final_mt_text không được để trống")

    sample.final_asr_text = asr_text
    sample.final_mt_text = mt_text
    sample.status = "done"
    db.commit()
    db.refresh(sample)

    return _to_detail(sample)


@router.post("/{sample_id:path}/reset", response_model=SampleDetail)
def reset_sample(sample_id: str, db: Session = Depends(get_db)):
    sample = db.get(Sample, sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy sample")

    sample.final_asr_text = None
    sample.final_mt_text = None
    sample.status = "pending"
    db.commit()
    db.refresh(sample)

    return _to_detail(sample)
