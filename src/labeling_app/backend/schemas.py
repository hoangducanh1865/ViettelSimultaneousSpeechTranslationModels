from datetime import datetime

from pydantic import BaseModel


class SampleListItem(BaseModel):
    id: str
    order_index: int
    source_dataset: str
    speaker_id: str | None
    duration_sec: float | None
    status: str

    class Config:
        from_attributes = True


class SampleDetail(BaseModel):
    id: str
    order_index: int
    source_dataset: str
    speaker_id: str | None
    duration_sec: float | None
    audio_url: str
    asr_google: str | None
    asr_internal: str | None
    asr_phowhisper: str | None
    asr_rover: str | None
    final_asr_text: str | None
    status: str
    updated_at: datetime | None

    class Config:
        from_attributes = True


class SubmitPayload(BaseModel):
    final_asr_text: str
