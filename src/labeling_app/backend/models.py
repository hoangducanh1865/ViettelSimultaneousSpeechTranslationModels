from sqlalchemy import Column, DateTime, Float, Integer, String, Text, func

from database import Base


class Sample(Base):
    __tablename__ = "samples"

    id = Column(String, primary_key=True)
    order_index = Column(Integer, nullable=False, default=0)
    source_dataset = Column(String, nullable=False)
    speaker_id = Column(String, nullable=True)
    duration_sec = Column(Float, nullable=True)
    audio_filename = Column(String, nullable=False)
    original_audio_path = Column(String, nullable=True)

    asr_google = Column(Text, nullable=True)
    asr_internal = Column(Text, nullable=True)
    asr_phowhisper = Column(Text, nullable=True)
    asr_rover = Column(Text, nullable=True)

    final_asr_text = Column(Text, nullable=True)
    status = Column(String, nullable=False, default="pending")  # "pending" | "done"

    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
