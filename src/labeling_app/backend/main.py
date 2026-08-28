import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from database import Base, engine
from routers import export, samples

AUDIO_DIR = Path(__file__).parent / "audio_data"
AUDIO_ACCESS_KEY = os.environ.get("AUDIO_ACCESS_KEY", "")
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",")]

Base.metadata.create_all(bind=engine)

app = FastAPI(title="ASR Labeling API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(samples.router)
app.include_router(export.router)


@app.get("/audio/{filename}")
def get_audio(filename: str, key: str = Query(default="")):
    if AUDIO_ACCESS_KEY and key != AUDIO_ACCESS_KEY:
        raise HTTPException(status_code=403, detail="Sai hoặc thiếu access key")

    # Chặn path traversal -- chỉ phục vụ file nằm thẳng trong AUDIO_DIR.
    path = (AUDIO_DIR / filename).resolve()
    if AUDIO_DIR.resolve() not in path.parents or not path.exists():
        raise HTTPException(status_code=404, detail="Không tìm thấy file audio")

    return FileResponse(path, media_type="audio/wav")


@app.get("/health")
def health():
    return {"status": "ok"}
