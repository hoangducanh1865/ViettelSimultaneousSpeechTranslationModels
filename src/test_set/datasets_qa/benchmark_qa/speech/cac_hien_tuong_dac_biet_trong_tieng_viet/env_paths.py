"""Path resolution cho 2 "vị trí" (--location): "drive" (Colab, Google Drive) hoặc "local" (máy
người dùng, thư mục <repo>/data -- data root local, KHÔNG còn dùng stuff/). Đây là nơi DUY NHẤT
chứa logic detect-môi-trường/ghép-path Drive-vs-local -- mọi script CLI (auto_model_relay.py,
code_switching_pipeline.py, code_switching_qa_pipeline.py, build_debate_seed.py, filter_qa_pipeline.py)
chỉ nhận --location rồi gọi hàm ở đây, KHÔNG tự viết lại logic này."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

LOCATIONS = ("drive", "local")

# Data root trên Google Drive (Colab) -- mọi subpath giống hệt data root local để 2 môi trường
# dùng chung một layout tương đối.
DRIVE_QA_DATASETS_DIR = Path(
    "/content/drive/MyDrive/it/vdt/voice_agent_for_edge_device/datasets/public/qa_datasets"
)
DRIVE_CREDENTIALS_ENV_FILE = Path("/content/drive/MyDrive/it/vdt/voice_agent_for_edge_device/.env")


def repo_root() -> Path:
    """Path(__file__) tự trỏ lên gốc repo -- KHÔNG phụ thuộc CWD của kernel đang chạy."""
    return Path(__file__).resolve().parents[6]


def _check_location(location: str) -> None:
    if location not in LOCATIONS:
        raise ValueError(f"location={location!r} không hợp lệ -- phải là 1 trong {LOCATIONS}.")


def data_root(location: str) -> Path:
    """Gốc dữ liệu QA theo location:
    - drive -> /content/drive/.../qa_datasets
    - local -> <repo>/data  (thay cho stuff/ cũ)
    Có thể override bằng biến môi trường QA_DATASETS_DIR."""
    _check_location(location)
    override = os.environ.get("QA_DATASETS_DIR")
    if override:
        return Path(override)
    if location == "drive":
        return DRIVE_QA_DATASETS_DIR
    return repo_root() / "data"


def base_dir(location: str, *, subpath: str) -> Path:
    """data_root(location)/subpath -- CÙNG subpath cho cả drive lẫn local."""
    return data_root(location) / subpath


def code_switching_dir(location: str) -> Path:
    """Override bằng CODE_SWITCHING_DIR nếu có (dùng cho test cô lập vào thư mục tạm)."""
    override = os.environ.get("CODE_SWITCHING_DIR")
    if override:
        return Path(override)
    return base_dir(location, subpath="benchmark_qa/speech/trich_xuat_thong_tin/code_switching")


def knowledge_dir(location: str) -> Path:
    """Override bằng KNOWLEDGE_DIR nếu có."""
    override = os.environ.get("KNOWLEDGE_DIR")
    if override:
        return Path(override)
    return base_dir(location, subpath="benchmark_qa/knowledge")


def credentials_env_file(location: str) -> Path:
    """File .env chứa credential (GEMINI_API_KEY/OPENAI_API_KEY, format
    `KEY="value" # base_url # model`):
    - drive -> /content/drive/.../voice_agent_for_edge_device/.env
    - local -> <repo>/.env
    Override bằng ENV_FILE nếu có."""
    _check_location(location)
    override = os.environ.get("ENV_FILE")
    if override:
        return Path(override)
    if location == "drive":
        return DRIVE_CREDENTIALS_ENV_FILE
    return repo_root() / ".env"


def hien_tuong_src_dir() -> Path:
    return repo_root() / "src/test_set/datasets_qa/benchmark_qa/speech/cac_hien_tuong_dac_biet_trong_tieng_viet"


def code_switching_src_dir() -> Path:
    return repo_root() / "src/test_set/datasets_qa/benchmark_qa/speech/trich_xuat_thong_tin/code_switching"


def gemini_service_account_path(location: str) -> Path:
    """CHỈ hợp lệ khi location="drive" (Vertex AI service account, người dùng tự upload lên
    Drive) -- raise ValueError nếu gọi với "local" (dùng credentials_env_file() thay thế)."""
    if location != "drive":
        raise ValueError('gemini_service_account_path() chỉ hợp lệ khi location="drive".')
    return Path("/content/drive/MyDrive/it/vdt/voice_agent_for_edge_device/gemini_service_account.json")


def default_env_file(location: str) -> Path:
    """(Giữ tên cũ cho tương thích) -> credentials_env_file(location)."""
    return credentials_env_file(location)


def add_location_arg(parser: argparse.ArgumentParser, *, default: str = "drive") -> None:
    """Helper dùng chung cho mọi script CLI cần chọn Drive/Local."""
    parser.add_argument("--location", choices=LOCATIONS, default=default, help='"drive" (Colab, Google Drive) hoặc "local" (máy người dùng, <repo>/data).')
