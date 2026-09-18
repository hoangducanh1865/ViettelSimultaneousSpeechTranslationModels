"""Path resolution cho 2 "vị trí" (--location): "drive" (Colab, Google Drive) hoặc "local" (máy
người dùng, thư mục repo/stuff/...). Đây là nơi DUY NHẤT chứa logic detect-môi-trường/ghép-path
Drive-vs-local -- mọi script CLI (auto_model_relay.py, code_switching_pipeline.py,
code_switching_qa_pipeline.py, build_debate_seed.py) chỉ nhận --location rồi gọi hàm ở đây, KHÔNG
tự viết lại logic này (từng bị lặp lại/viết sai trực tiếp trong nhiều cell notebook, là nguồn gốc
của rất nhiều lỗi vụn vặt trước đó -- IS_COLAB try/except, dò REPO_ROOT phụ thuộc CWD, ghép chuỗi
credential-arg khác nhau ở từng cell)."""

from __future__ import annotations

import argparse
from pathlib import Path

LOCATIONS = ("drive", "local")


def repo_root() -> Path:
    """Path(__file__) tự trỏ lên gốc repo -- KHÔNG phụ thuộc CWD của kernel đang chạy (notebook
    thường có CWD là thư mục chứa .ipynb, không phải gốc repo -- nguyên nhân 1 bug AssertionError
    trước đó khi cố dò ngược bằng CWD)."""
    return Path(__file__).resolve().parents[6]


def drive_qa_datasets_dir() -> Path:
    return Path("/content/drive/MyDrive/it/vdt/voice_agent_for_edge_device/datasets/public/qa_datasets")


def _check_location(location: str) -> None:
    if location not in LOCATIONS:
        raise ValueError(f"location={location!r} không hợp lệ -- phải là 1 trong {LOCATIONS}.")


def base_dir(location: str, *, drive_subpath: str, local_subpath: str) -> Path:
    """location="drive" -> drive_qa_datasets_dir()/drive_subpath.
    location="local" -> repo_root()/"stuff"/local_subpath."""
    _check_location(location)
    if location == "drive":
        return drive_qa_datasets_dir() / drive_subpath
    return repo_root() / "stuff" / local_subpath


def code_switching_dir(location: str) -> Path:
    return base_dir(
        location,
        drive_subpath="benchmark_qa/speech/trich_xuat_thong_tin/code_switching",
        local_subpath="benchmark_qa/speech/trich_xuat_thong_tin/code_switching",
    )


def knowledge_dir(location: str) -> Path:
    """drive -> QA_DATASETS_DIR/benchmark_qa/knowledge (cây thư mục dùng chung cho mọi task).
    local -> code_switching_dir(location) -- Code-switching hiện là task DUY NHẤT chạy local, nên
    gộp luôn knowledge_*.json vào cùng thư mục cho gọn, không cần cây thư mục riêng."""
    _check_location(location)
    if location == "drive":
        return drive_qa_datasets_dir() / "benchmark_qa/knowledge"
    return code_switching_dir(location)


def hien_tuong_src_dir() -> Path:
    return repo_root() / "src/test_set/datasets_qa/benchmark_qa/speech/cac_hien_tuong_dac_biet_trong_tieng_viet"


def code_switching_src_dir() -> Path:
    return repo_root() / "src/test_set/datasets_qa/benchmark_qa/speech/trich_xuat_thong_tin/code_switching"


def gemini_service_account_path(location: str) -> Path:
    """CHỈ hợp lệ khi location="drive" (Vertex AI service account, người dùng tự upload lên
    Drive) -- raise ValueError nếu gọi với "local" (dùng default_env_file() thay thế, vì local
    dùng chung 1 file .env cho cả Gemini lẫn OpenAI, không có service account riêng)."""
    if location != "drive":
        raise ValueError('gemini_service_account_path() chỉ hợp lệ khi location="drive".')
    return Path("/content/drive/MyDrive/it/vdt/voice_agent_for_edge_device/gemini_service_account.json")


def default_env_file(location: str) -> Path:
    """CHỈ hợp lệ khi location="local" -- trả về code_switching_dir(location)/".env" (người dùng
    tự tạo file này, xem docstring auto_model_relay.py về định dạng)."""
    if location != "local":
        raise ValueError('default_env_file() chỉ hợp lệ khi location="local".')
    return code_switching_dir(location) / ".env"


def add_location_arg(parser: argparse.ArgumentParser, *, default: str = "drive") -> None:
    """Helper dùng chung cho mọi script CLI cần chọn Drive/Local."""
    parser.add_argument("--location", choices=LOCATIONS, default=default, help='"drive" (Colab, Google Drive) hoặc "local" (máy người dùng, thư mục repo/stuff/...).')
