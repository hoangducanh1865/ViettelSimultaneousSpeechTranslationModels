"""Hạ tầng test E2E cho benchmark_qa (chạy REAL API trên local, gọi qua bash dispatcher).

Bật bằng biến môi trường RUN_REAL_API=1. Test tạo dữ liệu nhỏ (2-3 mẫu) trong <repo>/data/_e2e/
rồi chạy `run/benchmark_qa.sh <task> --location local ...`, kiểm tra sinh ra được sample QA hợp lệ.

Chạy:
    RUN_REAL_API=1 .venv/bin/python -m pytest tests/benchmark_qa -q
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "src/test_set/datasets_qa/benchmark_qa/run/benchmark_qa.sh"
ENV_FILE = REPO / ".env"
E2E_BASE = REPO / "data" / "_e2e"

PYTHON = sys.executable  # test chạy trong .venv -> dùng luôn interpreter này cho pipeline
RUN_TIMEOUT = 1800  # 30 phút/task (debate code-switching lâu nhất)


def _env_has_credentials() -> bool:
    if not ENV_FILE.exists():
        return False
    text = ENV_FILE.read_text(encoding="utf-8")
    return "GEMINI_API_KEY" in text and "OPENAI_API_KEY" in text


def require_real_api():
    if os.environ.get("RUN_REAL_API") != "1":
        pytest.skip("Bật RUN_REAL_API=1 để chạy test E2E real API.")
    if not _env_has_credentials():
        pytest.skip(f"Thiếu GEMINI_API_KEY/OPENAI_API_KEY trong {ENV_FILE}.")


@pytest.fixture()
def e2e_root(request) -> Path:
    """Tạo thư mục data/_e2e/<tên-test>/ sạch (nằm TRONG data/, không dùng stuff/)."""
    name = request.node.name.replace("[", "_").replace("]", "_")
    root = E2E_BASE / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def run_dispatcher(root: Path, task_args: list[str], *, timeout: int = RUN_TIMEOUT) -> subprocess.CompletedProcess:
    """Chạy benchmark_qa.sh với data root = <root>; trả CompletedProcess (raise nếu lỗi)."""
    cmd = [
        "bash", str(RUNNER), *task_args,
        "--location", "local",
        "--qa-datasets-dir", str(root),
        "--knowledge-dir", str(root / "knowledge"),
        "--env-file", str(ENV_FILE),
        "--python", PYTHON,
    ]
    # Cho phép override model qua biến môi trường nếu proxy local dùng tên model khác mặc định.
    if os.environ.get("E2E_GEMINI_MODEL"):
        cmd += ["--gemini-model", os.environ["E2E_GEMINI_MODEL"]]
    if os.environ.get("E2E_OPENAI_MODEL"):
        cmd += ["--openai-model", os.environ["E2E_OPENAI_MODEL"]]
    env = dict(os.environ)
    env["RUN_REAL_API"] = "1"
    print(f"\n$ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        print(proc.stdout[-4000:])
        print(proc.stderr[-4000:])
    assert proc.returncode == 0, f"dispatcher thất bại (exit {proc.returncode})"
    return proc


# ---------------------------------------------------------------------------------------------
# Kiểm tra record final hợp lệ
# ---------------------------------------------------------------------------------------------
REQUIRED_FINAL_FIELDS = [
    "id", "audio_id", "question", "choices", "answer", "dataset", "task", "split",
    "category", "sub-category", "difficulty",
]


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def assert_valid_final(path: Path, min_samples: int = 1) -> list[dict]:
    records = load_jsonl(path)
    assert len(records) >= min_samples, f"{path} chỉ có {len(records)} record (cần >= {min_samples})"
    for r in records:
        for field in REQUIRED_FINAL_FIELDS:
            assert r.get(field) is not None, f"record {r.get('id')} thiếu field {field!r}"
        assert len(r["choices"]) == 4, f"{r['id']}: cần đúng 4 choices"
        assert len(set(r["choices"])) == 4, f"{r['id']}: choices trùng nhau"
        assert r["answer"] in r["choices"], f"{r['id']}: answer không nằm trong choices"
    return records
