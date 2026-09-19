"""Unit test OFFLINE cho extend-final (join field) và error_log/failed_ids (log sample lỗi)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from conftest import REPO

_HT_DIR = REPO / "src/test_set/datasets_qa/benchmark_qa/speech/cac_hien_tuong_dac_biet_trong_tieng_viet"

sys.path.insert(0, str(_HT_DIR))
sys.path.insert(0, str(REPO / "src/test_set/datasets_qa/benchmark_qa/tools"))
import auto_model_relay  # noqa: E402,F401  (đảm bảo path hợp lệ)
import error_log  # noqa: E402
import extend_final  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_extend_final_join_by_id_va_base_id(tmp_path):
    final = tmp_path / "final.jsonl"
    kept = tmp_path / "kept.jsonl"
    classified = tmp_path / "classified.jsonl"
    out = tmp_path / "final_extended.jsonl"
    _write_jsonl(final, [{"id": "cs-001__A", "question": "Q", "answer": "x", "choices": ["x"]}])
    _write_jsonl(kept, [{"id": "cs-001__A", "question_type": "A", "target_terms": ["wifi"], "difficulty": "easy"}])
    _write_jsonl(classified, [{"id": "cs-001", "text": "câu transcript thật"}])

    extend_final.main([
        "--final", str(final), "--output", str(out),
        "--join", f"{kept}::id::question_type,target_terms,difficulty",
        "--join", f"{classified}::base_id::transcript=text",
    ])
    r = json.loads(out.read_text(encoding="utf-8").strip())
    assert r["question_type"] == "A"
    assert r["target_terms"] == ["wifi"]
    assert r["difficulty"] == "easy"
    assert r["transcript"] == "câu transcript thật"


def test_error_log_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHMARK_QA_LOG_DIR", str(tmp_path))
    error_log.clear_failures("t")
    error_log.log_failure("t", "generate-questions", "cs-001__A", RuntimeError("boom"), model="m")
    error_log.log_failure("t", "debate", "cs-002", ValueError("quota"))
    assert error_log.failed_ids("t") == {"cs-001", "cs-002"}
    assert error_log.failed_ids("t", stage="debate") == {"cs-002"}
    rec = json.loads((tmp_path / "t" / "failures.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert rec["sample_id"] == "cs-001" and "boom" in rec["error"]
