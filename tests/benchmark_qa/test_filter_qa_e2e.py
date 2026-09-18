import json
import os
from pathlib import Path

from conftest import ENV_FILE, PYTHON, REPO, require_real_api


def test_filter_qa_pipeline_e2e(e2e_root):
    """Kiểm tra filter_qa_pipeline.py (lọc 2 API, prompt theo task) chạy được trên local."""
    require_real_api()
    import subprocess

    pipeline = (
        REPO / "src/test_set/datasets_qa/benchmark_qa/speech/"
        "cac_hien_tuong_dac_biet_trong_tieng_viet/filter_qa_pipeline.py"
    )
    pre_final = e2e_root / "pre_final.jsonl"
    with open(pre_final, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "id": "q1", "question": "Có thể thêm từ nào để làm nhẹ màu sắc?",
            "choices": ["xanh xao", "nhè nhẹ", "mạnh mẽ", "đẹp đẽ"], "answer": "xanh xao",
            "base_word": "xanh", "tone_register": "high", "question_type": "cloze-tone-harmony",
        }, ensure_ascii=False) + "\n")

    kept = e2e_root / "kept.jsonl"
    rules = e2e_root / "rules.json"
    cmd = [
        str(PYTHON), str(pipeline), "filter-qa",
        "--task", "tu_lay", "--pre-final", str(pre_final),
        "--kept-output", str(kept), "--rules-output", str(rules),
        "--provider", "gemini", "--location", "local", "--env-file", str(ENV_FILE),
    ]
    # Chỉ override model nếu có biến môi trường; mặc định dùng model trong .env.
    if os.environ.get("E2E_GEMINI_MODEL"):
        cmd += ["--model", os.environ["E2E_GEMINI_MODEL"]]
    proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:])
    assert proc.returncode == 0
    assert kept.exists(), "filter không ghi file kept"
    assert rules.exists(), "filter không ghi file rules"
    summaries = json.loads(Path(rules).read_text(encoding="utf-8")).get("criteria_summaries")
    assert summaries, "rules.json thiếu criteria_summaries"
