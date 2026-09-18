from conftest import assert_valid_final, require_real_api, run_dispatcher
from fixtures import build_code_switching


def test_code_switching_e2e_with_debate(e2e_root):
    """Code-switching: scan/merge -> debate 2 API THẬT -> classify -> generate -> lọc 2 API -> finalize."""
    require_real_api()
    build_code_switching(e2e_root)

    run_dispatcher(e2e_root, ["code-switching", "--debate", "--debate-mode", "api"])

    cs = e2e_root / "benchmark_qa/speech/trich_xuat_thong_tin/code_switching"
    assert (cs / "knowledge_code_switching.json").exists(), "debate không sinh knowledge graph"
    assert_valid_final(cs / "code_switching_openai_kept_final.jsonl", min_samples=1)
