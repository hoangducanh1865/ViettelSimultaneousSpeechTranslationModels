from conftest import assert_valid_final, require_real_api, run_dispatcher
from fixtures import build_tu_muon


def test_tu_muon_e2e_with_debate(e2e_root):
    """Từ mượn: tiền xử lý local -> debate 2 API thật -> sinh -> lọc 2 API -> finalize."""
    require_real_api()
    build_tu_muon(e2e_root)

    run_dispatcher(e2e_root, ["local-preprocess", "--task", "tu-muon"])
    run_dispatcher(e2e_root, ["tu-muon", "--debate", "--debate-mode", "api"])

    final = e2e_root / "benchmark_qa/speech/cac_hien_tuong_dac_biet_trong_tieng_viet/tu_muon/tu_muon_multihop_qa_final.jsonl"
    assert_valid_final(final, min_samples=1)

    # debate phải sinh được knowledge graph
    kg = e2e_root / "knowledge" / "knowledge_tu_muon.json"
    assert kg.exists(), "debate không sinh knowledge_tu_muon.json"
