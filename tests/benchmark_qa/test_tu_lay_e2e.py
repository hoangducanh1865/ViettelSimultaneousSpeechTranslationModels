from conftest import assert_valid_final, require_real_api, run_dispatcher
from fixtures import build_tu_lay


def test_tu_lay_e2e(e2e_root):
    """Từ láy toan_bo: tiền xử lý -> classify/generate -> cloze (rule-based) -> lọc 2 API -> finalize."""
    require_real_api()
    build_tu_lay(e2e_root)

    run_dispatcher(e2e_root, ["local-preprocess", "--task", "tu-lay"])
    run_dispatcher(e2e_root, ["tu-lay", "--variant", "toan_bo"])

    base = e2e_root / "benchmark_qa/speech/cac_hien_tuong_dac_biet_trong_tieng_viet/tu_lay"
    assert_valid_final(base / "tu_lay_toan_bo_multihop_qa_final.jsonl", min_samples=1)
    # cloze/tone-harmony là rule-based -> phải có output (không cần API)
    assert_valid_final(base / "tu_lay_toan_bo_cloze_qa_final.jsonl", min_samples=1)
