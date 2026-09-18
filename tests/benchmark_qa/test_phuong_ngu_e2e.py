from conftest import assert_valid_final, require_real_api, run_dispatcher
from fixtures import build_phuong_ngu, speech_dir


def test_phuong_ngu_e2e(e2e_root):
    require_real_api()
    build_phuong_ngu(e2e_root)

    run_dispatcher(e2e_root, ["phuong-ngu"])

    final = speech_dir(e2e_root) / "phuong_ngu" / "phuong_ngu_region_qa_final.jsonl"
    assert_valid_final(final, min_samples=1)
