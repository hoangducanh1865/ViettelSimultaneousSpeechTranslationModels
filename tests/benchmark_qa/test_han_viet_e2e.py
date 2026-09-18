from conftest import assert_valid_final, require_real_api, run_dispatcher
from fixtures import build_han_viet, speech_dir


def test_han_viet_e2e(e2e_root):
    require_real_api()
    build_han_viet(e2e_root)

    run_dispatcher(e2e_root, ["han-viet"])

    final = speech_dir(e2e_root) / "han_viet" / "han_viet_multihop_qa_final.jsonl"
    assert_valid_final(final, min_samples=1)
