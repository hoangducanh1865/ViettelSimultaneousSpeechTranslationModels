#!/usr/bin/env python3
"""Entrypoint DUY NHẤT cho toàn bộ luồng code của benchmark_qa.

Trước đây mỗi script có `if __name__ == "__main__": main()` riêng và `benchmark_qa.sh`
gọi thẳng từng file. Giờ mọi thứ đi qua dispatcher này:

    python src/main.py <subcommand> [args...]

`args` được chuyển tiếp NGUYÊN VẸN cho `main(argv)` của script tương ứng (kể cả subcommand
nội bộ của script, ví dụ `python src/main.py han-viet classify-levels ...`).

Danh sách subcommand: xem SUBCOMMANDS bên dưới (hoặc `python src/main.py --help`).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent  # <repo>/src
_DSQA = _THIS_DIR / "test_set/datasets_qa"
_BQ = _DSQA / "benchmark_qa"
_HIEN_TUONG = _BQ / "speech/cac_hien_tuong_dac_biet_trong_tieng_viet"
_CODE_SWITCHING = _BQ / "speech/trich_xuat_thong_tin/code_switching"

# subcommand phẳng -> đường dẫn file script (đều có hàm main(argv)).
SUBCOMMANDS: dict[str, Path] = {
    "clotho-aqa": _BQ / "sound/clotho_aqa_pipeline.py",
    "sound-merge": _BQ / "sound/sound_dataset_merge.py",
    "apply-kg": _HIEN_TUONG / "apply_knowledge_graph.py",
    "relay": _HIEN_TUONG / "auto_model_relay.py",
    "debate-seed": _HIEN_TUONG / "build_debate_seed.py",
    "filter-qa": _HIEN_TUONG / "filter_qa_pipeline.py",
    "finalize-qa": _HIEN_TUONG / "finalize_qa.py",
    "han-viet": _HIEN_TUONG / "han_viet/han_viet_pipeline.py",
    "han-viet-seed": _HIEN_TUONG / "han_viet/han_viet_seed_csv.py",
    "hien-tuong": _HIEN_TUONG / "hien_tuong_filter_pipeline.py",
    "phuong-ngu": _HIEN_TUONG / "phuong_ngu/phuong_ngu_pipeline.py",
    "tu-lay": _HIEN_TUONG / "tu_lay/tu_lay_pipeline.py",
    "tu-muon": _HIEN_TUONG / "tu_muon/tu_muon_pipeline.py",
    "code-switching": _CODE_SWITCHING / "code_switching_pipeline.py",
    "code-switching-qa": _CODE_SWITCHING / "code_switching_qa_pipeline.py",
    "inspect-qa": _BQ / "tools/inspect_qa.py",
    "fetch-datasets": _BQ / "tools/fetch_datasets.py",
    "extend-final": _BQ / "tools/extend_final.py",
    "failed-ids": _BQ / "tools/failed_ids.py",
    "hf-pr-push": _DSQA / "translate_datasets/hf_pr_push.py",
    "translate-dataset": _DSQA / "translate_datasets/translate_dataset.py",
}


def _usage() -> str:
    lines = ["Cách dùng: python src/main.py <subcommand> [args...]", "", "Subcommand:"]
    width = max(len(name) for name in SUBCOMMANDS)
    for name in sorted(SUBCOMMANDS):
        lines.append(f"  {name.ljust(width)}  {SUBCOMMANDS[name]}")
    lines += [
        "",
        "Ví dụ:",
        "  python src/main.py han-viet classify-levels --input ... --output ...",
        "  python src/main.py relay run --task code_switching --debate-mode api",
        "  python src/main.py inspect-qa tree /path/test_sound.jsonl",
        "  python src/main.py filter-qa filter-qa --task tu_lay --provider gemini ...",
    ]
    return "\n".join(lines)


def _load_main(script_path: Path):
    if not script_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy script cho subcommand: {script_path}")
    # Emulate `python <script>`: thư mục chứa script nằm ở sys.path[0] để các import cùng
    # thư mục (env_paths, knowledge_graph, auto_model_relay, ...) resolve được.
    script_dir = str(script_path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    # Tên module duy nhất theo tên file để tránh đụng nhau giữa các thư mục.
    mod_name = f"_benchmark_qa_{script_path.stem}"
    spec = importlib.util.spec_from_file_location(mod_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Không load được module từ {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    main_fn = getattr(module, "main", None)
    if main_fn is None:
        raise AttributeError(f"{script_path} không có hàm main(argv).")
    return main_fn


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_usage())
        return 0 if argv else 1

    subcommand, rest = argv[0], argv[1:]
    script_path = SUBCOMMANDS.get(subcommand)
    if script_path is None:
        print(f"[main] Subcommand không hợp lệ: {subcommand!r}\n", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        return 2

    main_fn = _load_main(script_path)
    main_fn(rest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
