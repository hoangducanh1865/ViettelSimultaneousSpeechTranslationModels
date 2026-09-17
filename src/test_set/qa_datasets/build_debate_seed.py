"""Assembles the "debate seed" JSON that manual_model_relay.py's turn-1 prompt embeds verbatim.

The seed packages every piece of REAL evidence the 3-model relay (Gemini/ChatGPT/Claude, driven
by manual_model_relay.py) needs to review/expand/correct a task's candidate word list, so the
relay never has to invent evidence -- only reason over what's given:
  - the existing candidate CSV (if any -- Hán Việt has none yet, see han_viet_seed_csv.py),
  - a web-research candidate list (compiled by hand via WebSearch/WebFetch BEFORE running this
    script -- see the notebook/README for the exact search queries used per task),
  - a word-coverage report (from hien_tuong_filter_pipeline.py word-coverage-report) giving real
    hit counts/example transcripts per candidate, across whichever corpus was scanned,
  - a small sample of transcripts that matched NO candidate word at all, so the relay can also
    propose brand-new words grounded in real audio content instead of just reviewing the list.

Usage:
    python build_debate_seed.py --task tu_lay --variant toan_bo \\
        --candidate-csv tu_lay_toan_bo_final.csv --word-col "Từ láy" \\
        --web-candidates-json web_candidates_tu_lay_toan_bo.json \\
        --coverage-report word_coverage_report_tu_lay_toan_bo.json \\
        --unmatched-transcripts-sample unmatched_sample_tu_lay.json \\
        --out-json debate_seed_tu_lay_toan_bo.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional

from knowledge_graph import SCHEMA_VERSION

SCHEMA_TEXT = """{
  "schema_version": "%s", "task": "<task>", "variant": "<variant or null>",
  "generated_at": "<ISO8601>",
  "relay_meta": {"model_order": [...], "rounds_completed": int, "max_rounds": int,
                 "stopped_reason": "consensus"|"max_rounds", "consensus_round": int|null},
  "words": [{"word": str, "status": "confirmed"|"corrected"|"added"|"rejected",
             "source": "existing_csv"|"web_research"|"corpus_scan"|"model_proposed",
             "fields": {...tuy task...}, "extra_fields": {...}, "notes": str}],
  "rules": {"level_1": {"description": str, "distractor_strategy": str}, "level_2": {...},
            "level_3": {...}, "cloze_toan_bo": {...} (chỉ tu_lay/toan_bo),
            "van_new_mechanic": {"enabled": bool, ...} (chỉ tu_lay/van),
            "chung_new_mechanic": {"enabled": bool, ...} (chỉ tu_lay/chung)}
}""" % SCHEMA_VERSION

INSTRUCTIONS_BY_TASK = {
    "han_viet": (
        "Rà soát/mở rộng danh sách từ Hán Việt xuất hiện trong transcript ASR tiếng Việt của "
        "chúng tôi. Với MỖI từ (đã có trong CSV, đề xuất qua web-research, hoặc do bạn tự đề "
        "xuất từ transcript chưa khớp), quyết định status confirmed/corrected/added/rejected, "
        "cung cấp fields.meaning (nghĩa Hán Việt), fields.category_hint (phạm trù/lĩnh vực, có "
        "thể null), và fields.historical_fact (1 sự kiện/nhân vật/văn kiện lịch sử-văn hóa Việt "
        "Nam THẬT gắn với từ đó, CHỈ điền nếu bạn HOÀN TOÀN CHẮC CHẮN, else null). TUYỆT ĐỐI "
        "không thêm từ chỉ vì 'nghe giống Hán Việt' -- phải thực sự là từ Hán Việt phổ biến, "
        "đúng nghĩa, và (nếu có thể) có bằng chứng xuất hiện trong evidence bên dưới."
    ),
    "tu_muon": (
        "Rà soát/mở rộng danh sách từ mượn (đã Việt hóa) xuất hiện trong transcript ASR. Với "
        "mỗi từ, cung cấp fields.tu_goc (chữ viết gốc), fields.ngon_ngu_nguon_goc (chỉ điền nếu "
        "RÕ RÀNG, không mập mờ kiểu 'Pháp/Anh'), fields.nhom_linh_vuc (lĩnh vực sử dụng, có thể "
        "null), fields.y_nghia_ghi_chu, và fields.historical_fact (bối cảnh lịch sử du nhập THẬT "
        "vào tiếng Việt -- Pháp thuộc, giao thương người Hoa, ảnh hưởng tiếng Anh hiện đại... -- "
        "CHỈ điền nếu HOÀN TOÀN CHẮC CHẮN, else null)."
    ),
    "tu_lay": (
        "Rà soát/mở rộng danh sách từ láy (variant {variant}) xuất hiện trong transcript ASR. "
        "Với mỗi từ, cung cấp đủ các field cột CSV hiện có của variant này, PLUS fields.base_word "
        "(tiếng gốc trước khi láy toàn bộ -- CHỈ áp dụng variant toan_bo, else null), và "
        "fields.cultural_fact (1 câu ca dao/tục ngữ/thành ngữ/tác phẩm văn học Việt Nam THẬT, "
        "NỔI TIẾNG có dùng chính từ láy này theo cách tiêu biểu -- CHỈ điền nếu HOÀN TOÀN CHẮC "
        "CHẮN, else null). Nếu variant là 'van' hoặc 'chung': ngoài rà soát từ, hãy CÙNG ĐỀ XUẤT "
        "1 cơ chế câu hỏi khó tương đương với cơ chế 'điền vào chỗ trống + Tone Harmony' đã có "
        "cho variant toàn bộ (xem rules.cloze_toan_bo trong knowledge JSON hiện tại nếu có) -- "
        "ghi đề xuất đó vào rules.{variant}_new_mechanic (enabled=true kèm description + "
        "distractor_strategy cụ thể) khi cả 3 model đồng ý; nếu chưa đủ chắc chắn, để "
        "enabled=false."
    ),
    "phuong_ngu": (
        "Rà soát/mở rộng danh sách từ/cụm từ phương ngữ xuất hiện trong transcript ASR. Với mỗi "
        "từ, cung cấp fields.region_or_ethnic_group (vùng miền/dân tộc CỤ THỂ, ví dụ 'Nghệ Tĩnh', "
        "'Nam Bộ' -- CHỈ điền nếu chắc chắn, else null/rejected) và fields.confidence "
        "('high'/'medium'/'low')."
    ),
}

CONSENSUS_INSTRUCTIONS = (
    "\n\nKẾT THÚC câu trả lời của bạn bằng ĐÚNG 1 trong 2 dòng sau (không viết gì khác sau dòng "
    "này):\n"
    '  "CONSENSUS: FINAL" -- nếu bạn thấy knowledge graph đã đủ tốt, không cần sửa thêm.\n'
    '  "CONSENSUS: CONTINUE" -- nếu bạn vừa sửa/thêm gì đó và muốn model tiếp theo xem lại.\n'
    "Luôn trả về TOÀN BỘ knowledge graph (không phải diff) trong 1 khối ```json ... ``` duy nhất, "
    "đúng schema đã cho."
)


def _load_csv_words(candidate_csv: Path, word_col: str) -> list[dict]:
    with open(candidate_csv, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return [
        {"word": r[word_col].strip(), "source": "existing_csv", "fields": {k: v for k, v in r.items() if k != word_col}}
        for r in rows if r[word_col].strip()
    ]


def _load_web_candidates(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    return [{"word": item["word"], "source": "web_research", "notes": item.get("note", "")} for item in items]


def build_seed(
    task: str, *, variant: Optional[str] = None,
    candidate_csv: Optional[Path] = None, word_col: Optional[str] = None,
    web_candidates_json: Optional[Path] = None,
    coverage_report: Optional[dict] = None,
    unmatched_sample_transcripts: Optional[list[str]] = None,
    extra_instructions: str = "", max_unmatched_sample: int = 300,
) -> dict:
    """Hợp nhất 3 nguồn candidate (CSV hiện có + web-research + corpus-scan qua coverage_report)
    thành 1 danh sách duy nhất, đính kèm bằng chứng thật (hit count/example transcript), rồi gói
    thành seed JSON cho manual_model_relay.py. Không tự quyết định từ nào đúng/sai -- đó là việc
    của 3 model trong relay, seed chỉ cung cấp evidence."""
    candidates: dict[str, dict] = {}

    if candidate_csv is not None:
        for item in _load_csv_words(candidate_csv, word_col):
            candidates[item["word"]] = item

    if web_candidates_json is not None:
        for item in _load_web_candidates(web_candidates_json):
            candidates.setdefault(item["word"], {"word": item["word"], "source": "web_research", "notes": item.get("notes", "")})

    if coverage_report:
        for word, cov in coverage_report.items():
            if word in candidates:
                candidates[word]["coverage"] = cov
            elif cov.get("total_hits", 0) > 0:
                # Từ có hit thật trong corpus nhưng chưa nằm trong CSV/web-research -- vẫn đưa
                # vào seed để relay xem xét, đánh dấu rõ nguồn là corpus_scan.
                candidates[word] = {"word": word, "source": "corpus_scan", "coverage": cov}

    instructions = INSTRUCTIONS_BY_TASK[task].format(variant=variant) if task == "tu_lay" else INSTRUCTIONS_BY_TASK[task]
    if extra_instructions:
        instructions += "\n\n" + extra_instructions

    return {
        "task": task, "variant": variant,
        "instructions": instructions + CONSENSUS_INSTRUCTIONS,
        "schema_spec": SCHEMA_TEXT,
        "candidate_words": list(candidates.values()),
        "unmatched_transcript_sample": (unmatched_sample_transcripts or [])[:max_unmatched_sample],
    }


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=["han_viet", "tu_muon", "tu_lay", "phuong_ngu"])
    parser.add_argument("--variant", default=None, choices=["toan_bo", "van", "chung"])
    parser.add_argument("--candidate-csv", default=None)
    parser.add_argument("--word-col", default=None)
    parser.add_argument("--web-candidates-json", default=None, help='JSON list [{"word": str, "note": str}, ...].')
    parser.add_argument("--coverage-report", default=None, help="Output của hien_tuong_filter_pipeline.py word-coverage-report.")
    parser.add_argument("--unmatched-transcripts-sample", default=None, help="JSON list[str] transcript chưa khớp từ nào.")
    parser.add_argument("--extra-instructions", default="")
    parser.add_argument("--max-unmatched-sample", type=int, default=300)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args(argv)

    coverage_report = None
    if args.coverage_report:
        with open(args.coverage_report, encoding="utf-8") as f:
            coverage_report = json.load(f)

    unmatched = None
    if args.unmatched_transcripts_sample:
        with open(args.unmatched_transcripts_sample, encoding="utf-8") as f:
            unmatched = json.load(f)

    seed = build_seed(
        args.task, variant=args.variant,
        candidate_csv=Path(args.candidate_csv) if args.candidate_csv else None, word_col=args.word_col,
        web_candidates_json=Path(args.web_candidates_json) if args.web_candidates_json else None,
        coverage_report=coverage_report, unmatched_sample_transcripts=unmatched,
        extra_instructions=args.extra_instructions, max_unmatched_sample=args.max_unmatched_sample,
    )
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(seed, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu {args.out_json} ({len(seed['candidate_words'])} candidate word, "
          f"{len(seed['unmatched_transcript_sample'])} transcript chưa khớp mẫu).")


if __name__ == "__main__":
    main()
