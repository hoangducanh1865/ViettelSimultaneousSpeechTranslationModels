"""Sinh câu hỏi 5 khía cạnh cho Code-switching (đếm/từ loại, tác động ngữ nghĩa, tri thức nền,
Việt hóa, rủi ro ASR) + quan hệ đa-thuật-ngữ, dựa trên `code_switching_qa.jsonl` đã hợp nhất
(code_switching_pipeline.py merge-datasets) và knowledge_code_switching.json đã debate
(auto_model_relay.py -- xem cac_hien_tuong_dac_biet_trong_tieng_viet/).

3 subcommand, chạy theo thứ tự:
  1. classify-cs        -- với mỗi (sample, thuật ngữ), lấy field ĐÃ DEBATE trực tiếp từ knowledge
                            graph (không gọi Gemini); thuật ngữ CHƯA có trong knowledge graph (hiếm)
                            rơi về Gemini tự phân loại (fallback, batch).
  2. generate-questions  -- sinh câu hỏi cho từng loại (A-F) tuỳ field nào khác None:
                              A "count_classify" (luôn, rule-based -- KHÔNG gọi Gemini)
                              B "semantic_impact" (chỉ khi pos_role_typical=verb/adjective)
                              C "knowledge_grounding" (chỉ khi có knowledge_fact)
                              D "localization" (chỉ khi có vi_localized_term)
                              E "asr_risk" (luôn, mọi term_type)
                              F "relational" (chỉ sample có trong knowledge_graph["sample_relations"])
  3. filter-questions    -- lọc chất lượng (tự nhiên/đa dạng/độ khó/độ hợp lý), dùng CHUNG cho cả
                            lượt Gemini (--provider gemini) và lượt OpenAI-compatible
                            (--provider openai, input = output lượt Gemini).

Usage:
    python code_switching_qa_pipeline.py classify-cs \\
        --service-account-json gemini_service_account.json \\
        --input code_switching_qa.jsonl --knowledge-json knowledge_code_switching.json \\
        --output code_switching_classified.jsonl

    python code_switching_qa_pipeline.py generate-questions \\
        --service-account-json gemini_service_account.json \\
        --input code_switching_classified.jsonl --knowledge-json knowledge_code_switching.json \\
        --output code_switching_multihop_qa.jsonl

    python code_switching_qa_pipeline.py filter-questions --provider gemini \\
        --service-account-json gemini_service_account.json \\
        --input code_switching_multihop_qa.jsonl \\
        --kept-output code_switching_gemini_kept.jsonl --rules-output code_switching_gemini_rules.json

    python code_switching_qa_pipeline.py filter-questions --provider openai \\
        --openai-api-key-file openai_api_key.txt --openai-base-url https://r3wrrfi.abc-tunnel.us/v1 \\
        --input code_switching_gemini_kept.jsonl \\
        --kept-output code_switching_openai_kept.jsonl --rules-output code_switching_openai_rules.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from tqdm.auto import tqdm

_THIS_DIR = Path(__file__).resolve().parent
# knowledge_graph.py sống ở thư mục ANH EM cac_hien_tuong_dac_biet_trong_tieng_viet/ (cùng cấp
# speech/), KHÔNG phải cùng cây con trich_xuat_thong_tin/ -- phải trỏ path riêng cho nó.
sys.path.insert(0, str(_THIS_DIR.parents[1] / "cac_hien_tuong_dac_biet_trong_tieng_viet"))
sys.path.insert(0, str(_THIS_DIR.parents[3]))  # datasets_qa/, cho translate_dataset

import auto_model_relay  # noqa: E402 -- dùng chung call_model()/resolve_clients() cho auto-detect Colab/local
import env_paths  # noqa: E402
import error_log  # noqa: E402
from knowledge_graph import load_knowledge_graph, rule_addendum_text, words_index  # noqa: E402
from test_set.datasets_qa.translate_datasets.translate_dataset import DEFAULT_MODEL  # noqa: E402

TASK_NAME = "code_switching"

DEFAULT_BATCH_SIZE = 15
DEFAULT_GEN_BATCH_SIZE = 15
DEFAULT_MAX_WORKERS = 4
DEFAULT_MAX_RETRIES = 3

FIELD_KEYS = [
    "origin_language", "term_type", "pos_role_typical", "domain", "knowledge_fact",
    "vi_localized_term", "sentiment_impact_note", "asr_risk_level", "asr_risk_reason",
]

_POS_LABELS = {"noun": "danh từ", "verb": "động từ", "adjective": "tính từ", "other": "từ loại khác"}


def _ensure_parent(path):
    """Tạo thư mục cha trước khi ghi file (local/thư mục tạm có thể chưa có sẵn như trên Drive)."""
    from pathlib import Path as _P
    p = _P(path)
    if str(p.parent) and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    return p

def _audio_ref(sample: dict) -> Optional[str]:
    return sample.get("audio_filepath") or sample.get("audio")


def _load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ============================================================================================
# Step 1: classify-cs
# ============================================================================================

CS_CLASSIFY_FALLBACK_SYSTEM_PROMPT = """Bạn phân loại metadata cho 1 thuật ngữ code-switching
(từ nước ngoài chêm vào câu tiếng Việt) xuất hiện trong 1 đoạn audio, KHÔNG có sẵn trong đồ thị
tri thức đã debate -- bạn phải tự quyết định. Với mỗi mục, bạn được cho từ (`term`) và transcript
chứa từ đó (`transcript`).

Trả về ĐÚNG các field sau (null nếu không chắc chắn -- KHÔNG bịa):
- "origin_language": ngôn ngữ gốc của từ.
- "term_type": 1 trong "person_name"/"brand_org_name"/"place_name"/"technical_term"/"common_word".
- "pos_role_typical": từ loại trong CÂU NÀY -- 1 trong "noun"/"verb"/"adjective"/"other".
- "domain": lĩnh vực/hệ thống thuật ngữ thuộc về (thực phẩm, bệnh lý, công nghệ, thể thao...).
- "knowledge_fact": 1 dữ kiện THẬT ngắn gọn về thuật ngữ (null nếu không chắc).
- "vi_localized_term": bản dịch/Việt hóa CHUẨN XÁC theo chuyên ngành (null nếu không có/không nên dịch).
- "sentiment_impact_note": CHỈ điền nếu pos_role_typical là "verb"/"adjective" -- mô tả từ này làm
  thay đổi ý định/sắc thái câu như thế nào (else null).
- "asr_risk_level": "low"/"medium"/"high" -- mức độ nghiêm trọng nếu hệ thống nhận diện giọng nói
  nghe NHẦM từ này thành từ khác.
- "asr_risk_reason": lý do/hậu quả cụ thể cho asr_risk_level ở trên.

Input: JSON array các object {"id": str, "term": str, "transcript": str}.
Output: CHỈ trả về JSON array cùng độ dài, mỗi phần tử {"id": <id đầu vào>, "origin_language": ...,
"term_type": ..., "pos_role_typical": ..., "domain": ..., "knowledge_fact": ...,
"vi_localized_term": ..., "sentiment_impact_note": ..., "asr_risk_level": ..., "asr_risk_reason": ...}
-- không giải thích thêm, không markdown fence.
"""


def _classify_cs_fallback_batch(client, model, batch, max_retries):
    payload = [{"id": item["id"], "term": item["term"], "transcript": item["transcript"]} for item in batch]
    ids_sent = {item["id"] for item in batch}
    last_error = None
    for attempt in range(max_retries):
        try:
            raw = auto_model_relay.call_model(
                "gemini", json.dumps(payload, ensure_ascii=False),
                gemini_client=client, gemini_model=model,
                system_instruction=CS_CLASSIFY_FALLBACK_SYSTEM_PROMPT, temperature=0.0, max_output_tokens=4096,
            )
            raw = raw.strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            results = json.loads(raw)
            out = {}
            for item in results:
                rid = item.get("id")
                if rid in ids_sent:
                    out[rid] = {k: item.get(k) for k in FIELD_KEYS}
            missing = ids_sent - set(out)
            if missing:
                raise ValueError(f"Thiếu {len(missing)}/{len(ids_sent)} id trong response")
            return out
        except Exception as e:  # noqa: BLE001
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    tqdm.write(f"[LỖI classify-cs fallback] batch {len(batch)} mục: {last_error!r} -- dùng field rỗng.")
    error_log.log_failures(TASK_NAME, "classify-cs", [item["id"] for item in batch], last_error, model=model)
    return {item["id"]: {k: None for k in FIELD_KEYS} for item in batch}


def classify_cs(
    client, model, records: list[dict], knowledge_graph: Optional[dict] = None, *,
    batch_size: int = DEFAULT_BATCH_SIZE, max_workers: int = DEFAULT_MAX_WORKERS,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> list[dict]:
    """records: sample từ code_switching_qa.jsonl (đã lọc sẵn cs_terms không rỗng). Với MỖI term
    trong record["cs_terms"]: có trong words_index(knowledge_graph) -> dùng field đã debate,
    KHÔNG gọi Gemini; KHÔNG có -> gom vào batch fallback gọi Gemini 1 lần. Trả về list record MỚI
    (không sửa list đầu vào), mỗi record thêm "term_fields": {term: {field...}} và "relation"
    (entry khớp trong knowledge_graph["sample_relations"] theo sample id, None nếu không có)."""
    kg_words = words_index(knowledge_graph) if knowledge_graph else {}
    relations_by_sample = {
        r["sample_id"]: r for r in (knowledge_graph or {}).get("sample_relations", []) or []
    }

    pending_fallback = []
    for r in records:
        for term in r["cs_terms"]:
            if term not in kg_words:
                pending_fallback.append({"id": f"{r['id']}::{term}", "term": term, "transcript": r["text"]})

    fallback_fields: dict[str, dict] = {}
    if pending_fallback:
        print(f"Cần fallback Gemini cho {len(pending_fallback)} (sample, term) chưa có trong knowledge graph.")
        batches = [pending_fallback[i:i + batch_size] for i in range(0, len(pending_fallback), batch_size)]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_classify_cs_fallback_batch, client, model, b, max_retries): b for b in batches}
            for future in tqdm(as_completed(futures), total=len(futures), desc="classify-cs fallback"):
                fallback_fields.update(future.result())

    out = []
    for r in records:
        term_fields = {}
        for term in r["cs_terms"]:
            if term in kg_words:
                term_fields[term] = kg_words[term].get("fields", {})
            else:
                term_fields[term] = fallback_fields.get(f"{r['id']}::{term}", {k: None for k in FIELD_KEYS})
        new_r = dict(r)
        new_r["term_fields"] = term_fields
        new_r["relation"] = relations_by_sample.get(r["id"])
        out.append(new_r)
    return out


# ============================================================================================
# Step 2: generate-questions
# ============================================================================================

_DIFFICULTY_BY_TYPE = {"A": "easy", "B": "medium", "C": "medium", "D": "easy", "E": "hard", "F": "hard"}
QUESTION_RULE_KEY = {
    "A": "count_classify", "B": "semantic_impact", "C": "knowledge_grounding",
    "D": "localization", "E": "asr_risk", "F": "multi_term_policy",
}

QUESTION_TYPE_INSTRUCTIONS = {
    "B": (
        "Sinh câu hỏi trắc nghiệm 4 lựa chọn về TÁC ĐỘNG NGỮ NGHĨA của 1 thuật ngữ code-switching "
        "đóng vai trò động từ/tính từ trong câu. Đáp án đúng PHẢI dựa ĐÚNG vào "
        "\"sentiment_impact_note\" đã cho (diễn đạt lại, không đổi nghĩa). 3 phương án nhiễu là "
        "các mô tả tác động ngữ nghĩa/sắc thái KHÁC, hợp lý, cùng dạng, khác nhau, khác đáp án đúng."
    ),
    "C": (
        "Sinh câu hỏi trắc nghiệm 4 lựa chọn về TRI THỨC NỀN/ỨNG DỤNG của 1 thuật ngữ code-switching "
        "(ví dụ: thuật ngữ là thành phần chính trong loại thực phẩm/bệnh lý/hệ thống nào). Đáp án "
        "đúng PHẢI dựa ĐÚNG vào \"knowledge_fact\" đã cho, KHÔNG tự thêm/đổi dữ kiện. 3 phương án "
        "nhiễu là các dữ kiện/khái niệm THẬT KHÁC (không bịa), cùng dạng, khác nhau, khác đáp án đúng."
    ),
    "D": (
        "Sinh câu hỏi trắc nghiệm 4 lựa chọn hỏi BẢN VIỆT HÓA CHUẨN XÁC của 1 thuật ngữ code-switching "
        "theo đúng chuyên ngành. Đáp án đúng PHẢI là \"vi_localized_term\" đã cho. 3 phương án "
        "nhiễu là các bản dịch/Việt hóa SAI nhưng HỢP LÝ (nghe giống thuật ngữ chuyên ngành thật), "
        "khác nhau, khác đáp án đúng."
    ),
    "E": (
        "Sinh câu hỏi trắc nghiệm 4 lựa chọn về RỦI RO nếu hệ thống nhận diện giọng nói (ASR) NGHE "
        "NHẦM 1 thuật ngữ code-switching thành từ khác. Đáp án đúng PHẢI dựa ĐÚNG vào "
        "\"asr_risk_reason\" đã cho (diễn đạt lại, không đổi nghĩa). 3 phương án nhiễu là các "
        "hậu quả/mức độ rủi ro KHÁC, hợp lý, cùng dạng, khác nhau, khác đáp án đúng."
    ),
    "F": (
        "Sinh câu hỏi trắc nghiệm 4 lựa chọn về MỐI LIÊN HỆ giữa NHIỀU thuật ngữ code-switching "
        "cùng xuất hiện trong 1 câu. Đáp án đúng PHẢI dựa ĐÚNG vào \"relation_fact\" đã cho. 3 "
        "phương án nhiễu là các mối liên hệ/lĩnh vực KHÁC, hợp lý, cùng dạng, khác nhau, khác đáp "
        "án đúng."
    ),
}


def _build_qual_system_prompt(qtype: str, knowledge_graph: Optional[dict]) -> str:
    return (
        "Bạn sinh câu hỏi trắc nghiệm 4 lựa chọn cho benchmark nghe-hiểu tiếng Việt.\n\n"
        f"{QUESTION_TYPE_INSTRUCTIONS[qtype]}\n\n"
        "QUY TẮC BẮT BUỘC:\n"
        "- Câu hỏi KHÔNG được nhắc thẳng thuật ngữ mục tiêu (người nghe phải tự nhận ra qua audio).\n"
        "- Hệ thống trả lời câu hỏi CHỈ nghe được audio, KHÔNG có transcript -- \"question\" PHẢI "
        "luôn nói \"đoạn audio\"/\"câu vừa nghe\" (KHÔNG BAO GIỜ được dùng chữ \"transcript\").\n\n"
        "Input: JSON array các object {\"id\": str, ...field liên quan đã nêu ở trên...}.\n"
        "Output: CHỈ trả về JSON array cùng độ dài, mỗi phần tử {\"id\": <id đầu vào>, "
        "\"question\": str, \"choices\": [str, str, str, str], \"answer\": str} (answer PHẢI là 1 "
        "trong 4 choices) -- không giải thích thêm, không markdown fence."
    ) + rule_addendum_text(knowledge_graph, QUESTION_RULE_KEY[qtype])


def _units_semantic_impact(records: list[dict]) -> list[dict]:
    units = []
    for r in records:
        for idx, term in enumerate(r["cs_terms"]):
            fields = r["term_fields"].get(term, {})
            if fields.get("pos_role_typical") in ("verb", "adjective") and fields.get("sentiment_impact_note"):
                uid = f"{r['id']}__B__t{idx}"
                units.append({"id": uid, "sample": r, "terms": [term],
                              "payload": {"id": uid, "sentiment_impact_note": fields["sentiment_impact_note"]}})
    return units


def _units_knowledge_grounding(records: list[dict]) -> list[dict]:
    units = []
    for r in records:
        for idx, term in enumerate(r["cs_terms"]):
            fields = r["term_fields"].get(term, {})
            if fields.get("knowledge_fact"):
                uid = f"{r['id']}__C__t{idx}"
                units.append({"id": uid, "sample": r, "terms": [term],
                              "payload": {"id": uid, "knowledge_fact": fields["knowledge_fact"]}})
    return units


def _units_localization(records: list[dict]) -> list[dict]:
    units = []
    for r in records:
        for idx, term in enumerate(r["cs_terms"]):
            fields = r["term_fields"].get(term, {})
            if fields.get("vi_localized_term"):
                uid = f"{r['id']}__D__t{idx}"
                units.append({"id": uid, "sample": r, "terms": [term],
                              "payload": {"id": uid, "vi_localized_term": fields["vi_localized_term"]}})
    return units


def _units_asr_risk(records: list[dict]) -> list[dict]:
    units = []
    for r in records:
        for idx, term in enumerate(r["cs_terms"]):
            fields = r["term_fields"].get(term, {})
            if fields.get("asr_risk_reason"):
                uid = f"{r['id']}__E__t{idx}"
                units.append({"id": uid, "sample": r, "terms": [term],
                              "payload": {"id": uid, "asr_risk_reason": fields["asr_risk_reason"]}})
    return units


def _units_relational(records: list[dict]) -> list[dict]:
    units = []
    for r in records:
        relation = r.get("relation")
        if relation and relation.get("question_strategy") in ("relational", "both") and relation.get("relation_fact"):
            uid = f"{r['id']}__F"
            units.append({"id": uid, "sample": r, "terms": relation.get("terms") or r["cs_terms"],
                          "payload": {"id": uid, "terms": relation.get("terms"), "relation_fact": relation["relation_fact"]}})
    return units


_UNIT_BUILDERS = {
    "B": _units_semantic_impact, "C": _units_knowledge_grounding,
    "D": _units_localization, "E": _units_asr_risk, "F": _units_relational,
}


def _record_common_fields(sample: dict, qid: str, qtype: str, terms: list[str], gen: dict) -> dict:
    return {
        "id": qid,
        "audio_id": _audio_ref(sample),
        "question": gen["question"], "choices": gen["choices"], "answer": gen["answer"],
        "dataset": sample.get("dataset_source"),
        "dataset_source": sample.get("dataset_source"),
        "target_terms": terms,
        "question_type": qtype,
        "task": "speech", "split": "test", "category": "Reasoning",
        "sub-category": "Code-switching", "difficulty": _DIFFICULTY_BY_TYPE[qtype],
    }


def _build_count_classify_question(record: dict, rng: random.Random) -> Optional[dict]:
    """Rule-based, KHÔNG gọi Gemini -- terms/pos_role_typical đều đã có sẵn (debate hoặc
    fallback), đủ dữ kiện thật để tự dựng câu hỏi + nhiễu mà không cần suy luận thêm."""
    terms = record["cs_terms"]
    n = len(terms)
    if n >= 2:
        correct = str(n)
        distractor_pool = [str(x) for x in range(1, 6) if x != n]
        choices = [correct] + rng.sample(distractor_pool, 3)
        rng.shuffle(choices)
        return {
            "question": "Trong đoạn audio trên có bao nhiêu thuật ngữ code-switching (từ nước "
                         "ngoài chêm vào câu) xuất hiện?",
            "choices": choices, "answer": correct,
        }

    term = terms[0]
    pos = record["term_fields"].get(term, {}).get("pos_role_typical")
    if pos not in _POS_LABELS:
        return None
    correct = _POS_LABELS[pos]
    choices = [correct] + [v for k, v in _POS_LABELS.items() if k != pos]
    rng.shuffle(choices)
    return {
        "question": "Thuật ngữ code-switching xuất hiện trong đoạn audio trên đóng vai trò từ loại gì?",
        "choices": choices, "answer": correct,
    }


def _gen_qual_batch(client, model, system_prompt, batch, max_retries):
    payload = [item["payload"] for item in batch]
    ids_sent = {item["id"] for item in batch}
    last_error = None
    for attempt in range(max_retries):
        try:
            raw = auto_model_relay.call_model(
                "gemini", json.dumps(payload, ensure_ascii=False),
                gemini_client=client, gemini_model=model,
                system_instruction=system_prompt, temperature=0.7, max_output_tokens=8192,
            )
            raw = raw.strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            results = json.loads(raw)
            out = {}
            for item in results:
                rid = item.get("id")
                choices = item.get("choices") or []
                if rid in ids_sent and item.get("question") and len(choices) == 4 and item.get("answer") in choices:
                    out[rid] = item
            missing = ids_sent - set(out)
            if missing:
                raise ValueError(f"Thiếu/hỏng {len(missing)}/{len(ids_sent)} id trong response")
            return out
        except Exception as e:  # noqa: BLE001
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    tqdm.write(f"[LỖI sinh câu hỏi] batch {len(batch)} mục: {last_error!r} -- bỏ qua batch.")
    error_log.log_failures(TASK_NAME, "generate-questions", ids_sent, last_error, model=model)
    return {}


def generate_questions(
    client, model, classified_records: list[dict], output_path: Path, *,
    knowledge_graph: Optional[dict] = None,
    batch_size: int = DEFAULT_GEN_BATCH_SIZE, max_workers: int = DEFAULT_MAX_WORKERS,
    max_retries: int = DEFAULT_MAX_RETRIES, seed: int = 42,
) -> None:
    rng = random.Random(seed)

    done_ids = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line)["id"])
        print(f"Đã có sẵn {len(done_ids)} câu hỏi trong {output_path} -- resume.")

    # Loại A: rule-based, KHÔNG gọi Gemini.
    n_written_a = 0
    _ensure_parent(output_path)
    with open(output_path, "a", encoding="utf-8") as f:
        for r in classified_records:
            qid = f"{r['id']}__A"
            if qid in done_ids:
                continue
            qa = _build_count_classify_question(r, rng)
            if qa is None:
                continue
            f.write(json.dumps(_record_common_fields(r, qid, "A", r["cs_terms"], qa), ensure_ascii=False) + "\n")
            n_written_a += 1
    print(f"Loại A (đếm/từ loại, rule-based): ghi thêm {n_written_a} câu hỏi.")

    # Loại B-F: Gemini batched, mỗi loại 1 lượt riêng (system prompt khác nhau).
    for qtype, build_units_fn in _UNIT_BUILDERS.items():
        units = [u for u in build_units_fn(classified_records) if u["id"] not in done_ids]
        if not units:
            continue
        system_prompt = _build_qual_system_prompt(qtype, knowledge_graph)
        batches = [units[i:i + batch_size] for i in range(0, len(units), batch_size)]
        results: dict = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_gen_qual_batch, client, model, system_prompt, b, max_retries): b for b in batches}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"sinh câu hỏi loại {qtype}"):
                results.update(future.result())

        n_written = 0
        _ensure_parent(output_path)
        with open(output_path, "a", encoding="utf-8") as f:
            for u in units:
                gen = results.get(u["id"])
                if gen is None:
                    continue
                out_record = _record_common_fields(u["sample"], u["id"], qtype, u["terms"], gen)
                f.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                n_written += 1
        print(f"Loại {qtype}: ghi thêm {n_written}/{len(units)} câu hỏi.")


# ============================================================================================
# Step 3: filter-questions
# ============================================================================================

FILTER_SYSTEM_PROMPT = """Bạn chấm chất lượng câu hỏi trắc nghiệm 4 lựa chọn (benchmark nghe-hiểu
tiếng Việt, chủ đề code-switching) theo ĐÚNG 4 tiêu chí:
1. Tự nhiên: câu hỏi đọc lên nghe tự nhiên, không gượng ép/máy móc.
2. Đa dạng: không lặp lại y hệt cấu trúc/cách hỏi của các câu khác trong cùng batch.
3. Độ khó phù hợp: đáp án đúng không quá hiển nhiên (không thể đoán mà không cần nghe/hiểu).
4. Độ hợp lý của nhiễu: cả 3 phương án sai đều hợp lý, không có phương án nào vô lý/lạc đề dễ
   loại trừ ngay, và không phương án nào trùng nghĩa với đáp án đúng.

Với MỖI câu hỏi, quyết định "keep": true/false và "reason" (lý do ngắn gọn, nêu rõ tiêu chí nào
không đạt nếu loại).

Input: JSON array các object {"id": str, "question": str, "choices": [str,...], "answer": str}.
Output: CHỈ trả về JSON object {"items": [{"id": <id đầu vào>, "keep": true|false, "reason": str}, ...],
"criteria_summary": str (tóm tắt NGẮN GỌN các tiêu chí/ngưỡng bạn ĐÃ ÁP DỤNG thực tế cho batch này)}
-- không giải thích thêm, không markdown fence.
"""


def _filter_batch(client_or_provider_call, batch, max_retries):
    """client_or_provider_call: hàm (payload_json_str) -> response_text (đã trừu tượng hoá khác
    biệt Gemini/OpenAI ở call_provider() bên dưới)."""
    payload = [{"id": r["id"], "question": r["question"], "choices": r["choices"], "answer": r["answer"]} for r in batch]
    ids_sent = {r["id"] for r in batch}
    last_error = None
    for attempt in range(max_retries):
        try:
            raw = client_or_provider_call(json.dumps(payload, ensure_ascii=False))
            raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            parsed = json.loads(raw)
            items = {item["id"]: item for item in parsed.get("items", []) if item.get("id") in ids_sent}
            missing = ids_sent - set(items)
            if missing:
                raise ValueError(f"Thiếu {len(missing)}/{len(ids_sent)} id trong response")
            return items, parsed.get("criteria_summary", "")
        except Exception as e:  # noqa: BLE001
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    tqdm.write(f"[LỖI lọc câu hỏi] batch {len(batch)} câu: {last_error!r} -- GIỮ LẠI toàn bộ batch (mặc định an toàn).")
    error_log.log_failures(TASK_NAME, "filter-questions", [r["id"] for r in batch], last_error)
    return {r["id"]: {"keep": True, "reason": "Lỗi gọi API, mặc định giữ lại."} for r in batch}, ""


def filter_questions(
    provider_call, records: list[dict], kept_output: Path, rules_output: Path, *,
    batch_size: int = DEFAULT_BATCH_SIZE, max_workers: int = DEFAULT_MAX_WORKERS,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> None:
    batches = [records[i:i + batch_size] for i in range(0, len(records), batch_size)]
    all_items: dict = {}
    all_summaries = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_filter_batch, provider_call, b, max_retries): b for b in batches}
        for future in tqdm(as_completed(futures), total=len(futures), desc="lọc câu hỏi"):
            items, summary = future.result()
            all_items.update(items)
            if summary:
                all_summaries.append(summary)

    n_kept = 0
    _ensure_parent(kept_output)
    with open(kept_output, "w", encoding="utf-8") as f:
        for r in records:
            verdict = all_items.get(r["id"], {"keep": True, "reason": "Không có verdict (lỗi) -- mặc định giữ lại."})
            if not verdict.get("keep"):
                continue
            out_r = dict(r)
            out_r["filter_reason"] = verdict.get("reason", "")
            f.write(json.dumps(out_r, ensure_ascii=False) + "\n")
            n_kept += 1

    _ensure_parent(rules_output)
    with open(rules_output, "w", encoding="utf-8") as f:
        json.dump({"criteria_summaries": all_summaries}, f, ensure_ascii=False, indent=2)

    print(f"Đã lọc: giữ {n_kept}/{len(records)} câu hỏi vào {kept_output}; rule tổng hợp -> {rules_output}")


def make_gemini_provider_call(client, model: str):
    def _call(payload_json: str) -> str:
        return auto_model_relay.call_model(
            "gemini", payload_json, gemini_client=client, gemini_model=model,
            system_instruction=FILTER_SYSTEM_PROMPT, temperature=0.0, max_output_tokens=8192,
            model_candidates=auto_model_relay.FILTER_GEMINI_MODELS, rotation_key="gemini:filter",
        )

    return _call


def make_openai_provider_call(openai_client, model: str):
    def _call(payload_json: str) -> str:
        return auto_model_relay.call_model(
            "openai", payload_json, openai_client=openai_client, openai_model=model,
            system_instruction=FILTER_SYSTEM_PROMPT, temperature=0.0, max_output_tokens=8192,
            model_candidates=auto_model_relay.FILTER_OPENAI_MODELS, rotation_key="openai:filter",
        )

    return _call


# ============================================================================================
# CLI
# ============================================================================================

def _resolve_role_client(service_account_json, env_file, model_arg, role: str, input_path, location: str = "local"):
    """Ủy quyền cho auto_model_relay.resolve_role_client() (nguồn DUY NHẤT xử lý credential):
    ưu tiên service account (role GEMINI), nếu không thì đọc .env -> client OpenAI-compatible."""
    return auto_model_relay.resolve_role_client(
        role,
        service_account_json=service_account_json,
        env_file=env_file,
        location=location,
        model=model_arg,
        input_path=input_path,
    )


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("classify-cs", help="Gắn field debate (hoặc Gemini fallback) cho từng (sample, thuật ngữ).")
    env_paths.add_location_arg(p1)
    p1.add_argument("--service-account-json", default=None, help="Mặc định: env_paths.gemini_service_account_path(--location) khi --location drive; bỏ trống để lấy GEMINI_API_KEY/base_url/model từ --env-file.")
    p1.add_argument("--env-file", default=None, help='File .env local dạng KEY="value" # base_url # model (mặc định: env_paths.default_env_file(--location) khi --location local).')
    p1.add_argument("--input", default=None, help="Mặc định: {code_switching_dir}/code_switching_qa_has_terms.jsonl.")
    p1.add_argument("--knowledge-json", default=None, help="Mặc định: {knowledge_dir}/knowledge_code_switching.json.")
    p1.add_argument("--output", default=None, help="Mặc định: {code_switching_dir}/code_switching_classified.jsonl.")
    p1.add_argument("--model", default=None)
    p1.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p1.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    p1.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    p1.add_argument("--only-ids", default=None, help="JSON list base id -- chỉ chạy lại sample lỗi.")

    p2 = sub.add_parser("generate-questions", help="Sinh câu hỏi 6 loại (A-F) từ output classify-cs.")
    env_paths.add_location_arg(p2)
    p2.add_argument("--service-account-json", default=None, help="Mặc định: env_paths.gemini_service_account_path(--location) khi --location drive; bỏ trống để lấy GEMINI_API_KEY/base_url/model từ --env-file.")
    p2.add_argument("--env-file", default=None, help='File .env local dạng KEY="value" # base_url # model (mặc định: env_paths.default_env_file(--location) khi --location local).')
    p2.add_argument("--input", default=None, help="Mặc định: {code_switching_dir}/code_switching_classified.jsonl.")
    p2.add_argument("--knowledge-json", default=None, help="Mặc định: {knowledge_dir}/knowledge_code_switching.json.")
    p2.add_argument("--output", default=None, help="Mặc định: {code_switching_dir}/code_switching_multihop_qa.jsonl.")
    p2.add_argument("--model", default=None)
    p2.add_argument("--batch-size", type=int, default=DEFAULT_GEN_BATCH_SIZE)
    p2.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    p2.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    p2.add_argument("--only-ids", default=None, help="JSON list base id -- chỉ chạy lại sample lỗi.")
    p2.add_argument("--seed", type=int, default=42)

    p3 = sub.add_parser("filter-questions", help="Lọc chất lượng câu hỏi (dùng chung Gemini/OpenAI).")
    env_paths.add_location_arg(p3)
    p3.add_argument("--provider", required=True, choices=["gemini", "openai"])
    p3.add_argument("--input", default=None, help="Mặc định: {code_switching_dir}/code_switching_multihop_qa.jsonl (provider gemini) hoặc {code_switching_dir}/code_switching_gemini_kept.jsonl (provider openai).")
    p3.add_argument("--kept-output", default=None, help="Mặc định: {code_switching_dir}/code_switching_<provider>_kept.jsonl.")
    p3.add_argument("--rules-output", default=None, help="Mặc định: {code_switching_dir}/code_switching_<provider>_rules.json.")
    p3.add_argument("--env-file", default=None, help='File .env local dạng KEY="value" # base_url # model (mặc định: env_paths.default_env_file(--location) khi --location local) -- dùng khi thiếu --service-account-json/--openai-api-key-file.')
    p3.add_argument("--service-account-json", default=None, help="Vertex AI (--provider gemini). Mặc định: env_paths.gemini_service_account_path(--location) khi --location drive; bỏ trống để lấy GEMINI_API_KEY từ --env-file.")
    p3.add_argument("--gemini-model", default=None)
    p3.add_argument("--openai-api-key-file", default=None, help="File key OpenAI-compatible (--provider openai). Bỏ trống để lấy OPENAI_API_KEY từ --env-file.")
    p3.add_argument("--openai-base-url", default=None)
    p3.add_argument("--openai-model", default=None)
    p3.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p3.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    p3.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    p3.add_argument("--only-ids", default=None, help="JSON list base id -- chỉ chạy lại sample lỗi.")

    args = parser.parse_args(argv)

    if args.command == "classify-cs":
        cs_dir = env_paths.code_switching_dir(args.location)
        service_account_json = args.service_account_json or (
            str(env_paths.gemini_service_account_path(args.location)) if args.location == "drive" and not args.env_file else None
        )
        env_file = args.env_file or (str(env_paths.default_env_file(args.location)) if args.location == "local" else None)
        input_path = Path(args.input) if args.input else cs_dir / "code_switching_qa_has_terms.jsonl"
        knowledge_json = args.knowledge_json or str(env_paths.knowledge_dir(args.location) / "knowledge_code_switching.json")
        output = Path(args.output) if args.output else cs_dir / "code_switching_classified.jsonl"

        client, model = _resolve_role_client(service_account_json, env_file, args.model, "GEMINI", input_path, args.location)
        records = _load_jsonl(input_path)
        if getattr(args, "only_ids", None):
            _only = {str(i).split("__")[0] for i in json.load(open(args.only_ids, encoding="utf-8"))}
            records = [r for r in records if str(r["id"]).split("__")[0] in _only]
        kg = load_knowledge_graph(Path(knowledge_json)) if Path(knowledge_json).exists() else None
        out = classify_cs(
            client, model, records, kg,
            batch_size=args.batch_size, max_workers=args.max_workers, max_retries=args.max_retries,
        )
        _ensure_parent(output)
        with open(output, "w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Đã ghi {len(out)} sample đã classify vào {output}")

    elif args.command == "generate-questions":
        cs_dir = env_paths.code_switching_dir(args.location)
        service_account_json = args.service_account_json or (
            str(env_paths.gemini_service_account_path(args.location)) if args.location == "drive" and not args.env_file else None
        )
        env_file = args.env_file or (str(env_paths.default_env_file(args.location)) if args.location == "local" else None)
        input_path = Path(args.input) if args.input else cs_dir / "code_switching_classified.jsonl"
        knowledge_json = args.knowledge_json or str(env_paths.knowledge_dir(args.location) / "knowledge_code_switching.json")
        output = Path(args.output) if args.output else cs_dir / "code_switching_multihop_qa.jsonl"

        client, model = _resolve_role_client(service_account_json, env_file, args.model, "GEMINI", input_path, args.location)
        records = _load_jsonl(input_path)
        if getattr(args, "only_ids", None):
            _only = {str(i).split("__")[0] for i in json.load(open(args.only_ids, encoding="utf-8"))}
            records = [r for r in records if str(r["id"]).split("__")[0] in _only]
        kg = load_knowledge_graph(Path(knowledge_json)) if Path(knowledge_json).exists() else None
        generate_questions(
            client, model, records, output, knowledge_graph=kg,
            batch_size=args.batch_size, max_workers=args.max_workers, max_retries=args.max_retries,
            seed=args.seed,
        )

    elif args.command == "filter-questions":
        cs_dir = env_paths.code_switching_dir(args.location)
        default_input = cs_dir / "code_switching_multihop_qa.jsonl" if args.provider == "gemini" else cs_dir / "code_switching_gemini_kept.jsonl"
        input_path = Path(args.input) if args.input else default_input
        kept_output = Path(args.kept_output) if args.kept_output else cs_dir / f"code_switching_{args.provider}_kept.jsonl"
        rules_output = Path(args.rules_output) if args.rules_output else cs_dir / f"code_switching_{args.provider}_rules.json"
        env_file = args.env_file or (str(env_paths.default_env_file(args.location)) if args.location == "local" else None)

        records = _load_jsonl(input_path)
        if getattr(args, "only_ids", None):
            _only = {str(i).split("__")[0] for i in json.load(open(args.only_ids, encoding="utf-8"))}
            records = [r for r in records if str(r["id"]).split("__")[0] in _only]
        if args.provider == "gemini":
            service_account_json = args.service_account_json or (
                str(env_paths.gemini_service_account_path(args.location)) if args.location == "drive" and not env_file else None
            )
            client, model = _resolve_role_client(service_account_json, env_file,
                                                 args.gemini_model or auto_model_relay.DEFAULT_GEMINI_FILTER_MODEL,
                                                 "GEMINI", input_path, args.location)
            provider_call = make_gemini_provider_call(client, model)
        else:
            if args.openai_api_key_file:
                openai_client = auto_model_relay.load_openai_client(Path(args.openai_api_key_file), args.openai_base_url or auto_model_relay.DEFAULT_OPENAI_BASE_URL)
                openai_model = args.openai_model or auto_model_relay.DEFAULT_OPENAI_FILTER_MODEL
            else:
                openai_client, openai_model = _resolve_role_client(None, env_file,
                                                                   args.openai_model or auto_model_relay.DEFAULT_OPENAI_FILTER_MODEL,
                                                                   "OPENAI", input_path, args.location)
            provider_call = make_openai_provider_call(openai_client, openai_model)

        filter_questions(
            provider_call, records, kept_output, rules_output,
            batch_size=args.batch_size, max_workers=args.max_workers, max_retries=args.max_retries,
        )
