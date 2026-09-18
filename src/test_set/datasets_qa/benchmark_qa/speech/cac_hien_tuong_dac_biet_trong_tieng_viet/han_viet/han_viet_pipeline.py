"""Sino-Vietnamese ("Hán-Việt") multi-hop reasoning QA pipeline (Gemini/Vertex AI).

Extracted from noteboooks/benchmark_qa/Benchmarck_QA_contructer copy.ipynb -- the notebook
should only clone this repo and invoke this script's subcommands; all Gemini-calling logic
lives here so it can be reviewed, tested and reused outside Colab.

3 subcommands, run in order:
  1. classify-levels     -- extract the target Hán-Việt word from each sample's real transcript
                             (no invention), then ask Gemini to classify a 1/2/3-hop difficulty
                             ("max_level") per sample, grounded on the transcript plus Gemini's
                             own historical/cultural knowledge (level 3 only when Gemini is
                             fully confident of a real, well-known historical fact).
  2. generate-questions   -- for a sample with max_level=N, generate exactly N questions (one
                             per hop level 1..N), placed adjacent to each other in the output,
                             each question/answer grounded strictly in the transcript (and, for
                             level 3, the historical_fact) -- never inventing scenario details.
  3. fill-fields          -- join the generated questions back to the original test_speech.jsonl
                             (by "source_id") to fill in task/split/category/subcategory/
                             subsubcategory/difficulty, which aren't produced by the generation
                             step itself.

Usage:
    python han_viet_pipeline.py classify-levels \\
        --service-account-json /path/to/gemini_service_account.json \\
        --input han_viet/han_viet_qa.jsonl \\
        --output han_viet/han_viet_difficulty_levels.jsonl

    python han_viet_pipeline.py generate-questions \\
        --service-account-json /path/to/gemini_service_account.json \\
        --input han_viet/han_viet_difficulty_levels.jsonl \\
        --output han_viet/han_viet_multihop_qa.jsonl

    python han_viet_pipeline.py fill-fields \\
        --multihop han_viet/han_viet_multihop_qa.jsonl \\
        --original test_speech.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from tqdm.auto import tqdm

# Bootstrap sys.path for the split-directory layout: this file lives in .../han_viet/, but
# knowledge_graph.py sits 1 level up (cac_hien_tuong_dac_biet_trong_tieng_viet/) and
# translate_dataset.py sits 4 levels up (datasets_qa/) -- neither is on sys.path by default when
# Colab runs `!python .../han_viet_pipeline.py` directly.
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent))
sys.path.insert(0, str(_THIS_DIR.parents[3]))

from knowledge_graph import load_knowledge_graph, rule_addendum_text, words_index
from translate_dataset import DEFAULT_MODEL, load_gemini_client

DEFAULT_BATCH_SIZE = 20
DEFAULT_GEN_BATCH_SIZE = 15
DEFAULT_MAX_WORKERS = 4
DEFAULT_MAX_RETRIES = 3

_QUOTE_RE = re.compile(r'"([^"]+)"')

FIELDS_TO_FILL = ["task", "split", "category", "sub-category", "sub-sub-category", "difficulty"]


# ============================================================================================
# Step 1: classify-levels
# ============================================================================================

LEVEL_SYSTEM_PROMPT = """Bạn phân loại độ khó reasoning cho câu hỏi về 1 từ Hán-Việt xuất hiện
trong 1 đoạn audio tiếng Việt. Bạn PHỤ TRÁCH quyết định TOÀN BỘ 3 mức độ, dựa trên transcript
THẬT được cung cấp CỘNG kiến thức lịch sử/văn hóa Việt Nam của chính bạn.

3 mức:
- "level_3": từ Hán-Việt này gắn liền với 1 sự kiện/nhân vật/văn kiện lịch sử-văn hóa Việt Nam
  CÓ THẬT, NỔI TIẾNG, PHỔ BIẾN, mà bạn HOÀN TOÀN CHẮC CHẮN về tính chính xác (không suy đoán mơ
  hồ, không bịa nếu không chắc). Nếu KHÔNG chắc chắn 100% hoặc liên hệ chỉ là suy diễn gượng ép,
  TUYỆT ĐỐI KHÔNG gán level_3 -- hạ xuống level_2 hoặc level_1.
- "level_2": từ Hán-Việt trong CÂU (transcript) này thể hiện RÕ RÀNG 1 phạm trù/lĩnh vực khái
  niệm cụ thể (chính trị, kinh tế, y tế, gia đình, quân sự, giáo dục, đạo đức...) suy ra được từ
  chính ngữ cảnh câu nói.
- "level_1": từ Hán-Việt không có ngữ cảnh câu đủ rõ, hoặc nghĩa quá chung chung -- chỉ nên hỏi
  nghĩa đen.

Input: JSON array các object {"id": str, "target_word": str, "transcript": str}.
Output: CHỈ trả về JSON array cùng độ dài, mỗi phần tử {"id": <id đầu vào>,
"level": "level_1"|"level_2"|"level_3",
"historical_fact": "<sự kiện lịch sử THẬT nếu level_3, else null>",
"reason": "<lý do ngắn gọn>"} -- không giải thích thêm, không markdown fence.
"""

LEVEL_MAP = {"level_1": 1, "level_2": 2, "level_3": 3}


def extract_target_word(r: dict) -> Optional[str]:
    """Trích từ Hán-Việt mục tiêu TRỰC TIẾP từ transcript THẬT của sample -- không suy đoán,
    không bịa. Trả None nếu không trích được (caller phải mặc định mức 1 an toàn nhất)."""
    transcript = (r.get("transcript") or "").lower()
    if not transcript:
        return None

    m = _QUOTE_RE.search(r["question"])
    if m:
        quoted = m.group(1).lower()
        idx = transcript.find(quoted)
        if idx == -1:
            return None
        before = transcript[:idx].strip().split()
        after = transcript[idx + len(quoted):].strip().split()
        if "ngay sau" in r["question"].lower() and before:
            return before[-1]
        if "ngay trước" in r["question"].lower() and after:
            return after[0]
        return None

    answer = r["answer"].strip().lower()
    if answer in transcript:
        return answer
    return None


def _classify_level_batch(client, model, batch, max_retries):
    payload = [{"id": r["id"], "target_word": r["target_word"], "transcript": r["transcript"]} for r in batch]
    ids_sent = {r["id"] for r in batch}
    last_error = None
    for attempt in range(max_retries):
        try:
            from google.genai import types

            response = client.models.generate_content(
                model=model,
                contents=json.dumps(payload, ensure_ascii=False),
                config=types.GenerateContentConfig(
                    system_instruction=LEVEL_SYSTEM_PROMPT,
                    temperature=0.0,
                    max_output_tokens=4096,
                ),
            )
            raw = (response.text or "").strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            results = json.loads(raw)
            out = {}
            for item in results:
                if item.get("id") in ids_sent and item.get("level") in ("level_1", "level_2", "level_3"):
                    out[item["id"]] = item
            missing = ids_sent - set(out)
            if missing:
                raise ValueError(f"Thiếu {len(missing)}/{len(ids_sent)} id trong response")
            return out
        except Exception as e:  # noqa: BLE001
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    tqdm.write(f"[LỖI phân loại mức 1/2/3] batch {len(batch)} sample: {last_error!r} -- mặc định level_1.")
    return {r["id"]: {"level": "level_1", "historical_fact": None, "reason": "Lỗi gọi API, mặc định mức an toàn nhất."} for r in batch}


def classify_levels(
    client, model, records: list[dict], *,
    knowledge_graph: Optional[dict] = None,
    batch_size: int = DEFAULT_BATCH_SIZE, max_workers: int = DEFAULT_MAX_WORKERS,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> list[dict]:
    """Trả về BẢN SAO của records, mỗi record thêm target_word/max_level/historical_fact/
    level_reason.

    knowledge_graph=None (mặc định) -> hành vi CŨ y hệt (an toàn ngược). Khi có, từ đã được 3
    model debate xác thực (words_index) dùng TRỰC TIẾP fields.historical_fact/category_hint đã
    debate -- KHÔNG gọi Gemini tự quyết lại (rẻ hơn VÀ ít hallucination hơn so với để 1 lệnh gọi
    Gemini đơn lẻ tự quyết định mức 3); từ chưa có trong kg vẫn rơi về đường Gemini-tự-quyết cũ."""
    records = [dict(r) for r in records]
    for r in records:
        r["target_word"] = extract_target_word(r)

    n_no_target = sum(1 for r in records if r["target_word"] is None)
    print(f"CẢNH BÁO: {n_no_target}/{len(records)} sample KHÔNG trích được target_word "
          "thật từ transcript -- các sample này sẽ mặc định mức độ 1 (an toàn nhất).")

    kg_words = words_index(knowledge_graph) if knowledge_graph else {}
    n_from_kg = sum(1 for r in records if r["target_word"] in kg_words)
    if kg_words:
        print(f"{n_from_kg}/{len(records)} sample dùng trực tiếp dữ kiện đã debate xác thực từ knowledge graph.")

    pending = [r for r in records if r["target_word"] is not None and r["target_word"] not in kg_words]
    print(f"Cần Gemini phân loại mức 1/2/3 cho {len(pending)} sample "
          f"({len(records) - len(pending)} sample không cần gọi Gemini -- mặc định mức 1 hoặc lấy từ knowledge graph).")

    batches = [pending[i:i + batch_size] for i in range(0, len(pending), batch_size)]
    level_results: dict = {}
    if batches:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_classify_level_batch, client, model, b, max_retries): b for b in batches}
            for future in tqdm(as_completed(futures), total=len(futures), desc="phân loại mức 1/2/3"):
                level_results.update(future.result())

    for r in records:
        if r["target_word"] in kg_words:
            entry = kg_words[r["target_word"]]
            fact = entry.get("fields", {}).get("historical_fact")
            category_hint = entry.get("fields", {}).get("category_hint")
            r["max_level"] = 3 if fact else (2 if category_hint else 1)
            r["historical_fact"] = fact
            r["level_reason"] = f"Từ knowledge graph (source={entry.get('source')})."
            continue
        if r["target_word"] is None:
            r["max_level"] = 1
            r["historical_fact"] = None
            r["level_reason"] = "Không trích được target_word thật từ transcript -- mặc định mức 1."
            continue
        res = level_results.get(r["id"])
        if res is None:
            r["max_level"] = 1
            r["historical_fact"] = None
            r["level_reason"] = "Không có kết quả phân loại -- mặc định mức 1."
        else:
            r["max_level"] = LEVEL_MAP[res["level"]]
            r["historical_fact"] = res.get("historical_fact")
            r["level_reason"] = res["reason"]

    return records


# ============================================================================================
# Step 2: generate-questions
# ============================================================================================

MULTIHOP_SYSTEM_PROMPT = """Bạn sinh câu hỏi trắc nghiệm 4 lựa chọn (multi-hop reasoning) về 1
từ Hán-Việt xuất hiện trong 1 đoạn audio tiếng Việt. Mỗi sample có sẵn "level" (1/2/3) quy định
SỐ BƯỚC SUY LUẬN (hop) bắt buộc câu hỏi phải yêu cầu:

- level 1 (1-hop): nghe -> hiểu từ Hán-Việt -> chọn ĐÚNG nghĩa đen của từ đó. Đáp án đúng phải
  khớp với "known_meaning" đã cho (có thể diễn đạt lại tự nhiên hơn nhưng KHÔNG được đổi nghĩa).
- level 2 (2-hop): nghe -> hiểu từ Hán-Việt -> dựa vào CHÍNH transcript để xác định từ đó thuộc
  PHẠM TRÙ/LĨNH VỰC khái niệm nào (chính trị, kinh tế, y tế, gia đình, quân sự, giáo dục, đạo
  đức, xã hội...). Phạm trù phải suy ra được TỪ CHÍNH CÂU TRANSCRIPT, không bịa thêm bối cảnh.
- level 3 (3-hop): nghe -> hiểu từ Hán-Việt -> liên hệ đến sự kiện/nhân vật/văn kiện lịch sử-văn
  hóa Việt Nam THẬT đã cho sẵn trong "historical_fact". Đáp án đúng PHẢI dựa đúng vào
  "historical_fact" đã cho, KHÔNG được tự thêm/đổi sự kiện khác.

QUY TẮC BẮT BUỘC:
- MỖI sample trong input có field "target_level" (KHÁC "max_level" -- đây là mức CỤ THỂ cần
  sinh cho lượt gọi này, có thể nhỏ hơn max_level của sample). VD sample có max_level=3 sẽ xuất
  hiện 3 LẦN trong input (1 lần với target_level=1, 1 lần target_level=2, 1 lần target_level=3)
  -- mỗi lần sinh ĐÚNG 1 câu hỏi CHO ĐÚNG target_level đó (không phải max_level).
- Câu hỏi KHÔNG được nhắc thẳng tên từ Hán-Việt mục tiêu (người nghe phải tự nhận ra qua audio).
- Hệ thống trả lời câu hỏi CHỈ nghe được audio, KHÔNG có transcript. "transcript" trong input
  chỉ là tư liệu để BẠN suy luận -- nội dung "question" PHẢI luôn nói "đoạn audio"/"đoạn ghi âm"/
  "câu vừa nghe" (KHÔNG BAO GIỜ được dùng chữ "transcript" trong "question").
- 3 phương án nhiễu phải CÙNG DẠNG với đáp án đúng (level 1: nhiễu là nghĩa khác; level 2: nhiễu
  là phạm trù khác; level 3: nhiễu là sự kiện lịch sử THẬT khác, không bịa sự kiện giả), khác
  nhau, và khác đáp án đúng.
- TUYỆT ĐỐI không thêm chi tiết/bối cảnh nào ngoài transcript và historical_fact đã cho.

Input: JSON array các object {"id": str, "target_level": 1|2|3, "transcript": str,
"target_word": str, "known_meaning": str, "historical_fact": str|null}. "id" ở đây là ID GHÉP
(dạng "<id gốc>__L<target_level>"), không phải id gốc của sample.
Output: CHỈ trả về JSON array cùng độ dài, mỗi phần tử {"id": <id đầu vào, giữ NGUYÊN dạng ghép>,
"question": str, "choices": [str, str, str, str], "answer": str} (answer PHẢI là 1 trong 4
choices) -- không giải thích thêm, không markdown fence.
"""


def _gen_question_batch(client, model, batch, max_retries, system_prompt):
    # batch: list[(id_ghep, record, target_level)]
    payload = [
        {
            "id": id_ghep,
            "target_level": target_level,
            "transcript": r["transcript"],
            "target_word": r["target_word"],
            "known_meaning": r["answer"],
            "historical_fact": r.get("historical_fact"),
        }
        for id_ghep, r, target_level in batch
    ]
    ids_sent = {id_ghep for id_ghep, r, target_level in batch}
    last_error = None
    for attempt in range(max_retries):
        try:
            from google.genai import types

            response = client.models.generate_content(
                model=model,
                contents=json.dumps(payload, ensure_ascii=False),
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.7,
                    max_output_tokens=8192,
                ),
            )
            raw = (response.text or "").strip()
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
    tqdm.write(f"[LỖI sinh multi-hop] batch {len(batch)} sample: {last_error!r} -- các sample này bị bỏ qua.")
    return {}


def generate_questions(
    client, model, leveled_records: list[dict], output_path: Path, *,
    knowledge_graph: Optional[dict] = None,
    batch_size: int = DEFAULT_GEN_BATCH_SIZE, max_workers: int = DEFAULT_MAX_WORKERS,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> None:
    """Với mỗi sample có max_level=N, sinh ĐỦ N câu (level 1..N), ghi liền nhau theo thứ tự
    sample gốc vào output_path (append, resumable qua id ghép "<id>__L<level>").

    knowledge_graph=None -> hành vi CŨ y hệt; khi có, nối thêm luật debate đã tinh chỉnh (cho cả
    3 level) vào system prompt cho pha sinh hàng loạt."""
    system_prompt = MULTIHOP_SYSTEM_PROMPT + "".join(
        rule_addendum_text(knowledge_graph, f"level_{lvl}") for lvl in (1, 2, 3)
    )

    done_ghep_ids = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done_ghep_ids.add(json.loads(line)["id"])
        print(f"Đã có sẵn {len(done_ghep_ids)} câu hỏi trong {output_path} -- resume.")

    expanded = []
    for r in leveled_records:
        for target_level in range(1, r["max_level"] + 1):
            id_ghep = f"{r['id']}__L{target_level}"
            if id_ghep in done_ghep_ids:
                continue
            expanded.append((id_ghep, r, target_level))

    print(f"Cần sinh {len(expanded)} câu hỏi (tổng theo max_level của từng sample, "
          f"từ {len(leveled_records)} sample gốc).")

    batches = [expanded[i:i + batch_size] for i in range(0, len(expanded), batch_size)]
    batch_results: dict = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_gen_question_batch, client, model, b, max_retries, system_prompt): b for b in batches}
        for future in tqdm(as_completed(futures), total=len(futures), desc="sinh câu hỏi multi-hop"):
            batch_results.update(future.result())

    n_written = n_failed = 0
    with open(output_path, "a", encoding="utf-8") as f:
        for r in leveled_records:
            for target_level in range(1, r["max_level"] + 1):
                id_ghep = f"{r['id']}__L{target_level}"
                if id_ghep in done_ghep_ids:
                    continue
                gen = batch_results.get(id_ghep)
                if gen is None:
                    n_failed += 1
                    continue
                record = {
                    "id": id_ghep,
                    "source_id": r["id"],
                    "audio": r.get("audio"),
                    "audio_id": r.get("audio_id"),
                    "level": target_level,
                    "max_level": r["max_level"],
                    "question_type": {1: "1-hop", 2: "2-hop", 3: "3-hop"}[target_level],
                    "target_word": r["target_word"],
                    "question": gen["question"],
                    "choices": gen["choices"],
                    "answer": gen["answer"],
                    "transcript": r["transcript"],
                    "historical_fact": r.get("historical_fact") if target_level == 3 else None,
                    "dataset": r.get("dataset"),
                    # Chỉ có giá trị THẬT cho record từ build-new-word-records (đã điền sẵn, vì
                    # id "hv-new-..." không tồn tại trong test_speech.jsonl nên fill-fields sẽ
                    # bỏ qua) -- record từ 95 câu gốc thì các field này là None ở đây, sẽ được
                    # fill-fields điền đúng giá trị THẬT ngay sau bước này.
                    "task": r.get("task"), "split": r.get("split"), "category": r.get("category"),
                    "sub-category": r.get("sub-category"), "difficulty": r.get("difficulty"),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_written += 1

    print(f"\nĐã ghi thêm {n_written} câu hỏi multi-hop vào {output_path} "
          f"({n_failed} bị bỏ qua do lỗi/response hỏng).")

    from collections import Counter

    with open(output_path, encoding="utf-8") as f:
        all_out = [json.loads(line) for line in f if line.strip()]
    print(f"Phân bố level trong file kết quả: {dict(Counter(r.get('level') for r in all_out))}")
    print(f"Số sample gốc có mặt: {len({r.get('source_id', r['id']) for r in all_out})}")


# ============================================================================================
# Step 3: fill-fields
# ============================================================================================

def fill_fields(multihop_path: Path, original_path: Path) -> None:
    """Join lại task/split/category/subcategory/subsubcategory/difficulty từ file gốc
    (test_speech.jsonl) theo "source_id" (id gốc, KHÔNG phải id ghép "<id>__L<n>")."""
    with open(original_path, encoding="utf-8") as f:
        original_by_id = {json.loads(line)["id"]: json.loads(line) for line in f if line.strip()}

    with open(multihop_path, encoding="utf-8") as f:
        multihop_records = [json.loads(line) for line in f if line.strip()]

    n_filled = n_missing_original = 0
    filled_records = []
    for r in multihop_records:
        original = original_by_id.get(r.get("source_id", r["id"]))
        if original is None:
            n_missing_original += 1
            filled_records.append(r)
            continue

        new_r = dict(r)
        for field in FIELDS_TO_FILL:
            new_r[field] = original.get(field)
        filled_records.append(new_r)
        n_filled += 1

    print(f"Đã fill đủ field cho {n_filled}/{len(multihop_records)} sample "
          f"({n_missing_original} sample không tìm thấy id gốc trong {original_path}).")

    with open(multihop_path, "w", encoding="utf-8") as f:
        for r in filled_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Đã ghi đè {multihop_path} với đầy đủ field.")


# ============================================================================================
# Step 0 (bổ sung, TRƯỚC classify-levels): build-new-word-records
#
# 95 record gốc của han_viet_qa.jsonl là 1 danh sách CỐ ĐỊNH -- knowledge graph (mục "Tri thức
# nền") có thể debate/xác thực/THÊM những từ Hán Việt hoàn toàn MỚI (status="added") mà 95 record
# đó không hề có, nên classify-levels/generate-questions KHÔNG BAO GIỜ sinh được câu hỏi cho
# những từ mới đó nếu chỉ đọc han_viet_qa.jsonl. Bước này lấp lỗ hổng: nhận input là output của
# `hien_tuong_filter_pipeline.py build-samples` (quét 1 corpus THẬT SỰ CÓ AUDIO, ví dụ
# release_hf_transcripts_by_dataset.json -- KHÔNG dùng full_transcripts.json vì file đó không có
# field "audio") trên CSV do `apply_knowledge_graph.py` xuất ra từ knowledge graph, rồi build
# thẳng record đã có max_level/historical_fact (lấy TỪ knowledge graph, không gọi Gemini lại vì
# những từ này đã được 3-model debate xác thực) -- ghi THÊM (append) vào chính file
# --output của classify-levels, để generate-questions đọc 1 lần là ra cả 95 record cũ VÀ các từ
# mới cùng lúc.
# ============================================================================================

def build_new_word_records(samples: list[dict], list_key: str = "han_viet_xuat_hien", id_prefix: str = "hv-new") -> list[dict]:
    """samples: output của `hien_tuong_filter_pipeline.py build-samples` (mỗi sample có field
    list_key -- list các từ Hán Việt MỚI tìm thấy thật trong transcript của sample đó, kèm
    fields tu/meaning/category_hint/historical_fact lấy từ CSV do apply_knowledge_graph.py xuất
    ra). Trả về record ở ĐÚNG schema mà classify-levels ghi ra (id/target_word/transcript/answer/
    max_level/historical_fact/level_reason + audio/audio_id/dataset/task/split/category/
    sub-category/difficulty đã điền sẵn -- KHÔNG cần fill-fields nữa vì các id "hv-new-..." này
    không tồn tại trong test_speech.jsonl gốc, fill-fields sẽ bỏ qua chúng)."""
    records = []
    for i, s in enumerate(samples):
        found = s.get(list_key) or []
        for j, item in enumerate(found):
            fact = item.get("historical_fact") or None
            category_hint = item.get("category_hint") or None
            max_level = 3 if fact else (2 if category_hint else 1)
            audio = s.get("audio") or s.get("audio_filepath")
            records.append({
                "id": f"{id_prefix}-{i + 1:04d}-w{j}",
                "target_word": item["tu"],
                "answer": item.get("meaning", ""),
                "transcript": s.get("transcript") or s.get("text") or "",
                "audio": audio,
                "audio_id": s.get("audio_id") or audio,
                "max_level": max_level,
                "historical_fact": fact,
                "level_reason": "Từ MỚI đã được 3-model debate xác thực (knowledge graph) và tìm "
                                "thấy thật trong transcript qua build-samples -- không gọi Gemini tự phân loại lại.",
                "dataset": s.get("dataset"),
                "task": "speech", "split": "test", "category": "Reasoning",
                "sub-category": "Hiện tượng đặc biệt trong tiếng Việt", "difficulty": {1: "easy", 2: "medium", 3: "hard"}[max_level],
            })
    return records


# ============================================================================================
# CLI
# ============================================================================================

def _load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p0 = sub.add_parser("build-new-word-records", help="Ghép thêm record cho từ MỚI (knowledge graph) vào file --append-to của classify-levels.")
    p0.add_argument("--samples", required=True, help="Output của hien_tuong_filter_pipeline.py build-samples (dùng corpus CÓ audio, ví dụ release_hf_transcripts_by_dataset.json).")
    p0.add_argument("--list-key", default="han_viet_xuat_hien")
    p0.add_argument("--append-to", required=True, help="han_viet_difficulty_levels.jsonl (file --output của classify-levels -- ghi THÊM vào cuối).")

    p1 = sub.add_parser("classify-levels", help="Phân loại độ khó 1/2/3-hop cho từng sample.")
    p1.add_argument("--service-account-json", required=True)
    p1.add_argument("--input", required=True, help="han_viet_qa.jsonl (đã có field transcript).")
    p1.add_argument("--output", required=True, help="han_viet_difficulty_levels.jsonl")
    p1.add_argument("--knowledge-json", default=None, help="Knowledge graph đã debate (manual_model_relay.py) -- optional.")
    p1.add_argument("--model", default=DEFAULT_MODEL)
    p1.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p1.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    p1.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)

    p2 = sub.add_parser("generate-questions", help="Sinh N câu hỏi multi-hop cho mỗi sample (N=max_level).")
    p2.add_argument("--service-account-json", required=True)
    p2.add_argument("--input", required=True, help="han_viet_difficulty_levels.jsonl")
    p2.add_argument("--output", required=True, help="han_viet_multihop_qa.jsonl")
    p2.add_argument("--knowledge-json", default=None)
    p2.add_argument("--model", default=DEFAULT_MODEL)
    p2.add_argument("--batch-size", type=int, default=DEFAULT_GEN_BATCH_SIZE)
    p2.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    p2.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)

    p3 = sub.add_parser("fill-fields", help="Join lại task/category/... từ test_speech.jsonl gốc.")
    p3.add_argument("--multihop", required=True, help="han_viet_multihop_qa.jsonl (sẽ bị ghi đè).")
    p3.add_argument("--original", required=True, help="test_speech.jsonl gốc.")

    args = parser.parse_args(argv)

    if args.command == "build-new-word-records":
        with open(args.samples, encoding="utf-8") as f:
            samples = json.load(f)
        new_records = build_new_word_records(samples, list_key=args.list_key)
        with open(args.append_to, "a", encoding="utf-8") as f:
            for r in new_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Đã ghi thêm {len(new_records)} record từ MỚI vào {args.append_to}.")

    elif args.command == "classify-levels":
        client = load_gemini_client(args.service_account_json)
        records = _load_jsonl(Path(args.input))
        print(f"Đã đọc {len(records)} sample từ {args.input}")
        knowledge_graph = load_knowledge_graph(Path(args.knowledge_json)) if args.knowledge_json else None
        leveled = classify_levels(
            client, args.model, records, knowledge_graph=knowledge_graph,
            batch_size=args.batch_size, max_workers=args.max_workers, max_retries=args.max_retries,
        )
        with open(args.output, "w", encoding="utf-8") as f:
            for r in leveled:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        from collections import Counter

        print(f"\nPhân bố mức độ: {dict(Counter(r['max_level'] for r in leveled))}")
        print(f"Đã lưu {args.output}")

    elif args.command == "generate-questions":
        client = load_gemini_client(args.service_account_json)
        leveled_records = _load_jsonl(Path(args.input))
        print(f"Đã đọc {len(leveled_records)} sample đã phân loại mức độ.")
        knowledge_graph = load_knowledge_graph(Path(args.knowledge_json)) if args.knowledge_json else None
        generate_questions(
            client, args.model, leveled_records, Path(args.output), knowledge_graph=knowledge_graph,
            batch_size=args.batch_size, max_workers=args.max_workers, max_retries=args.max_retries,
        )

    elif args.command == "fill-fields":
        fill_fields(Path(args.multihop), Path(args.original))


if __name__ == "__main__":
    main()
