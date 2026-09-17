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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from tqdm.auto import tqdm

from translate_dataset import DEFAULT_MODEL, load_gemini_client

DEFAULT_BATCH_SIZE = 20
DEFAULT_GEN_BATCH_SIZE = 15
DEFAULT_MAX_WORKERS = 4
DEFAULT_MAX_RETRIES = 3

_QUOTE_RE = re.compile(r'"([^"]+)"')

FIELDS_TO_FILL = ["task", "split", "category", "subcategory", "subsubcategory", "difficulty"]


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
    batch_size: int = DEFAULT_BATCH_SIZE, max_workers: int = DEFAULT_MAX_WORKERS,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> list[dict]:
    """Trả về BẢN SAO của records, mỗi record thêm target_word/max_level/historical_fact/
    level_reason."""
    records = [dict(r) for r in records]
    for r in records:
        r["target_word"] = extract_target_word(r)

    n_no_target = sum(1 for r in records if r["target_word"] is None)
    print(f"CẢNH BÁO: {n_no_target}/{len(records)} sample KHÔNG trích được target_word "
          "thật từ transcript -- các sample này sẽ mặc định mức độ 1 (an toàn nhất).")

    pending = [r for r in records if r["target_word"] is not None]
    print(f"Cần Gemini phân loại mức 1/2/3 cho {len(pending)} sample "
          f"({len(records) - len(pending)} sample không có target_word sẽ mặc định mức 1).")

    batches = [pending[i:i + batch_size] for i in range(0, len(pending), batch_size)]
    level_results: dict = {}
    if batches:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_classify_level_batch, client, model, b, max_retries): b for b in batches}
            for future in tqdm(as_completed(futures), total=len(futures), desc="phân loại mức 1/2/3"):
                level_results.update(future.result())

    for r in records:
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


def _gen_question_batch(client, model, batch, max_retries):
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
                    system_instruction=MULTIHOP_SYSTEM_PROMPT,
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
    batch_size: int = DEFAULT_GEN_BATCH_SIZE, max_workers: int = DEFAULT_MAX_WORKERS,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> None:
    """Với mỗi sample có max_level=N, sinh ĐỦ N câu (level 1..N), ghi liền nhau theo thứ tự
    sample gốc vào output_path (append, resumable qua id ghép "<id>__L<level>")."""
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
        futures = {executor.submit(_gen_question_batch, client, model, b, max_retries): b for b in batches}
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
                    "target_word": r["target_word"],
                    "question": gen["question"],
                    "choices": gen["choices"],
                    "answer": gen["answer"],
                    "transcript": r["transcript"],
                    "historical_fact": r.get("historical_fact") if target_level == 3 else None,
                    "dataset": r.get("dataset"),
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
# CLI
# ============================================================================================

def _load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("classify-levels", help="Phân loại độ khó 1/2/3-hop cho từng sample.")
    p1.add_argument("--service-account-json", required=True)
    p1.add_argument("--input", required=True, help="han_viet_qa.jsonl (đã có field transcript).")
    p1.add_argument("--output", required=True, help="han_viet_difficulty_levels.jsonl")
    p1.add_argument("--model", default=DEFAULT_MODEL)
    p1.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p1.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    p1.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)

    p2 = sub.add_parser("generate-questions", help="Sinh N câu hỏi multi-hop cho mỗi sample (N=max_level).")
    p2.add_argument("--service-account-json", required=True)
    p2.add_argument("--input", required=True, help="han_viet_difficulty_levels.jsonl")
    p2.add_argument("--output", required=True, help="han_viet_multihop_qa.jsonl")
    p2.add_argument("--model", default=DEFAULT_MODEL)
    p2.add_argument("--batch-size", type=int, default=DEFAULT_GEN_BATCH_SIZE)
    p2.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    p2.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)

    p3 = sub.add_parser("fill-fields", help="Join lại task/category/... từ test_speech.jsonl gốc.")
    p3.add_argument("--multihop", required=True, help="han_viet_multihop_qa.jsonl (sẽ bị ghi đè).")
    p3.add_argument("--original", required=True, help="test_speech.jsonl gốc.")

    args = parser.parse_args(argv)

    if args.command == "classify-levels":
        client = load_gemini_client(args.service_account_json)
        records = _load_jsonl(Path(args.input))
        print(f"Đã đọc {len(records)} sample từ {args.input}")
        leveled = classify_levels(
            client, args.model, records,
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
        generate_questions(
            client, args.model, leveled_records, Path(args.output),
            batch_size=args.batch_size, max_workers=args.max_workers, max_retries=args.max_retries,
        )

    elif args.command == "fill-fields":
        fill_fields(Path(args.multihop), Path(args.original))


if __name__ == "__main__":
    main()
