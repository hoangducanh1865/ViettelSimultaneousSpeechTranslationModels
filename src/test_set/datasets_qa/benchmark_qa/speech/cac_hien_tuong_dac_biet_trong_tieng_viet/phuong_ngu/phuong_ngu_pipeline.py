"""Vietnamese dialect ("Phương ngữ") region/ethnic-group reasoning QA pipeline (Gemini/Vertex AI).

Extracted from noteboooks/benchmark_qa/Benchmarck_QA_contructer copy.ipynb -- the notebook
should only clone this repo and invoke this script's subcommands.

2 subcommands, run in order:
  1. classify-region  -- for each dialectal-form sample, ask Gemini whether the dialect word/
                          phrase can be attributed CONFIDENTLY to a real, well-known Vietnamese
                          region or ethnic group (never guessing when unsure -- identified=false
                          by default).
  2. generate-questions -- for samples with region_identified=True, generate a 4-choice question
                            whose correct answer is exactly the identified region/ethnic group
                            (never invented), with 3 real-but-wrong region/ethnic-group names as
                            distractors.

Usage:
    python phuong_ngu_pipeline.py classify-region \\
        --service-account-json /path/to/gemini_service_account.json \\
        --input phuong_ngu/phuong_ngu_qa.jsonl \\
        --output phuong_ngu/phuong_ngu_region_labels.jsonl

    python phuong_ngu_pipeline.py generate-questions \\
        --service-account-json /path/to/gemini_service_account.json \\
        --input phuong_ngu/phuong_ngu_region_labels.jsonl \\
        --output phuong_ngu/phuong_ngu_region_qa.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from tqdm.auto import tqdm

# Bootstrap sys.path for the split-directory layout: this file lives in .../phuong_ngu/, but
# knowledge_graph.py sits 1 level up (cac_hien_tuong_dac_biet_trong_tieng_viet/) and
# translate_dataset.py sits 4 levels up (datasets_qa/) -- neither is on sys.path by default when
# Colab runs `!python .../phuong_ngu_pipeline.py` directly.
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent))
sys.path.insert(0, str(_THIS_DIR.parents[3]))

from knowledge_graph import load_knowledge_graph, rule_addendum_text, words_index
from translate_dataset import DEFAULT_MODEL, load_gemini_client

DEFAULT_BATCH_SIZE = 20
DEFAULT_GEN_BATCH_SIZE = 15
DEFAULT_MAX_WORKERS = 4
DEFAULT_MAX_RETRIES = 3


# ============================================================================================
# Step 1: classify-region
# ============================================================================================

REGION_SYSTEM_PROMPT = """Bạn phân tích 1 từ/cụm từ mang đặc trưng phương ngữ tiếng Việt (lấy từ
1 câu hỏi trắc nghiệm về phương ngữ), xác định xem nó có gắn liền RÕ RÀNG, CHẮC CHẮN với 1
VÙNG MIỀN (Bắc Bộ / Trung Bộ / Nam Bộ / hoặc vùng cụ thể hơn như Nghệ Tĩnh, Huế, Nam Trung Bộ...)
hoặc 1 DÂN TỘC/nhóm dân cư cụ thể của Việt Nam hay không.

QUY TẮC:
- CHỈ gán "identified: true" khi bạn CHẮC CHẮN (từ/cụm từ này là đặc trưng ngôn ngữ học đã biết
  rõ, phổ biến, không mơ hồ) -- ví dụ "mô tê răng rứa" gắn với Nghệ Tĩnh (Bắc Trung Bộ), "hen"/
  "dạ hen" gắn với Nam Bộ. KHÔNG suy đoán nếu không chắc.
- Nếu không đủ căn cứ chắc chắn, PHẢI trả "identified: false", KHÔNG được đoán liều.
- KHÔNG bịa thêm chi tiết về dân tộc/vùng miền ngoài kiến thức ngôn ngữ học phổ biến, đã được
  công nhận rộng rãi.

Input: JSON array các object {"id": str, "dialect_form": str, "transcript": str|null}.
Output: CHỈ trả về JSON array cùng độ dài, mỗi phần tử {"id": <id đầu vào>,
"identified": true|false, "region_or_ethnic_group": "<tên vùng/dân tộc, null nếu identified=false>",
"reason": "<lý do ngắn gọn>"} -- không giải thích thêm, không markdown fence.
"""


def _classify_region_batch(client, model, batch, max_retries):
    payload = [
        {"id": r["id"], "dialect_form": r["answer"], "transcript": r.get("transcript")}
        for r in batch
    ]
    ids_sent = {r["id"] for r in batch}
    last_error = None
    for attempt in range(max_retries):
        try:
            from google.genai import types

            response = client.models.generate_content(
                model=model,
                contents=json.dumps(payload, ensure_ascii=False),
                config=types.GenerateContentConfig(
                    system_instruction=REGION_SYSTEM_PROMPT,
                    temperature=0.0,
                    max_output_tokens=4096,
                ),
            )
            raw = (response.text or "").strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            results = json.loads(raw)
            out = {}
            for item in results:
                if item.get("id") in ids_sent and "identified" in item:
                    out[item["id"]] = item
            missing = ids_sent - set(out)
            if missing:
                raise ValueError(f"Thiếu {len(missing)}/{len(ids_sent)} id trong response")
            return out
        except Exception as e:  # noqa: BLE001
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    tqdm.write(f"[LỖI phân loại vùng/dân tộc] batch {len(batch)} sample: {last_error!r} -- mặc định identified=false.")
    return {r["id"]: {"identified": False, "region_or_ethnic_group": None, "reason": "Lỗi gọi API."} for r in batch}


def classify_region(
    client, model, records: list[dict], output_path: Path, *,
    knowledge_graph: Optional[dict] = None,
    batch_size: int = DEFAULT_BATCH_SIZE, max_workers: int = DEFAULT_MAX_WORKERS,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> list[dict]:
    """Resumable: đọc lại output_path nếu đã có, chỉ phân loại tiếp phần chưa có. Trả về BẢN
    SAO của records với region_identified/region_or_ethnic_group/region_reason đã điền.

    knowledge_graph=None (mặc định) -> hành vi CŨ y hệt (an toàn ngược). Khi có, từ đã được 3
    model debate xác thực (words_index) LUÔN được dùng trực tiếp -- không gọi Gemini lại, và ưu
    tiên HƠN cả cache của lần chạy trước (knowledge graph coi là nguồn đáng tin hơn 1 lần tự
    quyết định đơn lẻ của Gemini)."""
    records = [dict(r) for r in records]
    kg_words = words_index(knowledge_graph) if knowledge_graph else {}

    cache: dict = {}
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    cached = json.loads(line)
                    cache[cached["id"]] = cached
        print(f"Đã có sẵn {len(cache)} sample đã phân loại trong {output_path} -- resume.")

    pending = [r for r in records if r["id"] not in cache and r["answer"] not in kg_words]
    n_from_kg = sum(1 for r in records if r["answer"] in kg_words)
    if kg_words:
        print(f"{n_from_kg}/{len(records)} sample dùng trực tiếp dữ kiện đã debate xác thực từ knowledge graph.")
    print(f"Cần phân loại vùng/dân tộc cho {len(pending)}/{len(records)} sample.")

    if pending:
        batches = [pending[i:i + batch_size] for i in range(0, len(pending), batch_size)]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_classify_region_batch, client, model, b, max_retries): b for b in batches}
            for future in tqdm(as_completed(futures), total=len(futures), desc="phân loại vùng/dân tộc"):
                cache.update(future.result())

    for r in records:
        if r["answer"] in kg_words:
            entry = kg_words[r["answer"]]
            region = entry.get("fields", {}).get("region_or_ethnic_group")
            r["region_identified"] = bool(region)
            r["region_or_ethnic_group"] = region
            r["region_reason"] = f"Từ knowledge graph (source={entry.get('source')})."
            continue
        res = cache.get(r["id"], {})
        # cache có thể ở 1 trong 2 dạng: dữ liệu THÔ từ API ("identified") hoặc record đã ghi
        # ra file ở lần chạy trước (đã đổi tên field thành "region_identified").
        r["region_identified"] = res.get("region_identified", res.get("identified", False))
        r["region_or_ethnic_group"] = res.get("region_or_ethnic_group")
        r["region_reason"] = res.get("region_reason", res.get("reason"))

    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_identified = sum(1 for r in records if r["region_identified"])
    print(f"\n{n_identified}/{len(records)} sample xác định được vùng/dân tộc rõ ràng.")
    print(f"Đã lưu {output_path}")
    return records


# ============================================================================================
# Step 2: generate-questions
# ============================================================================================

REGION_QA_SYSTEM_PROMPT = """Bạn sinh câu hỏi trắc nghiệm 4 lựa chọn về VÙNG MIỀN/DÂN TỘC gắn với
1 từ/cụm từ phương ngữ xuất hiện trong audio tiếng Việt. Đáp án ĐÚNG PHẢI dựa đúng theo
"region_or_ethnic_group" đã cho -- KHÔNG được đổi/thêm thông tin khác. Câu hỏi KHÔNG được nhắc
thẳng từ phương ngữ mục tiêu (người nghe phải tự nhận ra qua audio). 3 phương án nhiễu PHẢI là
tên vùng miền/dân tộc THẬT khác của Việt Nam (không bịa tên vùng/dân tộc không tồn tại), khác
nhau, khác đáp án đúng.

Input: JSON array các object {"id": str, "dialect_form": str, "region_or_ethnic_group": str}.
Output: CHỈ trả về JSON array cùng độ dài, mỗi phần tử {"id": <id đầu vào>, "question": str,
"choices": [str, str, str, str], "answer": str} (answer PHẢI khớp region_or_ethnic_group đã cho
và PHẢI là 1 trong 4 choices) -- không giải thích thêm, không markdown fence.
"""


def _gen_region_qa_batch(client, model, batch, max_retries, system_prompt):
    payload = [
        {"id": r["id"], "dialect_form": r["answer"], "region_or_ethnic_group": r["region_or_ethnic_group"]}
        for r in batch
    ]
    ids_sent = {r["id"] for r in batch}
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
    tqdm.write(f"[LỖI sinh câu hỏi vùng/dân tộc] batch {len(batch)} sample: {last_error!r} -- bỏ qua.")
    return {}


def generate_questions(
    client, model, region_records: list[dict], output_path: Path, *,
    knowledge_graph: Optional[dict] = None,
    batch_size: int = DEFAULT_GEN_BATCH_SIZE, max_workers: int = DEFAULT_MAX_WORKERS,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> None:
    """Sinh câu hỏi CHỈ cho sample có region_identified=True. Resumable (append, skip id đã
    sinh xong). knowledge_graph=None -> hành vi CŨ y hệt; khi có, nối thêm luật debate đã tinh
    chỉnh vào system prompt (rule key "region") cho pha sinh hàng loạt."""
    system_prompt = REGION_QA_SYSTEM_PROMPT + rule_addendum_text(knowledge_graph, "region")

    done_ids = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line)["id"])
        print(f"Đã có sẵn {len(done_ids)} câu hỏi trong {output_path} -- resume.")

    pending = [r for r in region_records if r["region_identified"] and r["id"] not in done_ids]
    print(f"Sinh câu hỏi cho {len(pending)}/{len(region_records)} sample đã xác định vùng/dân tộc.")

    batches = [pending[i:i + batch_size] for i in range(0, len(pending), batch_size)]
    n_written = n_failed = 0
    with open(output_path, "a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_gen_region_qa_batch, client, model, b, max_retries, system_prompt): b for b in batches}
            for future in tqdm(as_completed(futures), total=len(futures), desc="sinh câu hỏi vùng/dân tộc"):
                batch = futures[future]
                results = future.result()
                for r in batch:
                    gen = results.get(r["id"])
                    if gen is None:
                        n_failed += 1
                        continue
                    record = {
                        "id": r["id"],
                        "audio_id": r.get("audio_id"),
                        "question": gen["question"],
                        "choices": gen["choices"],
                        "answer": gen["answer"],
                        "transcript": r.get("transcript"),
                        "question_type": "region-1-hop",
                        "region_or_ethnic_group": r["region_or_ethnic_group"],
                        "dataset": r.get("dataset"),
                        "category": r.get("category"),
                        "sub-category": r.get("sub-category"),
                        "sub-sub-category": r.get("sub-sub-category"),
                        "difficulty": r.get("difficulty"),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_written += 1

    print(f"\nĐã ghi thêm {n_written} câu hỏi mới vào {output_path} ({n_failed} bị bỏ qua do lỗi).")


# ============================================================================================
# CLI
# ============================================================================================

def _load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("classify-region", help="Xác định vùng/dân tộc gắn với mỗi từ phương ngữ.")
    p1.add_argument("--service-account-json", required=True)
    p1.add_argument("--input", required=True, help="phuong_ngu_qa.jsonl (đã có field transcript).")
    p1.add_argument("--output", required=True, help="phuong_ngu_region_labels.jsonl")
    p1.add_argument("--knowledge-json", default=None, help="Knowledge graph đã debate (manual_model_relay.py) -- optional.")
    p1.add_argument("--model", default=DEFAULT_MODEL)
    p1.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p1.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    p1.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)

    p2 = sub.add_parser("generate-questions", help="Sinh câu hỏi vùng/dân tộc cho sample đã xác định.")
    p2.add_argument("--service-account-json", required=True)
    p2.add_argument("--input", required=True, help="phuong_ngu_region_labels.jsonl")
    p2.add_argument("--output", required=True, help="phuong_ngu_region_qa.jsonl")
    p2.add_argument("--knowledge-json", default=None)
    p2.add_argument("--model", default=DEFAULT_MODEL)
    p2.add_argument("--batch-size", type=int, default=DEFAULT_GEN_BATCH_SIZE)
    p2.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    p2.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)

    args = parser.parse_args(argv)

    if args.command == "classify-region":
        client = load_gemini_client(args.service_account_json)
        records = _load_jsonl(Path(args.input))
        n_ambiguous = sum(1 for r in records if r.get("_ambiguous_match_count"))
        n_no_transcript = sum(1 for r in records if r.get("transcript") is None)
        print(f"Đã đọc {len(records)} sample. {n_ambiguous} sample có transcript ghép MẬP MỜ, "
              f"{n_no_transcript} sample KHÔNG có transcript.")
        knowledge_graph = load_knowledge_graph(Path(args.knowledge_json)) if args.knowledge_json else None
        classify_region(
            client, args.model, records, Path(args.output), knowledge_graph=knowledge_graph,
            batch_size=args.batch_size, max_workers=args.max_workers, max_retries=args.max_retries,
        )

    elif args.command == "generate-questions":
        client = load_gemini_client(args.service_account_json)
        region_records = _load_jsonl(Path(args.input))
        knowledge_graph = load_knowledge_graph(Path(args.knowledge_json)) if args.knowledge_json else None
        generate_questions(
            client, args.model, region_records, Path(args.output), knowledge_graph=knowledge_graph,
            batch_size=args.batch_size, max_workers=args.max_workers, max_retries=args.max_retries,
        )


if __name__ == "__main__":
    main()
