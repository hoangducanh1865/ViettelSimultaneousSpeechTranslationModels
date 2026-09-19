"""ClothoAQA -> Vietnamese sound-QA pipeline (build manifest, VN-relevance filter, translate).

Extracted from noteboooks/benchmark_qa/Benchmarck_QA_contructer copy 2.ipynb -- the notebook
should only clone this repo and invoke this script's subcommands.

3 subcommands, run in order:
  1. build-manifest        -- download only the "test" parquet shards of CLAPv2/ClothoAQA (not
                               the full dataset), parse each sample's "Question: ... Answer: ..."
                               text field, write audio .wav files + a JSONL manifest.
  2. filter-vn-relevance    -- ask Gemini whether each (English) question/answer sample fits
                               Vietnamese life/culture context, keeping "relevant" and
                               "rare_but_plausible", dropping "not_relevant".
  3. translate              -- translate question/answer to Vietnamese AND generate 3 Vietnamese
                               distractors in the same Gemini call (ClothoAQA has no native
                               multiple-choice, only an open-ended answer).

Usage:
    python clotho_aqa_pipeline.py build-manifest \\
        --output-dir /path/to/clotho_aqa

    python clotho_aqa_pipeline.py filter-vn-relevance \\
        --service-account-json /path/to/gemini_service_account.json \\
        --input /path/to/clotho_aqa/clotho_aqa_test_non_yesno.jsonl \\
        --cache /path/to/clotho_aqa/clotho_aqa_non_yesno_vn_relevance_cache.json \\
        --output /path/to/clotho_aqa/clotho_aqa_test_non_yesno_vn.jsonl

    python clotho_aqa_pipeline.py translate \\
        --service-account-json /path/to/gemini_service_account.json \\
        --input /path/to/clotho_aqa/clotho_aqa_test_non_yesno_vn.jsonl \\
        --output /path/to/clotho_aqa/clotho_aqa_test_vi_qa.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from tqdm.auto import tqdm

# Bootstrap sys.path for the split-directory layout: this file lives in .../sound/, but
# translate_dataset.py sits 2 levels up (datasets_qa/) -- not on sys.path by default when Colab
# runs `!python .../clotho_aqa_pipeline.py` directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_set.datasets_qa.translate_datasets.translate_dataset import DEFAULT_MODEL, load_gemini_client

DEFAULT_BATCH_SIZE = 20
DEFAULT_MAX_WORKERS = 4
DEFAULT_MAX_RETRIES = 3
N_DISTRACTORS = 3

_QA_TEXT_RE = re.compile(r"^Question:\s*(.*?)\s*Answer:\s*(.*)$", re.DOTALL)
_YES_NO_ANSWERS = {"yes", "no"}


# ============================================================================================
# Step 1: build-manifest
# ============================================================================================

def build_manifest(output_dir: Path) -> None:
    """Tải chỉ các file parquet của split "test" (không tải cả dataset), ghi audio .wav +
    manifest JSONL, sau đó tách sẵn thành 2 file: yes/no và non-yes/no."""
    import soundfile as sf
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "clotho_aqa_test.jsonl"
    non_yesno_path = output_dir / "clotho_aqa_test_non_yesno.jsonl"

    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        print(f"{manifest_path} đã tồn tại -- đọc lại {len(records)} sample, không tải lại.")
    else:
        local_dir = snapshot_download(
            repo_id="CLAPv2/ClothoAQA",
            repo_type="dataset",
            allow_patterns=["data/test/*.parquet"],
            max_workers=8,
        )
        ds = load_dataset("parquet", data_files=f"{local_dir}/data/test/*.parquet", split="train")
        print(f"Đã load {len(ds)} sample (split test) từ {local_dir}.")

        def _write_one(idx_row):
            _, row = idx_row
            m = _QA_TEXT_RE.match(row["text"])
            if not m:
                return None
            question, answer = m.group(1).strip(), m.group(2).strip()
            sid = row["index"]
            out_wav = audio_dir / f"{sid}.wav"
            if not out_wav.exists():
                audio = row["audio"]
                sf.write(out_wav, audio["array"], audio["sampling_rate"])
            return {
                "id": sid,
                "audio_path": f"audio/{sid}.wav",
                "question": question,
                "answer": answer,
                "audio_len": row["audio_len"],
                "dataset": row["datasetname"],
            }

        # KHÔNG pre-list toàn bộ dataset vào RAM (gây OOM với dataset lớn nhiều audio) -- submit
        # trực tiếp từng row trong lúc duyệt (lazy load từ Arrow/parquet).
        records = []
        n_skipped_parse = 0
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(_write_one, idx_row)
                for idx_row in tqdm(enumerate(ds), total=len(ds), desc="submit task")
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="ghi audio ClothoAQA"):
                result = future.result()
                if result is None:
                    n_skipped_parse += 1
                else:
                    records.append(result)

        records.sort(key=lambda r: r["id"])
        if n_skipped_parse:
            print(f"Bỏ qua {n_skipped_parse} sample không parse được field 'text'.")

        with open(manifest_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Đã lưu {len(records)} sample (audio .wav + manifest) vào {output_dir}")

    non_yes_no = [r for r in records if r["answer"].strip().lower() not in _YES_NO_ANSWERS]
    with open(non_yesno_path, "w", encoding="utf-8") as f:
        for r in non_yes_no:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Đã lưu {len(non_yes_no)}/{len(records)} sample non-yes/no vào {non_yesno_path}")


# ============================================================================================
# Step 2: filter-vn-relevance
# ============================================================================================

VN_RELEVANCE_LEVELS = {"relevant", "rare_but_plausible", "not_relevant"}

VN_FILTER_SYSTEM_PROMPT = """Bạn đánh giá xem 1 sample audio-QA (mô tả bằng câu hỏi + đáp án
tiếng Anh, về 1 đoạn âm thanh môi trường/sự kiện) có PHÙ HỢP với ngữ cảnh, văn hóa, đời sống
Việt Nam hay không -- để chọn lọc dữ liệu benchmark AI dùng tại Việt Nam.

Phân loại theo 3 mức:
- "relevant": tình huống/âm thanh phổ biến, quen thuộc trong đời sống Việt Nam (mưa, xe cộ, chợ,
  động vật quen thuộc, âm nhạc, đám đông, nhà bếp...).
- "rare_but_plausible": hiếm gặp ở Việt Nam nhưng vẫn CÓ THỂ xảy ra/tồn tại (không phải hoàn toàn
  xa lạ), giữ lại để đa dạng dữ liệu.
- "not_relevant": gắn chặt với văn hóa/địa lý/khí hậu/sự vật KHÔNG tồn tại hoặc cực kỳ xa lạ ở
  Việt Nam (ví dụ: tuyết rơi, động vật vùng cực, lễ hội phương Tây đặc thù, thiết bị/công trình
  chỉ có ở nước ngoài).

Input: JSON array các object {"id": str, "question": str, "answer": str}.
Output: CHỈ trả về JSON array cùng độ dài, mỗi phần tử {"id": <id đầu vào>,
"verdict": "relevant"|"rare_but_plausible"|"not_relevant", "reason": "<lý do ngắn gọn bằng tiếng Việt>"}
-- không giải thích thêm, không markdown fence.
"""


def _classify_vn_relevance_batch(client, model, batch, max_retries):
    from google.genai import types

    payload = [{"id": r["id"], "question": r["question"], "answer": r["answer"]} for r in batch]
    ids_sent = {r["id"] for r in batch}
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=json.dumps(payload, ensure_ascii=False),
                config=types.GenerateContentConfig(
                    system_instruction=VN_FILTER_SYSTEM_PROMPT,
                    temperature=0.0,
                    max_output_tokens=4096,
                ),
            )
            raw = (response.text or "").strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            results = json.loads(raw)
            if not isinstance(results, list):
                raise ValueError("Response không phải JSON array")
            out = {}
            for item in results:
                if item.get("id") in ids_sent and item.get("verdict") in VN_RELEVANCE_LEVELS:
                    out[item["id"]] = {"verdict": item["verdict"], "reason": item.get("reason", "")}
            missing = ids_sent - set(out)
            if missing:
                raise ValueError(f"Thiếu {len(missing)}/{len(ids_sent)} id trong response")
            return out
        except Exception as e:  # noqa: BLE001
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    tqdm.write(f"[LỖI lọc VN-relevance] batch {len(batch)} sample, lỗi cuối: {last_error!r} "
               "-- mặc định GIỮ LẠI (relevant) để KHÔNG loại oan do lỗi kỹ thuật.")
    return {r["id"]: {"verdict": "relevant", "reason": "Lỗi gọi API, mặc định giữ lại."} for r in batch}


def filter_vn_relevance(
    client, model, records: list[dict], cache_path: Path, output_path: Path, *,
    batch_size: int = DEFAULT_BATCH_SIZE, max_workers: int = DEFAULT_MAX_WORKERS,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> None:
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
    else:
        cache = {}

    pending = [r for r in records if r["id"] not in cache]
    print(f"Cần chấm VN-relevance cho {len(pending)}/{len(records)} sample "
          f"(song song {max_workers} luồng, batch {batch_size}).")

    batches = [pending[i:i + batch_size] for i in range(0, len(pending), batch_size)]
    if batches:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_classify_vn_relevance_batch, client, model, b, max_retries): b for b in batches}
            for future in tqdm(as_completed(futures), total=len(futures), desc="VN-relevance ClothoAQA"):
                cache.update(future.result())
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)

    print(f"Xong: {len(cache)}/{len(records)} sample đã có nhãn VN-relevance.")

    from collections import Counter

    verdict_counter = Counter(cache[r["id"]]["verdict"] for r in records)
    print(f"Phân bố verdict: {dict(verdict_counter)}")

    kept = [r for r in records if cache[r["id"]]["verdict"] in ("relevant", "rare_but_plausible")]
    print(f"Giữ lại: {len(kept)}/{len(records)} sample phù hợp ngữ cảnh Việt Nam.")

    with open(output_path, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Đã lưu {output_path}")


# ============================================================================================
# Step 3: translate
# ============================================================================================

CLOTHO_AQA_SYSTEM_PROMPT = f"""Bạn xử lý câu hỏi-đáp về âm thanh (audio question answering) từ
tiếng Anh sang tiếng Việt, PHỤC VỤ tạo bộ trắc nghiệm 4 lựa chọn tiếng Việt.

Với mỗi sample (câu hỏi + đáp án ĐÚNG gốc tiếng Anh về 1 đoạn âm thanh), làm 2 việc:
1. Dịch "question" và "answer" sang tiếng Việt tự nhiên, đúng ngữ nghĩa, văn phong hội thoại
   (KHÔNG dịch máy móc từng chữ).
2. Sinh thêm {N_DISTRACTORS} PHƯƠNG ÁN NHIỄU (distractor) bằng tiếng Việt -- các đáp án SAI
   nhưng HỢP LÝ, CÙNG dạng/độ dài với đáp án đúng (ví dụ đáp án đúng là 1 danh từ chỉ vật/con
   vật/số lượng thì nhiễu cũng phải cùng dạng đó, không lạc format), đủ khó để đòi hỏi NGHE ĐÚNG
   âm thanh mới phân biệt được (không được vô lý/phi thực tế dễ loại trừ ngay). 3 nhiễu phải
   KHÁC NHAU và khác đáp án đúng.

Input: JSON array các object {{"id": str, "question": str, "answer": str}}.
Output: CHỈ trả về JSON array cùng độ dài, mỗi phần tử {{"id": <id đầu vào>, "question_vi": str,
"answer_vi": str, "distractors_vi": [str, ...]}} (đúng {N_DISTRACTORS} phần tử trong
distractors_vi) -- không giải thích, không markdown fence.
"""


def _translate_and_distract_batch(client, model, batch, max_retries):
    from google.genai import types

    payload = [{"id": r["id"], "question": r["question"], "answer": r["answer"]} for r in batch]
    ids_sent = {r["id"] for r in batch}
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=json.dumps(payload, ensure_ascii=False),
                config=types.GenerateContentConfig(
                    system_instruction=CLOTHO_AQA_SYSTEM_PROMPT,
                    temperature=0.7,
                    max_output_tokens=4096,
                ),
            )
            raw = (response.text or "").strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            results = json.loads(raw)
            if not isinstance(results, list):
                raise ValueError("Response không phải JSON array")
            out = {}
            for item in results:
                rid = item.get("id")
                distractors = item.get("distractors_vi") or []
                if (
                    rid in ids_sent
                    and item.get("question_vi")
                    and item.get("answer_vi")
                    and len(distractors) >= N_DISTRACTORS
                ):
                    out[rid] = {
                        "question_vi": item["question_vi"],
                        "answer_vi": item["answer_vi"],
                        "distractors_vi": distractors[:N_DISTRACTORS],
                    }
            missing = ids_sent - set(out)
            if missing:
                raise ValueError(f"Thiếu/hỏng {len(missing)}/{len(ids_sent)} id trong response")
            return out
        except Exception as e:  # noqa: BLE001
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    tqdm.write(f"[LỖI dịch+sinh nhiễu] batch {len(batch)} sample, lỗi cuối: {last_error!r} "
               "-- các sample này sẽ bị bỏ qua (KHÔNG giữ record lỗi/thiếu nhiễu).")
    return {}


def translate(
    client, model, records: list[dict], output_path: Path, *,
    batch_size: int = DEFAULT_BATCH_SIZE, max_workers: int = DEFAULT_MAX_WORKERS,
    max_retries: int = DEFAULT_MAX_RETRIES, seed: int = 42,
) -> None:
    random.seed(seed)

    done_ids = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line)["id"])
        print(f"Đã có sẵn {len(done_ids)} sample trong {output_path} -- resume, bỏ qua id đã có.")

    pending = [r for r in records if r["id"] not in done_ids]
    print(f"Cần dịch + sinh nhiễu cho {len(pending)}/{len(records)} sample.")

    batches = [pending[i:i + batch_size] for i in range(0, len(pending), batch_size)]
    n_written = n_failed = 0
    with open(output_path, "a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_translate_and_distract_batch, client, model, b, max_retries): b for b in batches}
            for future in tqdm(as_completed(futures), total=len(futures), desc="dịch + sinh nhiễu ClothoAQA"):
                batch = futures[future]
                results = future.result()
                for r in batch:
                    res = results.get(r["id"])
                    if res is None:
                        n_failed += 1
                        continue
                    choices = [res["answer_vi"]] + res["distractors_vi"]
                    random.shuffle(choices)
                    record = {
                        "id": r["id"],
                        "audio_id": f"clotho-aqa/audio/{Path(r['audio_path']).name}",
                        "question": res["question_vi"],
                        "choices": choices,
                        "answer": res["answer_vi"],
                        "dataset": r["dataset"],
                        "task": "sound",
                        "split": "test",
                        "category": "Audio Understanding",
                        "sub-category": "ClothoAQA",
                        "difficulty": "medium",
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_written += 1

    print(f"\nĐã ghi thêm {n_written} sample vào {output_path} "
          f"({n_failed} sample bị bỏ qua do lỗi/thiếu nhiễu).")


# ============================================================================================
# CLI
# ============================================================================================

def _load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("build-manifest", help="Tải + xây manifest audio ClothoAQA test split.")
    p1.add_argument("--output-dir", required=True)

    p2 = sub.add_parser("filter-vn-relevance", help="Lọc sample phù hợp ngữ cảnh Việt Nam.")
    p2.add_argument("--service-account-json", required=True)
    p2.add_argument("--input", required=True)
    p2.add_argument("--cache", required=True)
    p2.add_argument("--output", required=True)
    p2.add_argument("--model", default=DEFAULT_MODEL)
    p2.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p2.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    p2.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)

    p3 = sub.add_parser("translate", help="Dịch + sinh nhiễu 4 lựa chọn tiếng Việt.")
    p3.add_argument("--service-account-json", required=True)
    p3.add_argument("--input", required=True)
    p3.add_argument("--output", required=True)
    p3.add_argument("--model", default=DEFAULT_MODEL)
    p3.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p3.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    p3.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)

    args = parser.parse_args(argv)

    if args.command == "build-manifest":
        build_manifest(Path(args.output_dir))

    elif args.command == "filter-vn-relevance":
        client = load_gemini_client(args.service_account_json)
        records = _load_jsonl(Path(args.input))
        filter_vn_relevance(
            client, args.model, records, Path(args.cache), Path(args.output),
            batch_size=args.batch_size, max_workers=args.max_workers, max_retries=args.max_retries,
        )

    elif args.command == "translate":
        client = load_gemini_client(args.service_account_json)
        records = _load_jsonl(Path(args.input))
        translate(
            client, args.model, records, Path(args.output),
            batch_size=args.batch_size, max_workers=args.max_workers, max_retries=args.max_retries,
        )
