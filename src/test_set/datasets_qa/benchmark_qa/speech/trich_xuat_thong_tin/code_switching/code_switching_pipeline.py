"""Code-switching pipeline: (1) MMSU English comprehension subset, (2) từ code-switch THẬT trong
audio tiếng Việt (GigaSpeech2-vi).

MMSU (ddwang2000/MMSU) không có config/subset riêng cho từng hiện tượng -- toàn bộ 5000 sample
nằm trong 1 split "train" duy nhất, phân biệt qua cột "task_name" (47 giá trị). Subset
"Code-switching" tương ứng task_name == "code_switch_question_answering" (111/5000 sample,
category=Reasoning, sub-category=Linguistics, sub-sub-category=Semantics). Các câu hỏi gốc của
MMSU chỉ là listening comprehension chung (KHÔNG hỏi trực tiếp về từ bị code-switch) -- để hỏi
được "từ này thuộc loại gì/dùng làm gì/Việt hóa là gì" cần tự xác định (các) từ code-switch THẬT
trong 1 corpus tiếng Việt khác (GigaSpeech2-vi), vì MMSU không cung cấp span-level annotation nào.

3 subcommand, chạy độc lập theo 2 nhánh:
  1. build-manifest  -- (nhánh MMSU) tải 3 shard parquet của MMSU (dataset không hỗ trợ lọc theo
                         cột khi tải), giữ lại CHỈ sample thuộc subset Code-switching, ghi audio
                         .wav + manifest JSONL (question/choices/answer gốc tiếng Anh).
  2. build-vi-input -- (nhánh GigaSpeech2-vi) join 1 file transcript thô (audio_filepath nội bộ,
                         KHÔNG tải được) với release_hf_transcripts_by_dataset.json (CÓ audio
                         thật) theo transcript khớp NGUYÊN VĂN, để có (transcript, audio thật)
                         làm input cho bước sau.
  3. extract-terms   -- (nhánh GigaSpeech2-vi) Gemini phát hiện (các) từ/cụm code-switching THẬT
                         trong mỗi transcript (verify lại NGUYÊN VĂN xuất hiện trong câu để chống
                         hallucination), gán ngôn ngữ gốc + loại từ.

Usage:
    python code_switching_pipeline.py build-manifest \\
        --output-dir /path/to/code_switching

    python code_switching_pipeline.py build-vi-input \\
        --transcripts /path/to/giga_speech_test.jsonl \\
        --release-hf-json /path/to/release_hf_transcripts_by_dataset.json \\
        --output /path/to/code_switching_vi_input.jsonl

    python code_switching_pipeline.py extract-terms \\
        --service-account-json /path/to/gemini_service_account.json \\
        --input /path/to/code_switching_vi_input.jsonl \\
        --output /path/to/code_switching_vi_terms.jsonl
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

# Bootstrap sys.path cho layout thư mục tách rời: file này ở .../speech/trich_xuat_thong_tin/
# code_switching/, còn translate_dataset.py ở 3 cấp trên (datasets_qa/) -- không tự nằm trên
# sys.path khi Colab chạy `!python .../code_switching_pipeline.py` trực tiếp.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from translate_dataset import DEFAULT_MODEL, load_gemini_client  # noqa: E402

TASK_NAME = "code_switch_question_answering"
DEFAULT_BATCH_SIZE = 15
DEFAULT_MAX_WORKERS = 4
DEFAULT_MAX_RETRIES = 3


# ============================================================================================
# Step 1: build-manifest
# ============================================================================================

def build_manifest(output_dir: Path) -> None:
    """Tải toàn bộ 3 shard parquet của ddwang2000/MMSU, lọc lại CHỈ sample có
    task_name == TASK_NAME (subset Code-switching), ghi audio .wav + manifest JSONL."""
    import soundfile as sf
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "code_switching_test.jsonl"

    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        print(f"{manifest_path} đã tồn tại -- đọc lại {len(records)} sample, không tải lại.")
        return

    local_dir = snapshot_download(
        repo_id="ddwang2000/MMSU",
        repo_type="dataset",
        allow_patterns=["data/train-*.parquet"],
        max_workers=8,
    )
    ds = load_dataset("parquet", data_files=f"{local_dir}/data/train-*.parquet", split="train")
    print(f"Đã load {len(ds)} sample (toàn bộ MMSU) từ {local_dir}.")

    ds = ds.filter(lambda r: r["task_name"] == TASK_NAME)
    print(f"Giữ lại {len(ds)} sample thuộc subset Code-switching (task_name=\"{TASK_NAME}\").")

    def _write_one(idx_row):
        _, row = idx_row
        sid = row["id"]
        out_wav = audio_dir / f"{sid}.wav"
        if not out_wav.exists():
            audio = row["audio"]
            sf.write(out_wav, audio["array"], audio["sampling_rate"])
        return {
            "id": sid,
            "audio_path": f"audio/{sid}.wav",
            "question": row["question"],
            "choices": [row["choice_a"], row["choice_b"], row["choice_c"], row["choice_d"]],
            "answer": row["answer_gt"],
            "category": row["category"],
            "sub_category": row["sub-category"],
            "sub_sub_category": row["sub-sub-category"],
        }

    # KHÔNG pre-list toàn bộ dataset đã lọc vào RAM trước khi submit (giữ nguyên thói quen lazy
    # iterate như clotho_aqa_pipeline.py, dù ở đây dataset đã lọc nhỏ nên ít rủi ro OOM hơn).
    records = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(_write_one, idx_row)
            for idx_row in tqdm(enumerate(ds), total=len(ds), desc="submit task")
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="ghi audio Code-switching"):
            records.append(future.result())

    records.sort(key=lambda r: r["id"])
    with open(manifest_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Đã lưu {len(records)} sample (audio .wav + manifest) vào {output_dir}")


# ============================================================================================
# Step 2: build-vi-input
# ============================================================================================

def build_vi_input(
    transcripts_path: Path, release_hf_path: Path, output_path: Path, *,
    dataset_name: str = "gigaspeech2_vi",
) -> None:
    """Join 1 file transcript thô (audio_filepath trỏ vào server nội bộ, KHÔNG tải được -- ví dụ
    export từ /raid/... của máy huấn luyện ASR) với release_hf_transcripts_by_dataset.json (CÓ
    audio thật, tải được) theo transcript khớp NGUYÊN VĂN (chuẩn hoá khoảng trắng/hoa-thường) --
    lấy audio path THẬT thay cho audio_filepath cũ."""
    with open(transcripts_path, encoding="utf-8") as f:
        raw = [json.loads(line) for line in f if line.strip()]
    with open(release_hf_path, encoding="utf-8") as f:
        release = json.load(f)

    pool = release.get(dataset_name, [])
    by_text = {" ".join(it["transcript"].lower().split()): it["audio"] for it in pool}

    matched = []
    for i, r in enumerate(raw):
        key = " ".join(r["text"].lower().split())
        audio = by_text.get(key)
        if audio is None:
            continue
        matched.append({
            "id": f"cs-vi-{i:05d}",
            "transcript": r["text"],
            "audio": audio,
            "audio_id": audio,
            "dataset": dataset_name,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        for r in matched:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Khớp được {len(matched)}/{len(raw)} transcript với audio thật trong \"{dataset_name}\" "
          f"({len(raw) - len(matched)} không khớp) -- đã lưu {output_path}")


# ============================================================================================
# Step 3: extract-terms
# ============================================================================================

CS_EXTRACT_SYSTEM_PROMPT = """Bạn phát hiện CODE-SWITCHING THẬT trong 1 câu transcript tiếng Việt
(do ASR phiên âm, chữ thường, không dấu câu) -- tức các từ/cụm từ NƯỚC NGOÀI được người nói chêm
NGUYÊN VĂN vào câu, CHƯA được Việt hóa hoàn toàn về cách phát âm/chính tả (ví dụ: "ceo",
"guardian", "kevin de bruyne", "fbnc", "nato" là code-switching THẬT; còn "tivi", "cà phê",
"internet", "ga" (nhà ga) đã Việt hóa hoàn toàn thì KHÔNG tính).

QUY TẮC BẮT BUỘC:
- CHỈ trả về từ/cụm xuất hiện NGUYÊN VĂN, ĐÚNG TỪNG CHỮ trong transcript đã cho (không được thêm
  dấu, viết hoa, hay sửa chính tả) -- nếu không chắc chắn từ đó in y hệt trong câu, KHÔNG đưa vào.
- Mỗi từ/cụm phải gán "origin_language" (ngôn ngữ gốc, ví dụ "Anh", "Pháp", "Hàn", "Na Uy"...) và
  "term_type" là 1 trong: "person_name" (tên người), "brand_org_name" (tên thương hiệu/tổ chức),
  "place_name" (địa danh), "technical_term" (thuật ngữ chuyên ngành/chữ viết tắt), "common_word"
  (từ thông dụng khác).
- Nếu KHÔNG có code-switching thật nào (chỉ có từ mượn đã Việt hóa hoặc không có từ nước ngoài
  nào), trả "terms": [] (mảng rỗng) -- KHÔNG bịa để có kết quả.

Input: JSON array các object {"id": str, "transcript": str}.
Output: CHỈ trả về JSON array cùng độ dài, mỗi phần tử {"id": <id đầu vào>,
"terms": [{"term": str, "origin_language": str, "term_type": str}, ...]} -- không giải thích
thêm, không markdown fence.
"""

_TERM_TYPES = {"person_name", "brand_org_name", "place_name", "technical_term", "common_word"}


def _extract_terms_batch(client, model, batch, max_retries):
    from google.genai import types

    payload = [{"id": r["id"], "transcript": r["transcript"]} for r in batch]
    by_id = {r["id"]: r for r in batch}
    ids_sent = set(by_id)
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=json.dumps(payload, ensure_ascii=False),
                config=types.GenerateContentConfig(
                    system_instruction=CS_EXTRACT_SYSTEM_PROMPT, temperature=0.0, max_output_tokens=4096,
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
                if rid not in ids_sent:
                    continue
                transcript_lower = by_id[rid]["transcript"].lower()
                verified = [
                    t for t in (item.get("terms") or [])
                    if t.get("term") and t.get("term").lower() in transcript_lower
                    and t.get("term_type") in _TERM_TYPES
                ]
                out[rid] = verified
            missing = ids_sent - set(out)
            if missing:
                raise ValueError(f"Thiếu {len(missing)}/{len(ids_sent)} id trong response")
            return out
        except Exception as e:  # noqa: BLE001
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    tqdm.write(f"[LỖI phát hiện code-switch] batch {len(batch)} sample: {last_error!r} -- mặc định terms=[].")
    return {r["id"]: [] for r in batch}


def extract_terms(
    client, model, records: list[dict], output_path: Path, *,
    batch_size: int = DEFAULT_BATCH_SIZE, max_workers: int = DEFAULT_MAX_WORKERS,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> None:
    """CHỈ ghi record nào tìm thấy >=1 từ code-switch THẬT (đã verify nguyên văn) -- record không
    có từ nào bị loại khỏi output (không có giá trị cho bước sinh câu hỏi sau này)."""
    done_ids = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line)["id"])
        print(f"Đã có sẵn {len(done_ids)} sample trong {output_path} -- resume, bỏ qua id đã có.")

    pending = [r for r in records if r["id"] not in done_ids]
    print(f"Cần phát hiện code-switch cho {len(pending)}/{len(records)} sample.")

    batches = [pending[i:i + batch_size] for i in range(0, len(pending), batch_size)]
    n_with_terms = n_without = 0
    with open(output_path, "a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_extract_terms_batch, client, model, b, max_retries): b for b in batches}
            for future in tqdm(as_completed(futures), total=len(futures), desc="phát hiện code-switch"):
                batch = futures[future]
                results = future.result()
                for r in batch:
                    terms = results.get(r["id"]) or []
                    if not terms:
                        n_without += 1
                        continue
                    record = dict(r)
                    record["terms"] = terms
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_with_terms += 1

    print(f"\nTìm thấy code-switch thật ở {n_with_terms}/{len(pending)} sample vừa xử lý "
          f"({n_without} sample không có/không xác thực được -- đã bỏ qua) -- đã ghi vào {output_path}")


# ============================================================================================
# CLI
# ============================================================================================

def _load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("build-manifest", help="Tải + xây manifest audio subset Code-switching (MMSU).")
    p1.add_argument("--output-dir", required=True)

    p2 = sub.add_parser("build-vi-input", help="Join transcript thô (GigaSpeech2-vi) với audio thật.")
    p2.add_argument("--transcripts", required=True)
    p2.add_argument("--release-hf-json", required=True)
    p2.add_argument("--dataset-name", default="gigaspeech2_vi")
    p2.add_argument("--output", required=True)

    p3 = sub.add_parser("extract-terms", help="Gemini phát hiện từ code-switch THẬT trong transcript.")
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

    elif args.command == "build-vi-input":
        build_vi_input(
            Path(args.transcripts), Path(args.release_hf_json), Path(args.output),
            dataset_name=args.dataset_name,
        )

    elif args.command == "extract-terms":
        client = load_gemini_client(args.service_account_json)
        records = _load_jsonl(Path(args.input))
        extract_terms(
            client, args.model, records, Path(args.output),
            batch_size=args.batch_size, max_workers=args.max_workers, max_retries=args.max_retries,
        )


if __name__ == "__main__":
    main()
