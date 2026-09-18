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
# env_paths.py sống ở thư mục ANH EM cac_hien_tuong_dac_biet_trong_tieng_viet/ (cùng cấp speech/).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cac_hien_tuong_dac_biet_trong_tieng_viet"))

import env_paths  # noqa: E402
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
# Step 4: scan-dictionary
# ============================================================================================

def scan_dictionary(dictionary_path: Path, transcripts_path: Path, output_path: Path) -> None:
    """Đọc cs_broad_new.txt (1 token/dòng, đã lowercase, KHÔNG có phrase nhiều từ) thành 1 set.
    Với mỗi sample trong giga_speech_test.jsonl, tách "text" theo khoảng trắng (transcript đã ở
    dạng ASR-normalized, tách trắng là đủ ranh giới từ), so khớp CHÍNH XÁC (lowercase, word-
    boundary, KHÔNG lọc độ dài -- dùng đúng nguyên từ điển chuyên gia cung cấp) từng token với
    dictionary set. KHÔNG suy luận cụm nhiều từ (dictionary chỉ có token đơn)."""
    with open(dictionary_path, encoding="utf-8") as f:
        dictionary = {line.strip().lower() for line in f if line.strip()}

    with open(transcripts_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    n_with_terms = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            tokens = r["text"].lower().split()
            cs_terms = [t for t in tokens if t in dictionary]
            record = dict(r)
            record["cs_terms"] = cs_terms
            if cs_terms:
                n_with_terms += 1
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Đã quét {len(records)} sample với từ điển {len(dictionary)} token -- "
          f"{n_with_terms} sample có >=1 cs_terms -- đã lưu {output_path}")


# ============================================================================================
# Step 5: merge-datasets
# ============================================================================================

_DATASET_SOURCE_TEXT_FIELD = {
    "GigaSpeech": "text",
    "ViMed_Hard": "segment_text",
    "ViMed": "segment_text",
}


def merge_datasets(giga_scanned_path: Path, vimed_hard_path: Path, vimed_normal_path: Path, output_path: Path) -> None:
    """Hợp nhất 3 nguồn thành code_switching_qa.jsonl. MỖI record GIỮ NGUYÊN VẸN toàn bộ field
    gốc (không xóa field nào -- để truy vết ngược lại data gốc) + THÊM 2 field chuẩn hóa đồng
    nhất giữa mọi nguồn ("text", "cs_terms") + "dataset_source". Đường dẫn audio
    ("audio_filepath" ở GigaSpeech, "audio" ở ViMed) giữ NGUYÊN key + value gốc -- KHÔNG đổi tên,
    KHÔNG parse lại."""
    sources = [
        ("GigaSpeech", giga_scanned_path),
        ("ViMed_Hard", vimed_hard_path),
        ("ViMed", vimed_normal_path),
    ]

    n_written = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for dataset_source, path in sources:
            text_field = _DATASET_SOURCE_TEXT_FIELD[dataset_source]
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    record = dict(r)
                    record["id"] = f"cs-{n_written:05d}"
                    record["text"] = r[text_field]
                    record["cs_terms"] = r.get("cs_terms") or []
                    record["dataset_source"] = dataset_source
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_written += 1

    print(f"Đã hợp nhất {n_written} sample từ 3 nguồn (GigaSpeech + ViMed_Hard + ViMed) vào {output_path}")


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
    env_paths.add_location_arg(p1)
    p1.add_argument("--output-dir", default=None, help="Mặc định: env_paths.code_switching_dir(--location).")

    p2 = sub.add_parser("build-vi-input", help="Join transcript thô (GigaSpeech2-vi) với audio thật.")
    env_paths.add_location_arg(p2)
    p2.add_argument("--transcripts", default=None, help="Mặc định: {code_switching_dir}/giga_speech_test.jsonl.")
    p2.add_argument("--release-hf-json", required=True)
    p2.add_argument("--dataset-name", default="gigaspeech2_vi")
    p2.add_argument("--output", default=None, help="Mặc định: {code_switching_dir}/code_switching_vi_input.jsonl.")

    p3 = sub.add_parser("extract-terms", help="Gemini phát hiện từ code-switch THẬT trong transcript.")
    env_paths.add_location_arg(p3)
    p3.add_argument("--service-account-json", default=None, help="Mặc định: env_paths.gemini_service_account_path(--location) khi --location drive.")
    p3.add_argument("--input", default=None, help="Mặc định: {code_switching_dir}/code_switching_vi_input.jsonl.")
    p3.add_argument("--output", default=None, help="Mặc định: {code_switching_dir}/code_switching_vi_terms.jsonl.")
    p3.add_argument("--model", default=DEFAULT_MODEL)
    p3.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p3.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    p3.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)

    p4 = sub.add_parser("scan-dictionary", help="Quét cs_broad_new.txt trên giga_speech_test.jsonl (khớp token chính xác).")
    env_paths.add_location_arg(p4)
    p4.add_argument("--dictionary", default=None, help="Mặc định: {code_switching_dir}/cs_broad_new.txt.")
    p4.add_argument("--transcripts", default=None, help="Mặc định: {code_switching_dir}/giga_speech_test.jsonl.")
    p4.add_argument("--output", default=None, help="Mặc định: {code_switching_dir}/giga_speech_scanned.jsonl.")

    p5 = sub.add_parser("merge-datasets", help="Hợp nhất GigaSpeech (đã quét) + ViMed_Hard + ViMed thành code_switching_qa.jsonl.")
    env_paths.add_location_arg(p5)
    p5.add_argument("--giga-scanned", default=None, help="Mặc định: {code_switching_dir}/giga_speech_scanned.jsonl.")
    p5.add_argument("--vimed-hard", default=None, help="Mặc định: {code_switching_dir}/vimed_css_test_hard.jsonl.")
    p5.add_argument("--vimed-normal", default=None, help="Mặc định: {code_switching_dir}/vimed_css_test.jsonl.")
    p5.add_argument("--output", default=None, help="Mặc định: {code_switching_dir}/code_switching_qa.jsonl.")

    args = parser.parse_args(argv)

    if args.command == "build-manifest":
        cs_dir = env_paths.code_switching_dir(args.location)
        output_dir = Path(args.output_dir) if args.output_dir else cs_dir
        build_manifest(output_dir)

    elif args.command == "build-vi-input":
        cs_dir = env_paths.code_switching_dir(args.location)
        transcripts = Path(args.transcripts) if args.transcripts else cs_dir / "giga_speech_test.jsonl"
        output = Path(args.output) if args.output else cs_dir / "code_switching_vi_input.jsonl"
        build_vi_input(transcripts, Path(args.release_hf_json), output, dataset_name=args.dataset_name)

    elif args.command == "extract-terms":
        cs_dir = env_paths.code_switching_dir(args.location)
        service_account_json = args.service_account_json or (
            str(env_paths.gemini_service_account_path(args.location)) if args.location == "drive" else None
        )
        assert service_account_json, "--service-account-json bắt buộc khi --location local (không có Vertex service account cho local)."
        input_path = Path(args.input) if args.input else cs_dir / "code_switching_vi_input.jsonl"
        output = Path(args.output) if args.output else cs_dir / "code_switching_vi_terms.jsonl"
        client = load_gemini_client(service_account_json)
        records = _load_jsonl(input_path)
        extract_terms(
            client, args.model, records, output,
            batch_size=args.batch_size, max_workers=args.max_workers, max_retries=args.max_retries,
        )

    elif args.command == "scan-dictionary":
        cs_dir = env_paths.code_switching_dir(args.location)
        dictionary = Path(args.dictionary) if args.dictionary else cs_dir / "cs_broad_new.txt"
        transcripts = Path(args.transcripts) if args.transcripts else cs_dir / "giga_speech_test.jsonl"
        output = Path(args.output) if args.output else cs_dir / "giga_speech_scanned.jsonl"
        scan_dictionary(dictionary, transcripts, output)

    elif args.command == "merge-datasets":
        cs_dir = env_paths.code_switching_dir(args.location)
        giga_scanned = Path(args.giga_scanned) if args.giga_scanned else cs_dir / "giga_speech_scanned.jsonl"
        vimed_hard = Path(args.vimed_hard) if args.vimed_hard else cs_dir / "vimed_css_test_hard.jsonl"
        vimed_normal = Path(args.vimed_normal) if args.vimed_normal else cs_dir / "vimed_css_test.jsonl"
        output = Path(args.output) if args.output else cs_dir / "code_switching_qa.jsonl"
        merge_datasets(giga_scanned, vimed_hard, vimed_normal, output)


if __name__ == "__main__":
    main()
