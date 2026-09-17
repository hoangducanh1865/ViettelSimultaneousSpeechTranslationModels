"""Vietnamese-ized loanword ("Từ mượn") reasoning QA generation pipeline (Gemini/Vertex AI).

Extracted from noteboooks/Hien_tuong_dac_biet_trong_tieng_Viet.ipynb -- the Colab notebook
should only clone this repo and invoke this script's "generate" subcommand.

For each loanword found in a sample's transcript, and each answerable aspect (source language /
original spelling / meaning), this randomly picks one of 2 distractor-generation strategies per
question (configurable weights):
  - "rule_based": distractors are REAL values of the same aspect from OTHER loanwords in the
    CSV (prioritizing same-language-and-domain words first, for harder distractors), never
    invented.
  - "llm": Gemini generates 3 plausible-but-wrong values for that aspect, given only the correct
    value (grounded, not shown other samples).

Resumable: re-running skips any (sample, correct-answer) combination already present in the
output file, so it only fills in newly-discoverable (word, aspect) combinations.

Usage:
    python tu_muon_pipeline.py generate \\
        --service-account-json /path/to/gemini_service_account.json \\
        --samples /path/to/asr_samples_with_tu_muon.json \\
        --csv /path/to/tu_muon_tieng_viet_viet_hoa_final.csv \\
        --output /path/to/tu_muon_qa.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from tqdm.auto import tqdm

from translate_dataset import DEFAULT_MODEL, load_gemini_client

DEFAULT_BATCH_SIZE = 25
DEFAULT_MAX_WORKERS = 4
DEFAULT_MAX_RETRIES = 3
CANDIDATE_COL = "Từ Tiếng Việt (Việt Hóa)"

ASPECTS = ["ngon_ngu", "tu_goc", "y_nghia"]
ASPECT_QUESTION_TEMPLATES = {
    "ngon_ngu": [
        "Trong đoạn âm thanh trên có xuất hiện một từ mượn tiếng Việt. Từ đó có nguồn gốc từ "
        "ngôn ngữ/quốc gia nào?",
        "Đoạn âm thanh trên chứa một từ vay mượn từ tiếng nước ngoài. Từ đó bắt nguồn từ ngôn "
        "ngữ nào?",
        "Có một từ mượn tiếng Việt xuất hiện trong câu nói trên. Từ đó được Việt hóa từ ngôn "
        "ngữ nào?",
        "Trong câu vừa nghe có một từ tiếng Việt vốn vay mượn từ nước ngoài. Từ đó đến từ đâu?",
        "Đoạn âm thanh trên chứa một từ mượn tiếng Việt. Xuất xứ (ngôn ngữ gốc) của từ đó là gì?",
        "Có một từ vay mượn xuất hiện trong câu nói trên. Từ này bắt nguồn từ tiếng nước nào?",
    ],
    "tu_goc": [
        "Trong đoạn âm thanh trên có xuất hiện một từ mượn tiếng Việt. Từ gốc (trước khi Việt "
        "hóa) của nó viết như thế nào?",
        "Có một từ mượn tiếng Việt xuất hiện trong câu nói trên. Từ gốc của nó là gì?",
        "Đoạn âm thanh trên chứa một từ vay mượn từ tiếng nước ngoài. Từ nguyên gốc (trước khi "
        "phiên âm sang tiếng Việt) được viết ra sao?",
        "Trong câu vừa nghe có một từ tiếng Việt vốn được Việt hóa. Cách viết nguyên bản của từ "
        "đó (chưa Việt hóa) là gì?",
        "Có một từ mượn xuất hiện trong câu nói trên. Từ đó vốn được phiên âm từ chữ nào?",
    ],
    "y_nghia": [
        "Trong đoạn âm thanh trên có xuất hiện một từ mượn tiếng Việt. Từ đó mang ý nghĩa gì?",
        "Có một từ mượn tiếng Việt xuất hiện trong câu nói trên. Từ đó dùng để chỉ điều gì?",
        "Đoạn âm thanh trên chứa một từ vay mượn. Từ này biểu thị khái niệm/sự vật gì?",
        "Trong câu vừa nghe có một từ tiếng Việt vốn vay mượn từ nước ngoài. Từ đó nghĩa là gì?",
        "Có một từ mượn xuất hiện trong câu nói trên. Người nghe hiểu từ đó theo nghĩa nào?",
    ],
}
ASPECT_LABELS = {"ngon_ngu": "ngôn ngữ nguồn gốc", "tu_goc": "từ gốc (chưa Việt hóa)", "y_nghia": "ý nghĩa"}

LLM_DISTRACTOR_SYSTEM_PROMPT = """Bạn sinh PHƯƠNG ÁN NHIỄU cho câu hỏi trắc nghiệm về từ mượn
tiếng Việt. Với mỗi mục, bạn được cho: từ mượn tiếng Việt (`tu`), khía cạnh đang hỏi
(`aspect`: "ngôn ngữ nguồn gốc" / "từ gốc (chưa Việt hóa)" / "ý nghĩa"), và giá trị ĐÚNG của khía
cạnh đó (`correct`). Nhiệm vụ: sinh 3 giá trị SAI nhưng HỢP LÝ, PHONG CÁCH/ĐỘ DÀI tương tự giá
trị đúng, đủ khó để đòi hỏi biết chính xác dữ kiện mới phân biệt được (không được vô lý/lạc đề dễ
loại trừ), và bản thân 3 giá trị SAI phải khác nhau, khác giá trị đúng.

Input: JSON array các object {"id": str, "tu": str, "aspect": str, "correct": str}.
Output: CHỈ trả về JSON array cùng độ dài, mỗi phần tử {"id": <id đầu vào>,
"distractors": [str, str, str]} -- không giải thích, không markdown fence.
"""

_ID_PREFIX_RE = re.compile(r"^(tu-muon-asr-\d+)")


def _aspect_value(aspect, ngon_ngu, tu_goc, y_nghia):
    if aspect == "ngon_ngu":
        return ngon_ngu
    if aspect == "tu_goc":
        return tu_goc
    return y_nghia


def _dataset_from_audio_filepath(audio_filepath: str) -> str:
    parts = Path(audio_filepath).parts
    idx = parts.index("16k") if "16k" in parts else None
    if idx is not None and idx + 1 < len(parts):
        return parts[idx + 1]
    if "data" in parts:
        return parts[parts.index("data") + 1]
    return Path(audio_filepath).parent.parent.name


def _sample_audio_path(s: dict) -> str:
    return s.get("audio_filepath") or s["audio"]


def _is_unambiguous_language(ngon_ngu: str) -> bool:
    return "/" not in ngon_ngu


def _candidate_priority_order(all_loanwords, correct_word, correct_row):
    same_lang_same_field = [
        (w, r) for w, r in all_loanwords if w != correct_word
        and r["Ngôn Ngữ / Nguồn Gốc"] == correct_row["Ngôn Ngữ / Nguồn Gốc"]
        and r["Nhóm Lĩnh Vực"] == correct_row["Nhóm Lĩnh Vực"]
    ]
    same_lang_only = [
        (w, r) for w, r in all_loanwords if w != correct_word
        and r["Ngôn Ngữ / Nguồn Gốc"] == correct_row["Ngôn Ngữ / Nguồn Gốc"]
        and (w, r) not in same_lang_same_field
    ]
    others = [
        (w, r) for w, r in all_loanwords if w != correct_word
        and (w, r) not in same_lang_same_field and (w, r) not in same_lang_only
    ]
    random.shuffle(same_lang_same_field)
    random.shuffle(same_lang_only)
    random.shuffle(others)
    return same_lang_same_field + same_lang_only + others


def _pick_distractors_rule_based(all_loanwords, aspect, correct_word, correct_row, correct_text, n=3):
    seen_texts = {correct_text}
    picked_texts = []
    for w, r in _candidate_priority_order(all_loanwords, correct_word, correct_row):
        if aspect == "ngon_ngu" and not _is_unambiguous_language(r["Ngôn Ngữ / Nguồn Gốc"]):
            continue
        text = _aspect_value(aspect, r["Ngôn Ngữ / Nguồn Gốc"], r["Từ Gốc"], r["Ý Nghĩa / Ghi Chú"])
        if text in seen_texts:
            continue
        seen_texts.add(text)
        picked_texts.append(text)
        if len(picked_texts) == n:
            break
    assert len(picked_texts) == n, (
        f"Không đủ {n} nhiễu KHÁC NHAU cho khía cạnh {aspect!r} của từ {correct_word!r}."
    )
    return picked_texts


def _random_strategy(weights: dict) -> str:
    strategies, w = zip(*weights.items())
    return random.choices(strategies, weights=w, k=1)[0]


def _llm_batch_call(client, model, items, max_retries):
    from google.genai import types

    payload = [
        {"id": cid, "tu": tu, "aspect": ASPECT_LABELS[aspect], "correct": correct_text}
        for cid, tu, aspect, correct_text in items
    ]
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=json.dumps(payload, ensure_ascii=False),
                config=types.GenerateContentConfig(
                    system_instruction=LLM_DISTRACTOR_SYSTEM_PROMPT,
                    temperature=0.9,
                    max_output_tokens=8192,
                ),
            )
            raw = (response.text or "").strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            results = {item["id"]: item["distractors"] for item in json.loads(raw)}
            out = {}
            for cid, tu, aspect, correct_text in items:
                distractors = results.get(cid, [])
                cleaned, seen = [], {correct_text}
                for d in distractors:
                    d = d.strip()
                    if d and d not in seen:
                        seen.add(d)
                        cleaned.append(d)
                out[cid] = cleaned[:3] if len(cleaned) >= 3 else None
            return out
        except Exception as e:  # noqa: BLE001
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    tqdm.write(f"[LỖI llm batch, {len(items)} item] {last_error!r} -- các item này fallback rule_based.")
    return {cid: None for cid, _, _, _ in items}


def generate(
    client, model, samples: list[dict], csv_path: Path, output_path: Path, *,
    strategy_weights: dict = None, batch_size: int = DEFAULT_BATCH_SIZE,
    max_workers: int = DEFAULT_MAX_WORKERS, max_retries: int = DEFAULT_MAX_RETRIES, seed: int = 42,
) -> None:
    random.seed(seed)
    strategy_weights = strategy_weights or {"rule_based": 0.5, "llm": 0.5}

    final_word_rows = {}
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            w = row[CANDIDATE_COL].strip()
            if w:
                final_word_rows[w] = row
    all_loanwords = list(final_word_rows.items())

    # Đọc dữ liệu ĐÃ SINH trước đó để biết những (sample, đáp án) nào đã hỏi rồi -- so theo
    # `answer` verbatim trong CÙNG 1 sample (khớp qua tiền tố id, bất kể hậu tố). Nhờ vậy chạy
    # lại script KHÔNG hỏi trùng lại combo (từ, khía cạnh) đã có.
    existing_answers_by_sample: dict = {}
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                m = _ID_PREFIX_RE.match(rec["id"])
                if m:
                    existing_answers_by_sample.setdefault(m.group(1), set()).add(rec["answer"])
        n_existing = sum(len(v) for v in existing_answers_by_sample.values())
        print(f"Đã có sẵn {n_existing} câu hỏi (thuộc {len(existing_answers_by_sample)} sample) "
              f"trong {output_path} -- sẽ bỏ qua đúng các (sample, đáp án) đã có.")

    prepared = []
    skipped_no_word = 0
    for i, s in enumerate(samples):
        found = s.get("tu_muon_xuat_hien") or []
        if not found:
            skipped_no_word += 1
            continue
        sample_id_prefix = f"tu-muon-asr-{i + 1:04d}"
        already_asked = existing_answers_by_sample.get(sample_id_prefix, set())

        for word_idx, tu_info in enumerate(found):
            word = tu_info["tu"]
            if word not in final_word_rows:
                continue
            available_aspects = ASPECTS if _is_unambiguous_language(tu_info["ngon_ngu_nguon_goc"]) else \
                [a for a in ASPECTS if a != "ngon_ngu"]
            for aspect in available_aspects:
                correct_choice = _aspect_value(
                    aspect, tu_info["ngon_ngu_nguon_goc"], tu_info["tu_goc"], tu_info["y_nghia_ghi_chu"]
                )
                if correct_choice in already_asked:
                    continue
                record_id = f"{sample_id_prefix}-w{word_idx}-{aspect}"
                prepared.append((record_id, s, word, aspect, correct_choice, _random_strategy(strategy_weights)))

    if skipped_no_word:
        print(f"Bỏ qua {skipped_no_word} sample thiếu tu_muon_xuat_hien.")
    print(f"Cần sinh thêm {len(prepared)} câu hỏi MỚI (combo từ+khía cạnh chưa từng hỏi).")

    llm_entries = [
        (record_id, word, aspect, correct_choice)
        for record_id, s, word, aspect, correct_choice, strategy in prepared if strategy == "llm"
    ]
    print(f"{len(llm_entries)} câu dùng chiến lược 'llm', còn lại dùng 'rule_based'.")

    llm_distractors: dict = {}
    if llm_entries:
        batches = [llm_entries[i:i + batch_size] for i in range(0, len(llm_entries), batch_size)]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_llm_batch_call, client, model, b, max_retries): b for b in batches}
            for future in tqdm(as_completed(futures), total=len(futures), desc="tu_muon LLM batches"):
                llm_distractors.update(future.result())

    n_written = n_failed_distractor = 0
    with open(output_path, "a", encoding="utf-8") as f:
        for record_id, s, word, aspect, correct_choice, strategy in tqdm(prepared, desc="tu_muon build records"):
            distractor_choices = None
            if strategy == "llm":
                distractor_choices = llm_distractors.get(record_id)
            if distractor_choices is None:
                correct_row = final_word_rows[word]
                try:
                    distractor_choices = _pick_distractors_rule_based(all_loanwords, aspect, word, correct_row, correct_choice, n=3)
                except AssertionError:
                    n_failed_distractor += 1
                    continue

            choices = distractor_choices + [correct_choice]
            random.shuffle(choices)
            record = {
                "id": record_id,
                "audio_id": f"tu-muon-asr/audio/{Path(_sample_audio_path(s)).name}",
                "question": random.choice(ASPECT_QUESTION_TEMPLATES[aspect]),
                "choices": choices,
                "answer": correct_choice,
                "dataset": _dataset_from_audio_filepath(_sample_audio_path(s)),
                "task": "speech",
                "split": "test",
                "category": "Reasoning",
                "sub-category": "Hiện tượng đặc biệt trong tiếng Việt",
                "difficulty": "easy",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"\nĐã ghi thêm {n_written} câu hỏi mới vào {output_path} "
          f"({n_failed_distractor} không đủ nhiễu khác biệt nên bị bỏ qua).")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="Sinh câu hỏi trắc nghiệm về từ mượn tiếng Việt.")
    p.add_argument("--service-account-json", required=True)
    p.add_argument("--samples", required=True, help="asr_samples_with_tu_muon.json")
    p.add_argument("--csv", required=True, help="tu_muon_tieng_viet_viet_hoa_final.csv")
    p.add_argument("--output", required=True, help="tu_muon_qa.jsonl (append, resumable)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--rule-based-weight", type=float, default=0.5)
    p.add_argument("--llm-weight", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    p.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    p.add_argument("--seed", type=int, default=42)

    args = parser.parse_args(argv)

    client = load_gemini_client(args.service_account_json)
    with open(args.samples, encoding="utf-8") as f:
        samples = json.load(f)
    print(f"Đã đọc {len(samples)} sample từ {args.samples}")

    generate(
        client, args.model, samples, Path(args.csv), Path(args.output),
        strategy_weights={"rule_based": args.rule_based_weight, "llm": args.llm_weight},
        batch_size=args.batch_size, max_workers=args.max_workers, max_retries=args.max_retries, seed=args.seed,
    )


if __name__ == "__main__":
    main()
