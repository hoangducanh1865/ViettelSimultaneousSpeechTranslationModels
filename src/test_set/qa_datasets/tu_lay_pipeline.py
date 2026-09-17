"""Vietnamese reduplicative-word ("Từ láy") reasoning QA generation pipeline (Gemini/Vertex AI).

Extracted from noteboooks/Hien_tuong_dac_biet_trong_tieng_Viet.ipynb -- the Colab notebook
should only clone this repo and invoke this script's "generate" subcommand.

Supports 3 --variant values, each backed by a different CSV schema built earlier from a
different filter of tu_lay_tieng_viet*.csv:
  - "toan_bo" : "Láy toàn bộ" words -- asks about meaning / sắc thái biểu đạt.
  - "van"     : "Láy vần" words -- asks about the shared vần / meaning / sắc thái biểu đạt.
  - "chung"   : the remaining ("chung") words -- asks about cấu tạo (loại từ láy) / từ loại
                (ngữ pháp) / meaning / sắc thái biểu đạt.

Same dual distractor strategy (random per question) as tu_muon_pipeline.py: "rule_based" (real
values from other reduplicative words in the CSV, same-type words prioritized) or "llm" (Gemini
invents 3 plausible-but-wrong values, grounded only on the correct value).

Usage:
    python tu_lay_pipeline.py generate --variant toan_bo \\
        --service-account-json /path/to/gemini_service_account.json \\
        --samples /path/to/asr_samples_with_tu_lay_toan_bo.json \\
        --csv /path/to/tu_lay_toan_bo_final.csv \\
        --output /path/to/tu_lay_toan_bo_qa.jsonl

    python tu_lay_pipeline.py generate --variant van ... (tu_lay_van_final.csv)
    python tu_lay_pipeline.py generate --variant chung ... (tu_lay_tieng_viet_final.csv)
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

WORD_COL = "Từ láy"
_VAN_RE = re.compile(r"Láy vần \(([^)]+)\)")


# ============================================================================================
# Per-variant configuration
# ============================================================================================

VARIANT_ASPECTS = {
    "toan_bo": ["y_nghia", "sac_thai"],
    "van": ["van", "y_nghia", "sac_thai"],
    "chung": ["loai_tu_lay", "tu_loai", "y_nghia", "sac_thai"],
}

VARIANT_ID_PREFIX = {
    "toan_bo": "tu-lay-toan-bo-asr",
    "van": "tu-lay-van-asr",
    "chung": "tu-lay-chung-asr",
}

VARIANT_ASPECT_LABELS = {
    "toan_bo": {"y_nghia": "ý nghĩa", "sac_thai": "sắc thái biểu đạt"},
    "van": {"van": "phần vần lặp lại giữa 2 tiếng", "y_nghia": "ý nghĩa", "sac_thai": "sắc thái biểu đạt"},
    "chung": {
        "loai_tu_lay": "loại từ láy (cấu tạo ngữ âm)", "tu_loai": "từ loại (ngữ pháp)",
        "y_nghia": "ý nghĩa", "sac_thai": "sắc thái biểu đạt",
    },
}

QUESTION_TEMPLATES = {
    "toan_bo": {
        "y_nghia": [
            "Trong đoạn âm thanh trên có xuất hiện một từ láy tiếng Việt. Từ đó mang ý nghĩa gì?",
            "Có một từ láy tiếng Việt xuất hiện trong câu nói trên. Từ đó diễn tả điều gì?",
            "Đoạn âm thanh trên chứa một từ láy. Từ này miêu tả điều gì?",
            "Trong câu vừa nghe có một từ láy đặc trưng của tiếng Việt. Từ đó có nghĩa là gì?",
            "Có một từ láy xuất hiện trong câu nói trên. Người nghe hiểu từ đó theo ý nào?",
        ],
        "sac_thai": [
            "Trong đoạn âm thanh trên có xuất hiện một từ láy tiếng Việt. Từ đó mang sắc thái biểu "
            "đạt như thế nào?",
            "Có một từ láy tiếng Việt xuất hiện trong câu nói trên. Sắc thái biểu đạt của từ đó là gì?",
            "Đoạn âm thanh trên chứa một từ láy. Từ này gợi cảm xúc/mức độ theo chiều hướng nào?",
            "Trong câu vừa nghe có một từ láy đặc trưng của tiếng Việt. Sắc thái (tăng/giảm mức độ, "
            "tích cực/tiêu cực...) của từ đó là gì?",
            "Có một từ láy xuất hiện trong câu nói trên. Từ đó tạo cảm giác/ấn tượng gì cho người nghe?",
        ],
    },
    "van": {
        "van": [
            "Trong đoạn âm thanh trên có xuất hiện một từ láy vần tiếng Việt (2 tiếng chỉ giống "
            "nhau ở phần vần, khác phụ âm đầu). Phần vần được lặp lại giữa 2 tiếng đó là gì?",
            "Có một từ láy vần xuất hiện trong câu nói trên. Phần vần chung giữa 2 tiếng đó là gì?",
            "Đoạn âm thanh trên chứa một từ láy vần. Vần được lặp lại trong từ đó là gì?",
            "Trong câu vừa nghe có một từ láy mà 2 tiếng chỉ bắt vần với nhau (khác phụ âm đầu). "
            "Phần vần chung đó là gì?",
        ],
        "y_nghia": [
            "Trong đoạn âm thanh trên có xuất hiện một từ láy vần tiếng Việt. Từ đó mang ý nghĩa gì?",
            "Có một từ láy vần xuất hiện trong câu nói trên. Từ đó diễn tả điều gì?",
            "Đoạn âm thanh trên chứa một từ láy vần. Từ này miêu tả điều gì?",
            "Trong câu vừa nghe có một từ láy vần. Từ đó có nghĩa là gì?",
        ],
        "sac_thai": [
            "Trong đoạn âm thanh trên có xuất hiện một từ láy vần tiếng Việt. Từ đó mang sắc thái "
            "biểu đạt như thế nào?",
            "Có một từ láy vần xuất hiện trong câu nói trên. Sắc thái biểu đạt của từ đó là gì?",
            "Đoạn âm thanh trên chứa một từ láy vần. Từ này gợi cảm xúc/thái độ theo chiều hướng nào?",
            "Trong câu vừa nghe có một từ láy vần. Sắc thái (tích cực/tiêu cực, mạnh/nhẹ...) của từ "
            "đó là gì?",
        ],
    },
    "chung": {
        "loai_tu_lay": [
            "Trong đoạn âm thanh trên có xuất hiện một từ láy tiếng Việt. Từ đó thuộc loại từ láy nào "
            "(toàn bộ, âm đầu, vần...)?",
            "Có một từ láy tiếng Việt xuất hiện trong câu nói trên. Xét theo cấu tạo, từ đó là loại "
            "từ láy gì?",
            "Đoạn âm thanh trên chứa một từ láy. Về mặt cấu tạo ngữ âm, đây là loại từ láy nào?",
        ],
        "tu_loai": [
            "Trong đoạn âm thanh trên có xuất hiện một từ láy tiếng Việt. Từ đó thuộc từ loại nào "
            "(danh từ, động từ, tính từ...)?",
            "Có một từ láy tiếng Việt xuất hiện trong câu nói trên. Xét theo ngữ pháp, từ đó là từ "
            "loại gì?",
            "Đoạn âm thanh trên chứa một từ láy. Từ này đóng vai trò từ loại nào trong câu?",
        ],
        "y_nghia": [
            "Trong đoạn âm thanh trên có xuất hiện một từ láy tiếng Việt. Từ đó mang ý nghĩa gì?",
            "Có một từ láy tiếng Việt xuất hiện trong câu nói trên. Từ đó diễn tả điều gì?",
            "Đoạn âm thanh trên chứa một từ láy. Từ này miêu tả điều gì?",
        ],
        "sac_thai": [
            "Trong đoạn âm thanh trên có xuất hiện một từ láy tiếng Việt. Từ đó mang sắc thái biểu "
            "đạt như thế nào?",
            "Có một từ láy tiếng Việt xuất hiện trong câu nói trên. Sắc thái biểu đạt của từ đó là gì?",
            "Đoạn âm thanh trên chứa một từ láy. Từ này gợi cảm xúc/mức độ theo chiều hướng nào?",
        ],
    },
}

LLM_DISTRACTOR_SYSTEM_PROMPTS = {
    "toan_bo": """Bạn sinh PHƯƠNG ÁN NHIỄU cho câu hỏi trắc nghiệm về từ láy tiếng Việt. Với mỗi
mục, bạn được cho: từ láy (`tu`), khía cạnh đang hỏi (`aspect`: "ý nghĩa" / "sắc thái biểu đạt"),
và giá trị ĐÚNG của khía cạnh đó (`correct`). Nhiệm vụ: sinh 3 giá trị SAI nhưng HỢP LÝ, PHONG
CÁCH/ĐỘ DÀI tương tự giá trị đúng, đủ khó để đòi hỏi biết chính xác dữ kiện mới phân biệt được
(không được vô lý/lạc đề dễ loại trừ), và 3 giá trị SAI phải khác nhau, khác giá trị đúng.

Input: JSON array các object {"id": str, "tu": str, "aspect": str, "correct": str}.
Output: CHỈ trả về JSON array cùng độ dài, mỗi phần tử {"id": <id đầu vào>,
"distractors": [str, str, str]} -- không giải thích, không markdown fence.
""",
    "van": """Bạn sinh PHƯƠNG ÁN NHIỄU cho câu hỏi trắc nghiệm về từ láy vần tiếng Việt (từ láy
mà 2 tiếng chỉ giống nhau ở phần vần, khác phụ âm đầu). Với mỗi mục, bạn được cho: từ láy vần
(`tu`), khía cạnh đang hỏi (`aspect`: "phần vần lặp lại giữa 2 tiếng" / "ý nghĩa" / "sắc thái
biểu đạt"), và giá trị ĐÚNG của khía cạnh đó (`correct`). Nhiệm vụ: sinh 3 giá trị SAI nhưng HỢP
LÝ, PHONG CÁCH/ĐỘ DÀI tương tự giá trị đúng (nếu aspect là phần vần thì chỉ trả về 1 vần tiếng
Việt hợp lệ khác, ví dụ "oay", "inh", KHÔNG kèm giải thích), đủ khó để đòi hỏi biết chính xác dữ
kiện mới phân biệt được, và 3 giá trị SAI phải khác nhau, khác giá trị đúng.

Input: JSON array các object {"id": str, "tu": str, "aspect": str, "correct": str}.
Output: CHỈ trả về JSON array cùng độ dài, mỗi phần tử {"id": <id đầu vào>,
"distractors": [str, str, str]} -- không giải thích, không markdown fence.
""",
    "chung": """Bạn sinh PHƯƠNG ÁN NHIỄU cho câu hỏi trắc nghiệm về từ láy tiếng Việt. Với mỗi
mục, bạn được cho: từ láy (`tu`), khía cạnh đang hỏi (`aspect`: "loại từ láy (cấu tạo ngữ âm)" /
"từ loại (ngữ pháp)" / "ý nghĩa" / "sắc thái biểu đạt"), và giá trị ĐÚNG của khía cạnh đó
(`correct`). Nhiệm vụ: sinh 3 giá trị SAI nhưng HỢP LÝ, PHONG CÁCH/ĐỘ DÀI tương tự giá trị đúng
(nếu aspect là "loại từ láy" thì chỉ trả các nhãn cấu tạo hợp lệ khác như "Láy toàn bộ", "Láy âm
đầu", "Láy vần"; nếu là "từ loại" thì chỉ trả từ loại ngữ pháp hợp lệ khác như "Danh từ", "Động
từ"), đủ khó để đòi hỏi biết chính xác dữ kiện mới phân biệt được, và 3 giá trị SAI phải khác
nhau, khác giá trị đúng.

Input: JSON array các object {"id": str, "tu": str, "aspect": str, "correct": str}.
Output: CHỈ trả về JSON array cùng độ dài, mỗi phần tử {"id": <id đầu vào>,
"distractors": [str, str, str]} -- không giải thích, không markdown fence.
""",
}


def _extract_van(phan_loai: str) -> str:
    m = _VAN_RE.search(phan_loai)
    return m.group(1) if m else phan_loai


def _aspect_value(variant: str, aspect: str, row_or_info: dict) -> str:
    """row_or_info: dict-like object với các key CHUẨN HÓA sẵn (xem _normalize_row/_normalize_tu_info)."""
    return row_or_info[aspect]


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


def _normalize_csv_row(variant: str, row: dict) -> dict:
    """Chuẩn hóa 1 dòng CSV về dict {aspect: value} theo đúng tên cột của từng variant, cộng
    "_group_key" dùng để ưu tiên nhiễu CÙNG LOẠI (rule-based)."""
    if variant == "toan_bo":
        return {
            "y_nghia": row["Ý nghĩa"], "sac_thai": row["Sắc thái biểu đạt"],
            "_group_key": row["Phân loại"],
        }
    if variant == "van":
        return {
            "van": _extract_van(row["Phân loại"]), "y_nghia": row["Ý nghĩa"], "sac_thai": row["Sắc thái biểu đạt"],
            "_group_key": None,  # van KHÔNG ưu tiên cùng loại -- toàn bộ đều là "Láy vần (...)" khác nhau
        }
    # chung
    return {
        "loai_tu_lay": row["Loại từ láy"], "tu_loai": row["Từ loại"],
        "y_nghia": row["Ý nghĩa"], "sac_thai": row["Sắc thái biểu đạt"],
        "_group_key": row["Loại từ láy"],
    }


def _normalize_tu_info(variant: str, tu_info: dict) -> dict:
    """Chuẩn hóa 1 entry "tu_lay_xuat_hien" của sample (field name khác CSV 1 chút)."""
    if variant == "toan_bo":
        return {"y_nghia": tu_info["y_nghia"], "sac_thai": tu_info["sac_thai_bieu_dat"]}
    if variant == "van":
        return {"y_nghia": tu_info["y_nghia"], "sac_thai": tu_info["sac_thai_bieu_dat"]}  # "van" lấy từ CSV, xem generate()
    return {
        "loai_tu_lay": tu_info["loai_tu_lay"], "tu_loai": tu_info["tu_loai"],
        "y_nghia": tu_info["y_nghia"], "sac_thai": tu_info["sac_thai_bieu_dat"],
    }


def _candidate_priority_order(all_words: list, correct_word: str, correct_group_key) -> list:
    if correct_group_key is not None:
        same_type = [(w, r) for w, r in all_words if w != correct_word and r["_group_key"] == correct_group_key]
        others = [(w, r) for w, r in all_words if w != correct_word and (w, r) not in same_type]
        random.shuffle(same_type)
        random.shuffle(others)
        return same_type + others
    candidates = [(w, r) for w, r in all_words if w != correct_word]
    random.shuffle(candidates)
    return candidates


def _pick_distractors_rule_based(all_words, aspect, correct_word, correct_group_key, correct_text, n=3):
    seen_texts = {correct_text}
    picked_texts = []
    for w, r in _candidate_priority_order(all_words, correct_word, correct_group_key):
        text = r[aspect]
        if text in seen_texts:
            continue
        seen_texts.add(text)
        picked_texts.append(text)
        if len(picked_texts) == n:
            break
    assert len(picked_texts) == n, (
        f"Không đủ {n} nhiễu KHÁC NHAU cho khía cạnh {aspect!r} của từ láy {correct_word!r}."
    )
    return picked_texts


def _random_strategy(weights: dict) -> str:
    strategies, w = zip(*weights.items())
    return random.choices(strategies, weights=w, k=1)[0]


def _llm_batch_call(client, model, variant, items, max_retries):
    from google.genai import types

    labels = VARIANT_ASPECT_LABELS[variant]
    payload = [
        {"id": cid, "tu": tu, "aspect": labels[aspect], "correct": correct_text}
        for cid, tu, aspect, correct_text in items
    ]
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=json.dumps(payload, ensure_ascii=False),
                config=types.GenerateContentConfig(
                    system_instruction=LLM_DISTRACTOR_SYSTEM_PROMPTS[variant],
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
    client, model, variant: str, samples: list[dict], csv_path: Path, output_path: Path, *,
    strategy_weights: dict = None, batch_size: int = DEFAULT_BATCH_SIZE,
    max_workers: int = DEFAULT_MAX_WORKERS, max_retries: int = DEFAULT_MAX_RETRIES, seed: int = 42,
) -> None:
    random.seed(seed)
    strategy_weights = strategy_weights or {"rule_based": 0.5, "llm": 0.5}
    aspects = VARIANT_ASPECTS[variant]
    id_prefix_base = VARIANT_ID_PREFIX[variant]
    id_prefix_re = re.compile(r"^(" + re.escape(id_prefix_base) + r"-\d+)")

    word_rows = {}
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            w = row[WORD_COL].strip()
            if w:
                word_rows[w] = _normalize_csv_row(variant, row)
    all_words = list(word_rows.items())

    existing_answers_by_sample: dict = {}
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                m = id_prefix_re.match(rec["id"])
                if m:
                    existing_answers_by_sample.setdefault(m.group(1), set()).add(rec["answer"])
        n_existing = sum(len(v) for v in existing_answers_by_sample.values())
        print(f"Đã có sẵn {n_existing} câu hỏi (thuộc {len(existing_answers_by_sample)} sample) "
              f"trong {output_path} -- sẽ bỏ qua đúng các (sample, đáp án) đã có.")

    prepared = []
    skipped_no_word = 0
    for i, s in enumerate(samples):
        found = s.get("tu_lay_xuat_hien") or []
        if not found:
            skipped_no_word += 1
            continue
        sample_id_prefix = f"{id_prefix_base}-{i + 1:04d}"
        already_asked = existing_answers_by_sample.get(sample_id_prefix, set())

        for word_idx, tu_info in enumerate(found):
            word = tu_info["tu"]
            if word not in word_rows:
                continue
            info = _normalize_tu_info(variant, tu_info)
            if variant == "van":
                info["van"] = word_rows[word]["van"]  # "van" chỉ có trong CSV (đã trích từ Phân loại), không có trong tu_info
            for aspect in aspects:
                correct_choice = info[aspect]
                if correct_choice in already_asked:
                    continue
                record_id = f"{sample_id_prefix}-w{word_idx}-{aspect}"
                prepared.append((record_id, s, word, aspect, correct_choice, _random_strategy(strategy_weights)))

    if skipped_no_word:
        print(f"Bỏ qua {skipped_no_word} sample thiếu tu_lay_xuat_hien.")
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
            futures = {executor.submit(_llm_batch_call, client, model, variant, b, max_retries): b for b in batches}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"tu_lay_{variant} LLM batches"):
                llm_distractors.update(future.result())

    n_written = n_failed_distractor = 0
    with open(output_path, "a", encoding="utf-8") as f:
        for record_id, s, word, aspect, correct_choice, strategy in tqdm(prepared, desc=f"tu_lay_{variant} build records"):
            distractor_choices = None
            if strategy == "llm":
                distractor_choices = llm_distractors.get(record_id)
            if distractor_choices is None:
                correct_group_key = word_rows[word]["_group_key"]
                try:
                    distractor_choices = _pick_distractors_rule_based(all_words, aspect, word, correct_group_key, correct_choice, n=3)
                except AssertionError:
                    n_failed_distractor += 1
                    continue

            choices = distractor_choices + [correct_choice]
            random.shuffle(choices)
            record = {
                "id": record_id,
                "audio_id": f"{id_prefix_base}/audio/{Path(_sample_audio_path(s)).name}",
                "question": random.choice(QUESTION_TEMPLATES[variant][aspect]),
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

    p = sub.add_parser("generate", help="Sinh câu hỏi trắc nghiệm về từ láy tiếng Việt.")
    p.add_argument("--variant", required=True, choices=["toan_bo", "van", "chung"])
    p.add_argument("--service-account-json", required=True)
    p.add_argument("--samples", required=True, help="asr_samples_with_tu_lay_<variant>.json (hoặc _tu_lay.json cho 'chung')")
    p.add_argument("--csv", required=True, help="tu_lay_<variant>_final.csv (hoặc tu_lay_tieng_viet_final.csv cho 'chung')")
    p.add_argument("--output", required=True, help="tu_lay_<variant>_qa.jsonl (append, resumable)")
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
        client, args.model, args.variant, samples, Path(args.csv), Path(args.output),
        strategy_weights={"rule_based": args.rule_based_weight, "llm": args.llm_weight},
        batch_size=args.batch_size, max_workers=args.max_workers, max_retries=args.max_retries, seed=args.seed,
    )


if __name__ == "__main__":
    main()
