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

from knowledge_graph import load_knowledge_graph, rule_addendum_text, words_index
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


def _load_word_rows(csv_path: Path) -> dict:
    final_word_rows = {}
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            w = row[CANDIDATE_COL].strip()
            if w:
                final_word_rows[w] = row
    return final_word_rows


# ============================================================================================
# Multi-hop redesign: classify-levels + generate-questions
#
# Câu hỏi CŨ (subcommand "generate" ở trên) chỉ hỏi 1 trong 3 khía cạnh CÓ SẴN trong CSV cho MỖI
# từ mượn tìm được -- toàn bộ đều là lookup 1 bước (nghe -> nhận diện từ -> tra 1 field). Kiến
# trúc multi-hop dưới đây khai thác SÂU HƠN, giống han_viet_pipeline.py, với 3 mức tăng dần:
#   - level 1 (1-hop): giữ nguyên bản chất "tra 1 khía cạnh" cũ (ngôn ngữ/từ gốc/ý nghĩa), nhưng
#     chỉ hỏi ĐÚNG 1 khía cạnh/từ (ngẫu nhiên) thay vì hỏi hết -- tránh lặp lại quá nhiều câu quá
#     dễ cho cùng 1 từ.
#   - level 2 (2-hop): nghe -> nhận diện từ mượn -> suy ra LĨNH VỰC SỬ DỤNG (cột "Nhóm Lĩnh Vực"
#     đã có sẵn trong CSV kiểm duyệt -- vẫn là dữ kiện THẬT, không phải LLM bịa). Chỉ khả dụng
#     nếu CSV có giá trị lĩnh vực cho từ đó.
#   - level 3 (3-hop): nghe -> nhận diện từ mượn -> liên hệ 1 sự kiện/bối cảnh lịch sử-văn hóa
#     THẬT về việc từ đó du nhập vào tiếng Việt (thời Pháp thuộc, giao thương người Hoa, ảnh
#     hưởng tiếng Anh hiện đại...). Gemini tự quyết định có đủ chắc chắn để gán mức này không
#     (giống han_viet_pipeline.py -- không dùng whitelist thủ công).
# ============================================================================================

LEVEL_SYSTEM_PROMPT = """Bạn phân loại độ khó reasoning cho câu hỏi về 1 từ MƯỢN tiếng Việt (đã
Việt hóa) xuất hiện trong 1 đoạn audio. Với mỗi từ, bạn được cho: từ mượn (`tu`), ngôn ngữ nguồn
gốc (`ngon_ngu`), từ gốc (`tu_goc`), ý nghĩa (`y_nghia`), lĩnh vực sử dụng theo CSV kiểm duyệt
(`linh_vuc`, có thể rỗng), và transcript THẬT chứa từ đó.

Bạn quyết định "level_3_eligible": CHỈ true khi bạn HOÀN TOÀN CHẮC CHẮN về 1 sự kiện/bối cảnh
lịch sử-văn hóa THẬT, NỔI TIẾNG, PHỔ BIẾN gắn với việc từ này du nhập vào tiếng Việt (ví dụ: thời
kỳ Pháp thuộc, giao thương với người Hoa, ảnh hưởng tiếng Anh thời hiện đại, du nhập qua thể
thao/công nghệ/ẩm thực...). Nếu không chắc chắn 100% hoặc chỉ là suy đoán chung, PHẢI để false.
Nếu true, bạn phải tự cung cấp "historical_fact" (mô tả sự kiện/bối cảnh đó, ngắn gọn, CHÍNH XÁC).

Input: JSON array các object {"id": str, "tu": str, "ngon_ngu": str, "tu_goc": str,
"y_nghia": str, "linh_vuc": str, "transcript": str}.
Output: CHỈ trả về JSON array cùng độ dài, mỗi phần tử {"id": <id đầu vào>,
"level_3_eligible": true|false, "historical_fact": "<sự kiện THẬT nếu true, else null>",
"reason": "<lý do ngắn gọn>"} -- không giải thích thêm, không markdown fence.
"""

LEVEL2_QUESTION_TEMPLATES = [
    "Trong đoạn âm thanh trên có xuất hiện một từ mượn tiếng Việt. Từ đó thường được dùng "
    "trong lĩnh vực nào?",
    "Có một từ mượn tiếng Việt xuất hiện trong câu nói trên. Xét theo cách sử dụng, từ đó "
    "thuộc lĩnh vực/phạm trù nào?",
    "Đoạn âm thanh trên chứa một từ vay mượn từ tiếng nước ngoài. Từ này gắn với lĩnh vực nào "
    "trong đời sống?",
    "Trong câu vừa nghe có một từ tiếng Việt vốn vay mượn từ nước ngoài. Từ đó thường xuất "
    "hiện trong lĩnh vực nào?",
]

LEVEL3_SYSTEM_PROMPT = """Bạn sinh câu hỏi trắc nghiệm 4 lựa chọn về BỐI CẢNH LỊCH SỬ-VĂN HÓA du
nhập của 1 từ mượn tiếng Việt (đã Việt hóa) xuất hiện trong 1 đoạn audio. Với mỗi mục, bạn được
cho: từ mượn (`tu`), transcript chứa từ đó, và `historical_fact` (sự kiện/bối cảnh THẬT đã được
xác định trước -- đáp án đúng PHẢI dựa ĐÚNG vào historical_fact này, KHÔNG được tự đổi/thêm sự
kiện khác).

Nhiệm vụ: viết 1 câu hỏi (KHÔNG nhắc thẳng tên từ mượn -- người nghe phải tự nhận ra qua audio)
hỏi về bối cảnh/sự kiện lịch sử-văn hóa đó, đáp án đúng diễn đạt lại historical_fact (không đổi
nghĩa), và 3 phương án nhiễu là các bối cảnh/sự kiện lịch sử-văn hóa THẬT KHÁC (không bịa sự
kiện giả, không vô lý/lạc đề dễ loại trừ), khác nhau, khác đáp án đúng.

Input: JSON array các object {"id": str, "tu": str, "transcript": str, "historical_fact": str}.
Output: CHỈ trả về JSON array cùng độ dài, mỗi phần tử {"id": <id đầu vào>, "question": str,
"choices": [str, str, str, str], "answer": str} (answer PHẢI là 1 trong 4 choices) -- không giải
thích thêm, không markdown fence.
"""


def _classify_level_batch_tu_muon(client, model, batch, max_retries):
    from google.genai import types

    payload = [
        {"id": e["id"], "tu": e["tu"], "ngon_ngu": e["ngon_ngu"], "tu_goc": e["tu_goc"],
         "y_nghia": e["y_nghia"], "linh_vuc": e["linh_vuc"], "transcript": e["transcript"]}
        for e in batch
    ]
    ids_sent = {e["id"] for e in batch}
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=json.dumps(payload, ensure_ascii=False),
                config=types.GenerateContentConfig(
                    system_instruction=LEVEL_SYSTEM_PROMPT, temperature=0.0, max_output_tokens=4096,
                ),
            )
            raw = (response.text or "").strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            results = json.loads(raw)
            out = {}
            for item in results:
                if item.get("id") in ids_sent and "level_3_eligible" in item:
                    out[item["id"]] = item
            missing = ids_sent - set(out)
            if missing:
                raise ValueError(f"Thiếu {len(missing)}/{len(ids_sent)} id trong response")
            return out
        except Exception as e:  # noqa: BLE001
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    tqdm.write(f"[LỖI phân loại level_3] batch {len(batch)} từ: {last_error!r} -- mặc định level_3_eligible=False.")
    return {e["id"]: {"level_3_eligible": False, "historical_fact": None, "reason": "Lỗi gọi API."} for e in batch}


def _build_word_entries(samples: list[dict], final_word_rows: dict) -> list[dict]:
    """Trả về list entry {id, tu, ngon_ngu, tu_goc, y_nghia, linh_vuc, transcript, sample} --
    mỗi entry là 1 (sample, từ mượn) cần phân loại mức độ / sinh câu hỏi."""
    entries = []
    for i, s in enumerate(samples):
        found = s.get("tu_muon_xuat_hien") or []
        sample_id_prefix = f"tu-muon-asr-{i + 1:04d}"
        for word_idx, tu_info in enumerate(found):
            word = tu_info["tu"]
            if word not in final_word_rows:
                continue
            entries.append({
                "id": f"{sample_id_prefix}-w{word_idx}",
                "tu": word,
                "ngon_ngu": tu_info["ngon_ngu_nguon_goc"],
                "tu_goc": tu_info["tu_goc"],
                "y_nghia": tu_info["y_nghia_ghi_chu"],
                "linh_vuc": (final_word_rows[word].get("Nhóm Lĩnh Vực") or "").strip(),
                "transcript": s.get("transcript") or s.get("text") or "",
                "sample": s,
            })
    return entries


def classify_levels(
    client, model, samples: list[dict], csv_path: Path, *,
    knowledge_graph: Optional[dict] = None,
    batch_size: int = DEFAULT_BATCH_SIZE, max_workers: int = DEFAULT_MAX_WORKERS,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> list[dict]:
    """knowledge_graph=None -> hành vi CŨ y hệt (an toàn ngược). Khi có, từ mượn đã được 3 model
    debate xác thực (words_index) dùng TRỰC TIẾP fields.historical_fact/nhom_linh_vuc đã debate
    -- KHÔNG gọi Gemini tự quyết lại; từ chưa có trong kg vẫn rơi về đường Gemini-tự-quyết cũ."""
    final_word_rows = _load_word_rows(csv_path)
    entries = _build_word_entries(samples, final_word_rows)
    kg_words = words_index(knowledge_graph) if knowledge_graph else {}
    n_from_kg = sum(1 for e in entries if e["tu"] in kg_words)
    if kg_words:
        print(f"{n_from_kg}/{len(entries)} (sample, từ mượn) dùng trực tiếp dữ kiện đã debate xác thực từ knowledge graph.")

    pending = [e for e in entries if e["tu"] not in kg_words]
    print(f"Cần Gemini phân loại mức độ cho {len(pending)} (sample, từ mượn).")

    batches = [pending[i:i + batch_size] for i in range(0, len(pending), batch_size)]
    level_results: dict = {}
    if batches:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_classify_level_batch_tu_muon, client, model, b, max_retries): b for b in batches}
            for future in tqdm(as_completed(futures), total=len(futures), desc="tu_muon phân loại mức độ"):
                level_results.update(future.result())

    for e in entries:
        if e["tu"] in kg_words:
            entry = kg_words[e["tu"]]
            fields = entry.get("fields", {})
            fact = fields.get("historical_fact")
            linh_vuc = fields.get("nhom_linh_vuc") or e["linh_vuc"]
            e["max_level"] = 3 if fact else (2 if linh_vuc else 1)
            e["historical_fact"] = fact
            e["level_reason"] = f"Từ knowledge graph (source={entry.get('source')})."
            continue
        res = level_results.get(e["id"], {"level_3_eligible": False, "historical_fact": None, "reason": "Không có kết quả."})
        if res.get("level_3_eligible") and res.get("historical_fact"):
            e["max_level"] = 3
            e["historical_fact"] = res["historical_fact"]
        elif e["linh_vuc"]:
            e["max_level"] = 2
            e["historical_fact"] = None
        else:
            e["max_level"] = 1
            e["historical_fact"] = None
        e["level_reason"] = res.get("reason")

    return entries


def _gen_level3_batch(client, model, batch, max_retries, system_prompt):
    # batch: list[(id_ghep, entry)]
    from google.genai import types

    payload = [
        {"id": id_ghep, "tu": e["tu"], "transcript": e["transcript"], "historical_fact": e["historical_fact"]}
        for id_ghep, e in batch
    ]
    ids_sent = {id_ghep for id_ghep, e in batch}
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=json.dumps(payload, ensure_ascii=False),
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt, temperature=0.7, max_output_tokens=8192,
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
    tqdm.write(f"[LỖI sinh câu hỏi level 3] batch {len(batch)}: {last_error!r} -- các mục này bị bỏ qua.")
    return {}


def _record_common_fields(e: dict, id_ghep: str, level: int) -> dict:
    s = e["sample"]
    return {
        "id": id_ghep,
        "source_id": e["id"],
        "audio_id": f"tu-muon-asr/audio/{Path(_sample_audio_path(s)).name}",
        "dataset": _dataset_from_audio_filepath(_sample_audio_path(s)),
        "task": "speech",
        "split": "test",
        "category": "Reasoning",
        "sub-category": "Hiện tượng đặc biệt trong tiếng Việt",
        "difficulty": {1: "easy", 2: "medium", 3: "hard"}[level],
        "level": level,
        "max_level": e["max_level"],
        "question_type": {1: "1-hop", 2: "2-hop", 3: "3-hop"}[level],
    }


def generate_questions(
    client, model, entries: list[dict], csv_path: Path, output_path: Path, *,
    knowledge_graph: Optional[dict] = None,
    batch_size: int = DEFAULT_BATCH_SIZE, max_workers: int = DEFAULT_MAX_WORKERS,
    max_retries: int = DEFAULT_MAX_RETRIES, seed: int = 42,
) -> None:
    """knowledge_graph=None -> hành vi CŨ y hệt; khi có, nối thêm luật debate đã tinh chỉnh vào
    system prompt sinh câu hỏi level 3 (level 1/2 không gọi Gemini nên không có prompt để sửa)."""
    random.seed(seed)
    level3_system_prompt = LEVEL3_SYSTEM_PROMPT + rule_addendum_text(knowledge_graph, "level_3")
    final_word_rows = _load_word_rows(csv_path)
    all_loanwords = list(final_word_rows.items())
    all_linh_vuc = sorted({(r.get("Nhóm Lĩnh Vực") or "").strip() for _, r in all_loanwords} - {""})

    done_ghep_ids = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done_ghep_ids.add(json.loads(line)["id"])
        print(f"Đã có sẵn {len(done_ghep_ids)} câu hỏi trong {output_path} -- resume.")

    level3_pending = []
    n_written = n_failed = 0
    out_f = open(output_path, "a", encoding="utf-8")

    for e in entries:
        for level in range(1, e["max_level"] + 1):
            id_ghep = f"{e['id']}__L{level}"
            if id_ghep in done_ghep_ids:
                continue

            if level == 1:
                available_aspects = ASPECTS if _is_unambiguous_language(e["ngon_ngu"]) else \
                    [a for a in ASPECTS if a != "ngon_ngu"]
                aspect = random.choice(available_aspects)
                correct_choice = _aspect_value(aspect, e["ngon_ngu"], e["tu_goc"], e["y_nghia"])
                try:
                    distractors = _pick_distractors_rule_based(
                        all_loanwords, aspect, e["tu"], final_word_rows[e["tu"]], correct_choice, n=3,
                    )
                except AssertionError:
                    n_failed += 1
                    continue
                choices = distractors + [correct_choice]
                random.shuffle(choices)
                record = {
                    **_record_common_fields(e, id_ghep, 1),
                    "question": random.choice(ASPECT_QUESTION_TEMPLATES[aspect]),
                    "choices": choices,
                    "answer": correct_choice,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_written += 1

            elif level == 2:
                correct_choice = e["linh_vuc"]
                other_values = [v for v in all_linh_vuc if v != correct_choice]
                if len(other_values) < 3:
                    n_failed += 1
                    continue
                distractors = random.sample(other_values, 3)
                choices = distractors + [correct_choice]
                random.shuffle(choices)
                record = {
                    **_record_common_fields(e, id_ghep, 2),
                    "question": random.choice(LEVEL2_QUESTION_TEMPLATES),
                    "choices": choices,
                    "answer": correct_choice,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_written += 1

            else:  # level == 3
                level3_pending.append((id_ghep, e))

    if level3_pending:
        batches = [level3_pending[i:i + batch_size] for i in range(0, len(level3_pending), batch_size)]
        level3_results: dict = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_gen_level3_batch, client, model, b, max_retries, level3_system_prompt): b for b in batches}
            for future in tqdm(as_completed(futures), total=len(futures), desc="tu_muon sinh câu hỏi level 3"):
                level3_results.update(future.result())

        for id_ghep, e in level3_pending:
            gen = level3_results.get(id_ghep)
            if gen is None:
                n_failed += 1
                continue
            record = {
                **_record_common_fields(e, id_ghep, 3),
                "question": gen["question"],
                "choices": gen["choices"],
                "answer": gen["answer"],
                "historical_fact": e["historical_fact"],
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    out_f.close()
    print(f"\nĐã ghi thêm {n_written} câu hỏi multi-hop vào {output_path} "
          f"({n_failed} bị bỏ qua do thiếu nhiễu/lỗi API).")

    from collections import Counter

    with open(output_path, encoding="utf-8") as f:
        all_out = [json.loads(line) for line in f if line.strip()]
    print(f"Phân bố level trong file kết quả: {dict(Counter(r.get('level') for r in all_out))}")
    print(f"Số (sample, từ mượn) có mặt: {len({r.get('source_id', r['id']) for r in all_out})}")


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

    p = sub.add_parser("generate", help="[CŨ] Sinh câu hỏi flat 1-hop về từ mượn tiếng Việt (dùng classify-levels/generate-questions thay thế).")
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

    p1 = sub.add_parser("classify-levels", help="Phân loại độ khó 1/2/3-hop cho mỗi (sample, từ mượn).")
    p1.add_argument("--service-account-json", required=True)
    p1.add_argument("--samples", required=True, help="asr_samples_with_tu_muon.json")
    p1.add_argument("--csv", required=True, help="tu_muon_tieng_viet_viet_hoa_final.csv")
    p1.add_argument("--output", required=True, help="tu_muon_difficulty_levels.jsonl")
    p1.add_argument("--knowledge-json", default=None, help="Knowledge graph đã debate (manual_model_relay.py) -- optional.")
    p1.add_argument("--model", default=DEFAULT_MODEL)
    p1.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p1.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    p1.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)

    p2 = sub.add_parser("generate-questions", help="Sinh N câu hỏi multi-hop (N=max_level) cho mỗi (sample, từ mượn).")
    p2.add_argument("--service-account-json", required=True)
    p2.add_argument("--input", required=True, help="tu_muon_difficulty_levels.jsonl")
    p2.add_argument("--csv", required=True, help="tu_muon_tieng_viet_viet_hoa_final.csv")
    p2.add_argument("--output", required=True, help="tu_muon_multihop_qa.jsonl")
    p2.add_argument("--knowledge-json", default=None)
    p2.add_argument("--model", default=DEFAULT_MODEL)
    p2.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p2.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    p2.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    p2.add_argument("--seed", type=int, default=42)

    args = parser.parse_args(argv)

    if args.command == "generate":
        client = load_gemini_client(args.service_account_json)
        with open(args.samples, encoding="utf-8") as f:
            samples = json.load(f)
        print(f"Đã đọc {len(samples)} sample từ {args.samples}")
        generate(
            client, args.model, samples, Path(args.csv), Path(args.output),
            strategy_weights={"rule_based": args.rule_based_weight, "llm": args.llm_weight},
            batch_size=args.batch_size, max_workers=args.max_workers, max_retries=args.max_retries, seed=args.seed,
        )

    elif args.command == "classify-levels":
        client = load_gemini_client(args.service_account_json)
        with open(args.samples, encoding="utf-8") as f:
            samples = json.load(f)
        print(f"Đã đọc {len(samples)} sample từ {args.samples}")
        knowledge_graph = load_knowledge_graph(Path(args.knowledge_json)) if args.knowledge_json else None
        entries = classify_levels(
            client, args.model, samples, Path(args.csv), knowledge_graph=knowledge_graph,
            batch_size=args.batch_size, max_workers=args.max_workers, max_retries=args.max_retries,
        )
        with open(args.output, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        from collections import Counter

        print(f"\nPhân bố mức độ: {dict(Counter(e['max_level'] for e in entries))}")
        print(f"Đã lưu {args.output}")

    elif args.command == "generate-questions":
        client = load_gemini_client(args.service_account_json)
        with open(args.input, encoding="utf-8") as f:
            entries = [json.loads(line) for line in f if line.strip()]
        print(f"Đã đọc {len(entries)} entry đã phân loại mức độ.")
        knowledge_graph = load_knowledge_graph(Path(args.knowledge_json)) if args.knowledge_json else None
        generate_questions(
            client, args.model, entries, Path(args.csv), Path(args.output), knowledge_graph=knowledge_graph,
            batch_size=args.batch_size, max_workers=args.max_workers, max_retries=args.max_retries, seed=args.seed,
        )


if __name__ == "__main__":
    main()
