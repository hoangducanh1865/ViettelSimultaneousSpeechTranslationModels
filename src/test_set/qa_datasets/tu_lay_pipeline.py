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


def _load_word_rows(variant: str, csv_path: Path) -> dict:
    word_rows = {}
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            w = row[WORD_COL].strip()
            if w:
                word_rows[w] = _normalize_csv_row(variant, row)
    return word_rows


# ============================================================================================
# Multi-hop redesign: classify-levels + generate-questions
#
# Câu hỏi CŨ (subcommand "generate" ở trên) hỏi HẾT các khía cạnh có sẵn cho MỖI từ láy tìm
# được -- toàn bộ đều là lookup 1 bước (nghe -> nhận diện từ -> tra field). Kiến trúc multi-hop
# dưới đây (giống han_viet_pipeline.py/tu_muon_pipeline.py) khai thác sâu hơn với 3 mức:
#   - level 1 (1-hop): tra ĐÚNG 1 khía cạnh có sẵn (ngẫu nhiên trong các khía cạnh của variant),
#     thay vì hỏi hết -- tránh lặp quá nhiều câu quá dễ cho cùng 1 từ.
#   - level 2 (2-hop): nghe -> nhận diện từ láy -> suy ra NGỮ CẢNH SỬ DỤNG/PHONG CÁCH của từ đó
#     (ví dụ: văn nói hàng ngày, văn viết trang trọng, văn học miêu tả...) dựa trên "sắc thái
#     biểu đạt" đã có sẵn trong CSV kiểm duyệt (dữ kiện THẬT, không LLM bịa) kết hợp CHÍNH câu
#     transcript chứa từ đó.
#   - level 3 (3-hop): nghe -> nhận diện từ láy -> liên hệ 1 câu ca dao/tục ngữ/thành ngữ/tác
#     phẩm văn học Việt Nam THẬT, NỔI TIẾNG có sử dụng chính từ láy đó theo cách tiêu biểu. Gemini
#     tự quyết định có đủ chắc chắn để gán mức này không (giống han_viet_pipeline.py -- không
#     dùng whitelist thủ công).
# ============================================================================================

LEVEL_SYSTEM_PROMPT = """Bạn phân loại độ khó reasoning cho câu hỏi về 1 từ LÁY tiếng Việt xuất
hiện trong 1 đoạn audio. Với mỗi từ, bạn được cho: từ láy (`tu`), ý nghĩa (`y_nghia`), sắc thái
biểu đạt theo CSV kiểm duyệt (`sac_thai`), và transcript THẬT chứa từ đó.

Bạn quyết định "level_3_eligible": CHỈ true khi bạn HOÀN TOÀN CHẮC CHẮN có 1 câu ca dao/tục
ngữ/thành ngữ/tác phẩm văn học Việt Nam THẬT, NỔI TIẾNG, PHỔ BIẾN có sử dụng CHÍNH từ láy này
theo cách tiêu biểu, gắn được với ý nghĩa/sắc thái của từ. Nếu không chắc chắn 100% hoặc chỉ là
suy đoán/liên hệ gượng ép, PHẢI để false. Nếu true, bạn phải tự cung cấp "cultural_fact" (trích
dẫn/nêu tên câu ca dao-tục ngữ-thành ngữ-tác phẩm đó, ngắn gọn, CHÍNH XÁC).

Input: JSON array các object {"id": str, "tu": str, "y_nghia": str, "sac_thai": str,
"transcript": str}.
Output: CHỈ trả về JSON array cùng độ dài, mỗi phần tử {"id": <id đầu vào>,
"level_3_eligible": true|false, "cultural_fact": "<trích dẫn/tên tác phẩm THẬT nếu true, else null>",
"reason": "<lý do ngắn gọn>"} -- không giải thích thêm, không markdown fence.
"""

LEVEL2_QUESTION_TEMPLATES = [
    "Trong đoạn âm thanh trên có xuất hiện một từ láy tiếng Việt. Từ đó thường được dùng trong "
    "ngữ cảnh/phong cách nói nào?",
    "Có một từ láy tiếng Việt xuất hiện trong câu nói trên. Xét theo sắc thái, từ đó thường "
    "được dùng khi nào?",
    "Đoạn âm thanh trên chứa một từ láy. Từ này phù hợp với ngữ cảnh sử dụng nào nhất?",
    "Trong câu vừa nghe có một từ láy đặc trưng của tiếng Việt. Từ đó thường xuất hiện trong "
    "phong cách nói/viết nào?",
]

LEVEL3_SYSTEM_PROMPT = """Bạn sinh câu hỏi trắc nghiệm 4 lựa chọn liên hệ 1 từ láy tiếng Việt
xuất hiện trong 1 đoạn audio với 1 câu ca dao/tục ngữ/thành ngữ/tác phẩm văn học THẬT. Với mỗi
mục, bạn được cho: từ láy (`tu`), transcript chứa từ đó, và `cultural_fact` (trích dẫn/tên tác
phẩm THẬT đã được xác định trước -- đáp án đúng PHẢI dựa ĐÚNG vào cultural_fact này, KHÔNG được
tự đổi/thêm dẫn chứng khác).

Nhiệm vụ: viết 1 câu hỏi (KHÔNG nhắc thẳng tên từ láy -- người nghe phải tự nhận ra qua audio)
hỏi câu ca dao/tục ngữ/thành ngữ/tác phẩm nào có sử dụng từ láy đó, đáp án đúng diễn đạt lại
cultural_fact (không đổi nội dung), và 3 phương án nhiễu là các câu ca dao/tục ngữ/thành ngữ/tác
phẩm THẬT KHÁC (không bịa, không vô lý/lạc đề dễ loại trừ), khác nhau, khác đáp án đúng.

Input: JSON array các object {"id": str, "tu": str, "transcript": str, "cultural_fact": str}.
Output: CHỈ trả về JSON array cùng độ dài, mỗi phần tử {"id": <id đầu vào>, "question": str,
"choices": [str, str, str, str], "answer": str} (answer PHẢI là 1 trong 4 choices) -- không giải
thích thêm, không markdown fence.
"""


def _classify_level_batch_tu_lay(client, model, batch, max_retries):
    from google.genai import types

    payload = [
        {"id": e["id"], "tu": e["tu"], "y_nghia": e["y_nghia"], "sac_thai": e["sac_thai"], "transcript": e["transcript"]}
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
    return {e["id"]: {"level_3_eligible": False, "cultural_fact": None, "reason": "Lỗi gọi API."} for e in batch}


def _build_word_entries(variant: str, samples: list[dict], word_rows: dict) -> list[dict]:
    """Trả về list entry {id, tu, y_nghia, sac_thai, transcript, sample, _row} -- mỗi entry là 1
    (sample, từ láy) cần phân loại mức độ / sinh câu hỏi. "_row" là dict khía cạnh chuẩn hóa đầy
    đủ của variant (dùng cho level 1: chọn ngẫu nhiên 1 khía cạnh trong ASPECTS của variant)."""
    id_prefix_base = VARIANT_ID_PREFIX[variant]
    entries = []
    for i, s in enumerate(samples):
        found = s.get("tu_lay_xuat_hien") or []
        sample_id_prefix = f"{id_prefix_base}-{i + 1:04d}"
        for word_idx, tu_info in enumerate(found):
            word = tu_info["tu"]
            if word not in word_rows:
                continue
            row = dict(word_rows[word])
            if variant == "van":
                row["van"] = word_rows[word]["van"]
            entries.append({
                "id": f"{sample_id_prefix}-w{word_idx}",
                "tu": word,
                "y_nghia": row["y_nghia"],
                "sac_thai": row["sac_thai"],
                "transcript": s.get("transcript") or s.get("text") or "",
                "sample": s,
                "_row": row,
            })
    return entries


def classify_levels(
    client, model, variant: str, samples: list[dict], csv_path: Path, *,
    batch_size: int = DEFAULT_BATCH_SIZE, max_workers: int = DEFAULT_MAX_WORKERS,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> list[dict]:
    word_rows = _load_word_rows(variant, csv_path)
    entries = _build_word_entries(variant, samples, word_rows)
    print(f"Cần phân loại mức độ cho {len(entries)} (sample, từ láy).")

    batches = [entries[i:i + batch_size] for i in range(0, len(entries), batch_size)]
    level_results: dict = {}
    if batches:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_classify_level_batch_tu_lay, client, model, b, max_retries): b for b in batches}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"tu_lay_{variant} phân loại mức độ"):
                level_results.update(future.result())

    for e in entries:
        res = level_results.get(e["id"], {"level_3_eligible": False, "cultural_fact": None, "reason": "Không có kết quả."})
        if res.get("level_3_eligible") and res.get("cultural_fact"):
            e["max_level"] = 3
            e["cultural_fact"] = res["cultural_fact"]
        else:
            e["max_level"] = 2  # sắc thái luôn có sẵn trong CSV -> level 2 luôn khả dụng
            e["cultural_fact"] = None
        e["level_reason"] = res.get("reason")

    return entries


def _gen_level3_batch(client, model, batch, max_retries):
    # batch: list[(id_ghep, entry)]
    from google.genai import types

    payload = [
        {"id": id_ghep, "tu": e["tu"], "transcript": e["transcript"], "cultural_fact": e["cultural_fact"]}
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
                    system_instruction=LEVEL3_SYSTEM_PROMPT, temperature=0.7, max_output_tokens=8192,
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


def _record_common_fields(variant: str, e: dict, id_ghep: str, level: int) -> dict:
    s = e["sample"]
    id_prefix_base = VARIANT_ID_PREFIX[variant]
    return {
        "id": id_ghep,
        "source_id": e["id"],
        "audio_id": f"{id_prefix_base}/audio/{Path(_sample_audio_path(s)).name}",
        "dataset": _dataset_from_audio_filepath(_sample_audio_path(s)),
        "task": "speech",
        "split": "test",
        "category": "Reasoning",
        "sub-category": "Hiện tượng đặc biệt trong tiếng Việt",
        "difficulty": {1: "easy", 2: "medium", 3: "hard"}[level],
        "level": level,
        "max_level": e["max_level"],
    }


def generate_questions(
    client, model, variant: str, entries: list[dict], csv_path: Path, output_path: Path, *,
    batch_size: int = DEFAULT_BATCH_SIZE, max_workers: int = DEFAULT_MAX_WORKERS,
    max_retries: int = DEFAULT_MAX_RETRIES, seed: int = 42,
) -> None:
    random.seed(seed)
    word_rows = _load_word_rows(variant, csv_path)
    all_words = list(word_rows.items())
    aspects = VARIANT_ASPECTS[variant]
    level1_aspects = [a for a in aspects if a not in ("y_nghia", "sac_thai")] or ["y_nghia"]
    # level 1 ưu tiên khía cạnh ĐẶC TRƯNG của variant (loại từ láy/từ loại/phần vần) nếu có, vì
    # đó mới thực sự đòi hỏi phân tích cấu tạo; nếu variant không có khía cạnh đặc trưng nào
    # (không xảy ra với 3 variant hiện tại) thì mới rơi về "y_nghia".

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
                aspect = random.choice(level1_aspects)
                correct_choice = e["_row"][aspect]
                correct_group_key = e["_row"].get("_group_key")
                try:
                    distractors = _pick_distractors_rule_based(
                        all_words, aspect, e["tu"], correct_group_key, correct_choice, n=3,
                    )
                except AssertionError:
                    n_failed += 1
                    continue
                choices = distractors + [correct_choice]
                random.shuffle(choices)
                record = {
                    **_record_common_fields(variant, e, id_ghep, 1),
                    "question": random.choice(QUESTION_TEMPLATES[variant][aspect]),
                    "choices": choices,
                    "answer": correct_choice,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_written += 1

            elif level == 2:
                correct_choice = e["sac_thai"]
                other_values = sorted({r["sac_thai"] for _, r in all_words if r["sac_thai"] != correct_choice})
                if len(other_values) < 3:
                    n_failed += 1
                    continue
                distractors = random.sample(other_values, 3)
                choices = distractors + [correct_choice]
                random.shuffle(choices)
                record = {
                    **_record_common_fields(variant, e, id_ghep, 2),
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
            futures = {executor.submit(_gen_level3_batch, client, model, b, max_retries): b for b in batches}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"tu_lay_{variant} sinh câu hỏi level 3"):
                level3_results.update(future.result())

        for id_ghep, e in level3_pending:
            gen = level3_results.get(id_ghep)
            if gen is None:
                n_failed += 1
                continue
            record = {
                **_record_common_fields(variant, e, id_ghep, 3),
                "question": gen["question"],
                "choices": gen["choices"],
                "answer": gen["answer"],
                "cultural_fact": e["cultural_fact"],
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
    print(f"Số (sample, từ láy) có mặt: {len({r.get('source_id', r['id']) for r in all_out})}")


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

    p = sub.add_parser("generate", help="[CŨ] Sinh câu hỏi flat 1-hop về từ láy tiếng Việt (dùng classify-levels/generate-questions thay thế).")
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

    p1 = sub.add_parser("classify-levels", help="Phân loại độ khó 1/2/3-hop cho mỗi (sample, từ láy).")
    p1.add_argument("--variant", required=True, choices=["toan_bo", "van", "chung"])
    p1.add_argument("--service-account-json", required=True)
    p1.add_argument("--samples", required=True, help="asr_samples_with_tu_lay_<variant>.json")
    p1.add_argument("--csv", required=True, help="tu_lay_<variant>_final.csv")
    p1.add_argument("--output", required=True, help="tu_lay_<variant>_difficulty_levels.jsonl")
    p1.add_argument("--model", default=DEFAULT_MODEL)
    p1.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p1.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    p1.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)

    p2 = sub.add_parser("generate-questions", help="Sinh N câu hỏi multi-hop (N=max_level) cho mỗi (sample, từ láy).")
    p2.add_argument("--variant", required=True, choices=["toan_bo", "van", "chung"])
    p2.add_argument("--service-account-json", required=True)
    p2.add_argument("--input", required=True, help="tu_lay_<variant>_difficulty_levels.jsonl")
    p2.add_argument("--csv", required=True, help="tu_lay_<variant>_final.csv")
    p2.add_argument("--output", required=True, help="tu_lay_<variant>_multihop_qa.jsonl")
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
            client, args.model, args.variant, samples, Path(args.csv), Path(args.output),
            strategy_weights={"rule_based": args.rule_based_weight, "llm": args.llm_weight},
            batch_size=args.batch_size, max_workers=args.max_workers, max_retries=args.max_retries, seed=args.seed,
        )

    elif args.command == "classify-levels":
        client = load_gemini_client(args.service_account_json)
        with open(args.samples, encoding="utf-8") as f:
            samples = json.load(f)
        print(f"Đã đọc {len(samples)} sample từ {args.samples}")
        entries = classify_levels(
            client, args.model, args.variant, samples, Path(args.csv),
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
        generate_questions(
            client, args.model, args.variant, entries, Path(args.csv), Path(args.output),
            batch_size=args.batch_size, max_workers=args.max_workers, max_retries=args.max_retries, seed=args.seed,
        )


if __name__ == "__main__":
    main()
