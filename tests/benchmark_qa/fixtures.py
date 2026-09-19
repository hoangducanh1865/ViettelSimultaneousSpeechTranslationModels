"""Dựng dữ liệu nhỏ (2-3 mẫu) TRONG <repo>/data/_e2e/<test>/ cho từng task benchmark_qa."""

from __future__ import annotations

import csv
import json
from pathlib import Path

# Dataset audio path phải chứa segment "16k" để _dataset_from_audio_filepath() suy ra dataset.
DATASET = "tiny_ds"


def _audio(name: str) -> str:
    return f"release_hf/16k/{DATASET}/{name}"


def speech_dir(root: Path) -> Path:
    d = root / "benchmark_qa/speech/cac_hien_tuong_dac_biet_trong_tieng_viet"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------------------------------
# Corpus dùng chung: speech_sources.jsonl (file_name/text/dataset...)
# ---------------------------------------------------------------------------------------------
CORPUS = {
    DATASET: [
        # Từ mượn + Hán Việt
        {"audio": _audio("a1.wav"), "transcript": "hôm nay tôi uống cà phê và ăn bánh mì ở ga tàu"},
        {"audio": _audio("a2.wav"), "transcript": "tôi giặt xà phòng và xem tivi buổi tối"},
        {"audio": _audio("a3.wav"), "transcript": "chúng ta cần bảo vệ tổ quốc và giữ gìn văn hóa"},
        # Từ láy: full words (để filter-csv giữ) + phương ngữ
        {
            "audio": _audio("a4.wav"),
            "transcript": "trời xanh xao và mát mẻ, tôi thấy tươi tắn sáng sủa, nhè nhẹ mạnh mẽ, "
                          "đẹp đẽ lạnh lẽo, cô ấy lúng túng và vội vàng, mô tê răng rứa hen",
        },
        # Từ láy: CHỈ từ gốc (để build-cloze-samples nhận)
        {"audio": _audio("a5.wav"), "transcript": "trời hôm nay xanh quá, tuyết rơi nhẹ, gió mạnh, hôm nay đẹp và lạnh"},
    ]
}


def write_corpus(root: Path) -> None:
    speech = root / "benchmark_qa/speech"
    speech.mkdir(parents=True, exist_ok=True)
    # speech_sources.jsonl: corpus DUY NHẤT (thay full_transcripts.json + release_hf...json).
    _write_jsonl(speech / "speech_sources.jsonl", [
        {
            "file_name": s["audio"], "text": s["transcript"], "dataset": DATASET,
            "source_id": f"tiny-{i:03d}", "split": "test",
            "original_file": s["audio"], "transcript_status": "authoritative",
        }
        for i, s in enumerate(CORPUS[DATASET])
    ])
    # test_speech.jsonl -- runner đọc tại <root>/Vietnamese-Speech-QA/test_speech.jsonl (khớp
    # TEST_SPEECH_JSONL trong _lib.sh), dùng cho han_viet fill-fields (id khớp source_id).
    _write_jsonl(root / "Vietnamese-Speech-QA" / "test_speech.jsonl", [
        {
            "id": "hv-001", "task": "speech", "split": "test", "category": "Reasoning",
            "sub-category": "Hiện tượng đặc biệt trong tiếng Việt",
            "sub-sub-category": "Sino-Vietnamese Lexical Semantics", "difficulty": "medium",
        },
        {
            "id": "hv-002", "task": "speech", "split": "test", "category": "Reasoning",
            "sub-category": "Hiện tượng đặc biệt trong tiếng Việt",
            "sub-sub-category": "Sino-Vietnamese Lexical Semantics", "difficulty": "hard",
        },
    ])


# ---------------------------------------------------------------------------------------------
# Hán Việt
# ---------------------------------------------------------------------------------------------
def build_han_viet(root: Path) -> None:
    write_corpus(root)
    out = speech_dir(root) / "han_viet"
    out.mkdir(parents=True, exist_ok=True)
    # answer xuất hiện verbatim trong transcript -> extract_target_word() trích được target_word
    # (điều kiện để classify-levels gọi Gemini phân loại thay vì mặc định level 1).
    _write_jsonl(out / "han_viet_qa.jsonl", [
        {
            "id": "hv-001",
            "question": "Từ Hán-Việt nào dưới đây xuất hiện trong đoạn ghi âm?",
            "answer": "tổ quốc",
            "transcript": "chúng ta cần bảo vệ tổ quốc và giữ gìn văn hóa",
            "audio": _audio("a3.wav"), "audio_id": f"{DATASET}/a3.wav", "dataset": DATASET,
        },
        {
            "id": "hv-002",
            "question": "Từ Hán-Việt nào dưới đây xuất hiện trong đoạn ghi âm?",
            "answer": "văn hóa",
            "transcript": "chúng ta cần bảo vệ tổ quốc và giữ gìn văn hóa",
            "audio": _audio("a3.wav"), "audio_id": f"{DATASET}/a3.wav", "dataset": DATASET,
        },
    ])


# ---------------------------------------------------------------------------------------------
# Phương ngữ
# ---------------------------------------------------------------------------------------------
def build_phuong_ngu(root: Path) -> None:
    write_corpus(root)
    out = speech_dir(root) / "phuong_ngu"
    out.mkdir(parents=True, exist_ok=True)
    common = {
        "dataset": DATASET, "category": "Reasoning",
        "sub-category": "Hiện tượng đặc biệt trong tiếng Việt",
        "sub-sub-category": "Dialectal Form Normalization", "difficulty": "medium",
    }
    _write_jsonl(out / "phuong_ngu_qa.jsonl", [
        {
            "id": "pn-001", "answer": "mô tê răng rứa",
            "transcript": "anh đi mô tê răng rứa", "audio_id": f"{DATASET}/a4.wav",
            **common,
        },
        {
            "id": "pn-002", "answer": "hen",
            "transcript": "ngày mai ghé chơi hen", "audio_id": f"{DATASET}/a4.wav",
            **common,
        },
    ])


# ---------------------------------------------------------------------------------------------
# Từ mượn
# ---------------------------------------------------------------------------------------------
TU_MUON_COLUMNS = ["Từ Tiếng Việt (Việt Hóa)", "Từ Gốc", "Ngôn Ngữ / Nguồn Gốc", "Nhóm Lĩnh Vực", "Ý Nghĩa / Ghi Chú"]


def build_tu_muon(root: Path) -> None:
    write_corpus(root)
    d = root / "tu_muon"
    d.mkdir(parents=True, exist_ok=True)
    _write_csv(d / "tu_muon_tieng_viet_viet_hoa.csv", TU_MUON_COLUMNS, [
        {"Từ Tiếng Việt (Việt Hóa)": "cà phê", "Từ Gốc": "café", "Ngôn Ngữ / Nguồn Gốc": "Pháp", "Nhóm Lĩnh Vực": "ẩm thực", "Ý Nghĩa / Ghi Chú": "đồ uống có caffeine"},
        {"Từ Tiếng Việt (Việt Hóa)": "bánh mì", "Từ Gốc": "pain", "Ngôn Ngữ / Nguồn Gốc": "Pháp", "Nhóm Lĩnh Vực": "ẩm thực", "Ý Nghĩa / Ghi Chú": "bánh nướng"},
        {"Từ Tiếng Việt (Việt Hóa)": "ga", "Từ Gốc": "gare", "Ngôn Ngữ / Nguồn Gốc": "Pháp", "Nhóm Lĩnh Vực": "giao thông", "Ý Nghĩa / Ghi Chú": "nhà ga"},
        {"Từ Tiếng Việt (Việt Hóa)": "xà phòng", "Từ Gốc": "savon", "Ngôn Ngữ / Nguồn Gốc": "Pháp", "Nhóm Lĩnh Vực": "vệ sinh", "Ý Nghĩa / Ghi Chú": "chất tẩy rửa"},
        {"Từ Tiếng Việt (Việt Hóa)": "tivi", "Từ Gốc": "television", "Ngôn Ngữ / Nguồn Gốc": "Anh", "Nhóm Lĩnh Vực": "điện tử", "Ý Nghĩa / Ghi Chú": "máy thu hình"},
    ])


# ---------------------------------------------------------------------------------------------
# Từ láy
# ---------------------------------------------------------------------------------------------
def build_tu_lay(root: Path) -> None:
    write_corpus(root)
    d = root / "tu_lay"
    d.mkdir(parents=True, exist_ok=True)
    # Nguồn (1): toàn bộ + vần (cột "Phân loại" -> split-by-prefix "Láy toàn bộ")
    _write_csv(d / "tu_lay_toan_bo_va_van.csv",
               ["Từ láy", "Phân loại", "Từ gốc (cơ sở)", "Ý nghĩa", "Sắc thái biểu đạt"], [
        {"Từ láy": "xanh xao", "Phân loại": "Láy toàn bộ", "Từ gốc (cơ sở)": "xanh", "Ý nghĩa": "màu xanh nhạt", "Sắc thái biểu đạt": "giảm nhẹ"},
        {"Từ láy": "mát mẻ", "Phân loại": "Láy toàn bộ", "Từ gốc (cơ sở)": "mát", "Ý nghĩa": "mát mẻ dễ chịu", "Sắc thái biểu đạt": "tăng mạnh"},
        {"Từ láy": "tươi tắn", "Phân loại": "Láy toàn bộ", "Từ gốc (cơ sở)": "tươi", "Ý nghĩa": "tươi vui", "Sắc thái biểu đạt": "tăng mạnh"},
        {"Từ láy": "sáng sủa", "Phân loại": "Láy toàn bộ", "Từ gốc (cơ sở)": "sáng", "Ý nghĩa": "sáng rõ", "Sắc thái biểu đạt": "tăng mạnh"},
        {"Từ láy": "nhè nhẹ", "Phân loại": "Láy toàn bộ", "Từ gốc (cơ sở)": "nhẹ", "Ý nghĩa": "rất nhẹ", "Sắc thái biểu đạt": "giảm nhẹ"},
        {"Từ láy": "mạnh mẽ", "Phân loại": "Láy toàn bộ", "Từ gốc (cơ sở)": "mạnh", "Ý nghĩa": "rất mạnh", "Sắc thái biểu đạt": "tăng mạnh"},
        {"Từ láy": "đẹp đẽ", "Phân loại": "Láy toàn bộ", "Từ gốc (cơ sở)": "đẹp", "Ý nghĩa": "đẹp rõ ràng", "Sắc thái biểu đạt": "tăng mạnh"},
        {"Từ láy": "lạnh lẽo", "Phân loại": "Láy toàn bộ", "Từ gốc (cơ sở)": "lạnh", "Ý nghĩa": "rất lạnh", "Sắc thái biểu đạt": "tăng mạnh"},
        {"Từ láy": "lúng túng", "Phân loại": "Láy vần (ung)", "Từ gốc (cơ sở)": "", "Ý nghĩa": "vụng về", "Sắc thái biểu đạt": "giảm nhẹ"},
        {"Từ láy": "vội vàng", "Phân loại": "Láy vần (ang)", "Từ gốc (cơ sở)": "", "Ý nghĩa": "gấp gáp", "Sắc thái biểu đạt": "tăng mạnh"},
    ])
    # Nguồn (2): "chung" (schema rộng hơn, nguồn riêng)
    _write_csv(d / "tu_lay_tieng_viet.csv",
               ["Từ láy", "Loại từ láy", "Từ loại", "Ý nghĩa", "Sắc thái biểu đạt"], [
        {"Từ láy": "lúng túng", "Loại từ láy": "Láy vần", "Từ loại": "tính từ", "Ý nghĩa": "vụng về", "Sắc thái biểu đạt": "giảm nhẹ"},
        {"Từ láy": "vội vàng", "Loại từ láy": "Láy vần", "Từ loại": "tính từ", "Ý nghĩa": "gấp gáp", "Sắc thái biểu đạt": "tăng mạnh"},
    ])


# ---------------------------------------------------------------------------------------------
# Code-switching
# ---------------------------------------------------------------------------------------------
def build_code_switching(root: Path) -> None:
    write_corpus(root)
    cs = root / "benchmark_qa/speech/trich_xuat_thong_tin/code_switching"
    cs.mkdir(parents=True, exist_ok=True)
    # Từ điển 1 token/dòng (CHỈ dùng list này).
    (cs / "cs_broad_new.txt").write_text("wifi\nemail\nonline\n", encoding="utf-8")
    # GigaSpeech transcript chưa có cs_terms.
    _write_jsonl(cs / "giga_speech_test.jsonl", [
        {"id": "g1", "text": "hôm nay tôi dùng wifi và gửi email", "audio": _audio("a1.wav")},
        {"id": "g2", "text": "tôi mua hàng online", "audio": _audio("a2.wav")},
    ])
    # ViMed đã có cs_terms xác thực.
    _write_jsonl(cs / "vimed_css_test.jsonl", [
        {"id": "v1", "segment_text": "bệnh nhân cần đo ct scanner", "cs_terms": ["ct scanner"], "audio": _audio("a3.wav")},
    ])
    _write_jsonl(cs / "vimed_css_test_hard.jsonl", [
        {"id": "v2", "segment_text": "dùng thuốc paracetamol", "cs_terms": ["paracetamol"], "audio": _audio("a3.wav")},
    ])
