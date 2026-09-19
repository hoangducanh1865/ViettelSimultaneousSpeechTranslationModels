#!/usr/bin/env python3
"""Sinh file input GỘP cho task Từ láy: `tu_lay.csv` (1 file, cột "Loại láy" chú thích loại).

Nguồn:
  1. 3 CSV curated cũ (tu_lay_toan_bo.csv / tu_lay_van.csv / tu_lay_tieng_viet.csv).
  2. web_candidates_tu_lay_{toan_bo,van,chung}.json (word + note do web search tổng hợp).
  3. Sinh theo quy tắc láy toàn bộ (wiktionary Phụ lục:Từ láy tiếng Việt):
       - nhân bản nguyên:            to -> to to, xanh -> xanh xanh
       - đổi thanh B/C -> A cùng âm vực: thối -> thôi thối, nhẹ -> nhè nhẹ, đỏ -> đo đỏ, dễ -> dề dễ
       - âm cuối tắc -> mũi đồng vị:  mập -> mầm mập, nhạt -> nhàn nhạt, lệch -> lềnh lệch, điếc -> điêng điếc
  4. (tuỳ chọn --mine-corpus) mine bigram láy toàn bộ ĐÃ xuất hiện trong corpus ASR (độ chính xác cao).

Base word lấy từ: chính các CSV/web candidates (suy ra từ âm tiết thứ 2) + danh sách base phổ biến
nhúng trong EMBEDDED_BASES.

Usage:
    python tools/build_tu_lay_csv.py \
        --tu-lay-dir data/benchmark_qa/speech/cac_hien_tuong_dac_biet_trong_tieng_viet/tu_lay \
        --release-hf data/benchmark_qa/speech/speech_sources.jsonl \
        --out data/benchmark_qa/speech/cac_hien_tuong_dac_biet_trong_tieng_viet/tu_lay/tu_lay.csv \
        --mine-corpus

Ghi chú: Ý nghĩa/Sắc thái để rỗng cho dòng sinh mới nếu chưa có note; bước debate/LLM sau sẽ điền.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

WORD_COL = "Từ láy"
TYPE_COL = "Loại láy"
COLUMNS = ["STT", TYPE_COL, "Phân loại", "Loại từ láy", "Từ loại", "Ý nghĩa", "Sắc thái biểu đạt", "Từ gốc (cơ sở)"]
# Thứ tự cột thực tế xuất ra (đưa "Từ láy" lên đầu sau STT cho dễ đọc).
OUT_COLUMNS = ["STT", WORD_COL, TYPE_COL, "Phân loại", "Loại từ láy", "Từ loại", "Ý nghĩa", "Sắc thái biểu đạt", "Từ gốc (cơ sở)"]

_BASE_RE = re.compile(r"base\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)

# ---------------------------------------------------------------------------------------------
# Thanh điệu / âm cuối (NFD)
# ---------------------------------------------------------------------------------------------
_TONE_MARK = {"\u0301": "sac", "\u0300": "huyen", "\u0309": "hoi", "\u0303": "nga", "\u0323": "nang"}
_TRAC_TO_BANG = {"sac": "ngang", "nang": "huyen", "hoi": "ngang", "nga": "huyen"}
_STOP_TO_NASAL = {"p": "m", "t": "n", "ch": "nh", "c": "ng"}
_STOP_FINALS = tuple(sorted(_STOP_TO_NASAL, key=len, reverse=True))
# Anchor sắc/nặng/hỏi/ngã khi gỡ thanh để ghép lại (mọi nguyên âm tiếng Việt đều nhận dấu thanh).
_VOWELS = set("aăâeêioôơuưy")


def _strip_tone(syllable: str) -> str:
    """Bỏ dấu thanh, giữ dấu nguyên âm; trả về NFC."""
    out = []
    for ch in unicodedata.normalize("NFD", syllable):
        if ch in _TONE_MARK:
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def _get_tone(syllable: str) -> str:
    for ch in unicodedata.normalize("NFD", syllable):
        if ch in _TONE_MARK:
            return _TONE_MARK[ch]
    return "ngang"


def _split_final(no_tone: str) -> tuple[str, str]:
    """Tách (phần_đầu_không_dấu, âm_cuối). Âm cuối là ch/c/ng/nh/m/n/p/t hoặc rỗng."""
    for fin in ("ch", "nh", "ng", "c", "m", "n", "p", "t"):
        if no_tone.endswith(fin) and len(no_tone) > len(fin):
            return no_tone[: -len(fin)], fin
    return no_tone, ""


def _add_tone(no_tone: str, tone: str) -> str:
    """Ghép thanh vào nguyên âm chính (sau các dấu nguyên âm ă/â/ê/ô/ơ/ư của âm tiết đó)."""
    if tone == "ngang":
        return no_tone
    mark = {v: k for k, v in _TONE_MARK.items()}[tone]
    decompose = list(unicodedata.normalize("NFD", no_tone))
    for i in range(len(decompose) - 1, -1, -1):
        if decompose[i].lower() in _VOWELS:
            j = i
            while j + 1 < len(decompose) and unicodedata.combining(decompose[j + 1]) != 0:
                j += 1
            decompose.insert(j + 1, mark)
            break
    return unicodedata.normalize("NFC", "".join(decompose))


def mutate_first(base: str) -> str:
    """Âm tiết 1 của láy toàn bộ biến âm, suy từ base theo quy tắc wiktionary."""
    no_tone = _strip_tone(base)
    tone = _get_tone(base)
    head, fin = _split_final(no_tone)
    new_fin = fin
    if fin in _STOP_TO_NASAL:
        new_fin = _STOP_TO_NASAL[fin]
    new_tone = _TRAC_TO_BANG.get(tone, tone)
    return _add_tone(head + new_fin, new_tone)


def is_lay_toan_bo_pair(first: str, base: str) -> bool:
    """True nếu (first, base) là 1 cặp láy toàn bộ (nguyên hoặc biến âm thanh/âm cuối)."""
    if first == base:
        return True
    return mutate_first(base) == first


def _initial(syllable: str) -> str:
    """Phụ âm đầu (đã bỏ thanh) của 1 âm tiết."""
    no_tone = _strip_tone(syllable)
    for i, ch in enumerate(no_tone):
        if ch.lower() in _VOWELS:
            return no_tone[:i]
    return no_tone


def _starts_with_vowel(s: str) -> bool:
    if not s:
        return False
    first = unicodedata.normalize("NFD", s)[0].lower()
    return first in "aeiouy"


def _van(syllable: str) -> str:
    """Vần (bỏ thanh) = phần từ nguyên âm đầu tiên tới hết âm tiết."""
    no_tone = _strip_tone(syllable)
    for i, ch in enumerate(no_tone):
        if ch.lower() in _VOWELS:
            return no_tone[i:]
    return ""


def compute_rhyme(word: str) -> str:
    """Vần chung giữa 2 âm tiết của 1 từ láy vần (bỏ thanh): yêu cầu VẦN GIỐNG NHAU (>=2 ký tự,
    bắt đầu bằng nguyên âm)."""
    parts = word.split()
    if len(parts) < 2:
        return ""
    a, b = parts[0], parts[1]
    if a == b:
        return ""
    va, vb = _van(a), _van(b)
    return vb if (va and va == vb and len(vb) >= 2) else ""


# ---------------------------------------------------------------------------------------------
# Base words
# ---------------------------------------------------------------------------------------------
EMBEDDED_BASES = """
to nhỏ bé mềm cứng nóng lạnh ấm mát nguội nhanh chậm mau lâu cao thấp xa gần dài ngắn
rộng hẹp sâu cạn nông đầy vơi nặng nhẹ già trẻ non đẹp xấu xinh vui buồn sướng khổ đau
ngon ngọt mặn chua cay thơm thối nồng đắng nhạt bùi tươi héo khô ướt dai giòn mịn ráp
sáng tối mờ tỏ rõ đỏ xanh tím vàng hồng trắng đen nâu xám hung đục trong veo
nhiều ít thưa dày đặc loãng đặc sánh khét tanh giòn bở mềm dẻo
khỏe yếu mệt tỉnh say đói no rét bẩn sạch gọn gàng um tùm
hay dở giỏi kém khéo vụng hiền ác chăm lười siêng ngoan ngỗ ngáo
giàu nghèo sang hèn sang trọng quê mộc mạc
nóng bức lạnh giá buốt rét mướt nồm ẩm
đẹp đẽ mượt mà óng ả lấp lánh long lanh
ăn nói viết đi chạy cười khóc ngủ nằm ngồi đứng nhảy hát kêu gọi
nhìn nghe ngửi sờ nếm nghĩ thương nhớ giận hờn ghét yêu
gật lắc gõ đập vỗ xoa vuốt chà cọ
co duỗi gập gãy vỡ nát
thơm phức ngào ngạt thoang thoảng
lạnh lẽo âm ấm nóng nực
chậm chạp lanh lẹ
khẽ nhẹ khẽ khàng
xoay tròn méo vuông dài ngắn
trơ trọi đơn độc cô đơn
im lặng ồn ào náo nhiệt
sạch sẽ tinh tươm gọn ghẽ
cao lớn bé nhỏ to tướng
già nua trẻ trung non nớt
mập ốm gầy béo lùn cao
cứng rắn mềm mại dai dẻo
ngọt lịm chua loét mặn chát
đỏ ửng xanh rì tím biếc vàng óng trắng toát đen láy
nhanh nhẹn lề mề chậm rãi
dài dằng dặc ngắn ngủn
sâu thẳm cao vời rộng mênh
đầy tràn vơi cạn
sáng loáng tối om mờ ảo
ấm áp mát rượi nóng hổi
buồn bã vui vẻ hớn hở
im thin thít ồn ã
khéo léo vụng về
thật thà gian xảo
mạnh mẽ yếu ớt
lặng lẽ ầm ĩ
xanh xao vàng vọt
trắng trẻo đen đủi
lạnh nhạt nồng nàn
sạch bong tinh khôi
giàu có nghèo khó
tốt đẹp xấu xa
cao sang thấp hèn
vui sướng buồn rầu
nhanh gọn chậm chạp
"""


def load_bases(tu_lay_dir: Path) -> set[str]:
    bases: set[str] = set()
    for w in EMBEDDED_BASES.split():
        w = w.strip()
        if w:
            bases.add(w)
    # Chỉ suy base từ tu_lay_toan_bo.csv (âm tiết thứ 2 LÀ từ gốc); KHÔNG lấy từ van/chung vì âm
    # tiết thứ 2 của láy vần/láy âm không phải từ gốc (vd "lao xao" base không phải "xao").
    p = tu_lay_dir / "tu_lay_toan_bo.csv"
    if p.exists():
        with open(p, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                parts = (row.get(WORD_COL) or "").strip().lower().split()
                if parts:
                    bases.add(parts[-1])
    # từ note web candidates (base 'X')
    for name in ("web_candidates_tu_lay_toan_bo.json", "web_candidates_tu_lay_van.json", "web_candidates_tu_lay_chung.json"):
        p = tu_lay_dir / name
        if not p.exists():
            continue
        for item in json.load(open(p, encoding="utf-8")):
            note = item.get("note", "")
            m = _BASE_RE.search(note)
            if m:
                bases.add(m.group(1).strip().lower())
    return bases


# ---------------------------------------------------------------------------------------------
# Row collection
# ---------------------------------------------------------------------------------------------
class Rows:
    def __init__(self) -> None:
        self.by_word: dict[str, dict] = {}

    def add(self, word: str, *, loai_lay: str, y_nghia: str = "", sac_thai: str = "",
            loai_tu_lay: str = "", tu_loai: str = "", base: str = "", overwrite_empty: bool = True) -> bool:
        word = unicodedata.normalize("NFC", " ".join(word.lower().split()))
        loai_tu_lay = loai_tu_lay or loai_lay
        if not word or len(word.split()) < 2:
            return False
        cur = self.by_word.get(word)
        if cur is None:
            self.by_word[word] = {
                WORD_COL: word, TYPE_COL: loai_lay, "Phân loại": loai_lay,
                "Loại từ láy": loai_tu_lay, "Từ loại": tu_loai,
                "Ý nghĩa": y_nghia, "Sắc thái biểu đạt": sac_thai, "Từ gốc (cơ sở)": base,
            }
            return True
        if not overwrite_empty:
            return False
        for k, v in (("Loại từ láy", loai_tu_lay), ("Từ loại", tu_loai), ("Ý nghĩa", y_nghia),
                     ("Sắc thái biểu đạt", sac_thai), ("Từ gốc (cơ sở)", base)):
            if v and not cur.get(k):
                cur[k] = v
        return False


def collect_curated(rows: Rows, tu_lay_dir: Path) -> None:
    src = [
        ("tu_lay_toan_bo.csv", "Láy toàn bộ"),
        ("tu_lay_van.csv", "Láy vần"),
        ("tu_lay_tieng_viet.csv", "Láy âm đầu"),
    ]
    for name, default_type in src:
        p = tu_lay_dir / name
        if not p.exists():
            continue
        with open(p, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                word = (row.get(WORD_COL) or "").strip()
                if not word:
                    continue
                loai = (row.get("Loại từ láy") or row.get("Phân loại") or default_type).strip()
                if loai.startswith("Láy vần") and "(" not in loai:
                    rhyme = compute_rhyme(word)
                    if rhyme:
                        loai = f"Láy vần ({rhyme})"
                rows.add(
                    word, loai_lay=loai,
                    y_nghia=(row.get("Ý nghĩa") or "").strip(),
                    sac_thai=(row.get("Sắc thái biểu đạt") or "").strip(),
                    loai_tu_lay=(row.get("Loại từ láy") or "").strip(),
                    tu_loai=(row.get("Từ loại") or "").strip(),
                    base=(row.get("Từ gốc (cơ sở)") or "").strip(),
                )


def collect_web(rows: Rows, tu_lay_dir: Path) -> None:
    mapping = {
        "web_candidates_tu_lay_toan_bo.json": "Láy toàn bộ",
        "web_candidates_tu_lay_van.json": "Láy vần",
        "web_candidates_tu_lay_chung.json": "Láy âm đầu",
    }
    for name, default_type in mapping.items():
        p = tu_lay_dir / name
        if not p.exists():
            continue
        for item in json.load(open(p, encoding="utf-8")):
            word = (item.get("word") or "").strip()
            note = (item.get("note") or "").strip()
            loai = default_type
            low = note.lower()
            if "toàn bộ" in low:
                loai = "Láy toàn bộ"
            elif "láy vần" in low:
                loai = "Láy vần"
            elif "âm đầu" in low:
                loai = "Láy âm đầu"
            if loai == "Láy vần":
                rhyme = compute_rhyme(word)
                if rhyme:
                    loai = f"Láy vần ({rhyme})"
            base = ""
            m = _BASE_RE.search(note)
            if m:
                base = m.group(1).strip()
            rows.add(word, loai_lay=loai, y_nghia=note, sac_thai="", base=base)


def generate_toan_bo(rows: Rows, bases: set[str]) -> None:
    for base in sorted(bases):
        base = base.strip()
        if not base or " " in base:
            continue
        plain = f"{base} {base}"
        first = mutate_first(base)
        mutated = f"{first} {base}"
        if first != base:
            rows.add(mutated, loai_lay="Láy toàn bộ (biến âm)", base=base)
        rows.add(plain, loai_lay="Láy toàn bộ", base=base)


def _tokens(text: str) -> list[str]:
    out = []
    for tok in text.lower().split():
        tok = "".join(ch for ch in tok if ch.isalpha() or ch.isspace())
        if tok:
            out.append(tok)
    return out


def mine_corpus(rows: Rows, release_hf: Path, bases: set[str], min_freq: int = 2) -> None:
    transcripts: list[str] = []
    if release_hf.suffix == ".jsonl":
        with open(release_hf, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                it = json.loads(line)
                t = it.get("text") or it.get("transcript")
                if t:
                    transcripts.append(t)
    else:
        data = json.load(open(release_hf, encoding="utf-8"))
        entries = data.values() if isinstance(data, dict) else [data]
        for items in entries:
            if not isinstance(items, list):
                continue
            for it in items:
                t = it.get("text") or it.get("transcript") if isinstance(it, dict) else None
                if t:
                    transcripts.append(t)
    tb_counts: Counter = Counter()
    van_counts: Counter = Counter()
    for t in transcripts:
        toks = _tokens(t)
        for a, b in zip(toks, toks[1:]):
            if a == b or b not in bases or a not in bases:
                continue
            if is_lay_toan_bo_pair(a, b):
                tb_counts[(a, b)] += 1
            elif _initial(a) != _initial(b):
                if _van(a) and _van(a) == _van(b) and len(_van(b)) >= 2:
                    van_counts[(a, b)] += 1

    n_tb = n_van = n_am = 0
    for (a, b), c in tb_counts.items():
        if c < min_freq:
            continue
        rows.add(f"{a} {b}", loai_lay=("Láy toàn bộ" if a == b else "Láy toàn bộ (biến âm)"), base=b)
        n_tb += 1
    # Láy vần mine từ bigram chỉ nhận khi xuất hiện >=3 lần (giảm nhiễu bigram tình cờ).
    for (a, b), c in van_counts.items():
        if c < max(min_freq, 3):
            continue
        rhyme = _van(b)
        rows.add(f"{a} {b}", loai_lay=f"Láy vần ({rhyme})", base=b, overwrite_empty=True)
        n_van += 1
    print(f"Mine corpus: toàn bộ {n_tb} | vần {n_van} (âm đầu: KHÔNG mine vì quá nhiễu).")


def classify_chung_types(rows: "Rows", *, env_file: str, batch_size: int = 20,
                         limit: int | None = None, max_retries: int = 3) -> None:
    """LLM phân loại `Loại từ láy` + `Từ loại` cho các dòng nhóm "chung" (không phải toàn bộ/vần)."""
    import sys as _sys
    import time as _time

    hien = Path(__file__).resolve().parents[1] / "speech/cac_hien_tuong_dac_biet_trong_tieng_viet"
    if str(hien) not in _sys.path:
        _sys.path.insert(0, str(hien))
    import auto_model_relay

    auto_model_relay.usage_tracker.set_context(task="tu_lay", stage="build-tu-lay-csv")
    client, model = auto_model_relay.resolve_role_client("GEMINI", env_file=env_file, location="local")

    pending = [r for r in rows.by_word.values()
               if not r[TYPE_COL].startswith("Láy toàn bộ") and not r[TYPE_COL].startswith("Láy vần")]
    if limit is not None:
        pending = pending[:limit]
    print(f"Cần phân loại loại từ láy/từ loại cho {len(pending)} từ nhóm chung.")
    system = (
        "Bạn là chuyên gia ngữ pháp tiếng Việt. Với mỗi từ láy, trả về:\n"
        '  - "loai_tu_lay": một trong "Láy âm đầu", "Láy vần", "Láy toàn bộ", "Láy khuyết âm", "Láy ba tiếng".\n'
        '  - "tu_loai": một trong "Danh từ", "Động từ", "Tính từ", "Phó từ", "Đại từ", "Số từ", "Lượng từ", "Tình thái từ", "Trợ từ", "Quan hệ từ".\n'
        'Chỉ trả về MỘT JSON array: {"word": <đúng từ đầu vào>, "loai_tu_lay": str, "tu_loai": str}. Không markdown.'
    )
    done = 0
    for i in range(0, len(pending), batch_size):
        batch = pending[i:i + batch_size]
        payload = json.dumps([{"word": r[WORD_COL], "loai_lay_doan": r[TYPE_COL]} for r in batch], ensure_ascii=False)
        parsed = None
        for attempt in range(max_retries):
            try:
                raw = auto_model_relay.call_model(
                    "gemini", payload, gemini_client=client, gemini_model=model,
                    system_instruction=system, temperature=0.0, max_output_tokens=4096,
                )
                raw = (raw or "").strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                parsed = json.loads(raw)
                break
            except Exception as e:  # noqa: BLE001
                if attempt < max_retries - 1:
                    _time.sleep(2 ** attempt)
                else:
                    print(f"  [LỖI batch {i // batch_size}] {e!r} -- bỏ qua batch này.")
        if not parsed:
            continue
        by_word = {str(it.get("word", "")).strip().lower(): it for it in parsed if isinstance(it, dict)}
        for r in batch:
            it = by_word.get(r[WORD_COL].lower())
            if not it:
                continue
            if it.get("loai_tu_lay"):
                r["Loại từ láy"] = str(it["loai_tu_lay"]).strip()
                # cập nhật loại để split-by-prefix định tuyến lại (vd chuyển sang vần/toàn bộ)
                if r["Loại từ láy"].startswith(("Láy âm đầu", "Láy khuyết âm", "Láy ba tiếng", "Láy toàn bộ")):
                    r[TYPE_COL] = r["Loại từ láy"]
                    r["Phân loại"] = r["Loại từ láy"]
            if it.get("tu_loai"):
                r["Từ loại"] = str(it["tu_loai"]).strip()
        done += len(batch)
        print(f"  ... {done}/{len(pending)}")


# ---------------------------------------------------------------------------------------------
# Ghi CSV
# ---------------------------------------------------------------------------------------------
def write_csv(rows: Rows, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    items = sorted(rows.by_word.values(), key=lambda r: (r[TYPE_COL], r[WORD_COL]))
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLUMNS)
        w.writeheader()
        for i, r in enumerate(items, 1):
            r["STT"] = i
            w.writerow(r)
    print(f"Đã ghi {out} ({len(items)} từ).")


# ---------------------------------------------------------------------------------------------
# Fill Ý nghĩa / Sắc thái bằng LLM (tuỳ chọn, cần credential .env)
# ---------------------------------------------------------------------------------------------
FILL_SYSTEM_PROMPT = """Bạn là chuyên gia từ vựng tiếng Việt. Với mỗi từ láy đầu vào, hãy trả về:
  - "y_nghia": nghĩa ngắn gọn, chính xác (10-25 từ), nêu rõ từ láy của từ gốc nào và sắc thái mức độ.
  - "sac_thai": CỤM NGẮN (2-5 từ) mô tả sắc thái biểu đạt, chọn trong các nhóm như:
    "giảm nhẹ", "nhấn mạnh", "miêu tả âm thanh", "tượng hình dáng vẻ", "tăng mức độ",
    "đánh giá tích cực", "đánh giá tiêu cực", "trạng thái cảm xúc", "mức độ vừa phải", ...
Chỉ trả về MỘT JSON array, mỗi phần tử: {"word": <đúng từ đầu vào>, "y_nghia": str, "sac_thai": str}.
Không giải thích gì thêm, không bọc trong markdown."""


def fill_meanings(rows: "Rows", *, env_file: str, batch_size: int = 20, limit: int | None = None,
                  max_retries: int = 3, max_output_tokens: int = 8192) -> None:
    import sys as _sys
    import time as _time

    hien = Path(__file__).resolve().parents[1] / "speech/cac_hien_tuong_dac_biet_trong_tieng_viet"
    if str(hien) not in _sys.path:
        _sys.path.insert(0, str(hien))
    import auto_model_relay

    auto_model_relay.usage_tracker.set_context(task="tu_lay", stage="build-tu-lay-csv")
    client, model = auto_model_relay.resolve_role_client("GEMINI", env_file=env_file, location="local")
    pending = [r for r in rows.by_word.values() if not r["Ý nghĩa"].strip() or not r["Sắc thái biểu đạt"].strip()]
    if limit is not None:
        pending = pending[:limit]
    print(f"Cần điền Ý nghĩa/Sắc thái cho {len(pending)} từ (batch={batch_size}).")
    done = 0
    for i in range(0, len(pending), batch_size):
        batch = pending[i:i + batch_size]
        payload = json.dumps(
            [{"word": r[WORD_COL], "loai_lay": r[TYPE_COL], "tu_goc": r["Từ gốc (cơ sở)"]} for r in batch],
            ensure_ascii=False,
        )
        parsed = None
        for attempt in range(max_retries):
            try:
                raw = auto_model_relay.call_model(
                    "gemini", payload, gemini_client=client, gemini_model=model,
                    system_instruction=FILL_SYSTEM_PROMPT, temperature=0.0,
                    max_output_tokens=max_output_tokens,
                )
                raw = (raw or "").strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                parsed = json.loads(raw)
                break
            except Exception as e:  # noqa: BLE001
                if attempt < max_retries - 1:
                    _time.sleep(2 ** attempt)
                else:
                    print(f"  [LỖI batch {i // batch_size}] {e!r} -- bỏ qua batch này.")
        if not parsed:
            continue
        by_word = {str(item.get("word", "")).strip().lower(): item for item in parsed if isinstance(item, dict)}
        for r in batch:
            item = by_word.get(r[WORD_COL].lower())
            if not item:
                continue
            if not r["Ý nghĩa"].strip() and item.get("y_nghia"):
                r["Ý nghĩa"] = str(item["y_nghia"]).strip()
            if not r["Sắc thái biểu đạt"].strip() and item.get("sac_thai"):
                r["Sắc thái biểu đạt"] = str(item["sac_thai"]).strip()
        done += len(batch)
        print(f"  ... {done}/{len(pending)}")
    filled_y = sum(1 for r in rows.by_word.values() if r["Ý nghĩa"].strip())
    filled_s = sum(1 for r in rows.by_word.values() if r["Sắc thái biểu đạt"].strip())
    print(f"Sau fill: {filled_y} có Ý nghĩa, {filled_s} có Sắc thái / {len(rows.by_word)}.")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tu-lay-dir", required=True, help="Thư mục chứa CSV/web candidates từ láy (cũng là nơi ghi mặc định).")
    p.add_argument("--out", default=None, help="Đường dẫn tu_lay.csv (mặc định <tu-lay-dir>/tu_lay.csv).")
    p.add_argument("--release-hf", default=None, help="speech_sources.jsonl để mine corpus.")
    p.add_argument("--mine-corpus", action="store_true", help="Mine thêm cặp láy toàn bộ có thật trong corpus ASR.")
    p.add_argument("--min-freq", type=int, default=2, help="Tần suất tối thiểu khi mine corpus (mặc định 2).")
    p.add_argument("--fill-meaning", action="store_true", help="Điền Ý nghĩa/Sắc thái còn trống bằng LLM (cần .env).")
    p.add_argument("--classify-chung", action="store_true", help="LLM phân loại Loại từ láy/Từ loại cho nhóm chung (cần .env).")
    p.add_argument("--env-file", default=None, help="File .env (mặc định <repo>/.env).")
    p.add_argument("--fill-batch-size", type=int, default=20)
    p.add_argument("--fill-limit", type=int, default=None, help="Chỉ fill N từ đầu (để thử).")
    args = p.parse_args(argv)

    tu_lay_dir = Path(args.tu_lay_dir)
    out = Path(args.out) if args.out else tu_lay_dir / "tu_lay.csv"

    rows = Rows()
    collect_curated(rows, tu_lay_dir)
    n_curated = len(rows.by_word)
    collect_web(rows, tu_lay_dir)
    n_web = len(rows.by_word)
    bases = load_bases(tu_lay_dir)
    generate_toan_bo(rows, bases)
    n_gen = len(rows.by_word)
    if args.mine_corpus:
        if not args.release_hf:
            raise SystemExit("--mine-corpus cần --release-hf")
        mine_corpus(rows, Path(args.release_hf), bases, min_freq=args.min_freq)
    n_all = len(rows.by_word)

    print(f"Curated: {n_curated} | +web: {n_web} | +generated(toàn bộ): {n_gen} | +corpus: {n_all} | bases: {len(bases)}")

    if args.fill_meaning or args.classify_chung:
        env_file = args.env_file or str(Path(__file__).resolve().parents[5] / ".env")
        if args.classify_chung:
            classify_chung_types(rows, env_file=env_file, batch_size=args.fill_batch_size, limit=args.fill_limit)
        if args.fill_meaning:
            fill_meanings(rows, env_file=env_file, batch_size=args.fill_batch_size, limit=args.fill_limit)

    write_csv(rows, out)


if __name__ == "__main__":
    main()
