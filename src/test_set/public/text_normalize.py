from __future__ import annotations

import re
import unicodedata
from collections import Counter

# Keep letters (incl. combining marks / Vietnamese diacritics), digits and
# whitespace; strip everything else (punctuation, symbols).
_KEEP_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")

_VN_DIACRITIC_RE = re.compile(
    "[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợ"
    "ùúủũụưừứửữựỳýỷỹỵđ"
    "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢ"
    "ÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ]"
)

# Subset of the above that EXCLUDES the plain single-accent letters
# (à á â ã è é ê ì í ò ó ô õ ù ú ý) shared with French/Portuguese/Spanish/
# Italian -- those alone are common in real English text via loaned proper
# nouns (e.g. "René Descartes") and are NOT a reliable Vietnamese signal.
# Only Vietnamese-exclusive letters (ư ơ ă đ) and Vietnamese-exclusive tone
# marks (hook-above / dot-below / stacked circumflex+grave etc., e.g.
# ả ẹ ỉ ị ầ ấ ẫ ằ ắ ẵ ...) remain -- these genuinely never occur in those
# other languages, so this is a high-precision "this is Vietnamese" check.
_VN_DISTINCTIVE_DIACRITIC_RE = re.compile(
    "[ảạăằắẳẵặầấẩẫậẻẽẹềếểễệỉịỏọồốổỗộơờớởỡợủụưừứửữựỳỷỹỵđ"
    "ẢẠĂẰẮẲẴẶẦẤẨẪẬẺẼẸỀẾỂỄỆỈỊỎỌỒỐỔỖỘƠỜỚỞỠỢỦỤƯỪỨỬỮỰỲỶỸỴĐ]"
)


def normalize_text(text: str) -> str:
    """Normalize text so outputs from different ASR backends become
    comparable before word-level ROVER alignment.

    Steps: Unicode NFC normalization (backends aren't guaranteed to return
    the same NFC/NFD form -- an invisible mismatch here would silently
    corrupt alignment) -> lowercase -> strip punctuation -> collapse
    whitespace. Idempotent.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = text.replace("_", " ")
    text = _KEEP_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """normalize_text(text) split into words."""
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


def contains_vietnamese_diacritics(text: str) -> bool:
    """True if `text` contains at least one Vietnamese diacritic character.
    English essentially never contains these -- a cheap, low-false-positive
    signal for "this text_en accidentally contains Vietnamese" (vs. trying
    to positively classify "is this really English", which is unreliable).
    Also used the other way: a sufficiently long text_vi with ZERO
    diacritics is suspicious (Vietnamese relies heavily on tone marks)."""
    return bool(_VN_DIACRITIC_RE.search(text))


def contains_distinctive_vietnamese_diacritics(text: str) -> bool:
    """Stricter than `contains_vietnamese_diacritics`: True only for
    Vietnamese-EXCLUSIVE letters/tone marks (ư ơ ă đ, hook-above, dot-below,
    stacked circumflex+grave/acute/tilde combos). Excludes the plain single
    accents (é, à, ê, ã, ...) that French/Portuguese/Spanish/Italian proper
    nouns also use -- catches false positives like "René Descartes"
    correctly appearing in an English sentence, which the broad check
    above would (and did) wrongly flag as "text_en contains Vietnamese"."""
    return bool(_VN_DISTINCTIVE_DIACRITIC_RE.search(text))


def repetition_ratio(text: str, n: int = 2) -> float:
    """Fraction of `text`'s words taken up by its single most-repeated
    n-gram -- catches ASR/hallucination loops (e.g. "mấy mấy mấy mấy..."),
    not real speaker repetition. 0.0 for text too short to have any n-gram
    repeat meaningfully (fewer than 2*n words).

    Relocated (unchanged) from its original inline definition in
    `TestSet_construction copy 12.ipynb`'s mục 2.5 junk-filter cell, where
    it was applied to pre-ROVER ASR hypotheses; also reused by
    `dataset_versioning.validate_samples()` as a final degenerate-text
    check on the fully-assembled `text_vi`/`text_en`.
    """
    words = text.split()
    if len(words) < n * 2:
        return 0.0
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    most_common_count = Counter(ngrams).most_common(1)[0][1]
    return most_common_count * n / len(words)
