from __future__ import annotations

import re
import unicodedata
from collections import Counter

# Keep letters (incl. combining marks / Vietnamese diacritics), digits and
# whitespace; strip everything else (punctuation, symbols).
_KEEP_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")


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
