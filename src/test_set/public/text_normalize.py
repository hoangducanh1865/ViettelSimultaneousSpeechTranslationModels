from __future__ import annotations

import re
import unicodedata

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
