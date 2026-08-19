"""Optional LLM post-processing stage for the ROVER ensemble (Gemini, Vertex AI).

Two steps, both guarded by a before/after diff check so a hallucinating or
over-eager LLM call can never silently corrupt the transcript:

1. clean_transcript(): per-voter cleanup (fix spacing/punctuation/ASR
   artifacts, keep spoken style, must not change content).
2. fuse_transcripts(): given the ROVER-combined text + every voter's cleaned
   text, produce one final best transcript.

In both cases, if the LLM output diverges too much from its input (too
short/long, too dissimilar), the call is rejected and the *pre-LLM* text is
kept instead -- the LLM can refine, never replace wholesale.

Auth mirrors stuff/a_dat_gui/test_llm_flow.py's pattern (Vertex AI +
service-account credentials), but reads the key from an env var
(GEMINI_SERVICE_ACCOUNT_JSON, in-memory JSON string) instead of a local
file path, since the notebook that sets this up never gets pushed to
GitHub / cloned into Colab.
"""

from __future__ import annotations

import difflib
import json
import os
from typing import Optional

from text_normalize import normalize_text

DEFAULT_LOCATION = "global"
DEFAULT_CLEANUP_MODEL = "gemini-2.5-flash"
DEFAULT_FUSION_MODEL = "gemini-2.5-flash"
DEFAULT_TRANSLATE_MODEL = "gemini-2.5-flash"

CLEANUP_SYSTEM_PROMPT = (
    "Bạn là công cụ hậu xử lý văn bản ASR tiếng Việt (người nói có thể chêm "
    "từ/tên riêng tiếng Anh - code-switching). Nhiệm vụ: sửa lỗi chính tả, "
    "dấu câu, khoảng trắng; sửa các từ/tên riêng tiếng Anh bị nhận dạng sai "
    "thành cách viết phiên âm theo phát âm tiếng Việt về đúng chính tả gốc "
    "tiếng Anh (ví dụ: 'phây búc' -> 'Facebook', 'bưu' khi rõ ràng người nói "
    "đang nói 'boost' -> 'boost'); bỏ các từ/âm lặp do lỗi nhận dạng giọng "
    "nói gây ra, không phải do người nói lặp thật (ví dụ 'mấy mấy mấy' -> "
    "'mấy'). Giữ nguyên văn phong nói tự nhiên (spoken-style) và toàn bộ ý "
    "nghĩa, nội dung câu nói. TUYỆT ĐỐI không thêm thông tin mới, không diễn "
    "giải lại câu, không tóm tắt bớt nội dung, không dịch sang ngôn ngữ "
    "khác. Chỉ trả về đúng văn bản đã sửa, không thêm giải thích, không "
    "thêm dấu ngoặc kép, không thêm tiền tố."
)

FUSION_SYSTEM_PROMPT = (
    "Bạn nhận được kết quả ASR từ nhiều hệ thống khác nhau cho cùng một đoạn "
    "audio tiếng Việt (có thể chêm từ/tên riêng tiếng Anh - code-switching), "
    "cùng một bản đã hợp nhất bằng thuật toán ROVER (vote theo từng từ, dựa "
    "trên confidence của từng hệ thống). Hãy tổng hợp lại thành một câu "
    "chính xác nhất: ưu tiên nội dung xuất hiện ở đa số các bản; nếu các bản "
    "chỉ khác nhau ở cách viết một từ/tên riêng tiếng Anh (một bản viết đúng "
    "chính tả gốc tiếng Anh, các bản khác phiên âm theo phát âm tiếng Việt), "
    "hãy dùng đúng chính tả gốc tiếng Anh cho từ đó dù bản đó là thiểu số; "
    "giữ văn phong nói tự nhiên. TUYỆT ĐỐI không thêm thông tin không xuất "
    "hiện ở bất kỳ bản nào, không suy diễn, không tóm tắt bớt nội dung, "
    "không dịch. Chỉ trả về đúng câu văn bản cuối cùng, không thêm giải "
    "thích."
)

TRANSLATE_SYSTEM_PROMPT = (
    "Bạn là công cụ dịch thuật tiếng Việt sang tiếng Anh. Dịch chính xác câu "
    "sau sang tiếng Anh tự nhiên, giữ nguyên văn phong nói (spoken-style), "
    "giữ nguyên ý nghĩa và mọi chi tiết trong câu gốc. TUYỆT ĐỐI không thêm "
    "thông tin, không bỏ sót nội dung, không diễn giải, không thêm giải "
    "thích. Chỉ trả về đúng câu dịch tiếng Anh, không thêm dấu ngoặc kép, "
    "không thêm tiền tố."
)


# --------------------------------------------------------------------------
# Before/after safety guard
# --------------------------------------------------------------------------

def text_diff_metrics(before: str, after: str) -> dict:
    """Compare `after` (LLM output) against `before` (its input)."""
    before_norm = normalize_text(before or "")
    after_norm = normalize_text(after or "")
    before_words = before_norm.split()
    after_words = after_norm.split()

    similarity = difflib.SequenceMatcher(None, before_norm, after_norm).ratio()
    char_len_ratio = (len(after_norm) / len(before_norm)) if before_norm else float("inf")
    word_len_ratio = (len(after_words) / len(before_words)) if before_words else float("inf")

    return {
        "similarity": similarity,
        "char_len_ratio": char_len_ratio,
        "word_len_ratio": word_len_ratio,
        "before_chars": len(before_norm),
        "after_chars": len(after_norm),
        "before_words": len(before_words),
        "after_words": len(after_words),
    }


def is_safe_edit(
    metrics: dict,
    *,
    min_similarity: float = 0.45,
    min_len_ratio: float = 0.5,
    max_len_ratio: float = 1.6,
) -> bool:
    """Reject edits that are too dissimilar, or that inflate/shrink length
    too much -- typical symptoms of an LLM paraphrasing, hallucinating extra
    content, or truncating instead of just cleaning up."""
    if metrics["similarity"] < min_similarity:
        return False
    if not (min_len_ratio <= metrics["char_len_ratio"] <= max_len_ratio):
        return False
    if not (min_len_ratio <= metrics["word_len_ratio"] <= max_len_ratio):
        return False
    return True


# --------------------------------------------------------------------------
# Gemini client (Vertex AI, service-account auth)
# --------------------------------------------------------------------------

def create_client():
    from google import genai
    from google.oauth2 import service_account

    creds_raw = os.environ.get("GEMINI_SERVICE_ACCOUNT_JSON")
    project_id = os.environ.get("GEMINI_PROJECT_ID")
    if not creds_raw:
        raise RuntimeError("Missing GEMINI_SERVICE_ACCOUNT_JSON env var")
    if not project_id:
        raise RuntimeError("Missing GEMINI_PROJECT_ID env var")

    info = json.loads(creds_raw)
    credentials = service_account.Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    location = os.environ.get("GEMINI_LOCATION", DEFAULT_LOCATION)
    return genai.Client(vertexai=True, project=project_id, location=location, credentials=credentials)


def _call_gemini(client, model: str, system_prompt: str, user_content: str) -> str:
    from google.genai import types

    response = client.models.generate_content(
        model=model,
        contents=user_content,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.1,
            max_output_tokens=2048,
            # Gemini 2.5's internal "thinking" tokens are drawn from the same
            # max_output_tokens budget as the visible answer -- on longer
            # inputs that silently truncated the actual cleaned/fused text
            # mid-sentence (caught by the diff guard as an unsafe edit, but
            # the real bug was here). This is a plain cleanup/fusion task,
            # no reasoning needed, so turn thinking off entirely.
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return (response.text or "").strip()


# --------------------------------------------------------------------------
# The two refinement steps
# --------------------------------------------------------------------------

def clean_transcript(client, model: str, raw_text: str) -> dict:
    """Per-voter cleanup. Never raises -- errors/unsafe edits fall back to
    `final_text = raw_text` so a bad LLM call can't corrupt the pipeline."""
    result = {
        "input_text": raw_text,
        "llm_text": None,
        "final_text": raw_text,
        "metrics": None,
        "accepted": False,
        "error": None,
    }
    if not raw_text or not raw_text.strip():
        result["accepted"] = True  # nothing to clean
        return result

    try:
        llm_text = _call_gemini(client, model, CLEANUP_SYSTEM_PROMPT, raw_text)
        metrics = text_diff_metrics(raw_text, llm_text)
        accepted = bool(llm_text) and is_safe_edit(metrics)
        result.update(
            llm_text=llm_text,
            metrics=metrics,
            accepted=accepted,
            final_text=llm_text if accepted else raw_text,
        )
    except Exception as e:
        result["error"] = repr(e)

    return result


def fuse_transcripts(client, model: str, rover_text: str, per_model_texts: dict) -> dict:
    """Synthesize a final transcript from rover_text + each voter's
    (already-cleaned) text. Falls back to rover_text on error/unsafe edit."""
    result = {
        "input_rover_text": rover_text,
        "llm_text": None,
        "final_text": rover_text,
        "metrics": None,
        "best_reference": None,
        "accepted": False,
        "error": None,
    }
    if not rover_text or not rover_text.strip():
        result["accepted"] = True
        return result

    try:
        lines = [f"- {name}: {text}" for name, text in per_model_texts.items() if text]
        lines.append(f"- rover (hợp nhất bằng vote): {rover_text}")
        prompt = (
            "Các bản ASR cho cùng một đoạn audio:\n"
            + "\n".join(lines)
            + "\n\nHãy tổng hợp thành một câu văn bản cuối cùng chính xác nhất."
        )
        llm_text = _call_gemini(client, model, FUSION_SYSTEM_PROMPT, prompt)

        # Guard against drift from whichever input the fusion result actually
        # resembles most -- NOT just rover_text. rover_text is the raw,
        # unclean vote output (still has ASR errors like "buz"/"phây búc");
        # fusion's whole job is to fix exactly that using the *cleaned*
        # per-model texts, so a correct fusion can legitimately look quite
        # different from rover_text alone while matching a cleaned voter
        # almost exactly. Comparing only to rover_text would reject good
        # corrections as "too different".
        candidates = dict(per_model_texts)
        candidates["rover"] = rover_text
        best_name, best_metrics = None, None
        for name, ref_text in candidates.items():
            if not ref_text:
                continue
            m = text_diff_metrics(ref_text, llm_text)
            if best_metrics is None or m["similarity"] > best_metrics["similarity"]:
                best_name, best_metrics = name, m

        accepted = bool(llm_text) and best_metrics is not None and is_safe_edit(best_metrics)
        result.update(
            llm_text=llm_text,
            metrics=best_metrics,
            best_reference=best_name,
            accepted=accepted,
            final_text=llm_text if accepted else rover_text,
        )
    except Exception as e:
        result["error"] = repr(e)

    return result


# --------------------------------------------------------------------------
# Translation (VI -> EN) of the final ASR text
# --------------------------------------------------------------------------

def is_reasonable_translation_length(
    source_text: str,
    target_text: str,
    *,
    min_word_ratio: float = 0.3,
    max_word_ratio: float = 3.0,
) -> bool:
    """Loose sanity check for a translation.

    Similarity-based diffing (used for cleanup/fusion) doesn't apply across
    languages, so this only catches the failure modes that are still
    detectable without understanding English: an empty/near-empty
    translation (truncation, refusal) or a wildly longer one (rambling,
    repeated/hallucinated content).
    """
    if not target_text or not target_text.strip():
        return False
    src_words = len((source_text or "").split())
    tgt_words = len(target_text.split())
    if src_words == 0:
        return True
    ratio = tgt_words / src_words
    return min_word_ratio <= ratio <= max_word_ratio


def translate_text(client, model: str, source_text: str) -> dict:
    """Translate the final Vietnamese ASR text to English.

    Unlike clean_transcript/fuse_transcripts, there is no sensible
    pre-LLM fallback here (the source is Vietnamese, not English) -- if the
    translation is rejected or the call errors, final_text stays None
    rather than silently returning something wrong.
    """
    result = {
        "source_text": source_text,
        "llm_text": None,
        "final_text": None,
        "accepted": False,
        "error": None,
    }
    if not source_text or not source_text.strip():
        result["accepted"] = True
        result["final_text"] = ""
        return result

    try:
        llm_text = _call_gemini(client, model, TRANSLATE_SYSTEM_PROMPT, source_text)
        accepted = is_reasonable_translation_length(source_text, llm_text)
        result.update(
            llm_text=llm_text,
            accepted=accepted,
            final_text=llm_text if accepted else None,
        )
    except Exception as e:
        result["error"] = repr(e)

    return result
