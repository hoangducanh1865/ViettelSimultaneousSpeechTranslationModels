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
from pathlib import Path
from typing import Optional

from text_normalize import normalize_text

DEFAULT_LOCATION = "global"
DEFAULT_CLEANUP_MODEL = "gemini-2.5-flash"
DEFAULT_FUSION_MODEL = "gemini-2.5-flash"
DEFAULT_TRANSLATE_MODEL = "gemini-3.1-flash-lite"


def _load_context_doc(filename: str) -> str:
    """Read a small reference doc (e.g. vietnamese_pronoun_notes.md) sitting
    next to this module, to append to every prompt as extra context.

    Never raises: a missing/unreadable doc just means the corresponding
    context section is omitted, not a hard failure of the whole ensemble.
    """
    path = Path(__file__).parent / filename
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


# Loaded once at import time and appended to every cleanup/fusion/translate
# prompt below -- see vietnamese_pronoun_notes.md for the "chỉ"/"ảnh"/"ổng"/...
# contracted-pronoun phenomenon this exists to prevent the LLM from "fixing"
# away (which silently flips a 3rd-person reference into a 2nd-person one).
_PRONOUN_CONTEXT_NOTES = _load_context_doc("vietnamese_pronoun_notes.md")


def _with_context(prompt: str) -> str:
    if not _PRONOUN_CONTEXT_NOTES:
        return prompt
    return prompt + "\n\n---\n\n" + _PRONOUN_CONTEXT_NOTES


CLEANUP_SYSTEM_PROMPT = _with_context(
    "Bạn là công cụ hậu xử lý văn bản ASR tiếng Việt (người nói có thể chêm "
    "từ/tên riêng tiếng Anh - code-switching). Nhiệm vụ: sửa lỗi chính tả, "
    "dấu câu, khoảng trắng; sửa các từ/tên riêng tiếng Anh bị nhận dạng sai "
    "thành cách viết phiên âm theo phát âm tiếng Việt về đúng chính tả gốc "
    "tiếng Anh (ví dụ: 'phây búc' -> 'Facebook', 'bưu' khi rõ ràng người nói "
    "đang nói 'boost' -> 'boost'); bỏ các từ/âm lặp do lỗi nhận dạng giọng "
    "nói gây ra, không phải do người nói lặp thật (ví dụ 'mấy mấy mấy' -> "
    "'mấy'). Các đại từ nhân xưng (chị, anh, ông, bà, cô, chú, và dạng rút "
    "gọn ngôi thứ 3 kiểu Nam Bộ chỉ, ảnh, ổng, bả, cổ, chả -- xem bảng chi "
    "tiết bên dưới): TUYỆT ĐỐI giữ nguyên y hệt từ ASR đã nhận dạng, không "
    "đổi theo BẤT KỲ hướng nào -- không đổi dạng rút gọn về dạng gốc, và "
    "cũng không tự suy đoán rồi đổi dạng gốc thành dạng rút gọn (dù ngữ "
    "cảnh có vẻ đang nói về ngôi thứ 3). Chỉ được xem một từ là dạng rút "
    "gọn nếu ASR đã nhận dạng đúng y như vậy; không được suy diễn hay áp "
    "đặt cách viết khác cho từ ASR không thực sự nhận dạng ra. Giữ nguyên "
    "văn phong nói "
    "tự nhiên (spoken-style) và toàn bộ ý nghĩa, nội dung câu nói. TUYỆT ĐỐI "
    "không thêm thông tin mới, không diễn giải lại câu, không tóm tắt bớt "
    "nội dung, không dịch sang ngôn ngữ khác. Chỉ trả về đúng văn bản đã "
    "sửa, không thêm giải thích, không thêm dấu ngoặc kép, không thêm tiền "
    "tố."
)

FUSION_SYSTEM_PROMPT = _with_context(
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

TRANSLATE_SYSTEM_PROMPT = _with_context(
    "Bạn là công cụ dịch thuật tiếng Việt sang tiếng Anh. Dịch chính xác câu "
    "sau sang tiếng Anh tự nhiên, giữ nguyên văn phong nói (spoken-style), "
    "giữ nguyên ý nghĩa và mọi chi tiết trong câu gốc. Đại từ nhân xưng tiếng "
    "Việt (chị, em, anh, cô...) mơ hồ giữa ngôi thứ 2 (you) và ngôi thứ 3 "
    "(she/he) -- hãy dựa vào ngữ cảnh cả câu để xác định đúng: nếu câu đang "
    "kể chuyện/tường thuật về một người khác (có dấu hiệu như trích lời nghĩ "
    "trong ngoặc kép, 'kiểu...', 'là...') thì đó là ngôi thứ 3 (she/he), chỉ "
    "dịch là 'you' khi rõ ràng người nói đang nói trực tiếp với người đó. "
    "TUYỆT ĐỐI không thêm thông tin, không bỏ sót nội dung, không diễn giải, "
    "không thêm giải thích. Chỉ trả về đúng câu dịch tiếng Anh, không thêm "
    "dấu ngoặc kép, không thêm tiền tố."
)

# English-source variants: same job as the two prompts above, but for ASR
# output whose source language is English, not Vietnamese. The
# Vietnamese-phonetic-transliteration-fixing instruction ("phây búc" ->
# "Facebook") and the contracted-pronoun-preservation rule don't apply here
# -- those are phenomena specific to Vietnamese ASR/source text -- so they're
# dropped rather than reused. Deliberately NOT wrapped in
# _with_context()/_PRONOUN_CONTEXT_NOTES: that doc is about interpreting
# Vietnamese contracted 3rd-person forms in Vietnamese source text, which is
# irrelevant noise for cleaning English text.
CLEANUP_SYSTEM_PROMPT_EN = (
    "You are an ASR post-processing tool for English speech from Vietnamese "
    "speakers (may code-switch Vietnamese names, places, or words). Fix "
    "spelling, punctuation, and spacing; correct accent- or homophone-driven "
    "ASR errors (e.g. 'there'/'their', a misheard word that doesn't fit the "
    "sentence); remove word/phrase repetition caused by ASR recognition "
    "errors, not real speaker repetition (e.g. 'the the the' -> 'the'). "
    "Preserve Vietnamese proper nouns and names exactly as ASR transcribed "
    "them -- do not invent diacritics ASR didn't produce, do not guess a "
    "'corrected' spelling. Keep the natural spoken style and the full "
    "meaning of the sentence. Absolutely no new information, no "
    "paraphrasing, no summarizing, no translating. Return only the "
    "corrected text, no explanation, no quotes, no prefix."
)

FUSION_SYSTEM_PROMPT_EN = (
    "You receive ASR results from several different systems for the same "
    "English audio segment (speaker may be a Vietnamese person, possibly "
    "code-switching Vietnamese names/words), plus one version already "
    "merged by the ROVER algorithm (word-level vote weighted by each "
    "system's confidence). Synthesize the most accurate final sentence: "
    "prefer content that appears in most versions; if versions only differ "
    "in how a proper noun or Vietnamese word is spelled/transcribed, prefer "
    "the more clearly correct spelling even if it's the minority version; "
    "keep the natural spoken style. Absolutely do not add information that "
    "doesn't appear in any version, do not infer, do not summarize, do not "
    "translate. Return only the final sentence, no explanation."
)

TRANSLATE_EN_VI_SYSTEM_PROMPT = (
    "Bạn là công cụ dịch thuật tiếng Anh sang tiếng Việt. Dịch chính xác câu "
    "sau sang tiếng Việt tự nhiên, giữ văn phong nói (spoken-style), giữ "
    "nguyên ý nghĩa và mọi chi tiết trong câu gốc. Không có thông tin về "
    "tuổi/giới tính/quan hệ giữa người nói, nên mặc định xưng 'tôi', gọi "
    "người nghe là 'bạn' trừ khi chính câu cần dịch có tín hiệu rõ ràng khác "
    "(một cái tên được gọi trực tiếp, một danh xưng quan hệ gia đình, một "
    "kính ngữ...). Nếu có 'Phụ đề tham khảo' đi kèm: đó là phụ đề tiếng Việt "
    "gốc của video, có thể KHÔNG khớp thời gian chính xác với câu tiếng Anh "
    "cần dịch (do lệch giữa 2 track phụ đề) -- chỉ dùng nó để tham khảo văn "
    "phong, cách xưng hô, thuật ngữ/tên riêng đã được dịch sẵn trong video "
    "này, TUYỆT ĐỐI không chép lại nguyên văn phụ đề tham khảo nếu nội dung "
    "của nó không khớp với đúng câu tiếng Anh cần dịch -- luôn ưu tiên dịch "
    "đúng theo câu tiếng Anh thực tế. TUYỆT ĐỐI không thêm thông tin, không "
    "bỏ sót nội dung, không diễn giải, không thêm giải thích. Chỉ trả về "
    "đúng câu dịch tiếng Việt, không thêm dấu ngoặc kép, không thêm tiền tố."
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


def _thinking_config(model: str, minimal: bool):
    """Thinking tokens are drawn from the same max_output_tokens budget as
    the visible answer -- on longer inputs that let thinking run wild
    silently truncated the actual cleaned/fused/translated text
    mid-sentence (caught by the diff guard as an unsafe edit, but the real
    bug was excess thinking eating the budget).

    Cleanup/fusion are mechanical (fix spelling, pick a word) -> minimal
    thinking is fine and faster/cheaper. Translation, specifically
    resolving whether a contracted pronoun like "chỉ" means "you" or
    "she", genuinely benefits from reasoning about context -- forcing it
    to minimal made the model guess instead of reason, and it guessed
    wrong. So translate calls pass minimal=False to keep the model's
    default (higher) thinking level.

    Gemini 2.5.x controls this via `thinking_budget` (token count, 0 =
    off, unset = model default/dynamic). Gemini 3.x replaced that with
    `thinking_level` ("LOW"/"MEDIUM"/"HIGH", default HIGH if unset) --
    passing the wrong field name for a given model family raises, so
    branch on the model string.
    """
    from google.genai import types

    if not minimal:
        return None  # let the model use its own default thinking level

    if model.startswith("gemini-3"):
        return types.ThinkingConfig(thinking_level="LOW")
    return types.ThinkingConfig(thinking_budget=0)


def _call_gemini(
    client,
    model: str,
    system_prompt: str,
    user_content: str,
    *,
    minimal_thinking: bool = True,
    max_output_tokens: int = 2048,
) -> str:
    from google.genai import types

    response = client.models.generate_content(
        model=model,
        contents=user_content,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.1,
            max_output_tokens=max_output_tokens,
            thinking_config=_thinking_config(model, minimal_thinking),
        ),
    )
    return (response.text or "").strip()


# --------------------------------------------------------------------------
# The two refinement steps
# --------------------------------------------------------------------------

def clean_transcript(
    client, model: str, raw_text: str, *, system_prompt: str = CLEANUP_SYSTEM_PROMPT
) -> dict:
    """Per-voter cleanup. Never raises -- errors/unsafe edits fall back to
    `final_text = raw_text` so a bad LLM call can't corrupt the pipeline.

    `system_prompt` defaults to the Vietnamese-source prompt (zero behavior
    change for existing callers); pass `CLEANUP_SYSTEM_PROMPT_EN` for
    English-source ASR text."""
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
        llm_text = _call_gemini(client, model, system_prompt, raw_text)
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


def fuse_transcripts(
    client, model: str, rover_text: str, per_model_texts: dict,
    *, system_prompt: str = FUSION_SYSTEM_PROMPT,
) -> dict:
    """Synthesize a final transcript from rover_text + each voter's
    (already-cleaned) text. Falls back to rover_text on error/unsafe edit.

    `system_prompt` defaults to the Vietnamese-source prompt (zero behavior
    change for existing callers); pass `FUSION_SYSTEM_PROMPT_EN` for
    English-source ASR text."""
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
        llm_text = _call_gemini(client, model, system_prompt, prompt)

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
        llm_text = _call_gemini(
            client, model, TRANSLATE_SYSTEM_PROMPT, source_text,
            minimal_thinking=False, max_output_tokens=4096,
        )
        accepted = is_reasonable_translation_length(source_text, llm_text)
        result.update(
            llm_text=llm_text,
            accepted=accepted,
            final_text=llm_text if accepted else None,
        )
    except Exception as e:
        result["error"] = repr(e)

    return result


def translate_text_en_vi(
    client, model: str, source_text: str, *, reference_hint: Optional[str] = None
) -> dict:
    """Translate English ASR text to Vietnamese, optionally guided by a
    loosely-aligned Vietnamese reference caption for the same rough segment.

    `reference_hint` exists to make use of a video's own crawled Vietnamese
    captions without depending on their time-alignment to `source_text`
    being exact (it isn't reliable -- the two subtitle tracks can drift
    against each other over a video's runtime). The prompt explicitly tells
    the model to treat it as a style/terminology reference, not as the
    thing to translate, so a mismatched hint can't leak wrong content into
    `final_text`.

    Same no-fallback shape as `translate_text` (VI->EN): the source
    language differs from the output, so there's nothing sensible to fall
    back to on rejection/error -- `final_text` stays None.
    """
    result = {
        "source_text": source_text,
        "reference_hint": reference_hint,
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
        user_content = source_text
        if reference_hint and reference_hint.strip():
            user_content = f"{source_text}\n\n---\nPhụ đề tham khảo (có thể lệch thời gian): {reference_hint}"

        llm_text = _call_gemini(
            client, model, TRANSLATE_EN_VI_SYSTEM_PROMPT, user_content,
            minimal_thinking=False, max_output_tokens=4096,
        )
        accepted = is_reasonable_translation_length(source_text, llm_text)
        result.update(
            llm_text=llm_text,
            accepted=accepted,
            final_text=llm_text if accepted else None,
        )
    except Exception as e:
        result["error"] = repr(e)

    return result
