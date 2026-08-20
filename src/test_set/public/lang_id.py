"""Audio-based spoken-language identification, used to drop samples whose
audio isn't (at least partly) Vietnamese before spending ASR/LLM/translate
budget on them.

Uses a small multilingual Whisper checkpoint's built-in language-ID head
(`WhisperForConditionalGeneration.detect_language`, a single encoder+one
decoder-step forward pass -- no autoregressive generation, so this is much
cheaper than a real transcription call). Deliberately NOT PhoWhisper: the
existing PhoWhisper voter (asr_client_phowhisper.py) always calls
`generate(..., language="vi")`, forcing Vietnamese decoding regardless of
what the audio actually contains, so it has no free language-detection
signal to give -- it would "detect" every file as Vietnamese.

Whisper's feature extractor has a fixed 30s input window, so a single
detect_language() call only judges the first 30s of a file. To judge the
*whole* file (not just its start) for anything longer, split into
consecutive 30s windows and run detect_language() on each.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

LID_MODEL_ID = "openai/whisper-base"
SEGMENT_SECONDS = 30  # Whisper's fixed input window
SAMPLE_RATE = 16000

_model = None
_processor = None


def _ensure_model():
    global _model, _processor
    if _model is not None:
        return _model, _processor

    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _processor = WhisperProcessor.from_pretrained(LID_MODEL_ID)
    _model = WhisperForConditionalGeneration.from_pretrained(LID_MODEL_ID).to(device)
    _model.eval()
    return _model, _processor


def _load_audio_16k(wav_path: Path):
    import torchaudio

    waveform, sr = torchaudio.load(str(wav_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    return waveform.squeeze(0).numpy()


def _segment_language_probs(input_features) -> dict[str, float]:
    """One forward pass -> softmax over Whisper's language tokens only.

    Reimplements the core of WhisperForConditionalGeneration.detect_language
    (mask logits down to the language-token set, softmax) but keeps the
    full distribution instead of just the argmax, so callers can inspect
    e.g. the Vietnamese probability even when it isn't the top pick.
    """
    import torch

    model, _ = _ensure_model()
    generation_config = model.generation_config
    lang_to_id: dict[str, int] = generation_config.lang_to_id  # {"<|en|>": id, ...}
    id_to_lang = {v: k.strip("<|>") for k, v in lang_to_id.items()}
    lang_ids = list(lang_to_id.values())

    decoder_input_ids = torch.ones(
        (input_features.shape[0], 1), device=model.device, dtype=torch.long
    ) * generation_config.decoder_start_token_id

    with torch.no_grad():
        logits = model(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
        ).logits[:, -1]

    lang_logits = logits[0, lang_ids]
    probs = torch.softmax(lang_logits, dim=-1)
    return {id_to_lang[i]: p.item() for i, p in zip(lang_ids, probs)}


def detect_language_per_segment(wav_path: Path) -> list[dict[str, float]]:
    """Split `wav_path` into consecutive 30s windows and return one
    language-probability distribution (dict lang_code -> prob) per window,
    covering the file's entire duration."""
    model, processor = _ensure_model()
    audio = _load_audio_16k(wav_path)

    window = SEGMENT_SECONDS * SAMPLE_RATE
    chunks = [audio[i : i + window] for i in range(0, len(audio), window)]
    # A trailing sliver (< 1s) carries no reliable LID signal on its own;
    # drop it unless it's the *only* chunk (very short file).
    filtered = [c for c in chunks if len(c) >= SAMPLE_RATE]
    chunks = filtered or chunks[:1]

    segments = []
    for chunk in chunks:
        inputs = processor(chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        input_features = inputs.input_features.to(model.device)
        segments.append(_segment_language_probs(input_features))
    return segments


def audio_language_verdict(
    wav_path: Path,
    *,
    expected_language: str = "vi",
    min_prob: float = 0.15,
) -> dict:
    """Judge whether `wav_path` should be kept for a Vietnamese-source
    pipeline: kept unless the audio is entirely non-`expected_language`
    (no 30s window both picks `expected_language` as its top language AND
    gives it at least `min_prob`) -- a clip that's mostly Vietnamese with
    one English-quoted segment is still kept, only a wholly off-language
    clip is dropped.

    The `min_prob` floor matters in practice: on short/accented clips
    whisper-base's LID head is often barely confident at all, and picking
    whichever language narrowly edges out the rest (e.g. vi=0.12 vs
    en=0.11 among ~100 candidates) is not a meaningful "this is
    Vietnamese" signal -- it lets fully non-Vietnamese clips slip through
    just because "vi" happened to win a coin-flip-level tie. Requiring a
    real minimum probability, not just the argmax, catches those.

    Returns:
        {
          "keep": bool,
          "expected_language": str,
          "segment_top_langs": [str, ...],   # one per 30s window
          "segment_probs": [dict, ...],       # full distribution per window
          "error": Optional[str],
        }
    """
    try:
        segments = detect_language_per_segment(wav_path)
        top_langs = [max(seg, key=seg.get) for seg in segments]
        keep = any(
            top == expected_language and seg.get(expected_language, 0.0) >= min_prob
            for top, seg in zip(top_langs, segments)
        )
        return {
            "keep": keep,
            "expected_language": expected_language,
            "segment_top_langs": top_langs,
            "segment_probs": segments,
            "error": None,
        }
    except Exception as e:
        # Fail open: an LID error shouldn't silently drop a sample from
        # the pipeline -- keep it and surface the error so it's visible
        # (caller can decide whether to treat unknown-LID as fatal).
        return {
            "keep": True,
            "expected_language": expected_language,
            "segment_top_langs": [],
            "segment_probs": [],
            "error": repr(e),
        }
