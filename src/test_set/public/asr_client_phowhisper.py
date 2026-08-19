from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from asr_client_base import BaseASRClient, TranscriptionResult

# vinai/PhoWhisper-{tiny,base,small,medium,large} all exist on the HF Hub.
# Medium is the default for Colab-friendly load time / VRAM; bump to
# -large if the runtime has enough GPU memory.
PHOWHISPER_MODEL_ID = "vinai/PhoWhisper-medium"


class PhoWhisperClient(BaseASRClient):
    """Client for VinAI's PhoWhisper, run locally via transformers.

    Public HF model, no credentials needed. Model + processor are built
    lazily on first transcribe() call so simply instantiating this client
    (e.g. while building the full ensemble's client list) doesn't force a
    multi-GB model download/GPU allocation when this voter is excluded via
    --models.

    Uses `model.generate(..., output_scores=True)` directly (instead of the
    high-level `pipeline()`, which doesn't expose per-token scores) to
    compute a real per-utterance confidence: the average per-token
    probability of the greedily-generated sequence, exp(mean(log_softmax)).
    This is a proxy, not a calibrated probability, but it's a genuine
    signal from the model instead of a fixed guess.

    Whisper's feature extractor truncates/pads input to a 30s window, so
    audio longer than 30s will be silently cut -- fine for the short
    utterances this ensemble targets today, but a real limitation if this
    client is later pointed at longer recordings (would need chunking).
    """

    name = "phowhisper"

    def __init__(self, model_id: str = PHOWHISPER_MODEL_ID, device: Optional[str] = None):
        self.model_id = model_id
        self.device = device
        self._model = None
        self._processor = None

    def _ensure_model(self):
        if self._model is not None:
            return self._model, self._processor

        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        device = self.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._processor = WhisperProcessor.from_pretrained(self.model_id)
        self._model = WhisperForConditionalGeneration.from_pretrained(self.model_id).to(device)
        self._device = device
        return self._model, self._processor

    def _load_audio_16k(self, wav_path: Path):
        import torchaudio

        waveform, sr = torchaudio.load(str(wav_path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        return waveform.squeeze(0).numpy()

    @staticmethod
    def _avg_token_confidence(generated) -> Optional[float]:
        """exp(mean log-prob) of the generated tokens -> pseudo-confidence in (0, 1]."""
        import torch

        scores = generated.scores  # tuple of (1, vocab) logits, one per generated step
        if not scores:
            return None

        generated_ids = generated.sequences[0][-len(scores):]
        log_probs = [
            torch.log_softmax(step_logits[0], dim=-1)[token_id].item()
            for step_logits, token_id in zip(scores, generated_ids)
        ]
        if not log_probs:
            return None
        return float(math.exp(sum(log_probs) / len(log_probs)))

    def transcribe(self, wav_path: Path, *, language: Optional[str] = None) -> TranscriptionResult:
        try:
            import torch

            model, processor = self._ensure_model()
            audio = self._load_audio_16k(wav_path)

            inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(self._device)

            start = time.perf_counter()
            with torch.no_grad():
                generated = model.generate(
                    input_features,
                    language=language or "vi",
                    task="transcribe",
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            latency = time.perf_counter() - start

            text = processor.batch_decode(generated.sequences, skip_special_tokens=True)[0].strip()
            confidence = self._avg_token_confidence(generated)

            return TranscriptionResult(
                model_name=self.name,
                text=text,
                confidence=confidence,
                detected_language=language or "vi",
                latency_sec=latency,
                error=None,
                called_at=datetime.now(timezone.utc).isoformat(),
                extra={"model_id": self.model_id},
            )

        except Exception as e:
            return TranscriptionResult(
                model_name=self.name,
                text=None,
                confidence=None,
                detected_language=None,
                latency_sec=None,
                error=repr(e),
                called_at=datetime.now(timezone.utc).isoformat(),
            )

    def transcribe_batch(self, wav_paths: list[Path], *, language: Optional[str] = None) -> list[TranscriptionResult]:
        """Transcribe several files in a single generate() call.

        Whisper's feature extractor always produces a fixed-length (30s
        window) mel-spectrogram per file regardless of actual audio
        duration, so stacking several files into one batch needs no
        ragged-padding logic on the input side. Much better GPU
        utilization than looping transcribe() one file at a time.

        Raises on failure (unlike transcribe()) -- callers should catch
        and fall back to per-file transcribe() for that chunk rather than
        silently mis-attributing a batch-alignment bug to "no error".
        """
        import torch

        model, processor = self._ensure_model()
        audios = [self._load_audio_16k(p) for p in wav_paths]

        inputs = processor(audios, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(self._device)

        start = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(
                input_features,
                language=language or "vi",
                task="transcribe",
                output_scores=True,
                return_dict_in_generate=True,
            )
        latency = time.perf_counter() - start
        per_item_latency = latency / len(wav_paths)

        texts = processor.batch_decode(generated.sequences, skip_special_tokens=True)

        eos_id = processor.tokenizer.eos_token_id
        n_steps = len(generated.scores)
        total_len = generated.sequences.shape[1]
        generated_ids = generated.sequences[:, total_len - n_steps:]  # drop the forced prefix

        called_at = datetime.now(timezone.utc).isoformat()
        results = []
        for i in range(len(wav_paths)):
            # Batched generation runs every sample for the same number of
            # steps (padding shorter ones after their EOS) -- stop
            # accumulating log-prob right after this sample's own EOS so
            # padding doesn't dilute its confidence.
            log_probs = []
            for step in range(n_steps):
                token_id = generated_ids[i, step].item()
                step_logits = generated.scores[step][i]
                log_probs.append(torch.log_softmax(step_logits, dim=-1)[token_id].item())
                if token_id == eos_id:
                    break
            confidence = float(math.exp(sum(log_probs) / len(log_probs))) if log_probs else None

            results.append(TranscriptionResult(
                model_name=self.name,
                text=texts[i].strip(),
                confidence=confidence,
                detected_language=language or "vi",
                latency_sec=per_item_latency,
                error=None,
                called_at=called_at,
                extra={"model_id": self.model_id, "batched": True, "batch_size": len(wav_paths)},
            ))

        return results
