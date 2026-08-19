# cd "/Users/hoangducanh/Documents/tmp_it/python/test_files/py" && /opt/anaconda3/bin/python t6.py


"""
Zero-shot voice cloning with ZipVoice-Vietnamese-2500h
(https://huggingface.co/hynt/ZipVoice-Vietnamese-2500h)

Clones the voice from a reference wav and synthesizes new Vietnamese text
with it, using the k2-fsa ZipVoice inference pipeline.

Model license (CC-BY-NC-SA-4.0): research / experimentation only.

Prerequisites (run once):
    pip install zipvoice huggingface_hub
    pip install piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html
    # optional, only needed for automatic transcription of the reference audio:
    pip install faster-whisper
"""

import shutil
import subprocess
import sys
from pathlib import Path

HF_REPO_ID = "hynt/ZipVoice-Vietnamese-2500h"
CHECKPOINT_NAME = "iter-525000-avg-2.pt"
ZIPVOICE_GIT_URL = "https://github.com/k2-fsa/ZipVoice.git"

SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_MODEL_DIR = SCRIPT_DIR / "models" / "zipvoice-vi-2500h"
# The PyPI "zipvoice" package is just an empty stub, so the actual inference
# code is pulled from the GitHub repo and run from there (matches upstream's
# own "git clone" + "python3 -m zipvoice.bin.infer_zipvoice" instructions).
ZIPVOICE_REPO_DIR = SCRIPT_DIR / "ZipVoice_repo"
OUTPUT_WAV = SCRIPT_DIR / "outputs" / "zipvoice_clone_output.wav"

PROMPT_WAV = (
    "/Users/hoangducanh/Documents/tmp_it/vdt/vocie_agent_for_edge_device/"
    "datasets/tmp_VietSuperSpeech/audio (4).wav"
)
# Exact transcription of PROMPT_WAV. Leave as None to auto-transcribe with
# faster-whisper; set it manually for the best cloning quality.
PROMPT_TEXT = None

TARGET_TEXT = "Cái số đó là ba mươi... à không, ba mươi hai nghìn, hay ba mươi lăm nhỉ, thôi đại khái là hơn ba chục nghìn."


def ensure_zipvoice_repo(repo_dir: Path) -> Path:
    if not (repo_dir / "zipvoice").is_dir():
        subprocess.run(
            ["git", "clone", "--depth", "1", ZIPVOICE_GIT_URL, str(repo_dir)],
            check=True,
        )
    return repo_dir


def download_model(repo_id: str, local_dir: Path) -> Path:
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir))

    # ZipVoice's CLI expects the config file to be named "model.json".
    config_json = local_dir / "config.json"
    model_json = local_dir / "model.json"
    if config_json.exists() and not model_json.exists():
        shutil.copyfile(config_json, model_json)

    return local_dir


def transcribe_prompt(wav_path: str) -> str:
    from faster_whisper import WhisperModel

    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(wav_path, language="vi")
    text = " ".join(segment.text.strip() for segment in segments).strip()
    if not text:
        raise RuntimeError(
            "Auto-transcription returned empty text; set PROMPT_TEXT manually."
        )
    return text


def run_zipvoice_inference(
    repo_dir: Path,
    model_dir: Path,
    checkpoint_name: str,
    prompt_wav: str,
    prompt_text: str,
    text: str,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "zipvoice.bin.infer_zipvoice",
        "--model-name",
        "zipvoice",
        "--model-dir",
        str(model_dir),
        "--checkpoint-name",
        checkpoint_name,
        "--tokenizer",
        "espeak",
        "--lang",
        "vi",
        "--prompt-wav",
        prompt_wav,
        "--prompt-text",
        prompt_text,
        "--text",
        text,
        "--res-wav-path",
        str(out_path),
    ]
    # Run with cwd = the cloned repo root so "zipvoice" resolves as a local
    # package (it is not installed via pip, see ensure_zipvoice_repo above).
    subprocess.run(cmd, cwd=str(repo_dir), check=True)


def main() -> None:
    if not Path(PROMPT_WAV).exists():
        raise FileNotFoundError(f"Reference audio not found: {PROMPT_WAV}")

    print("Fetching ZipVoice source from GitHub ...")
    repo_dir = ensure_zipvoice_repo(ZIPVOICE_REPO_DIR)

    print(f"Downloading model from huggingface.co/{HF_REPO_ID} ...")
    model_dir = download_model(HF_REPO_ID, LOCAL_MODEL_DIR)

    prompt_text = PROMPT_TEXT
    if prompt_text is None:
        print("No PROMPT_TEXT set, transcribing reference audio with Whisper ...")
        prompt_text = transcribe_prompt(PROMPT_WAV)
    print(f"Prompt text: {prompt_text}")

    print("Running ZipVoice inference ...")
    run_zipvoice_inference(
        repo_dir=repo_dir,
        model_dir=model_dir,
        checkpoint_name=CHECKPOINT_NAME,
        prompt_wav=PROMPT_WAV,
        prompt_text=prompt_text,
        text=TARGET_TEXT,
        out_path=OUTPUT_WAV,
    )
    print(f"Done. Output saved to {OUTPUT_WAV}")


if __name__ == "__main__":
    main()
