"""ROVER (Recognizer Output Voting Error Reduction) ensemble for ASR.

Combines transcriptions from several ASR "voters" (internal Whisper API,
Google Chirp 3, VinAI PhoWhisper) into a single, hopefully-more-accurate
transcript per audio file, using the classic NIST ROVER algorithm
(Fiscus, 1997): each hypothesis is incrementally aligned into a growing
Word Transitive Network (WTN) via dynamic programming, then each network
column (aligned word slot, across all voters) is decided by a
confidence-weighted vote.

Usage (see also the notebook cells in noteboooks/Viettel_API_AST_models.ipynb):
    python rover.py --dataset-root /path/to/audio/root --language vi
    python rover.py --dataset-root ... --models internal_whisper,phowhisper  # skip Chirp-3
    python rover.py --dataset-root ... --force   # ignore cache, re-call every model
    python rover.py --dataset-root ... --limit 5 # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from asr_client_base import BaseASRClient, TranscriptionResult
from asr_client_internal_whisper import InternalWhisperClient
from asr_client_chirp3 import Chirp3Client
from asr_client_phowhisper import PhoWhisperClient
from text_normalize import tokenize
import llm_refine

DEFAULT_CONFIDENCE = 0.5  # vote weight used whenever a voter's confidence is None

# Multiplier applied to a voter's weight (confidence, or DEFAULT_CONFIDENCE if
# missing) before ROVER voting. 1.0 = no change. PhoWhisper is bumped up a
# bit: it's a Vietnamese-specific model that's held up well in spot checks,
# and its confidence now comes from a real (if uncalibrated) avg-token-prob
# rather than a fixed guess, so it deserves a slight edge over a plain 0.5
# default -- not enough to let it dominate internal_whisper's real
# high-90s confidence, just enough to win more ties against Chirp-3 (still
# stuck at the DEFAULT_CONFIDENCE fallback).
MODEL_WEIGHT_MULTIPLIER = {
    "phowhisper": 1.15,
}

CACHE_DIRNAME = "rover_cache"
ROVER_OUTPUT_SUBDIR = "rover_output"
AUDIO_EXTENSIONS = (".wav",)

ALL_CLIENT_BUILDERS = {
    "internal_whisper": lambda: InternalWhisperClient(),
    "chirp3": lambda: Chirp3Client(),
    "phowhisper": lambda: PhoWhisperClient(),
}


# --------------------------------------------------------------------------
# ROVER core: Word Transitive Network alignment + confidence-weighted voting
# --------------------------------------------------------------------------

@dataclass
class WTNColumn:
    # word-or-None(=NULL/deletion slot) -> list of (model_name, weight) votes
    votes: dict


def _sub_cost(column: WTNColumn, token: str) -> int:
    return 0 if token in column.votes else 1


def align_pairwise(
    network: list[WTNColumn],
    tokens: list[str],
    model_name: str,
    weight: float,
) -> list[WTNColumn]:
    """Align `tokens` (one hypothesis) into `network` via DP, return the new
    (possibly longer) network with this hypothesis's votes merged in.

    Standard ROVER alignment costs: match/substitution 0 if the token
    already exists as a candidate in that column else 1, insertion (token
    becomes a brand-new column) 1, deletion (network column gets a NULL
    vote from this model, token list not consumed) 1. Ties are broken by
    preferring diag > deletion > insertion, to keep the network compact.
    """
    n = len(network)
    m = len(tokens)

    # dp[i][j] = min cost aligning network[:i] with tokens[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bp = [[""] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        bp[i][0] = "up"
    for j in range(1, m + 1):
        dp[0][j] = j
        bp[0][j] = "left"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag_cost = dp[i - 1][j - 1] + _sub_cost(network[i - 1], tokens[j - 1])
            up_cost = dp[i - 1][j] + 1  # deletion: network col consumed, no token
            left_cost = dp[i][j - 1] + 1  # insertion: token consumed, no network col

            best_cost = min(diag_cost, up_cost, left_cost)
            dp[i][j] = best_cost
            if diag_cost == best_cost:
                bp[i][j] = "diag"
            elif up_cost == best_cost:
                bp[i][j] = "up"
            else:
                bp[i][j] = "left"

    # Backtrace from (n, m) to (0, 0), collecting moves, then replay forward.
    moves = []
    i, j = n, m
    while i > 0 or j > 0:
        move = bp[i][j]
        moves.append(move)
        if move == "diag":
            i, j = i - 1, j - 1
        elif move == "up":
            i -= 1
        else:  # "left"
            j -= 1
    moves.reverse()

    new_network: list[WTNColumn] = []
    i = j = 0
    for move in moves:
        if move == "diag":
            column = network[i]
            column.votes.setdefault(tokens[j], []).append((model_name, weight))
            new_network.append(column)
            i += 1
            j += 1
        elif move == "up":
            column = network[i]
            column.votes.setdefault(None, []).append((model_name, weight))
            new_network.append(column)
            i += 1
        else:  # "left"
            new_network.append(WTNColumn(votes={tokens[j]: [(model_name, weight)]}))
            j += 1

    return new_network


def build_word_transition_network(
    hypotheses: list[tuple[str, list[str], float]],
) -> list[WTNColumn]:
    """Fold hypotheses (model_name, tokens, weight) into a WTN, highest
    weight first, so the strongest voter shapes the network's backbone."""
    network: list[WTNColumn] = []
    for model_name, tokens, weight in sorted(hypotheses, key=lambda h: h[2], reverse=True):
        network = align_pairwise(network, tokens, model_name, weight)
    return network


def vote_column(column: WTNColumn) -> tuple[Optional[str], dict]:
    """Confidence-weighted vote among a column's candidates (incl. NULL).

    Primary key: summed vote weight. Ties broken by (1) raw vote count,
    (2) preferring a non-NULL winner, (3) whichever candidate was
    encountered first while folding hypotheses in (dict insertion order).
    """
    def score(item):
        _cand, vote_list = item
        total_weight = sum(w for _, w in vote_list)
        count = len(vote_list)
        return (total_weight, count, 1 if _cand is not None else 0)

    winner, vote_list = max(column.votes.items(), key=score)
    debug = {"votes": {str(k): list(v) for k, v in column.votes.items()}}
    return winner, debug


def rover_combine(hypotheses: list[tuple[str, list[str], float]]) -> tuple[str, list[dict]]:
    """Build the WTN from `hypotheses` and vote each column.

    Returns (combined_text, per_column_debug_list).
    """
    network = build_word_transition_network(hypotheses)
    words = []
    columns_debug = []
    for column in network:
        winner, debug = vote_column(column)
        debug["winner"] = winner
        columns_debug.append(debug)
        if winner is not None:
            words.append(winner)
    return " ".join(words), columns_debug


# --------------------------------------------------------------------------
# Cache / orchestration
# --------------------------------------------------------------------------

def discover_audio_files(dataset_root: Path) -> list[Path]:
    cache_root = dataset_root / CACHE_DIRNAME
    files = [
        p
        for p in dataset_root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in AUDIO_EXTENSIONS
        and cache_root not in p.parents
    ]
    return sorted(files)


def cache_path_for(dataset_root: Path, model_name: str, wav_path: Path) -> Path:
    relpath = wav_path.relative_to(dataset_root)
    return dataset_root / CACHE_DIRNAME / model_name / relpath.with_suffix(".json")


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def load_cached_result(cache_path: Path) -> Optional[TranscriptionResult]:
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, encoding="utf-8") as f:
            return TranscriptionResult.from_json_dict(json.load(f))
    except Exception:
        return None  # corrupt cache file -> treat as missing, will be re-called


def save_result(cache_path: Path, audio_relpath: str, result: TranscriptionResult) -> None:
    _atomic_write_json(cache_path, result.to_json_dict(audio_relpath))


def get_or_call(
    client: BaseASRClient,
    dataset_root: Path,
    wav_path: Path,
    *,
    language: Optional[str],
    force: bool,
    stats: dict,
) -> TranscriptionResult:
    """Cache-or-call: skip the model if cached output already exists,
    otherwise call it and persist the result."""
    cache_path = cache_path_for(dataset_root, client.name, wav_path)

    if not force:
        cached = load_cached_result(cache_path)
        if cached is not None:
            stats["cache_hits"][client.name] = stats["cache_hits"].get(client.name, 0) + 1
            return cached

    result = client.transcribe(wav_path, language=language)
    stats["fresh_calls"][client.name] = stats["fresh_calls"].get(client.name, 0) + 1
    if result.error is not None:
        stats["errors_by_model"][client.name] = stats["errors_by_model"].get(client.name, 0) + 1

    relpath = str(wav_path.relative_to(dataset_root))
    save_result(cache_path, relpath, result)
    return result


def llm_clean_cache_path(dataset_root: Path, model_name: str, wav_path: Path) -> Path:
    relpath = wav_path.relative_to(dataset_root)
    return dataset_root / CACHE_DIRNAME / "llm_clean" / model_name / relpath.with_suffix(".json")


def llm_fusion_cache_path(dataset_root: Path, wav_path: Path) -> Path:
    relpath = wav_path.relative_to(dataset_root)
    return dataset_root / CACHE_DIRNAME / "llm_fusion" / relpath.with_suffix(".json")


def _load_cached_json(cache_path: Path) -> Optional[dict]:
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None  # corrupt cache file -> treat as missing, will be re-called


def get_or_clean(
    llm_client,
    model: str,
    model_name: str,
    dataset_root: Path,
    wav_path: Path,
    raw_text: str,
    *,
    force: bool,
    stats: dict,
) -> dict:
    """Cache-or-call for the per-voter LLM cleanup step."""
    cache_path = llm_clean_cache_path(dataset_root, model_name, wav_path)

    if not force:
        cached = _load_cached_json(cache_path)
        if cached is not None:
            stats["llm_clean_cache_hits"][model_name] = stats["llm_clean_cache_hits"].get(model_name, 0) + 1
            return cached

    result = llm_refine.clean_transcript(llm_client, model, raw_text)
    stats["llm_clean_calls"][model_name] = stats["llm_clean_calls"].get(model_name, 0) + 1
    if not result.get("accepted", False):
        stats["llm_clean_rejected"][model_name] = stats["llm_clean_rejected"].get(model_name, 0) + 1
    _atomic_write_json(cache_path, result)
    return result


def get_or_fuse(
    llm_client,
    model: str,
    dataset_root: Path,
    wav_path: Path,
    rover_text: str,
    per_model_texts: dict,
    *,
    force: bool,
    stats: dict,
) -> dict:
    """Cache-or-call for the final fusion LLM step."""
    cache_path = llm_fusion_cache_path(dataset_root, wav_path)

    if not force:
        cached = _load_cached_json(cache_path)
        if cached is not None:
            stats["llm_fusion_cache_hits"] += 1
            return cached

    result = llm_refine.fuse_transcripts(llm_client, model, rover_text, per_model_texts)
    stats["llm_fusion_calls"] += 1
    if not result.get("accepted", False):
        stats["llm_fusion_rejected"] += 1
    _atomic_write_json(cache_path, result)
    return result


def combine_for_file(
    dataset_root: Path,
    wav_path: Path,
    per_model: dict[str, TranscriptionResult],
) -> dict:
    relpath = str(wav_path.relative_to(dataset_root))

    hypotheses = []
    per_model_summary = {}
    for model_name, result in per_model.items():
        per_model_summary[model_name] = {
            "text": result.text,
            "confidence": result.confidence,
            "error": result.error,
        }
        if result.error is not None or result.text is None:
            continue
        base_weight = result.confidence if result.confidence is not None else DEFAULT_CONFIDENCE
        weight = base_weight * MODEL_WEIGHT_MULTIPLIER.get(model_name, 1.0)
        hypotheses.append((model_name, tokenize(result.text), weight))

    combined = {
        "schema_version": 1,
        "audio_relpath": relpath,
        "rover_text": None,
        "voters_used": [h[0] for h in hypotheses],
        "per_model": per_model_summary,
        "wtn_columns": [],
        "combined_at": datetime.now(timezone.utc).isoformat(),
    }

    if hypotheses:
        combined_text, columns_debug = rover_combine(hypotheses)
        combined["rover_text"] = combined_text
        combined["wtn_columns"] = columns_debug

    return combined


def build_clients(selected: list[str]) -> list[BaseASRClient]:
    clients = []
    for model_name in selected:
        if model_name not in ALL_CLIENT_BUILDERS:
            raise ValueError(f"Unknown model '{model_name}'. Choices: {sorted(ALL_CLIENT_BUILDERS)}")
        clients.append(ALL_CLIENT_BUILDERS[model_name]())
    return clients


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, help="Root folder to recursively scan for .wav files")
    parser.add_argument("--language", default="vi", help="Source language hint passed to each voter (default: vi)")
    parser.add_argument(
        "--models",
        default="internal_whisper,chirp3,phowhisper",
        help="Comma-separated subset of {internal_whisper,chirp3,phowhisper} to run",
    )
    parser.add_argument("--force", action="store_true", help="Ignore cache, re-call every model for every file")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N audio files (smoke testing)")
    parser.add_argument(
        "--llm-refine",
        action="store_true",
        help=(
            "Enable Gemini (Vertex AI) post-processing: per-voter cleanup + final fusion. "
            "Requires GEMINI_SERVICE_ACCOUNT_JSON and GEMINI_PROJECT_ID env vars. "
            "Every LLM call is guarded by a before/after diff check (see llm_refine.is_safe_edit) "
            "-- an edit that's too dissimilar or too long/short than its input is rejected and "
            "the pre-LLM text is kept instead."
        ),
    )
    parser.add_argument("--llm-cleanup-model", default=llm_refine.DEFAULT_CLEANUP_MODEL)
    parser.add_argument("--llm-fusion-model", default=llm_refine.DEFAULT_FUSION_MODEL)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"--dataset-root does not exist or is not a directory: {dataset_root}")

    selected_models = [m.strip() for m in args.models.split(",") if m.strip()]
    clients = build_clients(selected_models)

    wav_files = discover_audio_files(dataset_root)
    if args.limit is not None:
        wav_files = wav_files[: args.limit]

    if not wav_files:
        raise SystemExit(f"No .wav files found under {dataset_root}")

    print(f"Dataset root : {dataset_root}")
    print(f"Models       : {', '.join(c.name for c in clients)}")
    print(f"Audio files  : {len(wav_files)}")
    print(f"LLM refine   : {'on (' + args.llm_cleanup_model + ' / ' + args.llm_fusion_model + ')' if args.llm_refine else 'off'}")

    stats = {"cache_hits": {}, "fresh_calls": {}, "errors_by_model": {}}
    llm_client = None
    if args.llm_refine:
        stats.update(
            llm_clean_cache_hits={},
            llm_clean_calls={},
            llm_clean_rejected={},
            llm_fusion_cache_hits=0,
            llm_fusion_calls=0,
            llm_fusion_rejected=0,
        )
        llm_client = llm_refine.create_client()

    output_dir = dataset_root / CACHE_DIRNAME / ROVER_OUTPUT_SUBDIR

    for wav_path in wav_files:
        relpath = wav_path.relative_to(dataset_root)
        per_model: dict[str, TranscriptionResult] = {}
        for client in clients:
            per_model[client.name] = get_or_call(
                client, dataset_root, wav_path, language=args.language, force=args.force, stats=stats
            )

        combined = combine_for_file(dataset_root, wav_path, per_model)
        final_text = combined["rover_text"]

        if args.llm_refine and llm_client is not None:
            clean_results = {}
            per_model_clean_texts = {}
            for model_name, result in per_model.items():
                if result.error is not None or not result.text:
                    continue
                clean = get_or_clean(
                    llm_client, args.llm_cleanup_model, model_name, dataset_root, wav_path, result.text,
                    force=args.force, stats=stats,
                )
                clean_results[model_name] = clean
                per_model_clean_texts[model_name] = clean["final_text"]

            fusion = None
            if combined["rover_text"]:
                fusion = get_or_fuse(
                    llm_client, args.llm_fusion_model, dataset_root, wav_path,
                    combined["rover_text"], per_model_clean_texts, force=args.force, stats=stats,
                )
                final_text = fusion["final_text"]

            combined["llm_refine"] = {
                "per_model_clean": clean_results,
                "fusion": fusion,
                "final_text": final_text,
            }

        out_path = (output_dir / relpath).with_suffix(".json")
        _atomic_write_json(out_path, combined)

        print(f"[{relpath}] -> {final_text!r}")

    run_summary = {
        "schema_version": 1,
        "dataset_root": str(dataset_root),
        "models": [c.name for c in clients],
        "n_files": len(wav_files),
        "cache_hits": stats["cache_hits"],
        "fresh_calls": stats["fresh_calls"],
        "errors_by_model": stats["errors_by_model"],
        "llm_refine": args.llm_refine,
        "run_at": datetime.now(timezone.utc).isoformat(),
    }
    if args.llm_refine:
        run_summary.update(
            llm_clean_cache_hits=stats["llm_clean_cache_hits"],
            llm_clean_calls=stats["llm_clean_calls"],
            llm_clean_rejected=stats["llm_clean_rejected"],
            llm_fusion_cache_hits=stats["llm_fusion_cache_hits"],
            llm_fusion_calls=stats["llm_fusion_calls"],
            llm_fusion_rejected=stats["llm_fusion_rejected"],
        )
    _atomic_write_json(dataset_root / CACHE_DIRNAME / "run_summary.json", run_summary)
    print("\nRun summary:")
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
