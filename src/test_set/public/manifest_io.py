"""I/O + resume-safety for `tmp.json`, the single continuous working file
for the test-set-construction pipeline (`TestSet_construction copy 12.ipynb`).

Replaces the notebook's old `manifest.jsonl` (JSONL, one dict per line) with
one JSON object -- `{"schema_version", "direction", "generation_mode",
"updated_at", "samples": {sample_id: {...}}}` -- since the new pipeline
needs `ver_N.json` version snapshots to have the exact same structure as the
live working file, and JSONL doesn't have a natural single-document shape
for that. The resume-safety properties `manifest.jsonl` had (atomic write,
safe to re-run any cell after a mid-pipeline failure/Colab disconnect
without losing other cells' progress) are preserved exactly, just
re-pointed at this new format.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

SCHEMA_VERSION = 1


def new_envelope(direction: str, generation_mode: str) -> dict:
    """A fresh, empty tmp.json envelope for a given direction/mode."""
    return {
        "schema_version": SCHEMA_VERSION,
        "direction": direction,
        "generation_mode": generation_mode,
        "updated_at": None,
        "samples": {},
    }


def load_tmp(path: Path) -> dict:
    """Load tmp.json, or None if it doesn't exist yet (caller decides
    whether that's an error or a reason to call `new_envelope()`)."""
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_tmp(path: Path, envelope: dict) -> None:
    """Atomic write: write to a `.tmp` sibling, then `os.replace()` -- a
    crash/interrupt mid-write can never leave `tmp.json` truncated/corrupt,
    same guarantee `manifest.jsonl`'s `save_manifest()` had."""
    envelope["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(envelope, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def reload_tmp_inplace(path: Path, envelope: dict) -> dict:
    """Re-read tmp.json from disk and merge it into the in-memory
    `envelope`'s `samples` dict, UPDATING EACH ROW-DICT IN PLACE (not
    replacing `envelope` with a new object) -- so any other code still
    holding a reference to a specific sample-row dict keeps seeing live
    data. Call this at the top of every processing cell before computing
    "pending" work, exactly like `manifest.jsonl`'s
    `reload_manifest_inplace()` -- this is what makes a cell safe to
    re-run standalone after a failure/disconnect without clobbering
    progress another cell already saved to disk.
    """
    fresh = load_tmp(path)
    if fresh is None:
        return envelope
    for sample_id, fresh_row in fresh.get("samples", {}).items():
        if sample_id in envelope["samples"]:
            envelope["samples"][sample_id].update(fresh_row)
        else:
            envelope["samples"][sample_id] = fresh_row
    return envelope


# ---------------------------------------------------------------------------
# Final-field derivation -- uniform view for normalize/validate/version/publish
# ---------------------------------------------------------------------------

def derive_final_fields(sample: dict, *, direction: str, generation_mode: str) -> None:
    """Populate `text_vi`/`text_en` (in place) from whichever raw fields the
    active direction/generation_mode produced, so every downstream stage
    (normalize, validate, version, publish) can read a uniform
    `text_vi`/`text_en` pair regardless of how the row was generated:

    - `heavy_pipeline` (either direction): `final_asr_text`/`final_mt_text`
      (post human-review) map onto whichever of `text_vi`/`text_en` matches
      the audio's actual source language.
    - `phost` (en_vi only): `text_vi`/`text_en` are already final -- no
      mapping needed, this is a no-op if both are already set.
    """
    if generation_mode == "phost":
        return  # already final, nothing to derive
    if generation_mode != "heavy_pipeline":
        raise ValueError(f"Unknown generation_mode {generation_mode!r}")

    final_source_text = sample.get("final_asr_text")
    final_target_text = sample.get("final_mt_text")
    if direction == "vi_en":
        sample["text_vi"] = final_source_text
        sample["text_en"] = final_target_text
    elif direction == "en_vi":
        sample["text_en"] = final_source_text
        sample["text_vi"] = final_target_text
    else:
        raise ValueError(f"Unknown direction {direction!r}")


# ---------------------------------------------------------------------------
# Bridge to prepare_data.py (unchanged, still expects manifest.jsonl-shaped
# JSONL input) -- exports a temporary JSONL view of tmp.json's samples so the
# labeling-app review round-trip keeps working without prepare_data.py
# needing to learn the new tmp.json envelope format.
# ---------------------------------------------------------------------------

def export_samples_jsonl(envelope: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in envelope["samples"].values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def import_csv_overlay(envelope: dict, csv_path: Path, fields: tuple[str, ...]) -> int:
    """Overlay reviewer-edited columns from a `mt_manual_check.csv`-shaped
    CSV (or the labeling app's exported equivalent) onto `envelope`'s
    samples -- the CSV always wins (authoritative human-reviewed data),
    same semantics as the old manifest pipeline's mục-0.5/mục-5 CSV merge.
    Returns the number of samples updated.
    """
    import pandas as pd

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    n_updated = 0
    for _, row in df.iterrows():
        sample_id = row["id"]
        sample = envelope["samples"].get(sample_id)
        if sample is None:
            continue
        touched = False
        for field in fields:
            value = row.get(field)
            if pd.notna(value) and str(value).strip():
                sample[field] = value
                touched = True
        if touched:
            n_updated += 1
    return n_updated
