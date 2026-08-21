"""Produce final (text_en, text_vi) ground truth for the segments
process_videos.py cut, using the SAME ASR+ROVER+LLM-cleanup pipeline as the
Vietnamese-audio side of this repo (rover.py), then LLM-translating that
clean English text to Vietnamese -- rather than trusting the crawled
caption tracks' own time-alignment (see process_videos.py's docstring for
why that alignment turned out to be unreliable).

Run rover.py FIRST, pointed at this dataset's audio tree, e.g.:
    python rover.py --dataset-root {OUT_DIR}/audio --language en \\
        --models internal_whisper,chirp3 --llm-refine
(no --translate: it's hardcoded VI->EN, see rover.py's guard)

Then run this script:
    python translate_en_to_vi.py --out-dir {OUT_DIR}
    python translate_en_to_vi.py --out-dir {OUT_DIR} --limit 5   # smoke test

For each manifest row, the English source text is rover_output's final ASR
text (llm_refine.final_text, falling back to rover_text) when available;
if a segment has no rover_output (e.g. the language filter in rover.py
dropped it, or rover.py hasn't been run on it yet), falls back to the
segment's own crawled caption text (text_en_caption) so nothing is silently
skipped -- callers can tell which happened from each row's "text_en_source".
The crawled Vietnamese caption for that segment (text_vi_caption_hint, if
any -- loosely time-aligned, may not exactly match) is passed to the LLM as
a style/terminology reference, never as the thing being translated.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "public"))
import llm_refine  # noqa: E402

CACHE_DIRNAME = "rover_cache"  # matches rover.py's own constant/layout


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def load_manifest_rows(out_dir: Path) -> list[dict]:
    manifest_path = out_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise SystemExit(f"Không tìm thấy {manifest_path} -- chạy process_videos.py trước.")
    with open(manifest_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_rover_output(audio_root: Path, row: dict) -> Optional[dict]:
    rover_output_path = (
        audio_root / CACHE_DIRNAME / "rover_output" / row["video_id"] / f"{row['id']}.json"
    )
    if not rover_output_path.exists():
        return None
    with open(rover_output_path, encoding="utf-8") as f:
        return json.load(f)


def resolve_english_source_text(row: dict, rover_output: Optional[dict]) -> tuple[str, str]:
    """Returns (text, source) where source is "rover" or "caption_fallback"."""
    if rover_output is not None:
        llm = rover_output.get("llm_refine") or {}
        text = llm.get("final_text") or rover_output.get("rover_text")
        if text:
            return text, "rover"
    return row.get("text_en_caption") or "", "caption_fallback"


def translate_cache_path(audio_root: Path, row: dict) -> Path:
    return audio_root / CACHE_DIRNAME / "llm_translate_en_vi" / row["video_id"] / f"{row['id']}.json"


def get_or_translate_segment(
    llm_client, model: str, audio_root: Path, row: dict, *, force: bool,
) -> dict:
    cache_path = translate_cache_path(audio_root, row)
    if not force and cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    rover_output = load_rover_output(audio_root, row)
    text_en, text_en_source = resolve_english_source_text(row, rover_output)

    translation = llm_refine.translate_text_en_vi(
        llm_client, model, text_en, reference_hint=row.get("text_vi_caption_hint"),
    )

    result = {
        "id": row["id"],
        "video_id": row["video_id"],
        "text_en": text_en,
        "text_en_source": text_en_source,
        "text_vi": translation.get("final_text"),
        "translate_accepted": translation.get("accepted", False),
        "translate_error": translation.get("error"),
        "reference_hint_used": row.get("text_vi_caption_hint"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(cache_path, result)
    return result


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True, help="Same --out-dir passed to process_videos.py")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N segments (smoke testing)")
    parser.add_argument("--force", action="store_true", help="Ignore cache, re-translate every segment")
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--llm-translate-model", default=llm_refine.DEFAULT_TRANSLATE_MODEL)
    args = parser.parse_args(argv)

    audio_root = args.out_dir / "audio"
    rows = load_manifest_rows(args.out_dir)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"manifest.jsonl rỗng trong {args.out_dir}")

    print(f"Segments      : {len(rows)}")
    print(f"Translate model: {args.llm_translate_model}")

    llm_client = llm_refine.create_client()

    lock = threading.Lock()
    stats = {"cache_hits": 0, "fresh_calls": 0, "accepted": 0, "rejected": 0, "caption_fallback": 0}
    results = []

    def _run(row):
        cache_path = translate_cache_path(audio_root, row)
        was_cached = cache_path.exists() and not args.force
        result = get_or_translate_segment(llm_client, args.llm_translate_model, audio_root, row, force=args.force)
        with lock:
            stats["cache_hits" if was_cached else "fresh_calls"] += 1
            stats["accepted" if result["translate_accepted"] else "rejected"] += 1
            if result["text_en_source"] == "caption_fallback":
                stats["caption_fallback"] += 1
        return result

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(_run, row): row for row in rows}
        for i, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"[{i}/{len(rows)}] {row['id']}: LỖI {e!r}")
                continue
            if i % 50 == 0 or i == len(rows):
                print(f"[{i}/{len(rows)}] ...")

    out_path = args.out_dir / "translated_manifest.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in sorted(results, key=lambda r: r["id"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\nStats:", json.dumps(stats, ensure_ascii=False))
    print(f"Đã lưu {len(results)} bản dịch vào {out_path}")


if __name__ == "__main__":
    main()
