"""Turn crawled videos (raw wav + vi/en subtitles) into a segment-level
(audio, text_en, text_vi) dataset -- no ASR, no LLM: both texts come
directly from the video's own captions.

Segmentation is driven by the English subtitle cues (merged up to a target
duration), not by speaker diarization: the captions already mark exactly
when speech happens. For each resulting segment window, the Vietnamese
text is whatever Vietnamese cues overlap that same time range -- this is a
time-overlap join, so the two languages' cue boundaries don't need to
match. Each cut segment is then checked with the existing lang_id module
(English-audio quality gate) before being kept.

Usage:
    python process_videos.py --raw-dir dataset/raw_audio --out-dir dataset/final
    python process_videos.py --raw-dir ... --out-dir ... --limit 2   # smoke test
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import soundfile as sf
import webvtt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "public"))
import lang_id  # noqa: E402

SAMPLE_RATE = 16000
SUBTITLE_EXTENSIONS = (".srt", ".vtt")


@dataclass
class Cue:
    start: float
    end: float
    text: str


def find_subtitle_file(search_dir: Path, video_id: str, lang: str) -> Optional[Path]:
    """yt-dlp writes subtitles as {video_id}.{lang}.{srt,vtt} (whichever
    format was available -- --sub-format "srt/vtt" prefers srt). Manual
    subs are written in preference to auto ones for a given language, so
    there's normally exactly one file per (video_id, lang). Looks in
    `search_dir` only (the wav's own parent directory) -- crawl.py's own
    output is flat, but this stays correct if the whole run got uploaded
    a directory level off (e.g. --output-dir raw_audio uploaded as a
    subfolder instead of its contents being uploaded directly)."""
    candidates = [
        p for ext in SUBTITLE_EXTENSIONS
        for p in search_dir.glob(f"{video_id}.{lang}{ext}")
    ]
    return candidates[0] if candidates else None


def _timestamp_to_seconds(ts) -> float:
    # webvtt-py's own Caption.start_in_seconds/end_in_seconds (and
    # Timestamp.in_seconds()) silently drop the milliseconds component --
    # fine for display, not for segment-boundary math -- so compute the
    # precise value from the timestamp's fields directly instead.
    return ts.hours * 3600 + ts.minutes * 60 + ts.seconds + ts.milliseconds / 1000.0


def load_cues(path: Path) -> list[Cue]:
    vtt = webvtt.from_srt(str(path)) if path.suffix.lower() == ".srt" else webvtt.read(str(path))
    return [
        Cue(
            start=_timestamp_to_seconds(c.start_time),
            end=_timestamp_to_seconds(c.end_time),
            text=c.text.strip(),
        )
        for c in vtt
        if c.text and c.text.strip()
    ]


def _dedupe_join(texts: list[str]) -> str:
    """Join consecutive cue texts, collapsing YouTube auto-caption's rolling
    display (each new cue often repeats the tail of the previous one plus a
    few new words) down to just the incremental content. Uses the longest
    matching block between the accumulated tail and the next cue's head;
    falls back to a plain space-join when there's no meaningful overlap
    (the normal case for manual/official subtitles, which aren't rolling)."""
    if not texts:
        return ""
    joined = texts[0]
    for text in texts[1:]:
        if not text:
            continue
        tail = joined[-120:]
        matcher = difflib.SequenceMatcher(None, tail, text, autojunk=False)
        match = matcher.find_longest_match(0, len(tail), 0, len(text))
        if match.size >= 8 and match.b == 0:
            # `text` starts by repeating the end of what we already have --
            # keep only the part after that repeated stretch.
            remainder = text[match.size:].strip()
            if remainder:
                joined = f"{joined} {remainder}"
        else:
            joined = f"{joined} {text}".strip()
    return " ".join(joined.split())


def merge_cues_into_segments(
    en_cues: list[Cue], *, target_max_sec: float = 20.0, min_gap_to_merge: float = 0.5,
) -> list[dict]:
    """Greedily merge consecutive English cues into segment windows, closing
    a segment once the next cue would push it past target_max_sec, or once
    the gap to the next cue exceeds min_gap_to_merge (a real pause, likely a
    natural break)."""
    if not en_cues:
        return []

    segments: list[dict] = []
    seg_start = en_cues[0].start
    seg_end = en_cues[0].end

    for cue in en_cues[1:]:
        gap = cue.start - seg_end
        would_span = cue.end - seg_start
        if gap > min_gap_to_merge or would_span > target_max_sec:
            segments.append({"start": seg_start, "end": seg_end})
            seg_start = cue.start
            seg_end = cue.end
        else:
            seg_end = cue.end
    segments.append({"start": seg_start, "end": seg_end})
    return segments


def merge_short_trailing_segments(
    segments: list[dict], *, min_seg_sec: float, target_max_sec: float,
) -> list[dict]:
    """Fold a too-short segment into its predecessor when that wouldn't
    exceed target_max_sec, instead of just dropping it outright."""
    merged: list[dict] = []
    for seg in segments:
        duration = seg["end"] - seg["start"]
        if (
            duration < min_seg_sec
            and merged
            and (seg["end"] - merged[-1]["start"]) <= target_max_sec
        ):
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(dict(seg))
    return merged


def extract_overlapping_text(cues: list[Cue], seg_start: float, seg_end: float) -> str:
    overlapping = [c.text for c in cues if c.start < seg_end and c.end > seg_start]
    return _dedupe_join(overlapping)


def cut_and_check_segment(
    audio, sr: int, seg_start: float, seg_end: float, out_path: Path,
) -> dict:
    start_sample = max(0, int(seg_start * sr))
    end_sample = min(len(audio), int(seg_end * sr))
    clip = audio[start_sample:end_sample]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, clip, sr)

    verdict = lang_id.audio_language_verdict(out_path, expected_language="en")
    return {
        "keep": verdict["keep"],
        "lang_id_top_langs": verdict["segment_top_langs"],
        "lang_id_error": verdict["error"],
    }


def process_one_video(
    video_id: str,
    wav_path: Path,
    out_dir: Path,
    *,
    target_max_sec: float,
    min_seg_sec: float,
    source_url: Optional[str] = None,
) -> tuple[list[dict], dict]:
    stats = {
        "video_id": video_id,
        "candidate_segments": 0,
        "dropped_missing_vi_text": 0,
        "dropped_lang_id_fail": 0,
        "kept": 0,
    }

    # Subtitles are looked up next to the wav itself, not a fixed raw_dir --
    # tolerates the crawl output having been uploaded a directory level off.
    sub_dir = wav_path.parent
    en_sub_path = find_subtitle_file(sub_dir, video_id, "en")
    vi_sub_path = find_subtitle_file(sub_dir, video_id, "vi")

    if not wav_path.exists() or en_sub_path is None or vi_sub_path is None:
        stats["error"] = (
            f"missing required file(s): wav={wav_path.exists()}, "
            f"en_sub={en_sub_path is not None}, vi_sub={vi_sub_path is not None}"
        )
        return [], stats

    en_cues = load_cues(en_sub_path)
    vi_cues = load_cues(vi_sub_path)

    segments = merge_cues_into_segments(en_cues, target_max_sec=target_max_sec)
    segments = merge_short_trailing_segments(
        segments, min_seg_sec=min_seg_sec, target_max_sec=target_max_sec
    )
    stats["candidate_segments"] = len(segments)

    audio, sr = sf.read(wav_path, dtype="float32")
    if sr != SAMPLE_RATE:
        raise RuntimeError(f"{wav_path}: expected {SAMPLE_RATE}Hz, got {sr}Hz")

    rows = []
    for idx, seg in enumerate(segments):
        seg_start, seg_end = seg["start"], seg["end"]
        if seg_end - seg_start < min_seg_sec:
            continue

        text_en = extract_overlapping_text(en_cues, seg_start, seg_end)
        text_vi = extract_overlapping_text(vi_cues, seg_start, seg_end)
        if not text_vi:
            stats["dropped_missing_vi_text"] += 1
            continue

        seg_id = f"{video_id}_seg{idx:04d}"
        out_wav = out_dir / "audio" / video_id / f"{seg_id}.wav"
        check = cut_and_check_segment(audio, sr, seg_start, seg_end, out_wav)

        if not check["keep"]:
            stats["dropped_lang_id_fail"] += 1
            out_wav.unlink(missing_ok=True)
            continue

        stats["kept"] += 1
        rows.append({
            "id": seg_id,
            "video_id": video_id,
            "segment_index": idx,
            "start": round(seg_start, 3),
            "end": round(seg_end, 3),
            "duration": round(seg_end - seg_start, 3),
            "text_en": text_en,
            "text_vi": text_vi,
            "audio_filepath": str(out_wav.relative_to(out_dir)),
            "source_url": source_url,
            "en_sub_ext": en_sub_path.suffix.lstrip("."),
            "vi_sub_ext": vi_sub_path.suffix.lstrip("."),
            "lang_id_top_langs": check["lang_id_top_langs"],
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

    return rows, stats


def load_processed_video_ids(out_dir: Path) -> set[str]:
    path = out_dir / "processed_videos.json"
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return set(json.load(f))


def save_processed_video_ids(out_dir: Path, video_ids: set[str]) -> None:
    with open(out_dir / "processed_videos.json", "w", encoding="utf-8") as f:
        json.dump(sorted(video_ids), f, ensure_ascii=False, indent=2)


def discover_wav_files(raw_dir: Path) -> list[Path]:
    # Recursive on purpose: crawl.py's own output is flat, but an upload
    # can easily land the whole output folder one level deeper (e.g. the
    # local raw_audio/ folder itself dragged into Drive instead of just
    # its contents) -- searching the whole tree tolerates that.
    return sorted(raw_dir.rglob("*.wav"))


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target-max-sec", type=float, default=20.0)
    parser.add_argument("--min-seg-sec", type=float, default=1.5)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N videos (smoke testing)")
    parser.add_argument("--force", action="store_true", help="Reprocess videos even if already in processed_videos.json")
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.jsonl"

    wav_files = discover_wav_files(args.raw_dir)
    wav_by_id: dict[str, Path] = {}
    for p in wav_files:
        if p.stem in wav_by_id and wav_by_id[p.stem] != p:
            print(f"WARNING: duplicate video id '{p.stem}' found at both "
                  f"{wav_by_id[p.stem]} and {p} -- keeping the first one found.")
            continue
        wav_by_id[p.stem] = p
    video_ids = sorted(wav_by_id)

    if args.limit is not None:
        video_ids = video_ids[: args.limit]
    if not video_ids:
        raise SystemExit(f"No {{id}}.wav files found under {args.raw_dir}")

    processed = set() if args.force else load_processed_video_ids(args.out_dir)
    pending = [v for v in video_ids if v not in processed]
    print(f"{len(video_ids) - len(pending)}/{len(video_ids)} video(s) already processed, {len(pending)} to do.")

    all_stats = []
    with open(manifest_path, "a", encoding="utf-8") as manifest_f:
        for i, video_id in enumerate(pending, 1):
            print(f"[{i}/{len(pending)}] {video_id}")
            rows, stats = process_one_video(
                video_id, wav_by_id[video_id], args.out_dir,
                target_max_sec=args.target_max_sec, min_seg_sec=args.min_seg_sec,
            )
            for row in rows:
                manifest_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            manifest_f.flush()

            all_stats.append(stats)
            processed.add(video_id)
            save_processed_video_ids(args.out_dir, processed)

            print(
                f"  candidates={stats['candidate_segments']} kept={stats.get('kept', 0)} "
                f"dropped_missing_vi={stats.get('dropped_missing_vi_text', 0)} "
                f"dropped_lang_id={stats.get('dropped_lang_id_fail', 0)}"
                + (f"  ERROR: {stats['error']}" if "error" in stats else "")
            )

    report_path = args.out_dir / "build_report.json"
    per_video_by_id: dict[str, dict] = {}
    if report_path.exists():
        with open(report_path, encoding="utf-8") as f:
            for s in json.load(f).get("per_video", []):
                per_video_by_id[s["video_id"]] = s
    for s in all_stats:
        per_video_by_id[s["video_id"]] = s
    merged_stats = list(per_video_by_id.values())

    report = {
        "schema_version": 1,
        "n_videos": len(merged_stats),
        "totals": {
            key: sum(s.get(key, 0) for s in merged_stats)
            for key in ("candidate_segments", "dropped_missing_vi_text", "dropped_lang_id_fail", "kept")
        },
        "per_video": merged_stats,
        "run_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\nTotals:", json.dumps(report["totals"], ensure_ascii=False))
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
