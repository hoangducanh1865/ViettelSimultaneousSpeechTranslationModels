"""Crawl YouTube videos into 16kHz mono wav + Vietnamese/English subtitles.

Wraps the yt-dlp invocation documented in Readme.md as a callable script
(Colab or local) instead of a copy-pasted shell command. Each line of
--urls-file may be a single video URL or a playlist URL -- yt-dlp expands
playlists on its own.

Usage:
    python crawl.py --urls-file urls.txt --output-dir dataset/raw_audio
    python crawl.py --urls-file urls.txt --output-dir dataset/raw_audio \
        --cookies cookies.txt   # only if YouTube starts rate-limiting/blocking
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def build_yt_dlp_command(
    urls_file: Path,
    output_dir: Path,
    *,
    cookies: Path | None,
    sleep_requests: float,
    sleep_interval: float,
    max_sleep_interval: float,
) -> list[str]:
    cmd = ["yt-dlp"]
    if cookies is not None:
        cmd += ["--cookies", str(cookies)]
    cmd += [
        "--js-runtimes", "deno",
        "--sleep-requests", str(sleep_requests),
        "--sleep-interval", str(sleep_interval),
        "--max-sleep-interval", str(max_sleep_interval),
        "-x",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--postprocessor-args", "ExtractAudio:-ar 16000 -ac 1",
        "--write-sub",
        "--write-auto-sub",
        "--sub-lang", "vi,en",
        "--sub-format", "srt/vtt",
        "-o", str(output_dir / "%(id)s.%(ext)s"),
        "-o", f"subtitle:{output_dir / '%(id)s.%(ext)s'}",
        "-a", str(urls_file),
    ]
    return cmd


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urls-file", type=Path, default=Path("urls.txt"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cookies", type=Path, default=None,
        help="Path to a cookies.txt (Netscape format, manually exported from a "
        "logged-in browser session). Only needed if YouTube starts returning "
        "429s / blocking requests -- most public videos don't need it.",
    )
    parser.add_argument("--sleep-requests", type=float, default=1.0)
    parser.add_argument("--sleep-interval", type=float, default=10.0)
    parser.add_argument("--max-sleep-interval", type=float, default=30.0)
    args = parser.parse_args(argv)

    if not args.urls_file.exists():
        raise SystemExit(f"--urls-file not found: {args.urls_file}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_yt_dlp_command(
        args.urls_file,
        args.output_dir,
        cookies=args.cookies,
        sleep_requests=args.sleep_requests,
        sleep_interval=args.sleep_interval,
        max_sleep_interval=args.max_sleep_interval,
    )
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"yt-dlp exited with code {result.returncode}")

    print(f"Done. Output in {args.output_dir}")


if __name__ == "__main__":
    main()
