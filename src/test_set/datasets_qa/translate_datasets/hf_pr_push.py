"""Push one or more local files to a HuggingFace dataset repo as a Pull Request.

Extracted from noteboooks/benchmark_qa/Benchmarck_QA_contructer copy 2.ipynb.

Disables the Xet upload backend before pushing: Xet requires DIRECT write access to the repo
even when create_pr=True, so an account that only has PR-creation rights (not a collaborator)
gets a 403 on "xet-write-token" for any file large enough to go through Xet. Falling back to
plain LFS supports creating a PR without direct write access.

Usage (repeat --file for each local:remote pair):
    python hf_pr_push.py \\
        --token hf_xxx \\
        --repo-id anhnbd2005/Vietnamese-Speech-QA \\
        --file /path/to/test_sound.jsonl=test_sound.jsonl \\
        --file /path/to/sound.zip=sound.zip \\
        --commit-message "Add test_sound.jsonl + sound.zip"
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

# PHẢI set TRƯỚC khi import huggingface_hub / gọi upload -- is_xet_available() có thể bị
# memoize ngay khi thư viện được import lần đầu trong session, set muộn hơn sẽ không có tác dụng.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def push_files(
    token: str, repo_id: str, file_pairs: list[tuple[str, str]], *,
    repo_type: str = "dataset", commit_message: Optional[str] = None,
) -> list[str]:
    """file_pairs: list of (local_path, path_in_repo). Trả về danh sách PR url."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    print("Đẩy PR bằng tài khoản:", api.whoami()["name"])

    pr_urls = []
    for local_path, path_in_repo in file_pairs:
        commit_info = api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=repo_type,
            create_pr=True,
            commit_message=commit_message or f"Add {path_in_repo}",
        )
        print(f"Đã tạo PR cho {path_in_repo}: {commit_info.pr_url}")
        pr_urls.append(commit_info.pr_url)

    print(f"\nXong -- chờ chủ sở hữu {repo_id} review & merge {len(pr_urls)} PR trên.")
    return pr_urls


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--token", required=True, help="HF write token CỦA BẠN (không cần quyền ghi trực tiếp repo).")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument(
        "--file", action="append", required=True, dest="files",
        help='Dạng "local_path=path_in_repo", lặp lại cờ này cho mỗi file cần đẩy.',
    )
    parser.add_argument("--commit-message", default=None)
    args = parser.parse_args(argv)

    file_pairs = []
    for entry in args.files:
        local_path, _, path_in_repo = entry.partition("=")
        if not path_in_repo:
            raise SystemExit(f"--file phải có dạng 'local_path=path_in_repo', nhận được: {entry!r}")
        file_pairs.append((local_path, path_in_repo))

    push_files(args.token, args.repo_id, file_pairs, repo_type=args.repo_type, commit_message=args.commit_message)


if __name__ == "__main__":
    main()
