"""Publish a chosen test-set version to Hugging Face.

Generalizes what used to be three copy-pasted, direction-specific notebook
cells (mục 10/10.5/10.6 of `TestSet_construction copy 12.ipynb`) into
path/repo-parameterized functions. Tokens are never embedded here -- the
notebook keeps them hardcoded (this file never gets pushed to git anyway;
`src/` does, so no secret ever enters version control) and passes them in
as arguments.
"""

from __future__ import annotations

import re
from pathlib import Path


def publish_test_set(
    samples: list[dict],
    repo_id: str,
    *,
    token: str,
    private: bool,
    split: str = "test",
) -> str:
    """Push `samples` (each with `id, source_dataset, speaker_id,
    duration_sec, text_vi, text_en, audio_path`) as an HF `Dataset`, audio
    embedded directly into Parquet via `datasets.Audio()`. Always an
    explicit `split` -- `push_to_hub()` defaults to "train" if omitted,
    which is wrong for a test-only dataset (the exact bug this project hit
    and had to clean up manually before this function existed).
    """
    from datasets import Audio, Dataset
    from huggingface_hub import login

    login(token=token)

    rows = [
        {
            "id": s["id"],
            "source_dataset": s["source_dataset"],
            "speaker_id": s.get("speaker_id"),
            "duration_sec": s.get("duration_sec"),
            "text_vi": s["text_vi"],
            "text_en": s["text_en"],
            "audio": s["audio_path"],  # path -- Dataset reads the real file + embeds bytes on push
        }
        for s in samples
    ]
    ds = Dataset.from_list(rows)
    ds = ds.cast_column("audio", Audio())
    ds.push_to_hub(repo_id, private=private, split=split)
    return f"https://huggingface.co/datasets/{repo_id}"


def upload_json_artifact(
    local_path: Path,
    repo_id: str,
    path_in_repo: str,
    *,
    token: str,
    private: bool = True,
) -> str:
    """Upload a single JSON file (e.g. an eval-results cache) to a dataset
    repo, creating it first if needed. Used today for
    `internal_ast_eval_results.json` cross-notebook (Colab -> Kaggle)
    caching -- generic enough for any single-artifact upload."""
    from huggingface_hub import HfApi, login

    login(token=token)
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
    )
    return f"https://huggingface.co/datasets/{repo_id}/blob/main/{path_in_repo}"


def strip_stale_split_from_readme(repo_id: str, *, token: str, stale_split: str = "train") -> None:
    """Regex-patch a leftover `stale_split` declaration out of a dataset
    repo's `README.md` YAML frontmatter (both the `dataset_info.splits`
    entry and the `configs[].data_files` entry). Needed because deleting a
    Parquet split's files via the Hub API does NOT auto-update this
    metadata, and a stale declaration referencing zero real files breaks
    `datasets.load_dataset()` for anyone downstream even though the actual
    data files are correct.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    readme_path = Path(api.hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="README.md"))
    content = readme_path.read_text(encoding="utf-8")

    content = re.sub(
        rf"  - name: {re.escape(stale_split)}\n    num_bytes: [\d.]+\n    num_examples: \d+\n", "", content
    )
    content = re.sub(
        rf"  - split: {re.escape(stale_split)}\n    path: data/{re.escape(stale_split)}-\*\n", "", content
    )

    fixed_path = readme_path.parent / "README_fixed.md"
    fixed_path.write_text(content, encoding="utf-8")
    api.upload_file(
        path_or_fileobj=str(fixed_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
