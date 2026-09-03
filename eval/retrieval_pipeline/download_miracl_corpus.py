#!/usr/bin/env python3
"""
Download the MIRACL passage collection from HuggingFace.

  https://huggingface.co/datasets/miracl/miracl-corpus

The corpus is stored as gzipped JSON lines, one passage per line, with the
fields docid / title / text. The docid schema is X#Y: all passages sharing X
come from the same Wikipedia article, Y is the passage index within it.

The pipeline reads these shards directly, so no conversion step is needed --
point paths.miracl_corpus_template at the per-language directory.

Examples
--------
    # one language
    python scripts/download_miracl_corpus.py --languages yo --output-dir /path/miracl/corpus

    # the small languages only (tractable for document expansion)
    python scripts/download_miracl_corpus.py --languages yo sw bn te th \
        --output-dir /path/miracl/corpus

    # see what you are about to pull before pulling it
    python scripts/download_miracl_corpus.py --languages en --list-only

Notes
-----
* miracl/miracl-corpus is Apache-2.0 and needs no authentication. The separate
  miracl/miracl topics+qrels repo does require accepting its terms first
  (`hf auth login`).
* Run this from a login/frontend node: compute nodes usually have no outbound
  network. Set HF_HOME to a work filesystem so the cache does not land in $HOME.
* Sizes vary enormously by language. English, German, French, Spanish and
  Russian are very large; yo, sw, bn, te and th are small. Check the table on
  the dataset card before pulling everything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

REPO_ID = "miracl/miracl-corpus"

LANGUAGES = [
    "ar", "bn", "de", "en", "es", "fa", "fi", "fr", "hi",
    "id", "ja", "ko", "ru", "sw", "te", "th", "yo", "zh",
]


def repo_files() -> List[str]:
    from huggingface_hub import list_repo_files

    return list_repo_files(REPO_ID, repo_type="dataset")


def files_by_language(all_files: List[str], languages: List[str]) -> Dict[str, List[str]]:
    """
    Data files live under miracl-corpus-v1.0-{lang}/. Match on the directory
    prefix, falling back to a looser match if the layout ever changes.
    """
    grouped: Dict[str, List[str]] = {}
    for lang in languages:
        prefix = f"miracl-corpus-v1.0-{lang}/"
        matches = [f for f in all_files if f.startswith(prefix)]
        if not matches:
            matches = [
                f
                for f in all_files
                if f.endswith((".jsonl.gz", ".json.gz", ".jsonl"))
                and f"-{lang}/" in f
            ]
        grouped[lang] = sorted(matches)
    return grouped


def download_language(
    lang: str,
    patterns: List[str],
    output_dir: Path,
    max_workers: int,
) -> Path:
    from huggingface_hub import snapshot_download

    target = output_dir / lang
    target.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=patterns,
        local_dir=str(target),
        max_workers=max_workers,
    )
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["yo"],
        choices=LANGUAGES + ["all"],
        help="Language codes to download, or 'all'.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Root directory; each language goes into <output-dir>/<lang>/.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="List the matching files and exit without downloading.",
    )
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    languages = LANGUAGES if "all" in args.languages else list(dict.fromkeys(args.languages))

    try:
        all_files = repo_files()
    except ImportError:
        sys.exit("huggingface_hub is required: pip install huggingface_hub")

    grouped = files_by_language(all_files, languages)

    for lang in languages:
        matches = grouped[lang]
        print(f"\n{lang}: {len(matches)} file(s)")
        for name in matches[:5]:
            print(f"  {name}")
        if len(matches) > 5:
            print(f"  ... and {len(matches) - 5} more")

        if not matches:
            print(f"  [warn] nothing matched for {lang}; check the repo layout")

    if args.list_only:
        return

    if args.output_dir is None:
        sys.exit("--output-dir is required unless --list-only is set")

    for lang in languages:
        matches = grouped[lang]
        if not matches:
            continue
        print(f"\n[download] {lang} -> {args.output_dir / lang}")
        target = download_language(
            lang, matches, args.output_dir, max_workers=args.max_workers
        )
        shards = sorted(target.rglob("*.jsonl.gz")) + sorted(target.rglob("*.jsonl"))
        print(f"[done] {lang}: {len(shards)} shard(s) under {target}")


if __name__ == "__main__":
    main()
