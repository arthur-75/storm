#!/usr/bin/env python3
"""
Stage-based generation, indexing and evaluation pipeline.

Usage:
    python run_retrieval_pipeline.py --config configs/experiment.yaml
    python run_retrieval_pipeline.py --config configs/experiment.yaml --dry-run

Sharded corpus generation:
    python run_retrieval_pipeline.py --config c.yaml --num-shards 4 --shard-id 0
    ...
    python run_retrieval_pipeline.py --config c.yaml --num-shards 4 --merge-corpus-shards
"""

from __future__ import annotations

import argparse
from pathlib import Path

from retrieval_pipeline import load_yaml, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate query/corpus keywords, build Lucene indexes and evaluate "
            "retrieval on BEIR, MIRACL or BRIGHT."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="YAML configuration file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate inputs and print the execution plan without "
            "generation/indexing/evaluation"
        ),
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of corpus-generation shards.",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="Zero-based shard assigned to this process.",
    )
    parser.add_argument(
        "--merge-corpus-shards",
        action="store_true",
        help=(
            "Merge completed corpus shard files into corpus.jsonl before "
            "indexing and evaluation."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    run_pipeline(
        cfg,
        dry_run=args.dry_run,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        merge_corpus_parts=args.merge_corpus_shards,
    )


if __name__ == "__main__":
    main()
