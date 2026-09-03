"""
Query/document keyword generation, Lucene indexing and retrieval evaluation.

Module map
----------
config.py      YAML loading, dotted lookups, path templating
io_utils.py    JSONL read/write, atomic writes, resume checkpoints, shards
artifacts.py   manifests and reuse validation
datasets.py    BEIR / MIRACL / BRIGHT loading, fingerprints
prompts.py     PromptSpec, PromptBuilder, per-dataset language handling
generation.py  model loading and resumable generation
indexing.py    derived corpora and Lucene index construction
retrieval.py   IndexSpec and searcher construction (BM25 / SPLADE impact)
evaluation.py  query variants and NDCG/MRR/Recall evaluation
stats.py       paired significance testing between runs
paths.py       artifact paths and per-dataset index resolution
pipeline.py    stage orchestration
"""

from .config import load_yaml
from .pipeline import run_pipeline

__all__ = ["load_yaml", "run_pipeline"]
