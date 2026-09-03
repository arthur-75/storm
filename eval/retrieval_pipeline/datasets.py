"""
Dataset loading.

Three dataset families are supported:

  beir    local BEIR folder (corpus + queries + qrels)  -> full pipeline
  miracl  queries + qrels only (JSON files or pyserini) -> query generation only
  bright  queries + qrels only (pyserini topics/qrels)  -> query generation only

MIRACL and BRIGHT ship no local corpus here, so document-side generation and
the `keywords` / `keywords_plus_original` indexes are not available for them.
The pipeline detects that from `DatasetBundle.has_corpus`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .config import deep_get, format_path
from .io_utils import load_corpus_any, load_json, normalize_text

# ---------------------------------------------------------------------------
# Known dataset families
# ---------------------------------------------------------------------------

MIRACL_LANGUAGE_NAMES: Dict[str, str] = {
    "ar": "Arabic",
    "bn": "Bengali",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "hi": "Hindi",
    "id": "Indonesian",
    "ja": "Japanese",
    "ko": "Korean",
    "ru": "Russian",
    "sw": "Swahili",
    "te": "Telugu",
    "th": "Thai",
    "yo": "Yoruba",
    "zh": "Chinese",
}

BRIGHT_DATASETS = frozenset(
    {
        "bright-biology",
        "bright-earth-science",
        "bright-economics",
        "bright-psychology",
        "bright-robotics",
        "bright-stackoverflow",
        "bright-sustainable-living",
        "bright-pony",
        "bright-leetcode",
        "bright-aops",
        "bright-theoremqa-theorems",
        "bright-theoremqa-questions",
    }
)

VALID_DATASET_KINDS = {"beir", "miracl", "bright"}

DEFAULT_MIRACL_TOPICS_TEMPLATE = "{miracl_root}/topics/{lang}/dev_queries.json"
DEFAULT_MIRACL_QRELS_TEMPLATE = "{miracl_root}/topics/{lang}/dev_qrels.json"


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------

@dataclass
class DatasetBundle:
    dataset: str
    split: str
    kind: str
    queries: Dict[str, str]
    qrels: Dict[str, Dict[str, int]]
    corpus: Optional[Dict[str, Dict[str, Any]]] = None
    language: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_corpus(self) -> bool:
        return bool(self.corpus)

    @property
    def language_name(self) -> Optional[str]:
        if self.language is None:
            return None
        return MIRACL_LANGUAGE_NAMES.get(self.language)

    @property
    def supports_self_docid_filter(self) -> bool:
        """
        BEIR indexes can contain a document whose id equals the query id
        (quora, arguana, ...). MIRACL/BRIGHT ids live in different spaces,
        so filtering there would silently drop correct hits.
        """
        return self.kind == "beir"

    @property
    def corpus_text(self) -> Dict[str, str]:
        if not self.corpus:
            return {}
        return {
            doc_id: normalize_text(f"{row.get('title', '')} {row.get('text', '')}")
            for doc_id, row in self.corpus.items()
        }


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------

def fingerprint_text_map(records: Mapping[str, str]) -> str:
    h = hashlib.sha256()
    for record_id in sorted(records):
        h.update(record_id.encode("utf-8"))
        h.update(b"\0")
        h.update(normalize_text(records[record_id]).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def fingerprint_corpus(corpus: Mapping[str, Mapping[str, Any]]) -> str:
    h = hashlib.sha256()
    for doc_id in sorted(corpus):
        row = corpus[doc_id]
        h.update(doc_id.encode("utf-8"))
        h.update(b"\0")
        h.update(normalize_text(row.get("title", "")).encode("utf-8"))
        h.update(b"\0")
        h.update(normalize_text(row.get("text", "")).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Kind detection
# ---------------------------------------------------------------------------

def detect_dataset_kind(cfg: Mapping[str, Any], dataset: str) -> str:
    """
    Resolution order:
      1. explicit override in data.dataset_kinds
      2. name-based detection (bright-*, miracl-*)
      3. data.miracl: true  -> bare language codes are MIRACL (old script style)
      4. beir
    """
    overrides = deep_get(cfg, "data.dataset_kinds", {}) or {}
    if isinstance(overrides, Mapping) and dataset in overrides:
        kind = str(overrides[dataset]).lower()
        if kind not in VALID_DATASET_KINDS:
            raise ValueError(
                f"data.dataset_kinds[{dataset}] must be one of "
                f"{sorted(VALID_DATASET_KINDS)}, got {kind!r}"
            )
        return kind

    if dataset in BRIGHT_DATASETS or dataset.startswith("bright-"):
        return "bright"

    if dataset.startswith("miracl-"):
        return "miracl"

    if bool(deep_get(cfg, "data.miracl", False)) and dataset in MIRACL_LANGUAGE_NAMES:
        return "miracl"

    return "beir"


def dataset_language(dataset: str) -> Optional[str]:
    lang = dataset[len("miracl-") :] if dataset.startswith("miracl-") else dataset
    return lang if lang in MIRACL_LANGUAGE_NAMES else None


def resolve_split(cfg: Mapping[str, Any], dataset: str, default_split: str) -> str:
    """
    data.split_by_dataset lets you keep a global split while overriding a few
    datasets, e.g. {msmarco: dev}. Empty by default, so existing configs behave
    exactly as before.
    """
    overrides = deep_get(cfg, "data.split_by_dataset", {}) or {}
    if isinstance(overrides, Mapping) and dataset in overrides:
        return str(overrides[dataset])
    return default_split


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _normalize_queries(raw: Mapping[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for qid, value in raw.items():
        if isinstance(value, Mapping):
            text = value.get("title") or value.get("text") or value.get("query") or ""
        else:
            text = value
        out[str(qid)] = str(text)
    return out


def _normalize_qrels(raw: Mapping[str, Any]) -> Dict[str, Dict[str, int]]:
    return {
        str(qid): {str(doc_id): int(rel) for doc_id, rel in docs.items()}
        for qid, docs in raw.items()
    }


def load_beir_bundle(data_root: Path, dataset: str, split: str) -> DatasetBundle:
    try:
        from beir.datasets.data_loader import GenericDataLoader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("BEIR is required: pip install beir") from exc

    data_path = Path(data_root) / dataset
    if not data_path.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_path}")

    corpus, queries, qrels = GenericDataLoader(data_folder=str(data_path)).load(
        split=split
    )

    corpus_norm = {
        str(doc_id): {
            "title": str(row.get("title", "")),
            "text": str(row.get("text", "")),
        }
        for doc_id, row in corpus.items()
    }

    return DatasetBundle(
        dataset=dataset,
        split=split,
        kind="beir",
        queries=_normalize_queries(queries),
        qrels=_normalize_qrels(qrels),
        corpus=corpus_norm,
    )


def load_miracl_bundle(cfg: Mapping[str, Any], dataset: str, split: str) -> DatasetBundle:
    lang = dataset_language(dataset)
    if lang is None:
        raise ValueError(
            f"Cannot infer a MIRACL language from dataset name {dataset!r}. "
            f"Known languages: {sorted(MIRACL_LANGUAGE_NAMES)}"
        )

    miracl_root = deep_get(cfg, "paths.miracl_root", None)
    topics_template = deep_get(
        cfg, "paths.miracl_topics_template", DEFAULT_MIRACL_TOPICS_TEMPLATE
    )
    qrels_template = deep_get(
        cfg, "paths.miracl_qrels_template", DEFAULT_MIRACL_QRELS_TEMPLATE
    )

    queries_path = qrels_path = None
    if miracl_root or "{miracl_root}" not in str(topics_template):
        queries_path = format_path(
            topics_template, miracl_root=miracl_root or "", lang=lang, split=split,
            dataset=dataset,
        )
        qrels_path = format_path(
            qrels_template, miracl_root=miracl_root or "", lang=lang, split=split,
            dataset=dataset,
        )

    if queries_path is not None and queries_path.is_file() and qrels_path.is_file():
        queries = _normalize_queries(load_json(queries_path))
        qrels = _normalize_qrels(load_json(qrels_path))
        source = f"json:{queries_path.parent}"
    else:
        # Fallback: pyserini's packaged MIRACL topics/qrels.
        try:
            from pyserini.search._base import get_qrels, get_topics
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pyserini is required for MIRACL topics") from exc

        topic_key = f"miracl-v1.0-{lang}-{split}"
        queries = _normalize_queries(get_topics(topic_key))
        qrels = _normalize_qrels(get_qrels(topic_key))
        source = f"pyserini:{topic_key}"

    print(f"[data] MIRACL {lang}: {len(queries):,} queries from {source}")

    corpus = load_miracl_corpus(cfg, lang=lang, dataset=dataset)

    return DatasetBundle(
        dataset=dataset,
        split=split,
        kind="miracl",
        queries=queries,
        qrels=qrels,
        corpus=corpus,
        language=lang,
        meta={"source": source},
    )


def load_miracl_corpus(
    cfg: Mapping[str, Any],
    *,
    lang: str,
    dataset: str,
) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Load a downloaded MIRACL passage collection, if one is configured and
    present. Returns None otherwise, in which case the pipeline runs
    query-side only and searches the prebuilt index.

    paths.miracl_corpus_template may point at either a single corpus.jsonl or
    a directory of raw shards (docs-0.jsonl.gz, ...) straight from
    huggingface.co/datasets/miracl/miracl-corpus.
    """
    template = deep_get(cfg, "paths.miracl_corpus_template", None)
    if not template:
        return None

    miracl_root = deep_get(cfg, "paths.miracl_root", None)
    corpus_path = format_path(
        template, miracl_root=miracl_root or "", lang=lang, dataset=dataset
    )

    if not corpus_path.exists():
        print(f"[data] no local MIRACL corpus at {corpus_path}; query-side only")
        return None

    print(f"[data] loading MIRACL {lang} corpus from {corpus_path}")
    return load_corpus_any(corpus_path)


def load_bright_bundle(dataset: str, split: str) -> DatasetBundle:
    try:
        from pyserini.search._base import get_qrels, get_topics
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pyserini is required for BRIGHT topics") from exc

    queries = _normalize_queries(get_topics(dataset))
    qrels = _normalize_qrels(get_qrels(dataset))

    print(f"[data] BRIGHT {dataset}: {len(queries):,} queries")

    return DatasetBundle(
        dataset=dataset,
        split=split,
        kind="bright",
        queries=queries,
        qrels=qrels,
        corpus=None,
        meta={"source": f"pyserini:{dataset}"},
    )


def load_dataset_bundle(
    cfg: Mapping[str, Any],
    *,
    dataset: str,
    data_root: Path,
    default_split: str,
) -> DatasetBundle:
    kind = detect_dataset_kind(cfg, dataset)
    split = resolve_split(cfg, dataset, default_split)

    if kind == "beir":
        return load_beir_bundle(data_root, dataset, split)
    if kind == "miracl":
        return load_miracl_bundle(cfg, dataset, split)
    if kind == "bright":
        return load_bright_bundle(dataset, split)

    raise ValueError(f"Unknown dataset kind: {kind}")