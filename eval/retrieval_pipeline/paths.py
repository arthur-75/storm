"""
Where every artifact lives, and which index each evaluation run searches.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

from .config import deep_get, format_path, require
from .datasets import DatasetBundle
from .retrieval import IndexSpec

DEFAULT_MIRACL_INDEX_TEMPLATE = "{miracl_root}/indexes/lucene-index.miracl-v1.0-{lang}"
DEFAULT_SPLADE_PREBUILT_TEMPLATE = "beir-v1.0.0-{dataset}.splade-pp-ed"


@dataclass
class ArtifactPaths:
    generated_queries: Path
    generated_corpus: Path
    combined_corpus: Path
    original_index: Path
    keywords_index: Path
    keywords_plus_original_index: Path
    result_dir: Path


def index_dataset_name(cfg: Mapping[str, Any], dataset: str) -> str:
    """
    Some query sets are evaluated against another dataset's index
    (e.g. trec-dl-2019 -> msmarco). Empty by default.
    """
    aliases = deep_get(cfg, "paths.index_dataset_aliases", {}) or {}
    if isinstance(aliases, Mapping) and dataset in aliases:
        return str(aliases[dataset])
    return dataset


def resolve_artifact_paths(
    cfg: Mapping[str, Any],
    *,
    dataset: str,
    exp_name: str,
    model_name: str,
) -> ArtifactPaths:
    output_root = format_path(
        require(cfg, "paths.output_root"),
        dataset=dataset,
        exp_name=exp_name,
        model=model_name,
    )
    index_root = format_path(
        require(cfg, "paths.original_index_root"),
        dataset=dataset,
        exp_name=exp_name,
        model=model_name,
    )

    original_index = format_path(
        deep_get(cfg, "paths.original_index_template", "{index_root}/{dataset}_docIndex"),
        index_root=str(index_root),
        dataset=index_dataset_name(cfg, dataset),
        exp_name=exp_name,
        model=model_name,
    )

    return ArtifactPaths(
        generated_queries=output_root / "keywords" / "text_data" / dataset / "queries.jsonl",
        generated_corpus=output_root / "keywords" / "text_data" / dataset / "corpus.jsonl",
        combined_corpus=(
            output_root / "keywords_corpus" / "text_data" / dataset / "corpus.jsonl"
        ),
        original_index=original_index,
        keywords_index=output_root / "keywords" / "index_data" / f"{dataset}_docIndex",
        keywords_plus_original_index=(
            output_root / "keywords_corpus" / "index_data" / f"{dataset}_docIndex"
        ),
        result_dir=output_root / "eval" / exp_name / model_name / dataset,
    )


def resolve_original_index_spec(
    cfg: Mapping[str, Any],
    bundle: DatasetBundle,
    paths: ArtifactPaths,
) -> IndexSpec:
    """
    The `original` index variant depends on the dataset family:

      beir    local Lucene index built from the BEIR corpus
      miracl  local lucene-index.miracl-v1.0-<lang> or the prebuilt equivalent
      bright  pyserini prebuilt index named after the dataset
    """
    retrieval_type = str(deep_get(cfg, "evaluation.retrieval_type", "bm25")).lower()
    encoder = (
        str(
            deep_get(
                cfg,
                "evaluation.splade_query_encoder",
                "naver/splade-cocondenser-ensembledistil",
            )
        )
        if retrieval_type == "splade"
        else None
    )

    override = deep_get(cfg, "evaluation.index_path", None)
    if override:
        return IndexSpec.from_path(
            format_path(override, dataset=bundle.dataset, lang=bundle.language or ""),
            language=bundle.language,
            encoder=encoder,
        )

    if bundle.kind == "bright":
        template = deep_get(cfg, "paths.bright_index_template", None)
        if template:
            return IndexSpec.from_path(
                format_path(template, dataset=bundle.dataset), encoder=encoder
            )
        return IndexSpec.prebuilt(bundle.dataset, encoder=encoder)

    if bundle.kind == "miracl":
        miracl_root = deep_get(cfg, "paths.miracl_root", None)
        template = deep_get(
            cfg, "paths.miracl_index_template", DEFAULT_MIRACL_INDEX_TEMPLATE
        )
        if miracl_root or "{miracl_root}" not in str(template):
            candidate = format_path(
                template,
                miracl_root=miracl_root or "",
                lang=bundle.language,
                dataset=bundle.dataset,
            )
            if candidate.is_dir():
                return IndexSpec.from_path(
                    candidate, language=bundle.language, encoder=encoder
                )
        return IndexSpec.prebuilt(
            f"miracl-v1.0-{bundle.language}",
            language=bundle.language,
            encoder=encoder,
        )

    # BEIR
    if retrieval_type == "splade":
        template = deep_get(
            cfg, "paths.splade_index_template", DEFAULT_SPLADE_PREBUILT_TEMPLATE
        )
        name = str(template).format(dataset=index_dataset_name(cfg, bundle.dataset))
        if Path(name).is_dir():
            return IndexSpec.from_path(name, encoder=encoder)
        return IndexSpec.prebuilt(name, encoder=encoder)

    return IndexSpec.from_path(paths.original_index)


def resolve_index_specs(
    cfg: Mapping[str, Any],
    bundle: DatasetBundle,
    paths: ArtifactPaths,
) -> Dict[str, IndexSpec]:
    return {
        "original": resolve_original_index_spec(cfg, bundle, paths),
        "keywords": IndexSpec.from_path(paths.keywords_index),
        "keywords_plus_original": IndexSpec.from_path(
            paths.keywords_plus_original_index
        ),
    }
