"""
Stage orchestration: generation -> derived corpora -> indexes -> evaluation.

Stage modes:
    auto  -> reuse a valid artifact; otherwise create it
    force -> recreate the artifact
    skip  -> never create it; reuse it only if already valid

The GPU model is loaded only when at least one generation artifact must
actually be created.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .artifacts import artifact_action, validate_ids, write_manifest
from .config import deep_get, expand, format_path, require, stage_mode
from .datasets import (
    DatasetBundle,
    fingerprint_corpus,
    fingerprint_text_map,
    load_dataset_bundle,
)
from .evaluation import (
    evaluate_queries,
    make_query_variant,
    metric_columns,
    run_needs_generated_corpus,
    run_needs_generated_queries,
    selected_runs,
)
from .generation import TextGenerator, generate_to_jsonl, generation_signature
from .indexing import build_lucene_index, ensure_keywords_plus_original_corpus
from .io_utils import (
    atomic_json_dump,
    load_corpus_jsonl,
    load_query_jsonl,
    merge_corpus_shards,
    select_contiguous_shard,
    shard_artifact_path,
)
from .paths import ArtifactPaths, resolve_artifact_paths, resolve_index_specs
from .prompts import PromptBuilder, prompt_for_dataset
from .retrieval import SearcherFactory
from .stats import compare_runs


# ---------------------------------------------------------------------------
# Planning helpers
# ---------------------------------------------------------------------------

def adapt_runs_to_bundle(
    cfg: Mapping[str, Any],
    bundle: DatasetBundle,
    runs: Sequence[Mapping[str, str]],
    index_targets: Sequence[str],
) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    MIRACL/BRIGHT have no local corpus, so document-side generation and the
    keyword indexes are impossible. Runs depending on them are dropped with a
    warning (or raise, if data.skip_corpus_runs_when_unavailable is false).
    """
    if bundle.has_corpus:
        return [dict(run) for run in runs], list(index_targets)

    blocked = [run["name"] for run in runs if run_needs_generated_corpus(run)]
    blocked_targets = [
        target
        for target in index_targets
        if target in {"keywords", "keywords_plus_original"}
    ]

    if blocked or blocked_targets:
        allow_skip = bool(
            deep_get(cfg, "data.skip_corpus_runs_when_unavailable", True)
        )
        message = (
            f"{bundle.dataset} ({bundle.kind}) has no local corpus. "
            f"Skipped runs: {blocked or 'none'}; "
            f"skipped index targets: {blocked_targets or 'none'}."
        )
        if not allow_skip:
            raise RuntimeError(message)
        print(f"[warn] {message}")

    kept_runs = [dict(run) for run in runs if not run_needs_generated_corpus(run)]
    kept_targets = [t for t in index_targets if t not in {"keywords", "keywords_plus_original"}]
    return kept_runs, kept_targets


def print_dataset_plan(
    *,
    bundle: DatasetBundle,
    query_action: str,
    corpus_action: str,
    index_mode: str,
    evaluation_enabled: bool,
    runs: Sequence[Mapping[str, str]],
    paths: ArtifactPaths,
    index_specs: Mapping[str, Any],
) -> None:
    print("\n" + "=" * 78)
    print(f"Dataset: {bundle.dataset}  [kind={bundle.kind}, split={bundle.split}]")
    if bundle.language:
        print(f"Language:          {bundle.language} ({bundle.language_name})")
    print(f"Queries:           {len(bundle.queries):,}")
    if bundle.has_corpus:
        print(f"Corpus:            {len(bundle.corpus):,} documents")
    else:
        print("Corpus:            not available locally")
    print(f"Generated queries: {query_action} -> {paths.generated_queries}")
    print(f"Generated corpus:  {corpus_action} -> {paths.generated_corpus}")
    print(f"Index mode:        {index_mode}")
    print(f"Evaluate:          {evaluation_enabled}")
    if evaluation_enabled:
        print("Runs:")
        for run in runs:
            spec = index_specs.get(run["index"])
            target = spec.describe() if spec is not None else run["index"]
            print(f"  - {run['name']}: query={run['query']}, index={target}")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    cfg: Mapping[str, Any],
    *,
    dry_run: bool = False,
    num_shards: int = 1,
    shard_id: int = 0,
    merge_corpus_parts: bool = False,
) -> None:
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if not merge_corpus_parts and not (0 <= shard_id < num_shards):
        raise ValueError(f"shard_id must be between 0 and {num_shards - 1}")

    shard_mode = num_shards > 1 and not merge_corpus_parts
    if shard_mode:
        print(f"[shard] Running corpus shard {shard_id + 1}/{num_shards}")

    exp_name = str(require(cfg, "experiment.name"))
    datasets_raw = require(cfg, "data.datasets")
    if not isinstance(datasets_raw, list) or not datasets_raw:
        raise ValueError("data.datasets must be a non-empty list")
    datasets = [str(x) for x in datasets_raw]
    default_split = str(deep_get(cfg, "data.split", "test"))

    data_root = format_path(require(cfg, "paths.data_root"), exp_name=exp_name)

    # Resolve the adapter path once so generation_signature and TextGenerator agree.
    adapter_path = format_path(require(cfg, "model.adapter_path"), exp_name=exp_name)
    cfg = dict(cfg)
    cfg["model"] = dict(cfg.get("model", {}))
    cfg["model"]["adapter_path"] = str(adapter_path)

    model_name = Path(expand(require(cfg, "model.base_model"))).name

    query_mode = stage_mode(cfg, "stages.generate_queries")
    corpus_mode = stage_mode(cfg, "stages.generate_corpus")
    index_mode = stage_mode(cfg, "stages.build_indexes")
    evaluation_enabled = bool(deep_get(cfg, "stages.evaluate", True))

    allow_legacy_artifacts = bool(
        deep_get(cfg, "reuse.allow_legacy_artifacts_without_manifest", True)
    )
    allow_legacy_indexes = bool(
        deep_get(cfg, "reuse.allow_legacy_indexes_without_manifest", False)
    )

    all_runs = selected_runs(cfg) if evaluation_enabled else []
    all_targets = [str(x) for x in (deep_get(cfg, "indexing.targets", []) or [])]

    generator: Optional[TextGenerator] = None
    searchers = SearcherFactory(cfg)
    all_summaries: List[Dict[str, Any]] = []

    try:
        for dataset in datasets:
            bundle = load_dataset_bundle(
                cfg,
                dataset=dataset,
                data_root=data_root,
                default_split=default_split,
            )
            paths = resolve_artifact_paths(
                cfg, dataset=dataset, exp_name=exp_name, model_name=model_name
            )
            index_specs = resolve_index_specs(cfg, bundle, paths)
            runs, index_targets = adapt_runs_to_bundle(cfg, bundle, all_runs, all_targets)

            prompt_spec = prompt_for_dataset(cfg, bundle)
            prompt_builder = PromptBuilder(prompt_spec)
            gen_signature = generation_signature(cfg, prompt_spec)

            source_query_fp = fingerprint_text_map(bundle.queries)
            source_corpus_fp = (
                fingerprint_corpus(bundle.corpus) if bundle.has_corpus else ""
            )

            query_decoding = deep_get(cfg, "generation.query_decoding", {}) or {}
            corpus_decoding = deep_get(cfg, "generation.corpus_decoding", {}) or {}

            needs_generated_queries = any(run_needs_generated_queries(r) for r in runs)
            needs_generated_corpus = bool(
                set(index_targets) & {"keywords", "keywords_plus_original"}
            ) or any(run_needs_generated_corpus(r) for r in runs)

            # ---------------------------------------------------------------
            # Corpus shard bookkeeping
            # ---------------------------------------------------------------
            if (merge_corpus_parts or shard_mode) and not bundle.has_corpus:
                print(
                    f"[shard] {dataset} has no local corpus; skipping corpus "
                    f"shard handling"
                )
                if shard_mode:
                    continue

            if merge_corpus_parts and bundle.has_corpus:
                merge_corpus_shards(
                    final_path=paths.generated_corpus,
                    num_shards=num_shards,
                    expected_ids=list(bundle.corpus),
                )
                write_manifest(
                    paths.generated_corpus,
                    artifact_type="generated_corpus",
                    source_fingerprint=source_corpus_fp,
                    count=len(bundle.corpus),
                    generation_signature=gen_signature,
                )

            all_corpus_items = list(bundle.corpus_text.items())

            if shard_mode and bundle.has_corpus:
                corpus_items = select_contiguous_shard(
                    all_corpus_items, shard_id=shard_id, num_shards=num_shards
                )
                corpus_output_path = shard_artifact_path(
                    paths.generated_corpus, shard_id=shard_id, num_shards=num_shards
                )
                corpus_expected_ids = [doc_id for doc_id, _ in corpus_items]
                corpus_validation_fp = fingerprint_text_map(dict(corpus_items))

                print(
                    f"[shard] Dataset {dataset}: {len(corpus_items):,}/"
                    f"{len(all_corpus_items):,} documents"
                )
                print(f"[shard] Output: {corpus_output_path}")
            else:
                corpus_items = all_corpus_items
                corpus_output_path = paths.generated_corpus
                corpus_expected_ids = list(bundle.corpus or {})
                corpus_validation_fp = source_corpus_fp

            # ---------------------------------------------------------------
            # Plan
            # ---------------------------------------------------------------
            query_validation = validate_ids(
                paths.generated_queries,
                list(bundle.queries),
                artifact_kind="queries",
                source_fingerprint=source_query_fp,
                allow_legacy_without_manifest=allow_legacy_artifacts,
            )
            query_action = artifact_action(query_mode, query_validation)

            if bundle.has_corpus:
                corpus_validation = validate_ids(
                    corpus_output_path,
                    corpus_expected_ids,
                    artifact_kind="corpus",
                    source_fingerprint=corpus_validation_fp,
                    allow_legacy_without_manifest=allow_legacy_artifacts,
                )
                corpus_action = artifact_action(corpus_mode, corpus_validation)
            else:
                corpus_validation = None
                corpus_action = "unavailable"

            if shard_mode and shard_id != 0:
                query_action = "skip-on-this-shard"

            print_dataset_plan(
                bundle=bundle,
                query_action=(
                    f"{query_action} ({query_validation.reason})"
                ),
                corpus_action=(
                    f"{corpus_action} ({corpus_validation.reason})"
                    if corpus_validation is not None
                    else "not applicable (no local corpus)"
                ),
                index_mode=index_mode,
                evaluation_enabled=evaluation_enabled,
                runs=runs,
                paths=paths,
                index_specs=index_specs,
            )

            if query_action == "unavailable" and needs_generated_queries:
                raise RuntimeError(
                    f"{dataset}: generated queries are required by evaluation, but "
                    f"stages.generate_queries=skip and the artifact is invalid: "
                    f"{query_validation.reason}"
                )
            if (
                corpus_action == "unavailable"
                and needs_generated_corpus
                and bundle.has_corpus
            ):
                raise RuntimeError(
                    f"{dataset}: generated corpus is required by indexing/evaluation, "
                    f"but stages.generate_corpus=skip and the artifact is invalid: "
                    f"{corpus_validation.reason}"
                )

            if dry_run:
                continue

            # ---------------------------------------------------------------
            # Generation
            # ---------------------------------------------------------------
            will_generate_corpus = bundle.has_corpus and corpus_action == "generate"
            if query_action == "generate" or will_generate_corpus:
                if generator is None:
                    print("[gpu] loading generation model")
                    generator = TextGenerator(cfg)

            if query_action == "generate":
                query_items = list(bundle.queries.items())
                generate_to_jsonl(
                    generator,  # type: ignore[arg-type]
                    items=query_items,
                    output_path=paths.generated_queries,
                    batch_size=int(deep_get(cfg, "generation.query_batch_size", 64)),
                    prompt_builder=prompt_builder,
                    artifact_kind="queries",
                    decoding=query_decoding,
                    max_input_tokens=deep_get(
                        cfg, "generation.query_max_input_tokens", None
                    ),
                    resume=bool(deep_get(cfg, "generation.resume", True)),
                    checkpoint_every_batches=int(
                        deep_get(cfg, "generation.checkpoint_every_batches", 1)
                    ),
                )
                write_manifest(
                    paths.generated_queries,
                    artifact_type="generated_queries",
                    source_fingerprint=source_query_fp,
                    count=len(query_items),
                    generation_signature=gen_signature,
                )
                query_validation = validate_ids(
                    paths.generated_queries,
                    list(bundle.queries),
                    artifact_kind="queries",
                    source_fingerprint=source_query_fp,
                    allow_legacy_without_manifest=False,
                )
                if not query_validation.valid:
                    raise RuntimeError(
                        f"Generated query validation failed: {query_validation.reason}"
                    )

            if will_generate_corpus:
                generate_to_jsonl(
                    generator,  # type: ignore[arg-type]
                    items=corpus_items,
                    output_path=corpus_output_path,
                    batch_size=int(deep_get(cfg, "generation.corpus_batch_size", 4)),
                    prompt_builder=prompt_builder,
                    artifact_kind="corpus",
                    decoding=corpus_decoding,
                    max_input_tokens=deep_get(
                        cfg, "generation.corpus_max_input_tokens", None
                    ),
                    resume=bool(deep_get(cfg, "generation.resume", True)),
                    checkpoint_every_batches=int(
                        deep_get(cfg, "generation.checkpoint_every_batches", 1)
                    ),
                )
                write_manifest(
                    corpus_output_path,
                    artifact_type=(
                        "generated_corpus_shard" if shard_mode else "generated_corpus"
                    ),
                    source_fingerprint=corpus_validation_fp,
                    count=len(corpus_items),
                    generation_signature=gen_signature,
                )
                corpus_validation = validate_ids(
                    corpus_output_path,
                    corpus_expected_ids,
                    artifact_kind="corpus",
                    source_fingerprint=corpus_validation_fp,
                    allow_legacy_without_manifest=False,
                )
                if not corpus_validation.valid:
                    raise RuntimeError(
                        f"Generated corpus validation failed: {corpus_validation.reason}"
                    )

            if shard_mode and bundle.has_corpus:
                print(
                    f"[shard] Corpus shard {shard_id + 1}/{num_shards} is complete: "
                    f"{corpus_output_path}"
                )
                # Never index or evaluate from an incomplete corpus.
                continue

            # ---------------------------------------------------------------
            # Derived corpora and indexes
            # ---------------------------------------------------------------
            if needs_generated_corpus and bundle.has_corpus:
                ensure_keywords_plus_original_corpus(
                    generated_corpus_path=paths.generated_corpus,
                    original_corpus=bundle.corpus,
                    output_path=paths.combined_corpus,
                    source_fingerprint=source_corpus_fp,
                    force=(corpus_mode == "force"),
                )

                index_language = bundle.language if bundle.kind == "miracl" else None

                if "keywords" in index_targets or any(
                    run["index"] == "keywords" for run in runs
                ):
                    keyword_corpus = load_corpus_jsonl(paths.generated_corpus)
                    build_lucene_index(
                        corpus=keyword_corpus,
                        index_path=paths.keywords_index,
                        source_fingerprint=fingerprint_corpus(keyword_corpus),
                        mode=index_mode,
                        threads=int(deep_get(cfg, "indexing.threads", 1)),
                        allow_legacy_without_manifest=allow_legacy_indexes,
                        language=index_language,
                    )

                if "keywords_plus_original" in index_targets or any(
                    run["index"] == "keywords_plus_original" for run in runs
                ):
                    combined_corpus = load_corpus_jsonl(paths.combined_corpus)
                    build_lucene_index(
                        corpus=combined_corpus,
                        index_path=paths.keywords_plus_original_index,
                        source_fingerprint=fingerprint_corpus(combined_corpus),
                        mode=index_mode,
                        threads=int(deep_get(cfg, "indexing.threads", 1)),
                        allow_legacy_without_manifest=allow_legacy_indexes,
                        language=index_language,
                    )

            # ---------------------------------------------------------------
            # Evaluation
            # ---------------------------------------------------------------
            if evaluation_enabled and runs:
                summaries = evaluate_dataset(
                    cfg,
                    bundle=bundle,
                    paths=paths,
                    runs=runs,
                    index_specs=index_specs,
                    searchers=searchers,
                    needs_generated_queries=needs_generated_queries,
                )
                all_summaries.extend(summaries)

        if not dry_run and all_summaries:
            write_global_summary(cfg, all_summaries, exp_name=exp_name, model_name=model_name)

    finally:
        if generator is not None:
            print("[gpu] releasing generation model")
            generator.close()
        searchers.close()


# ---------------------------------------------------------------------------
# Evaluation driver
# ---------------------------------------------------------------------------

def evaluate_dataset(
    cfg: Mapping[str, Any],
    *,
    bundle: DatasetBundle,
    paths: ArtifactPaths,
    runs: Sequence[Mapping[str, str]],
    index_specs: Mapping[str, Any],
    searchers: SearcherFactory,
    needs_generated_queries: bool,
) -> List[Dict[str, Any]]:
    import pandas as pd

    generated_queries: Optional[Dict[str, str]] = None
    if needs_generated_queries:
        generated_queries = load_query_jsonl(paths.generated_queries)

    paths.result_dir.mkdir(parents=True, exist_ok=True)

    filter_self_docid = bool(
        deep_get(cfg, "evaluation.filter_self_docid", True)
    ) and bundle.supports_self_docid_filter

    dataset_summaries: List[Dict[str, Any]] = []
    per_query: Dict[str, Any] = {}

    for run in runs:
        run_name = run["name"]
        spec = index_specs[run["index"]]
        if not spec.exists:
            raise FileNotFoundError(
                f"Evaluation index does not exist for {run_name}: {spec.describe()}"
            )

        query_map = make_query_variant(run["query"], bundle.queries, generated_queries)
        searcher = searchers.get(spec)

        results, summary = evaluate_queries(
            queries=query_map,
            searcher=searcher,
            qrels=bundle.qrels,
            ndcg_k=int(deep_get(cfg, "evaluation.ndcg_k", 10)),
            mrr_k=int(deep_get(cfg, "evaluation.mrr_k", 10)),
            recall_ks=[int(x) for x in deep_get(cfg, "evaluation.recall_ks", [100, 1000])],
            filter_self_docid=filter_self_docid,
            search_threads=int(deep_get(cfg, "evaluation.search_threads", 30)),
            search_one_by_one=bool(deep_get(cfg, "evaluation.search_one_by_one", False)),
        )

        results.to_csv(paths.result_dir / f"{run_name}.csv", index=False)
        per_query[run_name] = results

        summary_row = {
            "dataset": bundle.dataset,
            "kind": bundle.kind,
            "split": bundle.split,
            "run": run_name,
            "query_variant": run["query"],
            "index_variant": run["index"],
            "index": spec.value,
            **summary,
        }
        dataset_summaries.append(summary_row)
        print(f"[eval] {run_name}: {json.dumps(summary, indent=2)}")

    pd.DataFrame(dataset_summaries).to_csv(
        paths.result_dir / "summary.csv", index=False
    )
    atomic_json_dump(dataset_summaries, paths.result_dir / "summary.json")

    # Optional paired significance testing against a baseline run.
    if bool(deep_get(cfg, "evaluation.significance.enabled", False)):
        baseline_run = str(
            deep_get(cfg, "evaluation.significance.baseline_run", runs[0]["name"])
        )
        if baseline_run in per_query:
            table = compare_runs(
                per_query,
                baseline_run=baseline_run,
                metric_columns=metric_columns(cfg),
                alpha=float(deep_get(cfg, "evaluation.significance.alpha", 0.05)),
                correction=str(
                    deep_get(cfg, "evaluation.significance.correction", "holm")
                ),
            )
            if not table.empty:
                table.to_csv(paths.result_dir / "significance.csv", index=False)
                print(f"[eval] significance vs {baseline_run}:")
                print(table.to_string(index=False))
        else:
            print(
                f"[warn] significance baseline run {baseline_run!r} was not "
                f"evaluated for {bundle.dataset}"
            )

    return dataset_summaries


def write_global_summary(
    cfg: Mapping[str, Any],
    summaries: Sequence[Mapping[str, Any]],
    *,
    exp_name: str,
    model_name: str,
) -> None:
    import pandas as pd

    output_root = format_path(
        require(cfg, "paths.output_root"), exp_name=exp_name, model=model_name, dataset=""
    )
    global_summary_path = format_path(
        deep_get(
            cfg,
            "paths.global_summary",
            "{output_root}/eval/{exp_name}/{model}/all_datasets_summary.csv",
        ),
        output_root=str(output_root),
        exp_name=exp_name,
        model=model_name,
        dataset="all",
    )
    global_summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(summaries)).to_csv(global_summary_path, index=False)
    print(f"[write] global summary: {global_summary_path}")
