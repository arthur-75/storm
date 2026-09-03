"""
Query variant construction and retrieval evaluation (NDCG / MRR / Recall).
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .config import deep_get
from .io_utils import normalize_text

QUERY_VARIANTS = {"initial", "generated", "initial_plus_generated"}
INDEX_VARIANTS = {"original", "keywords", "keywords_plus_original"}


# ---------------------------------------------------------------------------
# Run selection
# ---------------------------------------------------------------------------

def selected_runs(cfg: Mapping[str, Any]) -> List[Dict[str, str]]:
    raw = deep_get(cfg, "evaluation.runs", [])
    if not isinstance(raw, list):
        raise ValueError("evaluation.runs must be a list")

    runs: List[Dict[str, str]] = []
    seen_names: set = set()

    for row in raw:
        if not isinstance(row, Mapping):
            raise ValueError(f"Each evaluation run must be a mapping: {row}")
        if not bool(row.get("enabled", True)):
            continue

        name = str(row.get("name", "")).strip()
        query_variant = str(row.get("query", "")).strip()
        index_variant = str(row.get("index", "")).strip()

        if not name:
            raise ValueError(f"Evaluation run has no name: {row}")
        if name in seen_names:
            raise ValueError(f"Duplicate evaluation run name: {name}")
        if query_variant not in QUERY_VARIANTS:
            raise ValueError(f"Invalid query variant in {name}: {query_variant}")
        if index_variant not in INDEX_VARIANTS:
            raise ValueError(f"Invalid index variant in {name}: {index_variant}")

        seen_names.add(name)
        runs.append({"name": name, "query": query_variant, "index": index_variant})

    return runs


def run_needs_generated_corpus(run: Mapping[str, str]) -> bool:
    return run["index"] in {"keywords", "keywords_plus_original"}


def run_needs_generated_queries(run: Mapping[str, str]) -> bool:
    return run["query"] in {"generated", "initial_plus_generated"}


def make_query_variant(
    variant: str,
    initial_queries: Mapping[str, str],
    generated_queries: Optional[Mapping[str, str]],
) -> Dict[str, str]:
    if variant not in QUERY_VARIANTS:
        raise ValueError(f"Unknown query variant: {variant}")

    if variant == "initial":
        return dict(initial_queries)

    if generated_queries is None:
        raise RuntimeError(f"Query variant {variant!r} requires generated queries")

    if set(generated_queries) != set(initial_queries):
        raise RuntimeError("Generated query IDs do not match initial query IDs")

    if variant == "generated":
        return dict(generated_queries)

    return {
        qid: normalize_text(f"{initial_queries[qid]} {generated_queries[qid]}")
        for qid in initial_queries
    }


# ---------------------------------------------------------------------------
# Retrieval evaluation
# ---------------------------------------------------------------------------

def evaluate_queries(
    *,
    queries: Mapping[str, str],
    searcher: Any,
    qrels: Mapping[str, Mapping[str, int]],
    ndcg_k: int,
    mrr_k: int,
    recall_ks: Sequence[int],
    filter_self_docid: bool,
    search_threads: int,
    search_one_by_one: bool = False,
) -> Tuple[Any, Dict[str, Any]]:
    try:
        import pandas as pd
        import pytrec_eval
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Evaluation requires pandas and pytrec_eval.") from exc

    import time

    valid_qids = [qid for qid in queries if qid in qrels]
    valid_queries = [queries[qid] for qid in valid_qids]
    if not valid_qids:
        raise RuntimeError("No query IDs overlap with qrels")

    search_k = max([ndcg_k, mrr_k, *recall_ks])
    internal_ids = [f"internal_{i}" for i in range(len(valid_qids))]

    search_seconds = 0.0
    hits: Dict[str, Any] = {}

    if search_one_by_one:
        # Per-query latency measurement (matches the old STORM timing setup).
        for internal_id, query_text in zip(internal_ids, valid_queries):
            start = time.perf_counter()
            try:
                hits[internal_id] = searcher.search(query_text, k=search_k)
            except Exception as exc:
                print(f"[eval] query failed ({type(exc).__name__}: {exc})")
                hits[internal_id] = []
            search_seconds += time.perf_counter() - start
    else:
        start = time.perf_counter()
        hits = searcher.batch_search(
            valid_queries,
            internal_ids,
            k=search_k,
            threads=min(max(1, search_threads), max(1, len(valid_qids))),
        )
        search_seconds = time.perf_counter() - start

    run: Dict[str, Dict[str, float]] = {}
    rankings: Dict[str, List[str]] = {}

    for qid, internal_id in zip(valid_qids, internal_ids):
        query_hits = list(hits.get(internal_id, []))
        if filter_self_docid:
            query_hits = [hit for hit in query_hits if str(hit.docid) != qid]

        run[qid] = {str(hit.docid): float(hit.score) for hit in query_hits}
        rankings[qid] = [str(hit.docid) for hit in query_hits]

    metric_names = {f"ndcg_cut.{ndcg_k}"}
    metric_names.update(f"recall.{k}" for k in recall_ks)

    evaluator = pytrec_eval.RelevanceEvaluator(
        {qid: dict(qrels[qid]) for qid in valid_qids}, metric_names
    )
    trec_results = evaluator.evaluate(run)

    rows: List[Dict[str, Any]] = []
    for qid in valid_qids:
        metrics = trec_results.get(qid, {})
        ndcg = metrics.get(
            f"ndcg_cut_{ndcg_k}", metrics.get(f"ndcg_cut.{ndcg_k}", 0.0)
        )

        mrr = 0.0
        for rank, doc_id in enumerate(rankings[qid][:mrr_k], start=1):
            if qrels[qid].get(doc_id, 0) > 0:
                mrr = 1.0 / rank
                break

        row: Dict[str, Any] = {
            "qid": qid,
            "retrieval_query": queries[qid],
            f"ndcg@{ndcg_k}": float(ndcg) * 100.0,
            f"mrr@{mrr_k}": float(mrr) * 100.0,
        }
        for k in recall_ks:
            recall = metrics.get(f"recall_{k}", metrics.get(f"recall.{k}", 0.0))
            row[f"recall@{k}"] = float(recall) * 100.0
        rows.append(row)

    results = pd.DataFrame(rows)
    summary: Dict[str, Any] = {
        "queries": int(len(results)),
        f"ndcg@{ndcg_k}": float(results[f"ndcg@{ndcg_k}"].mean()),
        f"mrr@{mrr_k}": float(results[f"mrr@{mrr_k}"].mean()),
    }
    for k in recall_ks:
        summary[f"recall@{k}"] = float(results[f"recall@{k}"].mean())

    summary["search_seconds"] = float(search_seconds)
    summary["avg_search_seconds_per_query"] = float(
        search_seconds / max(1, len(valid_qids))
    )
    return results, summary


def metric_columns(cfg: Mapping[str, Any]) -> List[str]:
    ndcg_k = int(deep_get(cfg, "evaluation.ndcg_k", 10))
    mrr_k = int(deep_get(cfg, "evaluation.mrr_k", 10))
    recall_ks = [int(x) for x in deep_get(cfg, "evaluation.recall_ks", [100, 1000])]
    return [f"ndcg@{ndcg_k}", f"mrr@{mrr_k}"] + [f"recall@{k}" for k in recall_ks]
