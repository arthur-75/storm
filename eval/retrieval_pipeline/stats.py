"""
Per-query significance testing between evaluation runs.

Runs are compared against a baseline run (typically `initial_on_original`) on
the per-query metric columns produced by evaluation.py. Tests are paired and
one-sided (treatment > baseline), with optional Holm-Bonferroni correction
across the comparisons made for one dataset.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence


def paired_tests(
    baseline: Sequence[float],
    treatment: Sequence[float],
    *,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    import numpy as np
    from scipy import stats

    base = np.asarray(baseline, dtype=float)
    treat = np.asarray(treatment, dtype=float)
    diffs = treat - base

    out: Dict[str, Any] = {
        "n": int(len(diffs)),
        "mean_baseline": float(base.mean()) if len(base) else 0.0,
        "mean_treatment": float(treat.mean()) if len(treat) else 0.0,
        "delta": float(diffs.mean()) if len(diffs) else 0.0,
    }

    if len(diffs) < 2 or not np.any(diffs != 0):
        out["ttest_pvalue"] = 1.0
        out["wilcoxon_pvalue"] = 1.0
        return out

    t_stat, p_two = stats.ttest_rel(treat, base)
    out["ttest_pvalue"] = float(p_two / 2 if t_stat > 0 else 1.0 - p_two / 2)

    _, p_wilcoxon = stats.wilcoxon(treat, base, alternative="greater")
    out["wilcoxon_pvalue"] = float(p_wilcoxon)
    return out


def holm_bonferroni(pvalues: Sequence[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values, order preserved."""
    m = len(pvalues)
    if m == 0:
        return []

    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        value = (m - rank) * pvalues[idx]
        running = max(running, min(1.0, value))
        adjusted[idx] = running
    return adjusted


def compare_runs(
    per_query: Mapping[str, Any],
    *,
    baseline_run: str,
    metric_columns: Sequence[str],
    alpha: float = 0.05,
    correction: str = "holm",
) -> Any:
    """
    per_query: {run_name: DataFrame with columns ['qid', *metric_columns]}
    Returns a DataFrame with one row per (run, metric).
    """
    import pandas as pd

    if baseline_run not in per_query:
        raise KeyError(
            f"Significance baseline run {baseline_run!r} is not among the "
            f"evaluated runs: {sorted(per_query)}"
        )

    baseline_df = per_query[baseline_run].set_index("qid")
    rows: List[Dict[str, Any]] = []

    for run_name, df in per_query.items():
        if run_name == baseline_run:
            continue
        treat_df = df.set_index("qid")
        shared = baseline_df.index.intersection(treat_df.index)

        for metric in metric_columns:
            if metric not in baseline_df.columns or metric not in treat_df.columns:
                continue
            result = paired_tests(
                baseline_df.loc[shared, metric].tolist(),
                treat_df.loc[shared, metric].tolist(),
                alpha=alpha,
            )
            rows.append(
                {
                    "run": run_name,
                    "baseline": baseline_run,
                    "metric": metric,
                    **result,
                }
            )

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    for test in ("ttest", "wilcoxon"):
        column = f"{test}_pvalue"
        if correction == "holm":
            table[f"{test}_pvalue_holm"] = holm_bonferroni(table[column].tolist())
            table[f"{test}_significant"] = table[f"{test}_pvalue_holm"] < alpha
        else:
            table[f"{test}_significant"] = table[column] < alpha

    return table
