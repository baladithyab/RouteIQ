"""Cross-strategy comparison seam (RouteIQ-833c).

Diff two harness runs head-to-head on the SAME workload to compare strategies.
The operator hot-swaps the active strategy via RouteIQ's registry
(``RoutingStrategyRegistry.set_active``) between runs, fires the identical
workload at each, and feeds both ``AnalysisResult``s here. The output is a
comparison structure (and a markdown table) over the model distribution and the
per-category routing so two strategies can be judged on identical traffic.

This is deliberately a pure function over two ``AnalysisResult``s — it touches no
network and no strategy internals, so it works for ANY pair of strategies
(bandit vs KNN, cost-aware vs centroid, ...), upholding the strategy-agnostic
design.
"""

from __future__ import annotations

from typing import Any

from .models import AnalysisResult


def compare_runs(
    run_a: AnalysisResult,
    run_b: AnalysisResult,
    *,
    label_a: str | None = None,
    label_b: str | None = None,
) -> dict[str, Any]:
    """Diff two runs' model + per-category distributions.

    Returns a dict with:
      * ``strategies``  — the (named) active strategy of each run,
      * ``model_distribution`` — per model: count + share under each run + delta,
      * ``per_category`` — per category: the top model + its share under each run,
      * ``summary``     — distinct-model counts + which run spread wider.

    Labels default to each run's active strategy (or ``run-a`` / ``run-b``).
    """
    la = label_a or run_a.active_strategy or "run-a"
    lb = label_b or run_b.active_strategy or "run-b"

    dist_a = run_a.model_distribution
    dist_b = run_b.model_distribution
    all_models = sorted(set(dist_a) | set(dist_b))
    model_rows: dict[str, dict[str, float]] = {}
    for model in all_models:
        ca = dist_a.get(model, {}).get("count", 0.0)
        cb = dist_b.get(model, {}).get("count", 0.0)
        sa = dist_a.get(model, {}).get("share", 0.0)
        sb = dist_b.get(model, {}).get("share", 0.0)
        model_rows[model] = {
            "count_a": ca,
            "count_b": cb,
            "share_a": sa,
            "share_b": sb,
            "share_delta": sb - sa,
        }

    all_categories = sorted(
        set(run_a.category_model_counts) | set(run_b.category_model_counts)
    )
    per_category: dict[str, dict[str, Any]] = {}
    for category in all_categories:
        per_category[category] = {
            label_a or "a": _top(run_a.category_model_counts.get(category, {})),
            label_b or "b": _top(run_b.category_model_counts.get(category, {})),
        }

    return {
        "strategies": {"a": la, "b": lb},
        "labels": {"a": label_a or "a", "b": label_b or "b"},
        "model_distribution": model_rows,
        "per_category": per_category,
        "summary": {
            "distinct_models_a": run_a.distinct_models,
            "distinct_models_b": run_b.distinct_models,
            "wider_spread": (
                la
                if run_a.distinct_models > run_b.distinct_models
                else lb
                if run_b.distinct_models > run_a.distinct_models
                else "tie"
            ),
            "successful_a": run_a.successful_requests,
            "successful_b": run_b.successful_requests,
        },
    }


def _top(counts: dict[str, int]) -> dict[str, Any]:
    """The top model + its share for a category's count map (or None)."""
    total = sum(counts.values())
    if total == 0:
        return {"top_model": None, "top_share": 0.0, "n": 0}
    top = max(counts, key=lambda m: counts[m])
    return {"top_model": top, "top_share": round(counts[top] / total, 4), "n": total}


def comparison_markdown(comparison: dict[str, Any]) -> str:
    """Render a cross-strategy comparison dict as a markdown section."""
    la = comparison["strategies"]["a"]
    lb = comparison["strategies"]["b"]
    lines: list[str] = []
    add = lines.append
    add(f"# Cross-strategy comparison: `{la}` vs `{lb}`")
    add("")
    summary = comparison["summary"]
    add(
        f"- `{la}`: {summary['distinct_models_a']} distinct models, "
        f"{summary['successful_a']} successful requests"
    )
    add(
        f"- `{lb}`: {summary['distinct_models_b']} distinct models, "
        f"{summary['successful_b']} successful requests"
    )
    add(f"- Wider model spread: **{summary['wider_spread']}**")
    add("")
    add("## Model distribution (share under each strategy)")
    add("")
    add(f"| model | {la} | {lb} | Δ share |")
    add("| --- | --- | --- | --- |")
    rows = comparison["model_distribution"]
    for model in sorted(rows, key=lambda m: -(rows[m]["count_a"] + rows[m]["count_b"])):
        r = rows[model]
        add(
            f"| {model} | {int(r['count_a'])} ({r['share_a']:.1%}) | "
            f"{int(r['count_b'])} ({r['share_b']:.1%}) | "
            f"{r['share_delta']:+.1%} |"
        )
    add("")
    add("## Per-category top model")
    add("")
    add(f"| category | {la} top | {lb} top |")
    add("| --- | --- | --- |")
    for category in sorted(comparison["per_category"]):
        cells = comparison["per_category"][category]
        keys = list(cells)
        a_cell = cells[keys[0]]
        b_cell = cells[keys[1]]
        add(
            f"| {category} | {a_cell['top_model'] or 'n/a'} "
            f"({a_cell['top_share']:.0%}) | {b_cell['top_model'] or 'n/a'} "
            f"({b_cell['top_share']:.0%}) |"
        )
    add("")
    return "\n".join(lines)
