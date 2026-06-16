"""Tests for the cross-strategy comparison seam (compare.py) — RouteIQ-833c."""

from __future__ import annotations

from stress_harness import analysis
from stress_harness.compare import compare_runs, comparison_markdown
from stress_harness.models import (
    EnrichedRecord,
    RequestRecord,
    RouteIQStats,
)


def _run_result(strategy, model_by_category):
    records = []
    i = 0
    for category, models in model_by_category.items():
        for model in models:
            i += 1
            records.append(
                EnrichedRecord(
                    request=RequestRecord(
                        my_category_tag=category,
                        prompt="p",
                        request_id=f"id-{i}",
                        body_model=model,
                        http_status=200,
                    )
                )
            )
    return analysis.analyze(
        records, server_stats=RouteIQStats(active_strategy=strategy)
    )


def test_compare_two_strategies_on_same_workload():
    # strategy A pins everything to one model; strategy B spreads.
    run_a = _run_result(
        "kumaraswamy-thompson",
        {"math": ["model-a", "model-a"], "code": ["model-a"]},
    )
    run_b = _run_result(
        "llmrouter-knn",
        {"math": ["math-model", "math-model"], "code": ["code-model"]},
    )
    cmp = compare_runs(run_a, run_b)
    assert cmp["strategies"]["a"] == "kumaraswamy-thompson"
    assert cmp["strategies"]["b"] == "llmrouter-knn"
    assert cmp["summary"]["distinct_models_a"] == 1
    assert cmp["summary"]["distinct_models_b"] == 2
    assert cmp["summary"]["wider_spread"] == "llmrouter-knn"
    # per-category top model differs between runs.
    assert cmp["per_category"]["math"]["a"]["top_model"] == "model-a"
    assert cmp["per_category"]["math"]["b"]["top_model"] == "math-model"


def test_comparison_markdown_renders():
    run_a = _run_result("s-a", {"math": ["m1"]})
    run_b = _run_result("s-b", {"math": ["m2"]})
    md = comparison_markdown(compare_runs(run_a, run_b))
    assert "Cross-strategy comparison" in md
    assert "s-a" in md and "s-b" in md
    assert "| model |" in md


def test_share_delta_signs():
    run_a = _run_result("a", {"math": ["m1", "m1", "m2", "m2"]})  # 50/50
    run_b = _run_result("b", {"math": ["m1", "m1", "m1", "m2"]})  # 75/25
    cmp = compare_runs(run_a, run_b)
    # m1 share goes up from 0.5 to 0.75 -> positive delta.
    assert cmp["model_distribution"]["m1"]["share_delta"] > 0
    assert cmp["model_distribution"]["m2"]["share_delta"] < 0
