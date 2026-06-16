"""Tests for the per-strategy verdict dispatch (verdicts.py) — RouteIQ-833c.

THE key requirement: the dispatch GENERALIZES over whatever strategy RouteIQ
runs. We cover the family routing, >=2 concrete strategy-family verdicts
(fan-out + consistency + cost-aware), and the unknown-strategy generic fallback
that must NEVER crash.
"""

from __future__ import annotations

from stress_harness.models import (
    EnrichedRecord,
    RequestRecord,
)
from stress_harness.verdicts import (
    dispatch_verdict,
    family_for,
    registered_families,
)


def _ok_record(category: str, model: str, *, user_id: str | None = None):
    rec = RequestRecord(
        my_category_tag=category,
        prompt="p",
        request_id=f"id-{id(object())}",
        body_model=model,
        http_status=200,
        user_id=user_id,
    )
    return EnrichedRecord(request=rec)


def _result_from(records):
    from stress_harness import analysis

    return analysis.analyze(records, server_stats=None)


# --- family routing -------------------------------------------------------


def test_family_routing_covers_all_families():
    assert family_for("kumaraswamy-thompson") == "fan-out"
    assert family_for("llmrouter-knn") == "consistency"
    assert family_for("llmrouter-svm") == "consistency"
    assert family_for("llmrouter-nadirclaw-centroid") == "consistency"
    assert family_for("llmrouter-cost-aware") == "cost-aware"
    assert family_for("llmrouter-gmt") == "personalized"
    assert family_for("personalized") == "personalized"
    assert family_for("llmrouter-r1") == "latency-cost"
    assert family_for("llmrouter-knn-multiround") == "latency-cost"
    assert family_for("some-future-strategy") == "generic"
    assert family_for(None) == "generic"


def test_at_least_three_registered_families():
    families = registered_families()
    assert len(families) >= 3
    assert {"fan-out", "consistency", "cost-aware"}.issubset(set(families))


# --- fan-out (bandit) -----------------------------------------------------


def test_fan_out_healthy_when_spread():
    records = [
        _ok_record("math", "model-a"),
        _ok_record("math", "model-b"),
        _ok_record("code", "model-c"),
    ]
    result = _result_from(records)
    verdict = dispatch_verdict("kumaraswamy-thompson", records, None, result)
    assert verdict.family == "fan-out"
    assert verdict.healthy is True
    assert verdict.findings["distinct_models"] == 3
    assert verdict.findings["pinned"] is False


def test_fan_out_unhealthy_when_pinned():
    records = [_ok_record("math", "model-a") for _ in range(8)]
    result = _result_from(records)
    verdict = dispatch_verdict("kumaraswamy-thompson", records, None, result)
    assert verdict.family == "fan-out"
    assert verdict.healthy is False
    assert verdict.findings["pinned"] is True
    assert "pinned" in verdict.summary.lower()


# --- consistency (learned routers) ----------------------------------------


def test_consistency_healthy_when_category_routes_dominantly():
    # each category routes to a single dominant model.
    records = [_ok_record("math", "math-model") for _ in range(5)] + [
        _ok_record("code", "code-model") for _ in range(5)
    ]
    result = _result_from(records)
    verdict = dispatch_verdict("llmrouter-knn", records, None, result)
    assert verdict.family == "consistency"
    assert verdict.healthy is True
    assert verdict.findings["per_category"]["math"]["top_model"] == "math-model"


def test_consistency_inconsistent_when_category_spreads():
    # math splits 50/50 across two models -> below 0.6 dominant share.
    records = [
        _ok_record("math", "model-a"),
        _ok_record("math", "model-b"),
        _ok_record("math", "model-a"),
        _ok_record("math", "model-b"),
    ]
    result = _result_from(records)
    verdict = dispatch_verdict("llmrouter-svm", records, None, result)
    assert verdict.family == "consistency"
    assert verdict.healthy is False
    assert "math" in verdict.findings["inconsistent_categories"]


# --- cost-aware -----------------------------------------------------------


def test_cost_aware_flags_premium_on_easy():
    # easy-chitchat dominated by an opus (premium) model -> wasteful.
    records = [
        _ok_record("easy-chitchat", "anthropic.claude-opus-4-8") for _ in range(4)
    ] + [_ok_record("easy-chitchat", "cheap-haiku")]
    result = _result_from(records)
    verdict = dispatch_verdict("llmrouter-cost-aware", records, None, result)
    assert verdict.family == "cost-aware"
    assert verdict.healthy is False
    assert "easy-chitchat" in verdict.findings["flagged_categories"]


def test_cost_aware_healthy_when_cheap_on_easy():
    records = [_ok_record("easy-chitchat", "cheap-haiku") for _ in range(5)]
    result = _result_from(records)
    verdict = dispatch_verdict("llmrouter-cost-aware", records, None, result)
    assert verdict.family == "cost-aware"
    assert verdict.healthy is True


# --- personalized ---------------------------------------------------------


def test_personalized_detects_per_user_drift():
    records = [
        _ok_record("math", "model-a", user_id="user-000"),
        _ok_record("math", "model-b", user_id="user-001"),
    ]
    result = _result_from(records)
    verdict = dispatch_verdict("llmrouter-gmt", records, None, result)
    assert verdict.family == "personalized"
    assert verdict.findings["users"] == 2
    assert verdict.findings["distinct_dominant_models"] == 2
    assert verdict.healthy is True  # drift observed


def test_personalized_not_assessable_without_users():
    records = [_ok_record("math", "model-a")]
    result = _result_from(records)
    verdict = dispatch_verdict("personalized", records, None, result)
    assert verdict.family == "personalized"
    assert verdict.healthy is None
    assert "--num-users" in verdict.summary


# --- latency-cost (router-r1) ---------------------------------------------


def test_latency_cost_reports_percentiles():
    records = []
    for i in range(4):
        rec = RequestRecord(
            my_category_tag="hard-reasoning",
            prompt="p",
            request_id=f"id-{i}",
            body_model="model-a",
            http_status=200,
        )
        rec.client_latency_ms = float((i + 1) * 100)
        rec.total_tokens = 50
        records.append(EnrichedRecord(request=rec))
    result = _result_from(records)
    verdict = dispatch_verdict("llmrouter-r1", records, None, result)
    assert verdict.family == "latency-cost"
    assert verdict.healthy is None  # informational tradeoff
    assert verdict.findings["n_latency_samples"] == 4
    assert verdict.findings["total_tokens"] == 200


# --- the UNKNOWN-strategy generic fallback (must never crash) -------------


def test_unknown_strategy_falls_back_to_generic():
    records = [_ok_record("math", "model-a"), _ok_record("code", "model-b")]
    result = _result_from(records)
    verdict = dispatch_verdict("totally-new-2027-strategy", records, None, result)
    assert verdict.family == "generic"
    assert verdict.healthy is None
    assert "no strategy-specific verdict" in verdict.summary.lower()
    assert "totally-new-2027-strategy" in verdict.summary


def test_none_strategy_falls_back_to_generic_without_crash():
    records = [_ok_record("math", "model-a")]
    result = _result_from(records)
    verdict = dispatch_verdict(None, records, None, result)
    assert verdict.family == "generic"
    assert verdict.healthy is None


def test_generic_fallback_handles_empty_run():
    result = _result_from([])
    verdict = dispatch_verdict("anything", [], None, result)
    assert verdict.family == "generic"
    assert verdict.findings["distinct_models"] == 0
