"""Tests for the strategy-agnostic analysis (analysis.py) — RouteIQ-4f19.

Asserts the ALWAYS-available distributions are emitted for any strategy, that
decision-line enrichment feeds the strategy + per-category-strategy counts, that
the active strategy names the run from the server stats, and that the verdict is
dispatched.
"""

from __future__ import annotations

from stress_harness import analysis
from stress_harness.models import (
    EnrichedRecord,
    RequestRecord,
    RouteIQStats,
    RoutingDecisionLine,
)


def _rec(
    category,
    model,
    *,
    rid,
    strategy=None,
    conversation_id=None,
    turn_index=0,
    num_turns=1,
    ok=True,
):
    rr = RequestRecord(
        my_category_tag=category,
        prompt="p",
        request_id=rid if ok else None,
        body_model=model,
        http_status=200 if ok else 500,
        conversation_id=conversation_id,
        turn_index=turn_index,
        num_turns=num_turns,
    )
    decision = None
    if strategy is not None:
        decision = RoutingDecisionLine(
            present=True, request_id=rid, model=model, strategy=strategy
        )
    return EnrichedRecord(request=rr, decision=decision)


def test_model_distribution_always_available():
    records = [
        _rec("math", "model-a", rid="1"),
        _rec("math", "model-a", rid="2"),
        _rec("code", "model-b", rid="3"),
    ]
    result = analysis.analyze(records)
    assert result.distinct_models == 2
    assert result.model_distribution["model-a"]["count"] == 2.0
    assert result.model_distribution["model-b"]["count"] == 1.0
    assert result.category_model_counts["math"] == {"model-a": 2}


def test_failed_requests_excluded_from_distribution():
    records = [
        _rec("math", "model-a", rid="1"),
        _rec("math", None, rid=None, ok=False),
    ]
    result = analysis.analyze(records)
    assert result.successful_requests == 1
    assert result.model_distribution["model-a"]["count"] == 1.0


def test_strategy_distribution_from_decision_lines():
    records = [
        _rec("math", "model-a", rid="1", strategy="llmrouter-knn"),
        _rec("code", "model-b", rid="2", strategy="llmrouter-knn"),
        _rec("code", "model-b", rid="3", strategy="cost-aware"),
    ]
    result = analysis.analyze(records)
    assert result.strategy_distribution == {"llmrouter-knn": 2, "cost-aware": 1}
    assert result.category_strategy_counts["code"] == {
        "llmrouter-knn": 1,
        "cost-aware": 1,
    }
    assert result.enriched_requests == 3


def test_active_strategy_named_from_server_stats():
    stats = RouteIQStats(
        active_strategy="llmrouter-cost-aware",
        available_strategies=["llmrouter-knn", "llmrouter-cost-aware"],
        model_distribution={"x": 10},
    )
    records = [_rec("easy-chitchat", "cheap", rid="1")]
    result = analysis.analyze(records, server_stats=stats)
    assert result.active_strategy == "llmrouter-cost-aware"
    assert result.verdict is not None
    assert result.verdict.family == "cost-aware"


def test_server_strategy_distribution_used_when_no_decision_lines():
    stats = RouteIQStats(
        active_strategy="llmrouter-knn",
        strategy_distribution={"llmrouter-knn": 99},
    )
    records = [_rec("math", "model-a", rid="1")]  # no decision-line strategy
    result = analysis.analyze(records, server_stats=stats)
    assert result.strategy_distribution == {"llmrouter-knn": 99}


def test_multi_turn_switching_detected():
    convs = [
        _rec(
            "math", "model-a", rid="t0", conversation_id="c1", turn_index=0, num_turns=2
        ),
        _rec(
            "math", "model-b", rid="t1", conversation_id="c1", turn_index=1, num_turns=2
        ),
    ]
    result = analysis.analyze(convs)
    switching = result.conversation_model_switching
    assert switching["conversations"] == 1
    assert switching["with_switch"] == 1
    assert switching["switch_rate"] == 1.0
    assert result.multi_turn_conversations == 1
    # turn-position distribution: turn 0 -> model-a, turn 1 -> model-b.
    assert result.turn_position_distribution[0] == {"model-a": 1}
    assert result.turn_position_distribution[1] == {"model-b": 1}


def test_unknown_strategy_yields_generic_verdict_no_crash():
    stats = RouteIQStats(active_strategy="future-strategy-xyz")
    records = [_rec("math", "model-a", rid="1")]
    result = analysis.analyze(records, server_stats=stats)
    assert result.verdict.family == "generic"


def test_no_stats_notes_unknown_strategy():
    result = analysis.analyze([_rec("math", "m", rid="1")], server_stats=None)
    assert result.active_strategy is None
    assert any("active_strategy unknown" in n for n in result.notes)
