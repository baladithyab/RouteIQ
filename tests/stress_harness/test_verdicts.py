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


# --- a BUGGY plugin must degrade to generic, never crash (RouteIQ-f239) ----


def test_buggy_plugin_degrades_to_generic_with_note(monkeypatch):
    """A verdict plugin that RAISES must not crash the run: the dispatch
    catches it, falls back to the generic distribution verdict, and APPENDS a
    note naming the failed family + exception so the failure is visible."""
    from stress_harness import verdicts

    def _boom(active_strategy, records, stats, result):
        raise ValueError("synthetic plugin failure")

    # Register the raising plugin under the family ``kumaraswamy-thompson``
    # resolves to (``fan-out``), so dispatch routes to it then has it blow up.
    monkeypatch.setitem(verdicts._REGISTRY, "fan-out", _boom)

    records = [_ok_record("math", "model-a"), _ok_record("code", "model-b")]
    result = _result_from(records)

    verdict = dispatch_verdict("kumaraswamy-thompson", records, None, result)

    # Degraded to the generic verdict (never raised).
    assert verdict.family == "generic"
    # The dispatch appended a note naming the failed family + the exception.
    joined = " ".join(verdict.messages)
    assert "fan-out" in joined
    assert "errored" in joined.lower()
    assert "ValueError" in joined
    assert "synthetic plugin failure" in joined


def test_buggy_plugin_note_carried_through_to_report_family(monkeypatch):
    """When a plugin errors, the report's dispatched family must read the
    DEGRADED family (generic), not recompute family_for(active_strategy)
    (RouteIQ-0be9 + RouteIQ-f239 together)."""
    from stress_harness import report, verdicts

    def _boom(active_strategy, records, stats, result):
        raise RuntimeError("kaboom")

    monkeypatch.setitem(verdicts._REGISTRY, "fan-out", _boom)

    records = [_ok_record("math", "model-a")]
    result = _result_from(records)
    result.active_strategy = "kumaraswamy-thompson"
    result.verdict = dispatch_verdict("kumaraswamy-thompson", records, None, result)

    # family_for would still say "fan-out"; the verdict degraded to "generic".
    assert family_for("kumaraswamy-thompson") == "fan-out"
    assert result.verdict.family == "generic"
    # The report must read the DEGRADED family from the verdict, not recompute.
    assert report._dispatched_family(result) == "generic"
    assert report.build_json(result)["verdict_family"] == "generic"


# ==========================================================================
# RouteIQ-f086 — cost-cascade + semantic-intent as first-class families
# ==========================================================================


def test_f086_new_families_registered_and_routed():
    """Both new families are registered AND the dispatch routes their tokens to
    them (not to the broader cost-aware / consistency families)."""
    families = set(registered_families())
    assert {"cost-cascade", "semantic-intent"}.issubset(families)
    # cost-cascade tokens win over the broad cost-aware family.
    assert family_for("llmrouter-cost-cascade") == "cost-cascade"
    assert family_for("cheap-first-cascade") == "cost-cascade"
    # semantic-intent tokens win over the broad consistency family.
    assert family_for("llmrouter-semantic-intent") == "semantic-intent"
    assert family_for("intent-classifier") == "semantic-intent"
    # the older families still route to themselves (no regression).
    assert family_for("llmrouter-cost-aware") == "cost-aware"
    assert family_for("llmrouter-knn") == "consistency"


# --- cost-cascade ---------------------------------------------------------


def test_cost_cascade_healthy_cheap_first():
    """Cheap tier carries the bulk of traffic and no easy bucket is premium-
    dominated -> the cheap-first cascade invariant holds."""
    records = (
        [_ok_record("easy-chitchat", "cheap-haiku") for _ in range(6)]
        + [_ok_record("math", "cheap-haiku") for _ in range(3)]
        # one escalation to a premium model on the hard bucket is fine.
        + [_ok_record("hard-reasoning", "anthropic.claude-opus-4-8")]
    )
    result = _result_from(records)
    verdict = dispatch_verdict("llmrouter-cost-cascade", records, None, result)
    assert verdict.family == "cost-cascade"
    assert verdict.healthy is True
    assert verdict.findings["cheap_first"] is True
    assert verdict.findings["cheap_total"] > verdict.findings["premium_total"]


def test_cost_cascade_broken_when_premium_dominates():
    """If the premium tier carries MORE traffic than cheap, the cascade has lost
    its cheap-first invariant -> unhealthy."""
    records = [
        _ok_record("hard-reasoning", "anthropic.claude-opus-4-8") for _ in range(6)
    ] + [_ok_record("math", "cheap-haiku")]
    result = _result_from(records)
    verdict = dispatch_verdict("cost-cascade", records, None, result)
    assert verdict.family == "cost-cascade"
    assert verdict.healthy is False
    assert verdict.findings["cheap_first"] is False
    assert "premium tier dominates" in verdict.summary


def test_cost_cascade_broken_when_easy_is_premium():
    """Even with overall cheap-first spread, a premium-dominated EASY bucket
    breaks the cascade floor (cheap-on-easy)."""
    records = [
        _ok_record("easy-chitchat", "anthropic.claude-opus-4-8") for _ in range(3)
    ] + [_ok_record("math", "cheap-haiku") for _ in range(10)]
    result = _result_from(records)
    verdict = dispatch_verdict("llmrouter-cost-cascade", records, None, result)
    assert verdict.family == "cost-cascade"
    # cheap_first holds (10 cheap > 3 premium) but easy bucket is premium-heavy.
    assert verdict.findings["cheap_first"] is True
    assert "easy-chitchat" in verdict.findings["easy_flagged_categories"]
    assert verdict.healthy is False


def test_cost_cascade_not_assessable_when_no_traffic():
    result = _result_from([])
    verdict = dispatch_verdict("cost-cascade", [], None, result)
    assert verdict.family == "cost-cascade"
    assert verdict.healthy is None


# --- semantic-intent ------------------------------------------------------


def test_semantic_intent_healthy_when_buckets_dispatch_distinctly():
    """Each bucket routes dominantly to a DISTINCT model -> intent dispatch
    differentiates by bucket."""
    records = [_ok_record("math", "math-model") for _ in range(5)] + [
        _ok_record("code", "code-model") for _ in range(5)
    ]
    result = _result_from(records)
    verdict = dispatch_verdict("llmrouter-semantic-intent", records, None, result)
    assert verdict.family == "semantic-intent"
    assert verdict.healthy is True
    assert verdict.findings["distinct_handlers"] == 2
    assert verdict.findings["differentiated"] is True


def test_semantic_intent_weak_when_bucket_smeared():
    """A bucket smeared 50/50 across two models falls below the dominance
    threshold -> weak intent dispatch."""
    records = [
        _ok_record("math", "model-a"),
        _ok_record("math", "model-b"),
        _ok_record("math", "model-a"),
        _ok_record("math", "model-b"),
        _ok_record("code", "code-model"),
    ]
    result = _result_from(records)
    verdict = dispatch_verdict("semantic-intent", records, None, result)
    assert verdict.family == "semantic-intent"
    assert verdict.healthy is False
    assert "math" in verdict.findings["smeared_buckets"]


def test_semantic_intent_weak_when_no_differentiation():
    """All buckets collapse onto ONE model -> the router is not discriminating by
    intent even though each bucket is internally dominant."""
    records = [_ok_record("math", "one-model") for _ in range(5)] + [
        _ok_record("code", "one-model") for _ in range(5)
    ]
    result = _result_from(records)
    verdict = dispatch_verdict("semantic-intent", records, None, result)
    assert verdict.family == "semantic-intent"
    assert verdict.findings["distinct_handlers"] == 1
    assert verdict.findings["differentiated"] is False
    assert verdict.healthy is False


def test_semantic_intent_not_assessable_without_traffic():
    result = _result_from([])
    verdict = dispatch_verdict("semantic-intent", [], None, result)
    assert verdict.family == "semantic-intent"
    assert verdict.healthy is None


# ==========================================================================
# RouteIQ-2bbe — personalized verdict reads /me/stats per-user routing
# ==========================================================================


def test_personalized_prefers_me_stats_per_user_view():
    """When stats.per_user_recent_models is populated (read from /me/stats), the
    AUTHORITATIVE server-side per-user view drives the drift verdict — even when
    the client-observed records carry NO user ids."""
    from stress_harness.models import RouteIQStats

    stats = RouteIQStats(
        active_strategy="personalized",
        per_user_recent_models={
            "user-000": ["model-a", "model-a", "model-b"],
            "user-001": ["model-c", "model-c"],
        },
    )
    # records carry NO user_id — the server view must still be used.
    records = [_ok_record("math", "model-x"), _ok_record("code", "model-y")]
    result = _result_from(records)
    verdict = dispatch_verdict("personalized", records, stats, result)
    assert verdict.family == "personalized"
    assert verdict.findings["users"] == 2
    # user-000 dominant=model-a, user-001 dominant=model-c -> 2 distinct -> drift.
    assert verdict.findings["distinct_dominant_models"] == 2
    assert verdict.healthy is True
    assert "/me/stats" in verdict.findings["source"]
    assert "/me/stats" in verdict.summary


def test_personalized_me_stats_no_drift_when_all_converge():
    """Server per-user view where every user converges to one model -> no drift
    (soft warning, not a hard fail)."""
    from stress_harness.models import RouteIQStats

    stats = RouteIQStats(
        per_user_recent_models={
            "user-000": ["model-a", "model-a"],
            "user-001": ["model-a"],
        },
    )
    records = [_ok_record("math", "model-a")]
    result = _result_from(records)
    verdict = dispatch_verdict("personalized", records, stats, result)
    assert verdict.findings["distinct_dominant_models"] == 1
    assert verdict.healthy is False  # >1 user but no drift


def test_personalized_falls_back_to_records_without_me_stats():
    """With no /me/stats per-user view, the verdict falls back to client-observed
    records grouped by user_id (the prior behaviour) and says so."""
    records = [
        _ok_record("math", "model-a", user_id="user-000"),
        _ok_record("math", "model-b", user_id="user-001"),
    ]
    result = _result_from(records)
    # stats present but WITHOUT per_user_recent_models.
    from stress_harness.models import RouteIQStats

    stats = RouteIQStats(active_strategy="personalized")
    verdict = dispatch_verdict("personalized", records, stats, result)
    assert verdict.findings["users"] == 2
    assert "client-observed records" in verdict.findings["source"]
