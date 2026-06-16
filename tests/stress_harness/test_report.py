"""Tests for the report assembler (report.py) — RouteIQ-0be9.

THE bug: ``build_json`` and ``build_markdown`` recomputed the dispatched verdict
family via ``family_for(result.active_strategy)`` instead of reading the
authoritative ``result.verdict.family``. The two DIVERGE when a verdict plugin
errors — ``dispatch_verdict`` degrades to the ``generic`` family while
``family_for(active_strategy)`` still names the matched-but-failed family,
mislabelling the report. These tests pin that the report reads the verdict's own
family and only falls back to ``family_for`` when no verdict was computed.
"""

from __future__ import annotations

from stress_harness import report
from stress_harness.models import AnalysisResult, StrategyVerdict
from stress_harness.verdicts import family_for


def _result_with_verdict(active_strategy: str, verdict_family: str) -> AnalysisResult:
    return AnalysisResult(
        active_strategy=active_strategy,
        verdict=StrategyVerdict(
            strategy=active_strategy,
            family=verdict_family,
            healthy=None,
            summary="synthetic",
            messages=["synthetic"],
        ),
    )


def test_dispatched_family_reads_verdict_family_not_recompute():
    """When the verdict family DIVERGES from family_for(active_strategy)
    (a plugin errored and degraded to generic), the report reads the verdict."""
    # active_strategy resolves to "fan-out" but the verdict degraded to generic.
    result = _result_with_verdict("kumaraswamy-thompson", "generic")
    assert family_for("kumaraswamy-thompson") == "fan-out"  # would mislabel
    assert report._dispatched_family(result) == "generic"  # reads verdict


def test_build_json_uses_verdict_family_on_divergence():
    result = _result_with_verdict("kumaraswamy-thompson", "generic")
    data = report.build_json(result)
    assert data["verdict_family"] == "generic"


def test_build_markdown_uses_verdict_family_on_divergence():
    result = _result_with_verdict("kumaraswamy-thompson", "generic")
    md = report.build_markdown(result)
    assert "Verdict family dispatched: `generic`" in md
    # The mislabelling family must NOT appear on the dispatched-family line.
    assert "Verdict family dispatched: `fan-out`" not in md


def test_build_json_matches_verdict_family_when_consistent():
    """No divergence (plugin succeeded): the family still comes from the verdict
    and equals family_for for the happy path."""
    result = _result_with_verdict("llmrouter-knn", "consistency")
    assert family_for("llmrouter-knn") == "consistency"
    assert report.build_json(result)["verdict_family"] == "consistency"


def test_dispatched_family_falls_back_to_family_for_without_verdict():
    """No verdict computed -> fall back to family_for(active_strategy)."""
    result = AnalysisResult(active_strategy="llmrouter-knn", verdict=None)
    assert report._dispatched_family(result) == "consistency"


def test_dispatched_family_generic_without_verdict_or_strategy():
    result = AnalysisResult(active_strategy=None, verdict=None)
    assert report._dispatched_family(result) == "generic"
    assert report.build_json(result)["verdict_family"] == "generic"
