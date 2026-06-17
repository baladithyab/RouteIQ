"""Unit tests for the MLOps closed-loop wiring into the eval pipeline
(Cluster H).

Verifies the FEEDBACK arm of the EXISTING eval loop reaches the drift detector
and the champion/challenger promoter end-to-end: ``EvalPipeline.push_feedback``
fans the aggregate ``{model/strategy: quality}`` to ``_mlops_aggregate_feedback``,
which drives a quality-gated promotion and a drift observation.

The wiring is gated; ``force=True`` bypasses the settings gate in tests so we do
not depend on the real Settings singleton. The drift / promoter consumers are
exercised through their public seams (singleton getters monkeypatched to live
instances) so the integration is real, not a mock tautology.
"""

from __future__ import annotations

import pytest

import litellm_llmrouter.eval_pipeline as ep
from litellm_llmrouter.eval_pipeline import EvalPipeline, reset_eval_pipeline
from litellm_llmrouter.mlops.drift import DriftDetector
from litellm_llmrouter.strategy_registry import (
    ChampionChallengerPromoter,
    RoutingContext,
    RoutingStrategy,
    get_routing_registry,
    reset_mlops_singletons,
    reset_routing_singletons,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_eval_pipeline()
    reset_routing_singletons()
    reset_mlops_singletons()
    yield
    reset_eval_pipeline()
    reset_routing_singletons()
    reset_mlops_singletons()


class _Stub(RoutingStrategy):
    def __init__(self, name: str):
        self._name = name

    def select_deployment(self, context: RoutingContext):
        return {"litellm_params": {"model": f"{self._name}-model"}}

    @property
    def name(self) -> str:
        return self._name


@pytest.mark.asyncio
async def test_wire_mlops_loop_drives_promotion(monkeypatch):
    """push_feedback aggregate -> promoter promotes the challenger."""
    reg = get_routing_registry()
    reg.register("champion", _Stub("champion"))
    reg.register("challenger", _Stub("challenger"))
    reg.set_active("champion")

    promoter = ChampionChallengerPromoter(
        champion="champion",
        challenger="challenger",
        margin=0.05,
        min_samples=2,
        registry=reg,
    )

    # Route the singleton getters to live instances (drift disabled here).
    monkeypatch.setattr(
        "litellm_llmrouter.strategy_registry.get_champion_challenger_promoter",
        lambda: promoter,
    )
    monkeypatch.setattr(
        "litellm_llmrouter.mlops.drift.get_drift_detector", lambda: None
    )

    pipeline = EvalPipeline(sample_rate=1.0)
    # Seed the tracker so the promoter's sample-count gate passes. The aggregate
    # is keyed by 'model' which here we deliberately name to match the strategy
    # names so the per-strategy promoter can read them.
    for _ in range(3):
        pipeline.tracker.record("champion", 0.70)
        pipeline.tracker.record("challenger", 0.90)

    assert ep.wire_mlops_loop(eval_pipeline=pipeline, force=True) is True

    result = await pipeline.push_feedback()
    assert result["updated"] is True

    # Promotion actually happened via the FEEDBACK arm.
    assert promoter.challenger_is_active is True
    assert reg.get_active() == "challenger"


@pytest.mark.asyncio
async def test_wire_mlops_loop_drives_drift(monkeypatch):
    """push_feedback aggregate -> drift detector observes + evaluates."""
    det = DriftDetector(quality_regression_threshold=0.1, min_samples=2, window_size=50)
    det.capture_baseline(quality=0.9)

    monkeypatch.setattr("litellm_llmrouter.mlops.drift.get_drift_detector", lambda: det)
    monkeypatch.setattr(
        "litellm_llmrouter.strategy_registry.get_champion_challenger_promoter",
        lambda: None,
    )

    pipeline = EvalPipeline(sample_rate=1.0)
    pipeline.tracker.record("gpt-4o", 0.5)  # low quality -> regression vs 0.9

    assert ep.wire_mlops_loop(eval_pipeline=pipeline, force=True) is True

    # Two pushes so the detector accrues >= min_samples current observations.
    await pipeline.push_feedback()
    await pipeline.push_feedback()

    report = det.evaluate()
    assert report.evaluated is True
    assert report.quality_regression_detected is True


@pytest.mark.asyncio
async def test_wire_idempotent(monkeypatch):
    monkeypatch.setattr(
        "litellm_llmrouter.mlops.drift.get_drift_detector", lambda: None
    )
    monkeypatch.setattr(
        "litellm_llmrouter.strategy_registry.get_champion_challenger_promoter",
        lambda: None,
    )
    pipeline = EvalPipeline(sample_rate=1.0)
    assert ep.wire_mlops_loop(eval_pipeline=pipeline, force=True) is True
    # Second wire is a no-op (callback already subscribed).
    assert ep.wire_mlops_loop(eval_pipeline=pipeline, force=True) is False


def test_wire_skipped_when_all_disabled():
    # Default settings: all mlops sub-loops off -> gate refuses (force=False).
    pipeline = EvalPipeline(sample_rate=1.0)
    assert ep.wire_mlops_loop(eval_pipeline=pipeline, force=False) is False
