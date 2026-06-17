"""Unit tests for the MLOps closed-loop LIVE callsites (RouteIQ-fc5c).

Three previously-dark mechanisms now have live callsites:

(a) ``wire_mlops_loop()`` -- called from the gateway lifespan after the eval
    pipeline's background loop starts. Tested here via its public seam: it
    subscribes the drift+promotion feedback callback when a flag is on, and is a
    no-op when all sub-loops are off.
(b) ``DriftDetector.record_request_bucket()`` + ``ShadowMirror.mirror()`` -- now
    called from the routing decision hot path
    (``router_decision_callback._record_mlops_hot_path``). Tested: a single
    decision records a bucket + a shadow record when the flags are on, and is a
    byte-stable no-op when off.
(c) Per-STRATEGY quality aggregation -- ``ModelQualityTracker`` now tracks
    per-strategy quality and ``_mlops_aggregate_feedback`` feeds the promoter the
    per-strategy aggregate (it keys by strategy, not model), so promotion acts on
    real quality.

All consumers are settings-gated and default-off; the hot-path tests drive the
singleton getters to live instances (no mock tautology).
"""

from __future__ import annotations

import pytest

import litellm_llmrouter.eval_pipeline as ep
from litellm_llmrouter.eval_pipeline import (
    EvalPipeline,
    ModelQualityTracker,
    reset_eval_pipeline,
    wire_mlops_loop,
)
from litellm_llmrouter.mlops.drift import (
    DriftDetector,
    reset_drift_detector,
)
import litellm_llmrouter.mlops.drift as drift_mod
import litellm_llmrouter.router_decision_callback as rdc
from litellm_llmrouter.strategy_registry import (
    ChampionChallengerPromoter,
    RoutingContext,
    RoutingStrategy,
    ShadowMirror,
    get_routing_registry,
    reset_mlops_singletons,
    reset_routing_singletons,
)
import litellm_llmrouter.strategy_registry as sr


@pytest.fixture(autouse=True)
def _reset():
    reset_eval_pipeline()
    reset_routing_singletons()
    reset_mlops_singletons()
    reset_drift_detector()
    yield
    reset_eval_pipeline()
    reset_routing_singletons()
    reset_mlops_singletons()
    reset_drift_detector()


class _Stub(RoutingStrategy):
    def __init__(self, name: str, model: str):
        self._name = name
        self._model = model

    def select_deployment(self, context: RoutingContext):
        return {"litellm_params": {"model": self._model}}

    @property
    def name(self) -> str:
        return self._name


# ===================================================================
# (a) wire_mlops_loop has a live callsite shape: subscribes when on
# ===================================================================


def test_wire_mlops_loop_subscribes_when_flag_on():
    pipeline = EvalPipeline()
    # force=True bypasses the settings gate (mirrors test_mlops_loop_wiring).
    assert wire_mlops_loop(eval_pipeline=pipeline, force=True) is True
    # Idempotent: a second call does not double-subscribe.
    assert wire_mlops_loop(eval_pipeline=pipeline, force=True) is False
    assert ep._mlops_aggregate_feedback in pipeline._feedback_callbacks


def test_wire_mlops_loop_no_op_when_all_disabled(monkeypatch):
    # No flag forced and the gate reports all sub-loops off -> no subscription.
    monkeypatch.setattr(ep, "_any_mlops_enabled", lambda: False)
    pipeline = EvalPipeline()
    assert wire_mlops_loop(eval_pipeline=pipeline) is False
    assert ep._mlops_aggregate_feedback not in pipeline._feedback_callbacks


# ===================================================================
# (b) hot path records bucket + shadow when on; no-op when off
# ===================================================================


def test_hot_path_records_bucket_and_shadow_when_on(monkeypatch):
    # Live drift detector + shadow mirror (candidate strategy registered).
    reg = get_routing_registry()
    reg.register("candidate", _Stub("candidate", "model-cand"))

    detector = DriftDetector(min_samples=1, window_size=50)
    mirror = ShadowMirror(
        candidate_strategy="candidate",
        sample_rate=1.0,  # always mirror
        registry=reg,
    )
    monkeypatch.setattr(drift_mod, "get_drift_detector", lambda: detector)
    monkeypatch.setattr(sr, "get_shadow_mirror", lambda: mirror)

    rdc._record_mlops_hot_path(
        model="gpt-4o",
        strategy="llmrouter-knn",
        messages=[{"role": "user", "content": "hello world"}],
        metadata={"_routing_profile": "balanced"},
    )

    # Drift: one bucket recorded (the tier from metadata).
    assert list(detector._cur_buckets) == ["balanced"]
    # Shadow: one mirror record produced.
    stats = mirror.get_stats()
    assert stats["total_seen"] == 1
    assert stats["total_mirrored"] == 1


def test_hot_path_is_no_op_when_disabled(monkeypatch):
    # Both singleton getters return None (the default-off state).
    monkeypatch.setattr(drift_mod, "get_drift_detector", lambda: None)
    monkeypatch.setattr(sr, "get_shadow_mirror", lambda: None)

    # Must not raise and must do nothing observable.
    rdc._record_mlops_hot_path(
        model="gpt-4o",
        strategy="llmrouter-knn",
        messages=[{"role": "user", "content": "x"}],
        metadata={},
    )


def test_prompt_length_bucket_is_low_cardinality():
    # Coarse, PII-free size classes only.
    assert rdc._prompt_length_bucket([]) == "empty"
    assert rdc._prompt_length_bucket([{"role": "user", "content": "hi"}]) == "xs"
    big = "a" * 5000
    assert rdc._prompt_length_bucket([{"role": "user", "content": big}]) == "l"


# ===================================================================
# (c) promoter sees PER-STRATEGY quality (not model-keyed)
# ===================================================================


def test_tracker_aggregates_per_strategy():
    tracker = ModelQualityTracker()
    # Two models, but both served by the same two strategies.
    tracker.record("claude", 0.9, strategy="challenger")
    tracker.record("nova", 0.7, strategy="challenger")
    tracker.record("gpt-4o", 0.6, strategy="champion")

    strat_q = tracker.get_all_strategy_qualities()
    assert strat_q["challenger"] == pytest.approx(0.8)  # (0.9 + 0.7) / 2
    assert strat_q["champion"] == pytest.approx(0.6)
    assert tracker.get_strategy_sample_counts() == {
        "challenger": 2,
        "champion": 1,
    }
    # Model-keyed window is preserved (byte-stable for existing consumers).
    assert tracker.get_quality("claude") == pytest.approx(0.9)


def test_aggregate_feedback_feeds_promoter_per_strategy(monkeypatch):
    reg = get_routing_registry()
    reg.register("champion", _Stub("champion", "model-champ"))
    reg.register("challenger", _Stub("challenger", "model-chall"))
    reg.set_active("champion")

    promoter = ChampionChallengerPromoter(
        champion="champion",
        challenger="challenger",
        margin=0.05,
        min_samples=1,
        registry=reg,
    )
    monkeypatch.setattr(sr, "get_champion_challenger_promoter", lambda: promoter)
    monkeypatch.setattr(drift_mod, "get_drift_detector", lambda: None)

    # A live pipeline whose tracker has per-strategy quality where the
    # challenger clearly beats the champion.
    pipeline = EvalPipeline()
    pipeline.tracker.record("model-chall", 0.90, strategy="challenger")
    pipeline.tracker.record("model-champ", 0.70, strategy="champion")
    monkeypatch.setattr(ep, "_pipeline", pipeline)

    # The callback receives the MODEL-keyed aggregate (no strategy keys at all),
    # but the promoter must still act -- proving it reads the per-strategy
    # aggregate from the tracker, not the model-keyed dict.
    model_keyed = {"model-chall": 0.90, "model-champ": 0.70}
    ep._mlops_aggregate_feedback(model_keyed)

    # Promotion fired on real per-strategy quality: challenger is now active.
    assert reg.get_active() == "challenger"
    assert promoter.challenger_is_active is True
