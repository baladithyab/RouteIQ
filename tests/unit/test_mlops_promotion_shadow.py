"""Unit tests for MLOps champion/challenger promotion (RouteIQ-2a1c) and
shadow/mirror canary traffic (RouteIQ-4fd1), Cluster H.

REAL behavior, not mock tautology:
- challenger above margin (with enough samples) -> PROMOTE, registry.set_active
  actually switches the active strategy;
- a promoted challenger that regresses -> ROLLBACK to the champion;
- insufficient samples -> HOLD, no registry mutation;
- shadow mirror records the candidate's counterfactual selection WITHOUT
  changing the served result; the sample rate is honored exactly.
"""

from __future__ import annotations

from typing import Optional

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from litellm_llmrouter.metrics import init_gateway_metrics, reset_gateway_metrics
from litellm_llmrouter.strategy_registry import (
    ChampionChallengerPromoter,
    PromotionAction,
    RoutingContext,
    RoutingResult,
    RoutingStrategy,
    ShadowMirror,
    get_champion_challenger_promoter,
    get_routing_registry,
    get_shadow_mirror,
    reset_mlops_singletons,
    reset_routing_singletons,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_routing_singletons()
    reset_mlops_singletons()
    reset_gateway_metrics()
    yield
    reset_routing_singletons()
    reset_mlops_singletons()
    reset_gateway_metrics()


class _Stub(RoutingStrategy):
    """Strategy that returns a fixed deployment model."""

    def __init__(self, name: str, model: str):
        self._name = name
        self._model = model
        self.calls = 0

    def select_deployment(self, context: RoutingContext) -> Optional[dict]:
        self.calls += 1
        return {"litellm_params": {"model": self._model}}

    @property
    def name(self) -> str:
        return self._name


def _ctx() -> RoutingContext:
    return RoutingContext(
        router=object(),
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
    )


# ===================================================================
# Promotion / rollback (RouteIQ-2a1c)
# ===================================================================


class TestPromotion:
    def _registry(self):
        reg = get_routing_registry()
        reg.register("champion", _Stub("champion", "model-champ"))
        reg.register("challenger", _Stub("challenger", "model-chall"))
        reg.set_active("champion")
        return reg

    def test_challenger_above_margin_promotes(self):
        reg = self._registry()
        promoter = ChampionChallengerPromoter(
            champion="champion",
            challenger="challenger",
            margin=0.05,
            min_samples=10,
            registry=reg,
        )
        decision = promoter.evaluate(
            {"champion": 0.70, "challenger": 0.80},  # +0.10 lead >= 0.05
            {"champion": 50, "challenger": 50},
        )
        assert decision.action == PromotionAction.PROMOTE
        assert decision.applied is True
        # Registry actually switched.
        assert reg.get_active() == "challenger"
        assert promoter.challenger_is_active is True

    def test_regression_after_promotion_rolls_back(self):
        reg = self._registry()
        promoter = ChampionChallengerPromoter(
            champion="champion",
            challenger="challenger",
            margin=0.05,
            min_samples=10,
            registry=reg,
        )
        # First: promote.
        promoter.evaluate(
            {"champion": 0.70, "challenger": 0.85},
            {"champion": 50, "challenger": 50},
        )
        assert reg.get_active() == "challenger"

        # Then: challenger regresses below champion by the margin.
        decision = promoter.evaluate(
            {"champion": 0.80, "challenger": 0.60},  # champ +0.20 >= margin
            {"champion": 50, "challenger": 50},
        )
        assert decision.action == PromotionAction.ROLLBACK
        assert decision.applied is True
        assert reg.get_active() == "champion"
        assert promoter.challenger_is_active is False

    def test_insufficient_samples_holds(self):
        reg = self._registry()
        promoter = ChampionChallengerPromoter(
            champion="champion",
            challenger="challenger",
            margin=0.05,
            min_samples=20,
            registry=reg,
        )
        decision = promoter.evaluate(
            {"champion": 0.50, "challenger": 0.99},  # huge lead...
            {"champion": 5, "challenger": 5},  # ...but too few samples
        )
        assert decision.action == PromotionAction.HOLD
        assert decision.applied is False
        assert "insufficient_samples" in decision.reason
        # No mutation.
        assert reg.get_active() == "champion"

    def test_challenger_not_better_holds(self):
        reg = self._registry()
        promoter = ChampionChallengerPromoter(
            champion="champion",
            challenger="challenger",
            margin=0.05,
            min_samples=10,
            registry=reg,
        )
        decision = promoter.evaluate(
            {"champion": 0.80, "challenger": 0.82},  # only +0.02 < margin
            {"champion": 50, "challenger": 50},
        )
        assert decision.action == PromotionAction.HOLD
        assert reg.get_active() == "champion"

    def test_missing_quality_signal_holds(self):
        reg = self._registry()
        promoter = ChampionChallengerPromoter(
            champion="champion",
            challenger="challenger",
            registry=reg,
        )
        decision = promoter.evaluate({"champion": 0.8})  # no challenger key
        assert decision.action == PromotionAction.HOLD
        assert decision.reason == "missing_quality_signal"

    def test_dry_run_apply_false_does_not_mutate(self):
        reg = self._registry()
        promoter = ChampionChallengerPromoter(
            champion="champion",
            challenger="challenger",
            margin=0.05,
            min_samples=10,
            registry=reg,
        )
        decision = promoter.evaluate(
            {"champion": 0.70, "challenger": 0.90},
            {"champion": 50, "challenger": 50},
            apply=False,
        )
        assert decision.action == PromotionAction.PROMOTE
        assert decision.applied is False
        assert reg.get_active() == "champion"  # unchanged

    def test_promotion_metric_emitted(self):
        reader = InMemoryMetricReader()
        provider = MeterProvider(metric_readers=[reader])
        init_gateway_metrics(provider.get_meter("t", "0.1.0"))
        reg = self._registry()
        promoter = ChampionChallengerPromoter(
            champion="champion",
            challenger="challenger",
            margin=0.05,
            min_samples=10,
            registry=reg,
        )
        promoter.evaluate(
            {"champion": 0.70, "challenger": 0.90},
            {"champion": 50, "challenger": 50},
        )

        def _count(action: str) -> float:
            total = 0.0
            data = reader.get_metrics_data()
            for rm in data.resource_metrics:
                for sm in rm.scope_metrics:
                    for metric in sm.metrics:
                        if metric.name == "gateway.mlops.promotion":
                            for dp in metric.data.data_points:
                                if dp.attributes.get("action") == action:
                                    total += dp.value
            return total

        assert _count("promote") == 1.0

    def test_singleton_disabled_by_default(self):
        # mlops.promotion.enabled defaults False.
        assert get_champion_challenger_promoter() is None


# ===================================================================
# Shadow / mirror (RouteIQ-4fd1)
# ===================================================================


class TestShadowMirror:
    def test_records_candidate_decision_without_affecting_served(self):
        reg = get_routing_registry()
        candidate = _Stub("candidate", "candidate-model")
        reg.register("candidate", candidate)

        mirror = ShadowMirror(
            candidate_strategy="candidate",
            sample_rate=1.0,  # always mirror
            registry=reg,
        )
        served = RoutingResult(
            deployment={"litellm_params": {"model": "served-model"}},
            strategy_name="active",
        )
        record = mirror.mirror(_ctx(), served)

        assert record is not None
        assert record.candidate_model == "candidate-model"
        assert record.served_model == "served-model"
        assert record.agreed is False
        # Served result is untouched.
        assert served.deployment["litellm_params"]["model"] == "served-model"
        # Candidate strategy was actually invoked (real counterfactual).
        assert candidate.calls == 1

    def test_agreement_recorded_when_same_model(self):
        reg = get_routing_registry()
        reg.register("candidate", _Stub("candidate", "same-model"))
        mirror = ShadowMirror(
            candidate_strategy="candidate", sample_rate=1.0, registry=reg
        )
        served = RoutingResult(deployment={"litellm_params": {"model": "same-model"}})
        record = mirror.mirror(_ctx(), served)
        assert record is not None
        assert record.agreed is True

    def test_sample_rate_zero_never_mirrors(self):
        reg = get_routing_registry()
        candidate = _Stub("candidate", "candidate-model")
        reg.register("candidate", candidate)
        mirror = ShadowMirror(
            candidate_strategy="candidate", sample_rate=0.0, registry=reg
        )
        served = RoutingResult(deployment={"litellm_params": {"model": "served-model"}})
        for _ in range(20):
            assert mirror.mirror(_ctx(), served) is None
        assert candidate.calls == 0  # candidate never run
        stats = mirror.get_stats()
        assert stats["total_seen"] == 20
        assert stats["total_mirrored"] == 0

    def test_sample_rate_honored_with_deterministic_rng(self):
        reg = get_routing_registry()
        reg.register("candidate", _Stub("candidate", "cand"))
        # rng returns a cycling sequence; sample_rate=0.5 mirrors when rng()<0.5.
        seq = iter([0.1, 0.9, 0.2, 0.8, 0.4, 0.99])
        mirror = ShadowMirror(
            candidate_strategy="candidate",
            sample_rate=0.5,
            registry=reg,
            rng=lambda: next(seq),
        )
        served = RoutingResult(deployment={"litellm_params": {"model": "served-model"}})
        results = [mirror.mirror(_ctx(), served) for _ in range(6)]
        mirrored = [r for r in results if r is not None]
        # rng values <0.5: 0.1, 0.2, 0.4 -> 3 mirrored.
        assert len(mirrored) == 3
        assert mirror.get_stats()["total_mirrored"] == 3

    def test_unregistered_candidate_returns_none(self):
        reg = get_routing_registry()  # no candidate registered
        mirror = ShadowMirror(candidate_strategy="ghost", sample_rate=1.0, registry=reg)
        served = RoutingResult(deployment={"litellm_params": {"model": "served-model"}})
        assert mirror.mirror(_ctx(), served) is None

    def test_candidate_exception_recorded_not_raised(self):
        reg = get_routing_registry()

        class _Boom(RoutingStrategy):
            def select_deployment(self, context: RoutingContext):
                raise RuntimeError("candidate exploded")

            @property
            def name(self) -> str:
                return "boom"

        reg.register("boom", _Boom())
        mirror = ShadowMirror(candidate_strategy="boom", sample_rate=1.0, registry=reg)
        served = RoutingResult(deployment={"litellm_params": {"model": "served-model"}})
        record = mirror.mirror(_ctx(), served)
        assert record is not None
        assert record.candidate_error is not None
        assert record.candidate_model is None
        # Served result is unaffected by the candidate's failure.
        assert served.deployment["litellm_params"]["model"] == "served-model"

    def test_singleton_disabled_by_default(self):
        assert get_shadow_mirror() is None
