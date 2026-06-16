"""Unit tests for the dark-subsystem instruments (metrics-2).

These tests verify that the previously-uninstrumented subsystems now emit their
metrics through the canonical ``GatewayMetrics`` seam. Each test installs a
``GatewayMetrics`` backed by an in-memory reader (so no live collector is
needed), drives the subsystem at a real decision/enforcement point, and asserts
the corresponding instrument recorded a data point with the expected labels.

The same pattern as ``tests/unit/test_metrics.py``: a ``MeterProvider`` with an
``InMemoryMetricReader`` is the metric backend, and ``init_gateway_metrics``
installs the singleton that subsystems fetch via ``get_gateway_metrics()``.

Telemetry must be a no-op (never raise) when OTel is disabled; the
``no_metrics_*`` tests pin that the subsystem still works when the singleton is
absent.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from litellm_llmrouter.metrics import (
    init_gateway_metrics,
    reset_gateway_metrics,
)


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_metrics_singleton():
    """Reset GatewayMetrics singleton before and after each test."""
    reset_gateway_metrics()
    yield
    reset_gateway_metrics()


@pytest.fixture()
def metric_reader() -> InMemoryMetricReader:
    return InMemoryMetricReader()


@pytest.fixture()
def gateway_metrics(metric_reader):
    """Install the GatewayMetrics singleton backed by an in-memory reader."""
    provider = MeterProvider(metric_readers=[metric_reader])
    meter = provider.get_meter("test-dark-meter", "0.1.0")
    return init_gateway_metrics(meter)


def _data_points(reader: InMemoryMetricReader, metric_name: str) -> list[Any]:
    """Return all data points recorded for ``metric_name`` (empty if none)."""
    data = reader.get_metrics_data()
    if data is None:
        return []
    points: list[Any] = []
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == metric_name:
                    points.extend(metric.data.data_points)
    return points


def _counter_value(
    reader: InMemoryMetricReader,
    metric_name: str,
    attrs: Optional[dict[str, Any]] = None,
) -> float:
    """Sum the counter value for ``metric_name`` filtered by ``attrs``."""
    total = 0.0
    for dp in _data_points(reader, metric_name):
        if attrs is None or all(dp.attributes.get(k) == v for k, v in attrs.items()):
            total += dp.value
    return total


# ---------------------------------------------------------------------------
# GatewayMetrics record helpers (direct)
# ---------------------------------------------------------------------------


class TestRecordHelpers:
    """The new convenience record helpers fire the right instruments."""

    def test_routing_selection(self, gateway_metrics, metric_reader):
        gateway_metrics.record_routing_selection("knn", "gpt-4")
        assert (
            _counter_value(
                metric_reader,
                "gateway.routing.selection",
                {"strategy": "knn", "model": "gpt-4"},
            )
            == 1
        )

    def test_governance_denial(self, gateway_metrics, metric_reader):
        gateway_metrics.record_governance_denial("budget")
        assert (
            _counter_value(
                metric_reader,
                "gateway.governance.enforcement.denial",
                {"reason": "budget"},
            )
            == 1
        )

    def test_guardrail_check(self, gateway_metrics, metric_reader):
        gateway_metrics.record_guardrail_check("pii", "deny")
        assert (
            _counter_value(
                metric_reader,
                "gateway.guardrail.check",
                {"check_type": "pii", "action": "deny"},
            )
            == 1
        )

    def test_semantic_cache_lookup(self, gateway_metrics, metric_reader):
        gateway_metrics.record_semantic_cache_lookup("hit")
        assert (
            _counter_value(
                metric_reader, "gateway.semantic_cache.lookup", {"result": "hit"}
            )
            == 1
        )

    def test_context_optimizer_tokens_saved(self, gateway_metrics, metric_reader):
        gateway_metrics.record_context_optimizer_tokens_saved(42)
        points = _data_points(metric_reader, "gateway.context_optimizer.tokens_saved")
        assert points
        assert points[0].sum == 42

    def test_mcp_tool_invocation(self, gateway_metrics, metric_reader):
        gateway_metrics.record_mcp_tool_invocation("success")
        assert (
            _counter_value(
                metric_reader, "gateway.mcp.tool.invocation", {"result": "success"}
            )
            == 1
        )

    def test_a2a_invocation(self, gateway_metrics, metric_reader):
        gateway_metrics.record_a2a_invocation("error")
        assert (
            _counter_value(metric_reader, "gateway.a2a.invocation", {"result": "error"})
            == 1
        )

    def test_eval_sample(self, gateway_metrics, metric_reader):
        gateway_metrics.record_eval_sample("pass")
        assert (
            _counter_value(metric_reader, "gateway.eval.sample", {"verdict": "pass"})
            == 1
        )


# ---------------------------------------------------------------------------
# Semantic cache
# ---------------------------------------------------------------------------


class TestSemanticCacheMetrics:
    def test_record_hit_fires_lookup_metric(self, gateway_metrics, metric_reader):
        from litellm_llmrouter.semantic_cache import CacheManager

        mgr = CacheManager()
        mgr.record_hit()
        assert (
            _counter_value(
                metric_reader, "gateway.semantic_cache.lookup", {"result": "hit"}
            )
            == 1
        )

    def test_record_miss_fires_lookup_metric(self, gateway_metrics, metric_reader):
        from litellm_llmrouter.semantic_cache import CacheManager

        mgr = CacheManager()
        mgr.record_miss()
        assert (
            _counter_value(
                metric_reader, "gateway.semantic_cache.lookup", {"result": "miss"}
            )
            == 1
        )

    def test_no_metrics_when_singleton_absent(self):
        """A miss/hit must not raise when OTel is disabled (no singleton)."""
        from litellm_llmrouter.semantic_cache import CacheManager

        mgr = CacheManager()
        mgr.record_hit()
        mgr.record_miss()
        assert mgr.get_stats()["total_hits"] == 1


# ---------------------------------------------------------------------------
# Context optimizer
# ---------------------------------------------------------------------------


class TestContextOptimizerMetrics:
    def test_optimize_records_tokens_saved(self, gateway_metrics, metric_reader):
        import json

        from litellm_llmrouter.gateway.plugins.context_optimizer import (
            ContextOptimizer,
        )

        # A large pretty-printed JSON blob gives the json_minify transform a
        # lot to compact, so tokens_saved > 0 deterministically.
        pretty = json.dumps({f"key_{i}": f"value_{i}" for i in range(50)}, indent=4)
        opt = ContextOptimizer(mode="safe")
        messages = [{"role": "user", "content": f"Here is data: {pretty}"}]
        _, result = opt.optimize(messages)
        assert result.tokens_saved > 0
        points = _data_points(metric_reader, "gateway.context_optimizer.tokens_saved")
        assert points
        assert points[0].sum == result.tokens_saved

    def test_no_metric_when_nothing_saved(self, gateway_metrics, metric_reader):
        from litellm_llmrouter.gateway.plugins.context_optimizer import (
            ContextOptimizer,
        )

        opt = ContextOptimizer(mode="safe")
        messages = [{"role": "user", "content": "hi"}]
        _, result = opt.optimize(messages)
        # No savings -> no histogram observation.
        if result.tokens_saved <= 0:
            assert not _data_points(
                metric_reader, "gateway.context_optimizer.tokens_saved"
            )


# ---------------------------------------------------------------------------
# Eval pipeline
# ---------------------------------------------------------------------------


class TestEvalPipelineMetrics:
    @pytest.mark.asyncio
    async def test_run_batch_records_pass_and_fail(
        self, gateway_metrics, metric_reader
    ):
        from litellm_llmrouter.eval_pipeline import EvalPipeline, EvalSample

        pipeline = EvalPipeline(sample_rate=1.0, batch_size=10)

        # Stub the judge so we control the verdicts deterministically.
        async def fake_evaluate_batch(samples):
            # First sample passes (0.9), second fails (0.1).
            return [{"quality": 0.9}, {"quality": 0.1}]

        pipeline._judge.evaluate_batch = fake_evaluate_batch  # type: ignore[assignment]

        def _sample(sid: str, content: str) -> EvalSample:
            return EvalSample(
                sample_id=sid,
                timestamp=0.0,
                model="gpt-4",
                strategy="knn",
                tier="complex",
                messages=[],
                response_content=content,
            )

        pipeline.collect(_sample("s1", "a"))
        pipeline.collect(_sample("s2", "b"))

        evaluated = await pipeline.run_evaluation_batch()
        assert evaluated == 2

        assert (
            _counter_value(metric_reader, "gateway.eval.sample", {"verdict": "pass"})
            == 1
        )
        assert (
            _counter_value(metric_reader, "gateway.eval.sample", {"verdict": "fail"})
            == 1
        )


# ---------------------------------------------------------------------------
# Guardrail policy engine (config-driven, 14 check types)
# ---------------------------------------------------------------------------


class TestGuardrailPolicyMetrics:
    @pytest.mark.asyncio
    async def test_deny_records_check_action_deny(self, gateway_metrics, metric_reader):
        from litellm_llmrouter.guardrail_policies import (
            GuardrailAction,
            GuardrailPolicy,
            GuardrailPolicyEngine,
            GuardrailType,
        )

        engine = GuardrailPolicyEngine()
        engine.add_policy(
            GuardrailPolicy(
                guardrail_id="g1",
                name="no-secrets",
                check_type=GuardrailType.REGEX_DENY,
                parameters={"patterns": ["(?i)password"]},
                action=GuardrailAction.DENY,
            )
        )

        results = await engine.evaluate_input(
            {"messages": [{"role": "user", "content": "my password is hunter2"}]}
        )
        assert engine.has_deny_result(results)
        assert (
            _counter_value(
                metric_reader,
                "gateway.guardrail.check",
                {"check_type": "regex_deny", "action": "deny"},
            )
            == 1
        )

    @pytest.mark.asyncio
    async def test_pass_records_check_action_pass(self, gateway_metrics, metric_reader):
        from litellm_llmrouter.guardrail_policies import (
            GuardrailAction,
            GuardrailPolicy,
            GuardrailPolicyEngine,
            GuardrailType,
        )

        engine = GuardrailPolicyEngine()
        engine.add_policy(
            GuardrailPolicy(
                guardrail_id="g1",
                name="no-secrets",
                check_type=GuardrailType.REGEX_DENY,
                parameters={"patterns": ["(?i)password"]},
                action=GuardrailAction.DENY,
            )
        )

        results = await engine.evaluate_input(
            {"messages": [{"role": "user", "content": "hello world"}]}
        )
        assert not engine.has_deny_result(results)
        assert (
            _counter_value(
                metric_reader,
                "gateway.guardrail.check",
                {"check_type": "regex_deny", "action": "pass"},
            )
            == 1
        )


# ---------------------------------------------------------------------------
# GuardrailPlugin base (covers pii_guard / prompt_injection_guard which inherit
# the metric wiring) -- block decision normalises to action=deny.
# ---------------------------------------------------------------------------


class TestGuardrailPluginBaseMetrics:
    @pytest.mark.asyncio
    async def test_block_decision_records_check_deny(
        self, gateway_metrics, metric_reader
    ):
        from litellm_llmrouter.gateway.plugins.guardrails_base import (
            GuardrailBlockError,
            GuardrailDecision,
            GuardrailPlugin,
        )

        class _BlockingGuard(GuardrailPlugin):
            def __init__(self) -> None:
                self._enabled = True
                self._action = "block"

            async def evaluate_input(self, model, messages, kwargs):
                return GuardrailDecision(
                    allowed=False,
                    action_taken="block",
                    guardrail_name="test-guard",
                    category="injection",
                    score=0.99,
                )

        guard = _BlockingGuard()
        with pytest.raises(GuardrailBlockError):
            await guard.on_llm_pre_call("gpt-4o", [], {})

        assert (
            _counter_value(
                metric_reader,
                "gateway.guardrail.check",
                {"check_type": "injection", "action": "deny"},
            )
            == 1
        )

    @pytest.mark.asyncio
    async def test_pass_decision_records_check_pass(
        self, gateway_metrics, metric_reader
    ):
        from litellm_llmrouter.gateway.plugins.guardrails_base import (
            GuardrailDecision,
            GuardrailPlugin,
        )

        class _PassingGuard(GuardrailPlugin):
            def __init__(self) -> None:
                self._enabled = True
                self._action = "block"

            async def evaluate_input(self, model, messages, kwargs):
                return GuardrailDecision(
                    allowed=True,
                    action_taken="pass",
                    guardrail_name="test-guard",
                    category="pii",
                    score=0.0,
                )

        guard = _PassingGuard()
        await guard.on_llm_pre_call("gpt-4o", [], {})

        assert (
            _counter_value(
                metric_reader,
                "gateway.guardrail.check",
                {"check_type": "pii", "action": "pass"},
            )
            == 1
        )


# ---------------------------------------------------------------------------
# Usage policies (share the governance enforcement.denial counter)
# ---------------------------------------------------------------------------


class TestUsagePolicyMetrics:
    @pytest.mark.asyncio
    async def test_request_limit_deny_records_rate_limit(
        self, gateway_metrics, metric_reader
    ):
        from unittest.mock import AsyncMock, MagicMock

        from litellm_llmrouter.usage_policies import (
            LimitPeriod,
            LimitType,
            PolicyAction,
            UsagePolicy,
            UsagePolicyEngine,
        )

        engine = UsagePolicyEngine()
        engine.add_policy(
            UsagePolicy(
                policy_id="p1",
                name="rps-cap",
                limit_type=LimitType.REQUESTS,
                limit_value=10,
                limit_period=LimitPeriod.MINUTE,
                action=PolicyAction.DENY,
            )
        )
        # Atomic check+incr script reports over-limit (allowed=0).
        engine._redis = MagicMock()
        engine._check_incr_script = AsyncMock(return_value=[b"10", 0, 30])

        results = await engine.evaluate({"api_key": "sk-test-123", "model": "gpt-4o"})
        assert any(r.exceeded for r in results)

        assert (
            _counter_value(
                metric_reader,
                "gateway.governance.enforcement.denial",
                {"reason": "rate_limit"},
            )
            == 1
        )

    @pytest.mark.asyncio
    async def test_cost_limit_deny_records_budget(self, gateway_metrics, metric_reader):
        from unittest.mock import AsyncMock, MagicMock

        from litellm_llmrouter.usage_policies import (
            LimitPeriod,
            LimitType,
            PolicyAction,
            UsagePolicy,
            UsagePolicyEngine,
        )

        engine = UsagePolicyEngine()
        engine.add_policy(
            UsagePolicy(
                policy_id="p1",
                name="cost-cap",
                limit_type=LimitType.COST,
                limit_value=10.0,
                limit_period=LimitPeriod.MONTH,
                action=PolicyAction.DENY,
            )
        )
        # Read-only cost counter already at the limit.
        engine._redis = MagicMock()
        engine._read_script = AsyncMock(return_value=[b"10.0"])

        results = await engine.evaluate({"api_key": "sk-test-123", "model": "gpt-4o"})
        assert any(r.exceeded for r in results)

        assert (
            _counter_value(
                metric_reader,
                "gateway.governance.enforcement.denial",
                {"reason": "budget"},
            )
            == 1
        )


# ---------------------------------------------------------------------------
# Routing strategies: KTS bandit selection
# ---------------------------------------------------------------------------


class _FakeRouter:
    """Minimal LiteLLM-router stand-in exposing healthy_deployments / model_list."""

    def __init__(self, deployments):
        self.healthy_deployments = deployments
        self.model_list = deployments


def _dep(model_name: str, arm: str) -> dict:
    return {"model_name": model_name, "litellm_params": {"model": arm}}


class TestKumaraswamyThompsonMetrics:
    def test_select_records_routing_selection(self, gateway_metrics, metric_reader):
        from litellm_llmrouter.kumaraswamy_thompson import (
            STRATEGY_NAME,
            KumaraswamyThompsonStrategy,
        )
        from litellm_llmrouter.strategy_registry import RoutingContext

        deployments = [_dep("g", "gpt-4o"), _dep("g", "claude-3")]
        router = _FakeRouter(deployments)
        strat = KumaraswamyThompsonStrategy(seed=7)
        ctx = RoutingContext(
            router=router,
            model="g",
            messages=[{"role": "user", "content": "hello world"}],
            request_id="r1",
        )
        chosen = strat.select_deployment(ctx)
        assert chosen is not None

        # Exactly one selection recorded for the KTS strategy.
        total = _counter_value(
            metric_reader,
            "gateway.routing.selection",
            {"strategy": STRATEGY_NAME},
        )
        assert total == 1


# ---------------------------------------------------------------------------
# Resilience: circuit breaker transitions (previously dead instrument)
# ---------------------------------------------------------------------------


class TestCircuitBreakerMetrics:
    @pytest.mark.asyncio
    async def test_force_open_records_transition(self, gateway_metrics, metric_reader):
        from litellm_llmrouter.resilience import (
            CircuitBreaker,
            reset_shared_circuit_breaker_state,
        )

        reset_shared_circuit_breaker_state()
        cb = CircuitBreaker("test-provider")
        await cb.force_open()

        val = _counter_value(
            metric_reader,
            "gateway.circuit_breaker.transitions",
            {"breaker": "test-provider", "from_state": "closed", "to_state": "open"},
        )
        assert val == 1
        reset_shared_circuit_breaker_state()

    @pytest.mark.asyncio
    async def test_open_then_close_records_two_transitions(
        self, gateway_metrics, metric_reader
    ):
        from litellm_llmrouter.resilience import (
            CircuitBreaker,
            reset_shared_circuit_breaker_state,
        )

        reset_shared_circuit_breaker_state()
        cb = CircuitBreaker("provider-2")
        await cb.force_open()
        await cb.force_closed()

        total = _counter_value(
            metric_reader,
            "gateway.circuit_breaker.transitions",
            {"breaker": "provider-2"},
        )
        assert total == 2
        reset_shared_circuit_breaker_state()

    @pytest.mark.asyncio
    async def test_no_transition_when_state_unchanged(
        self, gateway_metrics, metric_reader
    ):
        from litellm_llmrouter.resilience import (
            CircuitBreaker,
            reset_shared_circuit_breaker_state,
        )

        reset_shared_circuit_breaker_state()
        cb = CircuitBreaker("provider-3")
        # Already CLOSED; closing again is a no-op transition.
        await cb.force_closed()
        assert not _data_points(metric_reader, "gateway.circuit_breaker.transitions")
        reset_shared_circuit_breaker_state()


# ---------------------------------------------------------------------------
# Routing strategies: personalized selection (RouteIQ-2914)
# ---------------------------------------------------------------------------


class TestPersonalizedRoutingMetric:
    """The routing.selection metric must fire from the LIVE personalized path.

    metrics-2 originally emitted ``gateway.routing.selection`` from
    ``PersonalizedRouter.get_top_model()``, but that method is not on the live
    dispatch path -- ``_route_via_personalized`` calls ``rank_models`` directly
    and selects ``ranked[0][0]``. These tests drive that live path end-to-end
    (mocking ``rank_models``) and pin that the selection metric fires exactly
    once with ``strategy=personalized`` and the chosen model.
    """

    @pytest.mark.asyncio
    async def test_route_via_personalized_records_selection(
        self, gateway_metrics, metric_reader
    ):
        from unittest.mock import AsyncMock, MagicMock, patch

        from litellm_llmrouter.custom_routing_strategy import RouteIQRoutingStrategy

        deployments = [_dep("g", "gpt-4o"), _dep("g", "claude-3")]
        router = _FakeRouter(deployments)
        strategy = RouteIQRoutingStrategy(router_instance=router)

        # Mocked personalized router: rank_models picks claude-3 as the winner.
        p_router = MagicMock()
        p_router.rank_models = AsyncMock(
            return_value=[("claude-3", 0.9), ("gpt-4o", 0.4)]
        )

        with patch(
            "litellm_llmrouter.custom_routing_strategy.get_personalized_router",
            return_value=p_router,
        ):
            result = await strategy._route_via_personalized("g", {"user": "alice"})

        # The live path selected the top-ranked model and mapped it.
        assert result is not None
        assert result["litellm_params"]["model"] == "claude-3"
        p_router.rank_models.assert_awaited_once()

        # Exactly one selection recorded for the personalized strategy/model.
        assert (
            _counter_value(
                metric_reader,
                "gateway.routing.selection",
                {"strategy": "personalized", "model": "claude-3"},
            )
            == 1
        )
        # And nothing recorded for any other strategy.
        assert (
            _counter_value(
                metric_reader,
                "gateway.routing.selection",
                {"strategy": "personalized"},
            )
            == 1
        )

    @pytest.mark.asyncio
    async def test_no_selection_metric_when_no_user(
        self, gateway_metrics, metric_reader
    ):
        """No user id -> personalized routing bails before any selection."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from litellm_llmrouter.custom_routing_strategy import RouteIQRoutingStrategy

        deployments = [_dep("g", "gpt-4o"), _dep("g", "claude-3")]
        router = _FakeRouter(deployments)
        strategy = RouteIQRoutingStrategy(router_instance=router)

        p_router = MagicMock()
        p_router.rank_models = AsyncMock(return_value=[("claude-3", 0.9)])

        with patch(
            "litellm_llmrouter.custom_routing_strategy.get_personalized_router",
            return_value=p_router,
        ):
            result = await strategy._route_via_personalized("g", {})

        assert result is None
        p_router.rank_models.assert_not_awaited()
        assert not _data_points(metric_reader, "gateway.routing.selection")
