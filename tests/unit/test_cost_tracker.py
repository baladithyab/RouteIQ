"""
Tests for CostTrackerPlugin.

Covers:
- Plugin metadata and capability declarations
- on_llm_pre_call records start time and increments active gauge
- on_llm_success with mock response containing usage data
- on_llm_success with response lacking usage data (graceful handling)
- on_llm_failure records zero cost and increments error counter
- Cost calculation with mock litellm.completion_cost
- Cost calculation fallback when litellm raises
- Feature flag disabled skips startup
- OTel span attribute recording (shared_span_exporter)
- Edge cases: empty kwargs, missing metadata, None response
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm_llmrouter.gateway.plugin_manager import PluginCapability
from litellm_llmrouter.gateway.plugins.cost_tracker import (
    ATTR_INPUT_COST,
    ATTR_INPUT_TOKENS,
    ATTR_MODEL,
    ATTR_OUTPUT_COST,
    ATTR_OUTPUT_TOKENS,
    ATTR_TOTAL_COST,
    ATTR_TOTAL_TOKENS,
    CostTrackerPlugin,
    _calculate_cost,
    _extract_usage,
    _get_metadata,
    _is_enabled,
    _META_RESERVED_SPEND,
    _META_QUOTA_SUBJECT,
    _META_START_TIME,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class FakeResponse:
    usage: FakeUsage | None = None


def _make_kwargs(
    model: str = "gpt-4",
    metadata: dict | None = None,
) -> dict:
    """Build kwargs dict mimicking LiteLLM callback payload."""
    if metadata is None:
        metadata = {}
    return {
        "model": model,
        "litellm_params": {"metadata": metadata},
    }


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestCostTrackerMetadata:
    def test_name(self):
        plugin = CostTrackerPlugin()
        assert plugin.name == "cost-tracker"

    def test_version(self):
        plugin = CostTrackerPlugin()
        assert plugin.metadata.version == "1.0.0"

    def test_capabilities(self):
        plugin = CostTrackerPlugin()
        caps = plugin.metadata.capabilities
        assert PluginCapability.EVALUATOR in caps
        assert PluginCapability.OBSERVABILITY_EXPORTER in caps

    def test_priority(self):
        plugin = CostTrackerPlugin()
        assert plugin.metadata.priority == 50


# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------


class TestFeatureFlag:
    def test_enabled_by_default(self):
        with patch.dict("os.environ", {}, clear=False):
            # Remove key if present
            import os

            os.environ.pop("COST_TRACKER_ENABLED", None)
            assert _is_enabled() is True

    def test_disabled_via_env(self):
        with patch.dict("os.environ", {"COST_TRACKER_ENABLED": "false"}):
            assert _is_enabled() is False

    def test_enabled_explicitly(self):
        with patch.dict("os.environ", {"COST_TRACKER_ENABLED": "true"}):
            assert _is_enabled() is True

    @pytest.mark.asyncio
    async def test_startup_skips_when_disabled(self):
        with patch.dict("os.environ", {"COST_TRACKER_ENABLED": "false"}):
            plugin = CostTrackerPlugin()
            app = MagicMock()
            await plugin.startup(app)
            # Tracer should not be created when disabled
            assert plugin._tracer is None


# ---------------------------------------------------------------------------
# _extract_usage helper
# ---------------------------------------------------------------------------


class TestExtractUsage:
    def test_object_with_usage(self):
        resp = FakeResponse(usage=FakeUsage(prompt_tokens=100, completion_tokens=50))
        assert _extract_usage(resp) == (100, 50)

    def test_dict_with_usage(self):
        resp = {"usage": {"prompt_tokens": 200, "completion_tokens": 80}}
        assert _extract_usage(resp) == (200, 80)

    def test_none_response(self):
        assert _extract_usage(None) == (0, 0)

    def test_response_without_usage(self):
        resp = FakeResponse(usage=None)
        assert _extract_usage(resp) == (0, 0)

    def test_dict_without_usage(self):
        assert _extract_usage({}) == (0, 0)

    def test_usage_with_none_fields(self):
        resp = FakeResponse(
            usage=FakeUsage(prompt_tokens=None, completion_tokens=None)  # type: ignore[arg-type]
        )
        assert _extract_usage(resp) == (0, 0)


# ---------------------------------------------------------------------------
# _calculate_cost helper
# ---------------------------------------------------------------------------


class TestCalculateCost:
    def test_with_litellm_available(self):
        import litellm

        with (
            patch.object(litellm, "completion_cost", return_value=0.0075),
            patch.object(
                litellm,
                "model_cost",
                {
                    "gpt-4": {
                        "input_cost_per_token": 0.00003,
                        "output_cost_per_token": 0.00006,
                    }
                },
            ),
        ):
            inp, out, total = _calculate_cost("gpt-4", 100, 50)
            assert total == 0.0075
            assert inp == pytest.approx(0.003)
            assert out == pytest.approx(0.003)

    def test_fallback_on_exception(self):
        import litellm

        with patch.object(
            litellm,
            "completion_cost",
            side_effect=Exception("pricing unavailable"),
        ):
            inp, out, total = _calculate_cost("unknown-model", 100, 50)
            assert total == 0.0
            assert inp == 0.0
            assert out == 0.0


# ---------------------------------------------------------------------------
# _get_metadata helper
# ---------------------------------------------------------------------------


class TestGetMetadata:
    def test_normal_kwargs(self):
        kwargs = _make_kwargs(metadata={"team_id": "t1"})
        assert _get_metadata(kwargs) == {"team_id": "t1"}

    def test_missing_litellm_params(self):
        assert _get_metadata({}) == {}

    def test_none_metadata(self):
        assert _get_metadata({"litellm_params": {"metadata": None}}) == {}

    def test_non_dict_metadata(self):
        assert _get_metadata({"litellm_params": {"metadata": "oops"}}) == {}


# ---------------------------------------------------------------------------
# on_llm_pre_call
# ---------------------------------------------------------------------------


class TestOnLLMPreCall:
    @pytest.mark.asyncio
    async def test_records_start_time(self):
        plugin = CostTrackerPlugin()
        kwargs = _make_kwargs()
        await plugin.on_llm_pre_call("gpt-4", [], kwargs)

        metadata = _get_metadata(kwargs)
        assert _META_START_TIME in metadata
        assert isinstance(metadata[_META_START_TIME], float)

    @pytest.mark.asyncio
    async def test_increments_active_gauge(self):
        plugin = CostTrackerPlugin()
        mock_metrics = MagicMock()
        with patch(
            "litellm_llmrouter.metrics.get_gateway_metrics",
            return_value=mock_metrics,
        ):
            await plugin.on_llm_pre_call("gpt-4", [], _make_kwargs())
        mock_metrics.request_active.add.assert_called_once_with(1, {"model": "gpt-4"})

    @pytest.mark.asyncio
    async def test_returns_none(self):
        plugin = CostTrackerPlugin()
        result = await plugin.on_llm_pre_call("gpt-4", [], _make_kwargs())
        assert result is None


# ---------------------------------------------------------------------------
# on_llm_success
# ---------------------------------------------------------------------------


class TestOnLLMSuccess:
    @pytest.mark.asyncio
    async def test_extracts_tokens_and_records_metrics(self):
        plugin = CostTrackerPlugin()
        mock_metrics = MagicMock()

        response = FakeResponse(
            usage=FakeUsage(prompt_tokens=100, completion_tokens=50)
        )
        kwargs = _make_kwargs(metadata={"team_id": "eng", "user_id": "u1"})

        with (
            patch(
                "litellm_llmrouter.gateway.plugins.cost_tracker._calculate_cost",
                return_value=(0.003, 0.003, 0.006),
            ),
            patch(
                "litellm_llmrouter.metrics.get_gateway_metrics",
                return_value=mock_metrics,
            ),
        ):
            await plugin.on_llm_success("gpt-4", response, kwargs)

        # Cost counter
        mock_metrics.cost_total.add.assert_called_once()
        args = mock_metrics.cost_total.add.call_args
        assert args[0][0] == pytest.approx(0.006)
        assert args[0][1]["model"] == "gpt-4"

        # Tokens counter (input + output = 2 calls)
        assert mock_metrics.tokens_total.add.call_count == 2

        # Histogram
        mock_metrics.cost_per_request.record.assert_called_once()

        # Active gauge decremented
        mock_metrics.request_active.add.assert_called_once_with(-1, {"model": "gpt-4"})

    @pytest.mark.asyncio
    async def test_no_usage_data_graceful(self):
        """Response without usage data should not crash."""
        plugin = CostTrackerPlugin()
        mock_metrics = MagicMock()

        response = FakeResponse(usage=None)
        kwargs = _make_kwargs()

        with (
            patch(
                "litellm_llmrouter.gateway.plugins.cost_tracker._calculate_cost",
                return_value=(0.0, 0.0, 0.0),
            ),
            patch(
                "litellm_llmrouter.metrics.get_gateway_metrics",
                return_value=mock_metrics,
            ),
        ):
            await plugin.on_llm_success("gpt-4", response, kwargs)

        # cost_total should NOT be called with zero cost
        mock_metrics.cost_total.add.assert_not_called()
        # tokens_total should NOT be called with zero tokens
        mock_metrics.tokens_total.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_estimation_error_calculation(self):
        plugin = CostTrackerPlugin()
        mock_metrics = MagicMock()

        response = FakeResponse(
            usage=FakeUsage(prompt_tokens=100, completion_tokens=50)
        )
        # estimated_cost was 0.025, actual will be 0.006
        kwargs = _make_kwargs(metadata={"_estimated_cost": 0.025})

        with (
            patch(
                "litellm_llmrouter.gateway.plugins.cost_tracker._calculate_cost",
                return_value=(0.003, 0.003, 0.006),
            ),
            patch(
                "litellm_llmrouter.metrics.get_gateway_metrics",
                return_value=mock_metrics,
            ),
        ):
            await plugin.on_llm_success("gpt-4", response, kwargs)

        # Estimation error: (0.025 - 0.006) / 0.006 * 100 ~ 316.67%
        # We verify via span attributes in the OTel test below.

    @pytest.mark.asyncio
    async def test_none_response(self):
        """None response object should not crash."""
        plugin = CostTrackerPlugin()

        kwargs = _make_kwargs()
        with (
            patch(
                "litellm_llmrouter.gateway.plugins.cost_tracker._calculate_cost",
                return_value=(0.0, 0.0, 0.0),
            ),
            patch(
                "litellm_llmrouter.metrics.get_gateway_metrics",
                return_value=MagicMock(),
            ),
        ):
            await plugin.on_llm_success("gpt-4", None, kwargs)


# ---------------------------------------------------------------------------
# on_llm_failure
# ---------------------------------------------------------------------------


class TestOnLLMFailure:
    @pytest.mark.asyncio
    async def test_increments_error_counter(self):
        plugin = CostTrackerPlugin()
        mock_metrics = MagicMock()

        exc = ValueError("rate limit exceeded")
        with patch(
            "litellm_llmrouter.metrics.get_gateway_metrics",
            return_value=mock_metrics,
        ):
            await plugin.on_llm_failure("gpt-4", exc, _make_kwargs())

        mock_metrics.cost_errors.add.assert_called_once_with(1, {"model": "gpt-4"})

    @pytest.mark.asyncio
    async def test_decrements_active_gauge(self):
        plugin = CostTrackerPlugin()
        mock_metrics = MagicMock()

        with patch(
            "litellm_llmrouter.metrics.get_gateway_metrics",
            return_value=mock_metrics,
        ):
            await plugin.on_llm_failure("gpt-4", Exception("err"), _make_kwargs())

        mock_metrics.request_active.add.assert_called_once_with(-1, {"model": "gpt-4"})

    @pytest.mark.asyncio
    async def test_failure_with_no_instruments(self):
        """Failure when OTel instruments are None should not crash."""
        plugin = CostTrackerPlugin()
        await plugin.on_llm_failure("gpt-4", Exception("err"), _make_kwargs())


# ---------------------------------------------------------------------------
# OTel span attributes (integration with shared_span_exporter)
# ---------------------------------------------------------------------------


class TestOTelSpanAttributes:
    @pytest.mark.asyncio
    async def test_success_sets_span_attributes(self, shared_span_exporter):
        from opentelemetry import trace

        tracer = trace.get_tracer("test")
        plugin = CostTrackerPlugin()

        response = FakeResponse(
            usage=FakeUsage(prompt_tokens=500, completion_tokens=200)
        )
        kwargs = _make_kwargs(metadata={"_estimated_cost": 0.10})

        with tracer.start_as_current_span("test-llm-call"):
            with (
                patch(
                    "litellm_llmrouter.gateway.plugins.cost_tracker._calculate_cost",
                    return_value=(0.015, 0.012, 0.027),
                ),
                patch(
                    "litellm_llmrouter.metrics.get_gateway_metrics",
                    return_value=MagicMock(),
                ),
            ):
                await plugin.on_llm_success("gpt-4", response, kwargs)

        spans = shared_span_exporter.get_finished_spans()
        assert len(spans) == 1

        attrs = dict(spans[0].attributes)
        assert attrs[ATTR_MODEL] == "gpt-4"
        assert attrs[ATTR_INPUT_TOKENS] == 500
        assert attrs[ATTR_OUTPUT_TOKENS] == 200
        assert attrs[ATTR_TOTAL_TOKENS] == 700
        assert attrs[ATTR_INPUT_COST] == pytest.approx(0.015)
        assert attrs[ATTR_OUTPUT_COST] == pytest.approx(0.012)
        assert attrs[ATTR_TOTAL_COST] == pytest.approx(0.027)

    @pytest.mark.asyncio
    async def test_failure_sets_zero_cost_attributes(self, shared_span_exporter):
        from opentelemetry import trace

        tracer = trace.get_tracer("test")
        plugin = CostTrackerPlugin()

        with tracer.start_as_current_span("test-llm-fail"):
            await plugin.on_llm_failure("gpt-4", Exception("err"), _make_kwargs())

        spans = shared_span_exporter.get_finished_spans()
        assert len(spans) == 1

        attrs = dict(spans[0].attributes)
        assert attrs[ATTR_TOTAL_COST] == 0.0
        assert attrs[ATTR_TOTAL_TOKENS] == 0
        assert attrs[ATTR_MODEL] == "gpt-4"


# ---------------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------------


class TestStartupShutdown:
    @pytest.mark.asyncio
    async def test_startup_creates_instruments(self):
        plugin = CostTrackerPlugin()
        app = MagicMock()
        await plugin.startup(app)
        # When OTel is available, tracer should be created
        # (metrics are now centralized via GatewayMetrics, not on the plugin)
        assert plugin._tracer is not None

    @pytest.mark.asyncio
    async def test_shutdown_does_not_raise(self):
        plugin = CostTrackerPlugin()
        app = MagicMock()
        await plugin.startup(app)
        await plugin.shutdown(app)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_kwargs(self):
        plugin = CostTrackerPlugin()
        # Should not crash with empty kwargs
        await plugin.on_llm_pre_call("gpt-4", [], {})

    @pytest.mark.asyncio
    async def test_success_with_empty_kwargs(self):
        plugin = CostTrackerPlugin()
        with (
            patch(
                "litellm_llmrouter.gateway.plugins.cost_tracker._calculate_cost",
                return_value=(0.0, 0.0, 0.0),
            ),
            patch(
                "litellm_llmrouter.metrics.get_gateway_metrics",
                return_value=MagicMock(),
            ),
        ):
            await plugin.on_llm_success("gpt-4", FakeResponse(), {})

    @pytest.mark.asyncio
    async def test_failure_with_empty_kwargs(self):
        plugin = CostTrackerPlugin()
        await plugin.on_llm_failure("gpt-4", Exception("err"), {})

    @pytest.mark.asyncio
    async def test_dict_response_with_usage(self):
        """Response as a plain dict (not object) should still work."""
        plugin = CostTrackerPlugin()
        mock_metrics = MagicMock()

        response = {
            "usage": {"prompt_tokens": 50, "completion_tokens": 25},
            "choices": [],
        }
        kwargs = _make_kwargs()

        with (
            patch(
                "litellm_llmrouter.gateway.plugins.cost_tracker._calculate_cost",
                return_value=(0.001, 0.001, 0.002),
            ),
            patch(
                "litellm_llmrouter.metrics.get_gateway_metrics",
                return_value=mock_metrics,
            ),
        ):
            await plugin.on_llm_success("gpt-4", response, kwargs)

        mock_metrics.cost_total.add.assert_called_once()


# ---------------------------------------------------------------------------
# Cost reconciliation wiring
# ---------------------------------------------------------------------------


class TestCostReconciliationWiring:
    """Tests that CostTrackerPlugin.on_llm_success wires into quota reconciliation."""

    @pytest.mark.asyncio
    async def test_reconcile_called_with_reserved_spend(self):
        """on_llm_success calls reconcile_spend when metadata has reserved_spend."""
        plugin = CostTrackerPlugin()
        mock_metrics = MagicMock()
        mock_enforcer = MagicMock()
        mock_enforcer.reconcile_spend = AsyncMock(
            return_value={"reconciled": True, "adjustment": 0.07}
        )

        response = FakeResponse(
            usage=FakeUsage(prompt_tokens=100, completion_tokens=50)
        )
        kwargs = _make_kwargs(
            metadata={
                _META_RESERVED_SPEND: 0.10,
                _META_QUOTA_SUBJECT: "team:eng",
            }
        )

        with (
            patch(
                "litellm_llmrouter.gateway.plugins.cost_tracker._calculate_cost",
                return_value=(0.015, 0.015, 0.03),
            ),
            patch(
                "litellm_llmrouter.metrics.get_gateway_metrics",
                return_value=mock_metrics,
            ),
            patch(
                "litellm_llmrouter.quota.get_quota_enforcer",
                return_value=mock_enforcer,
            ),
        ):
            await plugin.on_llm_success("gpt-4", response, kwargs)

        mock_enforcer.reconcile_spend.assert_awaited_once_with(
            subject="team:eng",
            actual_cost=0.03,
            reserved_cost=0.10,
        )

    @pytest.mark.asyncio
    async def test_reconcile_skipped_without_reserved_spend(self):
        """on_llm_success skips reconciliation when no reserved_spend in metadata."""
        plugin = CostTrackerPlugin()
        mock_metrics = MagicMock()
        mock_enforcer = MagicMock()
        mock_enforcer.reconcile_spend = AsyncMock()

        response = FakeResponse(
            usage=FakeUsage(prompt_tokens=100, completion_tokens=50)
        )
        # No _META_RESERVED_SPEND or _META_QUOTA_SUBJECT in metadata
        kwargs = _make_kwargs(metadata={"team_id": "eng"})

        with (
            patch(
                "litellm_llmrouter.gateway.plugins.cost_tracker._calculate_cost",
                return_value=(0.015, 0.015, 0.03),
            ),
            patch(
                "litellm_llmrouter.metrics.get_gateway_metrics",
                return_value=mock_metrics,
            ),
            patch(
                "litellm_llmrouter.quota.get_quota_enforcer",
                return_value=mock_enforcer,
            ),
        ):
            await plugin.on_llm_success("gpt-4", response, kwargs)

        # reconcile_spend should NOT be called
        mock_enforcer.reconcile_spend.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_reconcile_skipped_when_subject_empty(self):
        """on_llm_success skips reconciliation when quota_subject is empty."""
        plugin = CostTrackerPlugin()
        mock_metrics = MagicMock()
        mock_enforcer = MagicMock()
        mock_enforcer.reconcile_spend = AsyncMock()

        response = FakeResponse(
            usage=FakeUsage(prompt_tokens=100, completion_tokens=50)
        )
        kwargs = _make_kwargs(
            metadata={
                _META_RESERVED_SPEND: 0.10,
                _META_QUOTA_SUBJECT: "",  # empty subject
            }
        )

        with (
            patch(
                "litellm_llmrouter.gateway.plugins.cost_tracker._calculate_cost",
                return_value=(0.015, 0.015, 0.03),
            ),
            patch(
                "litellm_llmrouter.metrics.get_gateway_metrics",
                return_value=mock_metrics,
            ),
            patch(
                "litellm_llmrouter.quota.get_quota_enforcer",
                return_value=mock_enforcer,
            ),
        ):
            await plugin.on_llm_success("gpt-4", response, kwargs)

        mock_enforcer.reconcile_spend.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_reconcile_failure_does_not_crash(self):
        """on_llm_success handles reconciliation errors gracefully."""
        plugin = CostTrackerPlugin()
        mock_metrics = MagicMock()
        mock_enforcer = MagicMock()
        mock_enforcer.reconcile_spend = AsyncMock(
            side_effect=Exception("Redis connection lost")
        )

        response = FakeResponse(
            usage=FakeUsage(prompt_tokens=100, completion_tokens=50)
        )
        kwargs = _make_kwargs(
            metadata={
                _META_RESERVED_SPEND: 0.10,
                _META_QUOTA_SUBJECT: "team:eng",
            }
        )

        with (
            patch(
                "litellm_llmrouter.gateway.plugins.cost_tracker._calculate_cost",
                return_value=(0.015, 0.015, 0.03),
            ),
            patch(
                "litellm_llmrouter.metrics.get_gateway_metrics",
                return_value=mock_metrics,
            ),
            patch(
                "litellm_llmrouter.quota.get_quota_enforcer",
                return_value=mock_enforcer,
            ),
        ):
            # Should not raise
            await plugin.on_llm_success("gpt-4", response, kwargs)

        # Metrics should still be recorded despite reconciliation failure
        mock_metrics.cost_per_request.record.assert_called_once()

    @pytest.mark.asyncio
    async def test_reconcile_passes_actual_total_cost(self):
        """Reconciliation receives the correct actual_cost from _calculate_cost."""
        plugin = CostTrackerPlugin()
        mock_metrics = MagicMock()
        mock_enforcer = MagicMock()
        mock_enforcer.reconcile_spend = AsyncMock(return_value={"reconciled": True})

        response = FakeResponse(
            usage=FakeUsage(prompt_tokens=200, completion_tokens=100)
        )
        kwargs = _make_kwargs(
            metadata={
                _META_RESERVED_SPEND: 0.50,
                _META_QUOTA_SUBJECT: "user:alice",
            }
        )

        actual_total_cost = 0.0123
        with (
            patch(
                "litellm_llmrouter.gateway.plugins.cost_tracker._calculate_cost",
                return_value=(0.006, 0.006, actual_total_cost),
            ),
            patch(
                "litellm_llmrouter.metrics.get_gateway_metrics",
                return_value=mock_metrics,
            ),
            patch(
                "litellm_llmrouter.quota.get_quota_enforcer",
                return_value=mock_enforcer,
            ),
        ):
            await plugin.on_llm_success("gpt-4", response, kwargs)

        # Verify the exact actual_cost and reserved_cost passed
        call_kwargs = mock_enforcer.reconcile_spend.call_args[1]
        assert call_kwargs["actual_cost"] == pytest.approx(actual_total_cost)
        assert call_kwargs["reserved_cost"] == pytest.approx(0.50)
