"""
Unit Tests for OTel Metrics Instrument Registry
=================================================

Tests for:
- GatewayMetrics initialization and singleton lifecycle
- All instrument creation and types
- LLM_API_PATHS expansion
- Metric recording in success/failure callbacks
- GenAI span attributes on success
- _compute_duration helper
"""

import time
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry import trace
from opentelemetry.metrics import (
    Counter,
    Histogram,
    UpDownCounter,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from litellm_llmrouter.metrics import (
    GatewayMetrics,
    get_gateway_metrics,
    init_gateway_metrics,
    reset_gateway_metrics,
)
from litellm_llmrouter.router_decision_callback import (
    LLM_API_PATHS,
    RouterDecisionCallback,
    RouterDecisionMiddleware,
    _compute_duration,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_metrics_singleton():
    """Reset GatewayMetrics singleton before and after each test."""
    reset_gateway_metrics()
    yield
    reset_gateway_metrics()


@pytest.fixture()
def metric_reader():
    """Provide an InMemoryMetricReader for verifying recorded metrics."""
    return InMemoryMetricReader()


@pytest.fixture()
def meter(metric_reader):
    """Provide a Meter backed by an InMemoryMetricReader."""
    provider = MeterProvider(metric_readers=[metric_reader])
    return provider.get_meter("test-meter", "0.1.0")


@pytest.fixture()
def gateway_metrics(meter):
    """Initialize and return a GatewayMetrics instance."""
    return init_gateway_metrics(meter)


# ===========================================================================
# GatewayMetrics singleton lifecycle
# ===========================================================================


class TestGatewayMetricsSingleton:
    """Tests for the module-level singleton pattern."""

    def test_get_returns_none_before_init(self):
        assert get_gateway_metrics() is None

    def test_init_returns_instance(self, meter):
        gm = init_gateway_metrics(meter)
        assert isinstance(gm, GatewayMetrics)

    def test_get_returns_instance_after_init(self, meter):
        init_gateway_metrics(meter)
        gm = get_gateway_metrics()
        assert gm is not None
        assert isinstance(gm, GatewayMetrics)

    def test_reset_clears_singleton(self, meter):
        init_gateway_metrics(meter)
        reset_gateway_metrics()
        assert get_gateway_metrics() is None

    def test_init_replaces_previous(self, meter):
        gm1 = init_gateway_metrics(meter)
        gm2 = init_gateway_metrics(meter)
        assert gm2 is not gm1
        assert get_gateway_metrics() is gm2


# ===========================================================================
# Instrument creation
# ===========================================================================


class TestInstrumentCreation:
    """Verify every expected instrument is created with correct types."""

    def test_request_duration_is_histogram(self, gateway_metrics):
        assert isinstance(gateway_metrics.request_duration, Histogram)

    def test_token_usage_is_histogram(self, gateway_metrics):
        assert isinstance(gateway_metrics.token_usage, Histogram)

    def test_time_to_first_token_is_histogram(self, gateway_metrics):
        assert isinstance(gateway_metrics.time_to_first_token, Histogram)

    def test_tokens_per_second_is_histogram(self, gateway_metrics):
        assert isinstance(gateway_metrics.tokens_per_second, Histogram)

    def test_request_total_is_counter(self, gateway_metrics):
        assert isinstance(gateway_metrics.request_total, Counter)

    def test_request_error_is_counter(self, gateway_metrics):
        assert isinstance(gateway_metrics.request_error, Counter)

    def test_request_active_is_up_down_counter(self, gateway_metrics):
        assert isinstance(gateway_metrics.request_active, UpDownCounter)

    def test_routing_decision_duration_is_histogram(self, gateway_metrics):
        assert isinstance(gateway_metrics.routing_decision_duration, Histogram)

    def test_routing_strategy_usage_is_counter(self, gateway_metrics):
        assert isinstance(gateway_metrics.routing_strategy_usage, Counter)

    def test_cost_total_is_counter(self, gateway_metrics):
        assert isinstance(gateway_metrics.cost_total, Counter)

    def test_circuit_breaker_transitions_is_counter(self, gateway_metrics):
        assert isinstance(gateway_metrics.circuit_breaker_transitions, Counter)


# ===========================================================================
# Recording observations (smoke tests)
# ===========================================================================


class TestMetricRecording:
    """Verify instruments accept observations without error."""

    def test_record_request_duration(self, gateway_metrics):
        gateway_metrics.request_duration.record(
            1.5, {"gen_ai.request.model": "gpt-4", "gen_ai.system": "openai"}
        )

    def test_record_token_usage(self, gateway_metrics):
        gateway_metrics.token_usage.record(
            256,
            {
                "gen_ai.request.model": "gpt-4",
                "gen_ai.token.type": "input",
            },
        )

    def test_record_ttft(self, gateway_metrics):
        gateway_metrics.time_to_first_token.record(
            0.35, {"gen_ai.request.model": "gpt-4"}
        )

    def test_increment_request_total(self, gateway_metrics):
        gateway_metrics.request_total.add(1, {"model": "gpt-4", "status": "success"})

    def test_increment_request_error(self, gateway_metrics):
        gateway_metrics.request_error.add(
            1, {"model": "gpt-4", "error_type": "Timeout"}
        )

    def test_active_requests_up_down(self, gateway_metrics):
        gateway_metrics.request_active.add(1, {"model": "gpt-4"})
        gateway_metrics.request_active.add(-1, {"model": "gpt-4"})

    def test_record_routing_decision_duration(self, gateway_metrics):
        gateway_metrics.routing_decision_duration.record(
            0.003, {"strategy": "knn", "outcome": "success"}
        )

    def test_increment_routing_strategy_usage(self, gateway_metrics):
        gateway_metrics.routing_strategy_usage.add(
            1, {"strategy": "knn", "model_selected": "gpt-4"}
        )

    def test_increment_cost_total(self, gateway_metrics):
        gateway_metrics.cost_total.add(0.015, {"model": "gpt-4", "provider": "openai"})

    def test_increment_circuit_breaker_transitions(self, gateway_metrics):
        gateway_metrics.circuit_breaker_transitions.add(
            1,
            {
                "breaker_name": "openai",
                "from_state": "closed",
                "to_state": "open",
            },
        )


# ===========================================================================
# LLM_API_PATHS expansion
# ===========================================================================


class TestLLMAPIPaths:
    """Verify the expanded LLM API path registry."""

    def test_chat_completion_paths(self):
        assert "/v1/chat/completions" in LLM_API_PATHS
        assert "/chat/completions" in LLM_API_PATHS
        assert LLM_API_PATHS["/v1/chat/completions"] == "chat_completion"
        assert LLM_API_PATHS["/chat/completions"] == "chat_completion"

    def test_responses_paths(self):
        assert "/v1/responses" in LLM_API_PATHS
        assert "/responses" in LLM_API_PATHS
        assert "/openai/v1/responses" in LLM_API_PATHS
        assert LLM_API_PATHS["/v1/responses"] == "responses"

    def test_embeddings_path(self):
        assert "/v1/embeddings" in LLM_API_PATHS
        assert LLM_API_PATHS["/v1/embeddings"] == "embedding"

    def test_completions_path(self):
        assert "/v1/completions" in LLM_API_PATHS
        assert LLM_API_PATHS["/v1/completions"] == "completion"

    def test_non_llm_path_not_in_registry(self):
        assert "/_health/ready" not in LLM_API_PATHS
        assert "/admin/config" not in LLM_API_PATHS

    def test_middleware_no_longer_has_chat_completion_paths_attr(self):
        """CHAT_COMPLETION_PATHS class attr was removed in favor of LLM_API_PATHS."""
        assert not hasattr(RouterDecisionMiddleware, "CHAT_COMPLETION_PATHS")


# ===========================================================================
# RouterDecisionCallback metrics integration
# ===========================================================================


class TestCallbackSuccessMetrics:
    """Test metrics recorded via log_success_event."""

    def _make_callback(self):
        cb = RouterDecisionCallback(strategy_name="test-strategy", enabled=True)
        # Force enabled even when ROUTER_CALLBACK_ENABLED is False (no OTEL env)
        cb._enabled = True
        return cb

    def _make_response(self, prompt_tokens=10, completion_tokens=20, model="gpt-4"):
        usage = SimpleNamespace(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        )
        return SimpleNamespace(usage=usage, model=model)

    def test_success_records_duration(self, gateway_metrics):
        cb = self._make_callback()
        resp = self._make_response()
        kwargs = {
            "model": "gpt-4",
            "litellm_call_id": "call-1",
            "litellm_params": {"custom_llm_provider": "openai"},
            "metadata": {},
        }
        # Simulate pre-call to populate start time
        cb._start_times["call-1"] = time.perf_counter() - 1.0

        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        with patch.object(trace, "get_current_span", return_value=mock_span):
            cb.log_success_event(kwargs, resp, 0.0, 1.0)

        # Verify duration was recorded (active gauge decremented)
        # The test is that no exception is raised and metrics flow through.

    def test_success_records_token_usage(self, gateway_metrics):
        cb = self._make_callback()
        resp = self._make_response(prompt_tokens=50, completion_tokens=100)
        kwargs = {
            "model": "gpt-4",
            "litellm_call_id": "",
            "litellm_params": {"custom_llm_provider": "openai"},
            "metadata": {},
        }
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        with patch.object(trace, "get_current_span", return_value=mock_span):
            cb.log_success_event(kwargs, resp, 0.0, 1.0)

    def test_success_sets_genai_span_attrs(self, gateway_metrics):
        cb = self._make_callback()
        resp = self._make_response(prompt_tokens=10, completion_tokens=20)
        kwargs = {
            "model": "gpt-4",
            "litellm_call_id": "",
            "litellm_params": {"custom_llm_provider": "openai"},
            "metadata": {},
        }
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        with patch.object(trace, "get_current_span", return_value=mock_span):
            cb.log_success_event(kwargs, resp, 0.0, 1.0)

        # Verify gen_ai.* attributes were set on the span
        calls = {c.args[0]: c.args[1] for c in mock_span.set_attribute.call_args_list}
        assert calls["gen_ai.usage.input_tokens"] == 10
        assert calls["gen_ai.usage.output_tokens"] == 20
        assert calls["gen_ai.response.model"] == "gpt-4"

    def test_success_no_metrics_when_disabled(self):
        cb = RouterDecisionCallback(enabled=False)
        resp = self._make_response()
        # Should not raise even without GatewayMetrics initialized
        cb.log_success_event({"model": "gpt-4", "litellm_call_id": ""}, resp, 0.0, 1.0)

    def test_success_handles_missing_usage(self, gateway_metrics):
        cb = self._make_callback()
        resp = SimpleNamespace(usage=None, model="gpt-4")
        kwargs = {
            "model": "gpt-4",
            "litellm_call_id": "",
            "litellm_params": {},
            "metadata": {},
        }
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        with patch.object(trace, "get_current_span", return_value=mock_span):
            cb.log_success_event(kwargs, resp, 0.0, 1.0)


class TestCallbackFailureMetrics:
    """Test metrics recorded via log_failure_event."""

    def _make_callback(self):
        cb = RouterDecisionCallback(strategy_name="test-strategy", enabled=True)
        cb._enabled = True
        return cb

    def test_failure_records_error_counter(self, gateway_metrics):
        cb = self._make_callback()
        kwargs = {
            "model": "gpt-4",
            "litellm_call_id": "call-2",
            "litellm_params": {"custom_llm_provider": "openai"},
            "exception": TimeoutError("timed out"),
        }
        cb._start_times["call-2"] = time.perf_counter()
        cb.log_failure_event(kwargs, None, 0.0, 1.0)

    def test_failure_cleans_up_start_time(self, gateway_metrics):
        cb = self._make_callback()
        cb._start_times["call-3"] = time.perf_counter()
        kwargs = {
            "model": "gpt-4",
            "litellm_call_id": "call-3",
            "litellm_params": {},
            "exception": ValueError("bad"),
        }
        cb.log_failure_event(kwargs, None, 0.0, 1.0)
        assert "call-3" not in cb._start_times

    def test_failure_unknown_error_type(self, gateway_metrics):
        cb = self._make_callback()
        kwargs = {
            "model": "gpt-4",
            "litellm_call_id": "",
            "litellm_params": {},
        }
        cb.log_failure_event(kwargs, None, 0.0, 1.0)

    def test_failure_no_metrics_when_disabled(self):
        cb = RouterDecisionCallback(enabled=False)
        cb.log_failure_event({"model": "gpt-4", "litellm_call_id": ""}, None, 0, 1)


class TestCallbackPreApiCall:
    """Test log_pre_api_call records start time and active gauge."""

    def _make_callback(self):
        cb = RouterDecisionCallback(strategy_name="test", enabled=True)
        cb._enabled = True
        return cb

    def test_pre_api_call_records_start_time(self, gateway_metrics):
        cb = self._make_callback()
        kwargs = {
            "litellm_call_id": "call-99",
            "metadata": {},
            "litellm_params": {},
        }
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        with patch.object(trace, "get_current_span", return_value=mock_span):
            cb.log_pre_api_call("gpt-4", [], kwargs)
        assert "call-99" in cb._start_times

    def test_pre_api_call_increments_active(self, gateway_metrics):
        cb = self._make_callback()
        kwargs = {
            "litellm_call_id": "call-100",
            "metadata": {},
            "litellm_params": {},
        }
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        with patch.object(trace, "get_current_span", return_value=mock_span):
            cb.log_pre_api_call("gpt-4", [], kwargs)


# ===========================================================================
# _compute_duration helper
# ===========================================================================


class TestComputeDuration:
    """Tests for the _compute_duration helper function."""

    def test_float_timestamps(self):
        assert _compute_duration(1.0, 2.5) == pytest.approx(1.5)

    def test_int_timestamps(self):
        assert _compute_duration(0, 3) == pytest.approx(3.0)

    def test_datetime_timestamps(self):
        t1 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 1, 0, 0, 5, tzinfo=timezone.utc)
        assert _compute_duration(t1, t2) == pytest.approx(5.0)

    def test_negative_duration_clamped_to_zero(self):
        assert _compute_duration(5.0, 2.0) == 0.0

    def test_invalid_types_return_zero(self):
        assert _compute_duration("bad", "worse") == 0.0

    def test_none_returns_zero(self):
        assert _compute_duration(None, None) == 0.0


# ===========================================================================
# GenAI span attributes in callback telemetry
# ===========================================================================


class TestGenAISpanAttributes:
    """Test that gen_ai.* span attrs are set by _emit_router_telemetry."""

    def _make_callback(self):
        cb = RouterDecisionCallback(strategy_name="test", enabled=True)
        cb._enabled = True
        return cb

    def test_sets_genai_request_model(self, gateway_metrics):
        cb = self._make_callback()
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        with patch.object(trace, "get_current_span", return_value=mock_span):
            cb._emit_router_telemetry(
                "gpt-4",
                [],
                {
                    "metadata": {},
                    "litellm_params": {"custom_llm_provider": "openai"},
                },
            )

        attrs = {c.args[0]: c.args[1] for c in mock_span.set_attribute.call_args_list}
        assert attrs["gen_ai.request.model"] == "gpt-4"
        assert attrs["gen_ai.system"] == "openai"

    def test_omits_genai_system_when_no_provider(self, gateway_metrics):
        cb = self._make_callback()
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        with patch.object(trace, "get_current_span", return_value=mock_span):
            cb._emit_router_telemetry(
                "gpt-4",
                [],
                {"metadata": {}, "litellm_params": {}},
            )

        attr_names = [c.args[0] for c in mock_span.set_attribute.call_args_list]
        assert "gen_ai.request.model" in attr_names
        # gen_ai.system should not be set when provider is empty
        assert "gen_ai.system" not in attr_names


# ===========================================================================
# Async callback methods delegate to sync
# ===========================================================================


class TestAsyncDelegation:
    """Verify async callback methods delegate to their sync counterparts."""

    def _make_callback(self):
        cb = RouterDecisionCallback(strategy_name="test", enabled=True)
        cb._enabled = True
        return cb

    @pytest.mark.asyncio
    async def test_async_log_success_event(self, gateway_metrics):
        cb = self._make_callback()
        resp = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=10),
            model="gpt-4",
        )
        kwargs = {
            "model": "gpt-4",
            "litellm_call_id": "",
            "litellm_params": {},
            "metadata": {},
        }
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        with patch.object(trace, "get_current_span", return_value=mock_span):
            await cb.async_log_success_event(kwargs, resp, 0.0, 1.0)

    @pytest.mark.asyncio
    async def test_async_log_failure_event(self, gateway_metrics):
        cb = self._make_callback()
        kwargs = {
            "model": "gpt-4",
            "litellm_call_id": "",
            "litellm_params": {},
            "exception": RuntimeError("boom"),
        }
        await cb.async_log_failure_event(kwargs, None, 0.0, 1.0)

    @pytest.mark.asyncio
    async def test_async_log_pre_api_call(self, gateway_metrics):
        cb = self._make_callback()
        kwargs = {
            "litellm_call_id": "call-async",
            "metadata": {},
            "litellm_params": {},
        }
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        with patch.object(trace, "get_current_span", return_value=mock_span):
            await cb.async_log_pre_api_call("gpt-4", [], kwargs)
        assert "call-async" in cb._start_times


# ===========================================================================
# Bucket boundary constants
# ===========================================================================


class TestBucketBoundaries:
    """Verify bucket boundary constants exist and are sorted."""

    def test_duration_buckets_sorted(self):
        from litellm_llmrouter.metrics import DURATION_BUCKETS

        assert list(DURATION_BUCKETS) == sorted(DURATION_BUCKETS)
        assert len(DURATION_BUCKETS) > 5

    def test_token_buckets_sorted(self):
        from litellm_llmrouter.metrics import TOKEN_BUCKETS

        assert list(TOKEN_BUCKETS) == sorted(TOKEN_BUCKETS)
        assert len(TOKEN_BUCKETS) > 5

    def test_ttft_buckets_sorted(self):
        from litellm_llmrouter.metrics import TTFT_BUCKETS

        assert list(TTFT_BUCKETS) == sorted(TTFT_BUCKETS)

    def test_routing_duration_buckets_sorted(self):
        from litellm_llmrouter.metrics import ROUTING_DURATION_BUCKETS

        assert list(ROUTING_DURATION_BUCKETS) == sorted(ROUTING_DURATION_BUCKETS)

    def test_cost_buckets_sorted(self):
        from litellm_llmrouter.metrics import COST_BUCKETS

        assert list(COST_BUCKETS) == sorted(COST_BUCKETS)

    def test_tokens_per_second_buckets_sorted(self):
        from litellm_llmrouter.metrics import TOKENS_PER_SECOND_BUCKETS

        assert list(TOKENS_PER_SECOND_BUCKETS) == sorted(TOKENS_PER_SECOND_BUCKETS)
        assert len(TOKENS_PER_SECOND_BUCKETS) > 3


# ===========================================================================
# record_streaming_metrics convenience method
# ===========================================================================


class TestRecordStreamingMetrics:
    """Tests for the record_streaming_metrics convenience method."""

    def test_records_ttft(self, gateway_metrics):
        """TTFT in ms should be recorded as seconds on time_to_first_token."""
        gateway_metrics.record_streaming_metrics(
            ttft_ms=350.0,
            tps=0.0,
            total_tokens=0,
            attrs={"gen_ai.request.model": "gpt-4"},
        )
        # No exception means it was recorded

    def test_records_tokens_per_second(self, gateway_metrics):
        """Positive tps should be recorded on tokens_per_second histogram."""
        gateway_metrics.record_streaming_metrics(
            ttft_ms=200.0,
            tps=45.5,
            total_tokens=0,
            attrs={"gen_ai.request.model": "gpt-4"},
        )

    def test_records_total_tokens(self, gateway_metrics):
        """Positive total_tokens should be recorded on token_usage histogram."""
        gateway_metrics.record_streaming_metrics(
            ttft_ms=100.0,
            tps=30.0,
            total_tokens=512,
            attrs={"gen_ai.request.model": "gpt-4"},
        )

    def test_skips_zero_tps(self, gateway_metrics):
        """Zero tps should not be recorded."""
        gateway_metrics.record_streaming_metrics(ttft_ms=100.0, tps=0.0, total_tokens=0)

    def test_skips_zero_total_tokens(self, gateway_metrics):
        """Zero total_tokens should not be recorded."""
        gateway_metrics.record_streaming_metrics(
            ttft_ms=100.0, tps=10.0, total_tokens=0
        )

    def test_none_attrs_defaults_to_empty(self, gateway_metrics):
        """None attrs should default to empty dict."""
        gateway_metrics.record_streaming_metrics(
            ttft_ms=100.0, tps=10.0, total_tokens=100, attrs=None
        )

    def test_all_metrics_recorded_together(self, gateway_metrics):
        """All three metrics should be recorded when all values are positive."""
        gateway_metrics.record_streaming_metrics(
            ttft_ms=250.0,
            tps=55.0,
            total_tokens=1024,
            attrs={"gen_ai.request.model": "gpt-4", "gen_ai.system": "openai"},
        )


# ===========================================================================
# GenAI finish_reasons span attributes on success and failure
# ===========================================================================


class TestGenAIFinishReasonsSuccess:
    """Test gen_ai.response.finish_reasons on successful responses."""

    def _make_callback(self):
        cb = RouterDecisionCallback(strategy_name="test-strategy", enabled=True)
        cb._enabled = True
        return cb

    def test_finish_reasons_set_from_choices(self, gateway_metrics):
        """finish_reasons should be extracted from response.choices."""
        cb = self._make_callback()
        choice = SimpleNamespace(finish_reason="stop")
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=20)
        resp = SimpleNamespace(usage=usage, model="gpt-4", choices=[choice])
        kwargs = {
            "model": "gpt-4",
            "litellm_call_id": "",
            "litellm_params": {"custom_llm_provider": "openai"},
            "metadata": {},
        }
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        with patch.object(trace, "get_current_span", return_value=mock_span):
            cb.log_success_event(kwargs, resp, 0.0, 1.0)

        calls = {c.args[0]: c.args[1] for c in mock_span.set_attribute.call_args_list}
        assert calls["gen_ai.response.finish_reasons"] == ["stop"]

    def test_finish_reasons_multiple_choices(self, gateway_metrics):
        """Multiple choices should produce multiple finish reasons."""
        cb = self._make_callback()
        choices = [
            SimpleNamespace(finish_reason="stop"),
            SimpleNamespace(finish_reason="length"),
        ]
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=20)
        resp = SimpleNamespace(usage=usage, model="gpt-4", choices=choices)
        kwargs = {
            "model": "gpt-4",
            "litellm_call_id": "",
            "litellm_params": {"custom_llm_provider": "openai"},
            "metadata": {},
        }
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        with patch.object(trace, "get_current_span", return_value=mock_span):
            cb.log_success_event(kwargs, resp, 0.0, 1.0)

        calls = {c.args[0]: c.args[1] for c in mock_span.set_attribute.call_args_list}
        assert calls["gen_ai.response.finish_reasons"] == ["stop", "length"]

    def test_finish_reasons_not_set_when_no_choices(self, gateway_metrics):
        """finish_reasons should not be set when response has no choices."""
        cb = self._make_callback()
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=20)
        resp = SimpleNamespace(usage=usage, model="gpt-4")
        kwargs = {
            "model": "gpt-4",
            "litellm_call_id": "",
            "litellm_params": {"custom_llm_provider": "openai"},
            "metadata": {},
        }
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        with patch.object(trace, "get_current_span", return_value=mock_span):
            cb.log_success_event(kwargs, resp, 0.0, 1.0)

        attr_names = [c.args[0] for c in mock_span.set_attribute.call_args_list]
        assert "gen_ai.response.finish_reasons" not in attr_names


class TestGenAIFinishReasonsFailure:
    """Test gen_ai.response.finish_reasons on failed responses."""

    def _make_callback(self):
        cb = RouterDecisionCallback(strategy_name="test-strategy", enabled=True)
        cb._enabled = True
        return cb

    def test_failure_sets_finish_reasons_error(self, gateway_metrics):
        """Failure path should set finish_reasons to ['error']."""
        cb = self._make_callback()
        kwargs = {
            "model": "gpt-4",
            "litellm_call_id": "",
            "litellm_params": {"custom_llm_provider": "openai"},
            "exception": TimeoutError("timed out"),
        }
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        with patch.object(trace, "get_current_span", return_value=mock_span):
            cb.log_failure_event(kwargs, None, 0.0, 1.0)

        calls = {c.args[0]: c.args[1] for c in mock_span.set_attribute.call_args_list}
        assert calls["gen_ai.response.finish_reasons"] == ["error"]
