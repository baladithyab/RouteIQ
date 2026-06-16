"""
Unit Tests for router_decision_callback.py
===========================================

Tests for the ASGI middleware and LiteLLM callback:
- RouterDecisionMiddleware (ASGI pattern, path filtering, telemetry emission, metrics)
- RouterDecisionCallback (log_pre_api_call, log_success_event, log_failure_event)
- _compute_duration helper
- register_router_decision_callback / register_router_decision_middleware
- LLM_API_PATHS coverage (all Responses API paths present)
"""

import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch


import os

from litellm_llmrouter.router_decision_callback import (
    LLM_API_PATHS,
    RouterDecisionCallback,
    RouterDecisionMiddleware,
    _compute_duration,
    _governance_spend_tracking_enabled,
    get_router_decision_callback,
    register_router_decision_callback,
    register_router_decision_middleware,
)
from litellm_llmrouter.settings import get_settings, reset_settings


# =============================================================================
# Helpers
# =============================================================================


def _make_http_scope(path: str, method: str = "POST") -> dict:
    """Create a minimal ASGI HTTP scope."""
    return {
        "type": "http",
        "path": path,
        "method": method,
    }


def _make_ws_scope(path: str) -> dict:
    """Create a minimal ASGI WebSocket scope."""
    return {"type": "websocket", "path": path}


async def _noop_receive():
    return {"type": "http.disconnect"}


async def _noop_send(message):
    pass


# =============================================================================
# LLM_API_PATHS Registry
# =============================================================================


class TestLLMAPIPaths:
    def test_chat_completions_paths(self):
        assert "/v1/chat/completions" in LLM_API_PATHS
        assert "/chat/completions" in LLM_API_PATHS

    def test_responses_api_paths(self):
        """All three Responses API paths must be present."""
        assert "/v1/responses" in LLM_API_PATHS
        assert "/responses" in LLM_API_PATHS
        assert "/openai/v1/responses" in LLM_API_PATHS

    def test_embeddings_path(self):
        assert "/v1/embeddings" in LLM_API_PATHS

    def test_completions_path(self):
        assert "/v1/completions" in LLM_API_PATHS

    def test_operation_types(self):
        assert LLM_API_PATHS["/v1/chat/completions"] == "chat_completion"
        assert LLM_API_PATHS["/v1/responses"] == "responses"
        assert LLM_API_PATHS["/v1/embeddings"] == "embedding"
        assert LLM_API_PATHS["/v1/completions"] == "completion"


# =============================================================================
# RouterDecisionMiddleware (ASGI)
# =============================================================================


class TestRouterDecisionMiddleware:
    async def test_passthrough_non_http(self):
        """WebSocket and lifespan scopes pass through unchanged."""
        inner_app = AsyncMock()
        mw = RouterDecisionMiddleware(inner_app)

        scope = _make_ws_scope("/v1/chat/completions")
        await mw(scope, _noop_receive, _noop_send)
        inner_app.assert_awaited_once_with(scope, _noop_receive, _noop_send)

    async def test_passthrough_get_request(self):
        """GET requests to LLM paths are not instrumented."""
        inner_app = AsyncMock()
        mw = RouterDecisionMiddleware(inner_app)

        scope = _make_http_scope("/v1/chat/completions", method="GET")
        await mw(scope, _noop_receive, _noop_send)
        inner_app.assert_awaited_once()

    async def test_passthrough_non_llm_path(self):
        """POST to non-LLM paths passes through."""
        inner_app = AsyncMock()
        mw = RouterDecisionMiddleware(inner_app)

        scope = _make_http_scope("/_health/live", method="POST")
        await mw(scope, _noop_receive, _noop_send)
        inner_app.assert_awaited_once()

    @patch("litellm_llmrouter.router_decision_callback.ROUTER_CALLBACK_ENABLED", True)
    async def test_instruments_post_to_llm_path(self):
        """POST to a known LLM path triggers telemetry emission."""
        inner_app = AsyncMock()
        mw = RouterDecisionMiddleware(inner_app)

        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        with (
            patch("litellm_llmrouter.router_decision_callback.trace") as mock_trace,
            patch(
                "litellm_llmrouter.observability.set_router_decision_attributes"
            ) as mock_set_attrs,
            patch(
                "litellm_llmrouter.metrics.get_gateway_metrics",
                return_value=None,
            ),
        ):
            mock_trace.get_current_span.return_value = mock_span

            scope = _make_http_scope("/v1/chat/completions")
            await mw(scope, _noop_receive, _noop_send)

            # Inner app should still be called (streaming-safe passthrough)
            inner_app.assert_awaited_once()
            # Telemetry should have been emitted
            mock_set_attrs.assert_called_once()

    @patch("litellm_llmrouter.router_decision_callback.ROUTER_CALLBACK_ENABLED", True)
    async def test_instruments_responses_api_path(self):
        """POST to /v1/responses triggers telemetry."""
        inner_app = AsyncMock()
        mw = RouterDecisionMiddleware(inner_app)

        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        with (
            patch("litellm_llmrouter.router_decision_callback.trace") as mock_trace,
            patch(
                "litellm_llmrouter.observability.set_router_decision_attributes"
            ) as mock_set_attrs,
            patch(
                "litellm_llmrouter.metrics.get_gateway_metrics",
                return_value=None,
            ),
        ):
            mock_trace.get_current_span.return_value = mock_span

            scope = _make_http_scope("/v1/responses")
            await mw(scope, _noop_receive, _noop_send)

            inner_app.assert_awaited_once()
            mock_set_attrs.assert_called_once()
            # Verify gen_ai.operation.name is set to "responses"
            mock_span.set_attribute.assert_called_with(
                "gen_ai.operation.name", "responses"
            )

    @patch("litellm_llmrouter.router_decision_callback.ROUTER_CALLBACK_ENABLED", False)
    async def test_disabled_skips_instrumentation(self):
        """When disabled, POST to LLM path passes through without telemetry."""
        inner_app = AsyncMock()
        mw = RouterDecisionMiddleware(inner_app)

        with patch(
            "litellm_llmrouter.observability.set_router_decision_attributes"
        ) as mock_set_attrs:
            scope = _make_http_scope("/v1/chat/completions")
            await mw(scope, _noop_receive, _noop_send)

            inner_app.assert_awaited_once()
            mock_set_attrs.assert_not_called()

    @patch("litellm_llmrouter.router_decision_callback.ROUTER_CALLBACK_ENABLED", True)
    async def test_metrics_increment(self):
        """POST to LLM path increments gateway metrics."""
        inner_app = AsyncMock()
        mw = RouterDecisionMiddleware(inner_app)

        mock_metrics = MagicMock()

        with (
            patch("litellm_llmrouter.router_decision_callback.trace") as mock_trace,
            patch("litellm_llmrouter.observability.set_router_decision_attributes"),
            patch(
                "litellm_llmrouter.metrics.get_gateway_metrics",
                return_value=mock_metrics,
            ),
        ):
            mock_span = MagicMock()
            mock_span.is_recording.return_value = True
            mock_trace.get_current_span.return_value = mock_span

            scope = _make_http_scope("/v1/chat/completions")
            await mw(scope, _noop_receive, _noop_send)

            mock_metrics.request_total.add.assert_called_once()
            mock_metrics.strategy_usage.add.assert_called_once()

    @patch("litellm_llmrouter.router_decision_callback.ROUTER_CALLBACK_ENABLED", True)
    async def test_non_recording_span_skips_telemetry(self):
        """Non-recording span doesn't get attributes set."""
        inner_app = AsyncMock()
        mw = RouterDecisionMiddleware(inner_app)

        mock_span = MagicMock()
        mock_span.is_recording.return_value = False

        with (
            patch("litellm_llmrouter.router_decision_callback.trace") as mock_trace,
            patch(
                "litellm_llmrouter.observability.set_router_decision_attributes"
            ) as mock_set_attrs,
            patch(
                "litellm_llmrouter.metrics.get_gateway_metrics",
                return_value=None,
            ),
        ):
            mock_trace.get_current_span.return_value = mock_span
            scope = _make_http_scope("/v1/chat/completions")
            await mw(scope, _noop_receive, _noop_send)

            # set_router_decision_attributes should NOT be called
            mock_set_attrs.assert_not_called()
            # But inner app should still be called
            inner_app.assert_awaited_once()


# =============================================================================
# register_router_decision_middleware
# =============================================================================


class TestRegisterMiddleware:
    @patch("litellm_llmrouter.router_decision_callback.ROUTER_CALLBACK_ENABLED", True)
    def test_wraps_asgi_app(self):
        mock_app = MagicMock()
        mock_app.app = MagicMock()

        result = register_router_decision_middleware(mock_app)
        assert result is True
        assert isinstance(mock_app.app, RouterDecisionMiddleware)

    @patch("litellm_llmrouter.router_decision_callback.ROUTER_CALLBACK_ENABLED", False)
    def test_disabled_returns_false(self):
        mock_app = MagicMock()
        result = register_router_decision_middleware(mock_app)
        assert result is False


# =============================================================================
# RouterDecisionCallback
# =============================================================================


class TestRouterDecisionCallback:
    @patch("litellm_llmrouter.router_decision_callback.ROUTER_CALLBACK_ENABLED", True)
    def test_log_pre_api_call_emits_telemetry(self):
        callback = RouterDecisionCallback(strategy_name="test-strategy", enabled=True)

        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        with (
            patch("litellm_llmrouter.router_decision_callback.trace") as mock_trace,
            patch(
                "litellm_llmrouter.observability.set_router_decision_attributes"
            ) as mock_set_attrs,
            patch(
                "litellm_llmrouter.metrics.get_gateway_metrics",
                return_value=None,
            ),
        ):
            mock_trace.get_current_span.return_value = mock_span

            callback.log_pre_api_call(
                model="gpt-4",
                messages=[{"role": "user", "content": "hello"}],
                kwargs={"metadata": {}},
            )

            mock_set_attrs.assert_called_once()
            call_kwargs = mock_set_attrs.call_args[1]
            assert call_kwargs["strategy"] == "test-strategy"
            assert call_kwargs["model_selected"] == "gpt-4"

    @patch("litellm_llmrouter.router_decision_callback.ROUTER_CALLBACK_ENABLED", True)
    def test_log_pre_api_call_disabled(self):
        callback = RouterDecisionCallback(enabled=False)

        with patch(
            "litellm_llmrouter.observability.set_router_decision_attributes"
        ) as mock_set_attrs:
            callback.log_pre_api_call(
                model="gpt-4",
                messages=[],
                kwargs={},
            )
            mock_set_attrs.assert_not_called()

    @patch("litellm_llmrouter.router_decision_callback.ROUTER_CALLBACK_ENABLED", True)
    def test_log_success_event_records_metrics(self):
        callback = RouterDecisionCallback(enabled=True)

        mock_metrics = MagicMock()
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        with (
            patch("litellm_llmrouter.router_decision_callback.trace") as mock_trace,
            patch(
                "litellm_llmrouter.metrics.get_gateway_metrics",
                return_value=mock_metrics,
            ),
        ):
            mock_trace.get_current_span.return_value = mock_span

            # Create a mock response with usage
            mock_response = MagicMock()
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 20
            mock_response.model = "gpt-4"

            callback.log_success_event(
                kwargs={"model": "gpt-4", "litellm_params": {}},
                response_obj=mock_response,
                start_time=time.time(),
                end_time=time.time() + 0.5,
            )

            mock_metrics.request_duration.record.assert_called_once()
            mock_metrics.request_total.add.assert_called_once()

    @patch("litellm_llmrouter.router_decision_callback.ROUTER_CALLBACK_ENABLED", True)
    def test_log_failure_event_records_error(self):
        callback = RouterDecisionCallback(enabled=True)

        mock_metrics = MagicMock()

        with patch(
            "litellm_llmrouter.metrics.get_gateway_metrics",
            return_value=mock_metrics,
        ):
            callback.log_failure_event(
                kwargs={
                    "model": "gpt-4",
                    "litellm_params": {"custom_llm_provider": "openai"},
                    "exception": ValueError("bad request"),
                },
                response_obj=None,
                start_time=time.time(),
                end_time=time.time() + 0.1,
            )

            mock_metrics.request_error.add.assert_called_once()
            call_args = mock_metrics.request_error.add.call_args
            assert call_args[0][0] == 1  # count
            attrs = call_args[0][1]  # attributes dict
            assert attrs["error_type"] == "ValueError"

    @patch("litellm_llmrouter.router_decision_callback.ROUTER_CALLBACK_ENABLED", True)
    def test_call_count_increments(self):
        callback = RouterDecisionCallback(enabled=True)
        assert callback._call_count == 0

        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        with (
            patch("litellm_llmrouter.router_decision_callback.trace") as mock_trace,
            patch("litellm_llmrouter.observability.set_router_decision_attributes"),
            patch(
                "litellm_llmrouter.metrics.get_gateway_metrics",
                return_value=None,
            ),
        ):
            mock_trace.get_current_span.return_value = mock_span

            callback.log_pre_api_call("gpt-4", [], {"metadata": {}})
            callback.log_pre_api_call("gpt-4", [], {"metadata": {}})
            assert callback._call_count == 2

    @patch("litellm_llmrouter.router_decision_callback.ROUTER_CALLBACK_ENABLED", True)
    def test_specific_deployment_reason(self):
        callback = RouterDecisionCallback(enabled=True)

        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        with (
            patch("litellm_llmrouter.router_decision_callback.trace") as mock_trace,
            patch(
                "litellm_llmrouter.observability.set_router_decision_attributes"
            ) as mock_set_attrs,
            patch(
                "litellm_llmrouter.metrics.get_gateway_metrics",
                return_value=None,
            ),
        ):
            mock_trace.get_current_span.return_value = mock_span

            callback.log_pre_api_call(
                "gpt-4", [], {"metadata": {"specific_deployment": True}}
            )

            call_kwargs = mock_set_attrs.call_args[1]
            assert call_kwargs["reason"] == "specific_deployment_requested"

    @patch("litellm_llmrouter.router_decision_callback.ROUTER_CALLBACK_ENABLED", True)
    def test_fallback_reason(self):
        callback = RouterDecisionCallback(enabled=True)

        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        with (
            patch("litellm_llmrouter.router_decision_callback.trace") as mock_trace,
            patch(
                "litellm_llmrouter.observability.set_router_decision_attributes"
            ) as mock_set_attrs,
            patch(
                "litellm_llmrouter.metrics.get_gateway_metrics",
                return_value=None,
            ),
        ):
            mock_trace.get_current_span.return_value = mock_span

            callback.log_pre_api_call("gpt-4", [], {"metadata": {"fallback": True}})

            call_kwargs = mock_set_attrs.call_args[1]
            assert call_kwargs["reason"] == "fallback_triggered"
            assert call_kwargs["fallback_triggered"] is True

    async def test_async_log_pre_api_call(self):
        """Async variant delegates to sync."""
        callback = RouterDecisionCallback(enabled=False)
        # Should not raise
        await callback.async_log_pre_api_call("gpt-4", [], {})

    async def test_async_log_success_event(self):
        callback = RouterDecisionCallback(enabled=False)
        await callback.async_log_success_event({}, None, 0.0, 0.0)

    async def test_async_log_failure_event(self):
        callback = RouterDecisionCallback(enabled=False)
        await callback.async_log_failure_event({}, None, 0.0, 0.0)

    async def test_async_post_call_hooks(self):
        callback = RouterDecisionCallback(enabled=False)
        await callback.async_post_call_success_hook({}, None, None)
        await callback.async_post_call_failure_hook({}, Exception("x"), None)


# =============================================================================
# _compute_duration
# =============================================================================


class TestComputeDuration:
    def test_float_timestamps(self):
        result = _compute_duration(100.0, 100.5)
        assert abs(result - 0.5) < 0.001

    def test_int_timestamps(self):
        result = _compute_duration(10, 12)
        assert result == 2.0

    def test_datetime_timestamps(self):
        start = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
        result = _compute_duration(start, end)
        assert abs(result - 1.0) < 0.01

    def test_negative_duration_clamped_to_zero(self):
        result = _compute_duration(100.0, 50.0)
        assert result == 0.0

    def test_incompatible_types_return_zero(self):
        result = _compute_duration("start", "end")
        assert result == 0.0

    def test_none_types_return_zero(self):
        result = _compute_duration(None, None)
        assert result == 0.0


# =============================================================================
# register_router_decision_callback
# =============================================================================


class TestRegisterCallback:
    @patch("litellm_llmrouter.router_decision_callback.ROUTER_CALLBACK_ENABLED", False)
    def test_disabled_returns_none(self):
        result = register_router_decision_callback()
        assert result is None

    @patch("litellm_llmrouter.router_decision_callback.ROUTER_CALLBACK_ENABLED", True)
    def test_registers_with_litellm(self):
        import litellm

        original_callbacks = getattr(litellm, "callbacks", [])
        # Clean slate
        litellm.callbacks = []

        try:
            result = register_router_decision_callback()
            assert result is not None
            assert isinstance(result, RouterDecisionCallback)
            assert result in litellm.callbacks
        finally:
            litellm.callbacks = original_callbacks

    @patch("litellm_llmrouter.router_decision_callback.ROUTER_CALLBACK_ENABLED", True)
    def test_avoids_duplicate_registration(self):
        import litellm

        original_callbacks = getattr(litellm, "callbacks", [])
        litellm.callbacks = []

        try:
            first = register_router_decision_callback()
            second = register_router_decision_callback()
            assert first is second
            # Should only be one instance
            count = sum(
                1 for c in litellm.callbacks if isinstance(c, RouterDecisionCallback)
            )
            assert count == 1
        finally:
            litellm.callbacks = original_callbacks


# =============================================================================
# get_router_decision_callback
# =============================================================================


class TestGetRouterDecisionCallback:
    def test_returns_class(self):
        cls = get_router_decision_callback()
        assert cls is RouterDecisionCallback


# =============================================================================
# RouteIQ-9f9f -- governance spend tracking gate reads settings (ADR-0013)
# =============================================================================
#
# The live reader migrated off a direct ``os.getenv`` to
# ``get_settings().llmrouter_governance_spend_tracking``, with the raw env read
# kept ONLY as a graceful fallback when ``get_settings()`` throws.


class TestGovernanceSpendTrackingGate:
    def test_default_is_on(self):
        """No override -> default ON (settings default True)."""
        reset_settings()
        get_settings()
        assert _governance_spend_tracking_enabled() is True

    def test_honors_settings_value_true(self):
        """RouteIQ-9f9f: the gate honors the settings field (set via override).

        The field carries a ``validation_alias`` so the override is supplied
        under the alias key (the Python-name kwarg is ignored by Pydantic when an
        alias is present)."""
        reset_settings()
        get_settings(LLMROUTER_GOVERNANCE_SPEND_TRACKING=True)
        assert _governance_spend_tracking_enabled() is True

    def test_honors_settings_value_false(self):
        """A False settings value disables the gate (read via get_settings)."""
        reset_settings()
        get_settings(LLMROUTER_GOVERNANCE_SPEND_TRACKING=False)
        assert _governance_spend_tracking_enabled() is False

    def test_settings_read_from_env_alias(self, monkeypatch):
        """The settings field binds the bare env name via validation_alias."""
        monkeypatch.setenv("LLMROUTER_GOVERNANCE_SPEND_TRACKING", "false")
        reset_settings()
        get_settings()
        assert _governance_spend_tracking_enabled() is False

    def test_env_fallback_when_settings_throws(self, monkeypatch):
        """If get_settings() throws, the gate falls back to the raw env read."""
        import litellm_llmrouter.settings as settings_mod

        def _boom(*a, **kw):
            raise RuntimeError("settings exploded")

        monkeypatch.setattr(settings_mod, "get_settings", _boom)

        # Env says OFF -> the fallback env reader must honor it.
        with patch.dict(
            os.environ, {"LLMROUTER_GOVERNANCE_SPEND_TRACKING": "no"}, clear=False
        ):
            assert _governance_spend_tracking_enabled() is False

        # Env says ON -> fallback honors it too.
        with patch.dict(
            os.environ, {"LLMROUTER_GOVERNANCE_SPEND_TRACKING": "on"}, clear=False
        ):
            assert _governance_spend_tracking_enabled() is True

    def test_flag_is_boot_time_not_per_request(self, monkeypatch):
        """RouteIQ-9fce: the flag is resolved at BOOT via the cached Settings
        (ADR-0013), NOT per request.

        Flipping the env var mid-process does NOT change the answer until
        ``reset_settings()`` rebuilds Settings.  This pins the documented
        behavior (flip + restart in prod, or reset_settings in tests).
        """
        # Boot with the flag ON (default) and cache the Settings instance.
        monkeypatch.delenv("LLMROUTER_GOVERNANCE_SPEND_TRACKING", raising=False)
        reset_settings()
        get_settings()
        assert _governance_spend_tracking_enabled() is True

        # Flip the env to OFF *without* resetting -> still ON (boot-cached).
        monkeypatch.setenv("LLMROUTER_GOVERNANCE_SPEND_TRACKING", "false")
        assert _governance_spend_tracking_enabled() is True

        # Only after reset_settings() does the new value take effect.
        reset_settings()
        get_settings()
        assert _governance_spend_tracking_enabled() is False
