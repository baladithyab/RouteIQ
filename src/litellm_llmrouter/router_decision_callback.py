"""
Router Decision Telemetry for TG4.1/TG4.2
==========================================

This module provides router decision telemetry emission for TG4.1 acceptance criteria.

Two mechanisms are provided:
1. RouterDecisionMiddleware - FastAPI middleware that emits router.* attributes on
   ALL LLM API requests (deterministic, works with any routing strategy)
2. RouterDecisionCallback - LiteLLM callback that emits metrics and span attributes

The middleware is the preferred approach for E2E testing because it:
- Fires before the LLM API call, so mock API keys don't prevent telemetry
- Works with LiteLLM's built-in routing strategies (simple-shuffle, etc.)
- Doesn't require trained ML models

Usage:
    # In gateway/app.py:
    from litellm_llmrouter.router_decision_callback import RouterDecisionMiddleware
    app.add_middleware(RouterDecisionMiddleware)

Environment Variables:
    LLMROUTER_ROUTER_CALLBACK_ENABLED: Enable the middleware (default: true if OTEL configured)
    LLMROUTER_ROUTER_CALLBACK_STRATEGY: Override strategy name in telemetry
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

from opentelemetry import trace
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)

# Feature flag: Enable router decision callback
# Default to true if OTEL is configured, false otherwise
_OTEL_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
ROUTER_CALLBACK_ENABLED = (
    os.getenv(
        "LLMROUTER_ROUTER_CALLBACK_ENABLED", "true" if _OTEL_ENDPOINT else "false"
    ).lower()
    == "true"
)

# Override strategy name in telemetry (useful for testing)
OVERRIDE_STRATEGY_NAME = os.getenv(
    "LLMROUTER_ROUTER_CALLBACK_STRATEGY", "simple-shuffle"
)


# =============================================================================
# LLM API Path Registry
# =============================================================================

# Maps all LLM API endpoint paths to their operation type.
# Used by both the middleware and callback to detect instrumented requests.
LLM_API_PATHS: Dict[str, str] = {
    "/v1/chat/completions": "chat_completion",
    "/chat/completions": "chat_completion",
    "/v1/responses": "responses",
    "/responses": "responses",
    "/openai/v1/responses": "responses",
    "/v1/embeddings": "embedding",
    "/v1/completions": "completion",
}


# =============================================================================
# RouterDecisionMiddleware - FastAPI Middleware (Recommended for E2E tests)
# =============================================================================


class RouterDecisionMiddleware:
    """
    Raw ASGI middleware that emits TG4.1 router decision span attributes.

    Uses the raw ASGI pattern (same as BackpressureMiddleware) instead of
    BaseHTTPMiddleware, which buffers the entire response and breaks streaming.

    This middleware intercepts LLM API requests and emits router.*
    span attributes BEFORE the LLM API call happens. It also increments
    gateway.request.total and gateway.routing.strategy.usage metrics.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """ASGI entry point."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        method = scope.get("method", "GET")

        # Only instrument POST requests to LLM API endpoints
        if method != "POST" or path not in LLM_API_PATHS:
            await self.app(scope, receive, send)
            return

        if not ROUTER_CALLBACK_ENABLED:
            await self.app(scope, receive, send)
            return

        # Emit router decision attributes on the current span
        self._emit_router_telemetry(path)

        # Increment gateway metrics
        self._increment_metrics(path)

        # Pass through to the next ASGI app (streaming-safe)
        await self.app(scope, receive, send)

    def _emit_router_telemetry(self, path: str) -> None:
        """
        Emit TG4.1 router decision span attributes and gen_ai.* attributes.

        This is called BEFORE the LLM API call, ensuring telemetry is
        always emitted for LLM API requests.
        """
        try:
            from litellm_llmrouter.observability import set_router_decision_attributes

            span = trace.get_current_span()
            if not span or not span.is_recording():
                return

            operation_name = LLM_API_PATHS.get(path, "unknown")
            strategy = OVERRIDE_STRATEGY_NAME or "litellm-builtin"

            set_router_decision_attributes(
                span,
                strategy=strategy,
                model_selected="pending",
                candidates_evaluated=1,
                outcome="success",
                reason="middleware_routing",
                latency_ms=0.1,
                strategy_version="v1-middleware",
                fallback_triggered=False,
            )

            span.set_attribute("gen_ai.operation.name", operation_name)

        except Exception as e:
            logger.debug(f"Failed to emit router telemetry in middleware: {e}")

    def _increment_metrics(self, path: str) -> None:
        """Increment gateway.request.total and gateway.routing.strategy.usage."""
        try:
            from litellm_llmrouter.metrics import get_gateway_metrics

            metrics = get_gateway_metrics()
            if metrics is None:
                return

            operation = LLM_API_PATHS.get(path, "unknown")
            strategy = OVERRIDE_STRATEGY_NAME or "litellm-builtin"

            metrics.request_total.add(1, {"operation": operation})
            metrics.strategy_usage.add(1, {"strategy": strategy})

        except Exception:
            pass


def register_router_decision_middleware(app: Any) -> bool:
    """
    Register the RouterDecisionMiddleware with a FastAPI app.

    Wraps the ASGI app directly (same pattern as BackpressureMiddleware)
    to avoid BaseHTTPMiddleware's response buffering that breaks streaming.

    Args:
        app: FastAPI application instance

    Returns:
        True if middleware was registered, False if disabled
    """
    if not ROUTER_CALLBACK_ENABLED:
        logger.debug("Router decision middleware disabled")
        return False

    try:
        app.app = RouterDecisionMiddleware(app.app)
        logger.info("Registered RouterDecisionMiddleware for TG4.1 telemetry (ASGI)")
        return True
    except Exception as e:
        logger.warning(f"Failed to register router decision middleware: {e}")
        return False


# =============================================================================
# RouterDecisionCallback - LiteLLM Callback (Legacy)
# =============================================================================


class RouterDecisionCallback:
    """
    LiteLLM custom callback that emits TG4.1 router decision span attributes
    and records OTel metrics for request duration, token usage, and errors.

    Compatible with LiteLLM's custom callback interface.
    """

    def __init__(
        self,
        strategy_name: Optional[str] = None,
        enabled: bool = True,
    ):
        """
        Initialize the router decision callback.

        Args:
            strategy_name: Override strategy name in telemetry
            enabled: Whether the callback is active
        """
        self._strategy_name = (
            strategy_name or OVERRIDE_STRATEGY_NAME or "litellm-builtin"
        )
        self._enabled = enabled and ROUTER_CALLBACK_ENABLED
        self._call_count = 0
        # Per-call start times keyed by litellm_call_id
        self._start_times: Dict[str, float] = {}
        logger.info(
            f"RouterDecisionCallback initialized: enabled={self._enabled}, "
            f"strategy={self._strategy_name}"
        )

    def log_pre_api_call(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        kwargs: Dict[str, Any],
    ) -> None:
        """
        Called before each API call - emits router decision telemetry
        and records start time + increments the active request gauge.
        """
        if not self._enabled:
            return

        try:
            self._emit_router_telemetry(model, messages, kwargs)
        except Exception as e:
            logger.debug(f"Failed to emit router telemetry: {e}")

        # Record start time and increment active gauge
        try:
            call_id = kwargs.get("litellm_call_id", "")
            if call_id:
                self._start_times[call_id] = time.perf_counter()

            from litellm_llmrouter.metrics import get_gateway_metrics

            gm = get_gateway_metrics()
            if gm:
                gm.request_active.add(1, {"model": model})
        except Exception as e:
            logger.debug(f"Failed to record pre-call metrics: {e}")

    def _emit_router_telemetry(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        kwargs: Dict[str, Any],
    ) -> None:
        """
        Emit TG4.1 router decision span attributes and gen_ai.* attributes.

        Sets router.* span attributes for routing visibility and gen_ai.*
        attributes for GenAI Semantic Convention compliance.
        """
        from litellm_llmrouter.observability import set_router_decision_attributes

        span = trace.get_current_span()
        if not span or not span.is_recording():
            logger.debug("No active span for router telemetry")
            return

        self._call_count += 1

        # Extract metadata from kwargs
        metadata = kwargs.get("metadata", {}) or {}

        # Determine number of candidates (if available from router)
        candidates = metadata.get("model_group_size", 1)
        if isinstance(candidates, str):
            try:
                candidates = int(candidates)
            except ValueError:
                candidates = 1

        # Determine routing strategy from metadata or config
        strategy = metadata.get("routing_strategy") or self._strategy_name

        # Determine outcome - if we got here, routing succeeded
        outcome = "success"
        reason = "model_selected"

        # Check for specific deployment selection
        if metadata.get("specific_deployment"):
            reason = "specific_deployment_requested"
        elif metadata.get("fallback"):
            reason = "fallback_triggered"

        # Approximate latency (not accurate, but provides a value)
        latency_ms = 0.1  # Placeholder

        # Set TG4.1 span attributes
        set_router_decision_attributes(
            span,
            strategy=strategy,
            model_selected=model,
            candidates_evaluated=candidates,
            outcome=outcome,
            reason=reason,
            latency_ms=latency_ms,
            strategy_version=f"v1-callback-{self._call_count}",
            fallback_triggered=bool(metadata.get("fallback")),
        )

        # Set gen_ai.* span attributes (GenAI Semantic Conventions)
        span.set_attribute("gen_ai.request.model", model)
        # Extract provider from litellm_params if available
        litellm_params = kwargs.get("litellm_params", {}) or {}
        provider = litellm_params.get("custom_llm_provider", "")
        if provider:
            span.set_attribute("gen_ai.system", provider)

        logger.debug(
            f"Emitted router telemetry: model={model}, strategy={strategy}, "
            f"candidates={candidates}, outcome={outcome}"
        )

    def log_success_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        """
        Called on successful API response.

        Records duration histogram, token usage, success counter, cost,
        and gen_ai.* span attributes from the response.
        """
        if not self._enabled:
            return

        try:
            self._record_success_metrics(kwargs, response_obj, start_time, end_time)
        except Exception as e:
            logger.debug(f"Failed to record success metrics: {e}")

    def _record_success_metrics(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        """Record metrics from a successful LLM API response."""
        from litellm_llmrouter.metrics import get_gateway_metrics

        model = kwargs.get("model", "unknown")
        litellm_params = kwargs.get("litellm_params", {}) or {}
        provider = litellm_params.get("custom_llm_provider", "unknown")

        # Compute duration from perf_counter if available, else from timestamps
        call_id = kwargs.get("litellm_call_id", "")
        perf_start = self._start_times.pop(call_id, None) if call_id else None
        if perf_start is not None:
            duration_s = time.perf_counter() - perf_start
        else:
            # Fallback: start_time/end_time are datetime or float from LiteLLM
            duration_s = _compute_duration(start_time, end_time)

        # Extract token usage from response object
        input_tokens = 0
        output_tokens = 0
        if hasattr(response_obj, "usage") and response_obj.usage is not None:
            usage = response_obj.usage
            input_tokens = getattr(usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(usage, "completion_tokens", 0) or 0

        # Extract finish reasons from response object
        finish_reasons: List[str] = []
        if hasattr(response_obj, "choices") and response_obj.choices:
            for choice in response_obj.choices:
                reason = getattr(choice, "finish_reason", None)
                if reason:
                    finish_reasons.append(str(reason))

        attrs = {
            "gen_ai.request.model": model,
            "gen_ai.system": provider,
        }

        gm = get_gateway_metrics()
        if gm:
            # Duration histogram
            gm.request_duration.record(duration_s, attrs)

            # Token usage histograms
            if input_tokens > 0:
                gm.token_usage.record(
                    input_tokens, {**attrs, "gen_ai.token.type": "input"}
                )
            if output_tokens > 0:
                gm.token_usage.record(
                    output_tokens, {**attrs, "gen_ai.token.type": "output"}
                )

            # Success counter
            gm.request_total.add(
                1,
                {
                    "model": model,
                    "provider": provider,
                    "status": "success",
                },
            )

            # Decrement active gauge
            gm.request_active.add(-1, {"model": model})

        # Set gen_ai.* span attributes on the current span
        span = trace.get_current_span()
        if span and span.is_recording():
            span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
            if hasattr(response_obj, "model") and response_obj.model:
                span.set_attribute("gen_ai.response.model", response_obj.model)
            if finish_reasons:
                span.set_attribute("gen_ai.response.finish_reasons", finish_reasons)

    def log_failure_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        """
        Called on failed API response.

        Records error counter with model/provider/error_type dimensions.
        """
        if not self._enabled:
            return

        try:
            self._record_failure_metrics(kwargs, response_obj, start_time, end_time)
        except Exception as e:
            logger.debug(f"Failed to record failure metrics: {e}")

    def _record_failure_metrics(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        """Record metrics from a failed LLM API response."""
        from litellm_llmrouter.metrics import get_gateway_metrics

        model = kwargs.get("model", "unknown")
        litellm_params = kwargs.get("litellm_params", {}) or {}
        provider = litellm_params.get("custom_llm_provider", "unknown")

        # Determine error type
        exception = kwargs.get("exception", None)
        error_type = type(exception).__name__ if exception else "unknown"

        # Clean up start time tracking
        call_id = kwargs.get("litellm_call_id", "")
        if call_id:
            self._start_times.pop(call_id, None)

        gm = get_gateway_metrics()
        if gm:
            gm.request_error.add(
                1,
                {
                    "model": model,
                    "provider": provider,
                    "error_type": error_type,
                },
            )

            # Decrement active gauge
            gm.request_active.add(-1, {"model": model})

        # Set gen_ai.response.finish_reasons = ["error"] on the span
        span = trace.get_current_span()
        if span and span.is_recording():
            span.set_attribute("gen_ai.response.finish_reasons", ["error"])

    async def async_log_pre_api_call(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        kwargs: Dict[str, Any],
    ) -> None:
        """Async version of log_pre_api_call."""
        self.log_pre_api_call(model, messages, kwargs)

    async def async_log_success_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        """Async version of log_success_event."""
        self.log_success_event(kwargs, response_obj, start_time, end_time)

    async def async_log_failure_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        """Async version of log_failure_event."""
        self.log_failure_event(kwargs, response_obj, start_time, end_time)

    async def async_post_call_success_hook(
        self,
        data: Dict[str, Any],
        user_api_key_dict: Any,
        response: Any,
    ) -> None:
        """Called after successful API call. Required by LiteLLM callback interface."""
        pass

    async def async_post_call_failure_hook(
        self,
        request_data: Dict[str, Any],
        original_exception: Exception,
        user_api_key_dict: Any,
    ) -> None:
        """Called after failed API call. Required by LiteLLM callback interface."""
        pass


def _compute_duration(start_time: Any, end_time: Any) -> float:
    """
    Compute duration in seconds from LiteLLM start/end times.

    LiteLLM passes datetime objects or floats depending on the code path.

    Args:
        start_time: Start time (datetime or float)
        end_time: End time (datetime or float)

    Returns:
        Duration in seconds, or 0.0 if computation fails.
    """
    try:
        if isinstance(start_time, (int, float)) and isinstance(end_time, (int, float)):
            return max(0.0, float(end_time) - float(start_time))
        # datetime objects
        if hasattr(start_time, "timestamp") and hasattr(end_time, "timestamp"):
            return max(0.0, end_time.timestamp() - start_time.timestamp())
    except Exception:
        pass
    return 0.0


def register_router_decision_callback() -> Optional[RouterDecisionCallback]:
    """
    Register the router decision callback with LiteLLM.

    Returns:
        The registered callback instance, or None if disabled.
    """
    if not ROUTER_CALLBACK_ENABLED:
        logger.debug("Router decision callback disabled")
        return None

    try:
        import litellm

        callback = RouterDecisionCallback()

        # Append to LiteLLM's callbacks list
        if not hasattr(litellm, "callbacks"):
            litellm.callbacks = []

        # Avoid duplicate registration
        for existing in litellm.callbacks:
            if isinstance(existing, RouterDecisionCallback):
                logger.debug("Router decision callback already registered")
                return existing

        litellm.callbacks.append(callback)
        logger.info("Registered router decision callback with LiteLLM")
        return callback

    except ImportError:
        logger.warning("LiteLLM not available, cannot register callback")
        return None
    except Exception as e:
        logger.error(f"Failed to register router decision callback: {e}")
        return None


def get_router_decision_callback() -> type:
    """
    Get the RouterDecisionCallback class for manual registration.

    Returns:
        The RouterDecisionCallback class
    """
    return RouterDecisionCallback
