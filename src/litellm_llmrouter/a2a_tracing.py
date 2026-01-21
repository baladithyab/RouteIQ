"""
A2A Tracing - OpenTelemetry Instrumentation for A2A Gateway
=============================================================

Provides OTel tracing for A2A (Agent-to-Agent) protocol operations with:
- Spans for agent invocations (a2a.agent.send, a2a.agent.stream)
- Attributes: agent_id, message_id, method, stream mode, HTTP method/path
- Error recording with exception details
- W3C trace context propagation on outbound HTTP calls
- Safe no-op behavior when tracing is disabled
- Proper span lifecycle management for streaming (disconnect-safe)
- FastAPI middleware for instrumenting LiteLLM's built-in /a2a/* routes

Usage:
    from litellm_llmrouter.a2a_tracing import (
        instrument_a2a_gateway,
        A2ATracingMiddleware,
    )

    # Option 1: Instrument A2A gateway (wraps gateway methods)
    instrument_a2a_gateway()

    # Option 2: Add middleware to FastAPI app (instruments HTTP endpoints)
    from fastapi import FastAPI
    app = FastAPI()
    app.add_middleware(A2ATracingMiddleware)
"""

import asyncio
import contextlib
import functools
import os
import time
from typing import Any, Callable, Generator

from litellm._logging import verbose_proxy_logger

# OTel imports - these are optional dependencies
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode, Span, SpanKind
    from opentelemetry.propagate import inject
    from opentelemetry.context import attach, detach, get_current

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None
    Status = None
    StatusCode = None
    Span = None
    SpanKind = None
    inject = None
    attach = None
    detach = None
    get_current = None


# Tracer name for A2A operations
TRACER_NAME = "litellm.a2a_gateway"

# Span attribute names (following OTel semantic conventions where applicable)
ATTR_A2A_AGENT_ID = "a2a.agent.id"
ATTR_A2A_AGENT_NAME = "a2a.agent.name"
ATTR_A2A_AGENT_URL = "a2a.agent.url"
ATTR_A2A_METHOD = "a2a.method"
ATTR_A2A_MESSAGE_ID = "a2a.message.id"
ATTR_A2A_STREAM = "a2a.stream"
ATTR_A2A_SUCCESS = "a2a.success"
ATTR_A2A_ERROR = "a2a.error"
ATTR_A2A_DURATION_MS = "a2a.duration_ms"
ATTR_HTTP_METHOD = "http.method"
ATTR_HTTP_URL = "http.url"
ATTR_HTTP_ROUTE = "http.route"
ATTR_HTTP_TARGET = "http.target"
ATTR_HTTP_STATUS_CODE = "http.status_code"


def get_tracer() -> Any:
    """
    Get the OTel tracer for A2A operations.

    Returns:
        OpenTelemetry Tracer instance, or None if OTel is not available
    """
    if not OTEL_AVAILABLE:
        return None

    try:
        return trace.get_tracer(TRACER_NAME)
    except Exception as e:
        verbose_proxy_logger.warning(f"A2A tracing: Failed to get OTel tracer: {e}")
        return None


def is_tracing_enabled() -> bool:
    """Check if A2A tracing is enabled and configured."""
    if not OTEL_AVAILABLE:
        return False
    if os.getenv("A2A_TRACING_ENABLED", "true").lower() != "true":
        return False
    return get_tracer() is not None


class _DummySpan:
    """Dummy span for when tracing is disabled."""

    def set_attribute(self, *args: Any, **kwargs: Any) -> None:
        pass

    def set_status(self, *args: Any, **kwargs: Any) -> None:
        pass

    def record_exception(self, *args: Any, **kwargs: Any) -> None:
        pass

    def end(self) -> None:
        pass


# =============================================================================
# A2A Tracing Middleware for FastAPI
# =============================================================================


class A2ATracingMiddleware:
    """
    FastAPI/Starlette middleware that instruments LiteLLM's /a2a/* routes.

    This middleware creates OpenTelemetry spans for A2A HTTP requests with:
    - Span name: "a2a.http /{agent_id}" for non-streaming, "a2a.http.stream /{agent_id}" for streaming
    - Attributes: a2a.agent.id, a2a.stream, http.route, http.target, http.method
    - Proper span lifecycle for streaming responses (span ends when stream completes)
    - Error recording when requests fail
    - Disconnect-safe handling for SSE streams

    Usage:
        from fastapi import FastAPI
        from litellm_llmrouter.a2a_tracing import A2ATracingMiddleware

        app = FastAPI()
        app.add_middleware(A2ATracingMiddleware)

    Note: This middleware only instruments paths matching /a2a/{agent_id}*.
    Other paths pass through without modification.
    """

    def __init__(self, app: Any) -> None:
        """Initialize the middleware with the ASGI app."""
        self.app = app
        self._tracer = get_tracer() if OTEL_AVAILABLE else None

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        """
        ASGI middleware entry point.

        For HTTP requests to /a2a/* paths, creates a span that covers the
        entire request lifecycle including streaming responses.
        """
        if scope["type"] != "http":
            # Not an HTTP request - pass through
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")

        # Only instrument /a2a/{agent_id} paths (LiteLLM's A2A routes)
        if not path.startswith("/a2a/"):
            await self.app(scope, receive, send)
            return

        # Extract agent_id from path: /a2a/{agent_id} or /a2a/{agent_id}/message/stream
        path_parts = path.split("/")
        agent_id = path_parts[2] if len(path_parts) > 2 else "unknown"

        # Determine if this is a streaming endpoint
        is_stream_endpoint = "/message/stream" in path

        # Get HTTP method
        method = scope.get("method", "POST")

        # Check if tracing is available
        if self._tracer is None:
            await self.app(scope, receive, send)
            return

        # Create span for A2A request
        span_name = f"a2a.http{'stream' if is_stream_endpoint else ''} /{agent_id}"

        # Use SERVER kind since this is an inbound request
        with self._tracer.start_as_current_span(
            span_name,
            kind=SpanKind.SERVER if SpanKind else None,
        ) as span:
            # Set initial attributes
            span.set_attribute(ATTR_A2A_AGENT_ID, agent_id)
            span.set_attribute(ATTR_A2A_STREAM, is_stream_endpoint)
            span.set_attribute(ATTR_HTTP_METHOD, method)
            span.set_attribute(
                ATTR_HTTP_ROUTE,
                "/a2a/{agent_id}" + ("/message/stream" if is_stream_endpoint else ""),
            )
            span.set_attribute(ATTR_HTTP_TARGET, path)

            # Track response status
            response_status = {"code": 0, "started": False}
            start_time = time.perf_counter()

            async def traced_send(message: dict) -> None:
                """Wrapper for send that captures response status."""
                if message["type"] == "http.response.start":
                    response_status["code"] = message.get("status", 0)
                    response_status["started"] = True
                    span.set_attribute(ATTR_HTTP_STATUS_CODE, response_status["code"])
                await send(message)

            try:
                await self.app(scope, receive, traced_send)

                # Set success status if we got a response
                if response_status["started"]:
                    if response_status["code"] < 400:
                        span.set_attribute(ATTR_A2A_SUCCESS, True)
                        span.set_status(Status(StatusCode.OK))
                    else:
                        span.set_attribute(ATTR_A2A_SUCCESS, False)
                        span.set_status(
                            Status(StatusCode.ERROR, f"HTTP {response_status['code']}")
                        )

            except asyncio.CancelledError:
                # Client disconnected - this is expected for SSE streams
                duration_ms = (time.perf_counter() - start_time) * 1000
                span.set_attribute(ATTR_A2A_DURATION_MS, round(duration_ms, 2))
                span.set_attribute("a2a.client_disconnected", True)
                span.set_status(Status(StatusCode.OK, "Client disconnected"))
                raise

            except Exception as e:
                # Request failed with error
                duration_ms = (time.perf_counter() - start_time) * 1000
                span.set_attribute(ATTR_A2A_DURATION_MS, round(duration_ms, 2))
                span.set_attribute(ATTR_A2A_SUCCESS, False)
                span.set_attribute(ATTR_A2A_ERROR, str(e))
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

            finally:
                # Record duration
                duration_ms = (time.perf_counter() - start_time) * 1000
                span.set_attribute(ATTR_A2A_DURATION_MS, round(duration_ms, 2))


def register_a2a_middleware(app: Any) -> bool:
    """
    Register the A2A tracing middleware with a FastAPI/Starlette app.

    This function adds the A2ATracingMiddleware to the app if tracing is enabled.
    It's safe to call even if OTel is not available (no-op in that case).

    Args:
        app: FastAPI or Starlette application instance

    Returns:
        True if middleware was registered, False otherwise
    """
    if not OTEL_AVAILABLE:
        verbose_proxy_logger.info(
            "A2A tracing middleware: OTel not available, skipping"
        )
        return False

    if os.getenv("A2A_TRACING_ENABLED", "true").lower() != "true":
        verbose_proxy_logger.info(
            "A2A tracing middleware: Disabled via A2A_TRACING_ENABLED=false"
        )
        return False

    try:
        # Add middleware to the app
        # Note: For Starlette/FastAPI, we use add_middleware with the class
        app.add_middleware(A2ATracingMiddleware)
        verbose_proxy_logger.info(
            "A2A tracing middleware: Registered for /a2a/* routes"
        )
        return True
    except Exception as e:
        verbose_proxy_logger.error(f"A2A tracing middleware: Failed to register: {e}")
        return False


class StreamingSpan:
    """
    A wrapper for managing span lifecycle during async streaming operations.

    This class ensures the span is properly ended even if the client disconnects
    or an exception occurs during streaming.
    """

    def __init__(
        self,
        tracer: Any,
        agent_id: str,
        agent_name: str,
        agent_url: str,
        method: str,
        message_id: str | None = None,
    ):
        self._tracer = tracer
        self._agent_id = agent_id
        self._agent_name = agent_name
        self._agent_url = agent_url
        self._method = method
        self._message_id = message_id
        self._span: Any = None
        self._token: Any = None
        self._start_time: float = 0
        self._ended: bool = False

    def __enter__(self) -> "StreamingSpan":
        """Start the span and make it current."""
        if self._tracer is None:
            return self

        self._start_time = time.perf_counter()

        # Create span manually so we can control when it ends
        self._span = self._tracer.start_span(f"a2a.agent.stream/{self._agent_id}")

        # Make it the current span
        self._token = trace.use_span(self._span, end_on_exit=False).__enter__()

        # Set initial attributes
        self._span.set_attribute(ATTR_A2A_AGENT_ID, self._agent_id)
        self._span.set_attribute(ATTR_A2A_AGENT_NAME, self._agent_name)
        self._span.set_attribute(ATTR_A2A_AGENT_URL, self._agent_url)
        self._span.set_attribute(ATTR_A2A_METHOD, self._method)
        self._span.set_attribute(ATTR_A2A_STREAM, True)
        self._span.set_attribute(ATTR_HTTP_METHOD, "POST")
        self._span.set_attribute(ATTR_HTTP_URL, self._agent_url)

        if self._message_id is not None:
            self._span.set_attribute(ATTR_A2A_MESSAGE_ID, str(self._message_id))

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """End the span, recording any exception that occurred."""
        self._end_span(exc_val)
        return False  # Don't suppress exceptions

    def _end_span(self, exception: Exception | None = None) -> None:
        """End the span with proper status and duration."""
        if self._ended or self._span is None:
            return

        self._ended = True
        duration_ms = (time.perf_counter() - self._start_time) * 1000

        try:
            self._span.set_attribute(ATTR_A2A_DURATION_MS, round(duration_ms, 2))

            if exception is not None:
                self._span.set_attribute(ATTR_A2A_SUCCESS, False)
                self._span.set_attribute(ATTR_A2A_ERROR, str(exception))
                self._span.record_exception(exception)
                self._span.set_status(Status(StatusCode.ERROR, str(exception)))
            else:
                self._span.set_attribute(ATTR_A2A_SUCCESS, True)
                self._span.set_status(Status(StatusCode.OK))

            self._span.end()
        except Exception as e:
            verbose_proxy_logger.debug(f"A2A tracing: Error ending span: {e}")

    @property
    def span(self) -> Any:
        """Get the underlying span (or a dummy span if tracing is disabled)."""
        return self._span if self._span is not None else _DummySpan()

    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute on the span."""
        if self._span is not None:
            self._span.set_attribute(key, value)

    def record_exception(self, exception: Exception) -> None:
        """Record an exception on the span."""
        if self._span is not None:
            self._span.record_exception(exception)

    def mark_success(self) -> None:
        """Mark the stream as successfully completed."""
        self._end_span(None)

    def mark_error(self, error: Exception) -> None:
        """Mark the stream as failed with an error."""
        self._end_span(error)


def trace_agent_stream(
    tracer: Any,
    agent_id: str,
    agent_name: str,
    agent_url: str,
    method: str = "message/stream",
    message_id: str | None = None,
) -> StreamingSpan:
    """
    Create a StreamingSpan for tracing an A2A streaming response.

    This is designed for use with async generators where the span must
    remain open until the stream completes or fails.

    Usage:
        streaming_span = trace_agent_stream(tracer, agent_id, ...)
        with streaming_span:
            async for chunk in agent_response:
                yield chunk
            streaming_span.mark_success()

    Args:
        tracer: OTel tracer instance
        agent_id: ID of the agent being invoked
        agent_name: Name of the agent
        agent_url: URL of the agent endpoint
        method: JSON-RPC method (default: "message/stream")
        message_id: Optional JSON-RPC message ID

    Returns:
        A StreamingSpan context manager
    """
    return StreamingSpan(
        tracer=tracer,
        agent_id=agent_id,
        agent_name=agent_name,
        agent_url=agent_url,
        method=method,
        message_id=message_id,
    )


def instrument_a2a_gateway() -> bool:
    """
    Instrument the A2A gateway with OTel tracing.

    This modifies the global A2AGateway instance to add tracing
    to key operations like agent invocations and streaming.

    Returns:
        True if instrumentation was successful, False otherwise
    """
    if not OTEL_AVAILABLE:
        verbose_proxy_logger.info(
            "A2A tracing: OTel not available, skipping instrumentation"
        )
        return False

    if os.getenv("A2A_TRACING_ENABLED", "true").lower() != "true":
        verbose_proxy_logger.info("A2A tracing: Disabled via A2A_TRACING_ENABLED=false")
        return False

    try:
        from .a2a_gateway import get_a2a_gateway

        gateway = get_a2a_gateway()
        tracer = get_tracer()

        if tracer is None:
            verbose_proxy_logger.warning("A2A tracing: Could not get OTel tracer")
            return False

        # Store tracer on gateway
        gateway._tracer = tracer

        # Wrap invoke_agent with tracing
        original_invoke_agent = gateway.invoke_agent

        @functools.wraps(original_invoke_agent)
        async def traced_invoke_agent(agent_id: str, request: Any):
            agent = gateway.get_agent(agent_id)
            agent_name = agent.name if agent else "unknown"
            agent_url = agent.url if agent else "unknown"
            method = request.method if hasattr(request, "method") else "unknown"
            message_id = request.id if hasattr(request, "id") else None

            # Prepare headers with trace context
            headers = {"Content-Type": "application/json"}
            inject_trace_headers(headers)

            with trace_agent_send(
                tracer,
                agent_id=agent_id,
                agent_name=agent_name,
                agent_url=agent_url,
                method=method,
                message_id=str(message_id) if message_id else None,
            ) as span:
                result = await original_invoke_agent(agent_id, request)
                if span and hasattr(span, "set_attribute"):
                    span.set_attribute(
                        ATTR_A2A_SUCCESS, result.error is None if result else False
                    )
                    if result and result.error:
                        error_msg = str(result.error.get("message", "Unknown error"))
                        span.set_attribute(ATTR_A2A_ERROR, error_msg)
                return result

        gateway.invoke_agent = traced_invoke_agent

        # Wrap stream_agent_response with tracing
        original_stream_response = gateway.stream_agent_response

        @functools.wraps(original_stream_response)
        async def traced_stream_response(agent_id: str, request: Any):
            agent = gateway.get_agent(agent_id)
            agent_name = agent.name if agent else "unknown"
            agent_url = agent.url if agent else "unknown"
            method = request.method if hasattr(request, "method") else "message/stream"
            message_id = request.id if hasattr(request, "id") else None

            streaming_span = trace_agent_stream(
                tracer,
                agent_id=agent_id,
                agent_name=agent_name,
                agent_url=agent_url,
                method=method,
                message_id=str(message_id) if message_id else None,
            )

            try:
                with streaming_span:
                    async for chunk in original_stream_response(agent_id, request):
                        yield chunk
                    streaming_span.mark_success()
            except Exception as e:
                streaming_span.mark_error(e)
                raise

        gateway.stream_agent_response = traced_stream_response

        verbose_proxy_logger.info("A2A tracing: Gateway instrumented successfully")
        return True

    except Exception as e:
        verbose_proxy_logger.error(f"A2A tracing: Failed to instrument gateway: {e}")
        return False


def inject_trace_headers(headers: dict[str, str]) -> dict[str, str]:
    """
    Inject W3C trace context headers into outbound HTTP request headers.

    This adds the `traceparent` and `tracestate` headers to propagate
    trace context to downstream services.

    Args:
        headers: The existing headers dictionary (will be modified in place)

    Returns:
        The headers dictionary with trace context injected
    """
    if not OTEL_AVAILABLE or inject is None:
        return headers

    try:
        inject(headers)
    except Exception as e:
        verbose_proxy_logger.debug(f"A2A tracing: Failed to inject trace headers: {e}")

    return headers


@contextlib.contextmanager
def trace_agent_send(
    tracer: Any,
    agent_id: str,
    agent_name: str,
    agent_url: str,
    method: str,
    message_id: str | None = None,
) -> Generator[Any, None, None]:
    """
    Context manager for tracing an A2A agent send operation.

    Uses start_as_current_span to ensure the span is:
    - Made the active span in the current context
    - Properly parented to any existing HTTP request span
    - Exported to the OTLP collector when the context manager exits

    Args:
        tracer: OTel tracer instance
        agent_id: ID of the agent being invoked
        agent_name: Name of the agent
        agent_url: URL of the agent endpoint
        method: JSON-RPC method (e.g., "message/send")
        message_id: Optional JSON-RPC message ID

    Yields:
        The span, or a DummySpan if tracing is disabled
    """
    if tracer is None:
        yield _DummySpan()
        return

    start_time = time.perf_counter()

    # Use start_as_current_span to ensure span is active and exported
    with tracer.start_as_current_span(f"a2a.agent.send/{agent_id}") as span:
        span.set_attribute(ATTR_A2A_AGENT_ID, agent_id)
        span.set_attribute(ATTR_A2A_AGENT_NAME, agent_name)
        span.set_attribute(ATTR_A2A_AGENT_URL, agent_url)
        span.set_attribute(ATTR_A2A_METHOD, method)
        span.set_attribute(ATTR_A2A_STREAM, False)
        span.set_attribute(ATTR_HTTP_METHOD, "POST")
        span.set_attribute(ATTR_HTTP_URL, agent_url)

        if message_id is not None:
            span.set_attribute(ATTR_A2A_MESSAGE_ID, str(message_id))

        try:
            yield span
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            span.set_attribute(ATTR_A2A_DURATION_MS, round(duration_ms, 2))
            span.set_attribute(ATTR_A2A_SUCCESS, False)
            span.set_attribute(ATTR_A2A_ERROR, str(e))
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
        else:
            duration_ms = (time.perf_counter() - start_time) * 1000
            span.set_attribute(ATTR_A2A_DURATION_MS, round(duration_ms, 2))
            span.set_attribute(ATTR_A2A_SUCCESS, True)
            span.set_status(Status(StatusCode.OK))
