"""
Unit Tests for A2A Tracing - OpenTelemetry Instrumentation
==========================================================

Tests for the A2A tracing module to verify:
- Span creation for agent send and stream operations
- Span attributes are correctly set (agent_id, message_id, stream flag, etc.)
- Error handling records exceptions on spans
- Spans are properly exported via context management
- StreamingSpan handles disconnect/completion correctly
- W3C trace header injection
- A2ATracingMiddleware for HTTP endpoint instrumentation
"""

import asyncio
import sys
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

# Mock all external dependencies before importing
sys.modules["litellm"] = MagicMock()
sys.modules["litellm._logging"] = MagicMock()
sys.modules["litellm.proxy"] = MagicMock()
sys.modules["litellm.proxy.proxy_server"] = MagicMock()
sys.modules["fastapi"] = MagicMock()
sys.modules["pydantic"] = MagicMock()

from opentelemetry import trace  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.trace import StatusCode  # noqa: E402

# Set up the TracerProvider once at module load time
_exporter = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_exporter))
trace.set_tracer_provider(_provider)

# Import the module under test directly (not through __init__.py)
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "a2a_tracing", "src/litellm_llmrouter/a2a_tracing.py"
)
a2a_tracing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a2a_tracing)

trace_agent_send = a2a_tracing.trace_agent_send
trace_agent_stream = a2a_tracing.trace_agent_stream
StreamingSpan = a2a_tracing.StreamingSpan
inject_trace_headers = a2a_tracing.inject_trace_headers
get_tracer = a2a_tracing.get_tracer
A2ATracingMiddleware = a2a_tracing.A2ATracingMiddleware
register_a2a_middleware = a2a_tracing.register_a2a_middleware
ATTR_A2A_AGENT_ID = a2a_tracing.ATTR_A2A_AGENT_ID
ATTR_A2A_AGENT_NAME = a2a_tracing.ATTR_A2A_AGENT_NAME
ATTR_A2A_AGENT_URL = a2a_tracing.ATTR_A2A_AGENT_URL
ATTR_A2A_METHOD = a2a_tracing.ATTR_A2A_METHOD
ATTR_A2A_MESSAGE_ID = a2a_tracing.ATTR_A2A_MESSAGE_ID
ATTR_A2A_STREAM = a2a_tracing.ATTR_A2A_STREAM
ATTR_A2A_SUCCESS = a2a_tracing.ATTR_A2A_SUCCESS
ATTR_A2A_ERROR = a2a_tracing.ATTR_A2A_ERROR
ATTR_A2A_DURATION_MS = a2a_tracing.ATTR_A2A_DURATION_MS
ATTR_HTTP_METHOD = a2a_tracing.ATTR_HTTP_METHOD
ATTR_HTTP_URL = a2a_tracing.ATTR_HTTP_URL
ATTR_HTTP_ROUTE = a2a_tracing.ATTR_HTTP_ROUTE
ATTR_HTTP_TARGET = a2a_tracing.ATTR_HTTP_TARGET
ATTR_HTTP_STATUS_CODE = a2a_tracing.ATTR_HTTP_STATUS_CODE


@pytest.fixture(autouse=True)
def clear_spans():
    """Clear spans before and after each test."""
    _exporter.clear()
    yield _exporter
    _exporter.clear()


class TestA2ATracingMiddleware:
    """Test suite for A2ATracingMiddleware ASGI middleware."""

    @pytest.mark.asyncio
    async def test_middleware_creates_span_for_a2a_route(self, clear_spans):
        """Test that middleware creates a span for /a2a/{agent_id} routes."""
        exporter = clear_spans

        # Track what the app received
        app_called = {"called": False, "scope": None}

        async def mock_app(scope, receive, send):
            app_called["called"] = True
            app_called["scope"] = scope
            # Send a minimal HTTP response
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b'{"result": "ok"}',
                }
            )

        middleware = A2ATracingMiddleware(mock_app)

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/a2a/weather-bot",
            "headers": [],
        }

        receive = AsyncMock()
        send = AsyncMock()

        await middleware(scope, receive, send)

        # Verify app was called
        assert app_called["called"]

        # Verify span was created
        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert "a2a.http" in span.name
        assert "weather-bot" in span.name

        attrs = dict(span.attributes)
        assert attrs[ATTR_A2A_AGENT_ID] == "weather-bot"
        assert attrs[ATTR_A2A_STREAM] is False
        assert attrs[ATTR_HTTP_METHOD] == "POST"
        assert attrs[ATTR_HTTP_TARGET] == "/a2a/weather-bot"
        assert attrs[ATTR_HTTP_STATUS_CODE] == 200
        assert attrs[ATTR_A2A_SUCCESS] is True

    @pytest.mark.asyncio
    async def test_middleware_detects_streaming_endpoint(self, clear_spans):
        """Test that middleware correctly detects /message/stream endpoints."""
        exporter = clear_spans

        async def mock_app(scope, receive, send):
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"text/event-stream")],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b'data: {"chunk": 1}\n\n',
                }
            )

        middleware = A2ATracingMiddleware(mock_app)

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/a2a/chat-agent/message/stream",
            "headers": [],
        }

        await middleware(scope, AsyncMock(), AsyncMock())

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert "a2a.httpstream" in span.name

        attrs = dict(span.attributes)
        assert attrs[ATTR_A2A_AGENT_ID] == "chat-agent"
        assert attrs[ATTR_A2A_STREAM] is True
        assert "/message/stream" in attrs[ATTR_HTTP_ROUTE]

    @pytest.mark.asyncio
    async def test_middleware_passes_through_non_a2a_routes(self, clear_spans):
        """Test that middleware passes through non-A2A routes without creating spans."""
        exporter = clear_spans

        app_called = {"called": False}

        async def mock_app(scope, receive, send):
            app_called["called"] = True
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        middleware = A2ATracingMiddleware(mock_app)

        # Test various non-A2A paths
        for path in ["/v1/chat/completions", "/health", "/mcp/servers"]:
            exporter.clear()
            app_called["called"] = False

            scope = {
                "type": "http",
                "method": "POST",
                "path": path,
                "headers": [],
            }

            await middleware(scope, AsyncMock(), AsyncMock())

            assert app_called["called"], f"App should be called for {path}"
            spans = exporter.get_finished_spans()
            assert len(spans) == 0, f"No span should be created for {path}"

    @pytest.mark.asyncio
    async def test_middleware_handles_error_response(self, clear_spans):
        """Test that middleware records error status for 4xx/5xx responses."""
        exporter = clear_spans

        async def mock_app(scope, receive, send):
            await send(
                {
                    "type": "http.response.start",
                    "status": 404,
                    "headers": [],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b'{"error": "Agent not found"}',
                }
            )

        middleware = A2ATracingMiddleware(mock_app)

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/a2a/nonexistent-agent",
            "headers": [],
        }

        await middleware(scope, AsyncMock(), AsyncMock())

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.status.status_code == StatusCode.ERROR

        attrs = dict(span.attributes)
        assert attrs[ATTR_A2A_SUCCESS] is False
        assert attrs[ATTR_HTTP_STATUS_CODE] == 404

    @pytest.mark.asyncio
    async def test_middleware_handles_exception(self, clear_spans):
        """Test that middleware records exceptions properly."""
        exporter = clear_spans

        async def mock_app(scope, receive, send):
            raise RuntimeError("Simulated server error")

        middleware = A2ATracingMiddleware(mock_app)

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/a2a/error-agent",
            "headers": [],
        }

        with pytest.raises(RuntimeError, match="Simulated server error"):
            await middleware(scope, AsyncMock(), AsyncMock())

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.status.status_code == StatusCode.ERROR

        attrs = dict(span.attributes)
        assert attrs[ATTR_A2A_SUCCESS] is False
        assert "Simulated server error" in attrs[ATTR_A2A_ERROR]

        # Verify exception was recorded
        events = span.events
        exception_events = [e for e in events if e.name == "exception"]
        assert len(exception_events) >= 1

    @pytest.mark.asyncio
    async def test_middleware_handles_client_disconnect(self, clear_spans):
        """Test that middleware handles CancelledError (client disconnect) gracefully."""
        exporter = clear_spans

        async def mock_app(scope, receive, send):
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [],
                }
            )
            # Simulate client disconnect during streaming
            raise asyncio.CancelledError()

        middleware = A2ATracingMiddleware(mock_app)

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/a2a/stream-agent/message/stream",
            "headers": [],
        }

        with pytest.raises(asyncio.CancelledError):
            await middleware(scope, AsyncMock(), AsyncMock())

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        # Client disconnect should be OK status (expected for SSE)
        assert span.status.status_code == StatusCode.OK

        attrs = dict(span.attributes)
        assert attrs["a2a.client_disconnected"] is True

    @pytest.mark.asyncio
    async def test_middleware_passes_through_websocket(self, clear_spans):
        """Test that middleware passes through non-HTTP requests."""
        exporter = clear_spans

        app_called = {"called": False}

        async def mock_app(scope, receive, send):
            app_called["called"] = True

        middleware = A2ATracingMiddleware(mock_app)

        scope = {
            "type": "websocket",
            "path": "/a2a/ws-agent",
        }

        await middleware(scope, AsyncMock(), AsyncMock())

        assert app_called["called"]
        spans = exporter.get_finished_spans()
        assert len(spans) == 0  # No span for non-HTTP

    def test_register_a2a_middleware_adds_to_app(self):
        """Test that register_a2a_middleware adds middleware to app."""
        mock_app = MagicMock()

        with patch.dict("os.environ", {"A2A_TRACING_ENABLED": "true"}):
            result = register_a2a_middleware(mock_app)

        assert result is True
        mock_app.add_middleware.assert_called_once_with(A2ATracingMiddleware)

    def test_register_a2a_middleware_respects_disabled_flag(self):
        """Test that register_a2a_middleware respects A2A_TRACING_ENABLED=false."""
        mock_app = MagicMock()

        with patch.dict("os.environ", {"A2A_TRACING_ENABLED": "false"}):
            result = register_a2a_middleware(mock_app)

        assert result is False
        mock_app.add_middleware.assert_not_called()


class TestTraceAgentSend:
    """Test suite for trace_agent_send context manager."""

    def test_span_is_created_and_exported(self, clear_spans):
        """Test that trace_agent_send creates a span that is properly exported."""
        exporter = clear_spans
        tracer = trace.get_tracer("test.a2a_gateway")

        with trace_agent_send(
            tracer,
            agent_id="weather-bot",
            agent_name="Weather Agent",
            agent_url="http://localhost:9001/a2a",
            method="message/send",
            message_id="req-123",
        ) as span:
            # Span should be active inside the context
            assert span is not None
            current_span = trace.get_current_span()
            assert current_span == span

        # Verify span was exported
        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        exported_span = spans[0]
        assert exported_span.name == "a2a.agent.send/weather-bot"
        assert exported_span.status.status_code == StatusCode.OK

    def test_span_has_correct_attributes(self, clear_spans):
        """Test that trace_agent_send sets the correct span attributes."""
        exporter = clear_spans
        tracer = trace.get_tracer("test.a2a_gateway")

        with trace_agent_send(
            tracer,
            agent_id="code-bot",
            agent_name="Code Assistant",
            agent_url="http://agents.example.com/code",
            method="message/send",
            message_id="msg-456",
        ):
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        attrs = dict(span.attributes)

        assert attrs[ATTR_A2A_AGENT_ID] == "code-bot"
        assert attrs[ATTR_A2A_AGENT_NAME] == "Code Assistant"
        assert attrs[ATTR_A2A_AGENT_URL] == "http://agents.example.com/code"
        assert attrs[ATTR_A2A_METHOD] == "message/send"
        assert attrs[ATTR_A2A_MESSAGE_ID] == "msg-456"
        assert attrs[ATTR_A2A_STREAM] is False
        assert attrs[ATTR_A2A_SUCCESS] is True
        assert attrs[ATTR_HTTP_METHOD] == "POST"
        assert attrs[ATTR_HTTP_URL] == "http://agents.example.com/code"
        assert ATTR_A2A_DURATION_MS in attrs
        assert attrs[ATTR_A2A_DURATION_MS] >= 0

    def test_span_records_exception_on_error(self, clear_spans):
        """Test that trace_agent_send records exceptions and sets error status."""
        exporter = clear_spans
        tracer = trace.get_tracer("test.a2a_gateway")

        with pytest.raises(ConnectionError, match="Agent unreachable"):
            with trace_agent_send(
                tracer,
                agent_id="failing-agent",
                agent_name="Failing Agent",
                agent_url="http://unreachable:9000",
                method="message/send",
            ):
                raise ConnectionError("Agent unreachable")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.status.status_code == StatusCode.ERROR
        assert "Agent unreachable" in span.status.description

        attrs = dict(span.attributes)
        assert attrs[ATTR_A2A_SUCCESS] is False
        assert attrs[ATTR_A2A_ERROR] == "Agent unreachable"

        # Verify exception was recorded (at least one exception event)
        events = span.events
        assert len(events) >= 1
        exception_events = [e for e in events if e.name == "exception"]
        assert len(exception_events) >= 1

    def test_span_is_parented_to_current_context(self, clear_spans):
        """Test that A2A spans are parented to the current trace context."""
        exporter = clear_spans
        tracer = trace.get_tracer("test.a2a_gateway")

        # Simulate an HTTP request span (parent)
        with tracer.start_as_current_span("HTTP POST /a2a/weather-bot") as parent_span:
            _ = parent_span.get_span_context()  # Used for hierarchical tracing

            # This should be a child of the HTTP span
            with trace_agent_send(
                tracer,
                agent_id="weather-bot",
                agent_name="Weather",
                agent_url="http://localhost:9001",
                method="message/send",
            ):
                pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 2

        # Find the agent send span
        agent_span = next(s for s in spans if "a2a.agent.send" in s.name)
        http_span = next(s for s in spans if "HTTP" in s.name)

        # Verify parent-child relationship
        assert agent_span.parent is not None
        assert agent_span.parent.span_id == http_span.context.span_id
        assert agent_span.context.trace_id == http_span.context.trace_id

    def test_dummy_span_when_tracer_is_none(self):
        """Test that a dummy span is yielded when tracer is None."""
        with trace_agent_send(
            tracer=None,
            agent_id="test-agent",
            agent_name="Test",
            agent_url="http://test",
            method="message/send",
        ) as span:
            # Should get a dummy span that accepts calls but does nothing
            span.set_attribute("key", "value")
            span.set_status("OK")
            span.record_exception(ValueError("test"))


class TestTraceAgentStream:
    """Test suite for trace_agent_stream and StreamingSpan."""

    def test_streaming_span_is_created_and_exported(self, clear_spans):
        """Test that StreamingSpan creates a span that is properly exported."""
        exporter = clear_spans
        tracer = trace.get_tracer("test.a2a_gateway")

        streaming_span = trace_agent_stream(
            tracer,
            agent_id="stream-agent",
            agent_name="Streaming Agent",
            agent_url="http://localhost:9002/a2a",
            method="message/stream",
            message_id="stream-123",
        )

        with streaming_span:
            # Span should be accessible
            assert streaming_span.span is not None
            streaming_span.mark_success()

        # Verify span was exported
        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        exported_span = spans[0]
        assert exported_span.name == "a2a.agent.stream/stream-agent"
        assert exported_span.status.status_code == StatusCode.OK

    def test_streaming_span_has_correct_attributes(self, clear_spans):
        """Test that StreamingSpan sets the correct span attributes."""
        exporter = clear_spans
        tracer = trace.get_tracer("test.a2a_gateway")

        streaming_span = trace_agent_stream(
            tracer,
            agent_id="chat-agent",
            agent_name="Chat Agent",
            agent_url="http://agents.example.com/chat",
            method="message/stream",
            message_id="chat-789",
        )

        with streaming_span:
            streaming_span.mark_success()

        spans = exporter.get_finished_spans()
        span = spans[0]
        attrs = dict(span.attributes)

        assert attrs[ATTR_A2A_AGENT_ID] == "chat-agent"
        assert attrs[ATTR_A2A_AGENT_NAME] == "Chat Agent"
        assert attrs[ATTR_A2A_AGENT_URL] == "http://agents.example.com/chat"
        assert attrs[ATTR_A2A_METHOD] == "message/stream"
        assert attrs[ATTR_A2A_MESSAGE_ID] == "chat-789"
        assert attrs[ATTR_A2A_STREAM] is True
        assert attrs[ATTR_A2A_SUCCESS] is True
        assert attrs[ATTR_HTTP_METHOD] == "POST"

    def test_streaming_span_records_error_on_failure(self, clear_spans):
        """Test that StreamingSpan handles errors correctly."""
        exporter = clear_spans
        tracer = trace.get_tracer("test.a2a_gateway")

        streaming_span = trace_agent_stream(
            tracer,
            agent_id="failing-stream",
            agent_name="Failing Stream",
            agent_url="http://fail:9000",
            method="message/stream",
        )

        with pytest.raises(TimeoutError):
            with streaming_span:
                raise TimeoutError("Stream timeout")

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.status.status_code == StatusCode.ERROR
        assert "Stream timeout" in span.status.description

        attrs = dict(span.attributes)
        assert attrs[ATTR_A2A_SUCCESS] is False
        assert attrs[ATTR_A2A_ERROR] == "Stream timeout"

    def test_streaming_span_mark_error_ends_span(self, clear_spans):
        """Test that mark_error properly ends the span with error status."""
        exporter = clear_spans
        tracer = trace.get_tracer("test.a2a_gateway")

        streaming_span = trace_agent_stream(
            tracer,
            agent_id="error-stream",
            agent_name="Error Stream",
            agent_url="http://error:9000",
            method="message/stream",
        )

        with streaming_span:
            # Simulate detecting an error during streaming
            streaming_span.mark_error(ConnectionError("Connection lost"))

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.status.status_code == StatusCode.ERROR

        attrs = dict(span.attributes)
        assert attrs[ATTR_A2A_SUCCESS] is False
        assert "Connection lost" in attrs[ATTR_A2A_ERROR]

    def test_streaming_span_set_attribute(self, clear_spans):
        """Test that set_attribute works on StreamingSpan."""
        exporter = clear_spans
        tracer = trace.get_tracer("test.a2a_gateway")

        streaming_span = trace_agent_stream(
            tracer,
            agent_id="attr-stream",
            agent_name="Attr Stream",
            agent_url="http://attr:9000",
            method="message/stream",
        )

        with streaming_span:
            streaming_span.set_attribute("custom.chunks", 42)
            streaming_span.set_attribute("custom.bytes", 1024)
            streaming_span.mark_success()

        spans = exporter.get_finished_spans()
        span = spans[0]
        attrs = dict(span.attributes)

        assert attrs["custom.chunks"] == 42
        assert attrs["custom.bytes"] == 1024

    def test_streaming_span_null_tracer(self):
        """Test that StreamingSpan works safely when tracer is None."""
        streaming_span = trace_agent_stream(
            tracer=None,
            agent_id="test",
            agent_name="Test",
            agent_url="http://test",
            method="message/stream",
        )

        with streaming_span:
            streaming_span.set_attribute("key", "value")
            streaming_span.mark_success()

        # Should not raise any exceptions


class TestInjectTraceHeaders:
    """Test suite for inject_trace_headers function."""

    def test_inject_headers_adds_traceparent(self, clear_spans):
        """Test that inject_trace_headers adds W3C traceparent header."""
        tracer = trace.get_tracer("test.a2a_gateway")

        with tracer.start_as_current_span("test-span"):
            headers = {"Content-Type": "application/json"}
            result = inject_trace_headers(headers)

            # Should contain traceparent header
            assert "traceparent" in result

    def test_inject_headers_preserves_existing(self, clear_spans):
        """Test that inject_trace_headers preserves existing headers."""
        tracer = trace.get_tracer("test.a2a_gateway")

        with tracer.start_as_current_span("test-span"):
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer token",
            }
            result = inject_trace_headers(headers)

            # Original headers should be preserved
            assert result["Content-Type"] == "application/json"
            assert result["Authorization"] == "Bearer token"


class TestGetTracer:
    """Test suite for get_tracer function."""

    def test_returns_tracer_when_otel_available(self):
        """Test that get_tracer returns a valid tracer."""
        tracer = get_tracer()
        assert tracer is not None

    def test_tracer_name_is_correct(self, clear_spans):
        """Test that the tracer can create spans."""
        tracer = get_tracer()
        # The tracer should be able to create spans
        with tracer.start_as_current_span("test-span"):
            current = trace.get_current_span()
            assert current is not None


class TestAsyncContextPreservation:
    """Tests for async context preservation in FastAPI-like environments."""

    @pytest.mark.asyncio
    async def test_span_context_preserved_across_await(self, clear_spans):
        """Test that span context is preserved across async await points."""
        import asyncio

        exporter = clear_spans
        tracer = trace.get_tracer("test.a2a_gateway")

        async def simulate_network_call():
            await asyncio.sleep(0.001)
            return {"result": "success"}

        with trace_agent_send(
            tracer,
            agent_id="async-agent",
            agent_name="Async Agent",
            agent_url="http://localhost:9000",
            method="message/send",
        ) as span:
            # Simulate async operations within the span
            result = await simulate_network_call()
            span.set_attribute("result.status", result["result"])

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "a2a.agent.send/async-agent"
        attrs = dict(span.attributes)
        assert attrs["result.status"] == "success"

    @pytest.mark.asyncio
    async def test_streaming_span_across_async_generator(self, clear_spans):
        """Test that StreamingSpan works correctly with async generators."""
        import asyncio

        exporter = clear_spans
        tracer = trace.get_tracer("test.a2a_gateway")

        async def mock_stream():
            for i in range(3):
                await asyncio.sleep(0.001)
                yield f"chunk-{i}"

        streaming_span = trace_agent_stream(
            tracer,
            agent_id="gen-stream",
            agent_name="Generator Stream",
            agent_url="http://localhost:9000",
            method="message/stream",
        )

        chunks_received = []
        with streaming_span:
            async for chunk in mock_stream():
                chunks_received.append(chunk)
            streaming_span.mark_success()

        assert chunks_received == ["chunk-0", "chunk-1", "chunk-2"]

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "a2a.agent.stream/gen-stream"
        assert span.status.status_code == StatusCode.OK
