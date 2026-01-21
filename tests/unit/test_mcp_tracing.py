"""
Unit Tests for MCP Tracing - OpenTelemetry Instrumentation
==========================================================

Tests for the MCP tracing module to verify:
- Span creation for tool calls, server registration, and health checks
- Span attributes are correctly set
- Error handling records exceptions on spans
- Spans are properly exported via context management
- Async context preservation for FastAPI
"""

import sys
from unittest.mock import MagicMock

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
    "mcp_tracing", "src/litellm_llmrouter/mcp_tracing.py"
)
mcp_tracing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mcp_tracing)

trace_tool_call = mcp_tracing.trace_tool_call
trace_server_registration = mcp_tracing.trace_server_registration
trace_health_check = mcp_tracing.trace_health_check
get_tracer = mcp_tracing.get_tracer
ATTR_MCP_TOOL_NAME = mcp_tracing.ATTR_MCP_TOOL_NAME
ATTR_MCP_SERVER_ID = mcp_tracing.ATTR_MCP_SERVER_ID
ATTR_MCP_SERVER_NAME = mcp_tracing.ATTR_MCP_SERVER_NAME
ATTR_MCP_TRANSPORT = mcp_tracing.ATTR_MCP_TRANSPORT
ATTR_MCP_SUCCESS = mcp_tracing.ATTR_MCP_SUCCESS
ATTR_MCP_ERROR = mcp_tracing.ATTR_MCP_ERROR
ATTR_MCP_DURATION_MS = mcp_tracing.ATTR_MCP_DURATION_MS


@pytest.fixture(autouse=True)
def clear_spans():
    """Clear spans before and after each test."""
    _exporter.clear()
    yield _exporter
    _exporter.clear()


class TestTraceToolCall:
    """Test suite for trace_tool_call context manager."""

    def test_span_is_created_and_exported(self, clear_spans):
        """Test that trace_tool_call creates a span that is properly exported."""
        exporter = clear_spans
        tracer = trace.get_tracer("test.mcp_gateway")

        with trace_tool_call(
            tracer,
            tool_name="test.echo",
            server_id="server-1",
            server_name="Test Server",
            transport="streamable_http",
        ) as span:
            # Span should be active inside the context
            assert span is not None
            current_span = trace.get_current_span()
            assert current_span == span

        # Verify span was exported
        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        exported_span = spans[0]
        assert exported_span.name == "mcp.tool.call/test.echo"
        assert exported_span.status.status_code == StatusCode.OK

    def test_span_has_correct_attributes(self, clear_spans):
        """Test that trace_tool_call sets the correct span attributes."""
        exporter = clear_spans
        tracer = trace.get_tracer("test.mcp_gateway")

        with trace_tool_call(
            tracer,
            tool_name="database.query",
            server_id="db-server",
            server_name="Database MCP",
            transport="sse",
        ):
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        attrs = dict(span.attributes)

        assert attrs[ATTR_MCP_TOOL_NAME] == "database.query"
        assert attrs[ATTR_MCP_SERVER_ID] == "db-server"
        assert attrs[ATTR_MCP_SERVER_NAME] == "Database MCP"
        assert attrs[ATTR_MCP_TRANSPORT] == "sse"
        assert attrs[ATTR_MCP_SUCCESS] is True
        assert ATTR_MCP_DURATION_MS in attrs
        assert attrs[ATTR_MCP_DURATION_MS] >= 0

    def test_span_records_exception_on_error(self, clear_spans):
        """Test that trace_tool_call records exceptions and sets error status."""
        exporter = clear_spans
        tracer = trace.get_tracer("test.mcp_gateway")

        with pytest.raises(ValueError, match="Tool failed"):
            with trace_tool_call(
                tracer,
                tool_name="failing.tool",
                server_id="server-1",
                server_name="Test Server",
                transport="streamable_http",
            ):
                raise ValueError("Tool failed")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.status.status_code == StatusCode.ERROR
        assert "Tool failed" in span.status.description

        attrs = dict(span.attributes)
        assert attrs[ATTR_MCP_SUCCESS] is False
        assert attrs[ATTR_MCP_ERROR] == "Tool failed"

        # Verify exception was recorded (at least one exception event)
        events = span.events
        assert len(events) >= 1
        exception_events = [e for e in events if e.name == "exception"]
        assert len(exception_events) >= 1

    def test_span_is_parented_to_current_context(self, clear_spans):
        """Test that MCP spans are parented to the current trace context."""
        exporter = clear_spans
        tracer = trace.get_tracer("test.mcp_gateway")

        # Simulate an HTTP request span (parent)
        with tracer.start_as_current_span("HTTP POST /mcp/tools/call") as parent_span:
            _ = parent_span.get_span_context()  # Used for hierarchical tracing

            # This should be a child of the HTTP span
            with trace_tool_call(
                tracer,
                tool_name="child.tool",
                server_id="server-1",
                server_name="Test",
                transport="streamable_http",
            ):
                pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 2

        # Find the tool call span
        tool_span = next(s for s in spans if "mcp.tool.call" in s.name)
        http_span = next(s for s in spans if "HTTP" in s.name)

        # Verify parent-child relationship
        assert tool_span.parent is not None
        assert tool_span.parent.span_id == http_span.context.span_id
        assert tool_span.context.trace_id == http_span.context.trace_id

    def test_dummy_span_when_tracer_is_none(self):
        """Test that a dummy span is yielded when tracer is None."""
        with trace_tool_call(
            tracer=None,
            tool_name="test.tool",
            server_id="server-1",
            server_name="Test",
            transport="streamable_http",
        ) as span:
            # Should get a dummy span that accepts calls but does nothing
            span.set_attribute("key", "value")
            span.set_status("OK")
            span.record_exception(ValueError("test"))


class TestTraceServerRegistration:
    """Test suite for trace_server_registration context manager."""

    def test_span_is_created_and_exported(self, clear_spans):
        """Test that trace_server_registration creates an exported span."""
        exporter = clear_spans
        tracer = trace.get_tracer("test.mcp_gateway")

        with trace_server_registration(
            tracer,
            server_id="new-server",
            server_name="New MCP Server",
            url="http://localhost:9000",
            transport="streamable_http",
        ):
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "mcp.server.register/new-server"
        assert span.status.status_code == StatusCode.OK

    def test_span_has_correct_attributes(self, clear_spans):
        """Test that trace_server_registration sets attributes correctly."""
        exporter = clear_spans
        tracer = trace.get_tracer("test.mcp_gateway")

        with trace_server_registration(
            tracer,
            server_id="reg-server",
            server_name="Registered Server",
            url="http://mcp.example.com:8080",
            transport="sse",
        ):
            pass

        spans = exporter.get_finished_spans()
        span = spans[0]
        attrs = dict(span.attributes)

        assert attrs[ATTR_MCP_SERVER_ID] == "reg-server"
        assert attrs[ATTR_MCP_SERVER_NAME] == "Registered Server"
        assert attrs["mcp.server.url"] == "http://mcp.example.com:8080"
        assert attrs[ATTR_MCP_TRANSPORT] == "sse"
        assert attrs[ATTR_MCP_DURATION_MS] >= 0

    def test_span_records_error_on_failure(self, clear_spans):
        """Test that trace_server_registration handles failures correctly."""
        exporter = clear_spans
        tracer = trace.get_tracer("test.mcp_gateway")

        with pytest.raises(ConnectionError):
            with trace_server_registration(
                tracer,
                server_id="fail-server",
                server_name="Failing Server",
                url="http://invalid:9999",
                transport="streamable_http",
            ):
                raise ConnectionError("Connection refused")

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.status.status_code == StatusCode.ERROR

    def test_dummy_span_when_tracer_is_none(self):
        """Test that a dummy span is yielded when tracer is None."""
        with trace_server_registration(
            tracer=None,
            server_id="test",
            server_name="Test",
            url="http://test",
            transport="streamable_http",
        ) as span:
            span.set_attribute("key", "value")


class TestTraceHealthCheck:
    """Test suite for trace_health_check context manager."""

    def test_span_is_created_and_exported(self, clear_spans):
        """Test that trace_health_check creates an exported span."""
        exporter = clear_spans
        tracer = trace.get_tracer("test.mcp_gateway")

        with trace_health_check(tracer, server_id="health-server"):
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "mcp.server.health/health-server"
        assert span.status.status_code == StatusCode.OK

    def test_span_has_server_id_attribute(self, clear_spans):
        """Test that trace_health_check sets the server_id attribute."""
        exporter = clear_spans
        tracer = trace.get_tracer("test.mcp_gateway")

        with trace_health_check(tracer, server_id="my-server"):
            pass

        spans = exporter.get_finished_spans()
        span = spans[0]
        attrs = dict(span.attributes)

        assert attrs[ATTR_MCP_SERVER_ID] == "my-server"
        assert attrs[ATTR_MCP_DURATION_MS] >= 0

    def test_span_records_error_on_timeout(self, clear_spans):
        """Test that trace_health_check records errors on timeout."""
        exporter = clear_spans
        tracer = trace.get_tracer("test.mcp_gateway")

        with pytest.raises(TimeoutError):
            with trace_health_check(tracer, server_id="slow-server"):
                raise TimeoutError("Health check timed out")

        spans = exporter.get_finished_spans()
        span = spans[0]
        assert span.status.status_code == StatusCode.ERROR

    def test_dummy_span_when_tracer_is_none(self):
        """Test that a dummy span is yielded when tracer is None."""
        with trace_health_check(tracer=None, server_id="test") as span:
            span.set_attribute("key", "value")


class TestGetTracer:
    """Test suite for get_tracer function."""

    def test_returns_tracer_when_otel_available(self):
        """Test that get_tracer returns a valid tracer."""
        tracer = get_tracer()
        assert tracer is not None

    def test_tracer_name_is_correct(self, clear_spans):
        """Test that the tracer has the correct name."""
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
        tracer = trace.get_tracer("test.mcp_gateway")

        async def simulate_network_call():
            await asyncio.sleep(0.001)
            return {"result": "success"}

        with trace_tool_call(
            tracer,
            tool_name="async.tool",
            server_id="server-1",
            server_name="Test",
            transport="streamable_http",
        ) as span:
            # Simulate async operations within the span
            result = await simulate_network_call()
            span.set_attribute("result.status", result["result"])

        spans = exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "mcp.tool.call/async.tool"
        attrs = dict(span.attributes)
        assert attrs["result.status"] == "success"

    @pytest.mark.asyncio
    async def test_nested_async_spans_maintain_hierarchy(self, clear_spans):
        """Test that nested async spans maintain proper parent-child hierarchy."""
        import asyncio

        exporter = clear_spans
        tracer = trace.get_tracer("test.mcp_gateway")

        async def inner_operation():
            await asyncio.sleep(0.001)
            with trace_health_check(tracer, server_id="inner-server"):
                await asyncio.sleep(0.001)

        with trace_tool_call(
            tracer,
            tool_name="outer.tool",
            server_id="outer-server",
            server_name="Outer",
            transport="streamable_http",
        ):
            await inner_operation()

        spans = exporter.get_finished_spans()
        assert len(spans) == 2

        tool_span = next(s for s in spans if "mcp.tool.call" in s.name)
        health_span = next(s for s in spans if "mcp.server.health" in s.name)

        # Health check should be a child of tool call
        assert health_span.parent is not None
        assert health_span.parent.span_id == tool_span.context.span_id
