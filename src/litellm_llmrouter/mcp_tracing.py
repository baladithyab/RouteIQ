"""
MCP Tracing - OpenTelemetry Instrumentation for MCP Gateway
============================================================

Provides OTel tracing for MCP tool calls with:
- Spans for tool invocations (mcp.tool.call)
- Attributes: server_id, tool_name, transport, success/failure, duration
- Error recording with exception details
- Integration with the global OTel tracer

Usage:
    from litellm_llmrouter.mcp_tracing import instrument_mcp_gateway

    instrument_mcp_gateway()
"""

import functools
import os
import time
from typing import Any

from litellm._logging import verbose_proxy_logger

# OTel imports - these are optional dependencies
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode, Span

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None
    Status = None
    StatusCode = None
    Span = None


# Tracer name for MCP operations
TRACER_NAME = "litellm.mcp_gateway"

# Span attribute names (following OTel semantic conventions where applicable)
ATTR_MCP_SERVER_ID = "mcp.server.id"
ATTR_MCP_SERVER_NAME = "mcp.server.name"
ATTR_MCP_TOOL_NAME = "mcp.tool.name"
ATTR_MCP_TRANSPORT = "mcp.transport"
ATTR_MCP_SUCCESS = "mcp.success"
ATTR_MCP_ERROR = "mcp.error"
ATTR_MCP_DURATION_MS = "mcp.duration_ms"
ATTR_MCP_RESULT_TYPE = "mcp.result.type"


def get_tracer() -> Any:
    """
    Get the OTel tracer for MCP operations.

    Returns:
        OpenTelemetry Tracer instance, or None if OTel is not available
    """
    if not OTEL_AVAILABLE:
        return None

    try:
        return trace.get_tracer(TRACER_NAME)
    except Exception as e:
        verbose_proxy_logger.warning(f"Failed to get OTel tracer: {e}")
        return None


def trace_tool_call(
    tracer: Any, tool_name: str, server_id: str, server_name: str, transport: str
):
    """
    Context manager for tracing an MCP tool call.

    Args:
        tracer: OTel tracer instance
        tool_name: Name of the tool being called
        server_id: ID of the MCP server
        server_name: Name of the MCP server
        transport: Transport type (e.g., "streamable_http", "sse")

    Yields:
        The span, or None if tracing is disabled
    """
    if tracer is None:
        # Tracing disabled - use a dummy context manager
        class DummySpan:
            def set_attribute(self, *args, **kwargs):
                pass

            def set_status(self, *args, **kwargs):
                pass

            def record_exception(self, *args, **kwargs):
                pass

        class DummyContext:
            def __enter__(self):
                return DummySpan()

            def __exit__(self, *args):
                pass

        return DummyContext()

    # Start span with tool call name
    span = tracer.start_span(f"mcp.tool.call/{tool_name}")
    span.set_attribute(ATTR_MCP_TOOL_NAME, tool_name)
    span.set_attribute(ATTR_MCP_SERVER_ID, server_id)
    span.set_attribute(ATTR_MCP_SERVER_NAME, server_name)
    span.set_attribute(ATTR_MCP_TRANSPORT, transport)

    class SpanContext:
        def __init__(self, span):
            self._span = span
            self._start_time = time.perf_counter()

        def __enter__(self):
            return self._span

        def __exit__(self, exc_type, exc_val, exc_tb):
            duration_ms = (time.perf_counter() - self._start_time) * 1000
            self._span.set_attribute(ATTR_MCP_DURATION_MS, round(duration_ms, 2))

            if exc_val is not None:
                self._span.set_attribute(ATTR_MCP_SUCCESS, False)
                self._span.set_attribute(ATTR_MCP_ERROR, str(exc_val))
                self._span.record_exception(exc_val)
                self._span.set_status(Status(StatusCode.ERROR, str(exc_val)))
            else:
                self._span.set_attribute(ATTR_MCP_SUCCESS, True)
                self._span.set_status(Status(StatusCode.OK))

            self._span.end()
            return False  # Don't suppress exceptions

    return SpanContext(span)


def trace_server_registration(
    tracer: Any, server_id: str, server_name: str, url: str, transport: str
):
    """
    Context manager for tracing MCP server registration.

    Args:
        tracer: OTel tracer instance
        server_id: ID of the MCP server
        server_name: Name of the MCP server
        url: URL of the MCP server
        transport: Transport type

    Yields:
        The span, or None if tracing is disabled
    """
    if tracer is None:

        class DummyContext:
            def __enter__(self):
                return type(
                    "DummySpan",
                    (),
                    {
                        "set_attribute": lambda *a, **k: None,
                        "set_status": lambda *a, **k: None,
                    },
                )()

            def __exit__(self, *args):
                pass

        return DummyContext()

    span = tracer.start_span(f"mcp.server.register/{server_id}")
    span.set_attribute(ATTR_MCP_SERVER_ID, server_id)
    span.set_attribute(ATTR_MCP_SERVER_NAME, server_name)
    span.set_attribute("mcp.server.url", url)
    span.set_attribute(ATTR_MCP_TRANSPORT, transport)

    class SpanContext:
        def __init__(self, span):
            self._span = span
            self._start_time = time.perf_counter()

        def __enter__(self):
            return self._span

        def __exit__(self, exc_type, exc_val, exc_tb):
            duration_ms = (time.perf_counter() - self._start_time) * 1000
            self._span.set_attribute(ATTR_MCP_DURATION_MS, round(duration_ms, 2))

            if exc_val is not None:
                self._span.set_status(Status(StatusCode.ERROR, str(exc_val)))
            else:
                self._span.set_status(Status(StatusCode.OK))

            self._span.end()
            return False

    return SpanContext(span)


def trace_health_check(tracer: Any, server_id: str):
    """
    Context manager for tracing MCP server health checks.

    Args:
        tracer: OTel tracer instance
        server_id: ID of the MCP server being checked

    Yields:
        The span, or None if tracing is disabled
    """
    if tracer is None:

        class DummyContext:
            def __enter__(self):
                return type(
                    "DummySpan",
                    (),
                    {
                        "set_attribute": lambda *a, **k: None,
                        "set_status": lambda *a, **k: None,
                    },
                )()

            def __exit__(self, *args):
                pass

        return DummyContext()

    span = tracer.start_span(f"mcp.server.health/{server_id}")
    span.set_attribute(ATTR_MCP_SERVER_ID, server_id)

    class SpanContext:
        def __init__(self, span):
            self._span = span
            self._start_time = time.perf_counter()

        def __enter__(self):
            return self._span

        def __exit__(self, exc_type, exc_val, exc_tb):
            duration_ms = (time.perf_counter() - self._start_time) * 1000
            self._span.set_attribute(ATTR_MCP_DURATION_MS, round(duration_ms, 2))

            if exc_val is not None:
                self._span.set_status(Status(StatusCode.ERROR, str(exc_val)))
            else:
                self._span.set_status(Status(StatusCode.OK))

            self._span.end()
            return False

    return SpanContext(span)


class TracedMCPGatewayMixin:
    """
    Mixin class that adds OTel tracing to MCPGateway methods.

    This is applied to the MCPGateway instance at runtime to add
    tracing without modifying the core class.
    """

    _tracer: Any = None

    def _get_tracer(self) -> Any:
        """Get or create the tracer instance."""
        if self._tracer is None:
            self._tracer = get_tracer()
        return self._tracer


def instrument_mcp_gateway() -> bool:
    """
    Instrument the MCP gateway with OTel tracing.

    This modifies the global MCPGateway instance to add tracing
    to key operations like tool calls, server registration, and health checks.

    Returns:
        True if instrumentation was successful, False otherwise
    """
    if not OTEL_AVAILABLE:
        verbose_proxy_logger.info(
            "MCP tracing: OTel not available, skipping instrumentation"
        )
        return False

    if os.getenv("MCP_TRACING_ENABLED", "true").lower() != "true":
        verbose_proxy_logger.info("MCP tracing: Disabled via MCP_TRACING_ENABLED=false")
        return False

    try:
        from .mcp_gateway import get_mcp_gateway

        gateway = get_mcp_gateway()
        tracer = get_tracer()

        if tracer is None:
            verbose_proxy_logger.warning("MCP tracing: Could not get OTel tracer")
            return False

        # Store tracer on gateway
        gateway._tracer = tracer

        # Wrap invoke_tool with tracing
        original_invoke_tool = gateway.invoke_tool

        @functools.wraps(original_invoke_tool)
        async def traced_invoke_tool(tool_name: str, arguments: dict[str, Any]):
            server = gateway.find_server_for_tool(tool_name)
            server_id = server.server_id if server else "unknown"
            server_name = server.name if server else "unknown"
            transport = server.transport.value if server else "unknown"

            with trace_tool_call(
                tracer, tool_name, server_id, server_name, transport
            ) as span:
                result = await original_invoke_tool(tool_name, arguments)
                if span and hasattr(span, "set_attribute"):
                    span.set_attribute(ATTR_MCP_SUCCESS, result.success)
                    if result.error:
                        span.set_attribute(ATTR_MCP_ERROR, result.error)
                return result

        gateway.invoke_tool = traced_invoke_tool

        # Wrap register_server with tracing
        original_register_server = gateway.register_server

        @functools.wraps(original_register_server)
        def traced_register_server(server):
            with trace_server_registration(
                tracer,
                server.server_id,
                server.name,
                server.url,
                server.transport.value,
            ):
                return original_register_server(server)

        gateway.register_server = traced_register_server

        # Wrap check_server_health with tracing
        original_check_health = gateway.check_server_health

        @functools.wraps(original_check_health)
        async def traced_check_health(server_id: str):
            with trace_health_check(tracer, server_id) as span:
                result = await original_check_health(server_id)
                if span and hasattr(span, "set_attribute"):
                    span.set_attribute(
                        "mcp.health.status", result.get("status", "unknown")
                    )
                return result

        gateway.check_server_health = traced_check_health

        verbose_proxy_logger.info("MCP tracing: Gateway instrumented successfully")
        return True

    except Exception as e:
        verbose_proxy_logger.error(f"MCP tracing: Failed to instrument gateway: {e}")
        return False


def is_tracing_enabled() -> bool:
    """Check if MCP tracing is enabled and configured."""
    if not OTEL_AVAILABLE:
        return False
    if os.getenv("MCP_TRACING_ENABLED", "true").lower() != "true":
        return False
    return get_tracer() is not None
