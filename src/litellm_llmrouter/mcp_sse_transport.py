"""
MCP SSE Transport - Server-Sent Events Transport for MCP
=========================================================

Provides native SSE transport for MCP protocol, enabling real-time
streaming of server events to clients (e.g., Claude Desktop, IDE MCP clients).

Transport Modes:
----------------
1. SSE (Server-Sent Events): GET /{path} with Accept: text/event-stream
   - Persistent connection for server → client events
   - Proper SSE framing: event:, data:, id:, retry:
   - Heartbeat/ping events to maintain connection
   - Supports tools/list, resources/list streaming

2. POST JSON-RPC: POST /{path} for client requests
   - Standard JSON-RPC 2.0 request/response
   - Works alongside SSE connection

SSE Event Format (per W3C spec):
--------------------------------
event: <event-type>
id: <event-id>
data: <json-payload>

<blank line>

Feature Flags:
--------------
- MCP_SSE_TRANSPORT_ENABLED: Enable SSE transport (default: true)
- MCP_SSE_LEGACY_MODE: Use pure HTTP fallback (default: false)
- MCP_SSE_HEARTBEAT_INTERVAL: Seconds between heartbeat events (default: 30)

Protocol References:
-------------------
- MCP Spec: https://modelcontextprotocol.io/specification/2024-11-05/transport/http
- SSE Spec: https://html.spec.whatwg.org/multipage/server-sent-events.html

Thread Safety:
--------------
- Session management uses threading locks
- Event emission is async-safe
"""

import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from litellm._logging import verbose_proxy_logger
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

from .auth import get_request_id
from .mcp_gateway import get_mcp_gateway, MCPToolResult

# ============================================================================
# Configuration & Feature Flags
# ============================================================================

# SSE transport feature flag (enabled by default for MCP compatibility)
MCP_SSE_TRANSPORT_ENABLED = (
    os.getenv("MCP_SSE_TRANSPORT_ENABLED", "true").lower() == "true"
)

# Legacy mode: disable SSE, use pure HTTP (for rollback safety)
MCP_SSE_LEGACY_MODE = os.getenv("MCP_SSE_LEGACY_MODE", "false").lower() == "true"

# Heartbeat interval in seconds (keeps SSE connection alive)
MCP_SSE_HEARTBEAT_INTERVAL = float(os.getenv("MCP_SSE_HEARTBEAT_INTERVAL", "30"))

# Maximum SSE connection duration in seconds (default: 30 minutes)
MCP_SSE_MAX_CONNECTION_DURATION = float(
    os.getenv("MCP_SSE_MAX_CONNECTION_DURATION", "1800")
)

# SSE retry interval hint for clients (milliseconds)
MCP_SSE_RETRY_INTERVAL_MS = int(os.getenv("MCP_SSE_RETRY_INTERVAL_MS", "3000"))

# Protocol version
MCP_PROTOCOL_VERSION = "2024-11-05"

# Server info
MCP_SERVER_NAME = os.getenv("MCP_SERVER_NAME", "routeiq-mcp-gateway")
MCP_SERVER_VERSION = os.getenv("MCP_SERVER_VERSION", "1.0.0")


# ============================================================================
# SSE Session Management
# ============================================================================


@dataclass
class SSESession:
    """Represents an active SSE connection session."""

    session_id: str
    client_id: str | None = None
    created_at: float = field(default_factory=time.time)
    last_event_id: int = 0
    is_active: bool = True
    protocol_version: str = MCP_PROTOCOL_VERSION
    capabilities: dict[str, Any] = field(default_factory=dict)

    def next_event_id(self) -> str:
        """Generate the next event ID for this session."""
        self.last_event_id += 1
        return str(self.last_event_id)


# In-memory session store (for HA, would need Redis)
_sse_sessions: dict[str, SSESession] = {}


def get_transport_mode() -> str:
    """
    Get the current transport mode.

    Returns:
        'sse': SSE transport enabled (default)
        'legacy': Pure HTTP transport (rollback mode)
        'disabled': SSE transport disabled
    """
    if not MCP_SSE_TRANSPORT_ENABLED:
        return "disabled"
    if MCP_SSE_LEGACY_MODE:
        return "legacy"
    return "sse"


# ============================================================================
# SSE Event Formatting
# ============================================================================


def format_sse_event(
    data: Any,
    event: str | None = None,
    event_id: str | None = None,
    retry: int | None = None,
) -> str:
    """
    Format a Server-Sent Event per W3C specification.

    Args:
        data: Event data (will be JSON-serialized if not a string)
        event: Optional event type (e.g., 'message', 'tools/list')
        event_id: Optional event ID for Last-Event-ID tracking
        retry: Optional retry interval in milliseconds

    Returns:
        SSE-formatted string with proper framing
    """
    lines: list[str] = []

    # Event type (optional)
    if event:
        lines.append(f"event: {event}")

    # Event ID (optional)
    if event_id:
        lines.append(f"id: {event_id}")

    # Retry interval (optional, typically sent once at connection start)
    if retry is not None:
        lines.append(f"retry: {retry}")

    # Data (required) - serialize JSON if needed
    if isinstance(data, str):
        data_str = data
    else:
        data_str = json.dumps(data)

    # SSE spec: multi-line data uses multiple "data:" lines
    for line in data_str.split("\n"):
        lines.append(f"data: {line}")

    # SSE events end with double newline
    lines.append("")
    lines.append("")

    return "\n".join(lines)


def format_sse_comment(comment: str) -> str:
    """
    Format an SSE comment (for heartbeats/keepalives).

    Comments start with ':' and are ignored by clients but keep
    the connection alive.
    """
    return f": {comment}\n\n"


# ============================================================================
# SSE Router
# ============================================================================

mcp_sse_router = APIRouter(
    prefix="/mcp",
    tags=["mcp-sse"],
    dependencies=[Depends(user_api_key_auth)],
)


# ============================================================================
# SSE Event Generators
# ============================================================================


async def generate_sse_events(
    session: SSESession,
    request: Request,
) -> AsyncGenerator[str, None]:
    """
    Generator for SSE event stream.

    Yields:
        SSE-formatted events including:
        - Connection established event
        - Heartbeat/ping events
        - Server data events (tools, resources, etc.)
    """
    start_time = time.time()

    try:
        # Send initial connection event with retry hint
        yield format_sse_event(
            data={
                "type": "session.created",
                "session_id": session.session_id,
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "serverInfo": {
                    "name": MCP_SERVER_NAME,
                    "version": MCP_SERVER_VERSION,
                },
                "capabilities": {
                    "tools": {"listChanged": True},
                    "resources": {"listChanged": True, "subscribe": False},
                },
            },
            event="session",
            event_id=session.next_event_id(),
            retry=MCP_SSE_RETRY_INTERVAL_MS,
        )

        verbose_proxy_logger.info(
            f"MCP SSE: Session {session.session_id} connected"
        )

        # Main event loop
        last_heartbeat = time.time()

        while session.is_active:
            # Check if client disconnected
            if await request.is_disconnected():
                verbose_proxy_logger.info(
                    f"MCP SSE: Client disconnected from session {session.session_id}"
                )
                break

            # Check max connection duration
            elapsed = time.time() - start_time
            if elapsed > MCP_SSE_MAX_CONNECTION_DURATION:
                verbose_proxy_logger.info(
                    f"MCP SSE: Session {session.session_id} exceeded max duration"
                )
                yield format_sse_event(
                    data={
                        "type": "session.expired",
                        "reason": "max_duration_exceeded",
                        "reconnect": True,
                    },
                    event="session",
                    event_id=session.next_event_id(),
                )
                break

            # Send heartbeat if needed
            now = time.time()
            if now - last_heartbeat >= MCP_SSE_HEARTBEAT_INTERVAL:
                yield format_sse_comment(f"ping {int(now)}")
                last_heartbeat = now

            # Small delay to reduce CPU usage
            await asyncio.sleep(0.1)

    except asyncio.CancelledError:
        verbose_proxy_logger.info(
            f"MCP SSE: Session {session.session_id} cancelled"
        )
    except Exception as e:
        verbose_proxy_logger.exception(
            f"MCP SSE: Error in session {session.session_id}: {e}"
        )
        yield format_sse_event(
            data={
                "type": "error",
                "code": -32603,
                "message": f"Internal error: {str(e)}",
            },
            event="error",
            event_id=session.next_event_id(),
        )
    finally:
        session.is_active = False
        if session.session_id in _sse_sessions:
            del _sse_sessions[session.session_id]
        verbose_proxy_logger.info(
            f"MCP SSE: Session {session.session_id} closed"
        )


# ============================================================================
# SSE Endpoint
# ============================================================================


@mcp_sse_router.get("/sse")
@mcp_sse_router.get("/sse/messages")
async def mcp_sse_endpoint(request: Request) -> StreamingResponse:
    """
    MCP SSE Transport Endpoint.

    GET /mcp/sse or /mcp/sse/messages

    Establishes a Server-Sent Events connection for real-time MCP events.
    Clients should connect with Accept: text/event-stream header.

    The connection will:
    1. Send initial session.created event with capabilities
    2. Periodically send heartbeat comments to keep connection alive
    3. Send events for tool/resource changes when they occur
    4. Close after MCP_SSE_MAX_CONNECTION_DURATION seconds

    Response headers:
    - Content-Type: text/event-stream
    - Cache-Control: no-cache
    - Connection: keep-alive
    - X-Accel-Buffering: no (disables nginx buffering)

    Example event stream:
    ```
    event: session
    id: 1
    retry: 3000
    data: {"type":"session.created","session_id":"...","protocolVersion":"2024-11-05"}

    : ping 1696012345

    event: tools/list_changed
    id: 2
    data: {"tools":[...]}
    ```
    """
    request_id = get_request_id() or "unknown"

    # Check transport mode
    transport_mode = get_transport_mode()
    if transport_mode == "disabled":
        raise HTTPException(
            status_code=404,
            detail={
                "error": "sse_transport_disabled",
                "message": "SSE transport is disabled. Set MCP_SSE_TRANSPORT_ENABLED=true",
                "request_id": request_id,
            },
        )

    if transport_mode == "legacy":
        raise HTTPException(
            status_code=404,
            detail={
                "error": "sse_legacy_mode",
                "message": "SSE transport in legacy mode. Use POST /mcp for JSON-RPC",
                "request_id": request_id,
            },
        )

    gateway = get_mcp_gateway()
    if not gateway.is_enabled():
        raise HTTPException(
            status_code=404,
            detail={
                "error": "mcp_gateway_disabled",
                "message": "MCP Gateway is not enabled. Set MCP_GATEWAY_ENABLED=true",
                "request_id": request_id,
            },
        )

    # Check Accept header
    accept = request.headers.get("accept", "")
    if "text/event-stream" not in accept and "*/*" not in accept:
        raise HTTPException(
            status_code=406,
            detail={
                "error": "not_acceptable",
                "message": "SSE endpoint requires Accept: text/event-stream",
                "request_id": request_id,
            },
        )

    # Create session
    session_id = str(uuid.uuid4())
    session = SSESession(
        session_id=session_id,
        client_id=request.headers.get("x-client-id"),
    )
    _sse_sessions[session_id] = session

    verbose_proxy_logger.info(
        f"MCP SSE: Creating session {session_id} for client {session.client_id or 'anonymous'}"
    )

    # Return SSE stream
    return StreamingResponse(
        generate_sse_events(session, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
            "X-SSE-Session-ID": session_id,
        },
    )


@mcp_sse_router.post("/sse/messages")
async def mcp_sse_post_endpoint(request: Request) -> JSONResponse:
    """
    MCP SSE Message POST Endpoint.

    POST /mcp/sse/messages

    Handles JSON-RPC requests associated with an SSE session.
    The session-id header links this request to an active SSE connection.

    This endpoint processes:
    - tools/call: Invoke a tool (response via SSE stream)
    - resources/read: Read a resource (response via SSE stream)

    Request headers:
    - Content-Type: application/json
    - X-SSE-Session-ID: <session-id> (optional, links to SSE stream)

    Request body (JSON-RPC 2.0):
    ```json
    {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "server.tool", "arguments": {...}}
    }
    ```
    """
    request_id = get_request_id() or "unknown"

    gateway = get_mcp_gateway()
    if not gateway.is_enabled():
        raise HTTPException(
            status_code=404,
            detail={
                "error": "mcp_gateway_disabled",
                "message": "MCP Gateway is not enabled",
                "request_id": request_id,
            },
        )

    # Parse request body
    try:
        body = await request.body()
        if not body:
            return JSONResponse(
                content={
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32700,
                        "message": "Empty request body",
                    },
                }
            )

        data = json.loads(body)
    except json.JSONDecodeError as e:
        return JSONResponse(
            content={
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32700,
                    "message": f"Invalid JSON: {str(e)}",
                },
            }
        )

    # Validate JSON-RPC structure
    if not isinstance(data, dict):
        return JSONResponse(
            content={
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32600,
                    "message": "Request must be a JSON object",
                },
            }
        )

    jsonrpc_version = data.get("jsonrpc")
    if jsonrpc_version != "2.0":
        return JSONResponse(
            content={
                "jsonrpc": "2.0",
                "id": data.get("id"),
                "error": {
                    "code": -32600,
                    "message": f"Invalid JSON-RPC version: {jsonrpc_version}",
                },
            }
        )

    request_id_jsonrpc = data.get("id")
    method = data.get("method")
    params = data.get("params", {})

    if not method:
        return JSONResponse(
            content={
                "jsonrpc": "2.0",
                "id": request_id_jsonrpc,
                "error": {
                    "code": -32600,
                    "message": "Missing 'method' field",
                },
            }
        )

    # Handle methods
    if method == "tools/call":
        return await _handle_sse_tools_call(request_id_jsonrpc, params, gateway)
    else:
        return JSONResponse(
            content={
                "jsonrpc": "2.0",
                "id": request_id_jsonrpc,
                "error": {
                    "code": -32601,
                    "message": f"Method '{method}' not supported via SSE POST. Use main /mcp endpoint.",
                },
            }
        )


async def _handle_sse_tools_call(
    request_id: int | str | None,
    params: dict[str, Any],
    gateway: Any,
) -> JSONResponse:
    """Handle tools/call via SSE POST endpoint."""
    if not gateway.is_tool_invocation_enabled():
        return JSONResponse(
            content={
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32002,
                    "message": "Remote tool invocation is disabled",
                },
            }
        )

    namespaced_name = params.get("name")
    arguments = params.get("arguments", {})

    if not namespaced_name:
        return JSONResponse(
            content={
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32602,
                    "message": "Missing required param: name",
                },
            }
        )

    # Parse namespaced tool name
    if "." in namespaced_name:
        parts = namespaced_name.split(".", 1)
        tool_name = parts[1]
    else:
        tool_name = namespaced_name

    # Invoke tool
    try:
        result: MCPToolResult = await gateway.invoke_tool(tool_name, arguments)

        if result.success:
            content = [
                {
                    "type": "text",
                    "text": (
                        json.dumps(result.result)
                        if not isinstance(result.result, str)
                        else result.result
                    ),
                }
            ]
            return JSONResponse(
                content={
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"content": content},
                }
            )
        else:
            content = [
                {
                    "type": "text",
                    "text": result.error or "Tool invocation failed",
                }
            ]
            return JSONResponse(
                content={
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"content": content, "isError": True},
                }
            )

    except Exception as e:
        return JSONResponse(
            content={
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": f"Tool invocation failed: {str(e)}",
                },
            }
        )


# ============================================================================
# Transport Info Endpoint
# ============================================================================


@mcp_sse_router.get("/transport")
async def mcp_transport_info() -> JSONResponse:
    """
    MCP Transport Information Endpoint.

    GET /mcp/transport

    Returns information about available transports and current configuration.

    Response:
    ```json
    {
        "transports": {
            "sse": {"enabled": true, "endpoint": "/mcp/sse"},
            "http": {"enabled": true, "endpoint": "/mcp"}
        },
        "current_mode": "sse",
        "config": {
            "heartbeat_interval": 30,
            "max_connection_duration": 1800,
            "retry_interval_ms": 3000
        }
    }
    ```
    """
    transport_mode = get_transport_mode()

    return JSONResponse(
        content={
            "transports": {
                "sse": {
                    "enabled": MCP_SSE_TRANSPORT_ENABLED and not MCP_SSE_LEGACY_MODE,
                    "endpoint": "/mcp/sse",
                    "messages_endpoint": "/mcp/sse/messages",
                },
                "http": {
                    "enabled": True,  # Always available
                    "endpoint": "/mcp",
                },
            },
            "current_mode": transport_mode,
            "legacy_mode": MCP_SSE_LEGACY_MODE,
            "config": {
                "heartbeat_interval": MCP_SSE_HEARTBEAT_INTERVAL,
                "max_connection_duration": MCP_SSE_MAX_CONNECTION_DURATION,
                "retry_interval_ms": MCP_SSE_RETRY_INTERVAL_MS,
            },
            "feature_flags": {
                "MCP_SSE_TRANSPORT_ENABLED": MCP_SSE_TRANSPORT_ENABLED,
                "MCP_SSE_LEGACY_MODE": MCP_SSE_LEGACY_MODE,
            },
        }
    )


# ============================================================================
# Active Sessions Endpoint (Admin)
# ============================================================================


@mcp_sse_router.get("/sse/sessions")
async def mcp_sse_sessions() -> JSONResponse:
    """
    List active SSE sessions (admin endpoint).

    GET /mcp/sse/sessions

    Returns information about all active SSE sessions.
    """
    sessions = []
    for session_id, session in _sse_sessions.items():
        sessions.append(
            {
                "session_id": session_id,
                "client_id": session.client_id,
                "created_at": session.created_at,
                "last_event_id": session.last_event_id,
                "is_active": session.is_active,
                "age_seconds": time.time() - session.created_at,
            }
        )

    return JSONResponse(
        content={
            "active_sessions": len(sessions),
            "sessions": sessions,
        }
    )
