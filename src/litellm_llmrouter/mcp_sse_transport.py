"""
MCP SSE Transport - Server-Sent Events Transport for MCP
=========================================================

Provides native SSE transport for MCP protocol, enabling real-time
streaming of server events to clients (e.g., Claude Desktop, IDE MCP clients).

Transport Modes:
----------------
1. Legacy SSE (MCP spec compliant):
   - GET /mcp/sse: Establishes SSE connection, emits `endpoint` event with POST URL
   - POST /mcp/messages?sessionId=<id>: Submit JSON-RPC requests, responses via SSE
   - Per-session async queues for routing responses to correct stream

2. Modern SSE (session-based):
   - Same flow but with enhanced session management
   - Heartbeat/ping events to maintain connection
   - Session timeouts and cleanup

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
- MCP_SSE_SESSION_TIMEOUT: Session cleanup timeout in seconds (default: 300)

Protocol References:
-------------------
- MCP Spec: https://modelcontextprotocol.io/specification/2024-11-05/transport/http
- SSE Spec: https://html.spec.whatwg.org/multipage/server-sent-events.html

Thread Safety:
--------------
- Session management uses asyncio locks
- Event emission is async-safe
- Per-session queues for message passing
"""

import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Query, Request
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

# Session timeout in seconds (default: 5 minutes of inactivity)
MCP_SSE_SESSION_TIMEOUT = float(os.getenv("MCP_SSE_SESSION_TIMEOUT", "300"))

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
    """
    Represents an active SSE connection session.

    Each session has an async queue for receiving JSON-RPC responses from
    POST /mcp/messages requests. The SSE stream generator reads from this queue
    and emits events to the client.
    """

    session_id: str
    client_id: str | None = None
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    last_event_id: int = 0
    is_active: bool = True
    is_initialized: bool = False
    protocol_version: str = MCP_PROTOCOL_VERSION
    capabilities: dict[str, Any] = field(default_factory=dict)
    # Async queue for JSON-RPC responses - created lazily
    _response_queue: asyncio.Queue | None = field(default=None, repr=False)
    # Lock for thread-safe queue creation
    _queue_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def next_event_id(self) -> str:
        """Generate the next event ID for this session."""
        self.last_event_id += 1
        return str(self.last_event_id)

    def touch(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = time.time()

    def is_expired(self) -> bool:
        """Check if session has timed out due to inactivity."""
        return time.time() - self.last_activity > MCP_SSE_SESSION_TIMEOUT

    async def get_queue(self) -> asyncio.Queue:
        """Get or create the response queue (thread-safe)."""
        if self._response_queue is None:
            async with self._queue_lock:
                if self._response_queue is None:
                    self._response_queue = asyncio.Queue()
        return self._response_queue

    async def send_response(self, response: dict[str, Any]) -> None:
        """Send a JSON-RPC response to the SSE stream."""
        queue = await self.get_queue()
        await queue.put(response)
        self.touch()


# In-memory session store (for HA, would need Redis)
_sse_sessions: dict[str, SSESession] = {}
# Lock for session store operations
_sessions_lock = asyncio.Lock()


async def get_session(session_id: str) -> SSESession | None:
    """Get a session by ID if it exists and is active."""
    session = _sse_sessions.get(session_id)
    if session and session.is_active and not session.is_expired():
        return session
    return None


async def cleanup_expired_sessions() -> int:
    """Remove expired sessions from the registry. Returns count removed."""
    async with _sessions_lock:
        expired = [
            sid for sid, session in _sse_sessions.items()
            if session.is_expired() or not session.is_active
        ]
        for sid in expired:
            del _sse_sessions[sid]
        return len(expired)


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


def get_messages_endpoint_url(request: Request, session_id: str) -> str:
    """
    Build the POST endpoint URL for a session.

    Uses the request's base URL to construct an absolute URL for the
    /mcp/messages endpoint with sessionId query parameter.
    """
    # Get the base URL from request (handles reverse proxy scenarios)
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host", request.headers.get("host", request.url.netloc))
    return f"{scheme}://{host}/mcp/messages?sessionId={session_id}"


async def generate_sse_events(
    session: SSESession,
    request: Request,
) -> AsyncGenerator[str, None]:
    """
    Generator for SSE event stream.

    Yields:
        SSE-formatted events including:
        - Legacy 'endpoint' event with POST URL (MCP SSE spec)
        - JSON-RPC responses from the session queue
        - Heartbeat/ping events to keep connection alive
        - Session events (created, expired)
    """
    start_time = time.time()

    try:
        # Legacy MCP SSE: Send 'endpoint' event with POST URL
        # This tells clients where to POST JSON-RPC requests
        messages_url = get_messages_endpoint_url(request, session.session_id)
        yield format_sse_event(
            data=messages_url,
            event="endpoint",
            event_id=session.next_event_id(),
            retry=MCP_SSE_RETRY_INTERVAL_MS,
        )

        verbose_proxy_logger.info(
            f"MCP SSE: Session {session.session_id} connected, endpoint: {messages_url}"
        )

        # Get the response queue for this session
        response_queue = await session.get_queue()

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

            # Try to get a response from the queue with timeout
            try:
                response = await asyncio.wait_for(
                    response_queue.get(),
                    timeout=0.1,  # Short timeout to allow heartbeat checks
                )
                # Emit the JSON-RPC response as a 'message' event
                yield format_sse_event(
                    data=response,
                    event="message",
                    event_id=session.next_event_id(),
                )
                session.touch()
            except asyncio.TimeoutError:
                pass  # No response in queue, continue

            # Send heartbeat if needed
            now = time.time()
            if now - last_heartbeat >= MCP_SSE_HEARTBEAT_INTERVAL:
                yield format_sse_comment(f"ping {int(now)}")
                last_heartbeat = now

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
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}",
                },
            },
            event="message",
            event_id=session.next_event_id(),
        )
    finally:
        session.is_active = False
        async with _sessions_lock:
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

    # Register session in global store
    async with _sessions_lock:
        # Cleanup expired sessions periodically
        await cleanup_expired_sessions()
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


@mcp_sse_router.post("/messages")
async def mcp_legacy_messages_endpoint(
    request: Request,
    sessionId: str = Query(..., description="Session ID from SSE endpoint event"),
) -> JSONResponse:
    """
    MCP Legacy SSE Messages Endpoint.

    POST /mcp/messages?sessionId=<session-id>

    Handles JSON-RPC requests for the legacy MCP SSE transport.
    Dispatches requests to the same handlers as POST /mcp, but sends
    responses via the SSE stream associated with the session.

    This endpoint supports all standard MCP methods:
    - initialize: Session initialization
    - tools/list: List available tools
    - tools/call: Invoke a tool
    - resources/list: List available resources

    The response is pushed to the session's SSE queue and emitted as
    a 'message' event on the SSE stream.

    Request headers:
    - Content-Type: application/json
    - Authorization: Bearer <api-key>

    Request body (JSON-RPC 2.0):
    ```json
    {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {...}
    }
    ```

    Response:
    - HTTP 202 Accepted (response sent via SSE)
    - HTTP 400 for invalid session
    - HTTP 404 for expired/unknown session
    """
    http_request_id = get_request_id() or "unknown"

    # Get session
    session = await get_session(sessionId)
    if not session:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "session_not_found",
                "message": f"Session '{sessionId}' not found or expired",
                "request_id": http_request_id,
            },
        )

    # Parse request body
    try:
        body = await request.body()
        if not body:
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32700,
                    "message": "Empty request body",
                },
            }
            await session.send_response(error_response)
            return JSONResponse(
                content={"status": "accepted", "session_id": sessionId},
                status_code=202,
            )

        data = json.loads(body)
    except json.JSONDecodeError as e:
        error_response = {
            "jsonrpc": "2.0",
            "id": None,
            "error": {
                "code": -32700,
                "message": f"Invalid JSON: {str(e)}",
            },
        }
        await session.send_response(error_response)
        return JSONResponse(
            content={"status": "accepted", "session_id": sessionId},
            status_code=202,
        )

    # Validate JSON-RPC structure
    if not isinstance(data, dict):
        error_response = {
            "jsonrpc": "2.0",
            "id": None,
            "error": {
                "code": -32600,
                "message": "Request must be a JSON object",
            },
        }
        await session.send_response(error_response)
        return JSONResponse(
            content={"status": "accepted", "session_id": sessionId},
            status_code=202,
        )

    jsonrpc_version = data.get("jsonrpc")
    if jsonrpc_version != "2.0":
        error_response = {
            "jsonrpc": "2.0",
            "id": data.get("id"),
            "error": {
                "code": -32600,
                "message": f"Invalid JSON-RPC version: {jsonrpc_version}. Expected '2.0'",
            },
        }
        await session.send_response(error_response)
        return JSONResponse(
            content={"status": "accepted", "session_id": sessionId},
            status_code=202,
        )

    request_id = data.get("id")
    method = data.get("method")
    params = data.get("params")

    if not method or not isinstance(method, str):
        error_response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32600,
                "message": "Missing or invalid 'method' field",
            },
        }
        await session.send_response(error_response)
        return JSONResponse(
            content={"status": "accepted", "session_id": sessionId},
            status_code=202,
        )

    # Dispatch to handler
    try:
        response = await _dispatch_jsonrpc_method(method, request_id, params, session)
        await session.send_response(response)
    except Exception as e:
        verbose_proxy_logger.exception(
            f"MCP SSE: Error dispatching {method} for session {sessionId}: {e}"
        )
        error_response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32603,
                "message": f"Internal error: {str(e)}",
            },
        }
        await session.send_response(error_response)

    return JSONResponse(
        content={"status": "accepted", "session_id": sessionId},
        status_code=202,
    )


async def _dispatch_jsonrpc_method(
    method: str,
    request_id: int | str | None,
    params: dict[str, Any] | None,
    session: SSESession,
) -> dict[str, Any]:
    """
    Dispatch a JSON-RPC method to the appropriate handler.

    Reuses the same logic as mcp_jsonrpc.py handlers but returns
    dict responses instead of JSONResponse.
    """
    gateway = get_mcp_gateway()

    if not gateway.is_enabled():
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32003,  # MCP_GATEWAY_DISABLED
                "message": "MCP Gateway is not enabled. Set MCP_GATEWAY_ENABLED=true",
            },
        }

    if method == "initialize":
        return await _handle_initialize_sse(request_id, params, session, gateway)
    elif method == "tools/list":
        return await _handle_tools_list_sse(request_id, params, gateway)
    elif method == "tools/call":
        return await _handle_tools_call_sse(request_id, params, gateway)
    elif method == "resources/list":
        return await _handle_resources_list_sse(request_id, params, gateway)
    else:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32601,  # JSONRPC_METHOD_NOT_FOUND
                "message": f"Method '{method}' not found. Supported: initialize, tools/list, tools/call, resources/list",
            },
        }


async def _handle_initialize_sse(
    request_id: int | str | None,
    params: dict[str, Any] | None,
    session: SSESession,
    gateway: Any,
) -> dict[str, Any]:
    """Handle MCP initialize request."""
    session.is_initialized = True
    session.touch()

    # Server capabilities per MCP spec
    capabilities = {
        "tools": {
            "listChanged": True,
        },
        "resources": {
            "listChanged": True,
            "subscribe": False,
        },
    }

    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": capabilities,
            "serverInfo": {
                "name": MCP_SERVER_NAME,
                "version": MCP_SERVER_VERSION,
            },
        },
    }


async def _handle_tools_list_sse(
    request_id: int | str | None,
    params: dict[str, Any] | None,
    gateway: Any,
) -> dict[str, Any]:
    """Handle MCP tools/list request."""
    tools = []
    for server in gateway.list_servers():
        for tool_name in server.tools:
            namespaced_name = f"{server.server_id}.{tool_name}"

            tool_entry = {
                "name": namespaced_name,
            }

            if tool_name in server.tool_definitions:
                tool_def = server.tool_definitions[tool_name]
                tool_entry["description"] = tool_def.description or f"Tool from {server.name}"
                tool_entry["inputSchema"] = tool_def.input_schema or {"type": "object"}
            else:
                tool_entry["description"] = f"Tool from {server.name}"
                tool_entry["inputSchema"] = {"type": "object"}

            tools.append(tool_entry)

    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {"tools": tools},
    }


async def _handle_tools_call_sse(
    request_id: int | str | None,
    params: dict[str, Any] | None,
    gateway: Any,
) -> dict[str, Any]:
    """Handle MCP tools/call request."""
    if not gateway.is_tool_invocation_enabled():
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32002,  # MCP_TOOL_INVOCATION_DISABLED
                "message": "Remote tool invocation is disabled. Set LLMROUTER_ENABLE_MCP_TOOL_INVOCATION=true",
            },
        }

    if not params:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32602,  # JSONRPC_INVALID_PARAMS
                "message": "Missing params for tools/call",
            },
        }

    namespaced_name = params.get("name")
    arguments = params.get("arguments", {})

    if not namespaced_name:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32602,
                "message": "Missing required param: name",
            },
        }

    # Parse namespaced tool name
    if "." in namespaced_name:
        parts = namespaced_name.split(".", 1)
        server_id = parts[0]
        tool_name = parts[1]
    else:
        tool_name = namespaced_name
        tool_def = gateway.get_tool(tool_name)
        if tool_def:
            server_id = tool_def.server_id
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32001,  # MCP_TOOL_NOT_FOUND
                    "message": f"Tool '{namespaced_name}' not found",
                },
            }

    # Verify server exists and has the tool
    server = gateway.get_server(server_id)
    if not server:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32001,
                "message": f"Server '{server_id}' not found for tool '{namespaced_name}'",
            },
        }

    if tool_name not in server.tools:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32001,
                "message": f"Tool '{tool_name}' not found on server '{server_id}'",
            },
        }

    # Invoke the tool
    try:
        result: MCPToolResult = await gateway.invoke_tool(tool_name, arguments)

        if result.success:
            content = [
                {
                    "type": "text",
                    "text": json.dumps(result.result) if not isinstance(result.result, str) else result.result,
                }
            ]
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"content": content},
            }
        else:
            content = [
                {
                    "type": "text",
                    "text": result.error or "Tool invocation failed",
                }
            ]
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"content": content, "isError": True},
            }

    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32603,
                "message": f"Tool invocation failed: {str(e)}",
            },
        }


async def _handle_resources_list_sse(
    request_id: int | str | None,
    params: dict[str, Any] | None,
    gateway: Any,
) -> dict[str, Any]:
    """Handle MCP resources/list request."""
    resources = []
    for res in gateway.list_resources():
        resources.append(
            {
                "uri": res.get("resource", ""),
                "name": res.get("resource", "").split("/")[-1] if res.get("resource") else "",
                "description": f"Resource from {res.get('server_name', 'unknown')}",
            }
        )

    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {"resources": resources},
    }


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
                    "messages_endpoint": "/mcp/messages",
                    "legacy_messages_endpoint": "/mcp/sse/messages",
                    "description": "Legacy SSE transport with endpoint event and async responses",
                },
                "http": {
                    "enabled": True,  # Always available
                    "endpoint": "/mcp",
                    "description": "Streamable HTTP transport (JSON-RPC over POST)",
                },
            },
            "current_mode": transport_mode,
            "legacy_mode": MCP_SSE_LEGACY_MODE,
            "config": {
                "heartbeat_interval": MCP_SSE_HEARTBEAT_INTERVAL,
                "max_connection_duration": MCP_SSE_MAX_CONNECTION_DURATION,
                "retry_interval_ms": MCP_SSE_RETRY_INTERVAL_MS,
                "session_timeout": MCP_SSE_SESSION_TIMEOUT,
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
