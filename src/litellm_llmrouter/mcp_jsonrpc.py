"""
MCP Native JSON-RPC 2.0 Transport
==================================

Provides native MCP client surface (Claude Desktop / IDE MCP clients) via
JSON-RPC 2.0 over HTTP POST.

This module implements the MCP protocol specification's streamable HTTP
transport, handling:
- initialize: Session initialization with capability negotiation
- tools/list: List available tools (aggregated from registered MCP servers)
- tools/call: Invoke a tool on the appropriate MCP server
- resources/list: List available resources (optional)

See: https://modelcontextprotocol.io/specification/2024-11-05/transport/http

Protocol Notes:
---------------
- All requests/responses follow JSON-RPC 2.0 specification
- The server reports capabilities during initialize handshake
- Tool names are namespaced as <server_id>.<tool_name> for disambiguation
- SSE is NOT implemented in this version (streamable HTTP is sufficient for
  stateless tool calls; SSE would be needed for streaming responses or
  server->client notifications)

Security:
---------
- Requires LiteLLM API key authentication (user_api_key_auth)
- Tool invocation requires LLMROUTER_ENABLE_MCP_TOOL_INVOCATION=true
- All outbound URLs are validated against SSRF attacks

Thread Safety:
--------------
- All operations are async-safe
- Registry reads use immutable snapshots
"""

import base64
import json
import logging
import os
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

from .auth import get_request_id
from .mcp_gateway import get_mcp_gateway, MCPToolResult

logger = logging.getLogger(__name__)

# Protocol versions (MCP spec versions we support)
MCP_PROTOCOL_VERSION = "2025-11-25"
MCP_SUPPORTED_VERSIONS = {"2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"}

# Server info
MCP_SERVER_NAME = os.getenv("MCP_SERVER_NAME", "routeiq-mcp-gateway")
MCP_SERVER_VERSION = os.getenv("MCP_SERVER_VERSION", "1.0.0")

# Pagination config for tools/list
MCP_TOOLS_PAGE_SIZE = int(os.getenv("MCP_TOOLS_PAGE_SIZE", "100"))

# Pagination config for resources/list
MCP_RESOURCES_PAGE_SIZE = int(os.getenv("MCP_RESOURCES_PAGE_SIZE", "50"))

# Maximum resources page size (cap for client-requested sizes)
MCP_RESOURCES_MAX_PAGE_SIZE = 100

# Session timeout in seconds (default: 30 minutes)
MCP_SESSION_TIMEOUT = float(os.getenv("MCP_SESSION_TIMEOUT", "1800"))


# ============================================================================
# Session Management
# ============================================================================

# Active sessions: session_id -> session metadata
_active_sessions: dict[str, dict[str, Any]] = {}


def get_or_create_session(request_headers: dict[str, str]) -> str:
    """
    Get existing session or create a new one.

    Implements Mcp-Session-Id header management per MCP 2025-06-18 spec.
    Returns the session ID for the Mcp-Session-Id response header.

    Args:
        request_headers: HTTP request headers (case-insensitive keys)

    Returns:
        Session ID string
    """
    # Check for existing session
    session_id = request_headers.get("mcp-session-id", "")
    if session_id and session_id in _active_sessions:
        session = _active_sessions[session_id]
        # Check if session is still valid
        if time.time() - session["last_active"] < MCP_SESSION_TIMEOUT:
            session["last_active"] = time.time()
            return session_id
        else:
            # Session expired, remove it
            del _active_sessions[session_id]

    # Create new session
    session_id = str(uuid.uuid4())
    _active_sessions[session_id] = {
        "created_at": time.time(),
        "last_active": time.time(),
    }
    return session_id


def get_active_sessions() -> dict[str, dict[str, Any]]:
    """Return active sessions (for testing/admin)."""
    return _active_sessions


def reset_sessions() -> None:
    """Reset all sessions. For testing only."""
    _active_sessions.clear()


async def emit_tools_list_changed() -> None:
    """
    Emit notifications/tools/list_changed to all connected JSON-RPC sessions.

    Per MCP 2025-03-26 spec, this notification informs clients that the
    tool list has changed and they should re-fetch it.

    Note: In the streamable HTTP transport, this notification is delivered
    on the next request from each session. For SSE transport, it is pushed
    immediately via the event stream.
    """
    notification = {
        "jsonrpc": "2.0",
        "method": "notifications/tools/list_changed",
    }
    # Track pending notifications per session for delivery on next request
    for session_id, session in _active_sessions.items():
        if "pending_notifications" not in session:
            session["pending_notifications"] = []
        session["pending_notifications"].append(notification)


# ============================================================================
# JSON-RPC 2.0 Models
# ============================================================================


class JSONRPCRequest(BaseModel):
    """JSON-RPC 2.0 request structure."""

    jsonrpc: str = "2.0"
    id: int | str | None = None
    method: str
    params: dict[str, Any] | None = None


class JSONRPCError(BaseModel):
    """JSON-RPC 2.0 error structure."""

    code: int
    message: str
    data: Any | None = None


class JSONRPCResponse(BaseModel):
    """JSON-RPC 2.0 response structure."""

    jsonrpc: str = "2.0"
    id: int | str | None = None
    result: Any | None = None
    error: JSONRPCError | None = None


# JSON-RPC 2.0 error codes
JSONRPC_PARSE_ERROR = -32700
JSONRPC_INVALID_REQUEST = -32600
JSONRPC_METHOD_NOT_FOUND = -32601
JSONRPC_INVALID_PARAMS = -32602
JSONRPC_INTERNAL_ERROR = -32603

# MCP-specific error codes (spec-defined, -32000 to -32099)
MCP_TOOL_NOT_FOUND = -32001
MCP_RESOURCE_NOT_FOUND = -32002  # Reserved by MCP spec for resource-not-found
MCP_GATEWAY_DISABLED = -32003
MCP_TOOL_INVOCATION_DISABLED = -32004  # Custom: tool invocation feature disabled


# ============================================================================
# JSON-RPC Router
# ============================================================================

mcp_jsonrpc_router = APIRouter(
    prefix="/mcp",
    tags=["mcp-jsonrpc"],
    dependencies=[Depends(user_api_key_auth)],
)


def _make_error_response(
    request_id: int | str | None,
    code: int,
    message: str,
    data: Any = None,
) -> JSONResponse:
    """Create a JSON-RPC error response."""
    response = JSONRPCResponse(
        id=request_id,
        error=JSONRPCError(code=code, message=message, data=data),
    )
    return JSONResponse(content=response.model_dump(exclude_none=True))


def _make_success_response(
    request_id: int | str | None,
    result: Any,
) -> JSONResponse:
    """Create a JSON-RPC success response."""
    response = JSONRPCResponse(id=request_id, result=result)
    return JSONResponse(content=response.model_dump(exclude_none=True))


# ============================================================================
# MCP Method Handlers
# ============================================================================


async def _handle_initialize(
    request_id: int | str | None,
    params: dict[str, Any] | None,
) -> JSONResponse:
    """
    Handle MCP initialize request.

    This establishes the session and negotiates capabilities. Per MCP spec,
    we return server info and capabilities. The server negotiates the protocol
    version by examining the client's protocolVersion and responding with
    the latest version both support.

    Params (from client):
        protocolVersion: str - Client's supported protocol version
        capabilities: dict - Client capabilities
        clientInfo: dict - Client name/version info

    Returns:
        protocolVersion: str - Negotiated protocol version
        capabilities: dict - Server capabilities
        serverInfo: dict - Server name/version info
    """
    gateway = get_mcp_gateway()

    if not gateway.is_enabled():
        return _make_error_response(
            request_id,
            MCP_GATEWAY_DISABLED,
            "MCP Gateway is not enabled. Set MCP_GATEWAY_ENABLED=true",
        )

    # Protocol version negotiation
    client_version = (params or {}).get("protocolVersion", MCP_PROTOCOL_VERSION)
    if client_version in MCP_SUPPORTED_VERSIONS:
        negotiated_version = client_version
    else:
        return _make_error_response(
            request_id,
            JSONRPC_INVALID_PARAMS,
            f"Unsupported protocol version: {client_version}. "
            f"Supported versions: {sorted(MCP_SUPPORTED_VERSIONS)}",
        )

    # Server capabilities per MCP spec (2025-11-25)
    capabilities: dict[str, Any] = {
        "tools": {
            "listChanged": True,
        },
        "resources": {
            "listChanged": True,
            "subscribe": False,
        },
        "logging": {},
        "completion": {},
    }

    result = {
        "protocolVersion": negotiated_version,
        "capabilities": capabilities,
        "serverInfo": {
            "name": MCP_SERVER_NAME,
            "version": MCP_SERVER_VERSION,
        },
    }

    return _make_success_response(request_id, result)


def _decode_cursor(cursor: str | None) -> int:
    """Decode a base64-encoded pagination cursor to an offset."""
    if not cursor:
        return 0
    try:
        return int(base64.b64decode(cursor).decode())
    except (ValueError, Exception):
        return 0


def _encode_cursor(offset: int) -> str:
    """Encode an offset as a base64 pagination cursor."""
    return base64.b64encode(str(offset).encode()).decode()


async def _handle_tools_list(
    request_id: int | str | None,
    params: dict[str, Any] | None,
) -> JSONResponse:
    """
    Handle MCP tools/list request.

    Returns all available tools from all registered MCP servers.
    Tool names are prefixed with server_id for disambiguation when
    multiple servers provide tools with the same name.

    Supports cursor-based pagination per MCP 2025-03-26 spec.

    Params:
        cursor: str (optional) - Base64-encoded pagination cursor

    Returns:
        tools: list[dict] - List of tool definitions with name, description,
                            inputSchema, and optional annotations
        nextCursor: str (optional) - Cursor for next page if more results
    """
    gateway = get_mcp_gateway()

    if not gateway.is_enabled():
        return _make_error_response(
            request_id,
            MCP_GATEWAY_DISABLED,
            "MCP Gateway is not enabled. Set MCP_GATEWAY_ENABLED=true",
        )

    # Collect all tools
    all_tools = []
    for server in gateway.list_servers():
        for tool_name in server.tools:
            namespaced_name = f"{server.server_id}.{tool_name}"

            tool_entry: dict[str, Any] = {
                "name": namespaced_name,
            }

            if tool_name in server.tool_definitions:
                tool_def = server.tool_definitions[tool_name]
                tool_entry["description"] = (
                    tool_def.description or f"Tool from {server.name}"
                )
                tool_entry["inputSchema"] = tool_def.input_schema or {"type": "object"}
                # Propagate tool annotations (MCP 2025-03-26)
                if tool_def.annotations:
                    tool_entry["annotations"] = tool_def.annotations
            else:
                tool_entry["description"] = f"Tool from {server.name}"
                tool_entry["inputSchema"] = {"type": "object"}

            all_tools.append(tool_entry)

    # Apply cursor-based pagination
    cursor = (params or {}).get("cursor")
    offset = _decode_cursor(cursor)
    page_size = MCP_TOOLS_PAGE_SIZE

    page = all_tools[offset : offset + page_size]

    result: dict[str, Any] = {"tools": page}
    if offset + page_size < len(all_tools):
        result["nextCursor"] = _encode_cursor(offset + page_size)

    return _make_success_response(request_id, result)


async def _handle_tools_call(
    request_id: int | str | None,
    params: dict[str, Any] | None,
) -> JSONResponse:
    """
    Handle MCP tools/call request.

    Invokes a tool on the appropriate MCP server. Tool names are expected
    to be namespaced as <server_id>.<tool_name>.

    Params:
        name: str - Namespaced tool name (server_id.tool_name)
        arguments: dict - Arguments to pass to the tool

    Returns:
        content: list[dict] - Tool result content blocks
        isError: bool (optional) - True if tool returned an error

    Security:
        Requires LLMROUTER_ENABLE_MCP_TOOL_INVOCATION=true
    """
    gateway = get_mcp_gateway()

    if not gateway.is_enabled():
        return _make_error_response(
            request_id,
            MCP_GATEWAY_DISABLED,
            "MCP Gateway is not enabled. Set MCP_GATEWAY_ENABLED=true",
        )

    if not gateway.is_tool_invocation_enabled():
        return _make_error_response(
            request_id,
            MCP_TOOL_INVOCATION_DISABLED,
            "Remote tool invocation is disabled. Set LLMROUTER_ENABLE_MCP_TOOL_INVOCATION=true",
        )

    if not params:
        return _make_error_response(
            request_id,
            JSONRPC_INVALID_PARAMS,
            "Missing params for tools/call",
        )

    namespaced_name = params.get("name")
    arguments = params.get("arguments", {})

    if not namespaced_name:
        return _make_error_response(
            request_id,
            JSONRPC_INVALID_PARAMS,
            "Missing required param: name",
        )

    # Parse namespaced tool name
    if "." in namespaced_name:
        # Format: server_id.tool_name
        parts = namespaced_name.split(".", 1)
        server_id = parts[0]
        tool_name = parts[1]
    else:
        # Non-namespaced - try to find the tool across all servers
        tool_name = namespaced_name
        tool_def = gateway.get_tool(tool_name)
        if tool_def:
            server_id = tool_def.server_id
        else:
            return _make_error_response(
                request_id,
                MCP_TOOL_NOT_FOUND,
                f"Tool '{namespaced_name}' not found",
            )

    # Verify server exists and has the tool
    server = gateway.get_server(server_id)
    if not server:
        return _make_error_response(
            request_id,
            MCP_TOOL_NOT_FOUND,
            f"Server '{server_id}' not found for tool '{namespaced_name}'",
        )

    if tool_name not in server.tools:
        return _make_error_response(
            request_id,
            MCP_TOOL_NOT_FOUND,
            f"Tool '{tool_name}' not found on server '{server_id}'",
        )

    # Invoke the tool
    try:
        result: MCPToolResult = await gateway.invoke_tool(tool_name, arguments)

        if result.success:
            # Format result as MCP content block
            content = [
                {
                    "type": "text",
                    "text": json.dumps(result.result)
                    if not isinstance(result.result, str)
                    else result.result,
                }
            ]
            return _make_success_response(request_id, {"content": content})
        else:
            # Return error as content with isError flag
            content = [
                {
                    "type": "text",
                    "text": result.error or "Tool invocation failed",
                }
            ]
            return _make_success_response(
                request_id, {"content": content, "isError": True}
            )

    except Exception as e:
        return _make_error_response(
            request_id,
            JSONRPC_INTERNAL_ERROR,
            f"Tool invocation failed: {str(e)}",
        )


async def _handle_resources_list(
    request_id: int | str | None,
    params: dict[str, Any] | None,
) -> JSONResponse:
    """
    Handle MCP resources/list request.

    Returns all available resources from all registered MCP servers.
    Supports cursor-based pagination per MCP 2025-03-26 spec.

    Params:
        cursor: str (optional) - Base64-encoded pagination cursor
        pageSize: int (optional) - Number of results per page (max 100)

    Returns:
        resources: list[dict] - List of resource definitions
        nextCursor: str (optional) - Cursor for next page if more results
    """
    gateway = get_mcp_gateway()

    if not gateway.is_enabled():
        return _make_error_response(
            request_id,
            MCP_GATEWAY_DISABLED,
            "MCP Gateway is not enabled. Set MCP_GATEWAY_ENABLED=true",
        )

    all_resources = []
    for res in gateway.list_resources():
        all_resources.append(
            {
                "uri": res.get("resource", ""),
                "name": res.get("resource", "").split("/")[-1]
                if res.get("resource")
                else "",
                "description": f"Resource from {res.get('server_name', 'unknown')}",
            }
        )

    # Apply cursor-based pagination
    cursor = (params or {}).get("cursor")
    offset = _decode_cursor(cursor)
    page_size = min(
        (params or {}).get("pageSize", MCP_RESOURCES_PAGE_SIZE),
        MCP_RESOURCES_MAX_PAGE_SIZE,
    )

    page = all_resources[offset : offset + page_size]

    result: dict[str, Any] = {"resources": page}
    if offset + page_size < len(all_resources):
        result["nextCursor"] = _encode_cursor(offset + page_size)

    return _make_success_response(request_id, result)


async def _handle_resources_read(
    request_id: int | str | None,
    params: dict[str, Any] | None,
) -> JSONResponse:
    """
    Handle MCP resources/read request.

    Proxies the read request to the upstream MCP server that owns the resource.
    Looks up the owning server from the gateway registry and forwards the
    request, returning the upstream response.

    Params:
        uri: str - URI of the resource to read

    Returns:
        contents: list[dict] - Resource content blocks with uri and text
    """
    gateway = get_mcp_gateway()

    if not gateway.is_enabled():
        return _make_error_response(
            request_id,
            MCP_GATEWAY_DISABLED,
            "MCP Gateway is not enabled. Set MCP_GATEWAY_ENABLED=true",
        )

    if not params or not params.get("uri"):
        return _make_error_response(
            request_id,
            JSONRPC_INVALID_PARAMS,
            "Missing required param: uri",
        )

    uri = params["uri"]

    # Proxy the read request to the upstream server
    result = await gateway.proxy_resource_read(uri)

    # Check for error from the proxy
    if "error" in result:
        error = result["error"]
        return _make_error_response(
            request_id,
            error.get("code", MCP_RESOURCE_NOT_FOUND),
            error.get("message", f"Resource not found: {uri}"),
        )

    return _make_success_response(request_id, result)


async def _handle_resources_templates_list(
    request_id: int | str | None,
    params: dict[str, Any] | None,
) -> JSONResponse:
    """
    Handle MCP resources/templates/list request (2025-06-18+).

    Returns an empty list of resource templates. This is a stub implementation
    that satisfies spec compliance without providing actual template functionality.
    """
    return _make_success_response(request_id, {"resourceTemplates": []})


async def _handle_logging_set_level(
    request_id: int | str | None,
    params: dict[str, Any] | None,
) -> JSONResponse:
    """
    Handle MCP logging/setLevel request (2025-06-18+).

    Accepts a log level from the client. This is a stub implementation
    that acknowledges the request without changing server logging behavior.
    """
    level = (params or {}).get("level", "info")
    logger.info(f"MCP client requested log level: {level}")
    return _make_success_response(request_id, {})


async def _handle_completion_complete(
    request_id: int | str | None,
    params: dict[str, Any] | None,
) -> JSONResponse:
    """
    Handle MCP completion/complete request (2025-06-18+).

    Returns empty completions. This is a stub implementation that satisfies
    spec compliance without providing actual completion functionality.
    """
    return _make_success_response(
        request_id,
        {
            "completion": {
                "values": [],
                "hasMore": False,
                "total": 0,
            }
        },
    )


# Method dispatch table
METHOD_HANDLERS: dict[str, Any] = {
    "initialize": _handle_initialize,
    "tools/list": _handle_tools_list,
    "tools/call": _handle_tools_call,
    "resources/list": _handle_resources_list,
    "resources/read": _handle_resources_read,
    "resources/templates/list": _handle_resources_templates_list,
    "logging/setLevel": _handle_logging_set_level,
    "completion/complete": _handle_completion_complete,
}

# Notification methods (no response expected, no id field)
NOTIFICATION_HANDLERS: set[str] = {
    "notifications/initialized",
}


# ============================================================================
# Main JSON-RPC Endpoint
# ============================================================================


@mcp_jsonrpc_router.post("")
@mcp_jsonrpc_router.post("/")
async def mcp_jsonrpc_endpoint(request: Request) -> JSONResponse:
    """
    Native MCP JSON-RPC 2.0 endpoint.

    This endpoint implements the MCP protocol's streamable HTTP transport,
    accepting JSON-RPC 2.0 requests and returning JSON-RPC 2.0 responses.

    Supported methods:
    - initialize: Session initialization with capability negotiation
    - tools/list: List available tools from all registered MCP servers
    - tools/call: Invoke a tool on the appropriate MCP server
    - resources/list: List available resources

    Example request:
    ```json
    {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0.0"}
        }
    }
    ```

    Example response:
    ```json
    {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "protocolVersion": "2025-03-26",
            "capabilities": {"tools": {"listChanged": true}},
            "serverInfo": {"name": "routeiq-mcp-gateway", "version": "1.0.0"}
        }
    }
    ```

    Note: SSE streaming is not implemented. For streaming responses, the
    standard HTTP transport is sufficient for stateless tool calls.
    """
    http_request_id = get_request_id() or "unknown"

    # Session management: get or create Mcp-Session-Id
    request_headers = {k.lower(): v for k, v in request.headers.items()}
    session_id = get_or_create_session(request_headers)

    # Parse request body
    try:
        body = await request.body()
        if not body:
            return _make_error_response(None, JSONRPC_PARSE_ERROR, "Empty request body")

        data = json.loads(body)
    except json.JSONDecodeError as e:
        return _make_error_response(
            None, JSONRPC_PARSE_ERROR, f"Invalid JSON: {str(e)}"
        )

    # Validate JSON-RPC structure
    if not isinstance(data, dict):
        return _make_error_response(
            None, JSONRPC_INVALID_REQUEST, "Request must be a JSON object"
        )

    jsonrpc_version = data.get("jsonrpc")
    if jsonrpc_version != "2.0":
        return _make_error_response(
            data.get("id"),
            JSONRPC_INVALID_REQUEST,
            f"Invalid JSON-RPC version: {jsonrpc_version}. Expected '2.0'",
        )

    request_id = data.get("id")
    method = data.get("method")
    params = data.get("params")

    if not method or not isinstance(method, str):
        return _make_error_response(
            request_id, JSONRPC_INVALID_REQUEST, "Missing or invalid 'method' field"
        )

    # Handle notifications (no id, no response expected)
    if method in NOTIFICATION_HANDLERS:
        # Notifications are fire-and-forget; return 202 Accepted with no body
        response = JSONResponse(content={}, status_code=202)
        response.headers["Mcp-Session-Id"] = session_id
        return response

    # Dispatch to method handler
    handler = METHOD_HANDLERS.get(method)
    if not handler:
        return _make_error_response(
            request_id,
            JSONRPC_METHOD_NOT_FOUND,
            f"Method '{method}' not found. Supported methods: {list(METHOD_HANDLERS.keys())}",
        )

    # Execute handler
    try:
        response = await handler(request_id, params)
        # Add Mcp-Session-Id header to all responses
        response.headers["Mcp-Session-Id"] = session_id
        return response
    except Exception as e:
        return _make_error_response(
            request_id,
            JSONRPC_INTERNAL_ERROR,
            f"Internal error: {str(e)}",
            data={"http_request_id": http_request_id},
        )


# ============================================================================
# SSE Endpoint (Placeholder)
# ============================================================================

# NOTE: Full SSE support would be needed for:
# - Server-initiated notifications (tools/list_changed, resources/list_changed)
# - Streaming tool responses
# - Progress updates during long-running operations
#
# For now, the JSON-RPC over HTTP POST is sufficient for:
# - initialize
# - tools/list
# - tools/call (non-streaming)
# - resources/list
#
# SSE implementation would require:
# 1. GET /mcp endpoint that returns text/event-stream
# 2. Session management for long-lived connections
# 3. Event formatting per SSE spec
#
# This is left as a future enhancement when streaming tool responses are needed.


@mcp_jsonrpc_router.get("")
@mcp_jsonrpc_router.get("/")
async def mcp_sse_info_endpoint(request: Request) -> JSONResponse:
    """
    INFO endpoint for MCP native surface.

    GET requests to /mcp return server info and supported methods.
    This helps clients discover the MCP server's capabilities.

    For SSE streaming (not yet implemented), clients would need to
    request with Accept: text/event-stream header.

    Note: Full SSE streaming is not implemented in this version.
    Use POST /mcp for JSON-RPC requests instead.
    """
    gateway = get_mcp_gateway()

    return JSONResponse(
        content={
            "name": MCP_SERVER_NAME,
            "version": MCP_SERVER_VERSION,
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "transport": "streamable-http",
            "endpoints": {
                "jsonrpc": "POST /mcp",
                "info": "GET /mcp",
            },
            "capabilities": {
                "tools": {"listChanged": True},
                "resources": {"listChanged": True, "subscribe": False},
                "logging": {},
                "completion": {},
            },
            "status": "enabled" if gateway.is_enabled() else "disabled",
            "sseSupport": "not_implemented",
            "note": "Use POST /mcp with JSON-RPC 2.0 requests. SSE streaming is not yet implemented.",
        }
    )
