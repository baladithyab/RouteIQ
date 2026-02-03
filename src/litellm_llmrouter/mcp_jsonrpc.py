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

import json
import os
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

from .auth import admin_api_key_auth, get_request_id
from .mcp_gateway import get_mcp_gateway, MCPToolResult

# Protocol version (MCP spec version we implement)
MCP_PROTOCOL_VERSION = "2024-11-05"

# Server info
MCP_SERVER_NAME = os.getenv("MCP_SERVER_NAME", "routeiq-mcp-gateway")
MCP_SERVER_VERSION = os.getenv("MCP_SERVER_VERSION", "1.0.0")


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
MCP_TOOL_INVOCATION_DISABLED = -32002
MCP_GATEWAY_DISABLED = -32003


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
    we return server info and capabilities. Clients should check protocolVersion
    compatibility.

    Params (from client):
        protocolVersion: str - Client's supported protocol version
        capabilities: dict - Client capabilities
        clientInfo: dict - Client name/version info

    Returns:
        protocolVersion: str - Server's protocol version
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

    # Server capabilities per MCP spec
    capabilities = {
        "tools": {
            "listChanged": True,  # We can notify when tool list changes (future)
        },
        "resources": {
            "listChanged": True,
            "subscribe": False,  # Resource subscriptions not implemented
        },
    }

    result = {
        "protocolVersion": MCP_PROTOCOL_VERSION,
        "capabilities": capabilities,
        "serverInfo": {
            "name": MCP_SERVER_NAME,
            "version": MCP_SERVER_VERSION,
        },
    }

    return _make_success_response(request_id, result)


async def _handle_tools_list(
    request_id: int | str | None,
    params: dict[str, Any] | None,
) -> JSONResponse:
    """
    Handle MCP tools/list request.

    Returns all available tools from all registered MCP servers.
    Tool names are prefixed with server_id for disambiguation when
    multiple servers provide tools with the same name.

    Params:
        cursor: str (optional) - Pagination cursor (not implemented)

    Returns:
        tools: list[dict] - List of tool definitions with name, description, inputSchema
        nextCursor: str (optional) - Pagination cursor (not implemented)
    """
    gateway = get_mcp_gateway()

    if not gateway.is_enabled():
        return _make_error_response(
            request_id,
            MCP_GATEWAY_DISABLED,
            "MCP Gateway is not enabled. Set MCP_GATEWAY_ENABLED=true",
        )

    tools = []
    for server in gateway.list_servers():
        for tool_name in server.tools:
            # Namespace tool name with server_id
            namespaced_name = f"{server.server_id}.{tool_name}"

            tool_entry = {
                "name": namespaced_name,
            }

            # Add detailed info if available from tool definitions
            if tool_name in server.tool_definitions:
                tool_def = server.tool_definitions[tool_name]
                tool_entry["description"] = tool_def.description or f"Tool from {server.name}"
                tool_entry["inputSchema"] = tool_def.input_schema or {"type": "object"}
            else:
                # Basic entry for tools without detailed definitions
                tool_entry["description"] = f"Tool from {server.name}"
                tool_entry["inputSchema"] = {"type": "object"}

            tools.append(tool_entry)

    return _make_success_response(request_id, {"tools": tools})


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
                    "text": json.dumps(result.result) if not isinstance(result.result, str) else result.result,
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
            return _make_success_response(request_id, {"content": content, "isError": True})

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

    Params:
        cursor: str (optional) - Pagination cursor (not implemented)

    Returns:
        resources: list[dict] - List of resource definitions
        nextCursor: str (optional) - Pagination cursor (not implemented)
    """
    gateway = get_mcp_gateway()

    if not gateway.is_enabled():
        return _make_error_response(
            request_id,
            MCP_GATEWAY_DISABLED,
            "MCP Gateway is not enabled. Set MCP_GATEWAY_ENABLED=true",
        )

    resources = []
    for res in gateway.list_resources():
        resources.append(
            {
                "uri": res.get("resource", ""),
                "name": res.get("resource", "").split("/")[-1] if res.get("resource") else "",
                "description": f"Resource from {res.get('server_name', 'unknown')}",
            }
        )

    return _make_success_response(request_id, {"resources": resources})


# Method dispatch table
METHOD_HANDLERS = {
    "initialize": _handle_initialize,
    "tools/list": _handle_tools_list,
    "tools/call": _handle_tools_call,
    "resources/list": _handle_resources_list,
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
            "protocolVersion": "2024-11-05",
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
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {"listChanged": true}},
            "serverInfo": {"name": "routeiq-mcp-gateway", "version": "1.0.0"}
        }
    }
    ```

    Note: SSE streaming is not implemented. For streaming responses, the
    standard HTTP transport is sufficient for stateless tool calls.
    """
    http_request_id = get_request_id() or "unknown"

    # Parse request body
    try:
        body = await request.body()
        if not body:
            return _make_error_response(None, JSONRPC_PARSE_ERROR, "Empty request body")

        data = json.loads(body)
    except json.JSONDecodeError as e:
        return _make_error_response(None, JSONRPC_PARSE_ERROR, f"Invalid JSON: {str(e)}")

    # Validate JSON-RPC structure
    if not isinstance(data, dict):
        return _make_error_response(None, JSONRPC_INVALID_REQUEST, "Request must be a JSON object")

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
        return _make_error_response(request_id, JSONRPC_INVALID_REQUEST, "Missing or invalid 'method' field")

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
        return await handler(request_id, params)
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
            },
            "status": "enabled" if gateway.is_enabled() else "disabled",
            "sseSupport": "not_implemented",
            "note": "Use POST /mcp with JSON-RPC 2.0 requests. SSE streaming is not yet implemented.",
        }
    )
