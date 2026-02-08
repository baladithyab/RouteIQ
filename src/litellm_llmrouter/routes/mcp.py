"""
MCP Gateway REST Endpoints.

These REST endpoints are prefixed with /llmrouter/mcp to avoid conflicts
with LiteLLM's native /mcp endpoint (which uses JSON-RPC over SSE).

Includes:
- Server CRUD (/llmrouter/mcp/servers/*)
- Tool CRUD (/llmrouter/mcp/tools/*)
- Server health (/v1/llmrouter/mcp/server/health)
- Registry (/v1/llmrouter/mcp/registry.json)
- Access groups (/v1/llmrouter/mcp/access_groups)
"""

from fastapi import Depends, HTTPException, Query

from ..auth import get_request_id, sanitize_error_response
from ..rbac import (
    requires_permission,
    PERMISSION_MCP_SERVER_WRITE,
    PERMISSION_MCP_TOOL_WRITE,
    PERMISSION_MCP_TOOL_CALL,
)
from ..audit import audit_log, AuditAction, AuditOutcome, AuditWriteError
from ..mcp_gateway import MCPServer, MCPTransport, MCPToolDefinition, get_mcp_gateway
from .models import ServerRegistration, MCPToolCall, MCPToolRegister
from . import admin_router, llmrouter_router


async def _handle_audit_write(
    action: AuditAction,
    resource_type: str,
    resource_id: str | None,
    outcome: AuditOutcome,
    rbac_info: dict | None,
    request_id: str,
    outcome_reason: str | None = None,
):
    """
    Handle audit write with fail-closed mode support.

    If fail-closed mode is enabled and audit write fails, raises 503.
    Otherwise, failure is logged and the request continues.
    """
    try:
        await audit_log(
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            outcome=outcome,
            outcome_reason=outcome_reason,
            actor_info=rbac_info,
        )
    except AuditWriteError:
        # Fail-closed: reject the request with 503
        raise HTTPException(
            status_code=503,
            detail={
                "error": "audit_log_unavailable",
                "message": "Cannot process request: audit logging is unavailable and fail-closed mode is enabled",
                "request_id": request_id,
            },
        )


# =============================================================================
# MCP Gateway Endpoints
# =============================================================================


# Read-only endpoints - user auth sufficient
@llmrouter_router.get("/llmrouter/mcp/servers")
async def list_mcp_servers():
    """List all registered MCP servers (REST API)."""
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

    return {
        "servers": [
            {
                "server_id": s.server_id,
                "name": s.name,
                "url": s.url,
                "transport": s.transport.value,
                "tools": s.tools,
                "resources": s.resources,
            }
            for s in gateway.list_servers()
        ]
    }


# Write operations - admin auth required + RBAC
@admin_router.post("/llmrouter/mcp/servers")
async def register_mcp_server(
    server: ServerRegistration,
    rbac_info: dict = Depends(requires_permission(PERMISSION_MCP_SERVER_WRITE)),
):
    """
    Register a new MCP server (REST API).

    Requires admin API key authentication or user with mcp.server.write permission.

    Security: Server URLs are validated against SSRF attacks. Private IPs are
    blocked by default. Configure LLMROUTER_SSRF_ALLOWLIST_HOSTS or
    LLMROUTER_SSRF_ALLOWLIST_CIDRS to allow specific endpoints.
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

    try:
        transport = MCPTransport(server.transport)
        mcp_server = MCPServer(
            server_id=server.server_id,
            name=server.name,
            url=server.url,
            transport=transport,
            tools=server.tools,
            resources=server.resources,
            auth_type=server.auth_type,
            metadata=server.metadata,
        )
        gateway.register_server(mcp_server)

        # Audit log the success
        await _handle_audit_write(
            AuditAction.MCP_SERVER_CREATE,
            "mcp_server",
            server.server_id,
            AuditOutcome.SUCCESS,
            rbac_info,
            request_id,
        )
        return {"status": "registered", "server_id": server.server_id}
    except ValueError as e:
        # SSRF validation or other URL validation errors
        error_msg = str(e)
        if "blocked for security reasons" in error_msg or "SSRF" in error_msg:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "ssrf_blocked",
                    "message": error_msg,
                    "request_id": request_id,
                },
            )
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_request",
                "message": error_msg,
                "request_id": request_id,
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        err = sanitize_error_response(e, request_id, "Failed to register MCP server")
        raise HTTPException(status_code=500, detail=err)


# Read-only - user auth
@llmrouter_router.get("/llmrouter/mcp/servers/{server_id}")
async def get_mcp_server(server_id: str):
    """Get a specific MCP server by ID (REST API)."""
    request_id = get_request_id() or "unknown"
    gateway = get_mcp_gateway()
    server = gateway.get_server(server_id)
    if not server:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "server_not_found",
                "message": f"Server {server_id} not found",
                "request_id": request_id,
            },
        )

    return {
        "server_id": server.server_id,
        "name": server.name,
        "url": server.url,
        "transport": server.transport.value,
        "tools": server.tools,
        "resources": server.resources,
        "auth_type": server.auth_type,
        "metadata": server.metadata,
    }


# Write operation - admin auth + RBAC
@admin_router.delete("/llmrouter/mcp/servers/{server_id}")
async def unregister_mcp_server(
    server_id: str,
    rbac_info: dict = Depends(requires_permission(PERMISSION_MCP_SERVER_WRITE)),
):
    """
    Unregister an MCP server (REST API).

    Requires admin API key authentication or user with mcp.server.write permission.
    """
    request_id = get_request_id() or "unknown"
    gateway = get_mcp_gateway()
    if gateway.unregister_server(server_id):
        # Audit log the success
        await _handle_audit_write(
            AuditAction.MCP_SERVER_DELETE,
            "mcp_server",
            server_id,
            AuditOutcome.SUCCESS,
            rbac_info,
            request_id,
        )
        return {"status": "unregistered", "server_id": server_id}
    raise HTTPException(
        status_code=404,
        detail={
            "error": "server_not_found",
            "message": f"Server {server_id} not found",
            "request_id": request_id,
        },
    )


# Write operation - admin auth + RBAC
@admin_router.put("/llmrouter/mcp/servers/{server_id}")
async def update_mcp_server(
    server_id: str,
    server: ServerRegistration,
    rbac_info: dict = Depends(requires_permission(PERMISSION_MCP_SERVER_WRITE)),
):
    """
    Update an MCP server (full update).

    Replaces all server fields with the provided values.
    Tools and resources are refreshed on update.

    Requires admin API key authentication or user with mcp.server.write permission.

    Security: Server URLs are validated against SSRF attacks. Private IPs are
    blocked by default. Configure LLMROUTER_SSRF_ALLOWLIST_HOSTS or
    LLMROUTER_SSRF_ALLOWLIST_CIDRS to allow specific endpoints.
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

    existing = gateway.get_server(server_id)
    if not existing:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "server_not_found",
                "message": f"Server {server_id} not found",
                "request_id": request_id,
            },
        )

    try:
        # Unregister old server to clean up tool mappings
        gateway.unregister_server(server_id)

        # Validate URL (SSRF guard)
        if not server.url.startswith(("http://", "https://")):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_url",
                    "message": f"Server URL '{server.url}' must start with http:// or https://",
                },
            )

        # Register updated server
        transport = MCPTransport(server.transport)
        mcp_server = MCPServer(
            server_id=server_id,
            name=server.name,
            url=server.url,
            transport=transport,
            tools=server.tools,
            resources=server.resources,
            auth_type=server.auth_type,
            metadata=server.metadata,
        )
        gateway.register_server(mcp_server)

        # Audit log the success
        await _handle_audit_write(
            AuditAction.MCP_SERVER_UPDATE,
            "mcp_server",
            server_id,
            AuditOutcome.SUCCESS,
            rbac_info,
            request_id,
        )

        return {
            "status": "updated",
            "server_id": server_id,
            "server": {
                "server_id": mcp_server.server_id,
                "name": mcp_server.name,
                "url": mcp_server.url,
                "transport": mcp_server.transport.value,
                "tools": mcp_server.tools,
                "resources": mcp_server.resources,
            },
        }
    except ValueError as e:
        # SSRF validation or other URL validation errors
        error_msg = str(e)
        if "blocked for security reasons" in error_msg or "SSRF" in error_msg:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "ssrf_blocked",
                    "message": error_msg,
                    "request_id": request_id,
                },
            )
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_request",
                "message": error_msg,
                "request_id": request_id,
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        err = sanitize_error_response(e, request_id, "Failed to update MCP server")
        raise HTTPException(status_code=500, detail=err)


# Read-only - user auth
@llmrouter_router.get("/llmrouter/mcp/tools")
async def list_mcp_tools():
    """List all available MCP tools across all servers."""
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

    return {"tools": gateway.list_tools()}


# Read-only - user auth
@llmrouter_router.get("/llmrouter/mcp/resources")
async def list_mcp_resources():
    """List all available MCP resources across all servers."""
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

    return {"resources": gateway.list_resources()}


# Read-only - user auth
@llmrouter_router.get("/llmrouter/mcp/tools/list")
async def list_mcp_tools_detailed():
    """
    List all available MCP tools with detailed information.

    Returns tool definitions including input schemas when available.
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

    tools = []
    for server in gateway.list_servers():
        for tool_name in server.tools:
            tool_info = {
                "name": tool_name,
                "server_id": server.server_id,
                "server_name": server.name,
            }
            # Add detailed definition if available
            if tool_name in server.tool_definitions:
                tool_def = server.tool_definitions[tool_name]
                tool_info["description"] = tool_def.description
                tool_info["input_schema"] = tool_def.input_schema
            tools.append(tool_info)

    return {"tools": tools, "count": len(tools)}


# Tool invocation - admin auth + RBAC (modifies state on external MCP servers)
@admin_router.post("/llmrouter/mcp/tools/call")
async def call_mcp_tool(
    request: MCPToolCall,
    rbac_info: dict = Depends(requires_permission(PERMISSION_MCP_TOOL_CALL)),
):
    """
    Invoke an MCP tool by name.

    The tool is looked up across all registered servers and invoked
    with the provided arguments. Arguments are validated against the
    tool's input schema if available.

    **Security Note**: Remote tool invocation is DISABLED by default.
    Enable via `LLMROUTER_ENABLE_MCP_TOOL_INVOCATION=true` environment variable.
    When disabled, this endpoint returns HTTP 501 (Not Implemented).

    Requires admin API key authentication.

    Request body:
    ```json
    {
        "tool_name": "create_issue",
        "arguments": {
            "title": "Bug report",
            "body": "Description of the bug"
        }
    }
    ```

    Response (when enabled and successful):
    ```json
    {
        "status": "success",
        "tool_name": "create_issue",
        "server_id": "github-mcp",
        "result": {...}
    }
    ```

    Response (when disabled - 501):
    ```json
    {
        "error": "not_implemented",
        "message": "Remote tool invocation is disabled. Enable via LLMROUTER_ENABLE_MCP_TOOL_INVOCATION=true",
        "request_id": "..."
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

    # Check if tool invocation is enabled (disabled by default for security)
    if not gateway.is_tool_invocation_enabled():
        raise HTTPException(
            status_code=501,
            detail={
                "error": "tool_invocation_disabled",
                "message": "Remote tool invocation is disabled. Enable via LLMROUTER_ENABLE_MCP_TOOL_INVOCATION=true",
                "request_id": request_id,
            },
        )

    # Find the tool
    tool = gateway.get_tool(request.tool_name)
    if not tool:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "tool_not_found",
                "message": f"Tool '{request.tool_name}' not found",
                "request_id": request_id,
            },
        )

    try:
        # Invoke the tool
        result = await gateway.invoke_tool(request.tool_name, request.arguments)

        if not result.success:
            # Check for specific error codes in the error message
            error_msg = result.error or "Tool invocation failed"
            if error_msg.startswith("tool_invocation_disabled:"):
                raise HTTPException(
                    status_code=501,
                    detail={
                        "error": "tool_invocation_disabled",
                        "message": error_msg.split(":", 1)[1].strip(),
                        "request_id": request_id,
                    },
                )
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "tool_invocation_failed",
                    "message": error_msg,
                    "request_id": request_id,
                },
            )

        # Audit log the success
        await _handle_audit_write(
            AuditAction.MCP_TOOL_CALL,
            "mcp_tool",
            request.tool_name,
            AuditOutcome.SUCCESS,
            rbac_info,
            request_id,
        )

        return {
            "status": "success",
            "tool_name": result.tool_name,
            "server_id": result.server_id,
            "result": result.result,
        }
    except HTTPException:
        raise
    except Exception as e:
        err = sanitize_error_response(e, request_id, "Failed to invoke MCP tool")
        raise HTTPException(status_code=500, detail=err)


# Read-only - user auth
@llmrouter_router.get("/llmrouter/mcp/tools/{tool_name}")
async def get_mcp_tool(tool_name: str):
    """Get details about a specific MCP tool."""
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

    tool = gateway.get_tool(tool_name)
    if not tool:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "tool_not_found",
                "message": f"Tool '{tool_name}' not found",
                "request_id": request_id,
            },
        )

    server = gateway.find_server_for_tool(tool_name)
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
        "server_id": tool.server_id,
        "server_name": server.name if server else None,
    }


# Tool registration - admin auth + RBAC
@admin_router.post("/llmrouter/mcp/servers/{server_id}/tools")
async def register_mcp_tool(
    server_id: str,
    tool: MCPToolRegister,
    rbac_info: dict = Depends(requires_permission(PERMISSION_MCP_TOOL_WRITE)),
):
    """
    Register a tool definition for an MCP server.

    This allows adding detailed tool definitions with input schemas
    to an existing server registration.

    Requires admin API key authentication or user with mcp.tool.write permission.
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

    server = gateway.get_server(server_id)
    if not server:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "server_not_found",
                "message": f"Server {server_id} not found",
                "request_id": request_id,
            },
        )

    try:
        tool_def = MCPToolDefinition(
            name=tool.name,
            description=tool.description,
            input_schema=tool.input_schema,
            server_id=server_id,
        )

        if gateway.register_tool_definition(server_id, tool_def):
            # Audit log the success
            await _handle_audit_write(
                AuditAction.MCP_TOOL_REGISTER,
                "mcp_tool",
                tool.name,
                AuditOutcome.SUCCESS,
                rbac_info,
                request_id,
            )
            return {
                "status": "registered",
                "tool_name": tool.name,
                "server_id": server_id,
            }
        else:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "tool_registration_failed",
                    "message": f"Failed to register tool '{tool.name}'",
                    "request_id": request_id,
                },
            )
    except HTTPException:
        raise
    except Exception as e:
        err = sanitize_error_response(e, request_id, "Failed to register MCP tool")
        raise HTTPException(status_code=500, detail=err)


# Read-only endpoints - user auth
@llmrouter_router.get("/v1/llmrouter/mcp/server/health")
async def get_mcp_servers_health():
    """
    Check the health of all registered MCP servers.

    Returns connectivity status and latency metrics for each server.
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

    try:
        health_results = await gateway.check_all_servers_health()

        healthy_count = sum(1 for h in health_results if h.get("status") == "healthy")
        unhealthy_count = len(health_results) - healthy_count

        return {
            "servers": health_results,
            "summary": {
                "total": len(health_results),
                "healthy": healthy_count,
                "unhealthy": unhealthy_count,
            },
        }
    except Exception as e:
        err = sanitize_error_response(
            e, request_id, "Failed to check MCP server health"
        )
        raise HTTPException(status_code=500, detail=err)


# Read-only endpoints - user auth
@llmrouter_router.get("/v1/llmrouter/mcp/server/{server_id}/health")
async def get_mcp_server_health(server_id: str):
    """
    Check the health of a specific MCP server.

    Returns connectivity status and latency metrics.
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

    try:
        health = await gateway.check_server_health(server_id)

        if health.get("status") == "not_found":
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "server_not_found",
                    "message": f"Server '{server_id}' not found",
                    "request_id": request_id,
                },
            )

        return health
    except HTTPException:
        raise
    except Exception as e:
        err = sanitize_error_response(
            e, request_id, "Failed to check MCP server health"
        )
        raise HTTPException(status_code=500, detail=err)


# Read-only endpoints - user auth
@llmrouter_router.get("/v1/llmrouter/mcp/registry.json")
async def get_mcp_registry(
    access_groups: str | None = Query(
        None, description="Comma-separated access groups"
    ),
):
    """
    Get the MCP registry document for discovery.

    Returns a registry document listing all servers and their capabilities.
    Optionally filter by access groups.
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

    groups = None
    if access_groups:
        groups = [g.strip() for g in access_groups.split(",")]

    registry = gateway.get_registry(access_groups=groups)
    return registry


# Read-only endpoints - user auth
@llmrouter_router.get("/v1/llmrouter/mcp/access_groups")
async def list_mcp_access_groups():
    """
    List all access groups across all MCP servers.

    Returns a list of unique access group names that can be used
    to filter server visibility.
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

    groups = gateway.list_access_groups()
    return {
        "access_groups": groups,
        "count": len(groups),
    }
