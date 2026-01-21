"""
FastAPI Routes for A2A Gateway, MCP Gateway, and Hot Reload
============================================================

These routes extend the LiteLLM proxy server with:
- A2A (Agent-to-Agent) convenience endpoints (/a2a/agents)
  Note: Main A2A functionality is provided by LiteLLM's built-in endpoints:
  - POST /v1/agents - Create agent (DB-backed)
  - GET /v1/agents - List agents (DB-backed)
  - DELETE /v1/agents/{agent_id} - Delete agent (DB-backed)
  - POST /a2a/{agent_id} - Invoke agent (A2A JSON-RPC protocol)
  - POST /a2a/{agent_id}/message/stream - Streaming alias (proxies to canonical)
- MCP (Model Context Protocol) gateway endpoints
- Hot reload and config sync endpoints

Usage:
    from litellm_llmrouter.routes import router
    app.include_router(router)
"""

import os
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .mcp_gateway import MCPServer, MCPTransport, MCPToolDefinition, get_mcp_gateway
from .hot_reload import get_hot_reload_manager
from .config_sync import get_sync_manager

# Main router for all LLMRouter routes
router = APIRouter(tags=["llmrouter"])


# =============================================================================
# Pydantic Models
# =============================================================================


class AgentRegistration(BaseModel):
    """Request model for A2A agent registration (compatibility layer)."""

    agent_name: str
    description: str = ""
    url: str
    capabilities: list[str] = []
    agent_card_params: dict[str, Any] = {}
    litellm_params: dict[str, Any] = {}


class ServerRegistration(BaseModel):
    """Request model for MCP server registration."""

    server_id: str
    name: str
    url: str
    transport: str = "streamable_http"
    tools: list[str] = []
    resources: list[str] = []
    auth_type: str = "none"
    metadata: dict[str, Any] = {}


class ReloadRequest(BaseModel):
    """Request model for reload operations."""

    strategy: str | None = None
    force_sync: bool = False


class MCPToolCall(BaseModel):
    """Request model for MCP tool invocation."""

    tool_name: str
    arguments: dict[str, Any] = {}


class MCPToolRegister(BaseModel):
    """Request model for registering an MCP tool definition."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = {}


# =============================================================================
# A2A Gateway Convenience Endpoints (/a2a/agents)
# =============================================================================
# These are thin wrappers around LiteLLM's global_agent_registry for convenience.
# The main A2A functionality is provided by LiteLLM's built-in endpoints:
# - POST /v1/agents - Create agent (DB-backed)
# - GET /v1/agents - List agents (DB-backed)
# - DELETE /v1/agents/{agent_id} - Delete agent (DB-backed)
# - POST /a2a/{agent_id} - Invoke agent (A2A JSON-RPC protocol)


@router.get("/a2a/agents")
async def list_a2a_agents_convenience():
    """
    List all registered A2A agents.

    This is a convenience endpoint that wraps LiteLLM's global_agent_registry.
    For full functionality, use GET /v1/agents (DB-backed, supports filtering).

    Returns agents from LiteLLM's in-memory registry (synced from DB+config).
    """
    try:
        from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry

        agents = global_agent_registry.get_agent_list()
        return {
            "agents": [
                {
                    "agent_id": a.agent_id,
                    "agent_name": a.agent_name,
                    "description": a.agent_card_params.get("description", "")
                    if a.agent_card_params
                    else "",
                    "url": a.agent_card_params.get("url", "")
                    if a.agent_card_params
                    else "",
                }
                for a in agents
            ]
        }
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="LiteLLM agent registry not available. Ensure LiteLLM is properly initialized.",
        )


@router.post("/a2a/agents")
async def register_a2a_agent_convenience(agent: AgentRegistration):
    """
    Register a new A2A agent.

    This is a convenience endpoint that wraps LiteLLM's global_agent_registry.
    For DB-backed persistence, use POST /v1/agents instead.

    Note: Agents registered via this endpoint are in-memory only and will be
    lost on restart. For HA consistency, use POST /v1/agents which persists
    to the database.
    """
    try:
        from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry
        from litellm.types.agents import AgentResponse
        import hashlib
        import json

        # Create agent config for hashing
        agent_config = {
            "agent_name": agent.agent_name,
            "agent_card_params": agent.agent_card_params
            or {
                "name": agent.agent_name,
                "description": agent.description,
                "url": agent.url,
                "capabilities": {"streaming": "streaming" in agent.capabilities},
            },
            "litellm_params": agent.litellm_params,
        }

        # Generate stable agent_id from config
        agent_id = hashlib.sha256(
            json.dumps(agent_config, sort_keys=True).encode()
        ).hexdigest()

        # Create AgentResponse (LiteLLM's agent type)
        agent_response = AgentResponse(
            agent_id=agent_id,
            agent_name=agent.agent_name,
            agent_card_params=agent_config["agent_card_params"],
            litellm_params=agent.litellm_params,
        )

        # Register with in-memory registry
        global_agent_registry.register_agent(agent_config=agent_response)

        return {
            "status": "registered",
            "agent_id": agent_id,
            "agent_name": agent.agent_name,
            "note": "Agent registered in-memory only. For HA persistence, use POST /v1/agents instead.",
        }
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="LiteLLM agent registry not available. Ensure LiteLLM is properly initialized.",
        )


@router.delete("/a2a/agents/{agent_id}")
async def unregister_a2a_agent_convenience(agent_id: str):
    """
    Unregister an A2A agent.

    This is a convenience endpoint that wraps LiteLLM's global_agent_registry.
    For DB-backed deletion, use DELETE /v1/agents/{agent_id} instead.

    Note: This only removes from in-memory registry. DB-backed agents will
    be re-loaded on restart. Use DELETE /v1/agents/{agent_id} for permanent deletion.
    """
    try:
        from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry

        # Get agent by ID first to find its name (needed for deregister_agent)
        agent = global_agent_registry.get_agent_by_id(agent_id)
        if agent:
            global_agent_registry.deregister_agent(agent_name=agent.agent_name)
            return {
                "status": "unregistered",
                "agent_id": agent_id,
                "note": "Agent removed from in-memory registry. For permanent deletion, use DELETE /v1/agents/{agent_id}",
            }

        # Try by name as fallback
        agent = global_agent_registry.get_agent_by_name(agent_id)
        if agent:
            global_agent_registry.deregister_agent(agent_name=agent_id)
            return {
                "status": "unregistered",
                "agent_name": agent_id,
                "note": "Agent removed from in-memory registry. For permanent deletion, use DELETE /v1/agents/{agent_id}",
            }

        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="LiteLLM agent registry not available. Ensure LiteLLM is properly initialized.",
        )


@router.post("/a2a/{agent_id}/message/stream")
async def a2a_streaming_alias(agent_id: str, request: Request):
    """
    Streaming alias endpoint for A2A JSON-RPC protocol.

    This is an alias that proxies to the canonical POST /a2a/{agent_id} endpoint.
    Use this endpoint when you want an explicit streaming URL for A2A messages.

    The request body should be a valid JSON-RPC message per the A2A protocol.
    The response is streamed back as Server-Sent Events (SSE).

    Example:
    ```bash
    curl -X POST http://localhost:8080/a2a/my-agent/message/stream \\
      -H "Content-Type: application/json" \\
      -H "Authorization: Bearer sk-xxx" \\
      -d '{"jsonrpc": "2.0", "method": "message/send", "id": "1", "params": {...}}'
    ```
    """
    # Read the request body
    body = await request.body()

    # Get upstream port from environment (default 4000 for LiteLLM)
    upstream_port = int(os.environ.get("LITELLM_PORT", "4000"))
    upstream_url = f"http://127.0.0.1:{upstream_port}/a2a/{agent_id}"

    # Forward headers, excluding hop-by-hop headers
    hop_by_hop = {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
    }
    headers = {k: v for k, v in request.headers.items() if k.lower() not in hop_by_hop}

    # Stream the response from upstream
    async def stream_upstream():
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(600.0, connect=60.0)
        ) as client:
            async with client.stream(
                "POST",
                upstream_url,
                content=body,
                headers=headers,
            ) as response:
                if response.status_code >= 400:
                    # For errors, read and yield the full response
                    error_body = await response.aread()
                    yield error_body
                    return

                async for chunk in response.aiter_bytes():
                    yield chunk

    # Determine content type from the request (preserve SSE if requested)
    accept = request.headers.get("accept", "application/json")
    if "text/event-stream" in accept:
        media_type = "text/event-stream"
    else:
        media_type = "application/json"

    return StreamingResponse(
        stream_upstream(),
        media_type=media_type,
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering for SSE
        },
    )


# =============================================================================
# MCP Gateway Endpoints
# =============================================================================
# These REST endpoints are prefixed with /llmrouter/mcp to avoid conflicts
# with LiteLLM's native /mcp endpoint (which uses JSON-RPC over SSE).


@router.get("/llmrouter/mcp/servers")
async def list_mcp_servers():
    """List all registered MCP servers (REST API)."""
    gateway = get_mcp_gateway()
    if not gateway.is_enabled():
        raise HTTPException(status_code=404, detail="MCP Gateway is not enabled")

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


@router.post("/llmrouter/mcp/servers")
async def register_mcp_server(server: ServerRegistration):
    """Register a new MCP server (REST API)."""
    gateway = get_mcp_gateway()
    if not gateway.is_enabled():
        raise HTTPException(status_code=404, detail="MCP Gateway is not enabled")

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
    return {"status": "registered", "server_id": server.server_id}


@router.get("/llmrouter/mcp/servers/{server_id}")
async def get_mcp_server(server_id: str):
    """Get a specific MCP server by ID (REST API)."""
    gateway = get_mcp_gateway()
    server = gateway.get_server(server_id)
    if not server:
        raise HTTPException(status_code=404, detail=f"Server {server_id} not found")

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


@router.delete("/llmrouter/mcp/servers/{server_id}")
async def unregister_mcp_server(server_id: str):
    """Unregister an MCP server (REST API)."""
    gateway = get_mcp_gateway()
    if gateway.unregister_server(server_id):
        return {"status": "unregistered", "server_id": server_id}
    raise HTTPException(status_code=404, detail=f"Server {server_id} not found")


@router.put("/llmrouter/mcp/servers/{server_id}")
async def update_mcp_server(server_id: str, server: ServerRegistration):
    """
    Update an MCP server (full update).

    Replaces all server fields with the provided values.
    Tools and resources are refreshed on update.
    """
    gateway = get_mcp_gateway()
    if not gateway.is_enabled():
        raise HTTPException(status_code=404, detail="MCP Gateway is not enabled")

    existing = gateway.get_server(server_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Server '{server_id}' not found")

    # Unregister old server to clean up tool mappings
    gateway.unregister_server(server_id)

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


@router.get("/llmrouter/mcp/tools")
async def list_mcp_tools():
    """List all available MCP tools across all servers."""
    gateway = get_mcp_gateway()
    if not gateway.is_enabled():
        raise HTTPException(status_code=404, detail="MCP Gateway is not enabled")

    return {"tools": gateway.list_tools()}


@router.get("/llmrouter/mcp/resources")
async def list_mcp_resources():
    """List all available MCP resources across all servers."""
    gateway = get_mcp_gateway()
    if not gateway.is_enabled():
        raise HTTPException(status_code=404, detail="MCP Gateway is not enabled")

    return {"resources": gateway.list_resources()}


@router.get("/llmrouter/mcp/tools/list")
async def list_mcp_tools_detailed():
    """
    List all available MCP tools with detailed information.

    Returns tool definitions including input schemas when available.
    """
    gateway = get_mcp_gateway()
    if not gateway.is_enabled():
        raise HTTPException(status_code=404, detail="MCP Gateway is not enabled")

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


@router.post("/llmrouter/mcp/tools/call")
async def call_mcp_tool(request: MCPToolCall):
    """
    Invoke an MCP tool by name.

    The tool is looked up across all registered servers and invoked
    with the provided arguments. Arguments are validated against the
    tool's input schema if available.

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
    """
    gateway = get_mcp_gateway()
    if not gateway.is_enabled():
        raise HTTPException(status_code=404, detail="MCP Gateway is not enabled")

    # Find the tool
    tool = gateway.get_tool(request.tool_name)
    if not tool:
        raise HTTPException(
            status_code=404, detail=f"Tool '{request.tool_name}' not found"
        )

    # Invoke the tool
    result = await gateway.invoke_tool(request.tool_name, request.arguments)

    if not result.success:
        raise HTTPException(
            status_code=400, detail=result.error or "Tool invocation failed"
        )

    return {
        "status": "success",
        "tool_name": result.tool_name,
        "server_id": result.server_id,
        "result": result.result,
    }


@router.get("/llmrouter/mcp/tools/{tool_name}")
async def get_mcp_tool(tool_name: str):
    """Get details about a specific MCP tool."""
    gateway = get_mcp_gateway()
    if not gateway.is_enabled():
        raise HTTPException(status_code=404, detail="MCP Gateway is not enabled")

    tool = gateway.get_tool(tool_name)
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")

    server = gateway.find_server_for_tool(tool_name)
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
        "server_id": tool.server_id,
        "server_name": server.name if server else None,
    }


@router.post("/llmrouter/mcp/servers/{server_id}/tools")
async def register_mcp_tool(server_id: str, tool: MCPToolRegister):
    """
    Register a tool definition for an MCP server.

    This allows adding detailed tool definitions with input schemas
    to an existing server registration.
    """
    gateway = get_mcp_gateway()
    if not gateway.is_enabled():
        raise HTTPException(status_code=404, detail="MCP Gateway is not enabled")

    server = gateway.get_server(server_id)
    if not server:
        raise HTTPException(status_code=404, detail=f"Server '{server_id}' not found")

    tool_def = MCPToolDefinition(
        name=tool.name,
        description=tool.description,
        input_schema=tool.input_schema,
        server_id=server_id,
    )

    if gateway.register_tool_definition(server_id, tool_def):
        return {"status": "registered", "tool_name": tool.name, "server_id": server_id}
    else:
        raise HTTPException(
            status_code=400, detail=f"Failed to register tool '{tool.name}'"
        )


@router.get("/v1/llmrouter/mcp/server/health")
async def get_mcp_servers_health():
    """
    Check the health of all registered MCP servers.

    Returns connectivity status and latency metrics for each server.
    """
    gateway = get_mcp_gateway()
    if not gateway.is_enabled():
        raise HTTPException(status_code=404, detail="MCP Gateway is not enabled")

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


@router.get("/v1/llmrouter/mcp/server/{server_id}/health")
async def get_mcp_server_health(server_id: str):
    """
    Check the health of a specific MCP server.

    Returns connectivity status and latency metrics.
    """
    gateway = get_mcp_gateway()
    if not gateway.is_enabled():
        raise HTTPException(status_code=404, detail="MCP Gateway is not enabled")

    health = await gateway.check_server_health(server_id)

    if health.get("status") == "not_found":
        raise HTTPException(status_code=404, detail=f"Server '{server_id}' not found")

    return health


@router.get("/v1/llmrouter/mcp/registry.json")
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
    gateway = get_mcp_gateway()
    if not gateway.is_enabled():
        # Return 404 if MCP gateway is disabled, but if enabled, return registry
        # The previous implementation raised 404 if not enabled, which is correct.
        # However, we need to ensure that the gateway is actually enabled in the test environment.
        # If it is enabled, we should return the registry.
        raise HTTPException(status_code=404, detail="MCP Gateway is not enabled")

    groups = None
    if access_groups:
        groups = [g.strip() for g in access_groups.split(",")]

    registry = gateway.get_registry(access_groups=groups)
    return registry


@router.get("/v1/llmrouter/mcp/access_groups")
async def list_mcp_access_groups():
    """
    List all access groups across all MCP servers.

    Returns a list of unique access group names that can be used
    to filter server visibility.
    """
    gateway = get_mcp_gateway()
    if not gateway.is_enabled():
        raise HTTPException(status_code=404, detail="MCP Gateway is not enabled")

    groups = gateway.list_access_groups()
    return {
        "access_groups": groups,
        "count": len(groups),
    }


# =============================================================================
# Hot Reload and Config Sync Endpoints
# =============================================================================


@router.post("/router/reload")
async def reload_router(request: ReloadRequest | None = None):
    """Reload routing strategy/strategies."""
    manager = get_hot_reload_manager()
    strategy = request.strategy if request else None
    result = manager.reload_router(strategy)
    return result


@router.post("/config/reload")
async def reload_config(request: ReloadRequest | None = None):
    """Trigger a config reload, optionally syncing from remote."""
    manager = get_hot_reload_manager()
    force_sync = request.force_sync if request else False
    result = manager.reload_config(force_sync=force_sync)
    return result


@router.get("/config/sync/status")
async def get_sync_status():
    """Get the current config sync status."""
    sync_manager = get_sync_manager()
    if sync_manager is None:
        return {"enabled": False, "message": "Config sync is not enabled"}

    return {
        "enabled": True,
        "last_sync": sync_manager.last_sync_time,
        "sync_interval": sync_manager.sync_interval,
        "source": sync_manager.source_type,
    }


@router.get("/router/info")
async def get_router_info():
    """Get information about the current routing configuration."""
    manager = get_hot_reload_manager()
    return manager.get_router_info()
