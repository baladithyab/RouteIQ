"""
FastAPI Routes for A2A Gateway, MCP Gateway, and Hot Reload
============================================================

These routes extend the LiteLLM proxy server with:
- A2A (Agent-to-Agent) protocol endpoints
- MCP (Model Context Protocol) gateway endpoints
- Hot reload and config sync endpoints

Usage:
    from litellm_llmrouter.routes import router
    app.include_router(router)
"""

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from .a2a_gateway import A2AAgent, get_a2a_gateway
from .mcp_gateway import MCPServer, MCPTransport, get_mcp_gateway
from .hot_reload import get_hot_reload_manager
from .config_sync import get_sync_manager

router = APIRouter(tags=["llmrouter"])


# =============================================================================
# Pydantic Models
# =============================================================================


class AgentRegistration(BaseModel):
    """Request model for A2A agent registration."""

    agent_id: str
    name: str
    description: str
    url: str
    capabilities: list[str] = []
    metadata: dict[str, Any] = {}


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


# =============================================================================
# A2A Gateway Endpoints
# =============================================================================


@router.get("/a2a/agents")
async def list_a2a_agents(capability: str | None = Query(None)):
    """List all registered A2A agents, optionally filtered by capability."""
    gateway = get_a2a_gateway()
    if not gateway.is_enabled():
        raise HTTPException(status_code=404, detail="A2A Gateway is not enabled")

    agents = gateway.discover_agents(capability)
    return {
        "agents": [
            {
                "agent_id": a.agent_id,
                "name": a.name,
                "description": a.description,
                "url": a.url,
                "capabilities": a.capabilities,
            }
            for a in agents
        ]
    }


@router.post("/a2a/agents")
async def register_a2a_agent(agent: AgentRegistration):
    """Register a new A2A agent."""
    gateway = get_a2a_gateway()
    if not gateway.is_enabled():
        raise HTTPException(status_code=404, detail="A2A Gateway is not enabled")

    a2a_agent = A2AAgent(
        agent_id=agent.agent_id,
        name=agent.name,
        description=agent.description,
        url=agent.url,
        capabilities=agent.capabilities,
        metadata=agent.metadata,
    )
    gateway.register_agent(a2a_agent)
    return {"status": "registered", "agent_id": agent.agent_id}


@router.get("/a2a/agents/{agent_id}")
async def get_a2a_agent(agent_id: str):
    """Get a specific A2A agent by ID."""
    gateway = get_a2a_gateway()
    if not gateway.is_enabled():
        raise HTTPException(status_code=404, detail="A2A Gateway is not enabled")

    agent = gateway.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

    return {
        "agent_id": agent.agent_id,
        "name": agent.name,
        "description": agent.description,
        "url": agent.url,
        "capabilities": agent.capabilities,
        "metadata": agent.metadata,
    }


@router.get("/a2a/agents/{agent_id}/card")
async def get_a2a_agent_card(agent_id: str):
    """Get the A2A agent card for an agent (A2A protocol format)."""
    gateway = get_a2a_gateway()
    card = gateway.get_agent_card(agent_id)
    if not card:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    return card


@router.delete("/a2a/agents/{agent_id}")
async def unregister_a2a_agent(agent_id: str):
    """Unregister an A2A agent."""
    gateway = get_a2a_gateway()
    if gateway.unregister_agent(agent_id):
        return {"status": "unregistered", "agent_id": agent_id}
    raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")


# =============================================================================
# MCP Gateway Endpoints
# =============================================================================


@router.get("/mcp/servers")
async def list_mcp_servers():
    """List all registered MCP servers."""
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


@router.post("/mcp/servers")
async def register_mcp_server(server: ServerRegistration):
    """Register a new MCP server."""
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


@router.get("/mcp/servers/{server_id}")
async def get_mcp_server(server_id: str):
    """Get a specific MCP server by ID."""
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


@router.delete("/mcp/servers/{server_id}")
async def unregister_mcp_server(server_id: str):
    """Unregister an MCP server."""
    gateway = get_mcp_gateway()
    if gateway.unregister_server(server_id):
        return {"status": "unregistered", "server_id": server_id}
    raise HTTPException(status_code=404, detail=f"Server {server_id} not found")


@router.get("/mcp/tools")
async def list_mcp_tools():
    """List all available MCP tools across all servers."""
    gateway = get_mcp_gateway()
    if not gateway.is_enabled():
        raise HTTPException(status_code=404, detail="MCP Gateway is not enabled")

    return {"tools": gateway.list_tools()}


@router.get("/mcp/resources")
async def list_mcp_resources():
    """List all available MCP resources across all servers."""
    gateway = get_mcp_gateway()
    if not gateway.is_enabled():
        raise HTTPException(status_code=404, detail="MCP Gateway is not enabled")

    return {"resources": gateway.list_resources()}


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
