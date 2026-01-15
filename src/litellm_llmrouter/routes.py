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

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from .a2a_gateway import A2AAgent, JSONRPCRequest, get_a2a_gateway
from .mcp_gateway import MCPServer, MCPTransport, MCPToolDefinition, get_mcp_gateway
from .hot_reload import get_hot_reload_manager
from .config_sync import get_sync_manager
from .database import A2AAgentDB, get_a2a_repository, get_a2a_activity_tracker

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


class AgentCreate(BaseModel):
    """Request model for creating an agent with DB persistence."""

    name: str
    description: str = ""
    url: str
    capabilities: list[str] = []
    metadata: dict[str, Any] = {}
    team_id: str | None = None
    user_id: str | None = None
    is_public: bool = False


class AgentUpdate(BaseModel):
    """Request model for full agent update."""

    name: str
    description: str = ""
    url: str
    capabilities: list[str] = []
    metadata: dict[str, Any] = {}
    team_id: str | None = None
    user_id: str | None = None
    is_public: bool = False


class AgentPatch(BaseModel):
    """Request model for partial agent update."""

    name: str | None = None
    description: str | None = None
    url: str | None = None
    capabilities: list[str] | None = None
    metadata: dict[str, Any] | None = None
    team_id: str | None = None
    user_id: str | None = None
    is_public: bool | None = None


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


@router.get("/a2a/{agent_id}/.well-known/agent-card.json")
async def get_a2a_well_known_agent_card(agent_id: str, request: Request):
    """
    Get the A2A agent card at the well-known URL (A2A protocol discovery).
    
    The URL in the agent card is rewritten to point to this gateway,
    so all subsequent A2A calls go through the gateway for logging and tracking.
    """
    gateway = get_a2a_gateway()
    if not gateway.is_enabled():
        raise HTTPException(status_code=404, detail="A2A Gateway is not enabled")

    card = gateway.get_agent_card(agent_id)
    if not card:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    # Rewrite URL to point to this gateway
    base_url = str(request.base_url).rstrip("/")
    card["url"] = f"{base_url}/a2a/{agent_id}"
    
    return JSONResponse(content=card)


@router.post("/a2a/{agent_id}")
async def invoke_a2a_agent(agent_id: str, request: Request):
    """
    Invoke an agent using the A2A protocol (JSON-RPC 2.0).
    
    Supported methods:
    - message/send: Send a message and get a response
    - message/stream: Send a message and stream the response
    
    Request body should be a JSON-RPC 2.0 request:
    ```json
    {
        "jsonrpc": "2.0",
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": "Hello"}]
            }
        },
        "id": "1"
    }
    ```
    """
    import time
    
    gateway = get_a2a_gateway()
    if not gateway.is_enabled():
        return JSONResponse(
            content={
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32000, "message": "A2A Gateway is not enabled"},
            },
            status_code=404,
        )

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            content={
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error: Invalid JSON"},
            },
            status_code=400,
        )

    # Validate JSON-RPC format
    if body.get("jsonrpc") != "2.0":
        return JSONResponse(
            content={
                "jsonrpc": "2.0",
                "id": body.get("id"),
                "error": {"code": -32600, "message": "Invalid Request: jsonrpc must be '2.0'"},
            },
            status_code=400,
        )

    method = body.get("method")
    if not method:
        return JSONResponse(
            content={
                "jsonrpc": "2.0",
                "id": body.get("id"),
                "error": {"code": -32600, "message": "Invalid Request: method is required"},
            },
            status_code=400,
        )

    # Create JSON-RPC request
    jsonrpc_request = JSONRPCRequest(
        method=method,
        params=body.get("params", {}),
        id=body.get("id"),
        jsonrpc=body.get("jsonrpc", "2.0"),
    )

    # Handle streaming vs non-streaming
    if method == "message/stream":
        return StreamingResponse(
            gateway.stream_agent_response(agent_id, jsonrpc_request),
            media_type="application/x-ndjson",
        )
    else:
        # Track invocation timing
        start_time = time.time()
        response = await gateway.invoke_agent(agent_id, jsonrpc_request)
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Record activity for analytics
        tracker = get_a2a_activity_tracker()
        success = response.error is None
        await tracker.record_invocation(agent_id, latency_ms, success)
        
        status_code = 200 if response.error is None else 400
        if response.error and response.error.get("code") == -32000:
            # Agent not found or similar
            if "not found" in response.error.get("message", "").lower():
                status_code = 404
        return JSONResponse(content=response.to_dict(), status_code=status_code)


@router.post("/a2a/{agent_id}/message/send")
async def invoke_a2a_agent_message_send(agent_id: str, request: Request):
    """
    Invoke an agent using message/send method (convenience endpoint).
    
    This is an alias for POST /a2a/{agent_id} with method="message/send".
    """
    return await invoke_a2a_agent(agent_id, request)


@router.post("/a2a/{agent_id}/message/stream")
async def invoke_a2a_agent_message_stream(agent_id: str, request: Request):
    """
    Invoke an agent using message/stream method (convenience endpoint).
    
    This is an alias for POST /a2a/{agent_id} with method="message/stream".
    Returns a streaming response using Server-Sent Events.
    """
    return await invoke_a2a_agent(agent_id, request)


# =============================================================================
# A2A Database-Backed Endpoints (/v1/agents)
# =============================================================================


@router.get("/v1/agents")
async def list_agents_v1(
    user_id: str | None = Query(None),
    team_id: str | None = Query(None),
    include_public: bool = Query(True),
):
    """
    List all agents with optional filtering by user, team, or public status.
    
    This endpoint uses database persistence for agent storage.
    """
    gateway = get_a2a_gateway()
    if not gateway.is_enabled():
        raise HTTPException(status_code=404, detail="A2A Gateway is not enabled")

    repo = get_a2a_repository()
    agents = await repo.list_all(
        user_id=user_id,
        team_id=team_id,
        include_public=include_public,
    )
    return {
        "agents": [agent.to_dict() for agent in agents]
    }


@router.post("/v1/agents")
async def create_agent_v1(agent: AgentCreate):
    """
    Create a new agent with database persistence.
    
    The agent is stored in PostgreSQL (if configured) and also registered
    with the in-memory A2A gateway for immediate use.
    """
    import uuid
    
    gateway = get_a2a_gateway()
    if not gateway.is_enabled():
        raise HTTPException(status_code=404, detail="A2A Gateway is not enabled")

    # Create agent in database
    repo = get_a2a_repository()
    agent_id = str(uuid.uuid4())
    
    db_agent = A2AAgentDB(
        agent_id=agent_id,
        name=agent.name,
        description=agent.description,
        url=agent.url,
        capabilities=agent.capabilities,
        metadata=agent.metadata,
        team_id=agent.team_id,
        user_id=agent.user_id,
        is_public=agent.is_public,
    )
    
    created_agent = await repo.create(db_agent)
    
    # Also register with in-memory gateway for immediate use
    a2a_agent = A2AAgent(
        agent_id=agent_id,
        name=agent.name,
        description=agent.description,
        url=agent.url,
        capabilities=agent.capabilities,
        metadata=agent.metadata,
    )
    gateway.register_agent(a2a_agent)
    
    return {"status": "created", "agent": created_agent.to_dict()}


@router.get("/v1/agents/{agent_id}")
async def get_agent_v1(agent_id: str):
    """Get a specific agent by ID from the database."""
    gateway = get_a2a_gateway()
    if not gateway.is_enabled():
        raise HTTPException(status_code=404, detail="A2A Gateway is not enabled")

    repo = get_a2a_repository()
    agent = await repo.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

    return agent.to_dict()


@router.put("/v1/agents/{agent_id}")
async def update_agent_v1(agent_id: str, agent: AgentUpdate):
    """
    Full update of an agent (replaces all fields).
    
    Requires all fields to be provided.
    """
    gateway = get_a2a_gateway()
    if not gateway.is_enabled():
        raise HTTPException(status_code=404, detail="A2A Gateway is not enabled")

    repo = get_a2a_repository()
    
    # Check if agent exists
    existing = await repo.get(agent_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    # Create updated agent
    updated_agent = A2AAgentDB(
        agent_id=agent_id,
        name=agent.name,
        description=agent.description,
        url=agent.url,
        capabilities=agent.capabilities,
        metadata=agent.metadata,
        team_id=agent.team_id,
        user_id=agent.user_id,
        is_public=agent.is_public,
    )
    
    result = await repo.update(agent_id, updated_agent)
    if not result:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    # Update in-memory gateway
    a2a_agent = A2AAgent(
        agent_id=agent_id,
        name=agent.name,
        description=agent.description,
        url=agent.url,
        capabilities=agent.capabilities,
        metadata=agent.metadata,
    )
    gateway.register_agent(a2a_agent)
    
    return {"status": "updated", "agent": result.to_dict()}


@router.patch("/v1/agents/{agent_id}")
async def patch_agent_v1(agent_id: str, agent: AgentPatch):
    """
    Partial update of an agent (only updates provided fields).
    """
    gateway = get_a2a_gateway()
    if not gateway.is_enabled():
        raise HTTPException(status_code=404, detail="A2A Gateway is not enabled")

    repo = get_a2a_repository()
    
    # Build updates dict from non-None fields
    updates = {}
    if agent.name is not None:
        updates["name"] = agent.name
    if agent.description is not None:
        updates["description"] = agent.description
    if agent.url is not None:
        updates["url"] = agent.url
    if agent.capabilities is not None:
        updates["capabilities"] = agent.capabilities
    if agent.metadata is not None:
        updates["metadata"] = agent.metadata
    if agent.team_id is not None:
        updates["team_id"] = agent.team_id
    if agent.user_id is not None:
        updates["user_id"] = agent.user_id
    if agent.is_public is not None:
        updates["is_public"] = agent.is_public
    
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    
    result = await repo.patch(agent_id, updates)
    if not result:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    # Update in-memory gateway
    a2a_agent = A2AAgent(
        agent_id=agent_id,
        name=result.name,
        description=result.description,
        url=result.url,
        capabilities=result.capabilities,
        metadata=result.metadata,
    )
    gateway.register_agent(a2a_agent)
    
    return {"status": "patched", "agent": result.to_dict()}


@router.delete("/v1/agents/{agent_id}")
async def delete_agent_v1(agent_id: str):
    """Delete an agent from the database."""
    gateway = get_a2a_gateway()
    if not gateway.is_enabled():
        raise HTTPException(status_code=404, detail="A2A Gateway is not enabled")

    repo = get_a2a_repository()
    deleted = await repo.delete(agent_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    # Also remove from in-memory gateway
    gateway.unregister_agent(agent_id)
    
    return {"status": "deleted", "agent_id": agent_id}


@router.post("/v1/agents/{agent_id}/make_public")
async def make_agent_public_v1(agent_id: str):
    """Make an agent publicly visible."""
    gateway = get_a2a_gateway()
    if not gateway.is_enabled():
        raise HTTPException(status_code=404, detail="A2A Gateway is not enabled")

    repo = get_a2a_repository()
    result = await repo.make_public(agent_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    return {"status": "public", "agent": result.to_dict()}


@router.get("/agent/daily/activity")
async def get_agent_daily_activity(
    agent_id: str | None = Query(None, description="Filter by agent ID"),
    start_date: str | None = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: str | None = Query(None, description="End date (YYYY-MM-DD)"),
    aggregated: bool = Query(False, description="Return aggregated statistics"),
):
    """
    Get agent daily activity analytics.
    
    Returns invocation counts, latency metrics, and success/error rates
    for A2A agents. Supports filtering by agent ID and date range.
    
    When `aggregated=True`, returns summary statistics instead of daily records.
    """
    from datetime import date as date_type
    
    gateway = get_a2a_gateway()
    if not gateway.is_enabled():
        raise HTTPException(status_code=404, detail="A2A Gateway is not enabled")

    tracker = get_a2a_activity_tracker()
    
    # Parse dates
    parsed_start = None
    parsed_end = None
    
    if start_date:
        try:
            parsed_start = date_type.fromisoformat(start_date)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid start_date format: {start_date}. Use YYYY-MM-DD."
            )
    
    if end_date:
        try:
            parsed_end = date_type.fromisoformat(end_date)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid end_date format: {end_date}. Use YYYY-MM-DD."
            )
    
    if aggregated:
        stats = await tracker.get_aggregated_activity(
            agent_id=agent_id,
            start_date=parsed_start,
            end_date=parsed_end,
        )
        return {"aggregated": True, "statistics": stats}
    else:
        activities = await tracker.get_daily_activity(
            agent_id=agent_id,
            start_date=parsed_start,
            end_date=parsed_end,
        )
        return {
            "aggregated": False,
            "activities": [a.to_dict() for a in activities],
            "count": len(activities),
        }


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


@router.put("/mcp/servers/{server_id}")
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
        }
    }


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


@router.get("/mcp/tools/list")
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


@router.post("/mcp/tools/call")
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
            status_code=404,
            detail=f"Tool '{request.tool_name}' not found"
        )

    # Invoke the tool
    result = await gateway.invoke_tool(request.tool_name, request.arguments)

    if not result.success:
        raise HTTPException(
            status_code=400,
            detail=result.error or "Tool invocation failed"
        )

    return {
        "status": "success",
        "tool_name": result.tool_name,
        "server_id": result.server_id,
        "result": result.result,
    }


@router.get("/mcp/tools/{tool_name}")
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


@router.post("/mcp/servers/{server_id}/tools")
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
            status_code=400,
            detail=f"Failed to register tool '{tool.name}'"
        )


@router.get("/v1/mcp/server/health")
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
        }
    }


@router.get("/v1/mcp/server/{server_id}/health")
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


@router.get("/v1/mcp/registry.json")
async def get_mcp_registry(
    access_groups: str | None = Query(None, description="Comma-separated access groups")
):
    """
    Get the MCP registry document for discovery.
    
    Returns a registry document listing all servers and their capabilities.
    Optionally filter by access groups.
    """
    gateway = get_mcp_gateway()
    if not gateway.is_enabled():
        raise HTTPException(status_code=404, detail="MCP Gateway is not enabled")

    groups = None
    if access_groups:
        groups = [g.strip() for g in access_groups.split(",")]
    
    registry = gateway.get_registry(access_groups=groups)
    return registry


@router.get("/v1/mcp/access_groups")
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
