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
- MCP Parity Layer - upstream-compatible endpoint aliases
- Hot reload and config sync endpoints
- Kubernetes health probe endpoints (/_health/live, /_health/ready)

Usage:
    from litellm_llmrouter.routes import (
        health_router,
        llmrouter_router,
        admin_router,
        mcp_parity_router,
        mcp_parity_admin_router,
        mcp_rest_router,
        mcp_proxy_router,
        oauth_callback_router,
        RequestIDMiddleware,
    )
    app.add_middleware(RequestIDMiddleware)  # Add first for request correlation
    app.include_router(health_router)  # Unauthenticated health probes
    app.include_router(llmrouter_router)  # User auth-protected routes
    app.include_router(admin_router)  # Admin auth-protected control-plane routes
    app.include_router(mcp_parity_router)  # Upstream-compatible MCP aliases
    app.include_router(mcp_parity_admin_router)  # Admin MCP parity routes
    app.include_router(mcp_rest_router)  # MCP REST API (/mcp-rest)
    # Feature-flagged:
    app.include_router(mcp_proxy_router)  # MCP protocol proxy (if enabled)
    app.include_router(oauth_callback_router)  # OAuth callback (if enabled)
"""

import asyncio
import os
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

from .auth import (
    admin_api_key_auth,
    get_request_id,
    sanitize_error_response,
    RequestIDMiddleware,
)
from .mcp_gateway import MCPServer, MCPTransport, MCPToolDefinition, get_mcp_gateway
from .hot_reload import get_hot_reload_manager
from .config_sync import get_sync_manager
from .url_security import validate_outbound_url, SSRFBlockedError

# Import MCP parity layer routers and feature flags
from .mcp_parity import (
    mcp_parity_router,
    mcp_parity_admin_router,
    mcp_rest_router,
    mcp_proxy_router,
    oauth_callback_router,
    MCP_OAUTH_ENABLED,
    MCP_PROTOCOL_PROXY_ENABLED,
)

# Health router - unauthenticated endpoints for Kubernetes probes
# These MUST remain accessible without credentials for K8s liveness/readiness
health_router = APIRouter(tags=["health"])

# Main router for user-facing LLMRouter routes - requires LiteLLM API key authentication
# This includes read-only endpoints like /router/info
llmrouter_router = APIRouter(
    tags=["llmrouter"],
    dependencies=[Depends(user_api_key_auth)],
)

# Admin router for control-plane operations - requires admin API key authentication
# This includes MCP server/tool registration, A2A agent registration, and config reload
# These are separate from user traffic and require elevated privileges
admin_router = APIRouter(
    tags=["admin"],
    dependencies=[Depends(admin_api_key_auth)],
)

# Legacy alias for backwards compatibility (deprecated - use health_router + llmrouter_router + admin_router)
router = llmrouter_router

# Re-export middleware for app setup
__all__ = [
    "health_router",
    "llmrouter_router",
    "admin_router",
    "router",
    "RequestIDMiddleware",
    # MCP Parity Layer (upstream-compatible aliases)
    "mcp_parity_router",
    "mcp_parity_admin_router",
    "mcp_rest_router",
    "mcp_proxy_router",
    "oauth_callback_router",
    "MCP_OAUTH_ENABLED",
    "MCP_PROTOCOL_PROXY_ENABLED",
]


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
# Kubernetes Health Probe Endpoints (/_health/*)
# =============================================================================
# These are minimal, unauthenticated endpoints for K8s probes.
# - /_health/live: Liveness probe - doesn't check external deps (DB/Redis)
# - /_health/ready: Readiness probe - checks optional deps with short timeouts
#
# Use these in K8s manifests instead of /health/* which may be auth-protected.


@health_router.get("/_health/live")
async def liveness_probe():
    """
    Kubernetes liveness probe endpoint.

    This endpoint verifies the application process is alive and responsive.
    It does NOT check external dependencies (database, Redis, etc.) because
    liveness failures trigger pod restarts, not traffic rerouting.

    Returns:
        200 OK if the process is alive
    """
    return {"status": "alive", "service": "litellm-llmrouter"}


@health_router.get("/_health/ready")
async def readiness_probe():
    """
    Kubernetes readiness probe endpoint.

    This endpoint verifies the application is ready to accept traffic.
    It checks optional external dependencies (database, Redis) with short
    timeouts (2s) so the probe doesn't hang.

    If a dependency is not configured, it's not checked (still returns ready).
    If a dependency is configured but unreachable, returns 503.

    Returns:
        200 OK if all configured dependencies are healthy
        503 Service Unavailable if any configured dependency is unhealthy
    """
    request_id = get_request_id() or "unknown"
    checks = {}
    is_ready = True

    # Check database if configured
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        try:
            # Import here to avoid circular imports and optional dependency
            import asyncpg

            # Use short timeout for health check
            conn = await asyncio.wait_for(
                asyncpg.connect(db_url, timeout=2.0),
                timeout=2.0,
            )
            await asyncio.wait_for(conn.execute("SELECT 1"), timeout=1.0)
            await conn.close()
            checks["database"] = {"status": "healthy"}
        except asyncio.TimeoutError:
            checks["database"] = {"status": "unhealthy", "error": "connection timeout"}
            is_ready = False
        except ImportError:
            # asyncpg not installed, try basic connectivity via litellm
            checks["database"] = {
                "status": "skipped",
                "reason": "asyncpg not installed",
            }
        except Exception:
            # Sanitize: don't leak exception details in health check response
            checks["database"] = {"status": "unhealthy", "error": "connection failed"}
            is_ready = False

    # Check Redis if configured
    redis_host = os.getenv("REDIS_HOST")
    if redis_host:
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        try:
            import redis.asyncio as aioredis

            r = aioredis.Redis(
                host=redis_host,
                port=redis_port,
                socket_connect_timeout=2.0,
                socket_timeout=2.0,
            )
            await asyncio.wait_for(r.ping(), timeout=2.0)
            await r.aclose()
            checks["redis"] = {"status": "healthy"}
        except asyncio.TimeoutError:
            checks["redis"] = {"status": "unhealthy", "error": "connection timeout"}
            is_ready = False
        except ImportError:
            checks["redis"] = {
                "status": "skipped",
                "reason": "redis package not installed",
            }
        except Exception:
            # Sanitize: don't leak exception details in health check response
            checks["redis"] = {"status": "unhealthy", "error": "connection failed"}
            is_ready = False

    # Check MCP gateway health if enabled
    if os.getenv("MCP_GATEWAY_ENABLED", "false").lower() == "true":
        try:
            gateway = get_mcp_gateway()
            if gateway.is_enabled():
                checks["mcp_gateway"] = {
                    "status": "healthy",
                    "servers": len(gateway.list_servers()),
                }
            else:
                checks["mcp_gateway"] = {"status": "disabled"}
        except Exception:
            # Sanitize: don't leak exception details
            checks["mcp_gateway"] = {"status": "unhealthy", "error": "check failed"}
            # MCP gateway failure is non-fatal for readiness
            # is_ready = False

    response = {
        "status": "ready" if is_ready else "not_ready",
        "service": "litellm-llmrouter",
        "checks": checks,
        "request_id": request_id,
    }

    if not is_ready:
        raise HTTPException(status_code=503, detail=response)

    return response


# =============================================================================
# A2A Gateway Convenience Endpoints (/a2a/agents)
# =============================================================================
# These are thin wrappers around LiteLLM's global_agent_registry for convenience.
# The main A2A functionality is provided by LiteLLM's built-in endpoints:
# - POST /v1/agents - Create agent (DB-backed)
# - GET /v1/agents - List agents (DB-backed)
# - DELETE /v1/agents/{agent_id} - Delete agent (DB-backed)
# - POST /a2a/{agent_id} - Invoke agent (A2A JSON-RPC protocol)
# - POST /a2a/{agent_id}/message/stream - Streaming alias (proxies to canonical)


# Read-only endpoint - user auth is sufficient
@llmrouter_router.get("/a2a/agents")
async def list_a2a_agents_convenience():
    """
    List all registered A2A agents.

    This is a convenience endpoint that wraps LiteLLM's global_agent_registry.
    For full functionality, use GET /v1/agents (DB-backed, supports filtering).

    Returns agents from LiteLLM's in-memory registry (synced from DB+config).
    """
    request_id = get_request_id() or "unknown"
    try:
        from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry

        agents = global_agent_registry.get_agent_list()
        return {
            "agents": [
                {
                    "agent_id": a.agent_id,
                    "agent_name": a.agent_name,
                    "description": (
                        a.agent_card_params.get("description", "")
                        if a.agent_card_params
                        else ""
                    ),
                    "url": (
                        a.agent_card_params.get("url", "")
                        if a.agent_card_params
                        else ""
                    ),
                }
                for a in agents
            ]
        }
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "agent_registry_unavailable",
                "message": "Agent registry not available. Ensure LiteLLM is properly initialized.",
                "request_id": request_id,
            },
        )
    except Exception as e:
        err = sanitize_error_response(e, request_id, "Failed to list agents")
        raise HTTPException(status_code=500, detail=err)


# Helper for A2A router
def ensure_a2a_server(transport: str):
    """
    Helper dependency to ensure the A2A server is streamable.
    """
    t = MCPTransport(transport)
    if not t.is_supported():
        raise HTTPException(
            status_code=500, detail=f"Transport '{transport}' is not supported"
        )
    return t


# Write operations - admin auth required
@admin_router.post("/a2a/agents")
async def register_a2a_agent_convenience(agent: AgentRegistration):
    """
    Register a new A2A agent.

    This is a convenience endpoint that wraps LiteLLM's global_agent_registry.
    For DB-backed persistence, use POST /v1/agents instead.

    Note: Agents registered via this endpoint are in-memory only and will be
    lost on restart. For HA consistency, use POST /v1/agents which persists
    to the database.

    Requires admin API key authentication.

    Security: URLs are validated against SSRF attacks.
    """
    request_id = get_request_id() or "unknown"

    # Security: Validate URL against SSRF attacks before registration
    if agent.url:
        try:
            validate_outbound_url(agent.url, resolve_dns=False)
        except SSRFBlockedError as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "ssrf_blocked",
                    "message": f"Agent URL blocked for security reasons: {e.reason}",
                    "request_id": request_id,
                },
            )
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_url",
                    "message": f"Agent URL is invalid: {str(e)}",
                    "request_id": request_id,
                },
            )

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
    except HTTPException:
        raise
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "agent_registry_unavailable",
                "message": "Agent registry not available. Ensure LiteLLM is properly initialized.",
                "request_id": request_id,
            },
        )
    except Exception as e:
        err = sanitize_error_response(e, request_id, "Failed to register agent")
        raise HTTPException(status_code=500, detail=err)


# Write operations - admin auth required
@admin_router.delete("/agents/{agent_id}")
async def unregister_a2a_agent_convenience(agent_id: str):
    """
    Unregister an A2A agent.

    This is a convenience endpoint that wraps LiteLLM's global_agent_registry.
    For DB-backed deletion, use DELETE /v1/agents/{agent_id} instead.

    Note: This only removes from in-memory registry. DB-backed agents will
    be re-loaded on restart. Use DELETE /v1/agents/{agent_id} for permanent deletion.

    Requires admin API key authentication.
    """
    request_id = get_request_id() or "unknown"
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

        raise HTTPException(
            status_code=404,
            detail={
                "error": "agent_not_found",
                "message": f"Agent {agent_id} not found",
                "request_id": request_id,
            },
        )
    except HTTPException:
        raise
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "agent_registry_unavailable",
                "message": "Agent registry not available. Ensure LiteLLM is properly initialized.",
                "request_id": request_id,
            },
        )
    except Exception as e:
        err = sanitize_error_response(e, request_id, "Failed to unregister agent")
        raise HTTPException(status_code=500, detail=err)


# Streaming alias - user auth sufficient for invocation
@llmrouter_router.post("/chat/completions")
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
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(600.0, connect=60.0)
            ) as client:
                async with client.stream(
                    "POST",
                    upstream_url,
                    content=body,
                    headers=headers,
                ) as response:
                    # For errors, read and yield the full response
                    error_body = await response.aread()
                    yield error_body
                    return

                    async for chunk in response.aiter_bytes():
                        yield chunk
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail={"error": "upstream_timeout"})
        except httpx.HTTPStatusError as e:
            # Upstream returned an HTTP error (not a timeout): include status text
            error_body = f"{e.response.status_code} {e.response.legacy_text or e.response.text or e.response.content or ''}"
            raise HTTPException(
                status_code=e.response.status_code, detail={"error": error_body}
            )
        except ConnectionError:
            raise HTTPException(
                status_code=502, detail={"error": "upstream_connection_error"}
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail={"error": str(e)})

    # Determine content type from the request (preserve SSE if requested)
    accept = request.headers.get("accept", "application/json")
    media_type = "application/json"
    if accept or "application/x-ndjson" in accept or "text/event-stream" in accept:
        media_type = "text/event-stream"

    headers_ = {
        "Content-Type": media_type,
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",  # Disable nginx buffering for SSE
    }
    if media_type == "text/event-stream":
        headers_.update({"Content-Type": media_type})
    else:
        headers_.update({"Content-Type": "application/json"})

    return StreamingResponse(stream_upstream(), headers=headers_)


# =============================================================================
# MCP Gateway Endpoints
# =============================================================================
# These REST endpoints are prefixed with /llmrouter/mcp to avoid conflicts
# with LiteLLM's native /mcp endpoint (which uses JSON-RPC over SSE).


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


# Write operations - admin auth required
@llmrouter_router.post("/llmrouter/mcp/servers")
async def register_mcp_server(server: ServerRegistration):
    """
    Register a new MCP server (REST API).

    Requires admin API key authentication.

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


# Write operation - admin auth
@llmrouter_router.delete("/llmrouter/mcp/servers/{server_id}")
async def unregister_mcp_server(server_id: str):
    """
    Unregister an MCP server (REST API).

    Requires admin API key authentication.
    """
    request_id = get_request_id() or "unknown"
    gateway = get_mcp_gateway()
    if gateway.unregister_server(server_id):
        return {"status": "unregistered", "server_id": server_id}
    raise HTTPException(
        status_code=404,
        detail={
            "error": "server_not_found",
            "message": f"Server {server_id} not found",
            "request_id": request_id,
        },
    )


# Write operation - admin auth
@llmrouter_router.put("/llmrouter/mcp/servers/{server_id}")
async def update_mcp_server(server_id: str, server: ServerRegistration):
    """
    Update an MCP server (full update).

    Replaces all server fields with the provided values.
    Tools and resources are refreshed on update.

    Requires admin API key authentication.

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


# Tool invocation - admin auth (modifies state on external MCP servers)
@llmrouter_router.post("/llmrouter/mcp/tools/call")
async def call_mcp_tool(request: MCPToolCall):
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


# Tool registration - admin auth
@llmrouter_router.post("/llmrouter/mcp/servers/{server_id}/tools")
async def register_mcp_tool(server_id: str, tool: MCPToolRegister):
    """
    Register a tool definition for an MCP server.

    This allows adding detailed tool definitions with input schemas
    to an existing server registration.

    Requires admin API key authentication.
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


# =============================================================================
# Hot Reload and Config Sync Endpoints
# =============================================================================


# Config reload - admin auth required
@llmrouter_router.post("/llmrouter/reload")
async def reload_config(request: ReloadRequest | None = None):
    """
    Trigger a config reload, optionally syncing from remote.

    Requires admin API key authentication.
    """
    request_id = get_request_id() or "unknown"
    try:
        manager = get_sync_manager()
        force_sync = request.force_sync if request else False
        result = manager.reload_config(force_sync=force_sync)
        return result
    except HTTPException:
        raise
    except Exception as e:
        err = sanitize_error_response(e, request_id, "Failed to reload config")
        raise HTTPException(status_code=500, detail=err)


# Config reload - admin auth required
@llmrouter_router.post("/config/reload")
async def reload_config_2(request: ReloadRequest | None = None):
    """
    Trigger a config reload, optionally syncing from remote.

    Requires admin API key authentication.
    """
    request_id = get_request_id() or "unknown"
    try:
        manager = get_hot_reload_manager()
        force_sync = request.force_sync if request else False
        result = manager.reload_config(force_sync=force_sync)
        return result
    except HTTPException:
        raise
    except Exception as e:
        err = sanitize_error_response(e, request_id, "Failed to reload config")
        raise HTTPException(status_code=500, detail=err)


# Read-only - user auth
@llmrouter_router.get("/config/sync/status")
async def get_sync_status():
    """Get the current config sync status."""
    sync_manager = get_sync_manager()
    if sync_manager is None:
        return {"enabled": False, "message": "Config sync is not enabled"}

    return sync_manager.get_status()


# Read-only - user auth
@llmrouter_router.get("/router/info")
async def get_router_info():
    """Get information about the current routing configuration."""
    manager = get_hot_reload_manager()
    return manager.get_router_info()
