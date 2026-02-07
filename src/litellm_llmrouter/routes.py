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
- MCP Namespaced Routes - /mcp for built-in, /{server_prefix}/mcp for per-server
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
        mcp_namespace_router,
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
    app.include_router(mcp_namespace_router)  # Namespaced /mcp routes
    # Feature-flagged:
    app.include_router(mcp_proxy_router)  # MCP protocol proxy (if enabled)
    app.include_router(oauth_callback_router)  # OAuth callback (if enabled)
"""

import asyncio
import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

from .auth import (
    admin_api_key_auth,
    get_request_id,
    sanitize_error_response,
    RequestIDMiddleware,
)
from .rbac import (
    requires_permission,
    PERMISSION_MCP_SERVER_WRITE,
    PERMISSION_MCP_TOOL_WRITE,
    PERMISSION_MCP_TOOL_CALL,
    PERMISSION_SYSTEM_CONFIG_RELOAD,
    PERMISSION_A2A_AGENT_WRITE,
)
from .audit import (
    audit_log,
    AuditAction,
    AuditOutcome,
    AuditWriteError,
)
from .mcp_gateway import MCPServer, MCPTransport, MCPToolDefinition, get_mcp_gateway
from .hot_reload import get_hot_reload_manager
from .config_sync import get_sync_manager
from .url_security import validate_outbound_url_async, SSRFBlockedError
from .resilience import get_drain_manager, get_circuit_breaker_manager

# Import MCP parity layer routers and feature flags
from .mcp_parity import (
    mcp_parity_router,
    mcp_parity_admin_router,
    mcp_rest_router,
    mcp_proxy_router,
    mcp_namespace_router,
    oauth_callback_router,
    MCP_OAUTH_ENABLED,
    MCP_PROTOCOL_PROXY_ENABLED,
)

# Import MCP native JSON-RPC router
from .mcp_jsonrpc import mcp_jsonrpc_router

# Import MCP SSE transport router
from .mcp_sse_transport import (
    mcp_sse_router,
    MCP_SSE_TRANSPORT_ENABLED,
    MCP_SSE_LEGACY_MODE,
    MCP_SSE_SESSION_TIMEOUT,
    get_transport_mode,
    get_session,
    cleanup_expired_sessions,
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
    "mcp_namespace_router",
    "oauth_callback_router",
    "MCP_OAUTH_ENABLED",
    "MCP_PROTOCOL_PROXY_ENABLED",
    # MCP Native JSON-RPC
    "mcp_jsonrpc_router",
    # MCP SSE Transport
    "mcp_sse_router",
    "MCP_SSE_TRANSPORT_ENABLED",
    "MCP_SSE_LEGACY_MODE",
    "MCP_SSE_SESSION_TIMEOUT",
    "get_transport_mode",
    "get_session",
    "cleanup_expired_sessions",
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
    If the server is draining (graceful shutdown), returns 503.
    If any circuit breaker is open, returns 200 but with degraded status.

    Returns:
        200 OK if all configured dependencies are healthy and not draining
        503 Service Unavailable if any configured dependency is unhealthy or draining
    """
    request_id = get_request_id() or "unknown"
    checks = {}
    is_ready = True
    is_degraded = False

    # Check drain status first (highest priority)
    drain_manager = get_drain_manager()
    if drain_manager.is_draining:
        checks["drain_status"] = {
            "status": "draining",
            "active_requests": drain_manager.active_requests,
        }
        is_ready = False
    else:
        checks["drain_status"] = {
            "status": "accepting",
            "active_requests": drain_manager.active_requests,
        }

    # Check circuit breakers - degraded mode detection
    cb_manager = get_circuit_breaker_manager()
    cb_status = cb_manager.get_status()
    if cb_status["is_degraded"]:
        is_degraded = True
        checks["circuit_breakers"] = {
            "status": "degraded",
            "open_breakers": cb_status["degraded_components"],
            "breakers": {
                name: {
                    "state": info["state"],
                    "failure_count": info["failure_count"],
                    "time_until_retry": info["time_until_retry"],
                }
                for name, info in cb_status["breakers"].items()
            },
        }
    else:
        checks["circuit_breakers"] = {
            "status": "healthy",
            "breakers": {
                name: {"state": info["state"]}
                for name, info in cb_status["breakers"].items()
            } if cb_status["breakers"] else {},
        }

    # Check database if configured
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        db_breaker = cb_manager.database
        # If circuit breaker is open, skip the actual check
        if db_breaker.is_open:
            checks["database"] = {
                "status": "degraded",
                "circuit_breaker": "open",
                "time_until_retry": round(db_breaker.time_until_retry, 1),
            }
            # Degraded is not a readiness failure - service can still work with cache
            is_degraded = True
        else:
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
                # Record success in circuit breaker
                await db_breaker.record_success()
            except asyncio.TimeoutError:
                await db_breaker.record_failure("connection timeout")
                checks["database"] = {"status": "unhealthy", "error": "connection timeout"}
                is_ready = False
            except ImportError:
                # asyncpg not installed, try basic connectivity via litellm
                checks["database"] = {
                    "status": "skipped",
                    "reason": "asyncpg not installed",
                }
            except Exception as e:
                await db_breaker.record_failure(str(e))
                # Sanitize: don't leak exception details in health check response
                checks["database"] = {"status": "unhealthy", "error": "connection failed"}
                is_ready = False

    # Check Redis if configured
    redis_host = os.getenv("REDIS_HOST")
    if redis_host:
        redis_breaker = cb_manager.redis
        # If circuit breaker is open, skip the actual check
        if redis_breaker.is_open:
            checks["redis"] = {
                "status": "degraded",
                "circuit_breaker": "open",
                "time_until_retry": round(redis_breaker.time_until_retry, 1),
            }
            is_degraded = True
        else:
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
                await redis_breaker.record_success()
            except asyncio.TimeoutError:
                await redis_breaker.record_failure("connection timeout")
                checks["redis"] = {"status": "unhealthy", "error": "connection timeout"}
                is_ready = False
            except ImportError:
                checks["redis"] = {
                    "status": "skipped",
                    "reason": "redis package not installed",
                }
            except Exception as e:
                await redis_breaker.record_failure(str(e))
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

    # Determine overall status
    if is_degraded and is_ready:
        status = "degraded"
    elif is_ready:
        status = "ready"
    else:
        status = "not_ready"

    response = {
        "status": status,
        "service": "litellm-llmrouter",
        "is_degraded": is_degraded,
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


# Write operations - admin auth required + RBAC
@admin_router.post("/a2a/agents")
async def register_a2a_agent_convenience(
    agent: AgentRegistration,
    rbac_info: dict = Depends(requires_permission(PERMISSION_A2A_AGENT_WRITE)),
):
    """
    Register a new A2A agent.

    This is a convenience endpoint that wraps LiteLLM's global_agent_registry.
    For DB-backed persistence, use POST /v1/agents instead.

    Note: Agents registered via this endpoint are in-memory only and will be
    lost on restart. For HA consistency, use POST /v1/agents which persists
    to the database.

    Requires admin API key authentication or user with a2a.agent.write permission.

    Security: URLs are validated against SSRF attacks.
    """
    request_id = get_request_id() or "unknown"

    # Security: Validate URL against SSRF attacks before registration
    # Use async version to avoid blocking the event loop
    if agent.url:
        try:
            await validate_outbound_url_async(agent.url, resolve_dns=False)
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

        # Audit log the success
        await _handle_audit_write(
            AuditAction.A2A_AGENT_CREATE,
            "a2a_agent",
            agent_id,
            AuditOutcome.SUCCESS,
            rbac_info,
            request_id,
        )

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


# Write operations - admin auth required + RBAC
@admin_router.delete("/agents/{agent_id}")
async def unregister_a2a_agent_convenience(
    agent_id: str,
    rbac_info: dict = Depends(requires_permission(PERMISSION_A2A_AGENT_WRITE)),
):
    """
    Unregister an A2A agent.

    This is a convenience endpoint that wraps LiteLLM's global_agent_registry.
    For DB-backed deletion, use DELETE /v1/agents/{agent_id} instead.

    Note: This only removes from in-memory registry. DB-backed agents will
    be re-loaded on restart. Use DELETE /v1/agents/{agent_id} for permanent deletion.

    Requires admin API key authentication or user with a2a.agent.write permission.
    """
    request_id = get_request_id() or "unknown"
    try:
        from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry

        # Get agent by ID first to find its name (needed for deregister_agent)
        agent = global_agent_registry.get_agent_by_id(agent_id)
        if agent:
            global_agent_registry.deregister_agent(agent_name=agent.agent_name)
            
            # Audit log the success
            await _handle_audit_write(
                AuditAction.A2A_AGENT_DELETE,
                "a2a_agent",
                agent_id,
                AuditOutcome.SUCCESS,
                rbac_info,
                request_id,
            )
            
            return {
                "status": "unregistered",
                "agent_id": agent_id,
                "note": "Agent removed from in-memory registry. For permanent deletion, use DELETE /v1/agents/{agent_id}",
            }

        # Try by name as fallback
        agent = global_agent_registry.get_agent_by_name(agent_id)
        if agent:
            global_agent_registry.deregister_agent(agent_name=agent_id)
            
            # Audit log the success
            await _handle_audit_write(
                AuditAction.A2A_AGENT_DELETE,
                "a2a_agent",
                agent_id,
                AuditOutcome.SUCCESS,
                rbac_info,
                request_id,
            )
            
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


# =============================================================================
# Hot Reload and Config Sync Endpoints
# =============================================================================


# Config reload - admin auth required + RBAC
@admin_router.post("/llmrouter/reload")
async def reload_config(
    request: ReloadRequest | None = None,
    rbac_info: dict = Depends(requires_permission(PERMISSION_SYSTEM_CONFIG_RELOAD)),
):
    """
    Trigger a config reload, optionally syncing from remote.

    Requires admin API key authentication or user with system.config.reload permission.
    """
    request_id = get_request_id() or "unknown"
    try:
        manager = get_sync_manager()
        force_sync = request.force_sync if request else False
        result = manager.reload_config(force_sync=force_sync)
        
        # Audit log the success
        await _handle_audit_write(
            AuditAction.CONFIG_RELOAD,
            "config",
            "llmrouter",
            AuditOutcome.SUCCESS,
            rbac_info,
            request_id,
        )
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        err = sanitize_error_response(e, request_id, "Failed to reload config")
        raise HTTPException(status_code=500, detail=err)


# Config reload - admin auth required + RBAC
@admin_router.post("/config/reload")
async def reload_config_2(
    request: ReloadRequest | None = None,
    rbac_info: dict = Depends(requires_permission(PERMISSION_SYSTEM_CONFIG_RELOAD)),
):
    """
    Trigger a config reload, optionally syncing from remote.

    Requires admin API key authentication or user with system.config.reload permission.
    """
    request_id = get_request_id() or "unknown"
    try:
        manager = get_hot_reload_manager()
        force_sync = request.force_sync if request else False
        result = manager.reload_config(force_sync=force_sync)
        
        # Audit log the success
        await _handle_audit_write(
            AuditAction.CONFIG_RELOAD,
            "config",
            "hot_reload",
            AuditOutcome.SUCCESS,
            rbac_info,
            request_id,
        )
        
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


# 	Read-only - user auth
@llmrouter_router.get("/router/info")
async def get_router_info():
    """Get information about the current routing configuration."""
    manager = get_hot_reload_manager()
    return manager.get_router_info()
