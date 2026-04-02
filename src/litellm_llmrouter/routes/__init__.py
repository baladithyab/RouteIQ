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
- MCP (Model Context Protocol) gateway REST endpoints (/llmrouter/mcp/*)
  Note: MCP parity layer, JSON-RPC, and SSE transport are now provided natively
  by LiteLLM and have been removed from RouteIQ.
- Hot reload and config sync endpoints
- Kubernetes health probe endpoints (/_health/live, /_health/ready)

Usage:
    from litellm_llmrouter.routes import (
        health_router,
        llmrouter_router,
        admin_router,
        RequestIDMiddleware,
    )
    app.add_middleware(RequestIDMiddleware)  # Add first for request correlation
    app.include_router(health_router)  # Unauthenticated health probes
    app.include_router(llmrouter_router)  # User auth-protected routes
    app.include_router(admin_router)  # Admin auth-protected control-plane routes
"""

from fastapi import APIRouter, Depends

from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

from ..auth import (
    admin_api_key_auth,
    RequestIDMiddleware,
)

# ---- Router definitions (must be defined BEFORE sub-module imports) ----

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

# ---- Import sub-modules (registers routes on the routers above) ----

from . import health as _health_routes  # noqa: E402, F401
from . import a2a as _a2a_routes  # noqa: E402, F401
from . import mcp as _mcp_routes  # noqa: E402, F401
from . import config as _config_routes  # noqa: E402, F401
from . import admin_ui as _admin_ui_routes  # noqa: E402, F401

# Re-export Pydantic models for backwards compatibility
from .models import (  # noqa: E402
    AgentRegistration,
    ServerRegistration,
    ReloadRequest,
    MCPToolCall,
    MCPToolRegister,
)

# Re-export middleware for app setup
__all__ = [
    "health_router",
    "llmrouter_router",
    "admin_router",
    "router",
    "RequestIDMiddleware",
    # Pydantic models
    "AgentRegistration",
    "ServerRegistration",
    "ReloadRequest",
    "MCPToolCall",
    "MCPToolRegister",
]
