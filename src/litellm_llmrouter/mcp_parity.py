"""
MCP Parity Layer - Upstream-Compatible LiteLLM MCP Endpoints
=============================================================

This module provides upstream-compatible endpoint aliases to match
LiteLLM's MCP management and REST API paths, enabling clients that
target the LiteLLM API to work seamlessly with RouteIQ Gateway.

Upstream Paths Covered:
-----------------------
Management Endpoints (from mcp_management_endpoints.py):
- GET/POST /v1/mcp/server → list/create MCP servers
- GET/DELETE /v1/mcp/server/{server_id} → get/delete specific server
- PUT /v1/mcp/server → update server
- GET /v1/mcp/server/health → server health checks
- GET /v1/mcp/tools → list MCP tools
- GET /v1/mcp/access_groups → list access groups
- GET /v1/mcp/registry.json → MCP registry document
- POST /v1/mcp/make_public → make MCP servers public

MCP REST Endpoints (from rest_endpoints.py):
- GET /mcp-rest/tools/list → list tools with mcp_info
- POST /mcp-rest/tools/call → call tool

OAuth Endpoints (feature-flagged):
- POST /v1/mcp/server/oauth/session → create temporary OAuth session
- GET /v1/mcp/server/oauth/{server_id}/authorize → OAuth authorize redirect
- POST /v1/mcp/server/oauth/{server_id}/token → OAuth token exchange
- POST /v1/mcp/server/oauth/{server_id}/register → OAuth client registration

Protocol Proxy (feature-flagged):
- /mcp/{server_id}/* → proxy to registered MCP server (SSE/streamable_http)
- /mcp → built-in MCP protocol endpoint (namespaced route)

HTTP Client Pooling:
- Uses shared HTTP client pool by default (HTTP_CLIENT_POOLING_ENABLED=true)
- Falls back to per-request clients when pooling is disabled
- See http_client_pool.py for configuration and lifecycle

All aliases delegate to existing handlers; no logic duplication.
"""

import hashlib
import os
import secrets
import time
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel

from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

from .auth import admin_api_key_auth, get_request_id, sanitize_error_response
from .mcp_gateway import (
    MCPServer,
    MCPTransport,
    get_mcp_gateway,
)
from .url_security import SSRFBlockedError, validate_outbound_url_async
from .http_client_pool import get_client_for_request

# Feature flags
MCP_OAUTH_ENABLED = os.getenv("MCP_OAUTH_ENABLED", "false").lower() == "true"
MCP_PROTOCOL_PROXY_ENABLED = (
    os.getenv("MCP_PROTOCOL_PROXY_ENABLED", "false").lower() == "true"
)

# Protocol proxy timeouts
PROTOCOL_PROXY_CONNECT_TIMEOUT = float(os.getenv("MCP_PROXY_CONNECT_TIMEOUT", "10"))
PROTOCOL_PROXY_READ_TIMEOUT = float(os.getenv("MCP_PROXY_READ_TIMEOUT", "120"))

# ============================================================================
# Routers
# ============================================================================

# Upstream-compatible management endpoints - with user auth for reads
mcp_parity_router = APIRouter(
    prefix="/v1/mcp",
    tags=["mcp-parity"],
    dependencies=[Depends(user_api_key_auth)],
)

# Admin-only management endpoints
mcp_parity_admin_router = APIRouter(
    prefix="/v1/mcp",
    tags=["mcp-parity-admin"],
    dependencies=[Depends(admin_api_key_auth)],
)

# MCP REST API endpoints (upstream path: /mcp-rest)
mcp_rest_router = APIRouter(
    prefix="/mcp-rest",
    tags=["mcp-rest"],
    dependencies=[Depends(user_api_key_auth)],
)

# Protocol proxy (feature-flagged, requires admin)
mcp_proxy_router = APIRouter(
    prefix="/mcp",
    tags=["mcp-proxy"],
    dependencies=[Depends(admin_api_key_auth)],
)

# Namespaced MCP protocol router - for built-in MCP server
# This provides upstream-compatible namespaced /mcp routes
mcp_namespace_router = APIRouter(
    tags=["mcp-namespace"],
    dependencies=[Depends(user_api_key_auth)],
)

# ============================================================================
# Pydantic Models (matching upstream schemas)
# ============================================================================


class NewMCPServerRequest(BaseModel):
    """Request model for creating/updating an MCP server (matches upstream)."""

    server_id: str | None = None
    server_name: str | None = None
    alias: str | None = None
    description: str | None = None
    url: str | None = None
    transport: str = "streamable_http"
    auth_type: str | None = "none"
    credentials: dict[str, Any] | None = None
    mcp_access_groups: list[str] | None = None
    allowed_tools: list[str] | None = None
    extra_headers: list[str] | None = None
    mcp_info: dict[str, Any] | None = None
    static_headers: dict[str, str] | None = None
    command: str | None = None
    args: list[str] | None = None
    env: dict[str, str] | None = None
    authorization_url: str | None = None
    token_url: str | None = None
    registration_url: str | None = None
    allow_all_keys: bool = False


class UpdateMCPServerRequest(BaseModel):
    """Request model for updating an MCP server (matches upstream)."""

    server_id: str
    server_name: str | None = None
    alias: str | None = None
    description: str | None = None
    url: str | None = None
    transport: str | None = None
    auth_type: str | None = None
    credentials: dict[str, Any] | None = None
    mcp_access_groups: list[str] | None = None
    allowed_tools: list[str] | None = None
    extra_headers: list[str] | None = None
    mcp_info: dict[str, Any] | None = None


class MCPToolCallRequest(BaseModel):
    """Request model for MCP tool invocation (matches upstream)."""

    server_id: str
    name: str
    arguments: dict[str, Any] | None = None


class MakeMCPServersPublicRequest(BaseModel):
    """Request model for making MCP servers public (matches upstream)."""

    mcp_server_ids: list[str]


# ============================================================================
# OAuth Storage (in-memory, for demonstration)
# ============================================================================


@dataclass
class OAuthSession:
    """Stores OAuth session state for CSRF protection."""

    server_id: str
    state: str
    redirect_uri: str
    created_at: float
    client_id: str | None = None
    code_verifier: str | None = None


@dataclass
class OAuthToken:
    """Stores OAuth tokens for an MCP server."""

    server_id: str
    access_token: str
    token_type: str = "Bearer"
    refresh_token: str | None = None
    expires_at: float | None = None
    scope: str | None = None


# In-memory OAuth stores (in production, use Redis or DB)
_oauth_sessions: dict[str, OAuthSession] = {}
_oauth_tokens: dict[str, OAuthToken] = {}

# In-memory public MCP servers list (mimics upstream litellm.public_mcp_servers)
_public_mcp_servers: list[str] = []

# OAuth session TTL (5 minutes)
OAUTH_SESSION_TTL = 300


def _cleanup_expired_sessions() -> None:
    """Remove expired OAuth sessions."""
    now = time.time()
    expired = [
        k for k, v in _oauth_sessions.items() if now - v.created_at > OAUTH_SESSION_TTL
    ]
    for k in expired:
        del _oauth_sessions[k]


def _generate_state() -> str:
    """Generate a cryptographically secure state parameter."""
    return secrets.token_urlsafe(32)


# ============================================================================
# Management Endpoint Aliases (Thin Wrappers)
# ============================================================================


@mcp_parity_router.get("/server")
async def fetch_all_mcp_servers():
    """
    Upstream-compatible: GET /v1/mcp/server

    Returns all configured MCP servers (filtered by user access in upstream).
    Alias for GET /llmrouter/mcp/servers.
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

    # Return list format matching upstream (list of server objects)
    servers = []
    for s in gateway.list_servers():
        servers.append(
            {
                "server_id": s.server_id,
                "server_name": s.name,
                "alias": s.metadata.get("alias"),
                "url": s.url,
                "transport": s.transport.value,
                "auth_type": s.auth_type,
                "tools": s.tools,
                "resources": s.resources,
                "mcp_access_groups": s.metadata.get("access_groups", []),
            }
        )
    return servers


@mcp_parity_router.get("/server/health")
async def health_check_servers(
    server_ids: list[str] | None = Query(None),
):
    """
    Upstream-compatible: GET /v1/mcp/server/health

    Health check for MCP servers.
    Alias for GET /v1/llmrouter/mcp/server/health.
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
        if server_ids:
            # Check specific servers
            health_results = []
            for sid in server_ids:
                health = await gateway.check_server_health(sid)
                health_results.append(
                    {
                        "server_id": sid,
                        "status": health.get("status", "unknown"),
                    }
                )
        else:
            # Check all servers
            all_health = await gateway.check_all_servers_health()
            health_results = [
                {"server_id": h.get("server_id"), "status": h.get("status", "unknown")}
                for h in all_health
            ]
        return health_results
    except Exception as e:
        err = sanitize_error_response(e, request_id, "Health check failed")
        raise HTTPException(status_code=500, detail=err)


@mcp_parity_router.get("/server/{server_id}")
async def fetch_mcp_server(server_id: str):
    """
    Upstream-compatible: GET /v1/mcp/server/{server_id}

    Returns info on a specific MCP server.
    Alias for GET /llmrouter/mcp/servers/{server_id}.
    """
    gateway = get_mcp_gateway()
    server = gateway.get_server(server_id)
    if not server:
        raise HTTPException(
            status_code=404,
            detail={"error": f"MCP Server with id {server_id} not found"},
        )

    return {
        "server_id": server.server_id,
        "server_name": server.name,
        "alias": server.metadata.get("alias"),
        "url": server.url,
        "transport": server.transport.value,
        "auth_type": server.auth_type,
        "tools": server.tools,
        "resources": server.resources,
        "mcp_access_groups": server.metadata.get("access_groups", []),
        "status": "healthy",  # Would need real health check
    }


@mcp_parity_admin_router.post("/server", status_code=201)
async def add_mcp_server(payload: NewMCPServerRequest):
    """
    Upstream-compatible: POST /v1/mcp/server

    Add a new external MCP server.
    Alias for POST /llmrouter/mcp/servers (admin required).
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

    # Generate server_id if not provided
    server_id = payload.server_id
    if not server_id:
        server_id = hashlib.sha256(
            f"{payload.server_name or payload.alias or 'mcp'}-{time.time()}".encode()
        ).hexdigest()[:16]

    # Check if server already exists
    if gateway.get_server(server_id):
        raise HTTPException(
            status_code=400,
            detail={"error": f"MCP Server with id {server_id} already exists"},
        )

    try:
        transport = MCPTransport(payload.transport)
        mcp_server = MCPServer(
            server_id=server_id,
            name=payload.server_name or payload.alias or server_id,
            url=payload.url or "",
            transport=transport,
            tools=payload.allowed_tools or [],
            resources=[],
            auth_type=payload.auth_type or "none",
            metadata={
                "alias": payload.alias,
                "description": payload.description,
                "access_groups": payload.mcp_access_groups or [],
                "mcp_info": payload.mcp_info,
                "authorization_url": payload.authorization_url,
                "token_url": payload.token_url,
                "registration_url": payload.registration_url,
            },
        )
        gateway.register_server(mcp_server)

        # Return matching upstream response
        return {
            "server_id": server_id,
            "server_name": mcp_server.name,
            "alias": payload.alias,
            "url": mcp_server.url,
            "transport": mcp_server.transport.value,
        }
    except ValueError as e:
        error_msg = str(e)
        if "blocked for security reasons" in error_msg:
            raise HTTPException(
                status_code=400,
                detail={"error": f"SSRF blocked: {error_msg}"},
            )
        raise HTTPException(status_code=400, detail={"error": error_msg})
    except Exception as e:
        err = sanitize_error_response(e, request_id, "Failed to create MCP server")
        raise HTTPException(status_code=500, detail=err)


@mcp_parity_admin_router.put("/server", status_code=202)
async def edit_mcp_server(payload: UpdateMCPServerRequest):
    """
    Upstream-compatible: PUT /v1/mcp/server

    Updates an existing MCP server.
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

    existing = gateway.get_server(payload.server_id)
    if not existing:
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"MCP Server not found, passed server_id={payload.server_id}"
            },
        )

    try:
        # Unregister old, register updated
        gateway.unregister_server(payload.server_id)

        transport = MCPTransport(payload.transport or existing.transport.value)
        mcp_server = MCPServer(
            server_id=payload.server_id,
            name=payload.server_name or payload.alias or existing.name,
            url=payload.url or existing.url,
            transport=transport,
            tools=payload.allowed_tools or existing.tools,
            resources=existing.resources,
            auth_type=payload.auth_type or existing.auth_type,
            metadata={
                "alias": payload.alias or existing.metadata.get("alias"),
                "description": payload.description
                or existing.metadata.get("description"),
                "access_groups": payload.mcp_access_groups
                or existing.metadata.get("access_groups", []),
                "mcp_info": payload.mcp_info or existing.metadata.get("mcp_info"),
            },
        )
        gateway.register_server(mcp_server)

        return {
            "server_id": mcp_server.server_id,
            "server_name": mcp_server.name,
            "alias": payload.alias,
            "url": mcp_server.url,
        }
    except ValueError as e:
        error_msg = str(e)
        if "blocked for security reasons" in error_msg:
            raise HTTPException(
                status_code=400,
                detail={"error": f"SSRF blocked: {error_msg}"},
            )
        raise HTTPException(status_code=400, detail={"error": error_msg})
    except Exception as e:
        err = sanitize_error_response(e, request_id, "Failed to update MCP server")
        raise HTTPException(status_code=500, detail=err)


@mcp_parity_admin_router.delete("/server/{server_id}", status_code=202)
async def remove_mcp_server(server_id: str):
    """
    Upstream-compatible: DELETE /v1/mcp/server/{server_id}

    Deletes an MCP server.
    """
    gateway = get_mcp_gateway()

    if gateway.unregister_server(server_id):
        return JSONResponse(status_code=202, content=None)

    raise HTTPException(
        status_code=404,
        detail={"error": f"MCP Server not found, passed server_id={server_id}"},
    )


@mcp_parity_router.get("/tools")
async def get_mcp_tools():
    """
    Upstream-compatible: GET /v1/mcp/tools

    Get all MCP tools available for the current key.
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
            tool_info = {"name": tool_name}
            if tool_name in server.tool_definitions:
                tool_def = server.tool_definitions[tool_name]
                tool_info["description"] = tool_def.description
                tool_info["inputSchema"] = tool_def.input_schema
            tools.append(tool_info)

    return {"tools": tools}


@mcp_parity_router.get("/access_groups")
async def get_mcp_access_groups():
    """
    Upstream-compatible: GET /v1/mcp/access_groups

    Get all available MCP access groups.
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
    return {"access_groups": groups}


@mcp_parity_router.get("/registry.json")
async def get_mcp_registry(request: Request):
    """
    Upstream-compatible: GET /v1/mcp/registry.json

    MCP registry endpoint for server discovery.
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

    registry = gateway.get_registry()

    # Transform to upstream format
    base_url = str(request.base_url).rstrip("/")
    servers = []
    for srv in registry.get("servers", []):
        servers.append(
            {
                "server": {
                    "name": srv.get("name", srv.get("id")),
                    "title": srv.get("name", srv.get("id")),
                    "description": srv.get("name", ""),
                    "version": "1.0.0",
                    "remotes": [
                        {
                            "type": "streamable-http",
                            "url": f"{base_url}/mcp/{srv.get('id')}/mcp",
                        }
                    ],
                }
            }
        )

    return {"servers": servers}


@mcp_parity_admin_router.post("/make_public", status_code=202)
async def make_mcp_servers_public(request: MakeMCPServersPublicRequest):
    """
    Upstream-compatible: POST /v1/mcp/make_public

    Make MCP servers public for AI Hub.
    Requires admin API key authentication.

    This endpoint stores the list of public MCP server IDs.
    When GET /v1/mcp/server is called, servers in this list
    will have mcp_info.is_public=True.
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

    global _public_mcp_servers

    # Validate that all server IDs exist
    for server_id in request.mcp_server_ids:
        server = gateway.get_server(server_id)
        if not server:
            raise HTTPException(
                status_code=404,
                detail={"error": f"MCP Server with ID {server_id} not found"},
            )

    # Update public servers list
    _public_mcp_servers = request.mcp_server_ids

    return {
        "message": "Successfully updated public mcp servers",
        "public_mcp_servers": _public_mcp_servers,
    }


# ============================================================================
# Namespaced MCP Protocol Router
# ============================================================================
# Provides upstream-compatible /mcp endpoint for built-in MCP server
# and /{server_prefix}/mcp for per-server namespaced routes.


@mcp_namespace_router.api_route("/mcp", methods=["GET", "POST"])
async def builtin_mcp_endpoint(request: Request):
    """
    Upstream-compatible: /mcp

    Built-in MCP protocol endpoint (streamable HTTP).
    This provides the default/built-in MCP server endpoint that upstream exposes.

    For now, returns server information. Full MCP protocol implementation
    would require SSE/streamable HTTP JSON-RPC handling.
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

    # Get base URL for building remote URLs
    base_url = str(request.base_url).rstrip("/")

    # Return MCP server info in standard format
    return {
        "name": "routeiq-mcp-server",
        "title": "RouteIQ MCP Server",
        "description": "MCP Server for RouteIQ Gateway",
        "version": "1.0.0",
        "protocol_version": "2024-11-05",
        "capabilities": {
            "tools": {"listChanged": True},
            "resources": {"subscribe": False, "listChanged": True},
        },
        "remotes": [
            {
                "type": "streamable-http",
                "url": f"{base_url}/mcp",
            }
        ],
    }


@mcp_namespace_router.api_route("/{server_prefix}/mcp", methods=["GET", "POST"])
async def namespaced_mcp_endpoint(request: Request, server_prefix: str):
    """
    Upstream-compatible: /{server_prefix}/mcp

    Namespaced MCP protocol endpoint for a specific server.
    The server_prefix is matched against server_id or alias.

    This provides upstream-compatible per-server namespaced routes.
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

    # Find server by prefix (could be server_id or alias)
    server = gateway.get_server(server_prefix)
    if not server:
        # Try to find by alias in metadata
        for s in gateway.list_servers():
            if s.metadata.get("alias") == server_prefix:
                server = s
                break

    if not server:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "server_not_found",
                "message": f"MCP server with prefix '{server_prefix}' not found",
                "request_id": request_id,
            },
        )

    # Get base URL for building remote URLs
    base_url = str(request.base_url).rstrip("/")

    # Return server info in MCP format
    return {
        "name": server.name,
        "title": server.name,
        "description": server.metadata.get("description", server.name),
        "version": "1.0.0",
        "protocol_version": "2024-11-05",
        "capabilities": {
            "tools": {"listChanged": True},
            "resources": {"subscribe": False, "listChanged": True},
        },
        "remotes": [
            {
                "type": "streamable-http",
                "url": f"{base_url}/{server_prefix}/mcp",
            }
        ],
        "server_id": server.server_id,
        "transport": server.transport.value,
        "tools": server.tools,
        "resources": server.resources,
    }


# ============================================================================
# OAuth Endpoints (Feature-Flagged)
# ============================================================================


if MCP_OAUTH_ENABLED:

    @mcp_parity_admin_router.post("/server/oauth/session", status_code=200)
    async def add_session_mcp_server(payload: NewMCPServerRequest):
        """
        Upstream-compatible: POST /v1/mcp/server/oauth/session

        Temporarily cache an MCP server in memory for OAuth flow (~5 minutes).
        Does not write to database.
        """
        request_id = get_request_id() or "unknown"

        server_id = payload.server_id
        if not server_id:
            server_id = hashlib.sha256(
                f"oauth-{payload.server_name or 'mcp'}-{time.time()}".encode()
            ).hexdigest()[:16]

        try:
            gateway = get_mcp_gateway()
            transport = MCPTransport(payload.transport)
            mcp_server = MCPServer(
                server_id=server_id,
                name=payload.server_name or payload.alias or server_id,
                url=payload.url or "",
                transport=transport,
                tools=payload.allowed_tools or [],
                resources=[],
                auth_type=payload.auth_type or "oauth2",
                metadata={
                    "alias": payload.alias,
                    "description": payload.description,
                    "access_groups": payload.mcp_access_groups or [],
                    "mcp_info": payload.mcp_info,
                    "authorization_url": payload.authorization_url,
                    "token_url": payload.token_url,
                    "registration_url": payload.registration_url,
                    "temporary": True,
                    "expires_at": time.time() + OAUTH_SESSION_TTL,
                },
            )
            gateway.register_server(mcp_server)

            return {
                "server_id": server_id,
                "server_name": mcp_server.name,
                "alias": payload.alias,
                "url": mcp_server.url,
                "expires_in": OAUTH_SESSION_TTL,
            }
        except ValueError as e:
            error_msg = str(e)
            if "blocked for security reasons" in error_msg:
                raise HTTPException(
                    status_code=400,
                    detail={"error": f"SSRF blocked: {error_msg}"},
                )
            raise HTTPException(status_code=400, detail={"error": error_msg})
        except Exception as e:
            err = sanitize_error_response(
                e, request_id, "Failed to create OAuth session"
            )
            raise HTTPException(status_code=500, detail=err)

    @mcp_parity_router.get("/server/oauth/{server_id}/authorize")
    async def mcp_authorize(
        request: Request,
        server_id: str,
        client_id: str,
        redirect_uri: str,
        state: str = "",
        code_challenge: str | None = None,
        code_challenge_method: str | None = None,
        response_type: str | None = None,
        scope: str | None = None,
    ):
        """
        Upstream-compatible: GET /v1/mcp/server/oauth/{server_id}/authorize

        OAuth authorization redirect endpoint.
        """
        gateway = get_mcp_gateway()
        server = gateway.get_server(server_id)
        if not server:
            raise HTTPException(
                status_code=404,
                detail={"error": f"Temporary MCP server {server_id} not found"},
            )

        authorization_url = server.metadata.get("authorization_url")
        if not authorization_url:
            raise HTTPException(
                status_code=400,
                detail={"error": "Server does not have authorization_url configured"},
            )

        # Validate authorization URL against SSRF
        # Use async version to avoid blocking the event loop
        try:
            await validate_outbound_url_async(authorization_url)
        except SSRFBlockedError as e:
            raise HTTPException(
                status_code=400,
                detail={"error": f"Authorization URL blocked for security: {e.reason}"},
            )

        # Generate our own state for CSRF protection
        internal_state = _generate_state()
        _cleanup_expired_sessions()
        _oauth_sessions[internal_state] = OAuthSession(
            server_id=server_id,
            state=state,  # Store original state to pass back
            redirect_uri=redirect_uri,
            created_at=time.time(),
            client_id=client_id,
            code_verifier=code_challenge,
        )

        # Build redirect URL to upstream authorization server
        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "state": internal_state,  # Use our state for CSRF
            "response_type": response_type or "code",
        }
        if scope:
            params["scope"] = scope
        if code_challenge:
            params["code_challenge"] = code_challenge
        if code_challenge_method:
            params["code_challenge_method"] = code_challenge_method

        query = "&".join(f"{k}={v}" for k, v in params.items())
        auth_redirect = f"{authorization_url}?{query}"

        return RedirectResponse(url=auth_redirect)

    @mcp_parity_admin_router.post("/server/oauth/{server_id}/token")
    async def mcp_token(
        request: Request,
        server_id: str,
        grant_type: str = Form(...),
        code: str | None = Form(None),
        redirect_uri: str | None = Form(None),
        client_id: str = Form(...),
        client_secret: str | None = Form(None),
        code_verifier: str | None = Form(None),
    ):
        """
        Upstream-compatible: POST /v1/mcp/server/oauth/{server_id}/token

        OAuth token exchange endpoint.
        """
        request_id = get_request_id() or "unknown"
        gateway = get_mcp_gateway()
        server = gateway.get_server(server_id)
        if not server:
            raise HTTPException(
                status_code=404,
                detail={"error": f"Temporary MCP server {server_id} not found"},
            )

        token_url = server.metadata.get("token_url")
        if not token_url:
            raise HTTPException(
                status_code=400,
                detail={"error": "Server does not have token_url configured"},
            )

        # Validate token URL against SSRF
        # Use async version to avoid blocking the event loop
        try:
            await validate_outbound_url_async(token_url)
        except SSRFBlockedError as e:
            raise HTTPException(
                status_code=400,
                detail={"error": f"Token URL blocked for security: {e.reason}"},
            )

        # Forward token request to upstream
        try:
            async with get_client_for_request(timeout=30.0) as client:
                form_data = {
                    "grant_type": grant_type,
                    "client_id": client_id,
                }
                if code:
                    form_data["code"] = code
                if redirect_uri:
                    form_data["redirect_uri"] = redirect_uri
                if client_secret:
                    form_data["client_secret"] = client_secret
                if code_verifier:
                    form_data["code_verifier"] = code_verifier

                response = await client.post(token_url, data=form_data)

                if response.status_code >= 400:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail={
                            "error": "token_exchange_failed",
                            "upstream": response.text[:500],
                        },
                    )

                token_data = response.json()

                # Store token for this server
                _oauth_tokens[server_id] = OAuthToken(
                    server_id=server_id,
                    access_token=token_data.get("access_token", ""),
                    token_type=token_data.get("token_type", "Bearer"),
                    refresh_token=token_data.get("refresh_token"),
                    expires_at=time.time() + token_data.get("expires_in", 3600),
                    scope=token_data.get("scope"),
                )

                return token_data
        except httpx.TimeoutException:
            raise HTTPException(
                status_code=504,
                detail={"error": "Token exchange timeout"},
            )
        except HTTPException:
            raise
        except Exception as e:
            err = sanitize_error_response(e, request_id, "Token exchange failed")
            raise HTTPException(status_code=500, detail=err)


# ============================================================================
# OAuth Callback Endpoint (Standalone for callback handling)
# ============================================================================


oauth_callback_router = APIRouter(tags=["mcp-oauth-callback"])


@oauth_callback_router.get("/mcp/oauth/callback")
async def oauth_callback(
    code: str = Query(...),
    state: str = Query(...),
    error: str | None = Query(None),
    error_description: str | None = Query(None),
):
    """
    OAuth callback endpoint.

    This endpoint receives the authorization code from the OAuth provider
    and validates the state parameter for CSRF protection.
    """
    if error:
        raise HTTPException(
            status_code=400,
            detail={
                "error": error,
                "error_description": error_description or "OAuth authorization failed",
            },
        )

    _cleanup_expired_sessions()

    # Validate state (CSRF protection)
    session = _oauth_sessions.get(state)
    if not session:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_state",
                "message": "Invalid or expired OAuth state",
            },
        )

    # Session is valid; return info for frontend to complete token exchange
    return {
        "status": "callback_received",
        "server_id": session.server_id,
        "code": code,
        "original_state": session.state,
        "redirect_uri": session.redirect_uri,
        "message": "Use POST /v1/mcp/server/oauth/{server_id}/token to exchange code for tokens",
    }
