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
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel

from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

from .auth import admin_api_key_auth, get_request_id, sanitize_error_response
from .mcp_gateway import (
    MCPServer,
    MCPTransport,
    get_mcp_gateway,
)
from .url_security import SSRFBlockedError, validate_outbound_url

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


# ============================================================================
# MCP REST Endpoints (/mcp-rest/*)
# ============================================================================


@mcp_rest_router.get("/tools/list")
async def list_tool_rest_api(
    request: Request,
    server_id: str | None = Query(None, description="The server id to list tools for"),
):
    """
    Upstream-compatible: GET /mcp-rest/tools/list

    List all available tools with server mcp_info.
    """
    gateway = get_mcp_gateway()
    if not gateway.is_enabled():
        return {
            "tools": [],
            "error": "mcp_gateway_disabled",
            "message": "MCP Gateway is not enabled",
        }

    tools = []
    try:
        if server_id:
            server = gateway.get_server(server_id)
            if not server:
                return {
                    "tools": [],
                    "error": "server_not_found",
                    "message": f"Server with id {server_id} not found",
                }
            for tool_name in server.tools:
                tool_info = {
                    "name": tool_name,
                    "description": "",
                    "inputSchema": {},
                    "mcp_info": server.metadata.get("mcp_info"),
                }
                if tool_name in server.tool_definitions:
                    tool_def = server.tool_definitions[tool_name]
                    tool_info["description"] = tool_def.description
                    tool_info["inputSchema"] = tool_def.input_schema
                tools.append(tool_info)
        else:
            for server in gateway.list_servers():
                for tool_name in server.tools:
                    tool_info = {
                        "name": tool_name,
                        "description": "",
                        "inputSchema": {},
                        "mcp_info": server.metadata.get("mcp_info"),
                    }
                    if tool_name in server.tool_definitions:
                        tool_def = server.tool_definitions[tool_name]
                        tool_info["description"] = tool_def.description
                        tool_info["inputSchema"] = tool_def.input_schema
                    tools.append(tool_info)

        return {
            "tools": tools,
            "error": None,
            "message": "Successfully retrieved tools",
        }
    except Exception as e:
        return {
            "tools": [],
            "error": "unexpected_error",
            "message": f"An unexpected error occurred: {str(e)}",
        }


@mcp_rest_router.post("/tools/call")
async def call_tool_rest_api(request: Request):
    """
    Upstream-compatible: POST /mcp-rest/tools/call

    REST API to call a specific MCP tool with provided arguments.
    """
    gateway = get_mcp_gateway()
    if not gateway.is_enabled():
        raise HTTPException(
            status_code=404,
            detail={
                "error": "mcp_gateway_disabled",
                "message": "MCP Gateway is not enabled",
            },
        )

    try:
        data = await request.json()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_json",
                "message": "Request body must be valid JSON",
            },
        )

    server_id = data.get("server_id")
    tool_name = data.get("name")
    arguments = data.get("arguments", {})

    if not server_id:
        raise HTTPException(
            status_code=400,
            detail={"error": "missing_parameter", "message": "server_id is required"},
        )
    if not tool_name:
        raise HTTPException(
            status_code=400,
            detail={"error": "missing_parameter", "message": "name is required"},
        )

    # Verify server exists
    server = gateway.get_server(server_id)
    if not server:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "access_denied",
                "message": f"Server {server_id} not found or not accessible",
            },
        )

    if not gateway.is_tool_invocation_enabled():
        raise HTTPException(
            status_code=501,
            detail={
                "error": "tool_invocation_disabled",
                "message": "Tool invocation is disabled. Set LLMROUTER_ENABLE_MCP_TOOL_INVOCATION=true",
            },
        )

    result = await gateway.invoke_tool(tool_name, arguments)
    if not result.success:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "tool_invocation_failed",
                "message": result.error or "Tool invocation failed",
            },
        )

    return {
        "status": "success",
        "tool_name": result.tool_name,
        "server_id": result.server_id,
        "result": result.result,
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
        try:
            validate_outbound_url(authorization_url)
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
        try:
            validate_outbound_url(token_url)
        except SSRFBlockedError as e:
            raise HTTPException(
                status_code=400,
                detail={"error": f"Token URL blocked for security: {e.reason}"},
            )

        # Forward token request to upstream
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
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

    @mcp_parity_admin_router.post("/server/oauth/{server_id}/register")
    async def mcp_register(request: Request, server_id: str):
        """
        Upstream-compatible: POST /v1/mcp/server/oauth/{server_id}/register

        OAuth dynamic client registration.
        """
        request_id = get_request_id() or "unknown"
        gateway = get_mcp_gateway()
        server = gateway.get_server(server_id)
        if not server:
            raise HTTPException(
                status_code=404,
                detail={"error": f"Temporary MCP server {server_id} not found"},
            )

        registration_url = server.metadata.get("registration_url")
        if not registration_url:
            raise HTTPException(
                status_code=400,
                detail={"error": "Server does not have registration_url configured"},
            )

        # Validate registration URL against SSRF
        try:
            validate_outbound_url(registration_url)
        except SSRFBlockedError as e:
            raise HTTPException(
                status_code=400,
                detail={"error": f"Registration URL blocked for security: {e.reason}"},
            )

        try:
            data = await request.json()
        except Exception:
            data = {}

        # Forward registration request to upstream
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(registration_url, json=data)

                if response.status_code >= 400:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail={
                            "error": "registration_failed",
                            "upstream": response.text[:500],
                        },
                    )

                return response.json()
        except httpx.TimeoutException:
            raise HTTPException(
                status_code=504,
                detail={"error": "Client registration timeout"},
            )
        except HTTPException:
            raise
        except Exception as e:
            err = sanitize_error_response(e, request_id, "Client registration failed")
            raise HTTPException(status_code=500, detail=err)


# ============================================================================
# Protocol Proxy (Feature-Flagged)
# ============================================================================


if MCP_PROTOCOL_PROXY_ENABLED:

    @mcp_proxy_router.api_route(
        "/{server_id}/{path:path}", methods=["GET", "POST", "PUT", "DELETE"]
    )
    async def mcp_protocol_proxy(request: Request, server_id: str, path: str):
        """
        MCP Protocol Proxy: /mcp/{server_id}/*

        Proxies requests to registered MCP servers.
        Feature-flagged via MCP_PROTOCOL_PROXY_ENABLED.

        Enforces:
        - SSRF allow/deny policy
        - Strict timeouts
        - Admin authentication

        """
        request_id = get_request_id() or "unknown"
        gateway = get_mcp_gateway()

        server = gateway.get_server(server_id)
        if not server:
            raise HTTPException(
                status_code=404,
                detail={"error": f"MCP server {server_id} not found"},
            )

        if not server.url:
            raise HTTPException(
                status_code=400,
                detail={"error": "MCP server does not have a URL configured"},
            )

        # Validate target URL against SSRF
        target_base = server.url.rstrip("/")
        target_url = f"{target_base}/{path}"

        try:
            validate_outbound_url(target_url)
        except SSRFBlockedError as e:
            raise HTTPException(
                status_code=400,
                detail={"error": f"Target URL blocked for security: {e.reason}"},
            )

        # Build timeout
        timeout = httpx.Timeout(
            connect=PROTOCOL_PROXY_CONNECT_TIMEOUT,
            read=PROTOCOL_PROXY_READ_TIMEOUT,
            write=PROTOCOL_PROXY_CONNECT_TIMEOUT,
            pool=PROTOCOL_PROXY_CONNECT_TIMEOUT,
        )

        # Forward headers (excluding hop-by-hop)
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
        headers = {
            k: v for k, v in request.headers.items() if k.lower() not in hop_by_hop
        }

        # Add auth header if configured
        if server.auth_type == "bearer_token" and server.metadata.get("auth_token"):
            headers["Authorization"] = f"Bearer {server.metadata['auth_token']}"
        elif server.auth_type == "api_key" and server.metadata.get("api_key"):
            headers["X-API-Key"] = server.metadata["api_key"]

        # Read request body
        body = await request.body()

        try:
            # Check if SSE is expected
            accept = request.headers.get("accept", "")
            is_sse = "text/event-stream" in accept

            async with httpx.AsyncClient(timeout=timeout) as client:
                if is_sse:
                    # Stream SSE response
                    async def stream_sse():
                        async with client.stream(
                            request.method,
                            target_url,
                            content=body,
                            headers=headers,
                            params=dict(request.query_params),
                        ) as response:
                            async for chunk in response.aiter_bytes():
                                yield chunk

                    return StreamingResponse(
                        stream_sse(),
                        media_type="text/event-stream",
                        headers={
                            "Cache-Control": "no-cache",
                            "X-Accel-Buffering": "no",
                        },
                    )
                else:
                    # Regular request
                    response = await client.request(
                        request.method,
                        target_url,
                        content=body,
                        headers=headers,
                        params=dict(request.query_params),
                    )

                    return JSONResponse(
                        status_code=response.status_code,
                        content=(
                            response.json()
                            if response.headers.get("content-type", "").startswith(
                                "application/json"
                            )
                            else {"raw": response.text[:1000]}
                        ),
                    )
        except httpx.TimeoutException:
            raise HTTPException(
                status_code=504,
                detail={"error": "Proxy timeout connecting to MCP server"},
            )
        except httpx.ConnectError:
            raise HTTPException(
                status_code=502,
                detail={"error": "Failed to connect to MCP server"},
            )
        except Exception as e:
            err = sanitize_error_response(e, request_id, "Proxy request failed")
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
