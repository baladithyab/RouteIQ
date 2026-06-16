"""
A2A Gateway Convenience Endpoints (/a2a/agents).

These are thin wrappers around LiteLLM's global_agent_registry for convenience.
The main A2A functionality is provided by LiteLLM's built-in endpoints:
- POST /v1/agents - Create agent (DB-backed)
- GET /v1/agents - List agents (DB-backed)
- DELETE /v1/agents/{agent_id} - Delete agent (DB-backed)
- POST /a2a/{agent_id} - Invoke agent (A2A JSON-RPC protocol)
- POST /a2a/{agent_id}/message/stream - Streaming alias (proxies to canonical)
"""

from fastapi import Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.convertors import Convertor, register_url_convertor

from ..a2a_gateway import (
    A2AAgent,
    A2AError,
    JSONRPCRequest,
    get_a2a_gateway,
)
from ..auth import get_request_id, sanitize_error_response
from ..rbac import requires_permission, PERMISSION_A2A_AGENT_WRITE
from ..audit import AuditAction, AuditOutcome
from ..url_security import validate_outbound_url_async, SSRFBlockedError
from .models import AgentRegistration
from . import admin_router, llmrouter_router, handle_audit_write


# Reserved collection literals under /a2a that must NOT be captured by the
# parametric invocation route POST /a2a/{agent_id}. Without this, a request to
# the admin CRUD literal POST /a2a/agents would be shadowed by the invocation
# route (Starlette matches in include order; llmrouter_router precedes
# admin_router). A custom path convertor with a negative-lookahead regex makes
# the literal win while keeping the documented /a2a/{agent_id} contract.
_A2A_RESERVED_LITERALS = ("agents",)


class _A2AAgentIdConvertor(Convertor):
    """Path convertor for A2A agent ids that excludes reserved collection
    literals (e.g. ``agents``) so the admin CRUD endpoints are reachable."""

    regex = r"(?!(?:agents)$)[^/]+"

    def convert(self, value: str) -> str:
        return value

    def to_string(self, value: str) -> str:
        return value


# Idempotent route-setup registration (not a behavioural side effect): registers
# the ``a2a_agent_id`` convertor used by the invocation routes below.
try:
    register_url_convertor("a2a_agent_id", _A2AAgentIdConvertor())
except Exception:  # pragma: no cover - already registered
    pass


# Read-only endpoint - user auth is sufficient
@llmrouter_router.get("/a2a/agents")
async def list_a2a_agents_convenience():
    """
    List all registered A2A agents.

    Lists the unified A2A registry: agents from the native A2AGateway merged
    with LiteLLM's global_agent_registry (deduplicated by agent_id). The native
    registry is the source of truth for invocation.

    For full functionality, use GET /v1/agents (DB-backed, supports filtering).
    """
    request_id = get_request_id() or "unknown"
    merged: dict[str, dict] = {}

    # Native A2AGateway registry first (source of truth for invocation).
    try:
        for a in get_a2a_gateway().list_agents():
            merged[a.agent_id] = {
                "agent_id": a.agent_id,
                "agent_name": a.name,
                "description": a.description,
                "url": a.url,
            }
    except Exception:
        pass

    # Merge in LiteLLM's registry (DB/config-backed) without overwriting native.
    try:
        from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry

        for a in global_agent_registry.get_agent_list():
            if a.agent_id in merged:
                continue
            card = a.agent_card_params or {}
            merged[a.agent_id] = {
                "agent_id": a.agent_id,
                "agent_name": a.agent_name,
                "description": card.get("description", "") if card else "",
                "url": card.get("url", "") if card else "",
            }
    except ImportError:
        # LiteLLM not initialized — return whatever the native registry has.
        if not merged:
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

    return {"agents": list(merged.values())}


# =============================================================================
# A2A Native Invocation (JSON-RPC 2.0 over HTTP + SSE)
# =============================================================================


@llmrouter_router.post("/a2a/{agent_id:a2a_agent_id}")
async def invoke_a2a_agent(agent_id: str, request: Request):
    """
    Invoke an A2A agent via JSON-RPC 2.0 (the A2A protocol data plane).

    This is the native invocation route backed by the unified A2AGateway
    registry (native registrations + LiteLLM global_agent_registry fallback).
    Streaming methods (``message/stream`` / ``tasks/sendSubscribe``) are served
    as Server-Sent Events when the client sends ``Accept: text/event-stream``.

    Request body is a JSON-RPC 2.0 envelope:
    ```json
    {"jsonrpc": "2.0", "id": 1, "method": "message/send",
     "params": {"message": {...}}}
    ```

    Security: outbound agent URLs are SSRF-validated by the gateway at
    invocation time (with DNS).
    """
    request_id = get_request_id() or "unknown"
    gateway = get_a2a_gateway()

    if not gateway.is_enabled():
        raise HTTPException(
            status_code=404,
            detail={
                "error": "a2a_gateway_disabled",
                "message": "A2A Gateway is not enabled",
                "request_id": request_id,
            },
        )

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_json",
                "message": "Request body must be a JSON-RPC 2.0 object",
                "request_id": request_id,
            },
        )

    if not isinstance(body, dict) or "method" not in body:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_request",
                "message": "Missing required JSON-RPC field: method",
                "request_id": request_id,
            },
        )

    rpc_request = JSONRPCRequest(
        method=body["method"],
        params=body.get("params", {}) or {},
        id=body.get("id"),
        jsonrpc=body.get("jsonrpc", "2.0"),
    )

    # Streaming dispatch: SSE when requested or for streaming methods.
    accept = request.headers.get("accept", "")
    wants_stream = "text/event-stream" in accept or gateway._is_streaming_method(
        rpc_request.method
    )

    if wants_stream:
        return StreamingResponse(
            gateway.stream_agent_response_sse(agent_id, rpc_request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    response = await gateway.invoke_agent(agent_id, rpc_request)
    return response.to_dict()


@llmrouter_router.post("/a2a/{agent_id}/message/stream")
async def stream_a2a_agent(agent_id: str, request: Request):
    """
    Streaming alias for A2A agent invocation (always SSE).

    Equivalent to POST /a2a/{agent_id} with ``Accept: text/event-stream``.
    Backed by the same unified A2AGateway registry.
    """
    request_id = get_request_id() or "unknown"
    gateway = get_a2a_gateway()

    if not gateway.is_enabled():
        raise HTTPException(
            status_code=404,
            detail={
                "error": "a2a_gateway_disabled",
                "message": "A2A Gateway is not enabled",
                "request_id": request_id,
            },
        )

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_json",
                "message": "Request body must be a JSON-RPC 2.0 object",
                "request_id": request_id,
            },
        )

    rpc_request = JSONRPCRequest(
        method=body.get("method", "message/stream"),
        params=body.get("params", {}) or {},
        id=body.get("id"),
        jsonrpc=body.get("jsonrpc", "2.0"),
    )

    return StreamingResponse(
        gateway.stream_agent_response_sse(agent_id, rpc_request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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

        # Register with LiteLLM's in-memory registry
        global_agent_registry.register_agent(agent_config=agent_response)

        # Registry unification: mirror into the native A2AGateway so the
        # invocation route (POST /a2a/{agent_id}) resolves the same agent.
        # This removes the "two registries" split (CRUD wrote LiteLLM only,
        # invocation read the native gateway only). Best-effort: a disabled
        # native gateway is a no-op.
        gateway = get_a2a_gateway()
        if gateway.is_enabled():
            try:
                gateway.register_agent(
                    A2AAgent(
                        agent_id=agent_id,
                        name=agent.agent_name,
                        description=agent.description,
                        url=agent.url,
                        capabilities=agent.capabilities,
                    )
                )
            except ValueError:
                # SSRF/validation already enforced above; ignore native dupe errs
                pass

        # Audit log the success
        await handle_audit_write(
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
            "note": "Agent registered in the unified A2A registry. For HA persistence, use POST /v1/agents.",
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

    # Registry unification: also remove from the native A2AGateway registry so
    # the invocation route stops resolving a deleted agent. Best-effort.
    try:
        get_a2a_gateway().unregister_agent(agent_id)
    except Exception:
        pass

    try:
        from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry

        # Get agent by ID first to find its name (needed for deregister_agent)
        agent = global_agent_registry.get_agent_by_id(agent_id)
        if agent:
            global_agent_registry.deregister_agent(agent_name=agent.agent_name)

            # Audit log the success
            await handle_audit_write(
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
            await handle_audit_write(
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
                "message": f"Agent '{agent_id}' not found",
                "code": A2AError.AGENT_NOT_FOUND["code"],
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
