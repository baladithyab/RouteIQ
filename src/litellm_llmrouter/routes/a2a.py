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

from fastapi import Depends, HTTPException

from ..a2a_gateway import A2AError
from ..auth import get_request_id, sanitize_error_response
from ..rbac import requires_permission, PERMISSION_A2A_AGENT_WRITE
from ..audit import audit_log, AuditAction, AuditOutcome, AuditWriteError
from ..mcp_gateway import MCPTransport
from ..url_security import validate_outbound_url_async, SSRFBlockedError
from .models import AgentRegistration
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
