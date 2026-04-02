"""
Hot Reload, Config Sync, Governance, Usage Policy, and Guardrail Policy Endpoints.

- POST /llmrouter/reload - Trigger config reload (sync manager)
- POST /config/reload - Trigger config reload (hot reload manager)
- GET /config/sync/status - Get config sync status
- GET /router/info - Get routing configuration info
- /api/v1/routeiq/governance/* - Workspace & key governance CRUD
- /api/v1/routeiq/governance/policies/* - Usage policy CRUD & counters
- /api/v1/routeiq/governance/guardrails/* - Guardrail policy CRUD
"""

from fastapi import Depends, HTTPException

import dataclasses
from typing import Optional

from ..auth import get_request_id, sanitize_error_response
from ..rbac import requires_permission, PERMISSION_SYSTEM_CONFIG_RELOAD
from ..audit import AuditAction, AuditOutcome
from ..hot_reload import get_hot_reload_manager
from ..config_sync import get_sync_manager, get_config_sync_status
from ..governance import (
    WorkspaceConfig,
    KeyGovernance,
    get_governance_engine,
)
from ..usage_policies import (
    UsagePolicy,
    get_usage_policy_engine,
)
from ..guardrail_policies import (
    GuardrailPolicy,
    GuardrailPhase,
    get_guardrail_policy_engine,
)
from fastapi import Request

from .models import ReloadRequest
from . import admin_router, llmrouter_router, health_router, handle_audit_write


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
        manager = get_hot_reload_manager()
        force_sync = request.force_sync if request else False
        result = manager.reload_config(force_sync=force_sync)

        # Audit log the success
        await handle_audit_write(
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
        await handle_audit_write(
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


# =============================================================================
# Cache Admin Endpoints
# =============================================================================


@admin_router.get("/admin/cache/stats")
async def get_cache_stats(
    rbac_info: dict = Depends(requires_permission(PERMISSION_SYSTEM_CONFIG_RELOAD)),
):
    """
    Get cache statistics: hit/miss counts, entry count, configuration.

    Requires admin API key authentication or user with system.config.reload permission.
    """
    from ..semantic_cache import get_cache_manager

    manager = get_cache_manager()
    if manager is None:
        return {"enabled": False}
    return manager.get_stats()


@admin_router.post("/admin/cache/flush")
async def flush_cache(
    prefix: str | None = None,
    rbac_info: dict = Depends(requires_permission(PERMISSION_SYSTEM_CONFIG_RELOAD)),
):
    """
    Flush all cache entries or entries matching a prefix.

    Requires admin API key authentication or user with system.config.reload permission.
    """
    request_id = get_request_id() or "unknown"
    from ..semantic_cache import get_cache_manager

    manager = get_cache_manager()
    if manager is None:
        return {"status": "cache_not_enabled"}
    count = await manager.flush(prefix=prefix)

    # Audit log the flush
    await handle_audit_write(
        AuditAction.CONFIG_RELOAD,
        "cache",
        prefix or "all",
        AuditOutcome.SUCCESS,
        rbac_info,
        request_id,
    )

    return {"status": "flushed", "entries_removed": count}


@admin_router.get("/admin/cache/entries")
async def list_cache_entries(
    limit: int = 100,
    rbac_info: dict = Depends(requires_permission(PERMISSION_SYSTEM_CONFIG_RELOAD)),
):
    """
    List cached keys with metadata.

    Requires admin API key authentication or user with system.config.reload permission.
    """
    from ..semantic_cache import get_cache_manager

    manager = get_cache_manager()
    if manager is None:
        return {"enabled": False, "entries": []}
    return manager.list_entries(limit=limit)


# =============================================================================
# Config Sync Endpoints
# =============================================================================


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


# Read-only - user auth
@llmrouter_router.get("/llmrouter/strategies/compare")
async def compare_strategies():
    """Compare active routing strategies.

    Returns all registered strategies with their state, version,
    and the current A/B testing configuration.
    """
    from ..strategy_registry import get_strategy_comparison

    return get_strategy_comparison()


# =============================================================================
# Config Status (unauthenticated, like health probes)
# =============================================================================


@health_router.get("/config/status")
async def config_status():
    """Config sync status. Unauthenticated (same as health probes)."""
    status = get_config_sync_status()
    return dataclasses.asdict(status)


# =============================================================================
# Spend Report Endpoint (v0.2.0 Wave C)
# =============================================================================


@admin_router.get("/llmrouter/spend/report")
async def get_spend_report(
    rbac_info: dict = Depends(requires_permission(PERMISSION_SYSTEM_CONFIG_RELOAD)),
):
    """
    Get an aggregated spend report from LiteLLM's spend tracking.

    Returns per-model token usage and cost estimates from the LiteLLM proxy's
    internal accounting. Falls back gracefully if spend tracking is not configured.

    Requires admin auth.
    """
    report: dict = {
        "status": "ok",
        "spend_tracking_enabled": False,
        "models": [],
        "total_cost_usd": 0.0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
    }

    try:
        from litellm.proxy.proxy_server import llm_router

        if llm_router is None:
            report["status"] = "no_router"
            return report

        # Get model list with cost info
        model_list = getattr(llm_router, "model_list", []) or []
        model_reports = []

        for model in model_list:
            model_name = model.get("model_name", "unknown")
            litellm_model = model.get("litellm_params", {}).get("model", "unknown")
            model_info = model.get("model_info", {})

            model_reports.append(
                {
                    "model_name": model_name,
                    "litellm_model": litellm_model,
                    "max_input_tokens": model_info.get("max_input_tokens"),
                    "max_output_tokens": model_info.get("max_output_tokens"),
                }
            )

        report["models"] = model_reports
        report["model_count"] = len(model_reports)

        # Try to get spend data from LiteLLM's internal spend tracking
        try:
            import litellm

            if hasattr(litellm, "model_cost") and litellm.model_cost:
                report["spend_tracking_enabled"] = True
                report["known_cost_models"] = len(litellm.model_cost)
        except Exception:
            pass

    except Exception as e:
        report["status"] = "error"
        report["error"] = str(e)

    return report


# =============================================================================
# Governance Endpoints (Workspace & Key CRUD)
# =============================================================================


# -- Workspace CRUD ---------------------------------------------------------


@admin_router.get("/api/v1/routeiq/governance/workspaces")
async def list_workspaces(
    org_id: Optional[str] = None,
    rbac_info: dict = Depends(requires_permission(PERMISSION_SYSTEM_CONFIG_RELOAD)),
):
    """List all workspaces.  Optionally filter by org_id.

    Requires admin API key or user with system.config.reload permission.
    """
    engine = get_governance_engine()
    workspaces = engine.list_workspaces(org_id=org_id)
    return {
        "workspaces": [w.model_dump() for w in workspaces],
        "count": len(workspaces),
    }


@admin_router.post("/api/v1/routeiq/governance/workspaces", status_code=201)
async def create_workspace(
    config: WorkspaceConfig,
    rbac_info: dict = Depends(requires_permission(PERMISSION_SYSTEM_CONFIG_RELOAD)),
):
    """Create or update a workspace.

    Requires admin API key or user with system.config.reload permission.
    """
    request_id = get_request_id() or "unknown"
    engine = get_governance_engine()

    existing = engine.get_workspace(config.workspace_id)
    engine.register_workspace(config)

    await handle_audit_write(
        AuditAction.CONFIG_RELOAD,
        "governance_workspace",
        config.workspace_id,
        AuditOutcome.SUCCESS,
        rbac_info,
        request_id,
    )

    return {
        "workspace_id": config.workspace_id,
        "created": existing is None,
        "workspace": engine.get_workspace(config.workspace_id).model_dump(),
    }


@admin_router.get("/api/v1/routeiq/governance/workspaces/{workspace_id}")
async def get_workspace(
    workspace_id: str,
    rbac_info: dict = Depends(requires_permission(PERMISSION_SYSTEM_CONFIG_RELOAD)),
):
    """Get a workspace by ID.

    Requires admin API key or user with system.config.reload permission.
    """
    engine = get_governance_engine()
    workspace = engine.get_workspace(workspace_id)
    if workspace is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "workspace_not_found",
                "message": f"Workspace '{workspace_id}' not found.",
                "request_id": get_request_id() or "unknown",
            },
        )
    return {"workspace": workspace.model_dump()}


@admin_router.put("/api/v1/routeiq/governance/workspaces/{workspace_id}")
async def update_workspace(
    workspace_id: str,
    config: WorkspaceConfig,
    rbac_info: dict = Depends(requires_permission(PERMISSION_SYSTEM_CONFIG_RELOAD)),
):
    """Update a workspace.  The workspace_id in the path takes precedence.

    Requires admin API key or user with system.config.reload permission.
    """
    request_id = get_request_id() or "unknown"
    engine = get_governance_engine()

    existing = engine.get_workspace(workspace_id)
    if existing is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "workspace_not_found",
                "message": f"Workspace '{workspace_id}' not found.",
                "request_id": request_id,
            },
        )

    # Ensure the path workspace_id is used
    config.workspace_id = workspace_id
    # Preserve original created_at
    config.created_at = existing.created_at
    engine.register_workspace(config)

    await handle_audit_write(
        AuditAction.CONFIG_RELOAD,
        "governance_workspace",
        workspace_id,
        AuditOutcome.SUCCESS,
        rbac_info,
        request_id,
    )

    return {"workspace": engine.get_workspace(workspace_id).model_dump()}


@admin_router.delete("/api/v1/routeiq/governance/workspaces/{workspace_id}")
async def delete_workspace(
    workspace_id: str,
    rbac_info: dict = Depends(requires_permission(PERMISSION_SYSTEM_CONFIG_RELOAD)),
):
    """Delete a workspace.

    Requires admin API key or user with system.config.reload permission.
    """
    request_id = get_request_id() or "unknown"
    engine = get_governance_engine()

    deleted = engine.delete_workspace(workspace_id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "workspace_not_found",
                "message": f"Workspace '{workspace_id}' not found.",
                "request_id": request_id,
            },
        )

    await handle_audit_write(
        AuditAction.CONFIG_RELOAD,
        "governance_workspace",
        workspace_id,
        AuditOutcome.SUCCESS,
        rbac_info,
        request_id,
    )

    return {"deleted": True, "workspace_id": workspace_id}


# -- Key Governance CRUD ---------------------------------------------------


@admin_router.get("/api/v1/routeiq/governance/keys/{key_id}")
async def get_key_governance(
    key_id: str,
    rbac_info: dict = Depends(requires_permission(PERMISSION_SYSTEM_CONFIG_RELOAD)),
):
    """Get governance rules for an API key.

    Requires admin API key or user with system.config.reload permission.
    """
    engine = get_governance_engine()
    governance = engine.get_key_governance(key_id)
    if governance is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "key_governance_not_found",
                "message": f"No governance rules found for key '{key_id}'.",
                "request_id": get_request_id() or "unknown",
            },
        )
    return {"key_governance": governance.model_dump()}


@admin_router.put("/api/v1/routeiq/governance/keys/{key_id}")
async def update_key_governance(
    key_id: str,
    governance: KeyGovernance,
    rbac_info: dict = Depends(requires_permission(PERMISSION_SYSTEM_CONFIG_RELOAD)),
):
    """Create or update governance rules for an API key.

    The key_id in the path takes precedence over the body.
    If workspace_id is specified in the governance body, the workspace must exist.

    Requires admin API key or user with system.config.reload permission.
    """
    request_id = get_request_id() or "unknown"
    engine = get_governance_engine()

    # Ensure path key_id is used
    governance.key_id = key_id

    # Validate workspace reference if provided
    if governance.workspace_id:
        ws = engine.get_workspace(governance.workspace_id)
        if ws is None:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "workspace_not_found",
                    "message": (
                        f"Referenced workspace '{governance.workspace_id}' "
                        f"does not exist."
                    ),
                    "request_id": request_id,
                },
            )

    engine.register_key_governance(governance)

    await handle_audit_write(
        AuditAction.CONFIG_RELOAD,
        "governance_key",
        key_id,
        AuditOutcome.SUCCESS,
        rbac_info,
        request_id,
    )

    return {"key_governance": engine.get_key_governance(key_id).model_dump()}


@admin_router.delete("/api/v1/routeiq/governance/keys/{key_id}")
async def delete_key_governance(
    key_id: str,
    rbac_info: dict = Depends(requires_permission(PERMISSION_SYSTEM_CONFIG_RELOAD)),
):
    """Delete governance rules for an API key.

    Requires admin API key or user with system.config.reload permission.
    """
    request_id = get_request_id() or "unknown"
    engine = get_governance_engine()

    deleted = engine.delete_key_governance(key_id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "key_governance_not_found",
                "message": f"No governance rules found for key '{key_id}'.",
                "request_id": request_id,
            },
        )

    await handle_audit_write(
        AuditAction.CONFIG_RELOAD,
        "governance_key",
        key_id,
        AuditOutcome.SUCCESS,
        rbac_info,
        request_id,
    )

    return {"deleted": True, "key_id": key_id}


# -- Governance Status ------------------------------------------------------


@admin_router.get("/api/v1/routeiq/governance/status")
async def governance_status(
    rbac_info: dict = Depends(requires_permission(PERMISSION_SYSTEM_CONFIG_RELOAD)),
):
    """Get governance engine status.

    Requires admin API key or user with system.config.reload permission.
    """
    engine = get_governance_engine()
    return engine.get_status()


# =============================================================================
# Usage Policy CRUD Endpoints
# =============================================================================


@admin_router.get("/api/v1/routeiq/governance/policies")
async def list_usage_policies(
    workspace_id: Optional[str] = None,
    rbac_info: dict = Depends(requires_permission(PERMISSION_SYSTEM_CONFIG_RELOAD)),
):
    """List all usage policies. Optionally filter by workspace_id.

    Returns policies sorted by priority (lowest first).
    Requires admin API key or user with system.config.reload permission.
    """
    engine = get_usage_policy_engine()
    policies = engine.list_policies(workspace_id=workspace_id)
    return {
        "policies": [p.model_dump() for p in policies],
        "count": len(policies),
    }


@admin_router.post("/api/v1/routeiq/governance/policies", status_code=201)
async def create_usage_policy(
    policy: UsagePolicy,
    rbac_info: dict = Depends(requires_permission(PERMISSION_SYSTEM_CONFIG_RELOAD)),
):
    """Create or update a usage policy.

    If a policy with the same policy_id already exists, it is overwritten.
    The alert_threshold must be between 0 and 1.
    The limit_value must be positive.

    Requires admin API key or user with system.config.reload permission.
    """
    request_id = get_request_id() or "unknown"

    # Validate
    if policy.limit_value <= 0:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_limit_value",
                "message": "limit_value must be positive.",
                "request_id": request_id,
            },
        )
    if not (0 <= policy.alert_threshold <= 1):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_alert_threshold",
                "message": "alert_threshold must be between 0 and 1.",
                "request_id": request_id,
            },
        )

    engine = get_usage_policy_engine()
    existing = engine.get_policy(policy.policy_id)
    engine.add_policy(policy)

    await handle_audit_write(
        AuditAction.CONFIG_RELOAD,
        "governance_usage_policy",
        policy.policy_id,
        AuditOutcome.SUCCESS,
        rbac_info,
        request_id,
    )

    return {
        "policy_id": policy.policy_id,
        "created": existing is None,
        "policy": engine.get_policy(policy.policy_id).model_dump(),
    }


@admin_router.get("/api/v1/routeiq/governance/policies/{policy_id}")
async def get_usage_policy(
    policy_id: str,
    rbac_info: dict = Depends(requires_permission(PERMISSION_SYSTEM_CONFIG_RELOAD)),
):
    """Get a usage policy by ID.

    Requires admin API key or user with system.config.reload permission.
    """
    engine = get_usage_policy_engine()
    policy = engine.get_policy(policy_id)
    if policy is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "policy_not_found",
                "message": f"Usage policy '{policy_id}' not found.",
                "request_id": get_request_id() or "unknown",
            },
        )
    return {"policy": policy.model_dump()}


@admin_router.put("/api/v1/routeiq/governance/policies/{policy_id}")
async def update_usage_policy(
    policy_id: str,
    policy: UsagePolicy,
    rbac_info: dict = Depends(requires_permission(PERMISSION_SYSTEM_CONFIG_RELOAD)),
):
    """Update an existing usage policy. The policy_id in the path takes precedence.

    Requires admin API key or user with system.config.reload permission.
    """
    request_id = get_request_id() or "unknown"
    engine = get_usage_policy_engine()

    existing = engine.get_policy(policy_id)
    if existing is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "policy_not_found",
                "message": f"Usage policy '{policy_id}' not found.",
                "request_id": request_id,
            },
        )

    # Validate
    if policy.limit_value <= 0:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_limit_value",
                "message": "limit_value must be positive.",
                "request_id": request_id,
            },
        )
    if not (0 <= policy.alert_threshold <= 1):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_alert_threshold",
                "message": "alert_threshold must be between 0 and 1.",
                "request_id": request_id,
            },
        )

    # Ensure path policy_id is used + preserve created_at
    policy.policy_id = policy_id
    policy.created_at = existing.created_at
    engine.add_policy(policy)

    await handle_audit_write(
        AuditAction.CONFIG_RELOAD,
        "governance_usage_policy",
        policy_id,
        AuditOutcome.SUCCESS,
        rbac_info,
        request_id,
    )

    return {"policy": engine.get_policy(policy_id).model_dump()}


@admin_router.delete("/api/v1/routeiq/governance/policies/{policy_id}")
async def delete_usage_policy(
    policy_id: str,
    rbac_info: dict = Depends(requires_permission(PERMISSION_SYSTEM_CONFIG_RELOAD)),
):
    """Delete a usage policy and its associated counters.

    Requires admin API key or user with system.config.reload permission.
    """
    request_id = get_request_id() or "unknown"
    engine = get_usage_policy_engine()

    deleted = engine.remove_policy(policy_id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "policy_not_found",
                "message": f"Usage policy '{policy_id}' not found.",
                "request_id": request_id,
            },
        )

    await handle_audit_write(
        AuditAction.CONFIG_RELOAD,
        "governance_usage_policy",
        policy_id,
        AuditOutcome.SUCCESS,
        rbac_info,
        request_id,
    )

    return {"deleted": True, "policy_id": policy_id}


@admin_router.get("/api/v1/routeiq/governance/policies/{policy_id}/usage")
async def get_usage_policy_usage(
    policy_id: str,
    group_key: Optional[str] = None,
    rbac_info: dict = Depends(requires_permission(PERMISSION_SYSTEM_CONFIG_RELOAD)),
):
    """Get current usage counters for a usage policy.

    If *group_key* is provided, returns the counter for that specific group.
    Otherwise returns a summary including the global counter.

    Requires admin API key or user with system.config.reload permission.
    """
    engine = get_usage_policy_engine()
    policy = engine.get_policy(policy_id)
    if policy is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "policy_not_found",
                "message": f"Usage policy '{policy_id}' not found.",
                "request_id": get_request_id() or "unknown",
            },
        )

    if group_key is not None:
        usage = await engine.get_usage(policy_id, group_key)
        return {
            "policy_id": policy_id,
            "group_key": group_key,
            "current_usage": usage,
            "limit_value": policy.limit_value,
            "limit_type": policy.limit_type.value,
            "limit_period": policy.limit_period.value,
            "usage_pct": (
                round(usage / policy.limit_value, 4) if policy.limit_value > 0 else 0.0
            ),
        }

    return await engine.get_policy_usage_summary(policy_id)


@admin_router.post("/api/v1/routeiq/governance/policies/{policy_id}/reset")
async def reset_usage_policy_counters(
    policy_id: str,
    group_key: Optional[str] = None,
    rbac_info: dict = Depends(requires_permission(PERMISSION_SYSTEM_CONFIG_RELOAD)),
):
    """Reset usage counters for a policy.

    If *group_key* is provided, only that group's counter is reset.
    Otherwise the global counter is reset.

    Requires admin API key or user with system.config.reload permission.
    """
    request_id = get_request_id() or "unknown"
    engine = get_usage_policy_engine()

    policy = engine.get_policy(policy_id)
    if policy is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "policy_not_found",
                "message": f"Usage policy '{policy_id}' not found.",
                "request_id": request_id,
            },
        )

    success = await engine.reset_usage(policy_id, group_key=group_key)

    await handle_audit_write(
        AuditAction.CONFIG_RELOAD,
        "governance_usage_policy_reset",
        policy_id,
        AuditOutcome.SUCCESS,
        rbac_info,
        request_id,
    )

    return {
        "reset": success,
        "policy_id": policy_id,
        "group_key": group_key or "__global__",
    }


# =============================================================================
# Personalized Routing Feedback Endpoints
# =============================================================================


@llmrouter_router.post("/api/v1/routeiq/routing/feedback")
async def submit_routing_feedback(request: Request):
    """Submit feedback on a routing decision to improve personalization.

    Accepts a JSON body with:
    - ``user_id`` (str, required): The user or team identifier.
    - ``model`` (str, required): The model that was used.
    - ``score`` (float, required): Feedback score in [-1.0, 1.0].
      Positive values indicate satisfaction, negative indicate dissatisfaction.
    - ``request_id`` (str, optional): The original request ID for correlation.

    Returns 200 with feedback confirmation, or 400/503 on error.
    Requires user API key authentication (inherited from llmrouter_router).
    """
    from ..personalized_routing import get_personalized_router

    router = get_personalized_router()
    if router is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "personalized_routing_disabled",
                "message": (
                    "Personalized routing is not enabled. "
                    "Set ROUTEIQ_PERSONALIZED_ROUTING=true to enable."
                ),
            },
        )

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_json",
                "message": "Request body must be valid JSON.",
            },
        )

    user_id = body.get("user_id")
    model = body.get("model")
    score = body.get("score")

    if not user_id or not isinstance(user_id, str):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "missing_field",
                "message": "'user_id' is required and must be a string.",
            },
        )
    if not model or not isinstance(model, str):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "missing_field",
                "message": "'model' is required and must be a string.",
            },
        )
    if score is None or not isinstance(score, (int, float)):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "missing_field",
                "message": "'score' is required and must be a number.",
            },
        )

    score = float(score)
    if score < -1.0 or score > 1.0:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_value",
                "message": "'score' must be between -1.0 and 1.0.",
            },
        )

    await router.record_feedback(user_id, model, score)

    return {
        "status": "ok",
        "user_id": user_id,
        "model": model,
        "score": score,
        "request_id": body.get("request_id"),
    }


@llmrouter_router.get("/api/v1/routeiq/routing/preferences/{user_id}")
async def get_user_preferences(user_id: str):
    """Get the current routing preferences for a user.

    Returns the user's preference state including interaction count,
    per-model scores, and last update time. Returns 404 for cold-start
    users with no preference data.

    Requires user API key authentication.
    """
    from ..personalized_routing import get_personalized_router

    router = get_personalized_router()
    if router is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "personalized_routing_disabled",
                "message": "Personalized routing is not enabled.",
            },
        )

    pref = await router.store.get_preference(user_id)
    if pref is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "no_preferences",
                "message": f"No preference data found for user '{user_id}'.",
            },
        )

    return {
        "user_id": pref.user_id,
        "interaction_count": pref.interaction_count,
        "last_updated": pref.last_updated,
        "model_scores": pref.model_scores,
    }


@admin_router.delete("/api/v1/routeiq/routing/preferences/{user_id}")
async def delete_user_preferences(
    user_id: str,
    rbac_info: dict = Depends(requires_permission(PERMISSION_SYSTEM_CONFIG_RELOAD)),
):
    """Delete all routing preference data for a user (GDPR/privacy).

    Requires admin API key authentication.
    """
    from ..personalized_routing import get_personalized_router

    router = get_personalized_router()
    if router is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "personalized_routing_disabled",
                "message": "Personalized routing is not enabled.",
            },
        )

    request_id = get_request_id() or "unknown"
    deleted = await router.delete_user_data(user_id)

    await handle_audit_write(
        AuditAction.CONFIG_RELOAD,
        "routing_preferences",
        user_id,
        AuditOutcome.SUCCESS,
        rbac_info,
        request_id,
    )

    return {"deleted": deleted, "user_id": user_id}


@admin_router.get("/api/v1/routeiq/routing/personalized/stats")
async def get_personalized_routing_stats(
    rbac_info: dict = Depends(requires_permission(PERMISSION_SYSTEM_CONFIG_RELOAD)),
):
    """Get personalized routing statistics.

    Returns configuration, model count, and preference store stats.
    Requires admin API key authentication.
    """
    from ..personalized_routing import get_personalized_router

    router = get_personalized_router()
    if router is None:
        return {"enabled": False}

    return {"enabled": True, **router.get_stats()}


# =============================================================================
# Guardrail Policy CRUD Endpoints
# =============================================================================


@admin_router.get("/api/v1/routeiq/governance/guardrails")
async def list_guardrail_policies(
    phase: Optional[str] = None,
    workspace_id: Optional[str] = None,
    rbac_info: dict = Depends(requires_permission(PERMISSION_SYSTEM_CONFIG_RELOAD)),
):
    """List all guardrail policies. Optionally filter by phase and workspace_id.

    Returns policies sorted by priority (lowest first).
    Requires admin API key or user with system.config.reload permission.
    """
    engine = get_guardrail_policy_engine()

    phase_enum = None
    if phase is not None:
        try:
            phase_enum = GuardrailPhase(phase)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_phase",
                    "message": f"Invalid phase '{phase}'. Must be 'input' or 'output'.",
                    "request_id": get_request_id() or "unknown",
                },
            )

    policies = engine.list_policies(phase=phase_enum, workspace_id=workspace_id)
    return {
        "guardrails": [p.model_dump() for p in policies],
        "count": len(policies),
    }


@admin_router.post("/api/v1/routeiq/governance/guardrails", status_code=201)
async def create_guardrail_policy(
    policy: GuardrailPolicy,
    rbac_info: dict = Depends(requires_permission(PERMISSION_SYSTEM_CONFIG_RELOAD)),
):
    """Create or update a guardrail policy.

    If a guardrail with the same guardrail_id already exists, it is overwritten.
    The guardrail_id and name are required.

    Requires admin API key or user with system.config.reload permission.
    """
    request_id = get_request_id() or "unknown"

    if not policy.guardrail_id:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "missing_guardrail_id",
                "message": "guardrail_id is required.",
                "request_id": request_id,
            },
        )
    if not policy.name:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "missing_name",
                "message": "name is required.",
                "request_id": request_id,
            },
        )

    engine = get_guardrail_policy_engine()
    existing = engine.get_policy(policy.guardrail_id)
    engine.add_policy(policy)

    await handle_audit_write(
        AuditAction.CONFIG_RELOAD,
        "governance_guardrail",
        policy.guardrail_id,
        AuditOutcome.SUCCESS,
        rbac_info,
        request_id,
    )

    return {
        "guardrail_id": policy.guardrail_id,
        "created": existing is None,
        "guardrail": engine.get_policy(policy.guardrail_id).model_dump(),
    }


@admin_router.get("/api/v1/routeiq/governance/guardrails/{guardrail_id}")
async def get_guardrail_policy(
    guardrail_id: str,
    rbac_info: dict = Depends(requires_permission(PERMISSION_SYSTEM_CONFIG_RELOAD)),
):
    """Get a guardrail policy by ID.

    Requires admin API key or user with system.config.reload permission.
    """
    engine = get_guardrail_policy_engine()
    policy = engine.get_policy(guardrail_id)
    if policy is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "guardrail_not_found",
                "message": f"Guardrail policy '{guardrail_id}' not found.",
                "request_id": get_request_id() or "unknown",
            },
        )
    return {"guardrail": policy.model_dump()}


@admin_router.put("/api/v1/routeiq/governance/guardrails/{guardrail_id}")
async def update_guardrail_policy(
    guardrail_id: str,
    policy: GuardrailPolicy,
    rbac_info: dict = Depends(requires_permission(PERMISSION_SYSTEM_CONFIG_RELOAD)),
):
    """Update an existing guardrail policy. The guardrail_id in the path takes precedence.

    Requires admin API key or user with system.config.reload permission.
    """
    request_id = get_request_id() or "unknown"
    engine = get_guardrail_policy_engine()

    existing = engine.get_policy(guardrail_id)
    if existing is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "guardrail_not_found",
                "message": f"Guardrail policy '{guardrail_id}' not found.",
                "request_id": request_id,
            },
        )

    # Ensure path guardrail_id is used + preserve created_at
    policy.guardrail_id = guardrail_id
    policy.created_at = existing.created_at
    engine.add_policy(policy)

    await handle_audit_write(
        AuditAction.CONFIG_RELOAD,
        "governance_guardrail",
        guardrail_id,
        AuditOutcome.SUCCESS,
        rbac_info,
        request_id,
    )

    return {"guardrail": engine.get_policy(guardrail_id).model_dump()}


@admin_router.delete("/api/v1/routeiq/governance/guardrails/{guardrail_id}")
async def delete_guardrail_policy(
    guardrail_id: str,
    rbac_info: dict = Depends(requires_permission(PERMISSION_SYSTEM_CONFIG_RELOAD)),
):
    """Delete a guardrail policy.

    Requires admin API key or user with system.config.reload permission.
    """
    request_id = get_request_id() or "unknown"
    engine = get_guardrail_policy_engine()

    deleted = engine.remove_policy(guardrail_id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "guardrail_not_found",
                "message": f"Guardrail policy '{guardrail_id}' not found.",
                "request_id": request_id,
            },
        )

    await handle_audit_write(
        AuditAction.CONFIG_RELOAD,
        "governance_guardrail",
        guardrail_id,
        AuditOutcome.SUCCESS,
        rbac_info,
        request_id,
    )

    return {"deleted": True, "guardrail_id": guardrail_id}


@admin_router.get("/api/v1/routeiq/governance/guardrails/status/summary")
async def guardrail_policy_status(
    rbac_info: dict = Depends(requires_permission(PERMISSION_SYSTEM_CONFIG_RELOAD)),
):
    """Get guardrail policy engine status summary.

    Returns counts of total/enabled/input/output policies and registered check types.
    Requires admin API key or user with system.config.reload permission.
    """
    engine = get_guardrail_policy_engine()
    return engine.get_status()
