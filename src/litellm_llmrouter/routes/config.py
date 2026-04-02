"""
Hot Reload, Config Sync, Governance, Usage Policy, Guardrail Policy, and Prompt
Management Endpoints.

- POST /llmrouter/reload - Trigger config reload (sync manager)
- POST /config/reload - Trigger config reload (hot reload manager)
- GET /config/sync/status - Get config sync status
- GET /router/info - Get routing configuration info
- /api/v1/routeiq/governance/* - Workspace & key governance CRUD
- /api/v1/routeiq/governance/policies/* - Usage policy CRUD & counters
- /api/v1/routeiq/governance/guardrails/* - Guardrail policy CRUD
- /api/v1/routeiq/prompts/* - Prompt management CRUD, versioning, A/B testing
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
from ..prompt_management import (
    get_prompt_manager,
    is_prompt_management_enabled,
    CreatePromptRequest,
    UpdatePromptRequest,
    RollbackRequest,
    ABTestRequest,
    ABTestStopRequest,
    ImportPromptsRequest,
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


# =============================================================================
# Prompt Management CRUD Endpoints
# =============================================================================


def _require_prompt_management() -> None:
    """Raise 404 if prompt management feature is disabled."""
    if not is_prompt_management_enabled():
        raise HTTPException(
            status_code=404,
            detail={
                "error": "feature_disabled",
                "message": (
                    "Prompt management is not enabled. "
                    "Set ROUTEIQ_PROMPT_MANAGEMENT=true to enable."
                ),
            },
        )


@admin_router.get("/api/v1/routeiq/prompts")
async def list_prompts(
    workspace_id: Optional[str] = None,
    tag: Optional[str] = None,
    rbac_info: dict = Depends(requires_permission(PERMISSION_SYSTEM_CONFIG_RELOAD)),
):
    """List all prompts. Optionally filter by workspace_id and/or tag.

    Requires admin API key or user with system.config.reload permission.
    """
    _require_prompt_management()
    manager = get_prompt_manager()
    tags = [tag] if tag else None
    prompts = manager.list_prompts(workspace_id=workspace_id, tags=tags)
    return {
        "prompts": [p.model_dump() for p in prompts],
        "count": len(prompts),
    }


@admin_router.post("/api/v1/routeiq/prompts", status_code=201)
async def create_prompt(
    body: CreatePromptRequest,
    rbac_info: dict = Depends(requires_permission(PERMISSION_SYSTEM_CONFIG_RELOAD)),
):
    """Create a new named prompt.

    The prompt name must be lowercase alphanumeric with hyphens (1-64 chars).
    Returns the created prompt definition with version 1.

    Requires admin API key or user with system.config.reload permission.
    """
    _require_prompt_management()
    request_id = get_request_id() or "unknown"
    manager = get_prompt_manager()

    try:
        prompt = manager.create_prompt(
            name=body.name,
            template=body.template,
            system_template=body.system_template,
            model=body.model,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
            description=body.description,
            workspace_id=body.workspace_id,
            created_by=rbac_info.get("user_id") if rbac_info else None,
            tags=body.tags,
            metadata=body.metadata,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_prompt",
                "message": str(exc),
                "request_id": request_id,
            },
        )

    await handle_audit_write(
        AuditAction.CONFIG_RELOAD,
        "prompt",
        body.name,
        AuditOutcome.SUCCESS,
        rbac_info,
        request_id,
    )

    return {"prompt": prompt.model_dump()}


@admin_router.get("/api/v1/routeiq/prompts/{name}")
async def get_prompt(
    name: str,
    workspace_id: Optional[str] = None,
    rbac_info: dict = Depends(requires_permission(PERMISSION_SYSTEM_CONFIG_RELOAD)),
):
    """Get a prompt definition by name.

    Requires admin API key or user with system.config.reload permission.
    """
    _require_prompt_management()
    manager = get_prompt_manager()
    prompt = manager.get_prompt(name, workspace_id=workspace_id)
    if prompt is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "prompt_not_found",
                "message": f"Prompt '{name}' not found.",
                "request_id": get_request_id() or "unknown",
            },
        )
    return {"prompt": prompt.model_dump()}


@admin_router.put("/api/v1/routeiq/prompts/{name}")
async def update_prompt(
    name: str,
    body: UpdatePromptRequest,
    workspace_id: Optional[str] = None,
    rbac_info: dict = Depends(requires_permission(PERMISSION_SYSTEM_CONFIG_RELOAD)),
):
    """Create a new version of an existing prompt.

    Each update creates a new version and sets it as active.
    The full version history is preserved for rollback.

    Requires admin API key or user with system.config.reload permission.
    """
    _require_prompt_management()
    request_id = get_request_id() or "unknown"
    manager = get_prompt_manager()

    try:
        prompt = manager.update_prompt(
            name=name,
            template=body.template,
            system_template=body.system_template,
            model=body.model,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
            change_note=body.change_note,
            created_by=rbac_info.get("user_id") if rbac_info else None,
            workspace_id=workspace_id,
            description=body.description,
            tags=body.tags,
            metadata=body.metadata,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "prompt_not_found",
                "message": str(exc),
                "request_id": request_id,
            },
        )

    await handle_audit_write(
        AuditAction.CONFIG_RELOAD,
        "prompt",
        name,
        AuditOutcome.SUCCESS,
        rbac_info,
        request_id,
    )

    return {"prompt": prompt.model_dump()}


@admin_router.delete("/api/v1/routeiq/prompts/{name}")
async def delete_prompt(
    name: str,
    workspace_id: Optional[str] = None,
    rbac_info: dict = Depends(requires_permission(PERMISSION_SYSTEM_CONFIG_RELOAD)),
):
    """Delete a prompt and all its versions.

    Requires admin API key or user with system.config.reload permission.
    """
    _require_prompt_management()
    request_id = get_request_id() or "unknown"
    manager = get_prompt_manager()

    deleted = manager.delete_prompt(name, workspace_id=workspace_id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "prompt_not_found",
                "message": f"Prompt '{name}' not found.",
                "request_id": request_id,
            },
        )

    await handle_audit_write(
        AuditAction.CONFIG_RELOAD,
        "prompt",
        name,
        AuditOutcome.SUCCESS,
        rbac_info,
        request_id,
    )

    return {"deleted": True, "name": name}


@admin_router.post("/api/v1/routeiq/prompts/{name}/rollback")
async def rollback_prompt(
    name: str,
    body: RollbackRequest,
    workspace_id: Optional[str] = None,
    rbac_info: dict = Depends(requires_permission(PERMISSION_SYSTEM_CONFIG_RELOAD)),
):
    """Roll back a prompt to a previous version.

    Sets the active version to the specified version number.
    The version must exist in the prompt's history.

    Requires admin API key or user with system.config.reload permission.
    """
    _require_prompt_management()
    request_id = get_request_id() or "unknown"
    manager = get_prompt_manager()

    try:
        prompt = manager.rollback(name, body.version, workspace_id=workspace_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "rollback_failed",
                "message": str(exc),
                "request_id": request_id,
            },
        )

    await handle_audit_write(
        AuditAction.CONFIG_RELOAD,
        "prompt_rollback",
        name,
        AuditOutcome.SUCCESS,
        rbac_info,
        request_id,
    )

    return {"prompt": prompt.model_dump(), "rolled_back_to": body.version}


@admin_router.post("/api/v1/routeiq/prompts/{name}/ab-test")
async def start_ab_test(
    name: str,
    body: ABTestRequest,
    workspace_id: Optional[str] = None,
    rbac_info: dict = Depends(requires_permission(PERMISSION_SYSTEM_CONFIG_RELOAD)),
):
    """Configure A/B testing between prompt versions.

    Provide a mapping of version numbers to traffic weights.
    Weights must sum to ~1.0 (tolerance: 0.01).
    All referenced versions must exist.

    Example body: {"versions": {"1": 0.9, "2": 0.1}}

    Requires admin API key or user with system.config.reload permission.
    """
    _require_prompt_management()
    request_id = get_request_id() or "unknown"
    manager = get_prompt_manager()

    try:
        prompt = manager.set_ab_test(name, body.versions, workspace_id=workspace_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "ab_test_failed",
                "message": str(exc),
                "request_id": request_id,
            },
        )

    await handle_audit_write(
        AuditAction.CONFIG_RELOAD,
        "prompt_ab_test",
        name,
        AuditOutcome.SUCCESS,
        rbac_info,
        request_id,
    )

    return {"prompt": prompt.model_dump(), "ab_test": "started"}


@admin_router.post("/api/v1/routeiq/prompts/{name}/ab-test/stop")
async def stop_ab_test(
    name: str,
    body: ABTestStopRequest,
    workspace_id: Optional[str] = None,
    rbac_info: dict = Depends(requires_permission(PERMISSION_SYSTEM_CONFIG_RELOAD)),
):
    """Stop an A/B test. Optionally promote a winning version.

    If ``winner`` is provided, that version becomes the new active version.
    Otherwise the current active version is kept.

    Requires admin API key or user with system.config.reload permission.
    """
    _require_prompt_management()
    request_id = get_request_id() or "unknown"
    manager = get_prompt_manager()

    try:
        prompt = manager.stop_ab_test(
            name, winner=body.winner, workspace_id=workspace_id
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "ab_test_stop_failed",
                "message": str(exc),
                "request_id": request_id,
            },
        )

    await handle_audit_write(
        AuditAction.CONFIG_RELOAD,
        "prompt_ab_test_stop",
        name,
        AuditOutcome.SUCCESS,
        rbac_info,
        request_id,
    )

    return {
        "prompt": prompt.model_dump(),
        "ab_test": "stopped",
        "winner": body.winner,
    }


@admin_router.get("/api/v1/routeiq/prompts-export")
async def export_prompts(
    workspace_id: Optional[str] = None,
    rbac_info: dict = Depends(requires_permission(PERMISSION_SYSTEM_CONFIG_RELOAD)),
):
    """Export all prompts as JSON.

    Optionally filter by workspace_id. Returns the full prompt definitions
    including all versions, suitable for backup/migration.

    Requires admin API key or user with system.config.reload permission.
    """
    _require_prompt_management()
    manager = get_prompt_manager()
    prompts = manager.list_prompts(workspace_id=workspace_id)
    return {
        "prompts": [p.model_dump() for p in prompts],
        "count": len(prompts),
    }


@admin_router.post("/api/v1/routeiq/prompts-import")
async def import_prompts(
    body: ImportPromptsRequest,
    rbac_info: dict = Depends(requires_permission(PERMISSION_SYSTEM_CONFIG_RELOAD)),
):
    """Import prompts from JSON.

    Accepts a list of prompt definitions. Existing prompts with the same
    name are overwritten. If ``workspace_id`` is provided, it overrides
    the workspace in the imported data.

    Requires admin API key or user with system.config.reload permission.
    """
    _require_prompt_management()
    request_id = get_request_id() or "unknown"
    manager = get_prompt_manager()

    count = manager.import_prompts(body.prompts, workspace_id=body.workspace_id)

    await handle_audit_write(
        AuditAction.CONFIG_RELOAD,
        "prompt_import",
        f"batch-{count}",
        AuditOutcome.SUCCESS,
        rbac_info,
        request_id,
    )

    return {"imported": count}


# =============================================================================
# Evaluation Pipeline Endpoints
# =============================================================================


@admin_router.get("/api/v1/routeiq/eval/stats")
async def get_eval_stats(
    rbac_info: dict = Depends(requires_permission(PERMISSION_SYSTEM_CONFIG_RELOAD)),
):
    """Get evaluation pipeline statistics.

    Returns pipeline state, pending/evaluated counts, model quality scores,
    and model rankings.

    Requires admin API key authentication.
    """
    from ..eval_pipeline import get_eval_pipeline

    pipeline = get_eval_pipeline()
    if pipeline is None:
        return {
            "enabled": False,
            "message": (
                "Evaluation pipeline is not enabled. "
                "Set ROUTEIQ_EVAL_PIPELINE=true to enable."
            ),
        }
    return {"enabled": True, **pipeline.get_stats()}


@admin_router.get("/api/v1/routeiq/eval/samples")
async def get_eval_samples(
    limit: int = 50,
    model: Optional[str] = None,
    evaluated_only: bool = False,
    rbac_info: dict = Depends(requires_permission(PERMISSION_SYSTEM_CONFIG_RELOAD)),
):
    """Get recent evaluation samples.

    Returns the most recent evaluation samples, optionally filtered by model
    or evaluation status.

    Args:
        limit: Maximum number of samples to return (default: 50, max: 500).
        model: Filter by model name (optional).
        evaluated_only: Only return evaluated samples (default: false).

    Requires admin API key authentication.
    """
    from ..eval_pipeline import get_eval_pipeline

    pipeline = get_eval_pipeline()
    if pipeline is None:
        return {
            "enabled": False,
            "samples": [],
            "message": "Evaluation pipeline is not enabled.",
        }

    # Clamp limit
    limit = min(max(limit, 1), 500)

    samples = pipeline.get_recent_samples(
        limit=limit, model=model, evaluated_only=evaluated_only
    )
    return {
        "samples": samples,
        "count": len(samples),
        "limit": limit,
        "filter_model": model,
        "evaluated_only": evaluated_only,
    }


@admin_router.post("/api/v1/routeiq/eval/run-batch")
async def run_eval_batch(
    rbac_info: dict = Depends(requires_permission(PERMISSION_SYSTEM_CONFIG_RELOAD)),
):
    """Manually trigger an evaluation batch.

    Evaluates up to ``batch_size`` pending samples immediately, regardless
    of the background loop schedule. Useful for testing or when you want
    immediate evaluation results.

    Requires admin API key authentication.
    """
    from ..eval_pipeline import get_eval_pipeline

    request_id = get_request_id() or "unknown"
    pipeline = get_eval_pipeline()
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "eval_pipeline_disabled",
                "message": (
                    "Evaluation pipeline is not enabled. "
                    "Set ROUTEIQ_EVAL_PIPELINE=true to enable."
                ),
                "request_id": request_id,
            },
        )

    count = await pipeline.run_evaluation_batch()

    await handle_audit_write(
        AuditAction.CONFIG_RELOAD,
        "eval_pipeline",
        "run_batch",
        AuditOutcome.SUCCESS,
        rbac_info,
        request_id,
    )

    return {
        "evaluated": count,
        "pending_remaining": len(pipeline._pending_samples),
    }


@admin_router.get("/api/v1/routeiq/eval/model-quality")
async def get_model_quality(
    rbac_info: dict = Depends(requires_permission(PERMISSION_SYSTEM_CONFIG_RELOAD)),
):
    """Get per-model quality scores from evaluations.

    Returns quality scores, rankings, and sample counts for all models
    that have been evaluated. Scores are normalized to 0.0-1.0.

    Requires admin API key authentication.
    """
    from ..eval_pipeline import get_eval_pipeline

    pipeline = get_eval_pipeline()
    if pipeline is None:
        return {
            "enabled": False,
            "models": {},
            "message": "Evaluation pipeline is not enabled.",
        }

    tracker = pipeline.tracker
    return {
        "models": tracker.get_all_qualities(),
        "ranking": tracker.get_ranking(),
        "sample_counts": tracker.get_sample_counts(),
    }


# =============================================================================
# Router-R1 Endpoint
# =============================================================================


@llmrouter_router.post("/api/v1/routeiq/routing/r1")
async def route_via_r1(request: Request):
    """Execute a query through the Router-R1 reasoning pipeline.

    This is a research/evaluation endpoint -- not for hot-path production use.
    The Router-R1 iteratively reasons about which model to route sub-queries to,
    using the configured router model as the reasoning agent.

    Accepts a JSON body with:
    - ``query`` (str, required): The user's query.
    - ``system_message`` (str, optional): System context for the router.

    Returns the final answer, execution trace, token usage, and latency.
    Requires user API key authentication.
    """
    from ..router_r1 import get_router_r1

    router = get_router_r1()
    if router is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "router_r1_disabled",
                "message": (
                    "Router-R1 is not enabled. "
                    "Set ROUTEIQ_ROUTER_R1_ENABLED=true to enable."
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

    query = body.get("query")
    if not query or not isinstance(query, str):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "missing_field",
                "message": "'query' is required and must be a string.",
            },
        )

    system_message = body.get("system_message")

    # Get model deployments from LiteLLM router
    try:
        from litellm.proxy.proxy_server import llm_router

        deployments = getattr(llm_router, "model_list", []) if llm_router else []
    except Exception:
        deployments = []

    result = await router.route(
        query=query,
        deployments=deployments,
        system_message=system_message,
    )

    return {
        "answer": result.answer,
        "router_model": result.router_model,
        "total_iterations": result.total_iterations,
        "total_tokens": result.total_tokens,
        "total_latency_ms": round(result.total_latency_ms, 1),
        "models_used": result.models_used,
        "steps": [
            {
                "iteration": s.iteration,
                "think": s.think,
                "routed_model": s.routed_model,
                "routed_query": s.routed_query,
                "result_preview": (
                    s.result[:200] + "..."
                    if s.result and len(s.result) > 200
                    else s.result
                ),
                "latency_ms": round(s.latency_ms, 1),
                "tokens_used": s.tokens_used,
            }
            for s in result.steps
        ],
    }


@admin_router.post("/api/v1/routeiq/eval/push-feedback")
async def push_eval_feedback(
    rbac_info: dict = Depends(requires_permission(PERMISSION_SYSTEM_CONFIG_RELOAD)),
):
    """Manually push evaluation quality scores to routing strategies.

    Immediately updates the personalized router's quality bias with
    the latest per-model scores from evaluations.

    Requires admin API key authentication.
    """
    from ..eval_pipeline import get_eval_pipeline

    request_id = get_request_id() or "unknown"
    pipeline = get_eval_pipeline()
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "eval_pipeline_disabled",
                "message": "Evaluation pipeline is not enabled.",
                "request_id": request_id,
            },
        )

    result = await pipeline.push_feedback()

    await handle_audit_write(
        AuditAction.CONFIG_RELOAD,
        "eval_pipeline",
        "push_feedback",
        AuditOutcome.SUCCESS,
        rbac_info,
        request_id,
    )

    return result
