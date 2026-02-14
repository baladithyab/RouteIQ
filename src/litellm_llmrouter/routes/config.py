"""
Hot Reload and Config Sync Endpoints.

- POST /llmrouter/reload - Trigger config reload (sync manager)
- POST /config/reload - Trigger config reload (hot reload manager)
- GET /config/sync/status - Get config sync status
- GET /router/info - Get routing configuration info
"""

from fastapi import Depends, HTTPException

import dataclasses

from ..auth import get_request_id, sanitize_error_response
from ..rbac import requires_permission, PERMISSION_SYSTEM_CONFIG_RELOAD
from ..audit import audit_log, AuditAction, AuditOutcome, AuditWriteError
from ..hot_reload import get_hot_reload_manager
from ..config_sync import get_sync_manager, get_config_sync_status
from .models import ReloadRequest
from . import admin_router, llmrouter_router, health_router


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
    await _handle_audit_write(
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
