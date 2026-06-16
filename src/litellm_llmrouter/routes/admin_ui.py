"""Admin UI API endpoints for RouteIQ gateway management.

Provides REST endpoints consumed by the RouteIQ Admin UI:

- ``GET  /api/v1/routeiq/status``         — Gateway status
- ``GET  /api/v1/routeiq/routing/stats``   — Routing statistics
- ``GET  /api/v1/routeiq/routing/config``  — Current routing configuration
- ``POST /api/v1/routeiq/routing/config``  — Update routing configuration
- ``GET  /api/v1/routeiq/models``          — Configured models

All endpoints require admin authentication (via ``admin_router``).
"""

import os
import time
from typing import Any

from fastapi import Depends, HTTPException
from pydantic import BaseModel

from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

from . import admin_router, llmrouter_router

# Module-level start time for uptime calculation
_start_time = time.time()

# Version
try:
    from importlib.metadata import version as _get_pkg_version

    _VERSION = _get_pkg_version("routeiq")
except Exception:
    _VERSION = "1.0.0rc1"  # fallback if not installed as package


# --- Pydantic Models ---


class GatewayStatusResponse(BaseModel):
    """Gateway status including version, uptime, and feature flags."""

    version: str
    uptime_seconds: float
    uptime_formatted: str
    worker_count: int
    active_strategy: str | None
    routing_profile: str
    centroid_routing_enabled: bool
    feature_flags: dict[str, bool]


class RoutingStatsResponse(BaseModel):
    """Routing decision statistics."""

    total_decisions: int
    strategy_distribution: dict[str, int]
    profile_distribution: dict[str, int]
    centroid_decisions: int
    average_latency_ms: float


class GlobalStatsResponse(BaseModel):
    """Org-wide routing rollups for the admin dashboard + future panels."""

    total_decisions: int
    strategy_distribution: dict[str, int]
    profile_distribution: dict[str, int]
    model_distribution: dict[str, int]
    key_distribution: dict[str, int]
    centroid_decisions: int
    average_latency_ms: float
    tracked_keys: int


class MyStatsResponse(BaseModel):
    """Caller-scoped usage stats for ``/me/stats``.

    Returns ONLY the authenticated caller's own key usage. Budget fields are
    populated only when the governance engine has a budget configured for the
    caller's key; otherwise they are ``None``.
    """

    key_id: str
    decision_count: int
    recent_models: list[str]
    budget_remaining_usd: float | None = None
    budget_used_pct: float | None = None
    spend_usd: float | None = None
    max_budget_usd: float | None = None


class ModelInfoResponse(BaseModel):
    """Model deployment information."""

    model_name: str
    provider: str
    model_id: str
    status: str  # "active", "degraded", "unavailable"


class RoutingConfigResponse(BaseModel):
    """Current routing configuration."""

    active_strategy: str | None
    available_strategies: list[str]
    routing_profile: str
    centroid_routing_enabled: bool
    ab_testing: dict[str, Any]


class UpdateRoutingConfigRequest(BaseModel):
    """Request body for updating routing configuration."""

    routing_profile: str | None = None
    centroid_routing_enabled: bool | None = None
    active_strategy: str | None = None


# --- Helper Functions ---


def _format_uptime(seconds: float) -> str:
    """Format seconds into human-readable uptime string."""
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    parts: list[str] = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def _get_feature_flags() -> dict[str, bool]:
    """Get current feature flag states from environment."""
    return {
        "mcp_gateway": os.environ.get("MCP_GATEWAY_ENABLED", "false").lower() == "true",
        "a2a_gateway": os.environ.get("A2A_GATEWAY_ENABLED", "false").lower() == "true",
        "policy_engine": os.environ.get("POLICY_ENGINE_ENABLED", "false").lower()
        == "true",
        "centroid_routing": os.environ.get("ROUTEIQ_CENTROID_ROUTING", "true").lower()
        == "true",
        "plugin_strategy": os.environ.get("ROUTEIQ_USE_PLUGIN_STRATEGY", "true").lower()
        == "true",
        "admin_ui": os.environ.get("ROUTEIQ_ADMIN_UI_ENABLED", "false").lower()
        == "true",
        "config_hot_reload": os.environ.get("CONFIG_HOT_RELOAD", "false").lower()
        == "true",
    }


def _get_routing_profile() -> str:
    """Get current routing profile from centroid strategy or env."""
    try:
        from litellm_llmrouter.centroid_routing import get_centroid_strategy

        strategy = get_centroid_strategy()
        if strategy and hasattr(strategy, "_profile"):
            profile = strategy._profile
            return profile.value if hasattr(profile, "value") else str(profile)
    except Exception:
        pass
    return os.environ.get("ROUTEIQ_ROUTING_PROFILE", "auto")


def _is_centroid_enabled() -> bool:
    """Check if centroid routing is enabled."""
    return os.environ.get("ROUTEIQ_CENTROID_ROUTING", "true").lower() == "true"


def _resolve_caller_key_id(auth: Any) -> str:
    """Resolve the CALLER's own raw key_id from the user_api_key_auth context.

    Fail-closed to the caller's own identity: this NEVER reads a key_id from
    request input (query/body), so a caller can only ever see their own scope.
    Accepts either LiteLLM's ``UserAPIKeyAuth`` object or the dict the test
    fixture overrides it with.

    Precedence mirrors the governance scope token: the RAW ``api_key`` (the
    same value the governance engine keys ``key_id`` on), then the hashed
    ``token``, then ``user_id``. Returns ``"anonymous"`` only when the auth
    context carries no identity at all.
    """

    def _get(field: str) -> Any:
        if isinstance(auth, dict):
            return auth.get(field)
        return getattr(auth, field, None)

    for field in ("api_key", "token", "user_id", "key_alias"):
        value = _get(field)
        if value:
            return str(value)
    return "anonymous"


def _get_models() -> list[dict[str, str]]:
    """Get configured models from LiteLLM proxy config."""
    models: list[dict[str, str]] = []
    try:
        import litellm

        if hasattr(litellm, "model_list") and litellm.model_list:
            for entry in litellm.model_list:
                model_name = entry.get("model_name", "unknown")
                litellm_params = entry.get("litellm_params", {})
                model_id = litellm_params.get("model", "unknown")
                # Parse provider from model_id (e.g., "anthropic/claude-3" -> "anthropic")
                provider = model_id.split("/")[0] if "/" in model_id else "unknown"
                models.append(
                    {
                        "model_name": model_name,
                        "provider": provider,
                        "model_id": model_id,
                        "status": "active",
                    }
                )
    except Exception:
        pass

    # Fallback: try to read from config
    if not models:
        try:
            from litellm_llmrouter.config_loader import load_config

            config = load_config(
                os.environ.get("LITELLM_CONFIG_PATH", "config/config.yaml")
            )
            if config and "model_list" in config:
                for entry in config["model_list"]:
                    model_name = entry.get("model_name", "unknown")
                    litellm_params = entry.get("litellm_params", {})
                    model_id = litellm_params.get("model", "unknown")
                    provider = model_id.split("/")[0] if "/" in model_id else "unknown"
                    models.append(
                        {
                            "model_name": model_name,
                            "provider": provider,
                            "model_id": model_id,
                            "status": "active",
                        }
                    )
        except Exception:
            pass

    return models


# --- Endpoints ---


@admin_router.get("/api/v1/routeiq/status", response_model=GatewayStatusResponse)
async def get_gateway_status():
    """Get gateway status including version, uptime, and feature flags."""
    uptime = time.time() - _start_time

    # Get active strategy from registry
    active_strategy = None
    try:
        from litellm_llmrouter.strategy_registry import get_routing_registry

        registry = get_routing_registry()
        active_strategy = registry.get_active()
    except Exception:
        pass

    return GatewayStatusResponse(
        version=_VERSION,
        uptime_seconds=round(uptime, 1),
        uptime_formatted=_format_uptime(uptime),
        worker_count=int(os.environ.get("ROUTEIQ_WORKERS", "1")),
        active_strategy=active_strategy,
        routing_profile=_get_routing_profile(),
        centroid_routing_enabled=_is_centroid_enabled(),
        feature_flags=_get_feature_flags(),
    )


@admin_router.get("/api/v1/routeiq/routing/stats", response_model=RoutingStatsResponse)
async def get_routing_stats():
    """Get routing decision statistics.

    Reads live aggregates from the in-process routing stats accumulator
    (fed by ``router_decision_callback`` on every decision). Seeds the
    strategy distribution with all registered strategies (at 0) so the
    dashboard shows the full strategy set, then overlays live counts.
    """
    from litellm_llmrouter.router_decision_callback import get_stats_accumulator

    snapshot = get_stats_accumulator().global_snapshot()

    # Seed strategy distribution with all registered strategies, then overlay
    # the live per-strategy decision counts from the accumulator.
    strategy_distribution: dict[str, int] = {}
    try:
        from litellm_llmrouter.strategy_registry import get_routing_registry

        registry = get_routing_registry()
        status = registry.get_status()
        for name in status.get("registered_strategies", []):
            strategy_distribution[name] = 0
    except Exception:
        pass
    strategy_distribution.update(snapshot["strategy_distribution"])

    return RoutingStatsResponse(
        total_decisions=snapshot["total_decisions"],
        strategy_distribution=strategy_distribution,
        profile_distribution=snapshot["profile_distribution"],
        centroid_decisions=snapshot["centroid_decisions"],
        average_latency_ms=snapshot["average_latency_ms"],
    )


@admin_router.get(
    "/api/v1/routeiq/routing/config", response_model=RoutingConfigResponse
)
async def get_routing_config():
    """Get current routing configuration."""
    active_strategy = None
    available_strategies: list[str] = []
    ab_testing: dict[str, Any] = {
        "enabled": False,
        "weights": {},
        "experiment_id": None,
    }

    try:
        from litellm_llmrouter.strategy_registry import get_routing_registry

        registry = get_routing_registry()
        active_strategy = registry.get_active()
        available_strategies = list(registry.list_strategies())

        experiment = registry.get_experiment()
        if experiment:
            ab_testing = {
                "enabled": experiment.enabled,
                "weights": dict(experiment.weights) if experiment.weights else {},
                "experiment_id": experiment.experiment_id,
            }
        else:
            weights = registry.get_weights()
            if weights:
                ab_testing = {
                    "enabled": True,
                    "weights": dict(weights),
                    "experiment_id": None,
                }
    except Exception:
        pass

    return RoutingConfigResponse(
        active_strategy=active_strategy,
        available_strategies=available_strategies,
        routing_profile=_get_routing_profile(),
        centroid_routing_enabled=_is_centroid_enabled(),
        ab_testing=ab_testing,
    )


@admin_router.post(
    "/api/v1/routeiq/routing/config", response_model=RoutingConfigResponse
)
async def update_routing_config(request: UpdateRoutingConfigRequest):
    """Update routing configuration."""
    # Update routing profile
    if request.routing_profile is not None:
        valid_profiles = {"auto", "eco", "premium", "free", "reasoning"}
        if request.routing_profile not in valid_profiles:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid routing profile: {request.routing_profile}. "
                    f"Must be one of: {', '.join(sorted(valid_profiles))}"
                ),
            )
        try:
            from litellm_llmrouter.centroid_routing import (
                RoutingProfile,
                get_centroid_strategy,
            )

            strategy = get_centroid_strategy()
            if strategy:
                strategy._profile = RoutingProfile(request.routing_profile)
            # Also update env for persistence across restarts
            os.environ["ROUTEIQ_ROUTING_PROFILE"] = request.routing_profile
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to update routing profile: {e}",
            )

    # Update centroid routing
    if request.centroid_routing_enabled is not None:
        os.environ["ROUTEIQ_CENTROID_ROUTING"] = str(
            request.centroid_routing_enabled
        ).lower()

    # Update active strategy
    if request.active_strategy is not None:
        try:
            from litellm_llmrouter.strategy_registry import get_routing_registry

            registry = get_routing_registry()
            if not registry.set_active(request.active_strategy):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Strategy '{request.active_strategy}' not found. "
                        f"Available: {registry.list_strategies()}"
                    ),
                )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to set active strategy: {e}",
            )

    # Return updated config
    return await get_routing_config()


@admin_router.get("/api/v1/routeiq/models", response_model=list[ModelInfoResponse])
async def get_models():
    """Get configured models with deployment info."""
    models = _get_models()
    return [ModelInfoResponse(**m) for m in models]


@llmrouter_router.get("/api/v1/routeiq/me/stats", response_model=MyStatsResponse)
async def get_my_stats(auth: Any = Depends(user_api_key_auth)):
    """Return the CALLER's OWN usage stats (user-auth, scope-isolated).

    The caller's key_id is resolved from the ``user_api_key_auth`` context
    only -- never from request input -- so a user can see ONLY their own
    stats and can never read another key's usage. Budget fields are populated
    when the governance engine has a budget configured for the caller's key.
    """
    key_id = _resolve_caller_key_id(auth)

    from litellm_llmrouter.router_decision_callback import get_stats_accumulator

    key_stats = get_stats_accumulator().key_snapshot(key_id)

    response = MyStatsResponse(
        key_id=key_id,
        decision_count=key_stats["decisions"],
        recent_models=key_stats["recent_models"],
    )

    # Overlay governance budget/spend for THIS key only, if configured.
    try:
        from litellm_llmrouter.governance import get_governance_engine

        engine = get_governance_engine()
        ctx = await engine.resolve_context(key_id)
        if ctx.effective_max_budget_usd is not None:
            # check_budget populates budget_remaining_usd / budget_used_pct
            # from the spend store (best-effort; tolerates store outage).
            try:
                await engine.check_budget(ctx)
            except Exception:
                pass
            response.max_budget_usd = ctx.effective_max_budget_usd
            response.budget_remaining_usd = ctx.budget_remaining_usd
            response.budget_used_pct = ctx.budget_used_pct
            if (
                ctx.budget_remaining_usd is not None
                and ctx.effective_max_budget_usd is not None
            ):
                response.spend_usd = max(
                    0.0, ctx.effective_max_budget_usd - ctx.budget_remaining_usd
                )
    except Exception:
        # Governance unavailable -> return usage stats without budget fields.
        pass

    return response


@admin_router.get("/api/v1/routeiq/stats/global", response_model=GlobalStatsResponse)
async def get_global_stats():
    """Org-wide routing rollups (admin auth).

    Exposes the per-strategy / per-model / per-key breakdowns the admin
    Dashboard and future analytics panels need, on top of the global totals.
    """
    from litellm_llmrouter.router_decision_callback import get_stats_accumulator

    snapshot = get_stats_accumulator().global_snapshot()
    return GlobalStatsResponse(
        total_decisions=snapshot["total_decisions"],
        strategy_distribution=snapshot["strategy_distribution"],
        profile_distribution=snapshot["profile_distribution"],
        model_distribution=snapshot["model_distribution"],
        key_distribution=snapshot["key_distribution"],
        centroid_decisions=snapshot["centroid_decisions"],
        average_latency_ms=snapshot["average_latency_ms"],
        tracked_keys=snapshot["tracked_keys"],
    )


@admin_router.get("/api/v1/routeiq/ui-config")
async def get_ui_config():
    """Return UI configuration for disaggregated deployments.

    The UI can call this endpoint to get feature flags, OIDC config, etc.
    This is useful when the UI is deployed separately from the gateway
    (e.g., on S3+CloudFront, Cloudflare Pages, Vercel) and needs to
    discover gateway capabilities at runtime.
    """
    oidc_enabled = os.environ.get("ROUTEIQ_OIDC_ENABLED", "false").lower() == "true"
    return {
        "version": _VERSION,
        "features": {
            "sso_login": oidc_enabled,
            "model_playground": False,  # future
            "cost_analytics": False,  # future
        },
        "oidc": {
            "enabled": oidc_enabled,
            "login_url": "/sso/login" if oidc_enabled else None,
        },
    }
