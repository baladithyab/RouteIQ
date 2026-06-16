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
from typing import Any, cast

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
    """Org-wide routing rollups for the admin dashboard + future panels.

    ``cluster_wide`` is ``True`` when the figures are the cross-replica
    aggregate read from the shared store (RouteIQ-78fd), and ``False`` when the
    shared store is disabled/unavailable and the numbers fall back to the
    single serving worker.  The UI surfaces this so a per-worker reading is not
    mistaken for the cluster total.
    """

    total_decisions: int
    strategy_distribution: dict[str, int]
    profile_distribution: dict[str, int]
    model_distribution: dict[str, int]
    key_distribution: dict[str, int]
    centroid_decisions: int
    average_latency_ms: float
    tracked_keys: int
    cluster_wide: bool = False


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


class ModelUpsertRequest(BaseModel):
    """Request body for adding/editing a ``model_list`` entry (admin auth).

    ``litellm_params`` mirrors the LiteLLM config block (must contain at least
    ``model``, e.g. ``"anthropic/claude-3-5-sonnet"``).  ``model_info`` is the
    optional LiteLLM metadata block.
    """

    model_name: str
    litellm_params: dict[str, Any]
    model_info: dict[str, Any] | None = None


class ModelMutationResponse(BaseModel):
    """Result of a model CRUD mutation."""

    model_name: str
    action: str  # "added" | "updated" | "removed"
    model_count: int


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


def _breaker_state_to_status(state: Any) -> str:
    """Map a circuit-breaker state to a model deployment status string.

    closed -> active, half_open -> degraded, open -> unavailable.  Accepts an
    enum (``CircuitBreakerState``) or a plain string; an unknown value maps to
    ``active`` so a model is never falsely reported unavailable on a surprise.
    """
    value = state.value if hasattr(state, "value") else str(state)
    if value == "open":
        return "unavailable"
    if value == "half_open":
        return "degraded"
    return "active"


def _build_provider_status_map() -> dict[str, str]:
    """Resolve a ``{provider: status}`` map from live circuit-breaker state.

    Backs the Model Overview Status (RouteIQ-c8d5) with REAL deployment health
    instead of a cosmetic always-``active``.  RouteIQ keys its per-provider
    breakers as ``provider:<name>`` (``resilience.get_or_create_provider_breaker``),
    so an OPEN breaker for a provider means its models are failing fast and the
    panel should show ``unavailable``.

    Only providers that ALREADY have a breaker are included; a provider with no
    breaker (no failures observed yet) is simply absent from the map and the
    caller defaults it to ``active``.  Fully fail-open: any error yields an
    empty map (every model then defaults to ``active``).
    """
    statuses: dict[str, str] = {}
    try:
        from litellm_llmrouter.resilience import get_circuit_breaker_manager

        manager = get_circuit_breaker_manager()
        breakers = getattr(manager, "_breakers", {}) or {}
        for name, breaker in breakers.items():
            if not name.startswith("provider:"):
                continue
            provider = name.split("provider:", 1)[1]
            statuses[provider] = _breaker_state_to_status(
                getattr(breaker, "state", "closed")
            )
    except Exception:
        return {}
    return statuses


def _resolve_model_status(provider: str, provider_status: dict[str, str]) -> str:
    """Resolve a model's status from its provider's live breaker state."""
    return provider_status.get(provider, "active")


def _model_entry_to_info(
    entry: dict[str, Any], provider_status: dict[str, str]
) -> dict[str, str]:
    """Shape one ``model_list`` entry into the Model Overview info dict.

    Status is derived from the provider's live circuit-breaker state
    (RouteIQ-c8d5) rather than hardcoded ``active``.
    """
    model_name = entry.get("model_name", "unknown")
    litellm_params = entry.get("litellm_params", {}) or {}
    model_id = litellm_params.get("model", "unknown")
    # Prefer an explicit provider override, else parse from the model_id
    # (e.g., "anthropic/claude-3" -> "anthropic").
    provider = litellm_params.get("custom_llm_provider") or (
        model_id.split("/")[0] if "/" in model_id else "unknown"
    )
    return {
        "model_name": model_name,
        "provider": provider,
        "model_id": model_id,
        "status": _resolve_model_status(provider, provider_status),
    }


def _get_models() -> list[dict[str, str]]:
    """Get configured models from LiteLLM proxy config.

    Status reflects REAL deployment health (RouteIQ-c8d5): each model's status
    is derived from its provider's live circuit-breaker state rather than a
    cosmetic always-``active``.
    """
    provider_status = _build_provider_status_map()
    models: list[dict[str, str]] = []
    try:
        import litellm

        if hasattr(litellm, "model_list") and litellm.model_list:
            for entry in litellm.model_list:
                if not isinstance(entry, dict):
                    continue
                entry_dict: dict[str, Any] = dict(entry)
                models.append(_model_entry_to_info(entry_dict, provider_status))
    except Exception:
        pass

    return models


# --- Model CRUD helpers (RouteIQ-eb2d) ---------------------------------------


def _get_model_list() -> list[dict[str, Any]]:
    """Return the live ``litellm.model_list`` (creating it if absent).

    This is the in-memory model catalog the live router reads.  Mutating it is
    the hot path for add/remove/edit; the LiteLLM router is also updated so the
    change takes effect without a restart (config hot-reload).
    """
    import litellm

    if not hasattr(litellm, "model_list") or litellm.model_list is None:
        litellm.model_list = []
    # Return the LIVE list object (callers mutate it in place); cast for mypy
    # since ``litellm.model_list`` is loosely typed.
    return cast("list[dict[str, Any]]", litellm.model_list)


def _find_model_index(model_list: list[dict[str, Any]], model_name: str) -> int:
    """Return the index of the FIRST entry with ``model_name``, or -1."""
    for i, entry in enumerate(model_list):
        if entry.get("model_name") == model_name:
            return i
    return -1


def _sync_router_add(entry: dict[str, Any]) -> None:
    """Best-effort: add/upsert the deployment on the live LiteLLM router.

    Keeps the running router's deployment table in sync with ``model_list`` so
    routing reflects the change immediately (config hot-reload).  Fail-open:
    a missing/older router never blocks the catalog mutation.
    """
    try:
        from litellm.proxy.proxy_server import llm_router

        if llm_router is None:
            return
        from litellm import Deployment

        deployment = Deployment(**entry)
        if hasattr(llm_router, "upsert_deployment"):
            llm_router.upsert_deployment(deployment)
        elif hasattr(llm_router, "add_deployment"):
            llm_router.add_deployment(deployment)
    except Exception:
        # Router sync is best-effort; the model_list mutation is the source of
        # truth and hot-reload / restart will reconcile the router.
        pass


def _sync_router_remove(model_name: str) -> None:
    """Best-effort: delete all deployments for ``model_name`` from the router."""
    try:
        from litellm.proxy.proxy_server import llm_router

        if llm_router is None:
            return
        model_list = getattr(llm_router, "model_list", []) or []
        for entry in list(model_list):
            if entry.get("model_name") != model_name:
                continue
            model_id = (entry.get("model_info") or {}).get("id")
            if model_id and hasattr(llm_router, "delete_deployment"):
                llm_router.delete_deployment(id=model_id)
    except Exception:
        pass


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


@admin_router.post(
    "/api/v1/routeiq/models", response_model=ModelMutationResponse, status_code=201
)
async def add_model(request: ModelUpsertRequest):
    """Add a new model to the catalog (admin auth, RouteIQ-eb2d).

    Appends a new ``model_list`` entry and syncs it onto the live router so the
    new deployment is routable without a restart (config hot-reload).  Rejects
    a duplicate ``model_name`` with 409 and a missing ``litellm_params.model``
    with 400.
    """
    if not request.litellm_params.get("model"):
        raise HTTPException(
            status_code=400,
            detail="litellm_params.model is required (e.g. 'anthropic/claude-3-5-sonnet')",
        )

    model_list = _get_model_list()
    if _find_model_index(model_list, request.model_name) != -1:
        raise HTTPException(
            status_code=409,
            detail=f"Model '{request.model_name}' already exists. Use PUT to edit it.",
        )

    entry: dict[str, Any] = {
        "model_name": request.model_name,
        "litellm_params": dict(request.litellm_params),
    }
    if request.model_info is not None:
        entry["model_info"] = dict(request.model_info)

    model_list.append(entry)
    _sync_router_add(entry)

    return ModelMutationResponse(
        model_name=request.model_name,
        action="added",
        model_count=len(model_list),
    )


@admin_router.put(
    "/api/v1/routeiq/models/{model_name}", response_model=ModelMutationResponse
)
async def update_model(model_name: str, request: ModelUpsertRequest):
    """Edit an existing model_list entry (admin auth, RouteIQ-eb2d).

    Replaces the entry's ``litellm_params`` / ``model_info`` in place and
    re-syncs the live router.  Returns 404 when the model is not in the
    catalog.  A body ``model_name`` that differs from the path is allowed and
    renames the entry.
    """
    if not request.litellm_params.get("model"):
        raise HTTPException(
            status_code=400,
            detail="litellm_params.model is required",
        )

    model_list = _get_model_list()
    idx = _find_model_index(model_list, model_name)
    if idx == -1:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found",
        )

    entry: dict[str, Any] = {
        "model_name": request.model_name,
        "litellm_params": dict(request.litellm_params),
    }
    if request.model_info is not None:
        entry["model_info"] = dict(request.model_info)

    # Router: drop the old deployment(s), add the (possibly renamed) new one.
    _sync_router_remove(model_name)
    model_list[idx] = entry
    _sync_router_add(entry)

    return ModelMutationResponse(
        model_name=request.model_name,
        action="updated",
        model_count=len(model_list),
    )


@admin_router.delete(
    "/api/v1/routeiq/models/{model_name}", response_model=ModelMutationResponse
)
async def delete_model(model_name: str):
    """Remove a model from the catalog (admin auth, RouteIQ-eb2d).

    Drops ALL ``model_list`` entries with the given ``model_name`` and removes
    the matching deployments from the live router.  Returns 404 when no entry
    matches.
    """
    model_list = _get_model_list()
    before = len(model_list)
    remaining = [e for e in model_list if e.get("model_name") != model_name]
    if len(remaining) == before:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found",
        )

    _sync_router_remove(model_name)
    # Mutate the live list in place so any other holder of the reference sees it.
    model_list[:] = remaining

    return ModelMutationResponse(
        model_name=model_name,
        action="removed",
        model_count=len(model_list),
    )


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

    Reads the CLUSTER-WIDE aggregate (RouteIQ-78fd): every worker mirrors its
    decision counters into a shared Redis store, and this endpoint sums them
    back across all replicas.  When the shared store is disabled/unavailable it
    falls back to the per-worker accumulator and flags ``cluster_wide=False``.
    """
    from litellm_llmrouter.router_decision_callback import cluster_global_snapshot

    snapshot = await cluster_global_snapshot()
    return GlobalStatsResponse(
        total_decisions=snapshot["total_decisions"],
        strategy_distribution=snapshot["strategy_distribution"],
        profile_distribution=snapshot["profile_distribution"],
        model_distribution=snapshot["model_distribution"],
        key_distribution=snapshot["key_distribution"],
        centroid_decisions=snapshot["centroid_decisions"],
        average_latency_ms=snapshot["average_latency_ms"],
        tracked_keys=snapshot["tracked_keys"],
        cluster_wide=snapshot.get("cluster_wide", False),
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
