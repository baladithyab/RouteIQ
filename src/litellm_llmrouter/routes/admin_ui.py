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

from fastapi import HTTPException
from pydantic import BaseModel

from . import admin_router

# Module-level start time for uptime calculation
_start_time = time.time()

# Version
_VERSION = "0.2.0"


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
    """Get routing decision statistics."""
    strategy_distribution: dict[str, int] = {}

    try:
        from litellm_llmrouter.strategy_registry import get_routing_registry

        registry = get_routing_registry()
        status = registry.get_status()
        # Build strategy distribution from registered strategies
        for name in status.get("registered_strategies", []):
            strategy_distribution[name] = 0
        # If there's an active strategy, mark it
        active = status.get("active_strategy")
        if active and active in strategy_distribution:
            strategy_distribution[active] = 1  # Placeholder
    except Exception:
        pass

    return RoutingStatsResponse(
        total_decisions=0,  # MVP: not yet tracked
        strategy_distribution=strategy_distribution,
        profile_distribution={},  # MVP: not yet tracked
        centroid_decisions=0,  # MVP: not yet tracked
        average_latency_ms=0.0,  # MVP: not yet tracked
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
