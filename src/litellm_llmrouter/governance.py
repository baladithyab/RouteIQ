"""RouteIQ Governance Layer — Workspace Isolation & API Key Governance.

Provides multi-tenant workspace isolation with per-key governance:
- Workspace hierarchy: Organization -> Workspace -> API Key
- Per-key: budget limits, rate limits, model access, config enforcement
- Per-workspace: aggregate budgets, model provisioning, guardrail policies
- Dynamic policies via CRUD API

Architecture:
  This module sits between auth (identity resolution) and routing (model selection).
  It resolves a request's governance context (workspace, key permissions, budgets,
  model allowlist) and enforces limits before the request reaches the router.

Storage:
  Redis for real-time counters (rate limits, spend tracking).
  PostgreSQL for workspace/key metadata (optional -- degrades to in-memory).
  In-memory cache with TTL for hot-path lookups.
"""

from __future__ import annotations

import fnmatch
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from fastapi import HTTPException
from pydantic import BaseModel, Field

from .auth import get_request_id

logger = logging.getLogger("litellm_llmrouter.governance")


# -- Roles ------------------------------------------------------------------


class OrgRole(str, Enum):
    """Roles within an organization."""

    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"


class WorkspaceRole(str, Enum):
    """Roles within a workspace."""

    ADMIN = "ws_admin"
    MANAGER = "ws_manager"
    MEMBER = "ws_member"


# -- Workspace Model --------------------------------------------------------


class WorkspaceConfig(BaseModel):
    """Configuration for a workspace."""

    workspace_id: str
    name: str
    org_id: Optional[str] = None

    # Model access control
    allowed_models: List[str] = Field(
        default_factory=list,
        description=(
            "Model allowlist with wildcard support (e.g., 'gpt-4o*', '@anthropic/*')"
        ),
    )
    blocked_models: List[str] = Field(default_factory=list)

    # Budget limits
    max_budget_usd: Optional[float] = None  # Monthly budget cap
    budget_alert_threshold: float = 0.8  # Alert at 80% of budget

    # Rate limits
    max_rpm: Optional[int] = None  # Requests per minute
    max_tpm: Optional[int] = None  # Tokens per minute

    # Guardrail enforcement
    enforced_guardrails: List[str] = Field(
        default_factory=list,
        description="Guardrail IDs enforced for all requests in this workspace",
    )

    # Default config (attached to all keys in workspace)
    default_routing_profile: Optional[str] = None
    config_override_allowed: bool = True  # Can keys override workspace config?

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[float] = None
    updated_at: Optional[float] = None


# -- API Key Governance -----------------------------------------------------


class KeyGovernance(BaseModel):
    """Governance rules attached to an API key."""

    key_id: str
    workspace_id: Optional[str] = None

    # Scoped permissions (subset of workspace permissions)
    scopes: Set[str] = Field(default_factory=lambda: {"completions.write"})

    # Per-key budget (within workspace budget)
    max_budget_usd: Optional[float] = None
    budget_period: str = "monthly"  # monthly, weekly, daily

    # Per-key rate limits (within workspace limits)
    max_rpm: Optional[int] = None
    max_tpm: Optional[int] = None

    # Model restrictions (intersection with workspace allowlist)
    allowed_models: List[str] = Field(default_factory=list)

    # Config enforcement
    enforced_config: Optional[Dict[str, Any]] = None
    config_override_allowed: bool = True

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


# -- Governance Context -----------------------------------------------------


@dataclass
class GovernanceContext:
    """Resolved governance context for a request.

    Contains the effective (intersected) limits from both workspace and key
    governance rules.  The ``enforce`` method on ``GovernanceEngine`` builds
    this context and verifies all limits before letting the request through.
    """

    workspace_id: Optional[str] = None
    workspace_name: Optional[str] = None
    key_id: Optional[str] = None
    org_id: Optional[str] = None

    # Effective limits (intersection of workspace + key)
    effective_allowed_models: Set[str] = field(default_factory=set)
    effective_blocked_models: Set[str] = field(default_factory=set)
    effective_max_rpm: Optional[int] = None
    effective_max_tpm: Optional[int] = None
    effective_max_budget_usd: Optional[float] = None
    effective_guardrails: List[str] = field(default_factory=list)
    effective_routing_profile: Optional[str] = None

    # Scopes
    scopes: Set[str] = field(default_factory=lambda: {"completions.write"})

    # Status
    budget_remaining_usd: Optional[float] = None
    budget_used_pct: float = 0.0


# -- Governance Engine ------------------------------------------------------


class GovernanceEngine:
    """Resolves and enforces governance rules for requests.

    The engine maintains an in-memory registry of workspaces and per-key
    governance rules.  On each request it:
    1. Resolves the full governance context (workspace + key -> effective limits)
    2. Checks model access, budget, and rate limits
    3. Raises ``HTTPException`` on violation

    The in-memory cache has a configurable TTL to avoid re-computing the
    governance context on every request.
    """

    def __init__(self) -> None:
        self._workspaces: Dict[str, WorkspaceConfig] = {}
        self._key_governance: Dict[str, KeyGovernance] = {}
        self._cache_ttl = 60  # seconds
        self._cache: Dict[str, tuple[float, GovernanceContext]] = {}

    # -- Workspace CRUD -----------------------------------------------------

    def register_workspace(self, config: WorkspaceConfig) -> None:
        """Register or update a workspace configuration."""
        config.updated_at = time.time()
        if config.created_at is None:
            config.created_at = time.time()
        self._workspaces[config.workspace_id] = config
        self._invalidate_cache(workspace_id=config.workspace_id)
        logger.info("Registered workspace %s (%s)", config.workspace_id, config.name)

    def get_workspace(self, workspace_id: str) -> Optional[WorkspaceConfig]:
        """Get a workspace by ID."""
        return self._workspaces.get(workspace_id)

    def list_workspaces(self, org_id: Optional[str] = None) -> List[WorkspaceConfig]:
        """List all workspaces, optionally filtered by org_id."""
        workspaces = list(self._workspaces.values())
        if org_id is not None:
            workspaces = [w for w in workspaces if w.org_id == org_id]
        return workspaces

    def delete_workspace(self, workspace_id: str) -> bool:
        """Delete a workspace.  Returns True if it existed."""
        removed = self._workspaces.pop(workspace_id, None)
        if removed is not None:
            self._invalidate_cache(workspace_id=workspace_id)
            logger.info("Deleted workspace %s", workspace_id)
            return True
        return False

    # -- Key Governance CRUD ------------------------------------------------

    def register_key_governance(self, governance: KeyGovernance) -> None:
        """Register governance rules for an API key."""
        self._key_governance[governance.key_id] = governance
        self._invalidate_cache(key_id=governance.key_id)
        logger.info(
            "Registered key governance for %s (workspace=%s)",
            governance.key_id,
            governance.workspace_id,
        )

    def get_key_governance(self, key_id: str) -> Optional[KeyGovernance]:
        """Get governance rules for a key."""
        return self._key_governance.get(key_id)

    def delete_key_governance(self, key_id: str) -> bool:
        """Delete governance rules for a key.  Returns True if it existed."""
        removed = self._key_governance.pop(key_id, None)
        if removed is not None:
            self._invalidate_cache(key_id=key_id)
            return True
        return False

    # -- Context Resolution -------------------------------------------------

    async def resolve_context(self, key_id: str) -> GovernanceContext:
        """Resolve the full governance context for an API key.

        Combines workspace + key rules with most-restrictive-wins semantics:
        - Allowed models: intersection of workspace and key allowlists
        - Rate limits: minimum of workspace and key limits
        - Budget: minimum of workspace and key budgets
        - Guardrails: union of workspace and key guardrails
        - Routing profile: key overrides workspace (if override allowed)
        """
        # Check cache first
        cached = self._cache.get(key_id)
        if cached is not None:
            ts, ctx = cached
            if time.time() - ts < self._cache_ttl:
                return ctx

        # Build fresh context
        key_gov = self._key_governance.get(key_id)

        ctx = GovernanceContext(key_id=key_id)

        if key_gov is not None:
            ctx.scopes = set(key_gov.scopes)

        # If there is a workspace, load it
        workspace: Optional[WorkspaceConfig] = None
        if key_gov is not None and key_gov.workspace_id:
            workspace = self._workspaces.get(key_gov.workspace_id)

        if workspace is not None:
            ctx.workspace_id = workspace.workspace_id
            ctx.workspace_name = workspace.name
            ctx.org_id = workspace.org_id
            ctx.effective_blocked_models = set(workspace.blocked_models)
            ctx.effective_guardrails = list(workspace.enforced_guardrails)
            ctx.effective_routing_profile = workspace.default_routing_profile

        # -- Model allowlist (intersection) ---------------------------------
        ws_models = set(workspace.allowed_models) if workspace else set()
        key_models = set(key_gov.allowed_models) if key_gov else set()

        if ws_models and key_models:
            # Intersection: a key model is effective only if it also
            # appears (or is covered by a wildcard) in the workspace list.
            effective: Set[str] = set()
            for km in key_models:
                for wm in ws_models:
                    if self._model_matches(km, wm) or self._model_matches(wm, km):
                        effective.add(km)
                        break
            ctx.effective_allowed_models = effective
        elif ws_models:
            ctx.effective_allowed_models = ws_models
        elif key_models:
            ctx.effective_allowed_models = key_models
        # else: both empty -> no restriction (all models allowed)

        # -- Rate limits (most restrictive wins) ----------------------------
        ctx.effective_max_rpm = self._min_optional(
            workspace.max_rpm if workspace else None,
            key_gov.max_rpm if key_gov else None,
        )
        ctx.effective_max_tpm = self._min_optional(
            workspace.max_tpm if workspace else None,
            key_gov.max_tpm if key_gov else None,
        )

        # -- Budget (most restrictive wins) ---------------------------------
        ctx.effective_max_budget_usd = self._min_optional(
            workspace.max_budget_usd if workspace else None,
            key_gov.max_budget_usd if key_gov else None,
        )

        # -- Routing profile (key overrides workspace if allowed) -----------
        if (
            key_gov is not None
            and key_gov.enforced_config
            and key_gov.enforced_config.get("routing_profile")
        ):
            allow_override = True
            if workspace is not None:
                allow_override = workspace.config_override_allowed
            if key_gov.config_override_allowed and allow_override:
                ctx.effective_routing_profile = key_gov.enforced_config[
                    "routing_profile"
                ]

        # -- Guardrails (union) ---------------------------------------------
        if key_gov is not None and key_gov.enforced_config:
            extra = key_gov.enforced_config.get("guardrails", [])
            if isinstance(extra, list):
                existing = set(ctx.effective_guardrails)
                for g in extra:
                    if g not in existing:
                        ctx.effective_guardrails.append(g)

        # Cache and return
        self._cache[key_id] = (time.time(), ctx)

        # Evict oldest 10% if cache exceeds max size
        _MAX_CACHE_SIZE = 10_000
        if len(self._cache) > _MAX_CACHE_SIZE:
            entries = sorted(self._cache.items(), key=lambda x: x[1][0])
            for key, _ in entries[: _MAX_CACHE_SIZE // 10]:
                del self._cache[key]

        return ctx

    # -- Enforcement Checks -------------------------------------------------

    async def check_model_access(self, ctx: GovernanceContext, model: str) -> bool:
        """Check if the resolved context allows access to the specified model.

        Supports wildcards: ``gpt-4o*``, ``@anthropic/*``, ``*``.
        If ``effective_allowed_models`` is empty the model is allowed
        (no restriction).
        """
        # Blocked models always take precedence
        for pattern in ctx.effective_blocked_models:
            if self._model_matches(model, pattern):
                return False

        # If no allowlist, everything (not blocked) is allowed
        if not ctx.effective_allowed_models:
            return True

        for pattern in ctx.effective_allowed_models:
            if self._model_matches(model, pattern):
                return True

        return False

    async def check_budget(self, ctx: GovernanceContext) -> bool:
        """Check if the request is within budget limits.

        Uses Redis for real-time spend tracking when available.
        If Redis is unavailable or no budget is configured, returns True
        (fail-open).
        """
        if ctx.effective_max_budget_usd is None:
            return True

        try:
            current_spend = await self._get_current_spend(ctx)
            if current_spend is None:
                # No spend data available (Redis down or not configured)
                return True

            ctx.budget_remaining_usd = max(
                0.0, ctx.effective_max_budget_usd - current_spend
            )
            ctx.budget_used_pct = (
                current_spend / ctx.effective_max_budget_usd
                if ctx.effective_max_budget_usd > 0
                else 0.0
            )

            return current_spend < ctx.effective_max_budget_usd
        except Exception as e:
            logger.warning("Budget check failed (fail-open): %s", e)
            return True

    async def check_rate_limit(self, ctx: GovernanceContext) -> bool:
        """Check if the request is within rate limits.

        Uses Redis sliding window counters when available.
        If Redis is unavailable or no limits are configured, returns True.
        """
        if ctx.effective_max_rpm is None:
            return True

        try:
            current_rpm = await self._get_current_rpm(ctx)
            if current_rpm is None:
                return True
            return current_rpm < ctx.effective_max_rpm
        except Exception as e:
            logger.warning("Rate limit check failed (fail-open): %s", e)
            return True

    async def enforce(self, key_id: str, model: str) -> GovernanceContext:
        """Full governance enforcement: resolve context, check all limits.

        Raises ``HTTPException`` on violation.

        Args:
            key_id: The API key identifier.
            model: The model being requested.

        Returns:
            The resolved ``GovernanceContext`` if all checks pass.
        """
        request_id = get_request_id() or "unknown"
        ctx = await self.resolve_context(key_id)

        # 1. Model access
        if not await self.check_model_access(ctx, model):
            logger.warning(
                "Governance: model access denied key=%s model=%s workspace=%s",
                key_id,
                model,
                ctx.workspace_id,
            )
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "model_access_denied",
                    "message": (
                        f"Model '{model}' is not allowed under the current "
                        f"workspace governance policy."
                    ),
                    "workspace_id": ctx.workspace_id,
                    "request_id": request_id,
                },
            )

        # 2. Budget
        if not await self.check_budget(ctx):
            logger.warning(
                "Governance: budget exceeded key=%s workspace=%s used_pct=%.1f%%",
                key_id,
                ctx.workspace_id,
                ctx.budget_used_pct * 100,
            )
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "budget_exceeded",
                    "message": (
                        "Workspace budget limit exceeded. "
                        f"Budget used: {ctx.budget_used_pct:.0%}"
                    ),
                    "workspace_id": ctx.workspace_id,
                    "budget_remaining_usd": ctx.budget_remaining_usd,
                    "request_id": request_id,
                },
                headers={"Retry-After": "3600"},
            )

        # 3. Rate limit
        if not await self.check_rate_limit(ctx):
            logger.warning(
                "Governance: rate limit exceeded key=%s workspace=%s",
                key_id,
                ctx.workspace_id,
            )
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "rate_limit_exceeded",
                    "message": (
                        f"Rate limit exceeded. Max RPM: {ctx.effective_max_rpm}"
                    ),
                    "workspace_id": ctx.workspace_id,
                    "request_id": request_id,
                },
                headers={"Retry-After": "60"},
            )

        return ctx

    # -- Cache Invalidation -------------------------------------------------

    def _invalidate_cache(
        self,
        workspace_id: Optional[str] = None,
        key_id: Optional[str] = None,
    ) -> None:
        """Invalidate cached contexts.

        If ``key_id`` is provided, only that entry is invalidated.
        If ``workspace_id`` is provided, all keys in that workspace are
        invalidated.  If neither is provided this is a no-op.
        """
        if key_id is not None:
            self._cache.pop(key_id, None)

        if workspace_id is not None:
            # Invalidate all keys belonging to this workspace
            to_remove: List[str] = []
            for kid, (_, ctx) in self._cache.items():
                if ctx.workspace_id == workspace_id:
                    to_remove.append(kid)
            for kid in to_remove:
                self._cache.pop(kid, None)

            # Also invalidate keys whose governance points to this workspace
            for kid, kg in self._key_governance.items():
                if kg.workspace_id == workspace_id:
                    self._cache.pop(kid, None)

    # -- Model Pattern Matching ---------------------------------------------

    def _model_matches(self, model: str, pattern: str) -> bool:
        """Check if *model* matches an allowlist/blocklist *pattern*.

        Supports:
        - ``*`` — matches everything
        - Glob patterns: ``gpt-4o*``, ``claude-3-*``
        - Provider prefix: ``@anthropic/*`` matches any model name
          containing 'anthropic' (case-insensitive)
        - Exact match (case-insensitive)
        """
        if pattern == "*":
            return True
        # Exact match (case-insensitive) — handles pattern-to-pattern comparison
        if model.lower() == pattern.lower():
            return True
        if pattern.startswith("@"):
            # Provider prefix: @anthropic/* -> match models with provider prefix
            provider = pattern[1:].split("/")[0].lower()
            model_lower = model.lower()
            return model_lower.startswith(provider + "/") or model_lower.startswith(
                provider + "-"
            )
        return fnmatch.fnmatch(model.lower(), pattern.lower())

    # -- Internal Helpers ---------------------------------------------------

    @staticmethod
    def _min_optional(
        a: Optional[int | float], b: Optional[int | float]
    ) -> Optional[int | float]:
        """Return the minimum of two optional numeric values, ignoring Nones."""
        if a is None:
            return b
        if b is None:
            return a
        return min(a, b)

    async def _get_current_spend(self, ctx: GovernanceContext) -> Optional[float]:
        """Retrieve current spend from Redis.

        Returns ``None`` if Redis is unavailable or not configured.
        The Redis key uses the workspace budget namespace so all keys in
        the same workspace share one budget counter.
        """
        try:
            from litellm_llmrouter.redis_pool import get_async_redis_client

            redis = await get_async_redis_client()
            if redis is None:
                return None

            # Workspace-scoped spend key (monthly bucket)
            bucket = int(time.time() // 2_592_000)  # ~30 day buckets
            scope = ctx.workspace_id or ctx.key_id or "global"
            key = f"governance:spend:{scope}:{bucket}"

            value = await redis.get(key)
            return float(value) if value else 0.0
        except Exception:
            return None

    async def _get_current_rpm(self, ctx: GovernanceContext) -> Optional[int]:
        """Retrieve current request count from Redis (per-minute window).

        Returns ``None`` if Redis is unavailable or not configured.
        """
        try:
            from litellm_llmrouter.redis_pool import get_async_redis_client

            redis = await get_async_redis_client()
            if redis is None:
                return None

            bucket = int(time.time() // 60)
            scope = ctx.workspace_id or ctx.key_id or "global"
            key = f"governance:rpm:{scope}:{bucket}"

            value = await redis.get(key)
            return int(value) if value else 0
        except Exception:
            return None

    # -- Status / Introspection ---------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return engine status for health / admin endpoints."""
        return {
            "workspace_count": len(self._workspaces),
            "governed_key_count": len(self._key_governance),
            "cache_entries": len(self._cache),
            "cache_ttl_seconds": self._cache_ttl,
        }


# -- Singleton ---------------------------------------------------------------

_engine: Optional[GovernanceEngine] = None


def get_governance_engine() -> GovernanceEngine:
    """Get or create the governance engine singleton."""
    global _engine
    if _engine is None:
        _engine = GovernanceEngine()
    return _engine


def reset_governance_engine() -> None:
    """Reset the governance engine singleton (for testing)."""
    global _engine
    _engine = None


# -- Exports -----------------------------------------------------------------

__all__ = [
    # Roles
    "OrgRole",
    "WorkspaceRole",
    # Models
    "WorkspaceConfig",
    "KeyGovernance",
    "GovernanceContext",
    # Engine
    "GovernanceEngine",
    "get_governance_engine",
    "reset_governance_engine",
]
