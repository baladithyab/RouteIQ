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
import json
import logging
import os
import time
from collections.abc import Mapping
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


# -- Organization Model -----------------------------------------------------


class OrgConfig(BaseModel):
    """First-class organization entity (top of the Org -> Workspace -> Key tree).

    Before P4 the org was only a string field on ``WorkspaceConfig.org_id``;
    this model makes the org a durable first-class row (Aurora-backed) so the
    full hierarchy the roadmap names is persistable.  Kept intentionally small
    -- workspaces still carry the ``org_id`` soft-reference; this just lets an
    org carry a name + metadata of its own.
    """

    org_id: str
    name: str
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


# -- Canonical scope derivation (single source of truth for WRITE == READ) ---


def derive_spend_scope_from_ctx(
    ctx: "GovernanceContext | Mapping[str, Any]",
) -> tuple[str, str]:
    """Derive the canonical ``(scope, scope_type)`` for a governance context.

    This is the SINGLE source of truth for the spend/RPM scope token used by
    BOTH the READ path (:meth:`GovernanceEngine._get_current_spend` /
    :meth:`_get_current_rpm`) AND the WRITE path
    (``router_decision_callback._derive_spend_scope`` ->
    ``record_governance_spend``).  Keeping one helper guarantees the write key
    (``governance:spend:{scope}:{bucket}``) is byte-identical to the read key,
    so workspace (RouteIQ-ed7a) and key (RouteIQ-08dd) budgets are actually
    enforced instead of silently fail-open.

    Accepts EITHER a :class:`GovernanceContext` (the READ path) OR the dict
    stamp written into ``metadata["_governance_ctx"]`` (the WRITE path), so both
    sides call this ONE code path instead of hand-copying the precedence inline
    (RouteIQ-9738 — eliminates the drift class by construction).

    Precedence (most-specific-wins, identical on both sides):
    ``workspace_id`` -> ``key_id`` -> ``"global"``.  ``scope_type`` is
    ``"workspace"`` when a workspace is resolved, else ``"key"`` when a key is
    resolved, else ``"global"``.  ``ctx.key_id`` is the RAW api_key the enforce
    path resolves (NOT a hash / user-id), so the write must reuse the same raw
    token for the read to see it.
    """
    if isinstance(ctx, Mapping):
        workspace_id = ctx.get("workspace_id")
        key_id = ctx.get("key_id")
    else:
        workspace_id = ctx.workspace_id
        key_id = ctx.key_id
    if workspace_id:
        return str(workspace_id), "workspace"
    if key_id:
        return str(key_id), "key"
    return "global", "global"


def _governance_fail_mode_closed() -> bool:
    """Return True when governance enforcement should fail CLOSED (deny on store down).

    Honours TWO equivalent operator controls (either selects fail-closed):
      * the canonical nested setting ``settings.governance.fail_mode`` (ADR-0013,
        env ``ROUTEIQ_GOVERNANCE__FAIL_MODE``), read via ``get_settings``, and
      * the legacy flat ``ROUTEIQ_GOVERNANCE_FAIL_MODE`` env var the seed names,
        read via ``os.getenv`` (mirrors ``QuotaConfig.from_env``'s flat fallback;
        pydantic-settings does NOT map the flat form onto the nested field).

    Default is OPEN (back-compat): returns True only when an operator has
    explicitly selected ``closed`` through either control.  A settings failure
    fails OPEN -- it must NOT silently start denying traffic.
    """
    # Legacy flat env: explicit operator opt-in to fail-closed.
    if os.getenv("ROUTEIQ_GOVERNANCE_FAIL_MODE", "open").lower() == "closed":
        return True
    try:
        from litellm_llmrouter.settings import GovernanceFailMode, get_settings

        return get_settings().governance.fail_mode == GovernanceFailMode.CLOSED
    except Exception:
        return False


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
        self._orgs: Dict[str, OrgConfig] = {}
        self._workspaces: Dict[str, WorkspaceConfig] = {}
        self._key_governance: Dict[str, KeyGovernance] = {}
        self._cache_ttl = 60  # seconds
        self._cache: Dict[str, tuple[float, GovernanceContext]] = {}

    # -- Org CRUD -----------------------------------------------------------

    def register_org(self, config: OrgConfig) -> None:
        """Register or update an organization (top of the hierarchy)."""
        config.updated_at = time.time()
        if config.created_at is None:
            config.created_at = time.time()
        self._orgs[config.org_id] = config
        logger.info("Registered org %s (%s)", config.org_id, config.name)

    def get_org(self, org_id: str) -> Optional[OrgConfig]:
        """Get an organization by ID."""
        return self._orgs.get(org_id)

    def list_orgs(self) -> List[OrgConfig]:
        """List all organizations."""
        return list(self._orgs.values())

    def delete_org(self, org_id: str) -> bool:
        """Delete an organization.  Returns True if it existed.

        Note: this does NOT cascade to workspaces (soft-FK model); orphaned
        workspaces keep their ``org_id`` string reference.
        """
        return self._orgs.pop(org_id, None) is not None

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

        Fail behaviour when the spend store cannot confirm current spend (Redis
        down / not configured) is governed by ``settings.governance.fail_mode``
        (RouteIQ-24fc).  When a budget IS configured but the store is
        unavailable: ``open`` (default, back-compat) allows; ``closed`` denies so
        a store outage cannot leak spend past the budget.  When NO budget is
        configured (``effective_max_budget_usd is None``) the request is always
        allowed -- fail-closed must NOT deny when there is no limit to enforce.
        """
        if ctx.effective_max_budget_usd is None:
            return True

        try:
            current_spend = await self._get_current_spend(ctx)
            if current_spend is None:
                # Store unavailable (Redis down / not configured) and a budget
                # IS set -> fail-closed denies, fail-open (default) allows.
                if _governance_fail_mode_closed():
                    logger.warning(
                        "Governance: budget store unavailable, denying "
                        "(fail-closed) workspace=%s key=%s",
                        ctx.workspace_id,
                        ctx.key_id,
                    )
                    return False
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
            # A budget IS configured (early-returned above otherwise) and the
            # store raised -> honour fail_mode (deny when closed).
            if _governance_fail_mode_closed():
                logger.warning(
                    "Governance: budget check errored, denying (fail-closed): %s",
                    e,
                )
                return False
            logger.warning("Budget check failed (fail-open): %s", e)
            return True

    async def check_rate_limit(self, ctx: GovernanceContext) -> bool:
        """Check if the request is within rate limits.

        Uses Redis sliding window counters when available.

        Fail behaviour when the store cannot confirm current RPM mirrors
        :meth:`check_budget` (RouteIQ-24fc): when an RPM limit IS configured but
        the store is unavailable, ``settings.governance.fail_mode == closed``
        denies while ``open`` (default) allows.  When NO RPM limit is configured
        the request is always allowed (fail-closed must not deny without a limit).
        """
        if ctx.effective_max_rpm is None:
            return True

        try:
            current_rpm = await self._get_current_rpm(ctx)
            if current_rpm is None:
                # Store unavailable and an RPM limit IS set -> honour fail_mode.
                if _governance_fail_mode_closed():
                    logger.warning(
                        "Governance: rate-limit store unavailable, denying "
                        "(fail-closed) workspace=%s key=%s",
                        ctx.workspace_id,
                        ctx.key_id,
                    )
                    return False
                return True
            return current_rpm < ctx.effective_max_rpm
        except Exception as e:
            if _governance_fail_mode_closed():
                logger.warning(
                    "Governance: rate-limit check errored, denying (fail-closed): %s",
                    e,
                )
                return False
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

            # Workspace-scoped spend key (epoch-aligned ~30-day bucket).
            # Note: Budget periods are aligned to Unix epoch 30-day windows
            # (2,592,000 seconds), NOT calendar months.  This means budget
            # resets occur at fixed intervals from epoch, not on the 1st of
            # each month.  This is intentional for simplicity and consistency
            # across timezones.
            bucket = int(time.time() // 2_592_000)  # ~30 day buckets
            scope, _scope_type = derive_spend_scope_from_ctx(ctx)
            key = f"governance:spend:{scope}:{bucket}"

            value = await redis.get(key)
            if value:
                return float(value)

            # Redis returned no hot counter (e.g. cache flush / node replacement).
            # Fall back to the durable Aurora rollup when the store is enabled so
            # budgets survive a Redis loss.  Epoch-aligned period_start matches the
            # Redis bucket (do NOT switch to calendar months -- byte-compatible).
            durable = await self._get_durable_spend(scope, bucket)
            if durable is not None:
                return durable
            return 0.0
        except Exception:
            return None

    async def _get_durable_spend(self, scope: str, bucket: int) -> Optional[float]:
        """Read the durable Aurora spend rollup for *scope* at *bucket*.

        Returns ``None`` when the governance store is disabled (no DATABASE_URL)
        or on any error -- callers degrade to the Redis/0.0 path.  The
        ``period_start`` is the epoch-aligned bucket start so it is byte-
        compatible with the Redis spend key bucketing.
        """
        try:
            from datetime import datetime, timezone

            from .governance_store import get_governance_store

            store = get_governance_store()
            if not store.enabled:
                return None
            period_start = datetime.fromtimestamp(bucket * 2_592_000, tz=timezone.utc)
            return await store.get_spend(scope, "monthly", period_start)
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
            scope, _scope_type = derive_spend_scope_from_ctx(ctx)
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


# -- Spend Write Path (ElastiCache hot counter + Aurora durable rollup) ------


async def record_governance_spend(
    scope: str,
    scope_type: str,
    *,
    cost: float = 0.0,
    tokens: int = 0,
    requests: int = 1,
) -> None:
    """Record post-response spend for a governance *scope*.

    This is the WRITE path for the ``governance:spend:`` / ``governance:rpm:``
    keys that :meth:`GovernanceEngine._get_current_spend` /
    :meth:`_get_current_rpm` already READ.  Before P4 nothing wrote these keys,
    so durable budget + RPM enforcement was inert (always saw 0.0 -> fail-open).

    Two stores, both fail-open (never break the response path):
      * ElastiCache (hot, in-window): ``INCRBYFLOAT governance:spend:{scope}:{bucket}``
        + ``EXPIRE`` ~30d, and ``INCR governance:rpm:{scope}:{minute_bucket}``
        + ``EXPIRE 60``.  Key shape is byte-identical to the read side.
      * Aurora (durable rollup, system-of-record): ``store.record_spend(...)``
        accumulating into ``governance_spend`` (only when the store is enabled).

    NEVER log cost/token VALUES (PII-adjacent); counts only, at debug.
    ``scope`` is the workspace_id / key_id / "global" (governance scope
    precedence); ``scope_type`` is "workspace" | "key" | "global".
    """
    # 1. ElastiCache hot counters (byte-identical key shape to the read path).
    spend_bucket = int(time.time() // 2_592_000)
    minute_bucket = int(time.time() // 60)
    spend_key = f"governance:spend:{scope}:{spend_bucket}"
    rpm_key = f"governance:rpm:{scope}:{minute_bucket}"
    try:
        from litellm_llmrouter.redis_pool import get_async_redis_client

        redis = await get_async_redis_client()
        if redis is not None:
            if cost and cost > 0:
                await redis.incrbyfloat(spend_key, float(cost))
                await redis.expire(spend_key, 2_592_000)
            if requests and requests > 0:
                await redis.incrby(rpm_key, int(requests))
                await redis.expire(rpm_key, 60)
    except Exception as exc:  # fail-open: never break the response path
        logger.debug("Governance spend redis write skipped: %s", exc)

    # 2. Aurora durable rollup (epoch-aligned period_start == the Redis bucket).
    try:
        from datetime import datetime, timezone

        from .governance_store import get_governance_store

        store = get_governance_store()
        if store.enabled:
            period_start = datetime.fromtimestamp(
                spend_bucket * 2_592_000, tz=timezone.utc
            )
            await store.record_spend(
                scope,
                scope_type,
                "monthly",
                period_start,
                cost=float(cost or 0.0),
                tokens=int(tokens or 0),
                requests=int(requests or 0),
            )
    except Exception as exc:  # fail-open
        logger.debug("Governance spend durable rollup skipped: %s", exc)


# -- File-Based Persistence --------------------------------------------------

_GOVERNANCE_STATE_PATH = os.getenv("ROUTEIQ_GOVERNANCE_STATE_PATH", "")


def save_governance_state(engine: GovernanceEngine) -> None:
    """Save governance state to file for persistence across restarts."""
    if not _GOVERNANCE_STATE_PATH:
        return
    state = {
        "workspaces": {k: v.model_dump() for k, v in engine._workspaces.items()},
        "key_governance": {
            k: v.model_dump() for k, v in engine._key_governance.items()
        },
    }
    try:
        with open(_GOVERNANCE_STATE_PATH, "w") as f:
            json.dump(state, f, indent=2, default=str)
        logger.debug("Governance state saved to %s", _GOVERNANCE_STATE_PATH)
    except Exception as exc:
        logger.warning("Failed to save governance state: %s", exc)


def load_governance_state(engine: GovernanceEngine) -> int:
    """Load governance state from file. Returns count of items loaded."""
    if not _GOVERNANCE_STATE_PATH or not os.path.exists(_GOVERNANCE_STATE_PATH):
        return 0
    try:
        with open(_GOVERNANCE_STATE_PATH) as f:
            state = json.load(f)
        count = 0
        for ws_data in state.get("workspaces", {}).values():
            engine.register_workspace(WorkspaceConfig(**ws_data))
            count += 1
        for kg_data in state.get("key_governance", {}).values():
            engine.register_key_governance(KeyGovernance(**kg_data))
            count += 1
        logger.info("Loaded %d governance items from %s", count, _GOVERNANCE_STATE_PATH)
        return count
    except Exception as exc:
        logger.warning("Failed to load governance state: %s", exc)
        return 0


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
    "OrgConfig",
    "WorkspaceConfig",
    "KeyGovernance",
    "GovernanceContext",
    # Engine
    "GovernanceEngine",
    "get_governance_engine",
    "reset_governance_engine",
    # Spend write path
    "record_governance_spend",
    # Canonical scope derivation (shared by WRITE + READ paths)
    "derive_spend_scope_from_ctx",
    # Persistence
    "save_governance_state",
    "load_governance_state",
]
