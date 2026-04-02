"""Dynamic Usage & Rate Limit Policy Engine for RouteIQ.

Provides condition-based, group-by policies for fine-grained budget and rate
limit enforcement. Unlike static per-key quotas, policies can match on any
request dimension (api_key, metadata.*, model, provider, workspace) and
enforce limits per unique combination of group-by dimensions.

Example policies:
  1. Per-user monthly budget: condition={}, group_by=["metadata._user"],
     limit={"type": "cost", "value": 50.0, "period": "monthly"}
  2. Per-team per-model RPM: condition={"metadata._team": "engineering"},
     group_by=["metadata._team", "model"], limit={"type": "rpm", "value": 100}
  3. Per-provider token budget: condition={"provider": "anthropic"},
     group_by=["provider"], limit={"type": "tokens", "value": 1000000, "period": "daily"}

Storage: Redis for counters, in-memory for policy definitions.

Configuration:
    Environment variables:
    - ROUTEIQ_USAGE_POLICIES_ENABLED: Enable usage policies (default: false)
    - ROUTEIQ_USAGE_POLICIES_FAIL_MODE: "open" (default) or "closed"
    - REDIS_HOST / REDIS_PORT: Redis for counter storage

Usage:
    from litellm_llmrouter.usage_policies import (
        get_usage_policy_engine,
        UsagePolicy,
        LimitType,
        LimitPeriod,
    )

    engine = get_usage_policy_engine()
    engine.add_policy(UsagePolicy(
        policy_id="per-user-monthly-budget",
        name="Per-user monthly budget",
        conditions={},
        group_by=["metadata._user"],
        limit_type=LimitType.COST,
        limit_value=50.0,
        limit_period=LimitPeriod.MONTH,
    ))
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel, Field

logger = logging.getLogger("litellm_llmrouter.usage_policies")


# =============================================================================
# Enums
# =============================================================================


class LimitType(str, Enum):
    """Type of resource being limited."""

    COST = "cost"  # USD budget
    TOKENS = "tokens"  # Token count (total)
    INPUT_TOKENS = "input_tokens"  # Input token count
    OUTPUT_TOKENS = "output_tokens"  # Output token count
    REQUESTS = "requests"  # Request count (RPM/RPH/RPD)


class LimitPeriod(str, Enum):
    """Time window for limit enforcement."""

    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    NONE = "none"  # Lifetime / no reset

    @property
    def seconds(self) -> int:
        """Duration in seconds (used for Redis TTL)."""
        return {
            LimitPeriod.MINUTE: 60,
            LimitPeriod.HOUR: 3600,
            LimitPeriod.DAY: 86400,
            LimitPeriod.WEEK: 604800,
            LimitPeriod.MONTH: 2592000,  # 30 days
            LimitPeriod.NONE: 0,  # No expiry
        }[self]

    @property
    def bucket_divisor(self) -> int:
        """Divisor for computing time-bucket keys."""
        return {
            LimitPeriod.MINUTE: 60,
            LimitPeriod.HOUR: 3600,
            LimitPeriod.DAY: 86400,
            LimitPeriod.WEEK: 604800,
            LimitPeriod.MONTH: 2592000,
            LimitPeriod.NONE: 1,  # Single bucket (lifetime)
        }[self]


class PolicyAction(str, Enum):
    """What happens when a limit is exceeded."""

    DENY = "deny"  # Block the request (429)
    LOG = "log"  # Log but allow
    ALERT = "alert"  # Allow + trigger alert callback


# =============================================================================
# Policy Model
# =============================================================================


class UsagePolicy(BaseModel):
    """A usage or rate limit policy definition."""

    policy_id: str
    name: str
    description: str = ""
    enabled: bool = True

    # Conditions — when does this policy apply?
    # All conditions must match (AND logic). Values support fnmatch wildcards.
    # Keys support dotted paths, e.g. "metadata._team", "model", "provider".
    conditions: dict[str, str] = Field(default_factory=dict)

    # Exclusions — skip this policy for matching requests.
    # Any matching exclusion exempts the request (OR logic).
    exclusions: dict[str, str] = Field(default_factory=dict)

    # Group-by — enforce limits per unique combination of dimension values.
    # e.g. ["metadata._user", "model"] → separate counter per user+model pair.
    group_by: list[str] = Field(default_factory=list)

    # Limit definition
    limit_type: LimitType = LimitType.REQUESTS
    limit_value: float = 100
    limit_period: LimitPeriod = LimitPeriod.MINUTE

    # Action on limit exceeded
    action: PolicyAction = PolicyAction.DENY

    # Alert threshold (0–1, fraction of limit). When usage crosses this
    # fraction an alert_triggered flag is set in the evaluation result.
    alert_threshold: float = 0.8

    # Priority (lower = evaluated first)
    priority: int = 100

    # Workspace scoping (None = global policy)
    workspace_id: str | None = None

    # Timestamps
    created_at: float | None = None
    updated_at: float | None = None


# =============================================================================
# Evaluation Result
# =============================================================================


@dataclass
class PolicyEvaluation:
    """Result of evaluating a single policy against a request."""

    policy_id: str
    policy_name: str
    matched: bool
    group_key: str = ""
    current_usage: float = 0.0
    limit_value: float = 0.0
    usage_pct: float = 0.0
    action: PolicyAction = PolicyAction.DENY
    exceeded: bool = False
    alert_triggered: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise for API responses."""
        return {
            "policy_id": self.policy_id,
            "policy_name": self.policy_name,
            "matched": self.matched,
            "group_key": self.group_key,
            "current_usage": self.current_usage,
            "limit_value": self.limit_value,
            "usage_pct": round(self.usage_pct, 4),
            "action": self.action.value,
            "exceeded": self.exceeded,
            "alert_triggered": self.alert_triggered,
            "error": self.error,
        }


# =============================================================================
# Redis Lua Scripts
# =============================================================================

# Atomic check-current + optional increment for float counters.
# Returns: [current_value_str, is_allowed (1/0), ttl]
_CHECK_AND_INCREMENT_LUA = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local increment = tonumber(ARGV[2])
local window_seconds = tonumber(ARGV[3])
local dry_run = tonumber(ARGV[4])

local current = tonumber(redis.call('GET', key) or '0')

if current + increment > limit then
    local ttl = redis.call('TTL', key)
    if ttl < 0 then ttl = window_seconds end
    return {tostring(current), 0, ttl}
end

if dry_run == 0 then
    local new_val = current + increment
    redis.call('SET', key, tostring(new_val))
    local existing_ttl = redis.call('TTL', key)
    if existing_ttl < 0 and window_seconds > 0 then
        redis.call('EXPIRE', key, window_seconds)
    end
    current = new_val
end

local ttl = redis.call('TTL', key)
if ttl < 0 then ttl = window_seconds end
return {tostring(current), 1, ttl}
"""

# Read-only counter fetch (no increment).
_READ_COUNTER_LUA = """
local key = KEYS[1]
local current = tonumber(redis.call('GET', key) or '0')
local ttl = redis.call('TTL', key)
return {tostring(current), ttl}
"""

# Increment counter unconditionally (used post-response for recording usage).
_INCREMENT_LUA = """
local key = KEYS[1]
local increment = tonumber(ARGV[1])
local window_seconds = tonumber(ARGV[2])

local new_val = tonumber(redis.call('GET', key) or '0') + increment
redis.call('SET', key, tostring(new_val))

local existing_ttl = redis.call('TTL', key)
if existing_ttl < 0 and window_seconds > 0 then
    redis.call('EXPIRE', key, window_seconds)
end

return tostring(new_val)
"""


# =============================================================================
# Usage Policy Engine
# =============================================================================


class UsagePolicyEngine:
    """Evaluates and enforces usage policies against request contexts.

    Policies are stored in-memory and can be managed via CRUD operations.
    Counters are stored in Redis with automatic time-bucket expiry.
    """

    def __init__(self) -> None:
        self._policies: dict[str, UsagePolicy] = {}
        self._redis: Any = None
        self._check_incr_script: Any = None
        self._read_script: Any = None
        self._incr_script: Any = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Redis bootstrap
    # ------------------------------------------------------------------

    async def _get_redis(self) -> Any:
        """Lazily create the Redis connection and register Lua scripts."""
        if self._redis is None:
            async with self._lock:
                if self._redis is None:
                    try:
                        from litellm_llmrouter.redis_pool import (
                            get_async_redis_client,
                        )

                        self._redis = await get_async_redis_client()
                        if self._redis is not None:
                            self._check_incr_script = self._redis.register_script(
                                _CHECK_AND_INCREMENT_LUA
                            )
                            self._read_script = self._redis.register_script(
                                _READ_COUNTER_LUA
                            )
                            self._incr_script = self._redis.register_script(
                                _INCREMENT_LUA
                            )
                    except Exception as e:
                        logger.warning("Usage policy engine: Redis unavailable: %s", e)
                        self._redis = None
        return self._redis

    # ------------------------------------------------------------------
    # Policy CRUD
    # ------------------------------------------------------------------

    def add_policy(self, policy: UsagePolicy) -> None:
        """Add or update a policy."""
        now = time.time()
        policy.updated_at = now
        if policy.created_at is None:
            policy.created_at = now
        self._policies[policy.policy_id] = policy
        logger.info(
            "Usage policy added/updated: %s (%s)", policy.policy_id, policy.name
        )

    def remove_policy(self, policy_id: str) -> bool:
        """Remove a policy. Returns True if it was found and removed."""
        removed = self._policies.pop(policy_id, None) is not None
        if removed:
            logger.info("Usage policy removed: %s", policy_id)
        return removed

    def get_policy(self, policy_id: str) -> UsagePolicy | None:
        """Get a single policy by ID."""
        return self._policies.get(policy_id)

    def list_policies(self, workspace_id: str | None = None) -> list[UsagePolicy]:
        """List all policies, optionally filtered by workspace.

        Global policies (workspace_id=None) are always included when filtering
        by workspace.
        """
        policies = sorted(self._policies.values(), key=lambda p: p.priority)
        if workspace_id is not None:
            policies = [p for p in policies if p.workspace_id in (workspace_id, None)]
        return policies

    # ------------------------------------------------------------------
    # Pre-request evaluation
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        request_context: dict[str, Any],
        *,
        dry_run: bool = False,
    ) -> list[PolicyEvaluation]:
        """Evaluate all matching policies against a request context.

        For ``REQUESTS``-type policies the counter is atomically incremented
        during evaluation (unless *dry_run* is True).  For ``COST`` / ``TOKENS``
        policies the counter is read-only here — actual usage is recorded later
        via :meth:`record_usage`.

        Args:
            request_context: Dict containing ``api_key``, ``model``,
                ``provider``, ``workspace_id``, ``metadata`` (dict with
                ``_user``, ``_team``, etc.)
            dry_run: If True, check limits without incrementing counters.

        Returns:
            List of :class:`PolicyEvaluation` for every matching policy.
        """
        results: list[PolicyEvaluation] = []
        redis = await self._get_redis()

        for policy in sorted(self._policies.values(), key=lambda p: p.priority):
            if not policy.enabled:
                continue

            # Check conditions
            if not self._matches_conditions(policy, request_context):
                continue

            group_key = self._build_group_key(policy, request_context)
            redis_key = self._get_redis_key(
                policy.policy_id, group_key, policy.limit_period
            )

            evaluation = PolicyEvaluation(
                policy_id=policy.policy_id,
                policy_name=policy.name,
                matched=True,
                group_key=group_key,
                limit_value=policy.limit_value,
                action=policy.action,
            )

            # For REQUEST-type limits, we increment atomically during evaluation.
            # For COST / TOKEN limits, we only read the current counter.
            is_request_limit = policy.limit_type == LimitType.REQUESTS

            if redis is None:
                # No Redis — fail open or closed based on env config
                fail_mode = os.getenv(
                    "ROUTEIQ_USAGE_POLICIES_FAIL_MODE", "open"
                ).lower()
                if fail_mode == "closed":
                    evaluation.exceeded = True
                    evaluation.error = "Redis unavailable (fail-closed)"
                else:
                    evaluation.error = "Redis unavailable (fail-open)"
                results.append(evaluation)
                continue

            try:
                if is_request_limit:
                    # Atomic check + increment (or dry-run read)
                    raw = await self._check_incr_script(
                        keys=[redis_key],
                        args=[
                            policy.limit_value,
                            1,  # increment by 1 request
                            policy.limit_period.seconds,
                            1 if dry_run else 0,
                        ],
                    )
                    current = float(raw[0])
                    allowed = int(raw[1]) == 1
                    evaluation.current_usage = current
                    evaluation.exceeded = not allowed
                else:
                    # Read-only for cost / token limits
                    raw = await self._read_script(
                        keys=[redis_key],
                        args=[],
                    )
                    current = float(raw[0])
                    evaluation.current_usage = current
                    evaluation.exceeded = current >= policy.limit_value

                # Compute usage percentage and alert flag
                if policy.limit_value > 0:
                    evaluation.usage_pct = current / policy.limit_value
                    evaluation.alert_triggered = (
                        evaluation.usage_pct >= policy.alert_threshold
                    )

            except Exception as e:
                logger.error(
                    "Usage policy evaluation error (policy=%s): %s",
                    policy.policy_id,
                    e,
                )
                evaluation.error = str(e)
                fail_mode = os.getenv(
                    "ROUTEIQ_USAGE_POLICIES_FAIL_MODE", "open"
                ).lower()
                if fail_mode == "closed":
                    evaluation.exceeded = True

            results.append(evaluation)

        return results

    # ------------------------------------------------------------------
    # Post-response usage recording
    # ------------------------------------------------------------------

    async def record_usage(
        self,
        request_context: dict[str, Any],
        *,
        tokens: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost: float = 0.0,
    ) -> int:
        """Record usage for all matching policies after a request completes.

        Increments counters for COST and TOKEN-type policies.  REQUEST-type
        counters were already incremented during :meth:`evaluate`.

        Returns:
            Number of policy counters updated.
        """
        redis = await self._get_redis()
        if redis is None:
            return 0

        updated = 0
        for policy in self._policies.values():
            if not policy.enabled:
                continue
            if not self._matches_conditions(policy, request_context):
                continue

            # Determine the increment value based on limit type
            increment: float | None = None
            if policy.limit_type == LimitType.COST:
                increment = cost
            elif policy.limit_type == LimitType.TOKENS:
                increment = float(tokens or (input_tokens + output_tokens))
            elif policy.limit_type == LimitType.INPUT_TOKENS:
                increment = float(input_tokens)
            elif policy.limit_type == LimitType.OUTPUT_TOKENS:
                increment = float(output_tokens)
            elif policy.limit_type == LimitType.REQUESTS:
                # Already incremented during evaluate()
                continue

            if increment is None or increment <= 0:
                continue

            group_key = self._build_group_key(policy, request_context)
            redis_key = self._get_redis_key(
                policy.policy_id, group_key, policy.limit_period
            )

            try:
                await self._incr_script(
                    keys=[redis_key],
                    args=[increment, policy.limit_period.seconds],
                )
                updated += 1
            except Exception as e:
                logger.error(
                    "Usage policy record_usage error (policy=%s): %s",
                    policy.policy_id,
                    e,
                )

        return updated

    # ------------------------------------------------------------------
    # Usage queries
    # ------------------------------------------------------------------

    async def get_usage(self, policy_id: str, group_key: str = "") -> float:
        """Get current usage counter for a policy/group combination."""
        policy = self._policies.get(policy_id)
        if policy is None:
            return 0.0

        if not group_key:
            group_key = "__global__"

        redis = await self._get_redis()
        if redis is None:
            return 0.0

        redis_key = self._get_redis_key(policy_id, group_key, policy.limit_period)
        try:
            raw = await self._read_script(keys=[redis_key], args=[])
            return float(raw[0])
        except Exception as e:
            logger.error("get_usage error (policy=%s): %s", policy_id, e)
            return 0.0

    async def get_policy_usage_summary(self, policy_id: str) -> dict[str, Any]:
        """Get a summary of usage across all known groups for a policy.

        Note: Since group keys are dynamic, we can only report usage for
        groups that the engine has seen. For a full scan you'd need to use
        Redis SCAN with the policy prefix — this method is a best-effort
        helper that uses the ``__global__`` group key.
        """
        policy = self._policies.get(policy_id)
        if policy is None:
            return {"error": "policy_not_found"}

        global_usage = await self.get_usage(policy_id, "__global__")
        return {
            "policy_id": policy_id,
            "policy_name": policy.name,
            "limit_type": policy.limit_type.value,
            "limit_value": policy.limit_value,
            "limit_period": policy.limit_period.value,
            "global_usage": global_usage,
            "usage_pct": (
                round(global_usage / policy.limit_value, 4)
                if policy.limit_value > 0
                else 0.0
            ),
        }

    async def reset_usage(self, policy_id: str, group_key: str | None = None) -> bool:
        """Reset usage counters for a policy or specific group.

        If *group_key* is ``None``, resets the ``__global__`` group.
        Returns True if the key was deleted.
        """
        policy = self._policies.get(policy_id)
        if policy is None:
            return False

        gk = group_key if group_key else "__global__"
        redis = await self._get_redis()
        if redis is None:
            return False

        redis_key = self._get_redis_key(policy_id, gk, policy.limit_period)
        try:
            deleted = await redis.delete(redis_key)
            return deleted > 0
        except Exception as e:
            logger.error("reset_usage error (policy=%s): %s", policy_id, e)
            return False

    # ------------------------------------------------------------------
    # Condition matching
    # ------------------------------------------------------------------

    def _matches_conditions(self, policy: UsagePolicy, context: dict[str, Any]) -> bool:
        """Check if all policy conditions match AND no exclusion matches."""
        # All conditions must match (AND)
        for key, pattern in policy.conditions.items():
            value = self._resolve_context_value(context, key)
            if value is None:
                return False
            if not fnmatch.fnmatch(str(value).lower(), pattern.lower()):
                return False

        # Any exclusion match exempts the request
        for key, pattern in policy.exclusions.items():
            value = self._resolve_context_value(context, key)
            if value is not None and fnmatch.fnmatch(
                str(value).lower(), pattern.lower()
            ):
                return False  # Excluded

        return True

    # ------------------------------------------------------------------
    # Context / key helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_context_value(context: dict[str, Any], key: str) -> str | None:
        """Resolve a dotted key path from the request context.

        Examples:
            "model" → context["model"]
            "metadata._user" → context["metadata"]["_user"]
            "metadata.custom.tier" → context["metadata"]["custom"]["tier"]
        """
        parts = key.split(".")
        current: Any = context
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return str(current) if current is not None else None

    def _build_group_key(self, policy: UsagePolicy, context: dict[str, Any]) -> str:
        """Build a group key from the policy's group_by dimensions."""
        if not policy.group_by:
            return "__global__"
        parts: list[str] = []
        for dim in policy.group_by:
            val = self._resolve_context_value(context, dim) or "__none__"
            # URL-encode to prevent delimiter injection
            parts.append(f"{dim}={quote(str(val), safe='')}")
        return "|".join(parts)

    @staticmethod
    def _get_redis_key(policy_id: str, group_key: str, period: LimitPeriod) -> str:
        """Build Redis key with time-bucket suffix for automatic rollover.

        Key pattern:
            routeiq:usage:{policy_id}:{group_key}:{bucket}

        The bucket is computed as ``int(now / period_seconds)`` so that
        counters roll over automatically when the period elapses.  For
        ``LimitPeriod.NONE`` (lifetime), the bucket is always ``0``.
        """
        if period == LimitPeriod.NONE:
            bucket = 0
        else:
            bucket = int(time.time() // period.bucket_divisor)
        return f"routeiq:usage:{policy_id}:{group_key}:{bucket}"


# =============================================================================
# Singleton Management
# =============================================================================

_engine: UsagePolicyEngine | None = None


def get_usage_policy_engine() -> UsagePolicyEngine:
    """Get or create the global UsagePolicyEngine singleton."""
    global _engine
    if _engine is None:
        _engine = UsagePolicyEngine()
    return _engine


def reset_usage_policy_engine() -> None:
    """Reset the global singleton (for testing)."""
    global _engine
    _engine = None


def is_usage_policies_enabled() -> bool:
    """Check whether usage policy enforcement is enabled."""
    return os.getenv("ROUTEIQ_USAGE_POLICIES_ENABLED", "false").lower() in (
        "true",
        "1",
        "yes",
        "on",
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "LimitType",
    "LimitPeriod",
    "PolicyAction",
    "UsagePolicy",
    "PolicyEvaluation",
    "UsagePolicyEngine",
    "get_usage_policy_engine",
    "reset_usage_policy_engine",
    "is_usage_policies_enabled",
]
