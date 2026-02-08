"""
Quota Enforcement Module for LLMRouter Gateway
==============================================

This module provides request/token/spend quota enforcement for the AI gateway.

Features:
- Multi-dimensional quotas: requests, input_tokens, output_tokens, total_tokens, spend_usd
- Time-based windows: minute, hour, day, month
- Redis-backed with atomic check+increment (Lua scripts)
- Pre-request enforcement via token reservation (no response buffering)
- Fail-open/fail-closed modes for Redis unavailability
- OpenTelemetry integration for observability

**CRITICAL: Streaming Safety**
Quota enforcement MUST NOT buffer response bodies. All enforcement happens BEFORE
the upstream call using:
- Request count: Always checked pre-request
- Token quotas: Reserved based on request fields (max_tokens, prompt estimation)
- Spend quotas: Reserved based on token reservation * cost multiplier

Usage:
    from litellm_llmrouter.quota import (
        QuotaEnforcer,
        get_quota_enforcer,
        quota_guard,
        derive_quota_subject,
    )

    # FastAPI dependency injection
    @router.post("/v1/chat/completions")
    async def chat_completions(
        request: Request,
        quota_check: QuotaGuardResult = Depends(quota_guard),
    ):
        # Request is allowed if we reach here
        ...

Configuration (environment variables):
    ROUTEIQ_QUOTA_ENABLED: Enable quota enforcement (default: false)
    ROUTEIQ_QUOTA_FAIL_MODE: "open" (default) or "closed" when Redis unavailable
    ROUTEIQ_QUOTA_LIMITS_JSON: JSON array of quota limit configurations
    REDIS_HOST: Redis host for quota storage
    REDIS_PORT: Redis port (default: 6379)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)


# =============================================================================
# Quota Enums and Data Classes
# =============================================================================


class QuotaMetric(str, Enum):
    """Metrics that can be quota-limited."""

    REQUESTS = "requests"
    INPUT_TOKENS = "input_tokens"
    OUTPUT_TOKENS = "output_tokens"
    TOTAL_TOKENS = "total_tokens"
    SPEND_USD = "spend_usd"


class QuotaWindow(str, Enum):
    """Time windows for quota enforcement."""

    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    MONTH = "month"

    @property
    def seconds(self) -> int:
        """Get window duration in seconds."""
        return {
            QuotaWindow.MINUTE: 60,
            QuotaWindow.HOUR: 3600,
            QuotaWindow.DAY: 86400,
            QuotaWindow.MONTH: 2592000,  # 30 days
        }[self]


class QuotaFailMode(str, Enum):
    """Behavior when Redis is unavailable."""

    OPEN = "open"  # Allow requests when Redis unavailable (default)
    CLOSED = "closed"  # Deny requests when Redis unavailable


@dataclass
class QuotaLimit:
    """A single quota limit configuration."""

    metric: QuotaMetric
    window: QuotaWindow
    limit: float
    # Optional: scope to specific models or routes (empty = all)
    models: list[str] = field(default_factory=list)
    routes: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "QuotaLimit":
        """Create QuotaLimit from dictionary."""
        return cls(
            metric=QuotaMetric(d["metric"]),
            window=QuotaWindow(d["window"]),
            limit=float(d["limit"]),
            models=d.get("models", []),
            routes=d.get("routes", []),
        )


@dataclass
class QuotaConfig:
    """Configuration for quota enforcement."""

    enabled: bool = False
    fail_mode: QuotaFailMode = QuotaFailMode.OPEN
    limits: list[QuotaLimit] = field(default_factory=list)
    # Default spend multiplier: USD per 1K tokens (when litellm cost not available)
    default_spend_per_1k_tokens: float = 0.002

    @classmethod
    def from_env(cls) -> "QuotaConfig":
        """Load configuration from environment variables."""
        enabled = os.getenv("ROUTEIQ_QUOTA_ENABLED", "false").lower() == "true"
        fail_mode_str = os.getenv("ROUTEIQ_QUOTA_FAIL_MODE", "open").lower()
        fail_mode = (
            QuotaFailMode.CLOSED if fail_mode_str == "closed" else QuotaFailMode.OPEN
        )

        limits: list[QuotaLimit] = []
        limits_json = os.getenv("ROUTEIQ_QUOTA_LIMITS_JSON", "")
        if limits_json:
            try:
                limits_data = json.loads(limits_json)
                if isinstance(limits_data, list):
                    for item in limits_data:
                        try:
                            limits.append(QuotaLimit.from_dict(item))
                        except (KeyError, ValueError) as e:
                            logger.warning(
                                f"Invalid quota limit config: {item}, error: {e}"
                            )
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid ROUTEIQ_QUOTA_LIMITS_JSON: {e}")

        spend_multiplier = float(
            os.getenv("ROUTEIQ_QUOTA_DEFAULT_SPEND_PER_1K_TOKENS", "0.002")
        )

        return cls(
            enabled=enabled,
            fail_mode=fail_mode,
            limits=limits,
            default_spend_per_1k_tokens=spend_multiplier,
        )


@dataclass
class QuotaSubject:
    """
    Identifies the entity being quota-limited.

    Subject precedence:
    1. team_id if present
    2. end_user_id if present
    3. API key hash (hash of the bearer token)
    4. client IP (last resort)
    """

    key: str
    type: str  # "team", "user", "api_key", "ip"

    @classmethod
    def derive(
        cls,
        request: Request,
        team_id: str | None = None,
        end_user_id: str | None = None,
        api_key: str | None = None,
    ) -> "QuotaSubject":
        """
        Derive quota subject from request context.

        Args:
            request: The FastAPI request
            team_id: Team ID from auth context
            end_user_id: End user ID from auth context
            api_key: API key from auth context

        Returns:
            QuotaSubject with stable key for quota tracking
        """
        if team_id:
            return cls(key=f"team:{team_id}", type="team")

        if end_user_id:
            return cls(key=f"user:{end_user_id}", type="user")

        if api_key:
            # Hash the API key for privacy/security
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
            return cls(key=f"apikey:{key_hash}", type="api_key")

        # Fall back to client IP
        client_ip = cls._get_client_ip(request)
        return cls(key=f"ip:{client_ip}", type="ip")

    @staticmethod
    def _get_client_ip(request: Request) -> str:
        """Extract client IP from request, respecting X-Forwarded-For."""
        # Check X-Forwarded-For header (for load balancer/proxy setups)
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            # Take the first IP (original client)
            return forwarded.split(",")[0].strip()

        # Check X-Real-IP header
        real_ip = request.headers.get("x-real-ip", "")
        if real_ip:
            return real_ip.strip()

        # Fall back to direct client
        if request.client:
            return request.client.host

        return "unknown"


@dataclass
class QuotaCheckResult:
    """Result of a quota check."""

    allowed: bool
    metric: QuotaMetric | None = None
    window: QuotaWindow | None = None
    current: float = 0.0
    limit: float = 0.0
    remaining: float = 0.0
    reset_at: float = 0.0  # Unix timestamp when quota resets
    error: str | None = None

    @property
    def retry_after(self) -> int:
        """Seconds until quota resets (for Retry-After header)."""
        if self.reset_at <= 0:
            return 60  # Default 1 minute
        remaining = int(self.reset_at - time.time())
        return max(1, remaining)


@dataclass
class QuotaGuardResult:
    """Result passed to route handlers after quota check."""

    subject: QuotaSubject
    allowed: bool = True
    checks: list[QuotaCheckResult] = field(default_factory=list)
    reserved_tokens: int = 0
    reserved_spend: float = 0.0


# =============================================================================
# Redis Lua Scripts for Atomic Quota Operations
# =============================================================================

# Lua script for atomic check and increment
# Returns: [current_value, limit, ttl_remaining, is_allowed (1/0)]
CHECK_AND_INCREMENT_LUA = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local increment = tonumber(ARGV[2])
local window_seconds = tonumber(ARGV[3])

local current = tonumber(redis.call('GET', key) or '0')

if current + increment > limit then
    local ttl = redis.call('TTL', key)
    if ttl < 0 then ttl = window_seconds end
    return {current, limit, ttl, 0}
end

local new_value = redis.call('INCRBY', key, increment)
if new_value == increment then
    redis.call('EXPIRE', key, window_seconds)
end

local ttl = redis.call('TTL', key)
if ttl < 0 then ttl = window_seconds end

return {new_value, limit, ttl, 1}
"""

# Lua script for atomic check and increment for float values (spend)
CHECK_AND_INCREMENT_FLOAT_LUA = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local increment = tonumber(ARGV[2])
local window_seconds = tonumber(ARGV[3])

local current = tonumber(redis.call('GET', key) or '0')

if current + increment > limit then
    local ttl = redis.call('TTL', key)
    if ttl < 0 then ttl = window_seconds end
    return {tostring(current), tostring(limit), ttl, 0}
end

local new_value = current + increment
redis.call('SET', key, tostring(new_value))
local existing_ttl = redis.call('TTL', key)
if existing_ttl < 0 then
    redis.call('EXPIRE', key, window_seconds)
end

local ttl = redis.call('TTL', key)
if ttl < 0 then ttl = window_seconds end

return {tostring(new_value), tostring(limit), ttl, 1}
"""

# Lua script for atomic float adjustment (credit or debit)
# Used by reconcile_spend to atomically adjust a quota counter.
# adjustment > 0 means credit back (subtract from usage)
# adjustment < 0 means debit more (add to usage)
# Returns: [new_value, adjustment_applied]
ADJUST_FLOAT_LUA = """
local key = KEYS[1]
local adjustment = tonumber(ARGV[1])

local current = tonumber(redis.call('GET', key) or '0')
local new_value = current - adjustment

if new_value < 0 then
    new_value = 0
end

redis.call('SET', key, tostring(new_value))

local existing_ttl = redis.call('TTL', key)
if existing_ttl > 0 then
    redis.call('EXPIRE', key, existing_ttl)
end

return {tostring(new_value), tostring(adjustment)}
"""


# =============================================================================
# Quota Repository (Redis Backend)
# =============================================================================


class QuotaRepository:
    """
    Redis-backed quota storage with atomic operations.

    Uses Lua scripts for atomic check-and-increment to prevent race conditions.
    Keys are structured as: quota:{subject}:{metric}:{window}:{bucket}
    where bucket is the current time window bucket (e.g., minute timestamp).
    """

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        key_prefix: str = "quota",
    ):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.key_prefix = key_prefix
        self._redis: Any = None
        self._check_incr_script: Any = None
        self._check_incr_float_script: Any = None
        self._adjust_float_script: Any = None
        self._lock = asyncio.Lock()

    async def _get_redis(self) -> Any:
        """Get or create Redis connection."""
        if self._redis is None:
            async with self._lock:
                if self._redis is None:
                    import redis.asyncio as aioredis

                    self._redis = aioredis.Redis(
                        host=self.redis_host,
                        port=self.redis_port,
                        decode_responses=True,
                        socket_connect_timeout=2.0,
                        socket_timeout=2.0,
                    )
                    # Register Lua scripts
                    self._check_incr_script = self._redis.register_script(
                        CHECK_AND_INCREMENT_LUA
                    )
                    self._check_incr_float_script = self._redis.register_script(
                        CHECK_AND_INCREMENT_FLOAT_LUA
                    )
                    self._adjust_float_script = self._redis.register_script(
                        ADJUST_FLOAT_LUA
                    )
        return self._redis

    def _bucket_key(
        self,
        subject: QuotaSubject,
        metric: QuotaMetric,
        window: QuotaWindow,
    ) -> str:
        """
        Generate Redis key for a quota bucket.

        Uses fixed time windows based on window type:
        - minute: floor to minute
        - hour: floor to hour
        - day: floor to day (UTC)
        - month: floor to month (UTC)
        """
        now = time.time()

        if window == QuotaWindow.MINUTE:
            bucket = int(now // 60)
        elif window == QuotaWindow.HOUR:
            bucket = int(now // 3600)
        elif window == QuotaWindow.DAY:
            bucket = int(now // 86400)
        else:  # MONTH
            bucket = int(now // 2592000)

        return f"{self.key_prefix}:{subject.key}:{metric.value}:{window.value}:{bucket}"

    async def check_and_increment(
        self,
        subject: QuotaSubject,
        metric: QuotaMetric,
        window: QuotaWindow,
        limit: float,
        increment: float = 1.0,
    ) -> QuotaCheckResult:
        """
        Atomically check quota and increment if allowed.

        Args:
            subject: The quota subject
            metric: The metric being checked
            window: The time window
            limit: The quota limit
            increment: Amount to increment (default: 1 for requests)

        Returns:
            QuotaCheckResult with current usage and remaining quota
        """
        key = self._bucket_key(subject, metric, window)

        try:
            await self._get_redis()

            # Use float script for spend, int script for everything else
            if metric == QuotaMetric.SPEND_USD:
                result = await self._check_incr_float_script(
                    keys=[key],
                    args=[limit, increment, window.seconds],
                )
                current = float(result[0])
                limit_v = float(result[1])
            else:
                result = await self._check_incr_script(
                    keys=[key],
                    args=[int(limit), int(increment), window.seconds],
                )
                current = float(result[0])
                limit_v = float(result[1])

            ttl = int(result[2])
            allowed = result[3] == 1

            reset_at = time.time() + ttl

            return QuotaCheckResult(
                allowed=allowed,
                metric=metric,
                window=window,
                current=current,
                limit=limit_v,
                remaining=max(0, limit_v - current),
                reset_at=reset_at,
            )

        except Exception as e:
            logger.error(f"Quota check failed for {subject.key}: {e}")
            return QuotaCheckResult(
                allowed=True,  # Fail open by default (caller handles fail mode)
                metric=metric,
                window=window,
                error=str(e),
            )

    async def adjust_float(
        self,
        subject: QuotaSubject,
        metric: QuotaMetric,
        window: QuotaWindow,
        adjustment: float,
    ) -> tuple[float, float]:
        """
        Atomically adjust a float quota counter.

        A positive adjustment credits back (reduces usage).
        A negative adjustment debits more (increases usage).

        Args:
            subject: The quota subject
            metric: The metric being adjusted
            window: The time window
            adjustment: Amount to adjust (positive = credit, negative = debit)

        Returns:
            Tuple of (new_value, adjustment_applied)
        """
        key = self._bucket_key(subject, metric, window)

        try:
            await self._get_redis()

            result = await self._adjust_float_script(
                keys=[key],
                args=[adjustment],
            )
            new_value = float(result[0])
            applied = float(result[1])
            return new_value, applied

        except Exception as e:
            logger.error(f"Quota adjustment failed for {subject.key}: {e}")
            return 0.0, 0.0

    async def get_current(
        self,
        subject: QuotaSubject,
        metric: QuotaMetric,
        window: QuotaWindow,
    ) -> float:
        """Get current usage without incrementing."""
        key = self._bucket_key(subject, metric, window)

        try:
            redis = await self._get_redis()
            value = await redis.get(key)
            return float(value) if value else 0.0
        except Exception as e:
            logger.error(f"Failed to get quota for {subject.key}: {e}")
            return 0.0

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.aclose()
            self._redis = None


# =============================================================================
# Quota Enforcer
# =============================================================================


class QuotaEnforcer:
    """
    Main quota enforcement logic.

    Provides:
    - Pre-request quota checks (request count)
    - Token/spend reservation based on request fields
    - Configurable fail-open/fail-closed behavior
    """

    def __init__(
        self,
        config: QuotaConfig | None = None,
        repository: QuotaRepository | None = None,
    ):
        self._config = config or QuotaConfig.from_env()
        self._repository = repository
        self._cumulative_savings: float = 0.0

    @property
    def config(self) -> QuotaConfig:
        """Get current quota configuration."""
        return self._config

    @property
    def is_enabled(self) -> bool:
        """Check if quota enforcement is enabled."""
        return self._config.enabled

    async def _get_repository(self) -> QuotaRepository:
        """Get or create the repository."""
        if self._repository is None:
            redis_host = os.getenv("REDIS_HOST", "localhost")
            redis_port = int(os.getenv("REDIS_PORT", "6379"))
            self._repository = QuotaRepository(
                redis_host=redis_host,
                redis_port=redis_port,
            )
        return self._repository

    def _estimate_tokens_from_request(self, body: dict[str, Any]) -> tuple[int, int]:
        """
        Estimate input and output tokens from request body.

        Returns:
            Tuple of (estimated_input_tokens, max_output_tokens)
        """
        # Estimate input tokens from messages
        input_tokens = 0
        messages = body.get("messages", [])
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                # Rough estimate: 1 token ≈ 4 characters
                input_tokens += len(content) // 4
            elif isinstance(content, list):
                # Multi-modal content
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        input_tokens += len(item.get("text", "")) // 4

        # Use max_tokens from request or default
        max_tokens = body.get("max_tokens", 4096)
        if max_tokens is None:
            max_tokens = 4096

        return max(input_tokens, 10), max_tokens

    def _calculate_spend_reservation(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str | None = None,
    ) -> float:
        """
        Calculate spend reservation based on tokens.

        Attempts to use litellm cost info if available, otherwise uses
        configurable default multiplier.
        """
        try:
            import litellm

            if model and hasattr(litellm, "model_cost"):
                cost_info = litellm.model_cost.get(model)
                if cost_info:
                    input_cost = (input_tokens / 1000) * cost_info.get(
                        "input_cost_per_token", 0
                    )
                    output_cost = (output_tokens / 1000) * cost_info.get(
                        "output_cost_per_token", 0
                    )
                    return input_cost + output_cost
        except (ImportError, Exception):
            pass

        # Fall back to default multiplier
        total_tokens = input_tokens + output_tokens
        return (total_tokens / 1000) * self._config.default_spend_per_1k_tokens

    async def check_and_increment_requests(
        self,
        subject: QuotaSubject,
        route: str | None = None,
        model: str | None = None,
    ) -> list[QuotaCheckResult]:
        """
        Check and increment request quota.

        Args:
            subject: The quota subject
            route: The route being accessed (for route-specific limits)
            model: The model being used (for model-specific limits)

        Returns:
            List of QuotaCheckResult for all applicable limits
        """
        if not self.is_enabled:
            return []

        results: list[QuotaCheckResult] = []
        repository = await self._get_repository()

        for limit in self._config.limits:
            if limit.metric != QuotaMetric.REQUESTS:
                continue

            # Check if limit applies to this route/model
            if limit.routes and route not in limit.routes:
                continue
            if limit.models and model not in limit.models:
                continue

            result = await repository.check_and_increment(
                subject=subject,
                metric=limit.metric,
                window=limit.window,
                limit=limit.limit,
                increment=1,
            )

            results.append(result)

            # If any check fails, handle based on fail mode
            if not result.allowed:
                break
            if result.error and self._config.fail_mode == QuotaFailMode.CLOSED:
                result.allowed = False
                break

        return results

    async def reserve_tokens_or_spend(
        self,
        subject: QuotaSubject,
        body: dict[str, Any],
        route: str | None = None,
        model: str | None = None,
    ) -> tuple[list[QuotaCheckResult], int, float]:
        """
        Reserve tokens and/or spend quota based on request.

        This is called PRE-REQUEST to enforce quotas without buffering responses.
        Reservation is based on:
        - Estimated input tokens from messages
        - max_tokens from request (for output)
        - Spend calculated from token estimates

        Args:
            subject: The quota subject
            body: The request body (parsed JSON)
            route: The route being accessed
            model: The model being used

        Returns:
            Tuple of (results, reserved_tokens, reserved_spend)
        """
        if not self.is_enabled:
            return [], 0, 0.0

        results: list[QuotaCheckResult] = []
        repository = await self._get_repository()

        # Extract model from body if not provided
        if not model:
            model = body.get("model")

        # Estimate tokens
        input_tokens, output_tokens = self._estimate_tokens_from_request(body)
        total_tokens = input_tokens + output_tokens

        # Calculate spend
        spend = self._calculate_spend_reservation(input_tokens, output_tokens, model)

        for limit in self._config.limits:
            # Check if limit applies to this route/model
            if limit.routes and route not in limit.routes:
                continue
            if limit.models and model not in limit.models:
                continue

            # Skip request limits (handled separately)
            if limit.metric == QuotaMetric.REQUESTS:
                continue

            # Determine increment based on metric
            if limit.metric == QuotaMetric.INPUT_TOKENS:
                increment = float(input_tokens)
            elif limit.metric == QuotaMetric.OUTPUT_TOKENS:
                increment = float(output_tokens)
            elif limit.metric == QuotaMetric.TOTAL_TOKENS:
                increment = float(total_tokens)
            elif limit.metric == QuotaMetric.SPEND_USD:
                increment = spend
            else:
                continue

            result = await repository.check_and_increment(
                subject=subject,
                metric=limit.metric,
                window=limit.window,
                limit=limit.limit,
                increment=increment,
            )

            results.append(result)

            # If any check fails, handle based on fail mode
            if not result.allowed:
                break
            if result.error and self._config.fail_mode == QuotaFailMode.CLOSED:
                result.allowed = False
                break

        return results, total_tokens, spend

    async def reconcile_spend(
        self,
        subject: str,
        actual_cost: float,
        reserved_cost: float,
    ) -> dict[str, Any]:
        """Reconcile actual spend against reserved amount.

        After an LLM call completes, the actual cost is known. This method
        adjusts the quota counter to reflect the actual spend instead of the
        reserved (estimated) amount.

        If actual < reserved, the difference is credited back.
        If actual > reserved, the difference is debited.

        Args:
            subject: The quota subject key (team/key identifier)
            actual_cost: Actual cost from the LLM response
            reserved_cost: Amount originally reserved before the call

        Returns:
            Dict with reconciliation details
        """
        # Check feature flag
        if not _is_cost_reconciliation_enabled():
            return {
                "reconciled": False,
                "reason": "cost_reconciliation_disabled",
            }

        if not self.is_enabled:
            return {
                "reconciled": False,
                "reason": "quota_disabled",
            }

        adjustment = reserved_cost - actual_cost

        # No adjustment needed if costs match
        if abs(adjustment) < 1e-10:
            return {
                "reconciled": True,
                "subject": subject,
                "actual_cost": actual_cost,
                "reserved_cost": reserved_cost,
                "adjustment": 0.0,
                "direction": "none",
                "savings": 0.0,
            }

        direction = "credit" if adjustment > 0 else "debit"
        savings = max(adjustment, 0.0)

        # Track cumulative savings
        self._cumulative_savings += savings

        # Apply adjustment to all spend quota limits
        repository = await self._get_repository()
        quota_subject = QuotaSubject(key=subject, type="reconciliation")

        adjusted_any = False
        for limit in self._config.limits:
            if limit.metric != QuotaMetric.SPEND_USD:
                continue

            new_value, applied = await repository.adjust_float(
                subject=quota_subject,
                metric=limit.metric,
                window=limit.window,
                adjustment=adjustment,
            )
            if abs(applied) > 1e-10:
                adjusted_any = True

        # Emit OTel metrics
        _emit_reconciliation_metrics(
            subject=subject,
            adjustment=adjustment,
            direction=direction,
            savings=savings,
        )

        logger.debug(
            "Quota reconciliation: subject=%s reserved=$%.6f actual=$%.6f "
            "adjustment=$%.6f direction=%s",
            subject,
            reserved_cost,
            actual_cost,
            adjustment,
            direction,
        )

        return {
            "reconciled": True,
            "subject": subject,
            "actual_cost": actual_cost,
            "reserved_cost": reserved_cost,
            "adjustment": adjustment,
            "direction": direction,
            "savings": savings,
            "cumulative_savings": self._cumulative_savings,
            "adjusted_quotas": adjusted_any,
        }

    async def close(self) -> None:
        """Close the enforcer and release resources."""
        if self._repository:
            await self._repository.close()


# =============================================================================
# Cost Reconciliation Feature Flag and OTel Helpers
# =============================================================================


def _is_cost_reconciliation_enabled() -> bool:
    """Check if cost reconciliation is enabled via environment variable."""
    return os.getenv("ROUTEIQ_COST_RECONCILIATION_ENABLED", "true").lower() != "false"


def _emit_reconciliation_metrics(
    subject: str,
    adjustment: float,
    direction: str,
    savings: float,
) -> None:
    """Emit OTel metrics for a reconciliation event."""
    try:
        from litellm_llmrouter.metrics import get_gateway_metrics

        metrics = get_gateway_metrics()
        if metrics is None:
            return

        # Use the reconciliation-specific instruments if available
        if hasattr(metrics, "reconciliation_savings"):
            metrics.reconciliation_savings.record(
                savings, {"subject": subject, "direction": direction}
            )
        if hasattr(metrics, "reconciliation_count"):
            metrics.reconciliation_count.add(
                1, {"subject": subject, "direction": direction}
            )
    except (ImportError, Exception):
        pass


# =============================================================================
# Global Enforcer Singleton and FastAPI Dependency
# =============================================================================

_quota_enforcer: QuotaEnforcer | None = None


def get_quota_enforcer() -> QuotaEnforcer:
    """Get or create the global quota enforcer singleton."""
    global _quota_enforcer
    if _quota_enforcer is None:
        _quota_enforcer = QuotaEnforcer()
    return _quota_enforcer


def reset_quota_enforcer() -> None:
    """Reset the global quota enforcer (for testing)."""
    global _quota_enforcer
    _quota_enforcer = None


def derive_quota_subject(request: Request) -> QuotaSubject:
    """
    Derive quota subject from FastAPI request.

    Extracts auth context from request state (set by LiteLLM's user_api_key_auth)
    and derives subject with precedence:
    1. team_id
    2. user_id / end_user_id
    3. API key hash
    4. Client IP
    """
    # Try to get auth context from request state (set by user_api_key_auth)
    team_id: str | None = None
    end_user_id: str | None = None
    api_key: str | None = None

    # LiteLLM sets user_api_key_dict in request state after auth
    user_info = getattr(request.state, "user_api_key_dict", None)
    if user_info:
        team_id = user_info.get("team_id")
        end_user_id = user_info.get("user_id") or user_info.get("end_user_id")
        api_key = user_info.get("api_key") or user_info.get("token")

    # Fall back to extracting API key from header
    if not api_key:
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            api_key = auth_header[7:].strip()

    return QuotaSubject.derive(
        request=request,
        team_id=team_id,
        end_user_id=end_user_id,
        api_key=api_key,
    )


# Paths to exclude from quota enforcement (health, docs, etc.)
QUOTA_EXCLUDED_PATHS = frozenset(
    {
        "/_health/live",
        "/_health/ready",
        "/health",
        "/health/liveliness",
        "/health/readiness",
        "/docs",
        "/openapi.json",
        "/redoc",
    }
)


async def quota_guard(request: Request) -> QuotaGuardResult:
    """
    FastAPI dependency for quota enforcement.

    This is designed to be used with `Depends(quota_guard)` on routes that
    should be quota-limited. It performs ALL quota checks PRE-REQUEST,
    ensuring streaming responses are not buffered.

    For token/spend quotas, it uses request body fields (max_tokens, messages)
    to estimate and reserve quota.

    Raises:
        HTTPException 429 if quota exceeded
        HTTPException 503 if quota storage unavailable (fail-closed mode)

    Returns:
        QuotaGuardResult with subject and reservation info
    """
    enforcer = get_quota_enforcer()

    # Skip if quota not enabled
    if not enforcer.is_enabled:
        # Return a minimal result
        return QuotaGuardResult(
            subject=QuotaSubject(key="disabled", type="disabled"),
            allowed=True,
        )

    # Skip excluded paths
    if request.url.path in QUOTA_EXCLUDED_PATHS:
        return QuotaGuardResult(
            subject=QuotaSubject(key="excluded", type="excluded"),
            allowed=True,
        )

    # Derive quota subject
    subject = derive_quota_subject(request)

    # Get route and model info
    route = request.url.path
    model: str | None = None

    # Try to parse request body for token estimation
    body: dict[str, Any] = {}
    try:
        # Check if body was already parsed (cached by FastAPI)
        cached_body = getattr(request.state, "_parsed_body", None)
        if cached_body is not None:
            body = cached_body
        else:
            # Read and cache body
            body_bytes = await request.body()
            if body_bytes:
                body = json.loads(body_bytes)
                request.state._parsed_body = body
                model = body.get("model")
    except (json.JSONDecodeError, Exception):
        pass

    all_results: list[QuotaCheckResult] = []

    # Check request quota
    request_results = await enforcer.check_and_increment_requests(
        subject=subject,
        route=route,
        model=model,
    )
    all_results.extend(request_results)

    # Check token/spend quotas (via reservation)
    reserved_tokens = 0
    reserved_spend = 0.0
    if body:
        (
            token_results,
            reserved_tokens,
            reserved_spend,
        ) = await enforcer.reserve_tokens_or_spend(
            subject=subject,
            body=body,
            route=route,
            model=model,
        )
        all_results.extend(token_results)

    # Check for any denials
    for result in all_results:
        if not result.allowed:
            # Create 429 response with quota info
            logger.warning(
                f"Quota exceeded for {subject.key}: "
                f"{result.metric.value if result.metric else 'unknown'} "
                f"{result.current}/{result.limit} in {result.window.value if result.window else 'unknown'}"
            )

            raise HTTPException(
                status_code=429,
                detail={
                    "error": "quota_exceeded",
                    "message": f"Quota exceeded for {result.metric.value if result.metric else 'metric'}",
                    "metric": result.metric.value if result.metric else None,
                    "window": result.window.value if result.window else None,
                    "current": result.current,
                    "limit": result.limit,
                    "retry_after": result.retry_after,
                },
                headers={"Retry-After": str(result.retry_after)},
            )

        if result.error and enforcer.config.fail_mode == QuotaFailMode.CLOSED:
            logger.error(f"Quota storage unavailable (fail-closed): {result.error}")
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "quota_storage_unavailable",
                    "message": "Quota enforcement unavailable",
                },
            )

    # Log successful quota check
    if all_results:
        logger.debug(
            f"Quota allowed for {subject.key}: "
            f"reserved_tokens={reserved_tokens}, reserved_spend=${reserved_spend:.4f}"
        )

    return QuotaGuardResult(
        subject=subject,
        allowed=True,
        checks=all_results,
        reserved_tokens=reserved_tokens,
        reserved_spend=reserved_spend,
    )


# =============================================================================
# Observability Integration
# =============================================================================


def add_quota_span_attributes(
    result: QuotaGuardResult,
    span: Any = None,
) -> None:
    """
    Add quota-related attributes to an OpenTelemetry span.

    Args:
        result: The quota guard result
        span: Optional span to add attributes to (uses current span if None)
    """
    try:
        from opentelemetry import trace

        if span is None:
            span = trace.get_current_span()

        if span and span.is_recording():
            span.set_attribute("quota.subject.key", result.subject.key)
            span.set_attribute("quota.subject.type", result.subject.type)
            span.set_attribute("quota.allowed", result.allowed)
            span.set_attribute("quota.reserved_tokens", result.reserved_tokens)
            span.set_attribute("quota.reserved_spend", result.reserved_spend)

            for i, check in enumerate(result.checks):
                prefix = f"quota.check.{i}"
                if check.metric:
                    span.set_attribute(f"{prefix}.metric", check.metric.value)
                if check.window:
                    span.set_attribute(f"{prefix}.window", check.window.value)
                span.set_attribute(f"{prefix}.current", check.current)
                span.set_attribute(f"{prefix}.limit", check.limit)
                span.set_attribute(f"{prefix}.allowed", check.allowed)
    except (ImportError, Exception):
        # OpenTelemetry not available or span error - fail silently
        pass
