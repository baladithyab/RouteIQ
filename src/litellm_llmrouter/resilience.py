"""
Resilience Primitives: Backpressure, Drain Mode, and Circuit Breakers
======================================================================

This module provides gateway-level resilience primitives for the LLMRouter gateway:

1. **BackpressureMiddleware**: ASGI middleware for concurrency/load shedding
   - Enforces bounded concurrency (max concurrent requests)
   - Returns 503 with JSON body when capacity is exhausted
   - Excludes health endpoints from limiting
   - Holds concurrency slot until response is fully sent (including streaming)

2. **DrainManager**: Manages graceful shutdown for long-lived/streaming requests
   - Tracks active requests via `app.state`
   - Exposes draining flag for readiness checks
   - Waits for in-flight requests during shutdown

3. **CircuitBreaker**: Implements the circuit breaker pattern for external dependencies
   - States: CLOSED (normal), OPEN (failing fast), HALF_OPEN (testing recovery)
   - Tracks failure rate and opens circuit when threshold exceeded
   - Automatic recovery testing after timeout

4. **CircuitBreakerManager**: Manages circuit breakers for DB, Redis, etc.
   - Provides named breakers for different services
   - Exposes degraded mode status for health checks
   - Supports graceful degradation (cached reads when writes fail)

5. **Per-Provider Circuit Breakers**: Protect outbound LLM HTTP calls per provider
   - Feature-flagged via ROUTEIQ_PROVIDER_CB_ENABLED (default: false)
   - Per-provider env overrides: ROUTEIQ_CB_{PROVIDER}_FAILURE_THRESHOLD, etc.
   - OTel event emission on state changes

Configuration via environment variables (disabled by default for non-breaking behavior):
- ROUTEIQ_MAX_CONCURRENT_REQUESTS: Max concurrent requests (0 = disabled)
- ROUTEIQ_DRAIN_TIMEOUT_SECONDS: Max time to wait for drain (default: 30)
- ROUTEIQ_CB_FAILURE_THRESHOLD: Failures before circuit opens (default: 5)
- ROUTEIQ_CB_SUCCESS_THRESHOLD: Successes before circuit closes (default: 2)
- ROUTEIQ_CB_TIMEOUT_SECONDS: Time before half-open test (default: 30)
- ROUTEIQ_CB_WINDOW_SECONDS: Failure tracking window (default: 60)
- ROUTEIQ_PROVIDER_CB_ENABLED: Enable per-provider LLM circuit breakers (default: false)

Usage:
    from litellm_llmrouter.resilience import (
        BackpressureMiddleware,
        DrainManager,
        get_drain_manager,
        CircuitBreaker,
        CircuitBreakerManager,
        get_circuit_breaker_manager,
        CircuitBreakerOpenError,
    )

    # In app factory:
    drain_manager = get_drain_manager()
    drain_manager.attach(app)

    # Circuit breaker for database calls:
    cb_manager = get_circuit_breaker_manager()
    try:
        async with cb_manager.get_breaker("database").execute():
            await db.execute("SELECT 1")
    except CircuitBreakerOpenError:
        # Fail fast - use cached data or return error
        pass

    # Per-provider circuit breaker for LLM calls:
    from litellm_llmrouter.resilience import execute_with_provider_circuit_breaker
    result = await execute_with_provider_circuit_breaker("openai", llm_call, prompt)

    # Middleware is added by calling add_backpressure_middleware(app)
    # It wraps the ASGI app to track concurrency
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable

from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_MAX_CONCURRENT_REQUESTS = 0  # 0 = disabled
DEFAULT_DRAIN_TIMEOUT_SECONDS = 30
DEFAULT_EXCLUDED_PATHS = frozenset(
    {
        "/_health/live",
        "/_health/ready",
        "/health/liveliness",
        "/health/readiness",
        "/health",
    }
)

# Circuit Breaker defaults
DEFAULT_CB_FAILURE_THRESHOLD = 5  # Number of failures before circuit opens
DEFAULT_CB_SUCCESS_THRESHOLD = 2  # Number of successes in half-open before closing
DEFAULT_CB_TIMEOUT_SECONDS = 30.0  # Time before attempting recovery (half-open)
DEFAULT_CB_WINDOW_SECONDS = 60.0  # Sliding window for failure tracking


@dataclass
class ResilienceConfig:
    """Configuration for resilience primitives."""

    max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_REQUESTS
    drain_timeout_seconds: float = DEFAULT_DRAIN_TIMEOUT_SECONDS
    excluded_paths: frozenset[str] = field(
        default_factory=lambda: DEFAULT_EXCLUDED_PATHS
    )

    @classmethod
    def from_env(cls) -> "ResilienceConfig":
        """Load configuration from environment variables."""
        max_concurrent_str = os.getenv("ROUTEIQ_MAX_CONCURRENT_REQUESTS", "0")
        drain_timeout_str = os.getenv(
            "ROUTEIQ_DRAIN_TIMEOUT_SECONDS", str(DEFAULT_DRAIN_TIMEOUT_SECONDS)
        )

        try:
            max_concurrent = int(max_concurrent_str)
        except ValueError:
            logger.warning(
                f"Invalid ROUTEIQ_MAX_CONCURRENT_REQUESTS value '{max_concurrent_str}', using default 0"
            )
            max_concurrent = 0

        try:
            drain_timeout = float(drain_timeout_str)
        except ValueError:
            logger.warning(
                f"Invalid ROUTEIQ_DRAIN_TIMEOUT_SECONDS value '{drain_timeout_str}', using default {DEFAULT_DRAIN_TIMEOUT_SECONDS}"
            )
            drain_timeout = DEFAULT_DRAIN_TIMEOUT_SECONDS

        # Additional excluded paths from env
        extra_excluded_str = os.getenv("ROUTEIQ_BACKPRESSURE_EXCLUDED_PATHS", "")
        extra_excluded = frozenset(
            p.strip() for p in extra_excluded_str.split(",") if p.strip()
        )
        excluded_paths = DEFAULT_EXCLUDED_PATHS | extra_excluded

        return cls(
            max_concurrent_requests=max_concurrent,
            drain_timeout_seconds=drain_timeout,
            excluded_paths=excluded_paths,
        )

    def is_enabled(self) -> bool:
        """Check if backpressure limiting is enabled."""
        return self.max_concurrent_requests > 0


class DrainManager:
    """
    Manages graceful shutdown drain mode for streaming/long-lived requests.

    This manager:
    - Tracks active requests count
    - Manages draining flag for readiness checks
    - Waits for in-flight requests during shutdown

    Usage:
        drain_manager = DrainManager()
        drain_manager.attach(app)  # Sets up app.state

        # During shutdown:
        await drain_manager.start_drain()
        await drain_manager.wait_for_drain()
    """

    def __init__(self, config: ResilienceConfig | None = None):
        self._config = config or ResilienceConfig.from_env()
        self._active_requests: int = 0
        self._draining: bool = False
        self._lock = asyncio.Lock()
        self._drain_event = asyncio.Event()
        self._app: Any = None

    @property
    def active_requests(self) -> int:
        """Current number of active requests."""
        return self._active_requests

    @property
    def is_draining(self) -> bool:
        """Whether the server is in drain mode."""
        return self._draining

    @property
    def drain_timeout_seconds(self) -> float:
        """Configured drain timeout."""
        return self._config.drain_timeout_seconds

    async def acquire(self) -> bool:
        """
        Try to acquire a request slot.

        Returns:
            True if acquired, False if draining or at capacity
        """
        async with self._lock:
            if self._draining:
                return False
            self._active_requests += 1
            return True

    async def release(self) -> None:
        """Release a request slot."""
        async with self._lock:
            self._active_requests -= 1
            if self._active_requests <= 0:
                self._active_requests = 0
                if self._draining:
                    self._drain_event.set()

    def attach(self, app: Any) -> None:
        """
        Attach the drain manager to a FastAPI/Starlette app.

        Sets up app.state with:
        - resilience_drain_manager: This DrainManager instance
        - active_requests: Property reference for backwards compat
        - is_draining: Property reference for backwards compat
        """
        self._app = app
        app.state.resilience_drain_manager = self
        # Expose as properties on app.state for convenience
        app.state.active_requests = property(lambda _: self._active_requests)
        app.state.is_draining = property(lambda _: self._draining)
        logger.info(
            f"DrainManager attached (timeout={self._config.drain_timeout_seconds}s)"
        )

    async def start_drain(self) -> None:
        """
        Start drain mode.

        After calling this, readiness checks should return non-200.
        """
        async with self._lock:
            if self._draining:
                logger.debug("Already in drain mode")
                return
            self._draining = True
            if self._active_requests == 0:
                self._drain_event.set()
        logger.info(
            f"Drain mode started, waiting for {self._active_requests} active requests"
        )

    async def wait_for_drain(self) -> bool:
        """
        Wait for all active requests to complete.

        Returns:
            True if all requests completed, False if timeout occurred
        """
        if self._active_requests == 0:
            logger.info("No active requests, drain complete")
            return True

        try:
            await asyncio.wait_for(
                self._drain_event.wait(),
                timeout=self._config.drain_timeout_seconds,
            )
            logger.info("Drain complete, all requests finished")
            return True
        except asyncio.TimeoutError:
            logger.warning(
                f"Drain timeout after {self._config.drain_timeout_seconds}s, "
                f"{self._active_requests} requests still active"
            )
            return False

    def get_status(self) -> dict[str, Any]:
        """Get current drain status for health checks."""
        return {
            "active_requests": self._active_requests,
            "is_draining": self._draining,
            "drain_timeout_seconds": self._config.drain_timeout_seconds,
        }


# Global singleton for the drain manager
_drain_manager: DrainManager | None = None


def get_drain_manager() -> DrainManager:
    """Get or create the global drain manager singleton."""
    global _drain_manager
    if _drain_manager is None:
        _drain_manager = DrainManager()
    return _drain_manager


def reset_drain_manager() -> None:
    """Reset the global drain manager (for testing)."""
    global _drain_manager
    _drain_manager = None


class BackpressureMiddleware:
    """
    ASGI middleware for concurrency limiting and load shedding.

    This middleware:
    - Tracks concurrent requests using DrainManager
    - Rejects requests with 503 when capacity is exhausted
    - Excludes health endpoints from limiting
    - Holds concurrency slot until response body is fully sent (streaming-safe)

    Unlike BaseHTTPMiddleware which doesn't properly handle streaming,
    this is a pure ASGI middleware that wraps the send callable to ensure
    the concurrency slot is released only after the response is complete.
    """

    def __init__(
        self,
        app: ASGIApp,
        config: ResilienceConfig | None = None,
        drain_manager: DrainManager | None = None,
    ):
        self.app = app
        self._config = config or ResilienceConfig.from_env()
        self._drain_manager = drain_manager or get_drain_manager()
        self._semaphore: asyncio.Semaphore | None = None
        if self._config.is_enabled():
            self._semaphore = asyncio.Semaphore(self._config.max_concurrent_requests)
        logger.info(
            f"BackpressureMiddleware initialized "
            f"(max_concurrent={self._config.max_concurrent_requests}, "
            f"enabled={self._config.is_enabled()})"
        )

    def _is_excluded(self, path: str) -> bool:
        """Check if a path should be excluded from rate limiting."""
        return path in self._config.excluded_paths

    async def _send_503_response(
        self, send: Send, request_id: str | None = None
    ) -> None:
        """Send a 503 over-capacity JSON response."""
        body = {
            "error": "over_capacity",
            "message": "Server is at capacity, please retry later",
        }
        if request_id:
            body["request_id"] = request_id

        import json

        body_bytes = json.dumps(body).encode("utf-8")

        await send(
            {
                "type": "http.response.start",
                "status": 503,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body_bytes)).encode()),
                    (b"retry-after", b"1"),
                ],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": body_bytes,
                "more_body": False,
            }
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """ASGI entry point."""
        # Only apply to HTTP requests
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")

        # Skip excluded paths (health checks)
        if self._is_excluded(path):
            await self.app(scope, receive, send)
            return

        # If backpressure is disabled, just track for drain purposes
        if not self._config.is_enabled():
            await self._handle_request_with_tracking(scope, receive, send)
            return

        # Check if draining
        if self._drain_manager.is_draining:
            # During drain, reject new requests
            request_id = self._extract_request_id(scope)
            await self._send_503_response(send, request_id)
            return

        # Try to acquire semaphore (non-blocking check first)
        assert self._semaphore is not None  # Guaranteed by is_enabled() check

        if self._semaphore.locked() and self._semaphore._value == 0:
            # At capacity - immediate 503
            request_id = self._extract_request_id(scope)
            logger.debug(f"Rejecting request {request_id or 'unknown'}: at capacity")
            await self._send_503_response(send, request_id)
            return

        # Acquire semaphore and process request
        try:
            # Use acquire with timeout of 0 to get immediate failure
            acquired = self._semaphore.locked()
            if not acquired:
                await self._semaphore.acquire()

                # Track in drain manager
                await self._drain_manager.acquire()

                try:
                    await self._handle_request_tracked(scope, receive, send)
                finally:
                    await self._drain_manager.release()
                    self._semaphore.release()
            else:
                # At capacity
                request_id = self._extract_request_id(scope)
                await self._send_503_response(send, request_id)
        except Exception:
            # Release on error
            raise

    async def _handle_request_with_tracking(
        self, scope: Scope, receive: Receive, send: Send
    ) -> None:
        """Handle request with drain tracking only (no concurrency limit)."""
        await self._drain_manager.acquire()
        try:
            await self.app(scope, receive, send)
        finally:
            await self._drain_manager.release()

    async def _handle_request_tracked(
        self, scope: Scope, receive: Receive, send: Send
    ) -> None:
        """
        Handle request with full tracking.

        Wraps the send callable to ensure we detect when response is complete,
        which is critical for streaming responses.
        """
        response_started = False
        response_complete = False

        async def send_wrapper(message: Message) -> None:
            nonlocal response_started, response_complete

            if message["type"] == "http.response.start":
                response_started = True
            elif message["type"] == "http.response.body":
                if not message.get("more_body", False):
                    response_complete = True

            await send(message)

        await self.app(scope, receive, send_wrapper)

    def _extract_request_id(self, scope: Scope) -> str | None:
        """Extract request ID from scope headers if present."""
        headers = scope.get("headers", [])
        for name, value in headers:
            if name.lower() == b"x-request-id":
                result: str = value.decode("utf-8", errors="replace")
                return result
        return None


def add_backpressure_middleware(app: Any) -> bool:
    """
    Add backpressure middleware to a FastAPI/Starlette app.

    This should be called early in app setup, before other middleware.

    Returns:
        True if middleware was added, False if disabled by config
    """
    config = ResilienceConfig.from_env()
    drain_manager = get_drain_manager()

    # Always attach drain manager for tracking
    drain_manager.attach(app)

    if not config.is_enabled():
        logger.info(
            "Backpressure middleware disabled (ROUTEIQ_MAX_CONCURRENT_REQUESTS not set or 0)"
        )
        # Still wrap for drain tracking even when limiting is disabled
        original_app = app.app if hasattr(app, "app") else None
        if original_app:
            app.app = BackpressureMiddleware(original_app, config, drain_manager)
        return False

    # Add as ASGI middleware by wrapping the app
    original_app = app.app if hasattr(app, "app") else None
    if original_app:
        app.app = BackpressureMiddleware(original_app, config, drain_manager)
    else:
        # For testing with raw ASGI apps
        logger.warning("Could not wrap app for backpressure middleware")
        return False

    logger.info(
        f"Backpressure middleware enabled (max_concurrent={config.max_concurrent_requests})"
    )
    return True


async def graceful_shutdown(app: Any, timeout: float | None = None) -> bool:
    """
    Perform graceful shutdown with drain.

    This should be called from the app's shutdown lifecycle.

    Args:
        app: The FastAPI/Starlette app
        timeout: Optional override for drain timeout

    Returns:
        True if drain completed, False if timeout occurred
    """
    drain_manager = get_drain_manager()

    if timeout is not None:
        drain_manager._config = ResilienceConfig(
            max_concurrent_requests=drain_manager._config.max_concurrent_requests,
            drain_timeout_seconds=timeout,
            excluded_paths=drain_manager._config.excluded_paths,
        )

    await drain_manager.start_drain()
    return await drain_manager.wait_for_drain()


# =============================================================================
# Circuit Breaker Implementation
# =============================================================================


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation - requests pass through
    OPEN = "open"  # Failing fast - all requests immediately fail
    HALF_OPEN = "half_open"  # Testing recovery - limited requests pass through


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open and request is rejected."""

    def __init__(self, breaker_name: str, time_until_retry: float):
        self.breaker_name = breaker_name
        self.time_until_retry = time_until_retry
        super().__init__(
            f"Circuit breaker '{breaker_name}' is open. "
            f"Retry in {time_until_retry:.1f}s"
        )


@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker."""

    failure_threshold: int = DEFAULT_CB_FAILURE_THRESHOLD
    success_threshold: int = DEFAULT_CB_SUCCESS_THRESHOLD
    timeout_seconds: float = DEFAULT_CB_TIMEOUT_SECONDS
    window_seconds: float = DEFAULT_CB_WINDOW_SECONDS

    @classmethod
    def from_env(cls, prefix: str = "") -> "CircuitBreakerConfig":
        """Load configuration from environment variables."""
        env_prefix = f"ROUTEIQ_CB_{prefix.upper()}_" if prefix else "ROUTEIQ_CB_"

        def get_int(key: str, default: int) -> int:
            try:
                return int(
                    os.getenv(
                        f"{env_prefix}{key}",
                        os.getenv(f"ROUTEIQ_CB_{key}", str(default)),
                    )
                )
            except ValueError:
                return default

        def get_float(key: str, default: float) -> float:
            try:
                return float(
                    os.getenv(
                        f"{env_prefix}{key}",
                        os.getenv(f"ROUTEIQ_CB_{key}", str(default)),
                    )
                )
            except ValueError:
                return default

        return cls(
            failure_threshold=get_int(
                "FAILURE_THRESHOLD", DEFAULT_CB_FAILURE_THRESHOLD
            ),
            success_threshold=get_int(
                "SUCCESS_THRESHOLD", DEFAULT_CB_SUCCESS_THRESHOLD
            ),
            timeout_seconds=get_float("TIMEOUT_SECONDS", DEFAULT_CB_TIMEOUT_SECONDS),
            window_seconds=get_float("WINDOW_SECONDS", DEFAULT_CB_WINDOW_SECONDS),
        )


class CircuitBreaker:
    """
    Circuit breaker for protecting external service calls.

    States:
    - CLOSED: Normal operation. Failures are tracked in a sliding window.
              When failure_threshold is reached, transitions to OPEN.
    - OPEN: All calls immediately fail with CircuitBreakerOpenError.
            After timeout_seconds, transitions to HALF_OPEN.
    - HALF_OPEN: A limited number of calls are allowed through.
                 If success_threshold is reached, transitions to CLOSED.
                 If any call fails, transitions back to OPEN.

    Usage:
        breaker = CircuitBreaker("database")

        try:
            async with breaker.execute():
                await db.execute("SELECT 1")
        except CircuitBreakerOpenError:
            # Handle fast failure
            pass
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ):
        self.name = name
        self._config = config or CircuitBreakerConfig.from_env(name)
        self._state = CircuitBreakerState.CLOSED
        self._failures: deque[float] = deque()  # Timestamps of failures
        self._success_count = 0  # Successes in half-open state
        self._opened_at: float | None = None  # When circuit opened
        self._lock = asyncio.Lock()
        self._last_failure_error: str | None = None

    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit state (checking for timeout transition)."""
        if self._state == CircuitBreakerState.OPEN:
            # Check if timeout has passed -> transition to half-open
            if (
                self._opened_at
                and time.monotonic() - self._opened_at >= self._config.timeout_seconds
            ):
                return CircuitBreakerState.HALF_OPEN
        return self._state

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing fast)."""
        return self.state == CircuitBreakerState.OPEN

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitBreakerState.CLOSED

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self.state == CircuitBreakerState.HALF_OPEN

    @property
    def time_until_retry(self) -> float:
        """Seconds until retry is allowed (0 if not open)."""
        if self._state != CircuitBreakerState.OPEN or not self._opened_at:
            return 0.0
        elapsed = time.monotonic() - self._opened_at
        remaining = self._config.timeout_seconds - elapsed
        return max(0.0, remaining)

    @property
    def failure_count(self) -> int:
        """Current failure count in sliding window."""
        self._cleanup_old_failures()
        return len(self._failures)

    @property
    def last_failure_error(self) -> str | None:
        """Last error message that caused a failure."""
        return self._last_failure_error

    def _cleanup_old_failures(self) -> None:
        """Remove failures outside the sliding window."""
        cutoff = time.monotonic() - self._config.window_seconds
        while self._failures and self._failures[0] < cutoff:
            self._failures.popleft()

    async def _transition_to(self, new_state: CircuitBreakerState) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state

        if new_state == CircuitBreakerState.OPEN:
            self._opened_at = time.monotonic()
            self._success_count = 0
            logger.warning(
                f"Circuit breaker '{self.name}' OPENED after {self.failure_count} failures. "
                f"Will retry in {self._config.timeout_seconds}s"
            )
        elif new_state == CircuitBreakerState.HALF_OPEN:
            self._success_count = 0
            logger.info(
                f"Circuit breaker '{self.name}' now HALF_OPEN, testing recovery"
            )
        elif new_state == CircuitBreakerState.CLOSED:
            self._failures.clear()
            self._opened_at = None
            self._success_count = 0
            self._last_failure_error = None
            if old_state != CircuitBreakerState.CLOSED:
                logger.info(f"Circuit breaker '{self.name}' CLOSED, service recovered")

    async def record_success(self) -> None:
        """Record a successful call."""
        async with self._lock:
            current_state = self.state  # Triggers timeout check

            if current_state == CircuitBreakerState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._config.success_threshold:
                    await self._transition_to(CircuitBreakerState.CLOSED)
            # In CLOSED state, successes are not specifically tracked

    async def record_failure(self, error: str | None = None) -> None:
        """Record a failed call."""
        async with self._lock:
            self._last_failure_error = error
            current_state = self.state  # Triggers timeout check

            if current_state == CircuitBreakerState.HALF_OPEN:
                # Any failure in half-open immediately opens the circuit
                await self._transition_to(CircuitBreakerState.OPEN)
            elif current_state == CircuitBreakerState.CLOSED:
                # Track failure in sliding window
                self._cleanup_old_failures()
                self._failures.append(time.monotonic())

                if len(self._failures) >= self._config.failure_threshold:
                    await self._transition_to(CircuitBreakerState.OPEN)

    async def allow_request(self) -> bool:
        """
        Check if a request should be allowed.

        Returns:
            True if request can proceed, False if circuit is open
        """
        async with self._lock:
            current_state = self.state

            if current_state == CircuitBreakerState.CLOSED:
                return True
            elif current_state == CircuitBreakerState.HALF_OPEN:
                # In half-open, allow limited requests
                return True
            else:  # OPEN
                return False

    @asynccontextmanager
    async def execute(self) -> AsyncIterator[None]:
        """
        Context manager for executing code with circuit breaker protection.

        Raises:
            CircuitBreakerOpenError: If circuit is open
        """
        async with self._lock:
            current_state = self.state

            if current_state == CircuitBreakerState.OPEN:
                raise CircuitBreakerOpenError(self.name, self.time_until_retry)

            # Transition to half-open if timeout passed (handled by state property)
            if (
                self._state == CircuitBreakerState.OPEN
                and current_state == CircuitBreakerState.HALF_OPEN
            ):
                await self._transition_to(CircuitBreakerState.HALF_OPEN)

        try:
            yield
            await self.record_success()
        except CircuitBreakerOpenError:
            # Re-raise circuit breaker errors without recording as failure
            raise
        except Exception as e:
            await self.record_failure(str(e))
            raise

    async def force_open(self) -> None:
        """Manually open the circuit (for testing or manual intervention)."""
        async with self._lock:
            await self._transition_to(CircuitBreakerState.OPEN)

    async def force_closed(self) -> None:
        """Manually close the circuit (for testing or manual intervention)."""
        async with self._lock:
            await self._transition_to(CircuitBreakerState.CLOSED)

    async def reset(self) -> None:
        """Reset the circuit breaker to initial state."""
        async with self._lock:
            self._state = CircuitBreakerState.CLOSED
            self._failures.clear()
            self._success_count = 0
            self._opened_at = None
            self._last_failure_error = None

    def get_status(self) -> dict[str, Any]:
        """Get current circuit breaker status."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self._success_count,
            "time_until_retry": round(self.time_until_retry, 1),
            "last_failure_error": self._last_failure_error,
            "config": {
                "failure_threshold": self._config.failure_threshold,
                "success_threshold": self._config.success_threshold,
                "timeout_seconds": self._config.timeout_seconds,
                "window_seconds": self._config.window_seconds,
            },
        }


# =============================================================================
# Circuit Breaker Manager and Degraded Mode
# =============================================================================


@dataclass
class DegradedComponent:
    """Information about a degraded component."""

    name: str
    is_degraded: bool
    reason: str | None = None
    since: float | None = None  # Timestamp when degradation started


class CircuitBreakerManager:
    """
    Manages circuit breakers for multiple services and tracks degraded mode.

    Provides named circuit breakers for:
    - database: PostgreSQL connections
    - redis: Redis cache connections
    - leader_election: Leader election operations
    - provider:<name>: Per-LLM-provider HTTP call protection

    Exposes aggregated degraded mode status for health checks.
    """

    # Standard breaker names
    DATABASE = "database"
    REDIS = "redis"
    LEADER_ELECTION = "leader_election"

    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    def get_breaker(self, name: str) -> CircuitBreaker:
        """
        Get or create a circuit breaker by name.

        Args:
            name: Name of the circuit breaker (e.g., "database", "redis")

        Returns:
            CircuitBreaker instance
        """
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name=name,
                config=CircuitBreakerConfig.from_env(name),
            )
        return self._breakers[name]

    def get_or_create_provider_breaker(self, provider_name: str) -> CircuitBreaker:
        """
        Get or create a circuit breaker for an LLM provider.

        Provider breakers use per-provider configuration from environment:
        ROUTEIQ_CB_{PROVIDER}_FAILURE_THRESHOLD, etc.

        Args:
            provider_name: LLM provider name (e.g., 'openai', 'anthropic', 'bedrock')

        Returns:
            CircuitBreaker instance keyed as ``provider:<provider_name>``
        """
        key = f"provider:{provider_name}"
        if key not in self._breakers:
            config = CircuitBreakerConfig.from_env(prefix=provider_name)
            self._breakers[key] = CircuitBreaker(name=key, config=config)
        return self._breakers[key]

    @property
    def database(self) -> CircuitBreaker:
        """Get the database circuit breaker."""
        return self.get_breaker(self.DATABASE)

    @property
    def redis(self) -> CircuitBreaker:
        """Get the Redis circuit breaker."""
        return self.get_breaker(self.REDIS)

    @property
    def leader_election(self) -> CircuitBreaker:
        """Get the leader election circuit breaker."""
        return self.get_breaker(self.LEADER_ELECTION)

    def is_degraded(self) -> bool:
        """Check if any circuit breaker is open (system is degraded)."""
        return any(breaker.is_open for breaker in self._breakers.values())

    def get_degraded_components(self) -> list[DegradedComponent]:
        """Get list of degraded components."""
        components = []
        for name, breaker in self._breakers.items():
            if breaker.is_open:
                components.append(
                    DegradedComponent(
                        name=name,
                        is_degraded=True,
                        reason=breaker.last_failure_error,
                        since=breaker._opened_at,
                    )
                )
        return components

    def get_status(self) -> dict[str, Any]:
        """Get aggregated status for all circuit breakers."""
        breakers_status = {
            name: breaker.get_status() for name, breaker in self._breakers.items()
        }

        degraded_names = [
            name for name, breaker in self._breakers.items() if breaker.is_open
        ]

        return {
            "is_degraded": self.is_degraded(),
            "degraded_components": degraded_names,
            "breakers": breakers_status,
        }

    async def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            await breaker.reset()


# Global singleton for the circuit breaker manager
_circuit_breaker_manager: CircuitBreakerManager | None = None


def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Get or create the global circuit breaker manager singleton."""
    global _circuit_breaker_manager
    if _circuit_breaker_manager is None:
        _circuit_breaker_manager = CircuitBreakerManager()
    return _circuit_breaker_manager


def reset_circuit_breaker_manager() -> None:
    """Reset the global circuit breaker manager (for testing)."""
    global _circuit_breaker_manager
    _circuit_breaker_manager = None


# =============================================================================
# Per-Provider Circuit Breaker Support
# =============================================================================

# Feature flag: opt-in per-provider circuit breakers for LLM HTTP calls
PROVIDER_CB_ENABLED = (
    os.getenv("ROUTEIQ_PROVIDER_CB_ENABLED", "false").lower() == "true"
)


def _notify_circuit_breaker_change(provider_name: str, breaker: CircuitBreaker) -> None:
    """Emit OTel event and log on circuit breaker state change."""
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if span and span.is_recording():
            span.add_event(
                "circuit_breaker.provider.state_change",
                attributes={
                    "provider": provider_name,
                    "state": breaker.state.value,
                    "failure_count": breaker.failure_count,
                },
            )
    except Exception:
        pass  # OTel is best-effort; never break the request path


async def execute_with_provider_circuit_breaker(
    provider_name: str,
    operation: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """
    Execute an operation with provider circuit breaker protection.

    Only applies if ``ROUTEIQ_PROVIDER_CB_ENABLED`` is true.
    Falls back to direct execution if disabled.

    Args:
        provider_name: LLM provider name (e.g., 'openai', 'anthropic')
        operation: Async callable to execute
        *args: Positional arguments forwarded to *operation*
        **kwargs: Keyword arguments forwarded to *operation*

    Returns:
        The result of *operation*

    Raises:
        CircuitBreakerOpenError: If the provider circuit is open
    """
    if not PROVIDER_CB_ENABLED:
        return await operation(*args, **kwargs)

    cb_manager = get_circuit_breaker_manager()
    breaker = cb_manager.get_or_create_provider_breaker(provider_name)

    try:
        async with breaker.execute():
            return await operation(*args, **kwargs)
    except CircuitBreakerOpenError:
        # Notify plugin hooks / OTel about state change
        _notify_circuit_breaker_change(provider_name, breaker)
        raise


# =============================================================================
# Wrapped Database Operations with Circuit Breaker
# =============================================================================


async def execute_with_circuit_breaker(
    breaker: CircuitBreaker,
    operation: Callable[..., Any],
    *args: Any,
    fallback: Callable[[], Any] | None = None,
    **kwargs: Any,
) -> Any:
    """
    Execute an async operation with circuit breaker protection.

    Args:
        breaker: The circuit breaker to use
        operation: The async callable to execute
        *args: Arguments to pass to the operation
        fallback: Optional fallback callable if circuit is open
        **kwargs: Keyword arguments to pass to the operation

    Returns:
        Result of the operation or fallback

    Raises:
        CircuitBreakerOpenError: If circuit is open and no fallback provided
    """
    try:
        async with breaker.execute():
            return await operation(*args, **kwargs)
    except CircuitBreakerOpenError:
        if fallback is not None:
            logger.debug(f"Circuit breaker '{breaker.name}' open, using fallback")
            result = fallback()
            if asyncio.iscoroutine(result):
                return await result
            return result
        raise
