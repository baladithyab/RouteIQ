"""
Resilience Primitives: Backpressure and Drain Mode
====================================================

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

Configuration via environment variables (disabled by default for non-breaking behavior):
- ROUTEIQ_MAX_CONCURRENT_REQUESTS: Max concurrent requests (0 = disabled)
- ROUTEIQ_DRAIN_TIMEOUT_SECONDS: Max time to wait for drain (default: 30)

Usage:
    from litellm_llmrouter.resilience import (
        BackpressureMiddleware,
        DrainManager,
        get_drain_manager,
    )

    # In app factory:
    drain_manager = get_drain_manager()
    drain_manager.attach(app)

    # Middleware is added by calling add_backpressure_middleware(app)
    # It wraps the ASGI app to track concurrency
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_MAX_CONCURRENT_REQUESTS = 0  # 0 = disabled
DEFAULT_DRAIN_TIMEOUT_SECONDS = 30
DEFAULT_EXCLUDED_PATHS = frozenset({
    "/_health/live",
    "/_health/ready",
    "/health/liveliness",
    "/health/readiness",
    "/health",
})


@dataclass
class ResilienceConfig:
    """Configuration for resilience primitives."""

    max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_REQUESTS
    drain_timeout_seconds: float = DEFAULT_DRAIN_TIMEOUT_SECONDS
    excluded_paths: frozenset[str] = field(default_factory=lambda: DEFAULT_EXCLUDED_PATHS)

    @classmethod
    def from_env(cls) -> "ResilienceConfig":
        """Load configuration from environment variables."""
        max_concurrent_str = os.getenv("ROUTEIQ_MAX_CONCURRENT_REQUESTS", "0")
        drain_timeout_str = os.getenv("ROUTEIQ_DRAIN_TIMEOUT_SECONDS", str(DEFAULT_DRAIN_TIMEOUT_SECONDS))

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

    async def _send_503_response(self, send: Send, request_id: str | None = None) -> None:
        """Send a 503 over-capacity JSON response."""
        body = {
            "error": "over_capacity",
            "message": "Server is at capacity, please retry later",
        }
        if request_id:
            body["request_id"] = request_id

        import json
        body_bytes = json.dumps(body).encode("utf-8")

        await send({
            "type": "http.response.start",
            "status": 503,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body_bytes)).encode()),
                (b"retry-after", b"1"),
            ],
        })
        await send({
            "type": "http.response.body",
            "body": body_bytes,
            "more_body": False,
        })

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
                return value.decode("utf-8", errors="replace")
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
        original_app = app.app if hasattr(app, 'app') else None
        if original_app:
            app.app = BackpressureMiddleware(original_app, config, drain_manager)
        return False

    # Add as ASGI middleware by wrapping the app
    original_app = app.app if hasattr(app, 'app') else None
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
