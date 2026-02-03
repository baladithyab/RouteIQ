"""
HTTP Client Pool - Shared Outbound HTTP Client Management
==========================================================

Provides a shared httpx.AsyncClient pool for outbound HTTP requests,
eliminating per-request client instantiation overhead and enabling
proper connection pooling.

Features:
- Singleton AsyncClient with configurable limits
- Proper lifecycle management (startup/shutdown hooks)
- Feature flag for rollback safety (ROUTEIQ_HTTP_CLIENT_POOLING_ENABLED)
- Instrumentation hooks for tracking instantiation counts

Usage:
    from litellm_llmrouter.http_client_pool import get_http_client

    # In async context
    client = get_http_client()
    response = await client.get("https://example.com")

Lifecycle:
    # On app startup (called by gateway/app.py)
    await startup_http_client_pool()

    # On app shutdown
    await shutdown_http_client_pool()

Rollback Safety:
    Set ROUTEIQ_HTTP_CLIENT_POOLING_ENABLED to false to disable pooling and
    fall back to per-request client creation. Default is 'true'.

See: https://www.python-httpx.org/advanced/clients/#client-instances
"""

import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx

logger = logging.getLogger(__name__)

# =============================================================================
# Feature Flag
# =============================================================================

# ROUTEIQ_HTTP_CLIENT_POOLING_ENABLED: When true (default), uses shared pooled client.
# When false, falls back to per-request client creation for rollback safety.
#
# Rollback Safety: Default is True to enable pooling. Set to "false" to disable
# and fall back to per-request clients if issues are detected.
#
# Toggle: Set environment variable ROUTEIQ_HTTP_CLIENT_POOLING_ENABLED=false to disable.
ROUTEIQ_HTTP_CLIENT_POOLING_ENABLED = (
    os.getenv("ROUTEIQ_HTTP_CLIENT_POOLING_ENABLED", "true").lower() == "true"
)

# =============================================================================
# Configuration
# =============================================================================

# Connection pool limits
# These are conservative defaults suitable for gateway workloads
HTTP_CLIENT_MAX_CONNECTIONS = int(os.getenv("HTTP_CLIENT_MAX_CONNECTIONS", "100"))
HTTP_CLIENT_MAX_KEEPALIVE = int(os.getenv("HTTP_CLIENT_MAX_KEEPALIVE", "20"))

# Default timeout configuration (can be overridden per-request)
HTTP_CLIENT_DEFAULT_TIMEOUT = float(os.getenv("HTTP_CLIENT_DEFAULT_TIMEOUT", "60.0"))

# =============================================================================
# Instrumentation (for testing and observability)
# =============================================================================

# Counter for tracking client instantiations (for testing)
_client_instantiation_count = 0
_client_instantiation_lock = threading.Lock()


def get_client_instantiation_count() -> int:
    """
    Get the number of times a pooled client has been instantiated.

    This is useful for testing to verify that the client is reused
    and not created on every request.

    Returns:
        Number of client instantiations since process start or reset.
    """
    return _client_instantiation_count


def reset_client_instantiation_count() -> None:
    """
    Reset the client instantiation counter.

    WARNING: For testing purposes only.
    """
    global _client_instantiation_count
    with _client_instantiation_lock:
        _client_instantiation_count = 0


def _increment_instantiation_count() -> int:
    """Increment and return the instantiation count."""
    global _client_instantiation_count
    with _client_instantiation_lock:
        _client_instantiation_count += 1
        return _client_instantiation_count


# =============================================================================
# Shared Client Singleton
# =============================================================================

_http_client: httpx.AsyncClient | None = None
_http_client_lock = threading.Lock()


def _create_client_limits() -> httpx.Limits:
    """
    Create connection pool limits for the shared client.

    Returns:
        httpx.Limits configured with pool settings.
    """
    return httpx.Limits(
        max_connections=HTTP_CLIENT_MAX_CONNECTIONS,
        max_keepalive_connections=HTTP_CLIENT_MAX_KEEPALIVE,
    )


def _create_client() -> httpx.AsyncClient:
    """
    Create a new AsyncClient instance with pooling configuration.

    This function is called once on startup (when pooling enabled)
    and tracks instantiation count for testing.

    Returns:
        Configured httpx.AsyncClient instance.
    """
    count = _increment_instantiation_count()
    logger.info(
        f"HTTP client pool: creating client #{count} "
        f"(max_connections={HTTP_CLIENT_MAX_CONNECTIONS}, "
        f"max_keepalive={HTTP_CLIENT_MAX_KEEPALIVE})"
    )

    return httpx.AsyncClient(
        limits=_create_client_limits(),
        timeout=httpx.Timeout(HTTP_CLIENT_DEFAULT_TIMEOUT),
        # HTTP/2 is disabled by default as h2 package may not be installed
        # Enable via optional dependency: httpx[http2]
        http2=False,
        # Follow redirects by default
        follow_redirects=True,
    )


async def startup_http_client_pool() -> None:
    """
    Initialize the shared HTTP client pool.

    Should be called during application startup. Safe to call multiple times
    (idempotent).

    This function is a no-op if ROUTEIQ_HTTP_CLIENT_POOLING_ENABLED=false.
    """
    global _http_client

    if not ROUTEIQ_HTTP_CLIENT_POOLING_ENABLED:
        logger.info(
            "HTTP client pool: pooling DISABLED (ROUTEIQ_HTTP_CLIENT_POOLING_ENABLED=false)"
        )
        return

    with _http_client_lock:
        if _http_client is None:
            _http_client = _create_client()
            logger.info("HTTP client pool: initialized")
        else:
            logger.debug("HTTP client pool: already initialized (no-op)")


async def shutdown_http_client_pool() -> None:
    """
    Shutdown the shared HTTP client pool.

    Should be called during application shutdown for proper cleanup.
    Closes all connections and releases resources.

    This function is a no-op if the client was never initialized.
    """
    global _http_client

    with _http_client_lock:
        if _http_client is not None:
            client = _http_client
            _http_client = None
        else:
            client = None

    if client is not None:
        logger.info("HTTP client pool: shutting down...")
        await client.aclose()
        logger.info("HTTP client pool: shutdown complete")
    else:
        logger.debug("HTTP client pool: no client to shutdown (no-op)")


def get_http_client() -> httpx.AsyncClient:
    """
    Get the shared HTTP client for outbound requests.

    Returns the pooled client if ROUTEIQ_HTTP_CLIENT_POOLING_ENABLED=true and
    the pool has been initialized. Otherwise, raises RuntimeError.

    For fallback scenarios (pooling disabled or not initialized),
    use get_http_client_safe() or create_fallback_client() instead.

    Returns:
        The shared httpx.AsyncClient instance.

    Raises:
        RuntimeError: If pooling is disabled or client not initialized.
    """
    if not ROUTEIQ_HTTP_CLIENT_POOLING_ENABLED:
        raise RuntimeError(
            "HTTP client pooling is disabled. "
            "Use create_fallback_client() for per-request clients."
        )

    if _http_client is None:
        raise RuntimeError(
            "HTTP client pool not initialized. "
            "Call startup_http_client_pool() first."
        )

    return _http_client


def is_pooling_enabled() -> bool:
    """
    Check if HTTP client pooling is enabled.

    Returns:
        True if pooling is enabled and client is initialized.
    """
    return ROUTEIQ_HTTP_CLIENT_POOLING_ENABLED and _http_client is not None


def is_pool_initialized() -> bool:
    """
    Check if the HTTP client pool has been initialized.

    Returns:
        True if the pool has been initialized.
    """
    return _http_client is not None


@asynccontextmanager
async def create_fallback_client(
    timeout: float | httpx.Timeout | None = None,
    **kwargs: Any,
) -> AsyncIterator[httpx.AsyncClient]:
    """
    Create a per-request fallback client (for rollback safety).

    Use this when ROUTEIQ_HTTP_CLIENT_POOLING_ENABLED=false or when you need
    specific client configuration that differs from the pool defaults.

    This is the original behavior before pooling was introduced.

    Args:
        timeout: Request timeout (default: HTTP_CLIENT_DEFAULT_TIMEOUT)
        **kwargs: Additional arguments passed to AsyncClient

    Yields:
        A new httpx.AsyncClient that will be closed on exit.

    Example:
        async with create_fallback_client(timeout=30.0) as client:
            response = await client.get("https://example.com")
    """
    _increment_instantiation_count()  # Track for testing

    if timeout is None:
        timeout = HTTP_CLIENT_DEFAULT_TIMEOUT

    async with httpx.AsyncClient(timeout=timeout, **kwargs) as client:
        yield client


@asynccontextmanager
async def get_client_for_request(
    timeout: float | httpx.Timeout | None = None,
    **kwargs: Any,
) -> AsyncIterator[httpx.AsyncClient]:
    """
    Get an HTTP client for a request, with automatic fallback.

    This is the recommended way to get a client for outbound requests.
    It will:
    1. Use the pooled client if pooling is enabled and initialized
    2. Fall back to per-request client if pooling is disabled
    3. Fall back to per-request client if pool not initialized (with warning)

    The context manager ensures proper cleanup on fallback.

    Args:
        timeout: Optional timeout override for this request.
                 When using pooled client, timeout is set per-request.
                 When using fallback, timeout is set on client creation.
        **kwargs: Additional arguments (only used for fallback clients)

    Yields:
        An httpx.AsyncClient suitable for making requests.

    Example:
        async with get_client_for_request(timeout=30.0) as client:
            response = await client.post(url, json=data)
    """
    if is_pooling_enabled():
        # Use pooled client
        client = _http_client
        if client is not None:
            yield client
            return

    # Fallback to per-request client
    if ROUTEIQ_HTTP_CLIENT_POOLING_ENABLED and not is_pool_initialized():
        logger.warning(
            "HTTP client pool not initialized; falling back to per-request client. "
            "Ensure startup_http_client_pool() is called during app startup."
        )

    async with create_fallback_client(timeout=timeout, **kwargs) as client:
        yield client


# =============================================================================
# Testing Helpers
# =============================================================================


async def reset_http_client_pool() -> None:
    """
    Reset the HTTP client pool for testing.

    WARNING: For testing purposes only. Not safe to call while
    requests are in flight.
    """
    global _http_client

    with _http_client_lock:
        client = _http_client
        _http_client = None

    if client is not None:
        await client.aclose()

    reset_client_instantiation_count()
    logger.debug("HTTP client pool: reset for testing")
