"""
Shared Redis Client Builder
============================

Centralises Redis connection configuration so every consumer
(health check, quota, MCP gateway, cache plugin, conversation affinity)
uses the same env-var-driven settings: REDIS_HOST, REDIS_PORT,
REDIS_PASSWORD, REDIS_SSL, and REDIS_DB.

Preferred usage (singleton, async):
    from litellm_llmrouter.redis_pool import get_async_redis_client

    client = await get_async_redis_client()
    if client:
        await client.ping()

Legacy usage (creates new client per call — deprecated):
    from litellm_llmrouter.redis_pool import create_async_redis_client

    client = create_async_redis_client()
    await client.ping()

Usage (sync):
    from litellm_llmrouter.redis_pool import create_sync_redis_client

    client = create_sync_redis_client()
    client.ping()

All factory functions read environment variables at call time so they
respect late-binding (hot-reload, test overrides via ``monkeypatch``).
The singleton ``get_async_redis_client()`` creates the client once and
reuses it across all callers, with automatic health-check ping.
"""

from __future__ import annotations

import asyncio
import logging
import os
import warnings
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------
_async_client: Optional[Any] = None  # redis.asyncio.Redis | None
_async_client_lock = asyncio.Lock()
_sync_client: Optional[Any] = None  # redis.Redis | None


def _redis_settings() -> dict[str, Any]:
    """Read common Redis settings from environment variables.

    Returns a dict suitable for unpacking into the ``redis.Redis`` /
    ``redis.asyncio.Redis`` constructors.
    """
    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", "6379"))
    password = os.getenv("REDIS_PASSWORD") or None
    ssl = os.getenv("REDIS_SSL", "false").lower() in ("true", "1", "yes")
    db = int(os.getenv("REDIS_DB", "0"))
    return {
        "host": host,
        "port": port,
        "password": password,
        "ssl": ssl,
        "db": db,
    }


def _warn_no_password() -> None:
    """Log a warning when Redis is configured without a password."""
    if not os.getenv("REDIS_PASSWORD"):
        logger.warning(
            "Redis connection configured without password (REDIS_PASSWORD not set). "
            "This is insecure for production deployments."
        )


# ---------------------------------------------------------------------------
# Async singleton
# ---------------------------------------------------------------------------


async def get_async_redis_client() -> Optional[Any]:
    """Get or create the shared async Redis client singleton.

    Returns the cached ``redis.asyncio.Redis`` instance, creating it on
    first call.  If the cached client fails a ``PING`` health-check it is
    discarded and a fresh one is created.

    Returns:
        A ``redis.asyncio.Redis`` instance, or ``None`` if Redis is not
        configured (``REDIS_HOST`` is not set).
    """
    global _async_client

    # Fast path: reuse existing healthy client
    if _async_client is not None:
        try:
            await _async_client.ping()
            return _async_client
        except Exception:
            logger.debug("Async Redis client health-check failed, reconnecting")
            _async_client = None

    async with _async_client_lock:
        # Double-check after acquiring the lock
        if _async_client is not None:
            return _async_client

        host = os.getenv("REDIS_HOST")
        if not host:
            return None

        import redis.asyncio as aioredis

        _warn_no_password()

        _async_client = aioredis.Redis(
            host=host,
            port=int(os.getenv("REDIS_PORT", "6379")),
            password=os.getenv("REDIS_PASSWORD") or None,
            ssl=os.getenv("REDIS_SSL", "false").lower() in ("true", "1", "yes"),
            db=int(os.getenv("REDIS_DB", "0")),
            decode_responses=True,
            socket_connect_timeout=2.0,
            socket_timeout=2.0,
            retry_on_timeout=True,
            health_check_interval=30,
        )
        logger.info("Async Redis client singleton created (host=%s)", host)
        return _async_client


async def close_async_redis_client() -> None:
    """Close the async Redis client singleton.

    Safe to call multiple times or when no client exists.
    Should be called during application shutdown.
    """
    global _async_client
    if _async_client is not None:
        try:
            await _async_client.close()
            logger.info("Async Redis client closed")
        except Exception as exc:
            logger.warning("Error closing async Redis client: %s", exc)
        finally:
            _async_client = None


# ---------------------------------------------------------------------------
# Sync singleton
# ---------------------------------------------------------------------------


def get_sync_redis_client() -> Optional[Any]:
    """Get or create the shared sync Redis client singleton.

    Returns the cached ``redis.Redis`` instance, creating it on first
    call.

    Returns:
        A ``redis.Redis`` instance, or ``None`` if Redis is not
        configured (``REDIS_HOST`` is not set).
    """
    global _sync_client

    if _sync_client is not None:
        return _sync_client

    host = os.getenv("REDIS_HOST")
    if not host:
        return None

    import redis

    _warn_no_password()

    _sync_client = redis.Redis(
        host=host,
        port=int(os.getenv("REDIS_PORT", "6379")),
        password=os.getenv("REDIS_PASSWORD") or None,
        ssl=os.getenv("REDIS_SSL", "false").lower() in ("true", "1", "yes"),
        db=int(os.getenv("REDIS_DB", "0")),
        decode_responses=True,
        socket_timeout=5.0,
    )
    logger.info("Sync Redis client singleton created (host=%s)", host)
    return _sync_client


# ---------------------------------------------------------------------------
# Reset for testing
# ---------------------------------------------------------------------------


def reset_redis_clients() -> None:
    """Reset all client singletons.

    Intended for test fixtures (``autouse=True``) to prevent cross-test
    contamination.  Does **not** close open connections — use
    :func:`close_async_redis_client` for graceful shutdown.
    """
    global _async_client, _sync_client
    _async_client = None
    _sync_client = None


# ---------------------------------------------------------------------------
# Legacy factory functions (backward-compatible, deprecated)
# ---------------------------------------------------------------------------


def create_async_redis_client(
    *,
    decode_responses: bool = True,
    socket_connect_timeout: float = 2.0,
    socket_timeout: float = 2.0,
    **overrides: Any,
) -> Any:
    """Create an async ``redis.asyncio.Redis`` client from env vars.

    .. deprecated::
        Use :func:`get_async_redis_client` instead to share a single
        connection across all callers.

    Any keyword *overrides* are merged on top of the environment-derived
    settings, so callers can customise individual options.

    Returns:
        A ``redis.asyncio.Redis`` instance (not yet connected).

    Raises:
        ImportError: If the ``redis`` package is not installed.
    """
    warnings.warn(
        "create_async_redis_client() creates a new connection on every call. "
        "Use get_async_redis_client() for the shared singleton instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    import redis.asyncio as aioredis

    kwargs = {
        **_redis_settings(),
        "decode_responses": decode_responses,
        "socket_connect_timeout": socket_connect_timeout,
        "socket_timeout": socket_timeout,
        **overrides,
    }
    return aioredis.Redis(**kwargs)


def create_sync_redis_client(
    *,
    decode_responses: bool = True,
    socket_timeout: float = 5.0,
    **overrides: Any,
) -> Any:
    """Create a synchronous ``redis.Redis`` client from env vars.

    .. deprecated::
        Use :func:`get_sync_redis_client` instead to share a single
        connection across all callers.

    Returns:
        A ``redis.Redis`` instance.

    Raises:
        ImportError: If the ``redis`` package is not installed.
    """
    warnings.warn(
        "create_sync_redis_client() creates a new connection on every call. "
        "Use get_sync_redis_client() for the shared singleton instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    import redis

    kwargs = {
        **_redis_settings(),
        "decode_responses": decode_responses,
        "socket_timeout": socket_timeout,
        **overrides,
    }
    return redis.Redis(**kwargs)


def build_redis_url() -> str:
    """Build a Redis URL from environment variables.

    Useful for consumers that accept a URL string rather than keyword
    arguments (e.g. ``redis.asyncio.from_url``).

    Returns:
        A ``redis://`` or ``rediss://`` URL string.
    """
    settings = _redis_settings()
    scheme = "rediss" if settings["ssl"] else "redis"
    auth = f":{settings['password']}@" if settings["password"] else ""
    return f"{scheme}://{auth}{settings['host']}:{settings['port']}/{settings['db']}"
