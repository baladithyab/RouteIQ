"""
Shared Redis Client Builder
============================

Centralises Redis connection configuration so every consumer
(health check, quota, MCP gateway, cache plugin, conversation affinity)
uses the same env-var-driven settings: REDIS_HOST, REDIS_PORT,
REDIS_PASSWORD, REDIS_SSL, and REDIS_DB.

Usage (async):
    from litellm_llmrouter.redis_pool import create_async_redis_client

    client = create_async_redis_client()
    await client.ping()

Usage (sync):
    from litellm_llmrouter.redis_pool import create_sync_redis_client

    client = create_sync_redis_client()
    client.ping()

All factory functions read environment variables at call time so they
respect late-binding (hot-reload, test overrides via ``monkeypatch``).
"""

from __future__ import annotations

import os
from typing import Any


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


def create_async_redis_client(
    *,
    decode_responses: bool = True,
    socket_connect_timeout: float = 2.0,
    socket_timeout: float = 2.0,
    **overrides: Any,
) -> Any:
    """Create an async ``redis.asyncio.Redis`` client from env vars.

    Any keyword *overrides* are merged on top of the environment-derived
    settings, so callers can customise individual options.

    Returns:
        A ``redis.asyncio.Redis`` instance (not yet connected).

    Raises:
        ImportError: If the ``redis`` package is not installed.
    """
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

    Returns:
        A ``redis.Redis`` instance.

    Raises:
        ImportError: If the ``redis`` package is not installed.
    """
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
