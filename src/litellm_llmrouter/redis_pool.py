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
import re
import warnings
from typing import Any, Optional

from litellm_llmrouter.settings import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------
_async_client: Optional[Any] = None  # redis.asyncio.Redis | None
_async_client_lock: Optional[asyncio.Lock] = None
_sync_client: Optional[Any] = None  # redis.Redis | None
_last_health_check: float = 0.0
_HEALTH_CHECK_INTERVAL = 30.0  # seconds


def _get_async_client_lock() -> asyncio.Lock:
    """Lazily initialize the async client lock to avoid binding to wrong event loop."""
    global _async_client_lock
    if _async_client_lock is None:
        _async_client_lock = asyncio.Lock()
    return _async_client_lock


def _redis_settings() -> dict[str, Any]:
    """Read common Redis settings from env vars, with typed settings fallback.

    Legacy env vars (``REDIS_HOST``, ``REDIS_PORT``, etc.) take precedence
    over the typed settings model (which uses ``ROUTEIQ_REDIS__*`` prefix).
    This ensures backward compatibility while supporting the new settings.

    Returns a dict suitable for unpacking into the ``redis.Redis`` /
    ``redis.asyncio.Redis`` constructors.
    """
    # Legacy env vars take precedence (most existing deployments use these)
    env_host = os.getenv("REDIS_HOST")
    env_port = os.getenv("REDIS_PORT")
    env_password = os.getenv("REDIS_PASSWORD")
    env_ssl = os.getenv("REDIS_SSL")
    env_db = os.getenv("REDIS_DB")
    # REDIS_USERNAME stays in the legacy REDIS_* namespace (read directly here, not
    # via get_settings) per the module's convention. On the ADR-0029 IAM-auth path
    # this is the CacheIamUserName (user_id == user_name) the chart emits.
    env_username = os.getenv("REDIS_USERNAME")

    # If any legacy env var is set, use env vars exclusively
    if any(
        v is not None
        for v in (env_host, env_port, env_password, env_ssl, env_db, env_username)
    ):
        return {
            "host": env_host or "localhost",
            "port": int(env_port) if env_port else 6379,
            "password": env_password or None,
            "ssl": (env_ssl or "false").lower() in ("true", "1", "yes"),
            "db": int(env_db) if env_db else 0,
            "username": env_username or None,
        }

    # Fall back to typed settings (reads ROUTEIQ_REDIS__* vars)
    try:
        settings = get_settings()
        return {
            "host": settings.redis.host or "localhost",
            "port": settings.redis.port,
            "password": settings.redis.password,
            "ssl": settings.redis.ssl,
            "db": settings.redis.db,
            "username": settings.redis.username,
        }
    except Exception:
        return {
            "host": "localhost",
            "port": 6379,
            "password": None,
            "ssl": False,
            "db": 0,
            "username": None,
        }


def _warn_no_password() -> None:
    """Log a warning when Redis is configured without a password."""
    # On the IAM-auth path a missing REDIS_PASSWORD is EXPECTED (the SigV4
    # elasticache:Connect token is the AUTH), so suppress the warning entirely.
    try:
        if get_settings().redis.iam_auth:
            return
    except Exception:
        pass
    has_password = bool(os.getenv("REDIS_PASSWORD"))
    if not has_password:
        try:
            settings = get_settings()
            has_password = bool(settings.redis.password)
        except Exception:
            pass
    if not has_password:
        logger.warning(
            "Redis connection configured without password (REDIS_PASSWORD not set). "
            "This is insecure for production deployments."
        )


_AWS_REGION_RE = re.compile(r"^[a-z]{2}-[a-z]+-\d+$")


def _region_from_cache_host(host: str | None) -> str | None:
    """Best-effort region from an ElastiCache endpoint hostname.

    Serverless endpoints use SHORTENED region tokens (``use1``, not
    ``us-east-1``), so prefer ``settings.redis.iam_region`` / ``AWS_REGION``.
    Returns the long-form region only when the host carries a label that
    matches the canonical AWS region shape (``<area>-<word>-<num>``, e.g.
    ``us-east-1``); otherwise None.

    NOTE: a naive "two hyphens" heuristic is WRONG -- a cache-name label such
    as ``routeiq-cache-sl`` also has two hyphens and would be mistaken for a
    region, signing the SigV4 token with a bogus region and breaking auth even
    when AWS_REGION is set. The strict regex avoids that (RouteIQ-d3a4).
    """
    if not host or ".cache.amazonaws.com" not in host:
        return None
    for p in host.split("."):
        # Match only the canonical long-form region label (e.g. us-east-1,
        # ap-southeast-2); never the shortened serverless token (use1) or a
        # cache-name label that merely happens to contain hyphens.
        if _AWS_REGION_RE.match(p):
            return p
    return None


def _mint_elasticache_token(user: str, cache_name: str, region: str) -> str:
    """Mint an ``elasticache:Connect`` SigV4-presigned AUTH token (local, no network).

    Token = the SigV4-signed ``GET https://<cache-name>/?Action=connect&User=<user>``
    URL with the ``https://`` scheme stripped. Valid 900s. Constructed inline (no
    module-level cache -> no new reset obligation). NEVER log the returned token.
    """
    import boto3
    from botocore.auth import SigV4QueryAuth
    from botocore.awsrequest import AWSRequest

    creds = boto3.session.Session().get_credentials()
    if creds is None:
        raise RuntimeError("no AWS credentials for elasticache:Connect token mint")
    signer = SigV4QueryAuth(creds, "elasticache", region, expires=900)
    req = AWSRequest(
        method="GET",
        url=f"https://{cache_name}/",
        params={"Action": "connect", "User": user},
    )
    signer.add_auth(req)
    return req.prepare().url.removeprefix("https://")


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
    global _async_client, _last_health_check

    # Fast path: reuse existing healthy client with time-gated health check
    if _async_client is not None:
        import time

        if time.monotonic() - _last_health_check > _HEALTH_CHECK_INTERVAL:
            try:
                await _async_client.ping()
                _last_health_check = time.monotonic()
            except Exception:
                logger.debug("Async Redis client health-check failed, reconnecting")
                _async_client = None
        else:
            return _async_client

    async with _get_async_client_lock():
        # Double-check after acquiring the lock
        if _async_client is not None:
            return _async_client

        # Check legacy env var first, then typed settings
        host = os.getenv("REDIS_HOST")
        if not host:
            try:
                settings = get_settings()
                host = settings.redis.host
            except Exception:
                pass
        if not host:
            return None

        import redis.asyncio as aioredis

        _warn_no_password()

        redis_cfg = _redis_settings()
        username = redis_cfg.get("username")
        password = redis_cfg["password"]
        # IAM auth (ADR-0029): when ROUTEIQ_REDIS_IAM_AUTH is set, present
        # REDIS_USERNAME (CacheIamUserName) + a short-lived elasticache:Connect
        # SigV4 token AS the password (not in a URL, to avoid quoting the token).
        # Minting fresh per client build gives the 15-min refresh (the singleton's
        # 30s health-check + reconnect rebuilds -> re-mints). Default OFF -> the
        # static path is unchanged. On mint failure we log (NEVER the token) and
        # fall through to the static AUTH (fail-soft, matches the Redis pattern).
        try:
            rsettings = get_settings().redis
            if rsettings.iam_auth:
                region = (
                    rsettings.iam_region
                    or _region_from_cache_host(host)
                    or os.getenv("AWS_REGION")
                )
                username = rsettings.username or username  # CacheIamUserName
                cache_name = host.split(".")[0]
                password = _mint_elasticache_token(username, cache_name, region)
                logger.info("Redis IAM auth enabled (user=%s)", username)  # no token
        except Exception as e:
            logger.error("Redis IAM token mint failed; using static AUTH: %s", e)

        _async_client = aioredis.Redis(
            host=redis_cfg["host"],
            port=redis_cfg["port"],
            username=username,
            password=password,
            ssl=redis_cfg["ssl"],
            db=redis_cfg["db"],
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

    # Check legacy env var first, then typed settings
    host = os.getenv("REDIS_HOST")
    if not host:
        try:
            settings = get_settings()
            host = settings.redis.host
        except Exception:
            pass
    if not host:
        return None

    import redis

    _warn_no_password()

    redis_cfg = _redis_settings()
    username = redis_cfg.get("username")
    password = redis_cfg["password"]
    # IAM auth (ADR-0029): mirror the async splice. _mint_elasticache_token is
    # sync (local SigV4 presign), fine for the sync client. Default OFF -> static
    # path unchanged; mint failure falls through to the static AUTH (fail-soft).
    try:
        rsettings = get_settings().redis
        if rsettings.iam_auth:
            region = (
                rsettings.iam_region
                or _region_from_cache_host(host)
                or os.getenv("AWS_REGION")
            )
            username = rsettings.username or username  # CacheIamUserName
            cache_name = host.split(".")[0]
            password = _mint_elasticache_token(username, cache_name, region)
            logger.info("Redis IAM auth enabled (user=%s)", username)  # no token
    except Exception as e:
        logger.error("Redis IAM token mint failed; using static AUTH: %s", e)

    _sync_client = redis.Redis(
        host=redis_cfg["host"],
        port=redis_cfg["port"],
        username=username,
        password=password,
        ssl=redis_cfg["ssl"],
        db=redis_cfg["db"],
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
    global _async_client, _sync_client, _async_client_lock, _last_health_check
    _async_client = None
    _sync_client = None
    _async_client_lock = None
    _last_health_check = 0.0


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
    """Build a Redis URL from typed settings (falling back to env vars).

    Useful for consumers that accept a URL string rather than keyword
    arguments (e.g. ``redis.asyncio.from_url``).

    Returns:
        A ``redis://`` or ``rediss://`` URL string.
    """
    settings = _redis_settings()
    scheme = "rediss" if settings["ssl"] else "redis"
    # Static-path URL only. NOT IAM-aware: a SigV4 token in a URL would need
    # percent-quoting, so the IAM path presents username + token-as-password on
    # the client kwargs instead (see get_async/sync_redis_client). username is
    # spliced here when present (harmless for the static path: "user:pw@").
    user = settings.get("username") or ""
    pw = settings["password"] or ""
    if user or pw:
        auth = f"{user}:{pw}@"
    else:
        auth = ""
    return f"{scheme}://{auth}{settings['host']}:{settings['port']}/{settings['db']}"
