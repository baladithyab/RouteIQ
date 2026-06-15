"""
Unit Tests for Redis Pool Module
==================================

Tests cover:
1. ``_redis_settings()`` env-var parsing (host, port, password, ssl, db)
2. ``create_async_redis_client()`` produces correct kwargs
3. ``create_sync_redis_client()`` produces correct kwargs
4. ``build_redis_url()`` URL construction (with/without password, ssl)
5. Override behaviour (keyword args win over env vars)

Run tests:
    uv run pytest tests/unit/test_redis_pool.py -v
"""

from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from litellm_llmrouter.redis_pool import (
    IamRegionUnresolvedError,
    _redis_settings,
    build_redis_url,
    create_async_redis_client,
    create_sync_redis_client,
)
from litellm_llmrouter.settings import reset_settings

# Placeholders only -- NEVER a real token value (RouteIQ-d3a4 discipline).
_FAKE_SIGNED_TOKEN = "SIGNED-TOKEN"  # noqa: S105 - test placeholder, not a secret
_FAKE_CACHE_USER = "routeiq-cache-user"
_SERVERLESS_HOST = "routeiq-dev-cache-abc123.serverless.use1.cache.amazonaws.com"


# =============================================================================
# _redis_settings
# =============================================================================


class TestRedisSettings:
    """Tests for _redis_settings helper."""

    def test_defaults(self):
        """Default values when no env vars are set."""
        with patch.dict(
            "os.environ",
            {},
            clear=True,
        ):
            settings = _redis_settings()
        assert settings["host"] == "localhost"
        assert settings["port"] == 6379
        assert settings["password"] is None
        assert settings["ssl"] is False
        assert settings["db"] == 0

    def test_reads_all_env_vars(self):
        """All REDIS_* env vars are respected."""
        env = {
            "REDIS_HOST": "my-redis.example.com",
            "REDIS_PORT": "6380",
            "REDIS_PASSWORD": "s3cret",
            "REDIS_SSL": "true",
            "REDIS_DB": "3",
        }
        with patch.dict("os.environ", env, clear=True):
            settings = _redis_settings()
        assert settings["host"] == "my-redis.example.com"
        assert settings["port"] == 6380
        assert settings["password"] == "s3cret"
        assert settings["ssl"] is True
        assert settings["db"] == 3

    def test_ssl_variants(self):
        """SSL flag accepts multiple truthy strings."""
        for val in ("true", "1", "yes", "True", "YES"):
            with patch.dict("os.environ", {"REDIS_SSL": val}, clear=True):
                assert _redis_settings()["ssl"] is True

        for val in ("false", "0", "no", ""):
            with patch.dict("os.environ", {"REDIS_SSL": val}, clear=True):
                assert _redis_settings()["ssl"] is False

    def test_empty_password_is_none(self):
        """Empty REDIS_PASSWORD string is treated as None."""
        with patch.dict("os.environ", {"REDIS_PASSWORD": ""}, clear=True):
            assert _redis_settings()["password"] is None

    def test_redis_settings_includes_username(self):
        """REDIS_USERNAME is surfaced on the legacy path (present + None)."""
        with patch.dict(
            "os.environ",
            {"REDIS_HOST": "h", "REDIS_USERNAME": _FAKE_CACHE_USER},
            clear=True,
        ):
            assert _redis_settings()["username"] == _FAKE_CACHE_USER
        # Unset: the key is present and None (a host alone still triggers legacy).
        with patch.dict("os.environ", {"REDIS_HOST": "h"}, clear=True):
            cfg = _redis_settings()
            assert "username" in cfg
            assert cfg["username"] is None


# =============================================================================
# build_redis_url
# =============================================================================


class TestBuildRedisUrl:
    """Tests for build_redis_url helper."""

    def test_basic_url(self):
        env = {"REDIS_HOST": "redis-host", "REDIS_PORT": "6379"}
        with patch.dict("os.environ", env, clear=True):
            url = build_redis_url()
        assert url == "redis://redis-host:6379/0"

    def test_url_with_password(self):
        env = {
            "REDIS_HOST": "redis-host",
            "REDIS_PORT": "6379",
            "REDIS_PASSWORD": "p@ss",
        }
        with patch.dict("os.environ", env, clear=True):
            url = build_redis_url()
        assert url == "redis://:p@ss@redis-host:6379/0"

    def test_url_with_ssl(self):
        env = {
            "REDIS_HOST": "redis-host",
            "REDIS_PORT": "6380",
            "REDIS_SSL": "true",
        }
        with patch.dict("os.environ", env, clear=True):
            url = build_redis_url()
        assert url == "rediss://redis-host:6380/0"

    def test_url_with_db(self):
        env = {
            "REDIS_HOST": "redis-host",
            "REDIS_PORT": "6379",
            "REDIS_DB": "5",
        }
        with patch.dict("os.environ", env, clear=True):
            url = build_redis_url()
        assert url == "redis://redis-host:6379/5"

    def test_url_full(self):
        env = {
            "REDIS_HOST": "prod.redis.io",
            "REDIS_PORT": "6380",
            "REDIS_PASSWORD": "secret",
            "REDIS_SSL": "1",
            "REDIS_DB": "2",
        }
        with patch.dict("os.environ", env, clear=True):
            url = build_redis_url()
        assert url == "rediss://:secret@prod.redis.io:6380/2"


# =============================================================================
# create_async_redis_client
# =============================================================================


class TestCreateAsyncRedisClient:
    """Tests for create_async_redis_client."""

    def test_creates_client_with_env_vars(self):
        """Client is created with settings derived from env vars."""
        env = {
            "REDIS_HOST": "async-host",
            "REDIS_PORT": "6380",
            "REDIS_PASSWORD": "async-pass",
            "REDIS_SSL": "true",
            "REDIS_DB": "1",
        }
        with patch.dict("os.environ", env, clear=True):
            with patch("redis.asyncio.Redis") as mock_cls:
                mock_cls.return_value = MagicMock()
                create_async_redis_client()

                mock_cls.assert_called_once_with(
                    host="async-host",
                    port=6380,
                    password="async-pass",
                    ssl=True,
                    db=1,
                    username=None,
                    decode_responses=True,
                    socket_connect_timeout=2.0,
                    socket_timeout=2.0,
                )

    def test_overrides_take_precedence(self):
        """Keyword overrides win over env-var-derived settings."""
        env = {
            "REDIS_HOST": "env-host",
            "REDIS_PORT": "6379",
        }
        with patch.dict("os.environ", env, clear=True):
            with patch("redis.asyncio.Redis") as mock_cls:
                mock_cls.return_value = MagicMock()
                create_async_redis_client(host="override-host", port=9999)

                call_kwargs = mock_cls.call_args.kwargs
                assert call_kwargs["host"] == "override-host"
                assert call_kwargs["port"] == 9999

    def test_custom_timeouts(self):
        """Custom timeout values are passed through."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("redis.asyncio.Redis") as mock_cls:
                mock_cls.return_value = MagicMock()
                create_async_redis_client(
                    socket_connect_timeout=10.0,
                    socket_timeout=10.0,
                )
                call_kwargs = mock_cls.call_args.kwargs
                assert call_kwargs["socket_connect_timeout"] == 10.0
                assert call_kwargs["socket_timeout"] == 10.0


# =============================================================================
# create_sync_redis_client
# =============================================================================


class TestCreateSyncRedisClient:
    """Tests for create_sync_redis_client."""

    def test_creates_client_with_env_vars(self):
        """Client is created with settings derived from env vars."""
        env = {
            "REDIS_HOST": "sync-host",
            "REDIS_PORT": "6381",
            "REDIS_PASSWORD": "sync-pass",
            "REDIS_SSL": "yes",
            "REDIS_DB": "2",
        }
        with patch.dict("os.environ", env, clear=True):
            with patch("redis.Redis") as mock_cls:
                mock_cls.return_value = MagicMock()
                create_sync_redis_client()

                mock_cls.assert_called_once_with(
                    host="sync-host",
                    port=6381,
                    password="sync-pass",
                    ssl=True,
                    db=2,
                    username=None,
                    decode_responses=True,
                    socket_timeout=5.0,
                )

    def test_overrides_take_precedence(self):
        """Keyword overrides win over env-var-derived settings."""
        env = {"REDIS_HOST": "env-host"}
        with patch.dict("os.environ", env, clear=True):
            with patch("redis.Redis") as mock_cls:
                mock_cls.return_value = MagicMock()
                create_sync_redis_client(host="override-host")

                call_kwargs = mock_cls.call_args.kwargs
                assert call_kwargs["host"] == "override-host"


# =============================================================================
# Health check AuthenticationError handling
# =============================================================================


class TestHealthCheckRedisAuth:
    """Tests that the health endpoint correctly catches AuthenticationError."""

    @pytest.fixture
    def mock_drain_manager(self):
        mgr = MagicMock()
        mgr.is_draining = False
        mgr.active_requests = 0
        return mgr

    @pytest.fixture
    def mock_cb_manager(self):
        mgr = MagicMock()
        mgr.get_status.return_value = {
            "is_degraded": False,
            "degraded_components": [],
            "breakers": {},
        }
        redis_breaker = MagicMock()
        redis_breaker.is_open = False
        redis_breaker.record_failure = MagicMock(
            side_effect=lambda *a, **kw: MagicMock(__await__=lambda s: iter([None]))
        )

        # Make record_failure awaitable
        async def record_failure(msg):
            pass

        redis_breaker.record_failure = record_failure
        mgr.redis = redis_breaker
        return mgr

    async def test_auth_error_reports_authentication_failed(
        self, mock_drain_manager, mock_cb_manager
    ):
        """When Redis raises AuthenticationError, health reports
        'authentication_failed' instead of generic 'connection failed'."""
        import redis.exceptions
        from fastapi import HTTPException

        from litellm_llmrouter.routes.health import readiness_probe

        async def mock_ping():
            raise redis.exceptions.AuthenticationError("WRONGPASS")

        mock_client = MagicMock()
        mock_client.ping = mock_ping
        mock_client.aclose = MagicMock(
            side_effect=lambda: MagicMock(__await__=lambda s: iter([None]))
        )

        with (
            patch.dict("os.environ", {"REDIS_HOST": "redis-host"}, clear=False),
            patch(
                "litellm_llmrouter.routes.health.get_drain_manager",
                return_value=mock_drain_manager,
            ),
            patch(
                "litellm_llmrouter.routes.health.get_circuit_breaker_manager",
                return_value=mock_cb_manager,
            ),
            patch(
                "litellm_llmrouter.routes.health.get_request_id",
                return_value="test-req",
            ),
            patch(
                "litellm_llmrouter.redis_pool.create_async_redis_client",
                return_value=mock_client,
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await readiness_probe()

            detail = exc_info.value.detail
            assert detail["checks"]["redis"]["error"] == "authentication_failed"
            assert detail["checks"]["redis"]["status"] == "unhealthy"

    async def test_generic_error_reports_connection_failed(
        self, mock_drain_manager, mock_cb_manager
    ):
        """Non-auth errors still report generic 'connection failed'."""
        from fastapi import HTTPException

        from litellm_llmrouter.routes.health import readiness_probe

        async def mock_ping():
            raise ConnectionError("Connection refused")

        mock_client = MagicMock()
        mock_client.ping = mock_ping
        mock_client.aclose = MagicMock(
            side_effect=lambda: MagicMock(__await__=lambda s: iter([None]))
        )

        with (
            patch.dict("os.environ", {"REDIS_HOST": "redis-host"}, clear=False),
            patch(
                "litellm_llmrouter.routes.health.get_drain_manager",
                return_value=mock_drain_manager,
            ),
            patch(
                "litellm_llmrouter.routes.health.get_circuit_breaker_manager",
                return_value=mock_cb_manager,
            ),
            patch(
                "litellm_llmrouter.routes.health.get_request_id",
                return_value="test-req",
            ),
            patch(
                "litellm_llmrouter.redis_pool.create_async_redis_client",
                return_value=mock_client,
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await readiness_probe()

            detail = exc_info.value.detail
            assert detail["checks"]["redis"]["error"] == "connection failed"
            assert detail["checks"]["redis"]["status"] == "unhealthy"


# =============================================================================
# IAM-auth splice (ADR-0029) -- elasticache:Connect SigV4 token + REDIS_USERNAME
# =============================================================================
#
# boto3/botocore are NOT imported here: the mint helper (_mint_elasticache_token)
# is monkeypatch-stubbed so the splice assertions are fully cred-free. redis /
# redis.asyncio ARE importable in the dev env, so the client classes are patched
# directly (mirrors the existing factory tests). reset_redis_clients +
# reset_settings run via the global autouse conftest fixture.


class TestRedisIamAuthSplice:
    """The IAM-auth path presents REDIS_USERNAME + a SigV4 token as the AUTH."""

    async def test_async_client_iam_auth_splice(self, monkeypatch):
        """get_async_redis_client mints + presents username + token-as-password."""
        import litellm_llmrouter.redis_pool as rp

        seen = {}

        def _fake_mint(user, cache_name, region):
            seen["user"] = user
            seen["cache_name"] = cache_name
            seen["region"] = region
            return _FAKE_SIGNED_TOKEN

        monkeypatch.setattr(rp, "_mint_elasticache_token", _fake_mint)
        env = {
            "REDIS_HOST": _SERVERLESS_HOST,
            "REDIS_SSL": "true",
            "REDIS_USERNAME": _FAKE_CACHE_USER,
            "ROUTEIQ_REDIS_IAM_AUTH": "true",
            "AWS_REGION": "us-east-1",
        }
        with patch.dict("os.environ", env, clear=True):
            reset_settings()  # rebuild settings under this env (see DB test note)
            with patch("redis.asyncio.Redis") as mock_cls:
                mock_cls.return_value = MagicMock()
                await rp.get_async_redis_client()

                kwargs = mock_cls.call_args.kwargs
                assert kwargs["username"] == _FAKE_CACHE_USER
                assert kwargs["password"] == _FAKE_SIGNED_TOKEN
                # The token is NOT a static REDIS_PASSWORD (none was set).
                assert kwargs["ssl"] is True
        assert seen["user"] == _FAKE_CACHE_USER
        # cache_name is the leftmost host label.
        assert seen["cache_name"] == _SERVERLESS_HOST.split(".")[0]
        assert seen["region"] == "us-east-1"

    def test_sync_client_iam_auth_splice(self, monkeypatch):
        """get_sync_redis_client mints + presents username + token-as-password."""
        import litellm_llmrouter.redis_pool as rp

        monkeypatch.setattr(
            rp,
            "_mint_elasticache_token",
            lambda user, cache_name, region: _FAKE_SIGNED_TOKEN,
        )
        env = {
            "REDIS_HOST": _SERVERLESS_HOST,
            "REDIS_SSL": "true",
            "REDIS_USERNAME": _FAKE_CACHE_USER,
            "ROUTEIQ_REDIS_IAM_AUTH": "true",
            "AWS_REGION": "us-east-1",
        }
        with patch.dict("os.environ", env, clear=True):
            reset_settings()
            with patch("redis.Redis") as mock_cls:
                mock_cls.return_value = MagicMock()
                rp.get_sync_redis_client()

                kwargs = mock_cls.call_args.kwargs
                assert kwargs["username"] == _FAKE_CACHE_USER
                assert kwargs["password"] == _FAKE_SIGNED_TOKEN

    async def test_async_client_static_path_no_token_when_flag_off(self, monkeypatch):
        """Flag OFF -> no mint; static password + username from REDIS_USERNAME."""
        import litellm_llmrouter.redis_pool as rp

        called = {"mint": False}

        def _should_not_be_called(*a, **kw):
            called["mint"] = True
            return _FAKE_SIGNED_TOKEN

        monkeypatch.setattr(rp, "_mint_elasticache_token", _should_not_be_called)
        env = {
            "REDIS_HOST": "static-host",
            "REDIS_PASSWORD": "static-pass",
            "REDIS_SSL": "true",
            # ROUTEIQ_REDIS_IAM_AUTH unset -> default OFF
        }
        with patch.dict("os.environ", env, clear=True):
            reset_settings()
            with patch("redis.asyncio.Redis") as mock_cls:
                mock_cls.return_value = MagicMock()
                await rp.get_async_redis_client()

                kwargs = mock_cls.call_args.kwargs
                assert kwargs["password"] == "static-pass"
                assert kwargs["username"] is None
        assert called["mint"] is False

    async def test_iam_mint_failure_falls_back_to_static(self, monkeypatch):
        """A mint error (e.g. no creds) degrades softly to the static AUTH."""
        import litellm_llmrouter.redis_pool as rp

        def _boom(*a, **kw):
            raise RuntimeError("no AWS credentials for elasticache:Connect token mint")

        monkeypatch.setattr(rp, "_mint_elasticache_token", _boom)
        env = {
            "REDIS_HOST": _SERVERLESS_HOST,
            "REDIS_PASSWORD": "static-fallback-pass",
            "REDIS_SSL": "true",
            "REDIS_USERNAME": _FAKE_CACHE_USER,
            "ROUTEIQ_REDIS_IAM_AUTH": "true",
            "AWS_REGION": "us-east-1",
        }
        with patch.dict("os.environ", env, clear=True):
            reset_settings()
            with patch("redis.asyncio.Redis") as mock_cls:
                mock_cls.return_value = MagicMock()
                client = await rp.get_async_redis_client()

                assert client is not None  # built despite mint failure
                kwargs = mock_cls.call_args.kwargs
                # Fell back to the static REDIS_PASSWORD, not the token.
                assert kwargs["password"] == "static-fallback-pass"
                assert kwargs["password"] != _FAKE_SIGNED_TOKEN

    def test_warn_no_password_suppressed_on_iam_path(self, monkeypatch, caplog):
        """On the IAM path a missing REDIS_PASSWORD emits NO insecure-config warning."""
        import logging

        import litellm_llmrouter.redis_pool as rp

        env = {
            "REDIS_HOST": _SERVERLESS_HOST,
            "ROUTEIQ_REDIS_IAM_AUTH": "true",
            # no REDIS_PASSWORD
        }
        with patch.dict("os.environ", env, clear=True):
            reset_settings()
            with caplog.at_level(logging.WARNING, logger=rp.logger.name):
                rp._warn_no_password()
        warnings_emitted = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert not warnings_emitted, [r.getMessage() for r in warnings_emitted]


# =============================================================================
# RouteIQ-f5c4 -- IAM token refresh-before-expiry (age-gated re-mint)
# =============================================================================
#
# The ElastiCache IAM SigV4 token (expires=900) is baked into the long-lived
# singleton at build time. Without an age-gate a healthy connection ages the
# token past 15 min and new pool connections AUTH with an expired token. These
# tests mock the signer (count mints, NEVER a real token) and inject a fake
# monotonic clock so staleness is fully deterministic -- no real time/sleep.

_IAM_ENV = {
    "REDIS_HOST": _SERVERLESS_HOST,
    "REDIS_SSL": "true",
    "REDIS_USERNAME": _FAKE_CACHE_USER,
    "ROUTEIQ_REDIS_IAM_AUTH": "true",
    "AWS_REGION": "us-east-1",
}


class TestRedisIamTokenRefresh:
    """RouteIQ-f5c4: an aged (>~15min) IAM token forces a rebuild + re-mint."""

    async def test_async_iam_token_remint_after_expiry(self):
        """A clock advance past (TTL - REFRESH_BEFORE) re-mints on the async path."""
        import litellm_llmrouter.redis_pool as rp

        clock = {"t": 1000.0}
        mints: list[float] = []

        def _fake_mint(user, cache_name, region):
            mints.append(clock["t"])
            return f"{_FAKE_SIGNED_TOKEN}-{len(mints)}"  # never a real token

        with patch.dict("os.environ", _IAM_ENV, clear=True):
            reset_settings()
            rp.reset_redis_clients()  # cold; restores real clock, then re-inject
            rp._set_clock(lambda: clock["t"])
            with patch.object(rp, "_mint_elasticache_token", _fake_mint):
                with patch("redis.asyncio.Redis") as mock_cls:
                    # ping() must SUCCEED so the ONLY rebuild trigger is token age.
                    inst = MagicMock()
                    inst.ping = AsyncMock(return_value=True)
                    mock_cls.return_value = inst

                    await rp.get_async_redis_client()  # mint #1
                    assert len(mints) == 1

                    clock["t"] += (rp._IAM_TOKEN_TTL - rp._IAM_REFRESH_BEFORE) + 1.0
                    await rp.get_async_redis_client()  # stale -> rebuild -> re-mint #2
                    assert len(mints) == 2, "aged IAM token must trigger a re-mint"

    async def test_async_iam_token_not_reminted_within_ttl(self):
        """A small advance (< threshold) does NOT re-mint (no over-eager rebuild)."""
        import litellm_llmrouter.redis_pool as rp

        clock = {"t": 5000.0}
        mints: list[float] = []

        def _fake_mint(user, cache_name, region):
            mints.append(clock["t"])
            return f"{_FAKE_SIGNED_TOKEN}-{len(mints)}"

        with patch.dict("os.environ", _IAM_ENV, clear=True):
            reset_settings()
            rp.reset_redis_clients()
            rp._set_clock(lambda: clock["t"])
            with patch.object(rp, "_mint_elasticache_token", _fake_mint):
                with patch("redis.asyncio.Redis") as mock_cls:
                    inst = MagicMock()
                    inst.ping = AsyncMock(return_value=True)
                    mock_cls.return_value = inst

                    await rp.get_async_redis_client()  # mint #1
                    assert len(mints) == 1

                    # Within the health-check gate AND well under the refresh
                    # threshold -> cached client returned, no new mint.
                    clock["t"] += 10.0
                    await rp.get_async_redis_client()
                    assert len(mints) == 1, "fresh IAM token must NOT be re-minted"

    def test_sync_iam_token_remint_after_expiry(self):
        """A clock advance past threshold re-mints on the sync path (no health-check)."""
        import litellm_llmrouter.redis_pool as rp

        clock = {"t": 2000.0}
        mints: list[float] = []

        def _fake_mint(user, cache_name, region):
            mints.append(clock["t"])
            return f"{_FAKE_SIGNED_TOKEN}-{len(mints)}"

        with patch.dict("os.environ", _IAM_ENV, clear=True):
            reset_settings()
            rp.reset_redis_clients()
            rp._set_clock(lambda: clock["t"])
            with patch.object(rp, "_mint_elasticache_token", _fake_mint):
                with patch("redis.Redis") as mock_cls:
                    mock_cls.return_value = MagicMock()

                    rp.get_sync_redis_client()  # mint #1
                    assert len(mints) == 1

                    clock["t"] += (rp._IAM_TOKEN_TTL - rp._IAM_REFRESH_BEFORE) + 1.0
                    rp.get_sync_redis_client()  # stale -> rebuild -> re-mint #2
                    assert len(mints) == 2, "aged sync IAM token must trigger a re-mint"

    async def test_token_value_never_logged_on_remint(self, caplog):
        """The token VALUE is never logged across mint + re-mint (only user=...)."""
        import logging

        import litellm_llmrouter.redis_pool as rp

        clock = {"t": 3000.0}
        secret_token = f"{_FAKE_SIGNED_TOKEN}-MUST-NOT-LEAK"

        with patch.dict("os.environ", _IAM_ENV, clear=True):
            reset_settings()
            rp.reset_redis_clients()
            rp._set_clock(lambda: clock["t"])
            with patch.object(
                rp, "_mint_elasticache_token", lambda u, c, r: secret_token
            ):
                with patch("redis.asyncio.Redis") as mock_cls:
                    inst = MagicMock()
                    inst.ping = AsyncMock(return_value=True)
                    mock_cls.return_value = inst
                    with caplog.at_level(logging.DEBUG, logger=rp.logger.name):
                        await rp.get_async_redis_client()
                        clock["t"] += (rp._IAM_TOKEN_TTL - rp._IAM_REFRESH_BEFORE) + 1.0
                        await rp.get_async_redis_client()

        for record in caplog.records:
            assert secret_token not in record.getMessage(), "IAM token leaked to logs"


# =============================================================================
# RouteIQ-89a6 / RouteIQ-6829 -- region resolution + fail-loud on unresolved
# =============================================================================
#
# A serverless ElastiCache endpoint carries only the SHORTENED region token
# (use1), which _region_from_cache_host deliberately refuses. When neither
# ROUTEIQ_REDIS__IAM_REGION nor AWS_REGION is set the region cannot be resolved,
# and signing the SigV4 token with region=None yields an invalid token that
# fails AUTH with an opaque error. The mint paths must FAIL LOUD instead.

# Serverless host with NO usable region label (only the use1 short token).
_NO_REGION_ENV = {
    "REDIS_HOST": _SERVERLESS_HOST,
    "REDIS_SSL": "true",
    "REDIS_USERNAME": _FAKE_CACHE_USER,
    "ROUTEIQ_REDIS_IAM_AUTH": "true",
    # NEITHER ROUTEIQ_REDIS__IAM_REGION NOR AWS_REGION -> unresolvable.
}


class TestResolveIamRegion:
    """RouteIQ-89a6/6829: region resolution order + fail-loud on unresolved."""

    def test_unresolved_region_raises_fail_loud(self):
        """No iam_region, no parseable host region, no AWS_REGION -> raise."""
        import litellm_llmrouter.redis_pool as rp

        with patch.dict("os.environ", _NO_REGION_ENV, clear=True):
            reset_settings()
            with pytest.raises(IamRegionUnresolvedError) as exc_info:
                rp._resolve_iam_region(_SERVERLESS_HOST)
        # The error names the missing env so the misconfig is actionable.
        msg = str(exc_info.value)
        assert "ROUTEIQ_REDIS__IAM_REGION" in msg
        assert "AWS_REGION" in msg

    def test_resolves_from_aws_region_env(self):
        """AWS_REGION supplies the region when host/settings cannot."""
        import litellm_llmrouter.redis_pool as rp

        env = {**_NO_REGION_ENV, "AWS_REGION": "ap-southeast-2"}
        with patch.dict("os.environ", env, clear=True):
            reset_settings()
            assert rp._resolve_iam_region(_SERVERLESS_HOST) == "ap-southeast-2"

    def test_resolves_from_settings_iam_region(self):
        """ROUTEIQ_REDIS__IAM_REGION takes precedence over host + AWS_REGION."""
        import litellm_llmrouter.redis_pool as rp

        env = {
            **_NO_REGION_ENV,
            "ROUTEIQ_REDIS__IAM_REGION": "eu-west-1",
            "AWS_REGION": "us-east-1",  # lower precedence
        }
        with patch.dict("os.environ", env, clear=True):
            reset_settings()
            assert rp._resolve_iam_region(_SERVERLESS_HOST) == "eu-west-1"

    def test_resolves_from_provisioned_host_label(self):
        """A provisioned cluster host carrying a long-form region label resolves
        even without iam_region / AWS_REGION (RouteIQ-d3a4 strict regex)."""
        import litellm_llmrouter.redis_pool as rp

        provisioned = "routeiq-cache.abc123.ng.0001.use1.cache.amazonaws.com"
        host_with_region = "routeiq-cache.abc.us-east-1.cache.amazonaws.com"
        with patch.dict("os.environ", _NO_REGION_ENV, clear=True):
            reset_settings()
            # The serverless-style host has no long-form label -> not resolvable.
            assert rp._region_from_cache_host(provisioned) is None
            # A host carrying the canonical region label IS resolved from the host.
            assert rp._resolve_iam_region(host_with_region) == "us-east-1"


class TestRedisIamFailLoudOnUnresolvedRegion:
    """The async + sync mint paths fail loud (do NOT sign region=None)."""

    async def test_async_unresolved_region_fails_loud(self, monkeypatch):
        """get_async_redis_client raises rather than minting with region=None."""
        import litellm_llmrouter.redis_pool as rp

        # The mint must NEVER be reached (no region=None sign attempt).
        def _should_not_mint(*a, **kw):
            raise AssertionError("mint reached despite unresolved region")

        monkeypatch.setattr(rp, "_mint_elasticache_token", _should_not_mint)
        with patch.dict("os.environ", _NO_REGION_ENV, clear=True):
            reset_settings()
            with patch("redis.asyncio.Redis"):
                with pytest.raises(IamRegionUnresolvedError):
                    await rp.get_async_redis_client()

    def test_sync_unresolved_region_fails_loud(self, monkeypatch):
        """get_sync_redis_client raises rather than minting with region=None."""
        import litellm_llmrouter.redis_pool as rp

        def _should_not_mint(*a, **kw):
            raise AssertionError("mint reached despite unresolved region")

        monkeypatch.setattr(rp, "_mint_elasticache_token", _should_not_mint)
        with patch.dict("os.environ", _NO_REGION_ENV, clear=True):
            reset_settings()
            with patch("redis.Redis"):
                with pytest.raises(IamRegionUnresolvedError):
                    rp.get_sync_redis_client()

    async def test_resolved_region_mints_ok(self, monkeypatch):
        """With a resolvable region, the mint runs and the token is presented."""
        import litellm_llmrouter.redis_pool as rp

        seen = {}

        def _fake_mint(user, cache_name, region):
            seen["region"] = region
            return _FAKE_SIGNED_TOKEN

        monkeypatch.setattr(rp, "_mint_elasticache_token", _fake_mint)
        env = {**_NO_REGION_ENV, "AWS_REGION": "us-east-1"}
        with patch.dict("os.environ", env, clear=True):
            reset_settings()
            with patch("redis.asyncio.Redis") as mock_cls:
                mock_cls.return_value = MagicMock()
                await rp.get_async_redis_client()
                kwargs = mock_cls.call_args.kwargs
                assert kwargs["password"] == _FAKE_SIGNED_TOKEN
        assert seen["region"] == "us-east-1"


class TestIamTokenTtlSingleSource:
    """RouteIQ-d3fb: the 900s TTL is a single source-of-truth constant referenced
    by BOTH the SigV4 presign (_mint_elasticache_token expires=...) and the
    age-gated refresh stamp (_iam_token_is_stale)."""

    def test_ttl_constant_value(self):
        """The single-source constant is 900 seconds; the float alias matches."""
        import litellm_llmrouter.redis_pool as rp

        assert rp._IAM_TOKEN_TTL_SECONDS == 900
        assert rp._IAM_TOKEN_TTL == float(rp._IAM_TOKEN_TTL_SECONDS)

    def test_mint_signs_with_ttl_constant(self, monkeypatch):
        """_mint_elasticache_token passes _IAM_TOKEN_TTL_SECONDS as expires=..,
        proving the presign reads the single-source constant (not a literal)."""
        import litellm_llmrouter.redis_pool as rp

        captured = {}

        class _FakeSigner:
            def __init__(self, creds, service, region, expires):
                captured["service"] = service
                captured["region"] = region
                captured["expires"] = expires

            def add_auth(self, req):
                return None

        class _FakeCreds:
            pass

        class _FakeSession:
            def get_credentials(self):
                return _FakeCreds()

        fake_boto3 = MagicMock()
        fake_boto3.session.Session.return_value = _FakeSession()
        fake_auth = MagicMock()
        fake_auth.SigV4QueryAuth = _FakeSigner

        class _FakePrepared:
            url = "https://cache.example/signed"

        class _FakeReq:
            def prepare(self):
                return _FakePrepared()

        fake_awsrequest = MagicMock()
        fake_awsrequest.AWSRequest.return_value = _FakeReq()

        import sys

        with patch.dict(
            sys.modules,
            {
                "boto3": fake_boto3,
                "botocore.auth": fake_auth,
                "botocore.awsrequest": fake_awsrequest,
            },
        ):
            token = rp._mint_elasticache_token("u", "cache.example", "us-east-1")

        assert token == "cache.example/signed"  # https:// stripped
        # The presign TTL is the single-source constant, not a hard-coded 900.
        assert captured["expires"] == rp._IAM_TOKEN_TTL_SECONDS
        assert captured["expires"] == 900
        assert captured["service"] == "elasticache"

    def test_stale_gate_uses_ttl_constant(self):
        """_iam_token_is_stale's threshold derives from the same TTL constant:
        an advance of exactly (TTL - REFRESH_BEFORE) is NOT yet stale, +1 IS."""
        import litellm_llmrouter.redis_pool as rp

        env = {**_NO_REGION_ENV, "AWS_REGION": "us-east-1"}
        with patch.dict("os.environ", env, clear=True):
            reset_settings()
            rp.reset_redis_clients()
            clock = {"t": 1000.0}
            rp._set_clock(lambda: clock["t"])
            # Simulate a mint at t=1000.
            rp._iam_token_minted_at = clock["t"]
            # Exactly at the threshold boundary -> NOT stale (strict >).
            clock["t"] = 1000.0 + (rp._IAM_TOKEN_TTL - rp._IAM_REFRESH_BEFORE)
            assert rp._iam_token_is_stale() is False
            # One second past the threshold -> stale.
            clock["t"] += 1.0
            assert rp._iam_token_is_stale() is True
