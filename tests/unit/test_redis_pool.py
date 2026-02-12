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

from unittest.mock import patch, MagicMock

import pytest

from litellm_llmrouter.redis_pool import (
    _redis_settings,
    build_redis_url,
    create_async_redis_client,
    create_sync_redis_client,
)


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
