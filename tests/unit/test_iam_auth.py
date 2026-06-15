"""
Unit Tests for IAM-Token Runtime Auth (RouteIQ-d3a4)
=====================================================

Covers the in-process IAM-token minting wired behind two DEFAULT-OFF flags:

- ``ROUTEIQ_DB_IAM_AUTH``  -> ``database.get_db_pool`` mints a 15-min
  ``rds-db:connect`` token (ADR-0028) and passes it to asyncpg as a CALLABLE
  password (refresh-per-connection).
- ``ROUTEIQ_REDIS_IAM_AUTH`` -> ``redis_pool`` presents ``REDIS_USERNAME`` +
  a short-lived ``elasticache:Connect`` SigV4 token as the AUTH (ADR-0029).

All tests are CRED-FREE: boto3 is mocked via ``sys.modules`` and the
module-level mint helpers (``_mint_db_token`` / ``_mint_elasticache_token``)
are ``monkeypatch``-stubbed so botocore signing internals never run. NEVER
assert on a token value appearing in logs.

Token placeholders only: ``TOKEN-15MIN`` (DB), ``SIGNED-TOKEN`` (Redis),
``routeiq-cache-user`` (IAM user). No real secrets.

The autouse ``_reset_all_singletons`` fixture (tests/unit/conftest.py) already
resets the DB pool, Redis clients, and settings between tests, so no new reset
obligation is introduced (the minted token lives inside the connection/client,
never in a module-level cache).
"""

from __future__ import annotations

import logging
import sys
from unittest.mock import MagicMock, patch

import pytest

import litellm_llmrouter.database as db
import litellm_llmrouter.redis_pool as rp
from litellm_llmrouter.settings import reset_settings


# A password-less RDS/Aurora DSN (the IAM path supplies the token as the
# callable password; the URL itself carries no password).
_DB_URL = (
    "postgresql://routeiq@db.cluster-x.us-east-1.rds.amazonaws.com:5432/"
    "litellm?sslmode=require"
)
# A serverless ElastiCache endpoint (shortened region token `use1`, so the
# host parse cannot recover the long-form region -> AWS_REGION is required).
_CACHE_HOST = "routeiq-cache-sl.abcd.serverless.use1.cache.amazonaws.com"


@pytest.fixture(autouse=True)
def _clean_iam_env(monkeypatch):
    """Strip any inherited IAM/DB/Redis env so each test starts from a clean
    slate; reset the settings + pool/client singletons before and after."""
    for key in (
        "DATABASE_URL",
        "ROUTEIQ_DB_IAM_AUTH",
        "ROUTEIQ_REDIS_IAM_AUTH",
        "REDIS_HOST",
        "REDIS_PORT",
        "REDIS_PASSWORD",
        "REDIS_SSL",
        "REDIS_DB",
        "REDIS_USERNAME",
        "AWS_REGION",
    ):
        monkeypatch.delenv(key, raising=False)
    reset_settings()
    db.reset_db_pool()
    rp.reset_redis_clients()
    yield
    reset_settings()
    db.reset_db_pool()
    rp.reset_redis_clients()


def _mock_asyncpg(capture: dict) -> MagicMock:
    """A mock ``asyncpg`` module whose ``create_pool`` records its args."""
    mod = MagicMock()

    async def _create_pool(url, **kwargs):
        capture["url"] = url
        capture["kwargs"] = kwargs
        return MagicMock()

    mod.create_pool = _create_pool
    return mod


# =============================================================================
# Region parsers (pure, no AWS)
# =============================================================================


class TestRegionParsers:
    def test_region_from_rds_host(self):
        host = "db.cluster-x.us-east-1.rds.amazonaws.com"
        assert db._region_from_rds_host(host) == "us-east-1"

    def test_region_from_rds_host_non_rds(self):
        assert db._region_from_rds_host("localhost") is None
        assert db._region_from_rds_host(None) is None

    def test_region_from_cache_host_literal_region(self):
        host = "x.yyyy.us-east-1.cache.amazonaws.com"
        assert rp._region_from_cache_host(host) == "us-east-1"

    def test_region_from_cache_host_serverless_shortened(self):
        # Shortened serverless token (`use1`) must NOT be returned as a region.
        assert rp._region_from_cache_host(_CACHE_HOST) is None

    def test_region_from_cache_host_cache_name_not_mistaken_for_region(self):
        # A cache-name label with two hyphens (`routeiq-cache-sl`) must NOT be
        # mistaken for a region (RouteIQ-d3a4 strict-regex guard).
        host = "routeiq-cache-sl.abcd.serverless.use1.cache.amazonaws.com"
        assert rp._region_from_cache_host(host) is None

    def test_region_from_cache_host_non_cache(self):
        assert rp._region_from_cache_host("localhost") is None
        assert rp._region_from_cache_host(None) is None


# =============================================================================
# DB IAM token (ADR-0028)
# =============================================================================


class TestDbIamAuth:
    async def test_passes_callable_password_with_refresh(self, monkeypatch):
        """On the IAM path, asyncpg gets a CALLABLE password that re-mints on
        each call (15-min refresh proof) and the URL stays password-less."""
        monkeypatch.setenv("DATABASE_URL", _DB_URL)
        monkeypatch.setenv("ROUTEIQ_DB_IAM_AUTH", "true")
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        reset_settings()
        db.reset_db_pool()

        monkeypatch.setattr(db, "_mint_db_token", lambda **kw: "TOKEN-15MIN")
        capture: dict = {}
        with patch.dict(sys.modules, {"asyncpg": _mock_asyncpg(capture)}):
            await db.get_db_pool()

        pw = capture["kwargs"].get("password")
        assert callable(pw), "IAM path must pass a callable password to asyncpg"
        assert pw() == "TOKEN-15MIN"
        assert pw() == "TOKEN-15MIN"  # re-mints per call (reconnect refresh)
        # URL is unchanged / password-less (token never spliced into the DSN).
        assert capture["url"] == _DB_URL

    async def test_mint_called_with_correct_args(self, monkeypatch):
        """The callable mints with host/port/user/region parsed from the DSN."""
        monkeypatch.setenv("DATABASE_URL", _DB_URL)
        monkeypatch.setenv("ROUTEIQ_DB_IAM_AUTH", "true")
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        reset_settings()
        db.reset_db_pool()

        spy = MagicMock(return_value="TOKEN-15MIN")
        monkeypatch.setattr(db, "_mint_db_token", spy)
        capture: dict = {}
        with patch.dict(sys.modules, {"asyncpg": _mock_asyncpg(capture)}):
            await db.get_db_pool()
            # Invoke the callable so the mint actually fires.
            capture["kwargs"]["password"]()

        kwargs = spy.call_args.kwargs
        assert kwargs["host"].endswith(".rds.amazonaws.com")
        assert kwargs["user"] == "routeiq"
        assert kwargs["port"] == 5432
        assert kwargs["region"] == "us-east-1"

    async def test_static_path_no_callable_when_flag_off(self, monkeypatch):
        """Flag OFF (default): no password kwarg, mint never called -> the
        static-cred path is byte-for-byte unchanged."""
        monkeypatch.setenv("DATABASE_URL", _DB_URL)
        # ROUTEIQ_DB_IAM_AUTH intentionally unset.
        reset_settings()
        db.reset_db_pool()

        spy = MagicMock(return_value="TOKEN-15MIN")
        monkeypatch.setattr(db, "_mint_db_token", spy)
        capture: dict = {}
        with patch.dict(sys.modules, {"asyncpg": _mock_asyncpg(capture)}):
            await db.get_db_pool()

        assert "password" not in capture["kwargs"]
        spy.assert_not_called()

    async def test_setup_failure_falls_back_to_static(self, monkeypatch):
        """If the IAM setup itself raises (e.g. settings unavailable), the pool
        is still built with no password kwarg -- no crash at boot."""
        monkeypatch.setenv("DATABASE_URL", _DB_URL)
        monkeypatch.setenv("ROUTEIQ_DB_IAM_AUTH", "true")
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        reset_settings()
        db.reset_db_pool()

        # Make the IAM-branch settings read raise inside get_db_pool's try block.
        def _boom():
            raise RuntimeError("settings unavailable")

        monkeypatch.setattr(db, "get_settings", _boom, raising=False)
        # Patch the symbol the IAM branch imports (`get_settings as _gs`).
        import litellm_llmrouter.settings as settings_mod

        monkeypatch.setattr(settings_mod, "get_settings", _boom)

        capture: dict = {}
        with patch.dict(sys.modules, {"asyncpg": _mock_asyncpg(capture)}):
            pool = await db.get_db_pool()

        assert pool is not None
        assert "password" not in capture["kwargs"]


# =============================================================================
# Redis IAM token (ADR-0029)
# =============================================================================


class TestRedisIamAuth:
    def test_redis_settings_includes_username(self, monkeypatch):
        monkeypatch.setenv("REDIS_USERNAME", "routeiq-cache-user")
        assert rp._redis_settings()["username"] == "routeiq-cache-user"

    def test_redis_settings_username_none_when_unset(self, monkeypatch):
        monkeypatch.setenv("REDIS_HOST", "localhost")
        cfg = rp._redis_settings()
        assert "username" in cfg
        assert cfg["username"] is None

    async def test_async_client_iam_splice(self, monkeypatch):
        monkeypatch.setenv("REDIS_HOST", _CACHE_HOST)
        monkeypatch.setenv("REDIS_SSL", "true")
        monkeypatch.setenv("REDIS_USERNAME", "routeiq-cache-user")
        monkeypatch.setenv("ROUTEIQ_REDIS_IAM_AUTH", "true")
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        reset_settings()
        rp.reset_redis_clients()

        spy = MagicMock(return_value="SIGNED-TOKEN")
        monkeypatch.setattr(rp, "_mint_elasticache_token", spy)

        captured: dict = {}

        def _fake_redis(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        with patch("redis.asyncio.Redis", side_effect=_fake_redis):
            await rp.get_async_redis_client()

        assert captured["username"] == "routeiq-cache-user"
        assert captured["password"] == "SIGNED-TOKEN"  # token, not REDIS_PASSWORD
        assert captured["ssl"] is True
        # cache_name = leftmost host label; region = AWS_REGION (serverless host
        # cannot yield a long-form region from its shortened token).
        spy.assert_called_once_with(
            "routeiq-cache-user", "routeiq-cache-sl", "us-east-1"
        )

    def test_sync_client_iam_splice(self, monkeypatch):
        monkeypatch.setenv("REDIS_HOST", _CACHE_HOST)
        monkeypatch.setenv("REDIS_SSL", "true")
        monkeypatch.setenv("REDIS_USERNAME", "routeiq-cache-user")
        monkeypatch.setenv("ROUTEIQ_REDIS_IAM_AUTH", "true")
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        reset_settings()
        rp.reset_redis_clients()

        spy = MagicMock(return_value="SIGNED-TOKEN")
        monkeypatch.setattr(rp, "_mint_elasticache_token", spy)

        captured: dict = {}

        def _fake_redis(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        with patch("redis.Redis", side_effect=_fake_redis):
            rp.get_sync_redis_client()

        assert captured["username"] == "routeiq-cache-user"
        assert captured["password"] == "SIGNED-TOKEN"
        spy.assert_called_once_with(
            "routeiq-cache-user", "routeiq-cache-sl", "us-east-1"
        )

    async def test_static_path_no_token_when_flag_off(self, monkeypatch):
        """Flag OFF (default): static AUTH used, mint never called, username
        flows from REDIS_USERNAME only."""
        monkeypatch.setenv("REDIS_HOST", "localhost")
        monkeypatch.setenv("REDIS_PASSWORD", "static-pass")
        # ROUTEIQ_REDIS_IAM_AUTH intentionally unset.
        reset_settings()
        rp.reset_redis_clients()

        spy = MagicMock(return_value="SIGNED-TOKEN")
        monkeypatch.setattr(rp, "_mint_elasticache_token", spy)

        captured: dict = {}

        def _fake_redis(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        with patch("redis.asyncio.Redis", side_effect=_fake_redis):
            await rp.get_async_redis_client()

        assert captured["password"] == "static-pass"
        assert captured["username"] is None
        spy.assert_not_called()

    async def test_iam_mint_failure_falls_back_to_static(self, monkeypatch, caplog):
        """If the mint raises (e.g. get_credentials() returns None), the client
        is still built with the static AUTH (fail-soft) and an error is logged
        WITHOUT any token value."""
        monkeypatch.setenv("REDIS_HOST", _CACHE_HOST)
        monkeypatch.setenv("REDIS_SSL", "true")
        monkeypatch.setenv("REDIS_USERNAME", "routeiq-cache-user")
        monkeypatch.setenv("REDIS_PASSWORD", "static-pass")
        monkeypatch.setenv("ROUTEIQ_REDIS_IAM_AUTH", "true")
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        reset_settings()
        rp.reset_redis_clients()

        def _boom(*a, **kw):
            raise RuntimeError("no AWS credentials for elasticache:Connect token mint")

        monkeypatch.setattr(rp, "_mint_elasticache_token", _boom)

        captured: dict = {}

        def _fake_redis(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        with caplog.at_level(logging.ERROR, logger=rp.logger.name):
            with patch("redis.asyncio.Redis", side_effect=_fake_redis):
                client = await rp.get_async_redis_client()

        assert client is not None  # graceful degradation
        assert captured["password"] == "static-pass"  # static AUTH used
        assert any("IAM token mint failed" in r.message for r in caplog.records)
        # The error log must NOT leak a token value.
        for record in caplog.records:
            assert "SIGNED-TOKEN" not in record.getMessage()

    def test_warn_no_password_suppressed_on_iam_path(self, monkeypatch, caplog):
        """With IAM auth on and no REDIS_PASSWORD, _warn_no_password() must NOT
        emit the insecure-no-password warning (the token IS the AUTH)."""
        monkeypatch.setenv("REDIS_HOST", _CACHE_HOST)
        monkeypatch.setenv("ROUTEIQ_REDIS_IAM_AUTH", "true")
        monkeypatch.delenv("REDIS_PASSWORD", raising=False)
        reset_settings()

        with caplog.at_level(logging.WARNING, logger=rp.logger.name):
            rp._warn_no_password()

        assert not any("without password" in r.getMessage() for r in caplog.records), (
            "IAM path must suppress the no-password warning"
        )

    def test_warn_no_password_still_warns_on_static_path(self, monkeypatch, caplog):
        """Sanity: with IAM OFF and no password, the warning still fires (so the
        suppression is scoped to the IAM path only)."""
        monkeypatch.setenv("REDIS_HOST", "localhost")
        monkeypatch.delenv("REDIS_PASSWORD", raising=False)
        monkeypatch.delenv("ROUTEIQ_REDIS_IAM_AUTH", raising=False)
        reset_settings()

        with caplog.at_level(logging.WARNING, logger=rp.logger.name):
            rp._warn_no_password()

        assert any("without password" in r.getMessage() for r in caplog.records)


# =============================================================================
# Settings flag wiring (flat env -> nested field)
# =============================================================================


class TestIamSettingsFlags:
    def test_flat_db_flag_maps_to_nested_field(self, monkeypatch):
        monkeypatch.setenv("ROUTEIQ_DB_IAM_AUTH", "true")
        reset_settings()
        from litellm_llmrouter.settings import get_settings

        assert get_settings().postgres.iam_auth is True

    def test_flat_redis_flag_maps_to_nested_field(self, monkeypatch):
        monkeypatch.setenv("ROUTEIQ_REDIS_IAM_AUTH", "true")
        reset_settings()
        from litellm_llmrouter.settings import get_settings

        assert get_settings().redis.iam_auth is True

    def test_flags_default_off(self, monkeypatch):
        monkeypatch.delenv("ROUTEIQ_DB_IAM_AUTH", raising=False)
        monkeypatch.delenv("ROUTEIQ_REDIS_IAM_AUTH", raising=False)
        reset_settings()
        from litellm_llmrouter.settings import get_settings

        s = get_settings()
        assert s.postgres.iam_auth is False
        assert s.redis.iam_auth is False

    def test_flat_flag_does_not_clobber_other_nested_redis_fields(self, monkeypatch):
        """Mapping the flat flag must deep-merge, not replace, the redis dict."""
        monkeypatch.setenv("ROUTEIQ_REDIS_IAM_AUTH", "true")
        monkeypatch.setenv("ROUTEIQ_REDIS__HOST", "configured-host")
        reset_settings()
        from litellm_llmrouter.settings import get_settings

        s = get_settings()
        assert s.redis.iam_auth is True
        assert s.redis.host == "configured-host"
