"""
Unit tests for the service_discovery module.

Tests cover:
- URL sanitization
- Individual service probes (mocked)
- Aggregate probing
- Feature availability calculation
- Startup table formatting
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm_llmrouter.service_discovery import (
    ExternalServices,
    ServiceStatus,
    _sanitize_url,
    format_service_status_table,
    get_feature_availability,
    get_last_probe_results,
    probe_all_services,
    probe_otel_collector,
    probe_oidc_provider,
    probe_postgresql,
    probe_redis,
)


# =============================================================================
# URL Sanitization
# =============================================================================


class TestSanitizeUrl:
    def test_removes_password(self):
        url = "postgresql://user:secret@host:5432/db"
        result = _sanitize_url(url)
        assert "secret" not in result
        assert "***@host:5432" in result

    def test_removes_username_and_password(self):
        url = "redis://admin:password@redis-host:6379/0"
        result = _sanitize_url(url)
        assert "admin" not in result
        assert "password" not in result
        assert "***@redis-host:6379" in result

    def test_no_credentials_unchanged(self):
        url = "http://otel-collector:4317"
        result = _sanitize_url(url)
        assert result == url

    def test_invalid_url_returns_placeholder(self):
        result = _sanitize_url("")
        # Empty string is valid but has no credentials
        assert "***" not in result

    def test_password_only(self):
        url = "redis://:secretpass@host:6379"
        result = _sanitize_url(url)
        assert "secretpass" not in result
        assert "***@host:6379" in result


# =============================================================================
# ServiceStatus
# =============================================================================


class TestServiceStatus:
    def test_to_dict(self):
        status = ServiceStatus(
            name="postgresql",
            available=True,
            url="postgresql://***@host:5432/db",
            latency_ms=3.5,
            version="PostgreSQL 16.1",
        )
        d = status.to_dict()
        assert d["name"] == "postgresql"
        assert d["available"] is True
        assert d["latency_ms"] == 3.5
        assert d["version"] == "PostgreSQL 16.1"
        assert d["error"] is None

    def test_to_dict_with_error(self):
        status = ServiceStatus(
            name="redis",
            available=False,
            url="redis://host:6379",
            error="Connection refused",
        )
        d = status.to_dict()
        assert d["available"] is False
        assert d["error"] == "Connection refused"


# =============================================================================
# probe_postgresql
# =============================================================================


class TestProbePostgresql:
    @pytest.mark.asyncio
    async def test_not_configured(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        status = await probe_postgresql()
        assert status.available is False
        assert "Not configured" in status.error

    @pytest.mark.asyncio
    async def test_successful_probe(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host:5432/db")

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=("PostgreSQL 16.1 on x86_64",))

        mock_pool = AsyncMock()
        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "litellm_llmrouter.database.get_db_pool",
            new_callable=AsyncMock,
            return_value=mock_pool,
        ):
            status = await probe_postgresql()

        assert status.available is True
        assert "pass" not in status.url
        assert status.latency_ms is not None
        assert status.version is not None

    @pytest.mark.asyncio
    async def test_pool_returns_none(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host:5432/db")

        with patch(
            "litellm_llmrouter.database.get_db_pool",
            new_callable=AsyncMock,
            return_value=None,
        ):
            status = await probe_postgresql()
        assert status.available is False
        assert "Pool creation" in status.error


# =============================================================================
# probe_redis
# =============================================================================


class TestProbeRedis:
    @pytest.mark.asyncio
    async def test_not_configured(self, monkeypatch):
        monkeypatch.delenv("REDIS_HOST", raising=False)
        status = await probe_redis()
        assert status.available is False
        assert "Not configured" in status.error

    @pytest.mark.asyncio
    async def test_successful_probe(self, monkeypatch):
        monkeypatch.setenv("REDIS_HOST", "redis-host")
        monkeypatch.setenv("REDIS_PORT", "6379")

        mock_client = AsyncMock()
        mock_client.info = AsyncMock(return_value={"redis_version": "7.2.4"})

        with patch(
            "litellm_llmrouter.redis_pool.get_async_redis_client",
            new_callable=AsyncMock,
            return_value=mock_client,
        ):
            status = await probe_redis()

        assert status.available is True
        assert status.version == "7.2.4"
        assert "redis-host" in status.url

    @pytest.mark.asyncio
    async def test_client_returns_none(self, monkeypatch):
        monkeypatch.setenv("REDIS_HOST", "redis-host")

        with patch(
            "litellm_llmrouter.redis_pool.get_async_redis_client",
            new_callable=AsyncMock,
            return_value=None,
        ):
            status = await probe_redis()

        assert status.available is False
        assert "Client creation" in status.error


# =============================================================================
# probe_otel_collector
# =============================================================================


class TestProbeOtelCollector:
    @pytest.mark.asyncio
    async def test_not_configured(self, monkeypatch):
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        status = await probe_otel_collector()
        assert status.available is False
        assert "Not configured" in status.error

    @pytest.mark.asyncio
    async def test_successful_probe(self, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4317")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"server": "otel-collector/0.99.0"}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            status = await probe_otel_collector()

        assert status.available is True
        assert "collector:4317" in status.url


# =============================================================================
# probe_oidc_provider
# =============================================================================


class TestProbeOidcProvider:
    @pytest.mark.asyncio
    async def test_not_configured(self, monkeypatch):
        monkeypatch.delenv("ROUTEIQ_OIDC_ENABLED", raising=False)
        monkeypatch.delenv("ROUTEIQ_OIDC_ISSUER_URL", raising=False)
        status = await probe_oidc_provider()
        assert status.available is False
        assert "Not configured" in status.error

    @pytest.mark.asyncio
    async def test_disabled(self, monkeypatch):
        monkeypatch.setenv("ROUTEIQ_OIDC_ENABLED", "false")
        monkeypatch.setenv("ROUTEIQ_OIDC_ISSUER_URL", "https://auth.example.com")
        status = await probe_oidc_provider()
        assert status.available is False

    @pytest.mark.asyncio
    async def test_successful_probe(self, monkeypatch):
        monkeypatch.setenv("ROUTEIQ_OIDC_ENABLED", "true")
        monkeypatch.setenv("ROUTEIQ_OIDC_ISSUER_URL", "https://auth.example.com")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"issuer": "https://auth.example.com"}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            status = await probe_oidc_provider()

        assert status.available is True
        assert "auth.example.com" in status.url


# =============================================================================
# probe_all_services
# =============================================================================


class TestProbeAllServices:
    @pytest.mark.asyncio
    async def test_returns_all_services(self, monkeypatch):
        """All four service types should be present in the result."""
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("REDIS_HOST", raising=False)
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        monkeypatch.delenv("ROUTEIQ_OIDC_ENABLED", raising=False)
        monkeypatch.delenv("ROUTEIQ_OIDC_ISSUER_URL", raising=False)

        services = await probe_all_services()
        assert ExternalServices.POSTGRESQL.value in services
        assert ExternalServices.REDIS.value in services
        assert ExternalServices.OTEL_COLLECTOR.value in services
        assert ExternalServices.OIDC_PROVIDER.value in services

    @pytest.mark.asyncio
    async def test_caches_results(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("REDIS_HOST", raising=False)
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        monkeypatch.delenv("ROUTEIQ_OIDC_ENABLED", raising=False)
        monkeypatch.delenv("ROUTEIQ_OIDC_ISSUER_URL", raising=False)

        await probe_all_services()
        cached = get_last_probe_results()
        assert len(cached) == 4


# =============================================================================
# get_feature_availability
# =============================================================================


class TestGetFeatureAvailability:
    def test_all_available(self):
        services = {
            ExternalServices.POSTGRESQL.value: ServiceStatus(
                name="postgresql", available=True, url=""
            ),
            ExternalServices.REDIS.value: ServiceStatus(
                name="redis", available=True, url=""
            ),
            ExternalServices.OTEL_COLLECTOR.value: ServiceStatus(
                name="otel-collector", available=True, url=""
            ),
            ExternalServices.OIDC_PROVIDER.value: ServiceStatus(
                name="oidc-provider", available=True, url=""
            ),
        }
        features = get_feature_availability(services)
        assert features["persistence"] is True
        assert features["caching"] is True
        assert features["ha_leader_election"] is True
        assert features["observability"] is True
        assert features["sso_login"] is True

    def test_none_available(self):
        services = {
            ExternalServices.POSTGRESQL.value: ServiceStatus(
                name="postgresql", available=False, url=""
            ),
            ExternalServices.REDIS.value: ServiceStatus(
                name="redis", available=False, url=""
            ),
            ExternalServices.OTEL_COLLECTOR.value: ServiceStatus(
                name="otel-collector", available=False, url=""
            ),
            ExternalServices.OIDC_PROVIDER.value: ServiceStatus(
                name="oidc-provider", available=False, url=""
            ),
        }
        features = get_feature_availability(services)
        assert features["persistence"] is False
        assert features["caching"] is False
        assert features["ha_leader_election"] is False
        assert features["observability"] is False
        assert features["sso_login"] is False

    def test_centroid_routing_from_env(self, monkeypatch):
        monkeypatch.setenv("ROUTEIQ_CENTROID_ROUTING", "true")
        features = get_feature_availability({})
        assert features["centroid_routing"] is True

        monkeypatch.setenv("ROUTEIQ_CENTROID_ROUTING", "false")
        features = get_feature_availability({})
        assert features["centroid_routing"] is False


# =============================================================================
# format_service_status_table
# =============================================================================


class TestFormatServiceStatusTable:
    def test_contains_service_names(self):
        services = {
            ExternalServices.POSTGRESQL.value: ServiceStatus(
                name="postgresql",
                available=True,
                url="postgresql://***@host:5432/db",
                latency_ms=3.0,
            ),
            ExternalServices.REDIS.value: ServiceStatus(
                name="redis",
                available=False,
                url="",
                error="Not configured (REDIS_HOST not set)",
            ),
            ExternalServices.OTEL_COLLECTOR.value: ServiceStatus(
                name="otel-collector",
                available=False,
                url="",
                error="Not configured (OTEL_EXPORTER_OTLP_ENDPOINT not set)",
            ),
            ExternalServices.OIDC_PROVIDER.value: ServiceStatus(
                name="oidc-provider",
                available=False,
                url="",
                error="Not configured (ROUTEIQ_OIDC_ENABLED=false or ROUTEIQ_OIDC_ISSUER_URL not set)",
            ),
        }
        features = get_feature_availability(services)
        table = format_service_status_table(services, features)

        assert "PostgreSQL" in table
        assert "Redis" in table
        assert "OTel Collector" in table
        assert "OIDC Provider" in table
        assert "OK" in table
        assert "Not configured" in table
        assert "Features:" in table
        assert "persistence=yes" in table
        assert "caching=no" in table

    def test_latency_displayed(self):
        services = {
            ExternalServices.POSTGRESQL.value: ServiceStatus(
                name="postgresql",
                available=True,
                url="",
                latency_ms=5.2,
            ),
            ExternalServices.REDIS.value: ServiceStatus(
                name="redis",
                available=True,
                url="",
                latency_ms=1.0,
            ),
            ExternalServices.OTEL_COLLECTOR.value: ServiceStatus(
                name="otel-collector",
                available=False,
                url="",
                error="Not configured (OTEL_EXPORTER_OTLP_ENDPOINT not set)",
            ),
            ExternalServices.OIDC_PROVIDER.value: ServiceStatus(
                name="oidc-provider",
                available=False,
                url="",
                error="Not configured (ROUTEIQ_OIDC_ENABLED=false or ROUTEIQ_OIDC_ISSUER_URL not set)",
            ),
        }
        features = get_feature_availability(services)
        table = format_service_status_table(services, features)

        assert "5ms" in table
        assert "1ms" in table


# =============================================================================
# ExternalServices enum
# =============================================================================


class TestExternalServicesEnum:
    def test_values(self):
        assert ExternalServices.POSTGRESQL.value == "postgresql"
        assert ExternalServices.REDIS.value == "redis"
        assert ExternalServices.OTEL_COLLECTOR.value == "otel-collector"
        assert ExternalServices.OIDC_PROVIDER.value == "oidc-provider"

    def test_is_str(self):
        """ExternalServices extends str for JSON serialization."""
        assert isinstance(ExternalServices.POSTGRESQL, str)
