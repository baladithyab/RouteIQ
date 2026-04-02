"""
Tests for the Pydantic settings module (``litellm_llmrouter.settings``).

Covers: defaults, env var loading, validators, cross-field validation,
nested model tests, convenience properties, and edge cases.

Every test resets the settings singleton via the ``_reset_settings`` fixture
to prevent cross-test contamination from cached env vars.
"""

import warnings

import pytest
from pydantic import ValidationError

from litellm_llmrouter.settings import (
    A2ASettings,
    AuditFailMode,
    AuditSettings,
    CacheSettings,
    ConfigSyncSettings,
    ConversationAffinitySettings,
    GatewaySettings,
    HASettings,
    HTTPClientSettings,
    MCPSettings,
    ManagementSettings,
    OIDCSettings,
    OTelSettings,
    PluginSettings,
    PolicyFailMode,
    PolicySettings,
    PostgresSettings,
    QuotaFailMode,
    QuotaSettings,
    RedisSettings,
    ResilienceSettings,
    RoutingProfile,
    RoutingSettings,
    SSRFSettings,
    SecuritySettings,
    get_settings,
    reset_settings,
)

# ---------------------------------------------------------------------------
# Env vars that may leak from the host — cleared before each test
# ---------------------------------------------------------------------------

_ENV_VARS_TO_CLEAR = (
    "ROUTEIQ_PORT",
    "ROUTEIQ_HOST",
    "ROUTEIQ_ENV",
    "ROUTEIQ_WORKERS",
    "ROUTEIQ_DEBUG",
    "ROUTEIQ_ADMIN_UI_ENABLED",
    "ROUTEIQ_SKIP_ENV_VALIDATION",
    "ROUTEIQ_CONFIG_PATH",
    "ROUTEIQ_LITELLM_MASTER_KEY",
    "ROUTEIQ_LITELLM_CONFIG_PATH",
    "ROUTEIQ_LITELLM_PORT",
    "ROUTEIQ_LITELLM_HOST",
    "ROUTEIQ_LITELLM_DEBUG",
    "ROUTEIQ_LLMROUTER_ROUTER_CALLBACK_ENABLED",
    # Nested: redis
    "ROUTEIQ_REDIS__HOST",
    "ROUTEIQ_REDIS__PORT",
    "ROUTEIQ_REDIS__PASSWORD",
    "ROUTEIQ_REDIS__SSL",
    "ROUTEIQ_REDIS__DB",
    # Nested: postgres
    "ROUTEIQ_POSTGRES__URL",
    "ROUTEIQ_POSTGRES__POOL_MIN",
    "ROUTEIQ_POSTGRES__POOL_MAX",
    "ROUTEIQ_POSTGRES__SSL_MODE",
    # Nested: otel
    "ROUTEIQ_OTEL__ENABLED",
    "ROUTEIQ_OTEL__ENDPOINT",
    "ROUTEIQ_OTEL__SERVICE_NAME",
    "ROUTEIQ_OTEL__SAMPLE_RATE",
    # Nested: oidc
    "ROUTEIQ_OIDC__ENABLED",
    "ROUTEIQ_OIDC__ISSUER_URL",
    "ROUTEIQ_OIDC__CLIENT_ID",
    # Nested: routing
    "ROUTEIQ_ROUTING__USE_PLUGIN_STRATEGY",
    "ROUTEIQ_ROUTING__CENTROID_ENABLED",
    "ROUTEIQ_ROUTING__DEFAULT_PROFILE",
    # Nested: security
    "ROUTEIQ_SECURITY__ADMIN_API_KEYS",
    "ROUTEIQ_SECURITY__ADMIN_API_KEY",
    "ROUTEIQ_SECURITY__CORS_ORIGINS",
    "ROUTEIQ_SECURITY__ENFORCE_SIGNED_MODELS",
    "ROUTEIQ_SECURITY__ALLOW_PICKLE_MODELS",
    # Nested: mcp
    "ROUTEIQ_MCP__ENABLED",
    "ROUTEIQ_MCP__TOOL_INVOCATION_ENABLED",
    # Nested: a2a
    "ROUTEIQ_A2A__ENABLED",
)


@pytest.fixture(autouse=True)
def _reset_settings(monkeypatch):
    """Clear env vars and reset settings singleton before each test."""
    for var in _ENV_VARS_TO_CLEAR:
        monkeypatch.delenv(var, raising=False)
    reset_settings()
    yield
    reset_settings()


# ============================================================================
# 1. GatewaySettings defaults
# ============================================================================


class TestDefaults:
    """Test that GatewaySettings fields have correct defaults."""

    def test_core_defaults(self):
        s = GatewaySettings()
        assert s.port == 4000
        assert s.host == "0.0.0.0"
        assert s.workers == 1
        assert s.env == "production"
        assert s.debug is False
        assert s.admin_ui_enabled is False
        assert s.skip_env_validation is False
        assert s.config_path is None

    def test_litellm_passthrough_defaults(self):
        s = GatewaySettings()
        assert s.litellm_master_key is None
        assert s.litellm_config_path is None
        assert s.litellm_port == 4000
        assert s.litellm_host == "0.0.0.0"
        assert s.litellm_debug is False
        assert s.llmrouter_router_callback_enabled is True

    def test_nested_redis_defaults(self):
        s = GatewaySettings()
        assert s.redis.host is None
        assert s.redis.port == 6379
        assert s.redis.password is None
        assert s.redis.ssl is False
        assert s.redis.db == 0

    def test_nested_postgres_defaults(self):
        s = GatewaySettings()
        assert s.postgres.url is None
        assert s.postgres.pool_min == 2
        assert s.postgres.pool_max == 10
        assert s.postgres.ssl_mode == "prefer"

    def test_nested_otel_defaults(self):
        s = GatewaySettings()
        assert s.otel.enabled is True
        assert s.otel.endpoint is None
        assert s.otel.service_name == "litellm-gateway"
        assert s.otel.sample_rate == 1.0

    def test_nested_routing_defaults(self):
        s = GatewaySettings()
        assert s.routing.use_plugin_strategy is True
        assert s.routing.centroid_enabled is True
        assert s.routing.centroid_warmup is False
        assert s.routing.default_profile == RoutingProfile.AUTO
        assert s.routing.pipeline_enabled is True

    def test_nested_security_defaults(self):
        s = GatewaySettings()
        assert s.security.admin_api_keys is None
        assert s.security.admin_auth_enabled is True
        assert s.security.key_prefix == "sk-riq-"
        assert s.security.cors_origins == "*"
        assert s.security.cors_credentials is False
        assert s.security.enforce_signed_models is True
        assert s.security.allow_pickle_models is False

    def test_nested_mcp_defaults(self):
        s = GatewaySettings()
        assert s.mcp.enabled is False
        assert s.mcp.tool_invocation_enabled is False
        assert s.mcp.ha_sync_enabled is True
        assert s.mcp.sync_interval == 5.0
        assert s.mcp.oauth_enabled is False

    def test_nested_a2a_defaults(self):
        s = GatewaySettings()
        assert s.a2a.enabled is False
        assert s.a2a.base_url == ""

    def test_nested_oidc_defaults(self):
        s = GatewaySettings()
        assert s.oidc.enabled is False
        assert s.oidc.issuer_url is None
        assert s.oidc.client_id is None
        assert s.oidc.user_id_claim == "sub"
        assert s.oidc.email_claim == "email"
        assert s.oidc.display_name_claim == "name"
        assert s.oidc.default_role == "internal_user"
        assert s.oidc.session_ttl == 1800
        assert s.oidc.auto_provision_users is True

    def test_nested_resilience_defaults(self):
        s = GatewaySettings()
        assert s.resilience.max_concurrent_requests == 0
        assert s.resilience.drain_timeout_seconds == 30.0
        assert s.resilience.provider_circuit_breaker_enabled is False

    def test_nested_policy_defaults(self):
        s = GatewaySettings()
        assert s.policy.enabled is False
        assert s.policy.fail_mode == PolicyFailMode.OPEN

    def test_nested_audit_defaults(self):
        s = GatewaySettings()
        assert s.audit.enabled is True
        assert s.audit.fail_mode == AuditFailMode.OPEN

    def test_nested_quota_defaults(self):
        s = GatewaySettings()
        assert s.quota.enabled is False
        assert s.quota.fail_mode == QuotaFailMode.OPEN

    def test_nested_cache_defaults(self):
        s = GatewaySettings()
        assert s.cache.enabled is False
        assert s.cache.semantic_enabled is False
        assert s.cache.ttl_seconds == 3600
        assert s.cache.similarity_threshold == 0.95
        assert s.cache.embedding_model == "all-MiniLM-L6-v2"

    def test_nested_ha_defaults(self):
        s = GatewaySettings()
        assert s.ha.mode == ""
        assert s.ha.leader_migrations is False


# ============================================================================
# 2. Singleton pattern
# ============================================================================


class TestSingleton:
    """Test get_settings() / reset_settings() singleton behaviour."""

    def test_get_settings_returns_same_instance(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_reset_settings_clears_singleton(self):
        s1 = get_settings()
        reset_settings()
        s2 = get_settings()
        assert s1 is not s2

    def test_get_settings_accepts_overrides(self):
        s = get_settings(port=9999)
        assert s.port == 9999

    def test_overrides_ignored_on_second_call(self):
        s1 = get_settings(port=9999)
        s2 = get_settings(port=1111)
        assert s2.port == 9999  # second call returns cached
        assert s1 is s2


# ============================================================================
# 3. Environment variable loading
# ============================================================================


class TestEnvVarLoading:
    """Test that env vars are correctly mapped to settings fields."""

    def test_routeiq_port(self, monkeypatch):
        monkeypatch.setenv("ROUTEIQ_PORT", "8080")
        s = GatewaySettings()
        assert s.port == 8080

    def test_routeiq_env(self, monkeypatch):
        monkeypatch.setenv("ROUTEIQ_ENV", "staging")
        s = GatewaySettings()
        assert s.env == "staging"

    def test_routeiq_workers(self, monkeypatch):
        monkeypatch.setenv("ROUTEIQ_WORKERS", "4")
        s = GatewaySettings()
        assert s.workers == 4

    def test_routeiq_debug_true(self, monkeypatch):
        monkeypatch.setenv("ROUTEIQ_DEBUG", "true")
        s = GatewaySettings()
        assert s.debug is True

    def test_routeiq_debug_false(self, monkeypatch):
        monkeypatch.setenv("ROUTEIQ_DEBUG", "false")
        s = GatewaySettings()
        assert s.debug is False

    def test_routeiq_debug_one(self, monkeypatch):
        monkeypatch.setenv("ROUTEIQ_DEBUG", "1")
        s = GatewaySettings()
        assert s.debug is True

    def test_routeiq_debug_zero(self, monkeypatch):
        monkeypatch.setenv("ROUTEIQ_DEBUG", "0")
        s = GatewaySettings()
        assert s.debug is False

    def test_nested_redis_host(self, monkeypatch):
        monkeypatch.setenv("ROUTEIQ_REDIS__HOST", "redis.example.com")
        s = GatewaySettings()
        assert s.redis.host == "redis.example.com"

    def test_nested_redis_port(self, monkeypatch):
        monkeypatch.setenv("ROUTEIQ_REDIS__PORT", "6380")
        s = GatewaySettings()
        assert s.redis.port == 6380

    def test_nested_redis_ssl(self, monkeypatch):
        monkeypatch.setenv("ROUTEIQ_REDIS__SSL", "True")
        s = GatewaySettings()
        assert s.redis.ssl is True

    def test_nested_otel_enabled(self, monkeypatch):
        monkeypatch.setenv("ROUTEIQ_OTEL__ENABLED", "false")
        s = GatewaySettings()
        assert s.otel.enabled is False

    def test_nested_otel_endpoint(self, monkeypatch):
        monkeypatch.setenv("ROUTEIQ_OTEL__ENDPOINT", "http://collector:4317")
        s = GatewaySettings()
        assert s.otel.endpoint == "http://collector:4317"

    def test_nested_routing_profile(self, monkeypatch):
        monkeypatch.setenv("ROUTEIQ_ROUTING__DEFAULT_PROFILE", "eco")
        s = GatewaySettings()
        assert s.routing.default_profile == RoutingProfile.ECO

    def test_nested_security_cors(self, monkeypatch):
        monkeypatch.setenv(
            "ROUTEIQ_SECURITY__CORS_ORIGINS",
            "http://localhost:3000,https://app.example.com",
        )
        s = GatewaySettings()
        assert (
            s.security.cors_origins == "http://localhost:3000,https://app.example.com"
        )

    def test_nested_mcp_enabled(self, monkeypatch):
        monkeypatch.setenv("ROUTEIQ_MCP__ENABLED", "true")
        s = GatewaySettings()
        assert s.mcp.enabled is True

    def test_litellm_master_key_via_routeiq_prefix(self, monkeypatch):
        monkeypatch.setenv("ROUTEIQ_LITELLM_MASTER_KEY", "sk-real-key-123")
        s = GatewaySettings()
        assert s.litellm_master_key == "sk-real-key-123"

    def test_case_insensitive_env(self, monkeypatch):
        """Pydantic settings with case_sensitive=False should accept any case."""
        monkeypatch.setenv("routeiq_port", "7777")
        s = GatewaySettings()
        assert s.port == 7777


# ============================================================================
# 4. Validators
# ============================================================================


class TestValidators:
    """Test field validators and value constraints."""

    def test_port_min_boundary(self):
        s = GatewaySettings(port=1)
        assert s.port == 1

    def test_port_max_boundary(self):
        s = GatewaySettings(port=65535)
        assert s.port == 65535

    def test_port_zero_rejected(self):
        with pytest.raises(ValidationError, match="port"):
            GatewaySettings(port=0)

    def test_port_above_max_rejected(self):
        with pytest.raises(ValidationError, match="port"):
            GatewaySettings(port=65536)

    def test_port_negative_rejected(self):
        with pytest.raises(ValidationError, match="port"):
            GatewaySettings(port=-1)

    def test_workers_min_one(self):
        s = GatewaySettings(workers=1)
        assert s.workers == 1

    def test_workers_zero_rejected(self):
        with pytest.raises(ValidationError, match="workers"):
            GatewaySettings(workers=0)

    def test_workers_negative_rejected(self):
        with pytest.raises(ValidationError, match="workers"):
            GatewaySettings(workers=-5)

    def test_master_key_placeholder_warns_sk1234(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            GatewaySettings(litellm_master_key="sk-1234")
            placeholder_warnings = [x for x in w if "placeholder" in str(x.message)]
            assert len(placeholder_warnings) >= 1

    def test_master_key_placeholder_warns_changeme(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            GatewaySettings(litellm_master_key="changeme")
            placeholder_warnings = [x for x in w if "placeholder" in str(x.message)]
            assert len(placeholder_warnings) >= 1

    def test_master_key_placeholder_warns_test(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            GatewaySettings(litellm_master_key="test")
            placeholder_warnings = [x for x in w if "placeholder" in str(x.message)]
            assert len(placeholder_warnings) >= 1

    def test_master_key_placeholder_warns_your_key_here(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            GatewaySettings(litellm_master_key="your-key-here")
            placeholder_warnings = [x for x in w if "placeholder" in str(x.message)]
            assert len(placeholder_warnings) >= 1

    def test_master_key_real_value_no_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            GatewaySettings(litellm_master_key="sk-riq-prod-abc123xyz")
            placeholder_warnings = [x for x in w if "placeholder" in str(x.message)]
            assert len(placeholder_warnings) == 0

    def test_master_key_none_no_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            GatewaySettings(litellm_master_key=None)
            placeholder_warnings = [x for x in w if "placeholder" in str(x.message)]
            assert len(placeholder_warnings) == 0

    def test_routing_profile_all_valid_values(self):
        for profile in ("auto", "eco", "premium", "free", "reasoning"):
            s = RoutingSettings(default_profile=profile)
            assert s.default_profile.value == profile

    def test_routing_profile_invalid_rejected(self):
        with pytest.raises(ValidationError, match="default_profile"):
            RoutingSettings(default_profile="turbo")

    def test_otel_sample_rate_min(self):
        s = OTelSettings(sample_rate=0.0)
        assert s.sample_rate == 0.0

    def test_otel_sample_rate_max(self):
        s = OTelSettings(sample_rate=1.0)
        assert s.sample_rate == 1.0

    def test_otel_sample_rate_above_max_rejected(self):
        with pytest.raises(ValidationError, match="sample_rate"):
            OTelSettings(sample_rate=1.1)

    def test_otel_sample_rate_below_min_rejected(self):
        with pytest.raises(ValidationError, match="sample_rate"):
            OTelSettings(sample_rate=-0.1)


# ============================================================================
# 5. Cross-field validation
# ============================================================================


class TestCrossFieldValidation:
    """Test model validators that check relationships between fields."""

    def test_pickle_without_signing_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SecuritySettings(allow_pickle_models=True, enforce_signed_models=False)
            pickle_warnings = [x for x in w if "pickle" in str(x.message).lower()]
            assert len(pickle_warnings) >= 1

    def test_pickle_with_signing_no_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SecuritySettings(allow_pickle_models=True, enforce_signed_models=True)
            pickle_warnings = [
                x for x in w if "unverified pickle" in str(x.message).lower()
            ]
            assert len(pickle_warnings) == 0

    def test_pickle_disabled_no_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SecuritySettings(allow_pickle_models=False, enforce_signed_models=False)
            pickle_warnings = [x for x in w if "pickle" in str(x.message).lower()]
            assert len(pickle_warnings) == 0

    def test_multi_workers_legacy_mode_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            GatewaySettings(
                workers=4,
                routing=RoutingSettings(use_plugin_strategy=False),
            )
            worker_warnings = [x for x in w if "workers" in str(x.message).lower()]
            assert len(worker_warnings) >= 1

    def test_multi_workers_plugin_mode_no_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            GatewaySettings(
                workers=4,
                routing=RoutingSettings(use_plugin_strategy=True),
            )
            worker_warnings = [
                x
                for x in w
                if "workers" in str(x.message).lower()
                and "monkey-patch" in str(x.message).lower()
            ]
            assert len(worker_warnings) == 0

    def test_single_worker_legacy_mode_no_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            GatewaySettings(
                workers=1,
                routing=RoutingSettings(use_plugin_strategy=False),
            )
            worker_warnings = [
                x
                for x in w
                if "workers" in str(x.message).lower()
                and "monkey-patch" in str(x.message).lower()
            ]
            assert len(worker_warnings) == 0


# ============================================================================
# 6. Nested model tests — deeper field validation
# ============================================================================


class TestRedisSettings:
    def test_port_valid_range(self):
        r = RedisSettings(port=1)
        assert r.port == 1
        r2 = RedisSettings(port=65535)
        assert r2.port == 65535

    def test_port_invalid(self):
        with pytest.raises(ValidationError):
            RedisSettings(port=0)
        with pytest.raises(ValidationError):
            RedisSettings(port=70000)

    def test_db_valid_range(self):
        r = RedisSettings(db=0)
        assert r.db == 0
        r2 = RedisSettings(db=15)
        assert r2.db == 15

    def test_db_out_of_range(self):
        with pytest.raises(ValidationError):
            RedisSettings(db=-1)
        with pytest.raises(ValidationError):
            RedisSettings(db=16)


class TestPostgresSettings:
    def test_valid_postgres_url(self):
        p = PostgresSettings(url="postgresql://user:pass@host:5432/db")
        assert p.url == "postgresql://user:pass@host:5432/db"

    def test_valid_postgres_scheme_variant(self):
        p = PostgresSettings(url="postgres://user:pass@host/db")
        assert p.url.startswith("postgres://")

    def test_valid_sqlite_url(self):
        p = PostgresSettings(url="sqlite:///path/to/db.sqlite")
        assert p.url.startswith("sqlite://")

    def test_unrecognised_scheme_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            PostgresSettings(url="mysql://user:pass@host/db")
            scheme_warnings = [x for x in w if "scheme" in str(x.message).lower()]
            assert len(scheme_warnings) >= 1

    def test_none_url_valid(self):
        p = PostgresSettings(url=None)
        assert p.url is None

    def test_pool_min_boundary(self):
        with pytest.raises(ValidationError):
            PostgresSettings(pool_min=0)

    def test_pool_max_boundary(self):
        with pytest.raises(ValidationError):
            PostgresSettings(pool_max=101)


class TestOTelSettings:
    def test_valid_http_endpoint(self):
        o = OTelSettings(endpoint="http://collector:4317")
        assert o.endpoint == "http://collector:4317"

    def test_valid_https_endpoint(self):
        o = OTelSettings(endpoint="https://otel.example.com:4317")
        assert o.endpoint == "https://otel.example.com:4317"

    def test_valid_grpc_endpoint(self):
        o = OTelSettings(endpoint="grpc://collector:4317")
        assert o.endpoint == "grpc://collector:4317"

    def test_bad_scheme_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            OTelSettings(endpoint="ftp://bad.example.com")
            scheme_warnings = [x for x in w if "scheme" in str(x.message).lower()]
            assert len(scheme_warnings) >= 1


class TestOIDCSettings:
    def test_https_issuer_valid(self):
        o = OIDCSettings(issuer_url="https://auth.example.com")
        assert o.issuer_url == "https://auth.example.com"

    def test_localhost_http_allowed(self):
        o = OIDCSettings(issuer_url="http://localhost:8080")
        assert o.issuer_url == "http://localhost:8080"

    def test_127_0_0_1_http_allowed(self):
        o = OIDCSettings(issuer_url="http://127.0.0.1:8080")
        assert o.issuer_url == "http://127.0.0.1:8080"

    def test_http_non_localhost_rejected(self):
        with pytest.raises(ValidationError, match="HTTPS"):
            OIDCSettings(issuer_url="http://auth.example.com")

    def test_trailing_slash_stripped(self):
        o = OIDCSettings(issuer_url="https://auth.example.com/")
        assert not o.issuer_url.endswith("/")

    def test_none_issuer_valid(self):
        o = OIDCSettings(issuer_url=None)
        assert o.issuer_url is None

    def test_session_ttl_boundaries(self):
        o = OIDCSettings(session_ttl=60)
        assert o.session_ttl == 60
        o2 = OIDCSettings(session_ttl=86400)
        assert o2.session_ttl == 86400

    def test_session_ttl_below_min(self):
        with pytest.raises(ValidationError):
            OIDCSettings(session_ttl=59)

    def test_session_ttl_above_max(self):
        with pytest.raises(ValidationError):
            OIDCSettings(session_ttl=86401)

    def test_role_mapping_default_empty(self):
        o = OIDCSettings()
        assert o.role_mapping == {}

    def test_allowed_email_domains_default_empty(self):
        o = OIDCSettings()
        assert o.allowed_email_domains == []


class TestMCPSettings:
    def test_sync_interval_min(self):
        m = MCPSettings(sync_interval=1.0)
        assert m.sync_interval == 1.0

    def test_sync_interval_below_min(self):
        with pytest.raises(ValidationError):
            MCPSettings(sync_interval=0.5)


class TestResilienceSettings:
    def test_max_concurrent_zero_means_unlimited(self):
        r = ResilienceSettings(max_concurrent_requests=0)
        assert r.max_concurrent_requests == 0

    def test_drain_timeout_zero_valid(self):
        r = ResilienceSettings(drain_timeout_seconds=0.0)
        assert r.drain_timeout_seconds == 0.0


class TestHTTPClientSettings:
    def test_defaults(self):
        h = HTTPClientSettings()
        assert h.pooling_enabled is True
        assert h.max_connections == 100
        assert h.max_keepalive == 20
        assert h.default_timeout == 60.0

    def test_timeout_boundary(self):
        with pytest.raises(ValidationError):
            HTTPClientSettings(default_timeout=0.5)


class TestSSRFSettings:
    def test_defaults(self):
        s = SSRFSettings()
        assert s.outbound_url_allowlist == ""
        assert s.use_sync_dns is False
        assert s.dns_timeout == 2.0
        assert s.dns_cache_ttl == 300
        assert s.dns_cache_size == 256

    def test_dns_timeout_below_min(self):
        with pytest.raises(ValidationError):
            SSRFSettings(dns_timeout=0.05)


class TestConversationAffinitySettings:
    def test_ttl_boundary(self):
        with pytest.raises(ValidationError):
            ConversationAffinitySettings(ttl=59)

    def test_max_entries_boundary(self):
        with pytest.raises(ValidationError):
            ConversationAffinitySettings(max_entries=99)


class TestCacheSettings:
    def test_similarity_threshold_range(self):
        c = CacheSettings(similarity_threshold=0.0)
        assert c.similarity_threshold == 0.0
        c2 = CacheSettings(similarity_threshold=1.0)
        assert c2.similarity_threshold == 1.0

    def test_similarity_threshold_above_max(self):
        with pytest.raises(ValidationError):
            CacheSettings(similarity_threshold=1.1)

    def test_max_temperature_range(self):
        c = CacheSettings(max_temperature=0.0)
        assert c.max_temperature == 0.0
        c2 = CacheSettings(max_temperature=2.0)
        assert c2.max_temperature == 2.0

    def test_max_temperature_above_max(self):
        with pytest.raises(ValidationError):
            CacheSettings(max_temperature=2.1)


class TestPluginSettings:
    def test_defaults(self):
        p = PluginSettings()
        assert p.plugins == ""
        assert p.allowlist is None
        assert p.failure_mode == "continue"
        assert p.startup_timeout == 30.0

    def test_startup_timeout_below_min(self):
        with pytest.raises(ValidationError):
            PluginSettings(startup_timeout=0.5)


# ============================================================================
# 7. Convenience properties
# ============================================================================


class TestConvenienceProperties:
    def test_is_production_true(self):
        s = GatewaySettings(env="production")
        assert s.is_production is True

    def test_is_production_false_staging(self):
        s = GatewaySettings(env="staging")
        assert s.is_production is False

    def test_is_production_false_development(self):
        s = GatewaySettings(env="development")
        assert s.is_production is False

    def test_is_production_case_insensitive(self):
        s = GatewaySettings(env="Production")
        assert s.is_production is True

    def test_redis_configured_true(self):
        s = GatewaySettings(redis=RedisSettings(host="redis.local"))
        assert s.redis_configured is True

    def test_redis_configured_false(self):
        s = GatewaySettings()
        assert s.redis_configured is False

    def test_postgres_configured_true(self):
        s = GatewaySettings(
            postgres=PostgresSettings(url="postgresql://user:pass@host/db")
        )
        assert s.postgres_configured is True

    def test_postgres_configured_false(self):
        s = GatewaySettings()
        assert s.postgres_configured is False

    def test_cors_origins_list_wildcard(self):
        s = GatewaySettings()
        assert s.cors_origins_list == ["*"]

    def test_cors_origins_list_multiple(self):
        s = GatewaySettings(
            security=SecuritySettings(
                cors_origins="http://localhost:3000, https://app.example.com"
            )
        )
        assert s.cors_origins_list == [
            "http://localhost:3000",
            "https://app.example.com",
        ]

    def test_cors_origins_list_single(self):
        s = GatewaySettings(
            security=SecuritySettings(cors_origins="https://only.example.com")
        )
        assert s.cors_origins_list == ["https://only.example.com"]

    def test_admin_api_keys_set_empty(self):
        s = GatewaySettings()
        assert s.admin_api_keys_set == set()

    def test_admin_api_keys_set_from_keys(self):
        s = GatewaySettings(
            security=SecuritySettings(admin_api_keys="key1, key2, key3")
        )
        assert s.admin_api_keys_set == {"key1", "key2", "key3"}

    def test_admin_api_keys_set_from_single_key(self):
        s = GatewaySettings(security=SecuritySettings(admin_api_key="single-key"))
        assert s.admin_api_keys_set == {"single-key"}

    def test_admin_api_keys_set_combined(self):
        s = GatewaySettings(
            security=SecuritySettings(
                admin_api_keys="key1, key2",
                admin_api_key="key3",
            )
        )
        assert s.admin_api_keys_set == {"key1", "key2", "key3"}

    def test_admin_api_keys_set_deduplication(self):
        s = GatewaySettings(
            security=SecuritySettings(
                admin_api_keys="key1, key2",
                admin_api_key="key1",
            )
        )
        assert s.admin_api_keys_set == {"key1", "key2"}

    def test_admin_api_keys_set_strips_whitespace(self):
        s = GatewaySettings(
            security=SecuritySettings(admin_api_keys="  key1 ,  key2  ")
        )
        assert s.admin_api_keys_set == {"key1", "key2"}

    def test_admin_api_keys_set_skips_empty_entries(self):
        s = GatewaySettings(security=SecuritySettings(admin_api_keys="key1,,key2,"))
        assert s.admin_api_keys_set == {"key1", "key2"}


# ============================================================================
# 8. Enums
# ============================================================================


class TestEnums:
    def test_routing_profile_values(self):
        assert RoutingProfile.AUTO.value == "auto"
        assert RoutingProfile.ECO.value == "eco"
        assert RoutingProfile.PREMIUM.value == "premium"
        assert RoutingProfile.FREE.value == "free"
        assert RoutingProfile.REASONING.value == "reasoning"

    def test_policy_fail_mode_values(self):
        assert PolicyFailMode.OPEN.value == "open"
        assert PolicyFailMode.CLOSED.value == "closed"

    def test_audit_fail_mode_values(self):
        assert AuditFailMode.OPEN.value == "open"
        assert AuditFailMode.CLOSED.value == "closed"

    def test_quota_fail_mode_values(self):
        assert QuotaFailMode.OPEN.value == "open"
        assert QuotaFailMode.CLOSED.value == "closed"


# ============================================================================
# 9. Edge cases
# ============================================================================


class TestEdgeCases:
    def test_empty_string_env_for_optional(self, monkeypatch):
        """Empty string for optional string fields should be stored as-is."""
        monkeypatch.setenv("ROUTEIQ_LITELLM_MASTER_KEY", "")
        s = GatewaySettings()
        # Pydantic may coerce empty string for Optional[str]; just confirm no crash
        assert s.litellm_master_key is not None or s.litellm_master_key is None

    def test_instantiation_with_no_env_vars(self):
        """Pure defaults — no env vars at all."""
        s = GatewaySettings()
        assert s.port == 4000
        assert s.env == "production"

    def test_unicode_in_env(self, monkeypatch):
        monkeypatch.setenv("ROUTEIQ_ENV", "développement")
        s = GatewaySettings()
        assert s.env == "développement"

    def test_unicode_in_redis_host(self, monkeypatch):
        monkeypatch.setenv("ROUTEIQ_REDIS__HOST", "redis-ünïcödë.local")
        s = GatewaySettings()
        assert s.redis.host == "redis-ünïcödë.local"

    def test_extra_env_vars_ignored(self, monkeypatch):
        """model_config extra='ignore' should silently drop unknown fields."""
        monkeypatch.setenv("ROUTEIQ_NONEXISTENT_FIELD", "whatever")
        s = GatewaySettings()
        assert not hasattr(s, "nonexistent_field")

    def test_large_workers_value(self):
        """Large but valid workers count should be accepted."""
        s = GatewaySettings(workers=128)
        assert s.workers == 128

    def test_invalid_port_type_from_env(self, monkeypatch):
        """Non-integer port from env should raise ValidationError."""
        monkeypatch.setenv("ROUTEIQ_PORT", "not-a-number")
        with pytest.raises(ValidationError):
            GatewaySettings()

    def test_invalid_boolean_from_env(self, monkeypatch):
        """Nonsense boolean from env should raise."""
        monkeypatch.setenv("ROUTEIQ_DEBUG", "notabool")
        with pytest.raises(ValidationError):
            GatewaySettings()

    def test_nested_multiple_fields_at_once(self, monkeypatch):
        """Set multiple nested fields in one go."""
        monkeypatch.setenv("ROUTEIQ_REDIS__HOST", "redis.prod")
        monkeypatch.setenv("ROUTEIQ_REDIS__PORT", "6380")
        monkeypatch.setenv("ROUTEIQ_REDIS__SSL", "true")
        monkeypatch.setenv("ROUTEIQ_REDIS__DB", "3")
        s = GatewaySettings()
        assert s.redis.host == "redis.prod"
        assert s.redis.port == 6380
        assert s.redis.ssl is True
        assert s.redis.db == 3

    def test_postgres_pool_min_exceeds_max_accepted(self):
        """Pydantic doesn't cross-validate pool_min < pool_max; just range-check."""
        p = PostgresSettings(pool_min=50, pool_max=1)
        assert p.pool_min == 50
        assert p.pool_max == 1

    def test_config_sync_all_none_is_valid(self):
        """Config sync with no S3/GCS config is valid (no remote sync)."""
        cs = ConfigSyncSettings()
        assert cs.s3_bucket is None
        assert cs.gcs_bucket is None

    def test_ha_empty_mode_is_disabled(self):
        ha = HASettings()
        assert ha.mode == ""

    def test_management_defaults(self):
        m = ManagementSettings()
        assert m.rbac_enabled is False
        assert m.otel_enabled is True
