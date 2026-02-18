"""
Tests for environment variable validation.

Each test manipulates ``os.environ`` via the ``clean_env`` fixture to ensure
isolation between test cases.
"""


import pytest

from litellm_llmrouter.env_validation import ValidationResult, validate_environment


# ---------------------------------------------------------------------------
# Fixture: ensure a clean environment for every test
# ---------------------------------------------------------------------------

# All env vars the validation module inspects — we remove them before each
# test so that the host machine's environment doesn't leak in.
_ENV_VARS_UNDER_TEST = (
    "ROUTEIQ_SKIP_ENV_VALIDATION",
    "LITELLM_MASTER_KEY",
    "LITELLM_CONFIG_PATH",
    "DATABASE_URL",
    "REDIS_HOST",
    "REDIS_PORT",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "ADMIN_API_KEYS",
    "OTEL_ENABLED",
    "MCP_GATEWAY_ENABLED",
    "A2A_GATEWAY_ENABLED",
    "CONFIG_HOT_RELOAD",
    "POLICY_ENGINE_ENABLED",
    "MCP_SSE_TRANSPORT_ENABLED",
    "MCP_PROTOCOL_PROXY_ENABLED",
    "LLMROUTER_ALLOW_PICKLE_MODELS",
    "LLMROUTER_ENFORCE_SIGNED_MODELS",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Remove all env vars under test before each test case."""
    for var in _ENV_VARS_UNDER_TEST:
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# Skip validation
# ---------------------------------------------------------------------------


class TestSkipValidation:
    """Tests for ROUTEIQ_SKIP_ENV_VALIDATION behaviour."""

    def test_skip_returns_empty_result(self, monkeypatch):
        monkeypatch.setenv("ROUTEIQ_SKIP_ENV_VALIDATION", "true")
        result = validate_environment()
        assert result.errors == []
        assert result.warnings == []

    def test_skip_with_one(self, monkeypatch):
        monkeypatch.setenv("ROUTEIQ_SKIP_ENV_VALIDATION", "1")
        result = validate_environment()
        assert result.errors == []
        assert result.warnings == []

    def test_skip_with_yes(self, monkeypatch):
        monkeypatch.setenv("ROUTEIQ_SKIP_ENV_VALIDATION", "yes")
        result = validate_environment()
        assert result.errors == []
        assert result.warnings == []

    def test_skip_ignores_bad_config(self, monkeypatch):
        """Even blatant errors are ignored when skip is on."""
        monkeypatch.setenv("ROUTEIQ_SKIP_ENV_VALIDATION", "true")
        monkeypatch.setenv("LITELLM_CONFIG_PATH", "/nonexistent/path.yaml")
        monkeypatch.setenv("DATABASE_URL", "mongodb://bad")
        result = validate_environment()
        assert result.errors == []
        assert result.warnings == []


# ---------------------------------------------------------------------------
# Valid configuration (no errors, no warnings)
# ---------------------------------------------------------------------------


class TestValidConfiguration:
    """A fully-valid configuration should produce zero errors and warnings."""

    def test_minimal_valid_config(self, monkeypatch):
        """Only master key set — still valid, no warnings."""
        monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-real-key")
        result = validate_environment()
        assert result.errors == []
        assert result.warnings == []

    def test_full_valid_config(self, monkeypatch, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("model_list: []")

        monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-real-key")
        monkeypatch.setenv("LITELLM_CONFIG_PATH", str(config_file))
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost/db")
        monkeypatch.setenv("REDIS_HOST", "localhost")
        monkeypatch.setenv("REDIS_PORT", "6379")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        monkeypatch.setenv("ADMIN_API_KEYS", "sk-admin-key-1,sk-admin-key-2")
        monkeypatch.setenv("OTEL_ENABLED", "true")
        monkeypatch.setenv("MCP_GATEWAY_ENABLED", "false")

        result = validate_environment()
        assert result.errors == []
        assert result.warnings == []


# ---------------------------------------------------------------------------
# LITELLM_MASTER_KEY
# ---------------------------------------------------------------------------


class TestMasterKey:
    def test_missing_master_key_warns(self):
        result = validate_environment()
        assert any("LITELLM_MASTER_KEY" in w for w in result.warnings)

    def test_present_master_key_no_warning(self, monkeypatch):
        monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-prod-key")
        result = validate_environment()
        assert not any("LITELLM_MASTER_KEY" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# LITELLM_CONFIG_PATH
# ---------------------------------------------------------------------------


class TestConfigPath:
    def test_nonexistent_config_path_errors(self, monkeypatch):
        monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-key")
        monkeypatch.setenv("LITELLM_CONFIG_PATH", "/nonexistent/config.yaml")
        result = validate_environment()
        assert any("LITELLM_CONFIG_PATH" in e for e in result.errors)
        assert any("does not exist" in e for e in result.errors)

    def test_existing_config_path_no_error(self, monkeypatch, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("model_list: []")
        monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-key")
        monkeypatch.setenv("LITELLM_CONFIG_PATH", str(config_file))
        result = validate_environment()
        assert result.errors == []

    def test_unset_config_path_no_error(self, monkeypatch):
        monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-key")
        result = validate_environment()
        assert not any("LITELLM_CONFIG_PATH" in e for e in result.errors)


# ---------------------------------------------------------------------------
# DATABASE_URL
# ---------------------------------------------------------------------------


class TestDatabaseUrl:
    @pytest.mark.parametrize(
        "url",
        [
            "postgresql://user:pass@localhost:5432/db",
            "postgres://user:pass@localhost:5432/db",
            "sqlite:///path/to/db.sqlite",
        ],
    )
    def test_valid_database_url(self, monkeypatch, url):
        monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-key")
        monkeypatch.setenv("DATABASE_URL", url)
        result = validate_environment()
        assert not any("DATABASE_URL" in w for w in result.warnings)

    @pytest.mark.parametrize(
        "url",
        [
            "mongodb://localhost:27017/db",
            "mysql://user:pass@localhost/db",
            "not-a-url",
        ],
    )
    def test_invalid_database_url_warns(self, monkeypatch, url):
        monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-key")
        monkeypatch.setenv("DATABASE_URL", url)
        result = validate_environment()
        assert any("DATABASE_URL" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# REDIS_HOST / REDIS_PORT
# ---------------------------------------------------------------------------


class TestRedis:
    def test_host_without_port_warns(self, monkeypatch):
        monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-key")
        monkeypatch.setenv("REDIS_HOST", "localhost")
        result = validate_environment()
        assert any("REDIS_PORT" in w for w in result.warnings)

    def test_port_without_host_warns(self, monkeypatch):
        monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-key")
        monkeypatch.setenv("REDIS_PORT", "6379")
        result = validate_environment()
        assert any("REDIS_HOST" in w for w in result.warnings)

    def test_both_set_no_warning(self, monkeypatch):
        monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-key")
        monkeypatch.setenv("REDIS_HOST", "localhost")
        monkeypatch.setenv("REDIS_PORT", "6379")
        result = validate_environment()
        assert not any("REDIS" in w for w in result.warnings)

    def test_neither_set_no_warning(self, monkeypatch):
        monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-key")
        result = validate_environment()
        assert not any("REDIS" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# OTEL_EXPORTER_OTLP_ENDPOINT
# ---------------------------------------------------------------------------


class TestOtelEndpoint:
    @pytest.mark.parametrize(
        "endpoint",
        [
            "http://localhost:4317",
            "https://otel.example.com:4318",
            "grpc://collector:4317",
        ],
    )
    def test_valid_otel_endpoint(self, monkeypatch, endpoint):
        monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-key")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", endpoint)
        result = validate_environment()
        assert not any("OTEL_EXPORTER_OTLP_ENDPOINT" in w for w in result.warnings)

    @pytest.mark.parametrize(
        "endpoint",
        [
            "localhost:4317",
            "ftp://otel.example.com",
            "otel-collector",
        ],
    )
    def test_invalid_otel_endpoint_warns(self, monkeypatch, endpoint):
        monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-key")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", endpoint)
        result = validate_environment()
        assert any("OTEL_EXPORTER_OTLP_ENDPOINT" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# ADMIN_API_KEYS
# ---------------------------------------------------------------------------


class TestAdminApiKeys:
    def test_placeholder_changeme_warns(self, monkeypatch):
        monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-key")
        monkeypatch.setenv("ADMIN_API_KEYS", "changeme")
        result = validate_environment()
        assert any("placeholder" in w for w in result.warnings)

    def test_placeholder_test_warns(self, monkeypatch):
        monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-key")
        monkeypatch.setenv("ADMIN_API_KEYS", "test")
        result = validate_environment()
        assert any("placeholder" in w for w in result.warnings)

    def test_mixed_keys_warns_only_for_placeholder(self, monkeypatch):
        monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-key")
        monkeypatch.setenv("ADMIN_API_KEYS", "sk-good-key,changeme,sk-another")
        result = validate_environment()
        placeholder_warnings = [w for w in result.warnings if "placeholder" in w]
        assert len(placeholder_warnings) == 1

    def test_real_keys_no_warning(self, monkeypatch):
        monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-key")
        monkeypatch.setenv("ADMIN_API_KEYS", "sk-admin-key-1,sk-admin-key-2")
        result = validate_environment()
        assert not any("placeholder" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Boolean env vars
# ---------------------------------------------------------------------------


class TestBooleanEnvVars:
    @pytest.mark.parametrize("value", ["true", "false", "1", "0", "yes", "no", ""])
    def test_valid_boolean_values(self, monkeypatch, value):
        monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-key")
        monkeypatch.setenv("OTEL_ENABLED", value)
        result = validate_environment()
        assert not any("OTEL_ENABLED" in w for w in result.warnings)

    @pytest.mark.parametrize("value", ["maybe", "2", "enabled", "on", "off", "TRUE "])
    def test_invalid_boolean_values_warn(self, monkeypatch, value):
        monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-key")
        monkeypatch.setenv("OTEL_ENABLED", value)
        result = validate_environment()
        assert any("OTEL_ENABLED" in w for w in result.warnings)

    def test_multiple_boolean_vars_checked(self, monkeypatch):
        monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-key")
        monkeypatch.setenv("MCP_GATEWAY_ENABLED", "nope")
        monkeypatch.setenv("A2A_GATEWAY_ENABLED", "nah")
        result = validate_environment()
        assert any("MCP_GATEWAY_ENABLED" in w for w in result.warnings)
        assert any("A2A_GATEWAY_ENABLED" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Port env vars
# ---------------------------------------------------------------------------


class TestPortEnvVars:
    def test_valid_port(self, monkeypatch):
        monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-key")
        monkeypatch.setenv("REDIS_HOST", "localhost")
        monkeypatch.setenv("REDIS_PORT", "6379")
        result = validate_environment()
        assert not any("not a valid integer" in w for w in result.warnings)

    def test_non_integer_port_warns(self, monkeypatch):
        monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-key")
        monkeypatch.setenv("REDIS_HOST", "localhost")
        monkeypatch.setenv("REDIS_PORT", "abc")
        result = validate_environment()
        assert any("not a valid integer" in w for w in result.warnings)

    def test_out_of_range_port_warns(self, monkeypatch):
        monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-key")
        monkeypatch.setenv("REDIS_HOST", "localhost")
        monkeypatch.setenv("REDIS_PORT", "99999")
        result = validate_environment()
        assert any("outside the valid port range" in w for w in result.warnings)

    def test_zero_port_warns(self, monkeypatch):
        monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-key")
        monkeypatch.setenv("REDIS_HOST", "localhost")
        monkeypatch.setenv("REDIS_PORT", "0")
        result = validate_environment()
        assert any("outside the valid port range" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# ValidationResult dataclass
# ---------------------------------------------------------------------------


class TestValidationResult:
    def test_default_empty(self):
        r = ValidationResult()
        assert r.errors == []
        assert r.warnings == []

    def test_mutable_lists(self):
        r = ValidationResult()
        r.errors.append("err1")
        r.warnings.append("warn1")
        assert r.errors == ["err1"]
        assert r.warnings == ["warn1"]

    def test_independent_instances(self):
        """Each instance should have its own lists."""
        r1 = ValidationResult()
        r2 = ValidationResult()
        r1.errors.append("err")
        assert r2.errors == []
