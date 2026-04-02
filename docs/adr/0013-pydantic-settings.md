# ADR-0013: Pydantic Settings for Typed Configuration

**Status**: Accepted
**Date**: 2026-04-02
**Decision Makers**: RouteIQ Core Team

## Context

### Problem: Configuration Sprawl

RouteIQ has 124 environment variables spread across 20+ prefixes:

| Prefix | Count | Example |
|--------|------:|---------|
| `LITELLM_*` | ~40 | `LITELLM_MASTER_KEY`, `LITELLM_CONFIG_PATH` |
| `REDIS_*` | 5 | `REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD` |
| `OTEL_*` | 8 | `OTEL_ENABLED`, `OTEL_EXPORTER_OTLP_ENDPOINT` |
| `MCP_*` | 6 | `MCP_GATEWAY_ENABLED`, `MCP_SSE_TRANSPORT_ENABLED` |
| `LLMROUTER_*` | 15 | `LLMROUTER_ALLOW_PICKLE_MODELS` |
| `ROUTEIQ_*` | 12 | `ROUTEIQ_USE_PLUGIN_STRATEGY`, `ROUTEIQ_WORKERS` |
| `POLICY_*` | 3 | `POLICY_ENGINE_ENABLED`, `POLICY_CONFIG_PATH` |
| `ADMIN_*` | 3 | `ADMIN_API_KEYS`, `ADMIN_AUTH_ENABLED` |
| `A2A_*` | 2 | `A2A_GATEWAY_ENABLED` |
| `CONFIG_*` | 4 | `CONFIG_HOT_RELOAD`, `CONFIG_S3_BUCKET` |
| `DATABASE_*` | 1 | `DATABASE_URL` |
| Other | ~25 | Various one-off env vars |

Of these 124 env vars, approximately 70 are undocumented.

Each module reads its own env vars independently using `os.getenv()` with
scattered defaults. This creates:

1. **No startup validation**: Invalid values (e.g., `REDIS_PORT=abc`) cause
   runtime errors, not startup errors.
2. **Inconsistent defaults**: The same concept may have different defaults
   in different modules.
3. **No discoverability**: Users must read source code to find all env vars.
4. **Type unsafety**: All values are strings until manually cast.

## Decision

Create a single `GatewaySettings(BaseSettings)` class using Pydantic v2's
Settings management, consolidating all configuration into ~35 typed fields
with startup validation.

### Implementation

```python
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator

class GatewaySettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ROUTEIQ_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    # Core
    config_path: str = Field("config/config.yaml", description="Path to config YAML")
    workers: int = Field(1, ge=1, le=32, description="Uvicorn workers")
    port: int = Field(4000, ge=1, le=65535)

    # Database
    database_url: str | None = Field(None, description="PostgreSQL connection string")
    db_pool_min: int = Field(2, ge=1)
    db_pool_max: int = Field(10, ge=1)

    # Redis
    redis_host: str = Field("localhost")
    redis_port: int = Field(6379)
    redis_password: str | None = Field(None)
    redis_ssl: bool = Field(False)

    # Routing
    routing_profile: str = Field("auto", pattern="^(auto|eco|premium|free|reasoning)$")
    use_plugin_strategy: bool = Field(True)
    centroid_routing: bool = Field(True)
    centroid_warmup: bool = Field(False)

    # Security
    admin_api_keys: list[str] = Field(default_factory=list)
    admin_auth_enabled: bool = Field(True)
    enforce_signed_models: bool = Field(True)
    allow_pickle_models: bool = Field(False)

    # Features
    mcp_gateway_enabled: bool = Field(False)
    a2a_gateway_enabled: bool = Field(False)
    policy_engine_enabled: bool = Field(False)
    admin_ui_enabled: bool = Field(False)

    # Observability
    otel_enabled: bool = Field(True)
    otel_endpoint: str | None = Field(None)
    otel_service_name: str = Field("routeiq-gateway")
```

### Validation at Startup

```python
try:
    settings = GatewaySettings()
except ValidationError as e:
    logger.error("Configuration validation failed:\n%s", e)
    sys.exit(1)
```

Invalid configuration fails loud at startup, not at first use.

### Migration Path

1. Create `GatewaySettings` with all fields
2. Modules gradually migrate from `os.getenv()` to `settings.field`
3. Old env var names are supported via Pydantic aliases
4. Deprecation warnings for old env var names

## Consequences

### Positive

- **Startup validation**: All configuration validated before serving requests.
- **Type safety**: `int`, `bool`, `str`, `list` fields with proper parsing.
- **Discoverability**: Single class documents all configuration options.
- **IDE support**: Autocomplete and type checking for settings access.
- **Fewer env vars**: Consolidate ~124 env vars to ~35 typed settings.

### Negative

- **Migration effort**: All modules must be updated to use the settings object.
- **Backward compatibility**: Old env var names must be supported via aliases.
- **Pydantic Settings dependency**: Adds `pydantic-settings` to core deps.

## Alternatives Considered

### Alternative A: Keep os.getenv()

- **Pros**: No migration; works today.
- **Cons**: All problems listed in Context persist.
- **Rejected**: Unscalable as env var count grows.

### Alternative B: dataclasses + Manual Validation

- **Pros**: No extra dependency.
- **Cons**: Must reimplement env var parsing, validation, defaults, nesting.
- **Rejected**: Pydantic Settings does this better with less code.

## References

- `src/litellm_llmrouter/env_validation.py` — Current advisory validation
- Pydantic Settings: https://docs.pydantic.dev/latest/concepts/pydantic_settings/
- [ADR-0006: Security Hardening](0006-security-hardening-defaults.md)
