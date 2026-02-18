"""
Environment Variable Validation
================================

Validates environment variables on startup to catch misconfigurations early.
This module is advisory only — it logs warnings and errors but never raises
exceptions or prevents the gateway from starting.

Usage::

    from litellm_llmrouter.env_validation import validate_environment

    result = validate_environment()
    # result.errors  -> list[str]
    # result.warnings -> list[str]
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Holds errors and warnings produced by environment validation."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Known boolean env vars
# ---------------------------------------------------------------------------

_BOOLEAN_ENV_VARS: tuple[str, ...] = (
    "OTEL_ENABLED",
    "MCP_GATEWAY_ENABLED",
    "A2A_GATEWAY_ENABLED",
    "CONFIG_HOT_RELOAD",
    "POLICY_ENGINE_ENABLED",
    "MCP_SSE_TRANSPORT_ENABLED",
    "MCP_PROTOCOL_PROXY_ENABLED",
    "LLMROUTER_ALLOW_PICKLE_MODELS",
    "LLMROUTER_ENFORCE_SIGNED_MODELS",
    "ROUTEIQ_USE_PLUGIN_STRATEGY",
)

_VALID_BOOLEAN_VALUES: frozenset[str] = frozenset(
    {"true", "false", "1", "0", "yes", "no", ""}
)

# ---------------------------------------------------------------------------
# Known port env vars
# ---------------------------------------------------------------------------

_PORT_ENV_VARS: tuple[str, ...] = ("REDIS_PORT",)

# ---------------------------------------------------------------------------
# Known positive-integer env vars (workers, counts, etc.)
# ---------------------------------------------------------------------------

_POSITIVE_INT_ENV_VARS: tuple[str, ...] = ("ROUTEIQ_WORKERS",)

# ---------------------------------------------------------------------------
# Placeholder values that shouldn't appear in production admin keys
# ---------------------------------------------------------------------------

_PLACEHOLDER_TOKENS: tuple[str, ...] = ("changeme", "test")

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_environment() -> ValidationResult:
    """Validate known environment variables and return a summary.

    Returns a :class:`ValidationResult` with ``errors`` (critical issues like
    missing files) and ``warnings`` (potential problems like missing master key).

    The function **never** raises an exception — the gateway should always
    continue to start regardless of validation results.

    Set ``ROUTEIQ_SKIP_ENV_VALIDATION=true`` to skip all checks.
    """
    result = ValidationResult()

    # Allow skipping entirely (useful in CI / testing)
    if os.environ.get("ROUTEIQ_SKIP_ENV_VALIDATION", "").lower() in (
        "true",
        "1",
        "yes",
    ):
        logger.info("Environment validation skipped (ROUTEIQ_SKIP_ENV_VALIDATION)")
        return result

    _validate_master_key(result)
    _validate_config_path(result)
    _validate_database_url(result)
    _validate_redis(result)
    _validate_otel_endpoint(result)
    _validate_admin_api_keys(result)
    _validate_boolean_vars(result)
    _validate_port_vars(result)
    _validate_positive_int_vars(result)

    # Summary log
    n_errors = len(result.errors)
    n_warnings = len(result.warnings)
    logger.info(
        "Environment validation complete: %d error(s), %d warning(s)",
        n_errors,
        n_warnings,
    )

    return result


# ---------------------------------------------------------------------------
# Individual validators (private)
# ---------------------------------------------------------------------------


def _validate_master_key(result: ValidationResult) -> None:
    """Warn if LITELLM_MASTER_KEY is not set (security risk)."""
    if not os.environ.get("LITELLM_MASTER_KEY"):
        result.warnings.append(
            "LITELLM_MASTER_KEY is not set — admin endpoints are unprotected"
        )


def _validate_config_path(result: ValidationResult) -> None:
    """Error if LITELLM_CONFIG_PATH is set but the file does not exist."""
    path = os.environ.get("LITELLM_CONFIG_PATH")
    if path and not Path(path).is_file():
        result.errors.append(
            f"LITELLM_CONFIG_PATH is set to '{path}' but the file does not exist"
        )


def _validate_database_url(result: ValidationResult) -> None:
    """Warn if DATABASE_URL has an unrecognised scheme."""
    url = os.environ.get("DATABASE_URL")
    if url is None:
        return
    valid_prefixes = ("postgresql://", "postgres://", "sqlite://")
    if not url.startswith(valid_prefixes):
        result.warnings.append(
            f"DATABASE_URL does not start with a recognised scheme "
            f"(expected one of {', '.join(valid_prefixes)})"
        )


def _validate_redis(result: ValidationResult) -> None:
    """Warn if only one of REDIS_HOST / REDIS_PORT is set."""
    host = os.environ.get("REDIS_HOST")
    port = os.environ.get("REDIS_PORT")
    if host and not port:
        result.warnings.append(
            "REDIS_HOST is set but REDIS_PORT is not — "
            "Redis may not connect on the expected port"
        )
    if port and not host:
        result.warnings.append(
            "REDIS_PORT is set but REDIS_HOST is not — "
            "Redis may not connect to the expected host"
        )


def _validate_otel_endpoint(result: ValidationResult) -> None:
    """Warn if OTEL_EXPORTER_OTLP_ENDPOINT does not look like a URL."""
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint is None:
        return
    valid_prefixes = ("http://", "https://", "grpc://")
    if not endpoint.startswith(valid_prefixes):
        result.warnings.append(
            f"OTEL_EXPORTER_OTLP_ENDPOINT ('{endpoint}') does not start with "
            f"a recognised scheme (expected one of {', '.join(valid_prefixes)})"
        )


def _validate_admin_api_keys(result: ValidationResult) -> None:
    """Warn if ADMIN_API_KEYS contains obvious placeholder values."""
    raw = os.environ.get("ADMIN_API_KEYS")
    if raw is None:
        return
    keys = [k.strip() for k in raw.split(",")]
    for key in keys:
        lower = key.lower()
        for placeholder in _PLACEHOLDER_TOKENS:
            if placeholder in lower:
                result.warnings.append(
                    f"ADMIN_API_KEYS contains a placeholder-like value "
                    f"('{key}') — this is a security risk in production"
                )
                # Only warn once per key
                break


def _validate_boolean_vars(result: ValidationResult) -> None:
    """Warn for boolean env vars whose value is not recognisable."""
    for var in _BOOLEAN_ENV_VARS:
        value = os.environ.get(var)
        if value is not None and value.lower() not in _VALID_BOOLEAN_VALUES:
            result.warnings.append(
                f"{var} has unexpected value '{value}' "
                f"(expected one of: true, false, 1, 0, yes, no)"
            )


def _validate_port_vars(result: ValidationResult) -> None:
    """Warn for port env vars that are not valid integers."""
    for var in _PORT_ENV_VARS:
        value = os.environ.get(var)
        if value is None:
            continue
        try:
            port = int(value)
            if not (1 <= port <= 65535):
                result.warnings.append(
                    f"{var} value '{value}' is outside the valid port range (1-65535)"
                )
        except ValueError:
            result.warnings.append(f"{var} value '{value}' is not a valid integer")


def _validate_positive_int_vars(result: ValidationResult) -> None:
    """Warn for env vars that should be positive integers."""
    for var in _POSITIVE_INT_ENV_VARS:
        value = os.environ.get(var)
        if value is None:
            continue
        try:
            parsed = int(value)
            if parsed < 1:
                result.warnings.append(
                    f"{var} value '{value}' must be a positive integer (>= 1)"
                )
        except ValueError:
            result.warnings.append(f"{var} value '{value}' is not a valid integer")
