"""
Unit tests for multi-worker support via ROUTEIQ_WORKERS.

Tests cover the ``resolve_worker_count()`` function in ``startup.py`` which:
- Reads ROUTEIQ_WORKERS env var
- Always allows multi-worker (plugin strategy is the only routing path)
- Handles invalid values gracefully (non-integer, zero, negative)
- Respects CLI --workers as a fallback when env var is not set

Tests also cover the ``_validate_positive_int_vars()`` function in
``env_validation.py`` for ROUTEIQ_WORKERS validation.
"""

from __future__ import annotations

import logging

import pytest

from litellm_llmrouter.env_validation import validate_environment
from litellm_llmrouter.startup import resolve_worker_count

# ======================================================================
# Env vars to clean before each test
# ======================================================================

_ENV_VARS = (
    "ROUTEIQ_WORKERS",
    "ROUTEIQ_USE_PLUGIN_STRATEGY",
    "ROUTEIQ_SKIP_ENV_VALIDATION",
    "LITELLM_MASTER_KEY",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Remove all env vars under test before each test case."""
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


# ======================================================================
# TestResolveWorkerCount — plugin strategy mode (default)
# ======================================================================


class TestResolveWorkerCountPluginMode:
    """Tests for resolve_worker_count when plugin strategy is active."""

    def test_default_is_one_worker(self) -> None:
        """Without ROUTEIQ_WORKERS or CLI flag, defaults to 1."""
        workers = resolve_worker_count()
        assert workers == 1

    def test_env_var_sets_workers(self, monkeypatch) -> None:
        """ROUTEIQ_WORKERS=4 results in 4 workers."""
        monkeypatch.setenv("ROUTEIQ_WORKERS", "4")
        workers = resolve_worker_count()
        assert workers == 4

    def test_env_var_overrides_cli(self, monkeypatch) -> None:
        """ROUTEIQ_WORKERS env var takes precedence over CLI --workers."""
        monkeypatch.setenv("ROUTEIQ_WORKERS", "8")
        workers = resolve_worker_count(cli_workers=2)
        assert workers == 8

    def test_cli_workers_used_when_no_env(self) -> None:
        """CLI --workers is used when ROUTEIQ_WORKERS is not set."""
        workers = resolve_worker_count(cli_workers=3)
        assert workers == 3

    def test_explicit_plugin_true(self, monkeypatch) -> None:
        """ROUTEIQ_USE_PLUGIN_STRATEGY=true allows multi-worker."""
        monkeypatch.setenv("ROUTEIQ_USE_PLUGIN_STRATEGY", "true")
        monkeypatch.setenv("ROUTEIQ_WORKERS", "4")
        workers = resolve_worker_count()
        assert workers == 4

    def test_plugin_strategy_yes_variant(self, monkeypatch) -> None:
        """ROUTEIQ_USE_PLUGIN_STRATEGY=yes allows multi-worker."""
        monkeypatch.setenv("ROUTEIQ_USE_PLUGIN_STRATEGY", "yes")
        monkeypatch.setenv("ROUTEIQ_WORKERS", "2")
        workers = resolve_worker_count()
        assert workers == 2

    def test_plugin_strategy_1_variant(self, monkeypatch) -> None:
        """ROUTEIQ_USE_PLUGIN_STRATEGY=1 allows multi-worker."""
        monkeypatch.setenv("ROUTEIQ_USE_PLUGIN_STRATEGY", "1")
        monkeypatch.setenv("ROUTEIQ_WORKERS", "6")
        workers = resolve_worker_count()
        assert workers == 6


# ======================================================================
# TestResolveWorkerCount — ROUTEIQ_USE_PLUGIN_STRATEGY is ignored
# ======================================================================


class TestResolveWorkerCountPluginStrategyIgnored:
    """Verify that ROUTEIQ_USE_PLUGIN_STRATEGY does not restrict worker count.

    Legacy monkey-patch mode has been removed.  The plugin strategy
    (RouteIQRoutingStrategy) is always used, so worker count is never
    capped.  Setting ``ROUTEIQ_USE_PLUGIN_STRATEGY=false`` is a no-op
    for worker resolution.
    """

    def test_false_flag_does_not_restrict_workers(self, monkeypatch) -> None:
        """ROUTEIQ_USE_PLUGIN_STRATEGY=false does NOT force workers=1."""
        monkeypatch.setenv("ROUTEIQ_USE_PLUGIN_STRATEGY", "false")
        monkeypatch.setenv("ROUTEIQ_WORKERS", "4")
        workers = resolve_worker_count()
        assert workers == 4

    def test_false_flag_with_cli_workers(self, monkeypatch) -> None:
        """CLI --workers is respected even when flag is false."""
        monkeypatch.setenv("ROUTEIQ_USE_PLUGIN_STRATEGY", "false")
        workers = resolve_worker_count(cli_workers=4)
        assert workers == 4

    def test_false_flag_default_is_one(self, monkeypatch) -> None:
        """Default is still 1 when no workers env or CLI arg is given."""
        monkeypatch.setenv("ROUTEIQ_USE_PLUGIN_STRATEGY", "false")
        workers = resolve_worker_count()
        assert workers == 1

    def test_false_flag_single_worker_no_warning(self, monkeypatch, caplog) -> None:
        """No legacy warning emitted for single worker."""
        monkeypatch.setenv("ROUTEIQ_USE_PLUGIN_STRATEGY", "false")
        monkeypatch.setenv("ROUTEIQ_WORKERS", "1")
        with caplog.at_level(logging.WARNING):
            workers = resolve_worker_count()
        assert workers == 1
        assert not any("legacy" in record.message.lower() for record in caplog.records)


# ======================================================================
# TestResolveWorkerCount — invalid values
# ======================================================================


class TestResolveWorkerCountInvalidValues:
    """Tests for graceful handling of invalid ROUTEIQ_WORKERS values."""

    def test_non_integer_defaults_to_one(self, monkeypatch) -> None:
        """Non-integer value falls back to 1."""
        monkeypatch.setenv("ROUTEIQ_WORKERS", "abc")
        workers = resolve_worker_count()
        assert workers == 1

    def test_zero_defaults_to_one(self, monkeypatch) -> None:
        """Zero value falls back to 1."""
        monkeypatch.setenv("ROUTEIQ_WORKERS", "0")
        workers = resolve_worker_count()
        assert workers == 1

    def test_negative_defaults_to_one(self, monkeypatch) -> None:
        """Negative value falls back to 1."""
        monkeypatch.setenv("ROUTEIQ_WORKERS", "-3")
        workers = resolve_worker_count()
        assert workers == 1

    def test_float_defaults_to_one(self, monkeypatch) -> None:
        """Float value falls back to 1."""
        monkeypatch.setenv("ROUTEIQ_WORKERS", "2.5")
        workers = resolve_worker_count()
        assert workers == 1

    def test_empty_string_defaults_to_one(self, monkeypatch) -> None:
        """Empty string falls back to 1."""
        monkeypatch.setenv("ROUTEIQ_WORKERS", "")
        workers = resolve_worker_count()
        assert workers == 1

    def test_non_integer_logs_warning(self, monkeypatch, caplog) -> None:
        """A warning is logged for non-integer values."""
        monkeypatch.setenv("ROUTEIQ_WORKERS", "not-a-number")
        with caplog.at_level(logging.WARNING):
            resolve_worker_count()
        assert any("not a valid integer" in r.message for r in caplog.records)

    def test_zero_logs_warning(self, monkeypatch, caplog) -> None:
        """A warning is logged for zero value."""
        monkeypatch.setenv("ROUTEIQ_WORKERS", "0")
        with caplog.at_level(logging.WARNING):
            resolve_worker_count()
        assert any("invalid" in r.message.lower() for r in caplog.records)

    def test_cli_workers_none_uses_default(self) -> None:
        """cli_workers=None uses default of 1."""
        workers = resolve_worker_count(cli_workers=None)
        assert workers == 1

    def test_cli_workers_zero_uses_default(self) -> None:
        """cli_workers=0 uses default of 1."""
        workers = resolve_worker_count(cli_workers=0)
        assert workers == 1

    def test_cli_workers_negative_uses_default(self) -> None:
        """cli_workers=-1 uses default of 1."""
        workers = resolve_worker_count(cli_workers=-1)
        assert workers == 1


# ======================================================================
# TestEnvValidation — ROUTEIQ_WORKERS
# ======================================================================


class TestEnvValidationWorkers:
    """Tests for ROUTEIQ_WORKERS in env_validation.py."""

    def test_valid_workers_no_warning(self, monkeypatch) -> None:
        """Valid ROUTEIQ_WORKERS=4 produces no warning."""
        monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-test")
        monkeypatch.setenv("ROUTEIQ_WORKERS", "4")
        result = validate_environment()
        assert not any("ROUTEIQ_WORKERS" in w for w in result.warnings)

    def test_one_worker_no_warning(self, monkeypatch) -> None:
        """ROUTEIQ_WORKERS=1 produces no warning."""
        monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-test")
        monkeypatch.setenv("ROUTEIQ_WORKERS", "1")
        result = validate_environment()
        assert not any("ROUTEIQ_WORKERS" in w for w in result.warnings)

    def test_non_integer_warns(self, monkeypatch) -> None:
        """ROUTEIQ_WORKERS=abc produces a warning."""
        monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-test")
        monkeypatch.setenv("ROUTEIQ_WORKERS", "abc")
        result = validate_environment()
        assert any("ROUTEIQ_WORKERS" in w for w in result.warnings)
        assert any("not a valid integer" in w for w in result.warnings)

    def test_zero_warns(self, monkeypatch) -> None:
        """ROUTEIQ_WORKERS=0 produces a warning."""
        monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-test")
        monkeypatch.setenv("ROUTEIQ_WORKERS", "0")
        result = validate_environment()
        assert any("ROUTEIQ_WORKERS" in w for w in result.warnings)
        assert any("positive integer" in w for w in result.warnings)

    def test_negative_warns(self, monkeypatch) -> None:
        """ROUTEIQ_WORKERS=-5 produces a warning."""
        monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-test")
        monkeypatch.setenv("ROUTEIQ_WORKERS", "-5")
        result = validate_environment()
        assert any("ROUTEIQ_WORKERS" in w for w in result.warnings)
        assert any("positive integer" in w for w in result.warnings)

    def test_unset_no_warning(self, monkeypatch) -> None:
        """Unset ROUTEIQ_WORKERS produces no warning."""
        monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-test")
        result = validate_environment()
        assert not any("ROUTEIQ_WORKERS" in w for w in result.warnings)
