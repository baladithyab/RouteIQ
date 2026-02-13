"""Tests for /config/status endpoint."""

import dataclasses

from litellm_llmrouter.config_sync import ConfigSyncStatus, get_config_sync_status


def test_config_status_returns_dataclass():
    """get_config_sync_status returns a ConfigSyncStatus instance."""
    status = get_config_sync_status()
    assert isinstance(status, ConfigSyncStatus)


def test_config_status_has_expected_fields():
    """ConfigSyncStatus has all required fields."""
    status = get_config_sync_status()
    as_dict = dataclasses.asdict(status)
    expected_keys = {
        "config_source",
        "sync_enabled",
        "sync_interval_seconds",
        "last_sync_attempt",
        "last_sync_success",
        "last_sync_error",
        "config_version_hash",
        "model_count",
        "next_sync_at",
    }
    assert set(as_dict.keys()) == expected_keys


def test_config_status_defaults():
    """Default status has sensible values."""
    status = get_config_sync_status()
    assert isinstance(status.sync_enabled, bool)
    assert isinstance(status.sync_interval_seconds, int)
    assert isinstance(status.model_count, int)
