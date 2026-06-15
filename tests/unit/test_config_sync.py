"""
Unit tests for ConfigSyncManager AppConfig poll-interval honoring.

The AppConfig data-plane (``get_latest_configuration``) returns a
``NextPollIntervalInSeconds`` hint on every poll. RouteIQ honors it for the next
sleep (clamped to a floor) in preference to the static ``sync_interval`` — see
``_poll_appconfig_if_changed`` + ``_next_poll_delay``. These tests prove the
server value is captured and used.
"""

import io
import sys
from contextlib import contextmanager
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from litellm_llmrouter.config_sync import (
    _APPCONFIG_MIN_POLL_INTERVAL_SECONDS,
    ConfigSyncManager,
)


@contextmanager
def _patched_boto3(client):
    """Install a fake ``boto3`` module so ``import boto3`` inside the poll works.

    boto3 is an optional dependency (dev extra only); the AppConfig poll imports
    it lazily inside the function, so we stub ``sys.modules['boto3']`` rather than
    patching ``boto3.client`` (which would require boto3 to be installed).
    """
    fake = ModuleType("boto3")
    fake.client = MagicMock(return_value=client)
    saved = sys.modules.get("boto3")
    sys.modules["boto3"] = fake
    try:
        yield
    finally:
        if saved is not None:
            sys.modules["boto3"] = saved
        else:
            del sys.modules["boto3"]


def _make_appconfig_manager(tmp_path) -> ConfigSyncManager:
    """Build a manager with AppConfig forced on, no env/settings coupling."""
    manager = ConfigSyncManager(
        local_config_path=str(tmp_path / "config.yaml"),
        sync_interval_seconds=300,  # deliberately large so server hint is distinct
    )
    # Force AppConfig active without depending on env/settings resolution.
    manager.s3_sync_enabled = False
    manager.gcs_sync_enabled = False
    manager.appconfig_enabled = True
    manager.appconfig_application = "app-id"
    manager.appconfig_environment = "env-id"
    manager.appconfig_profile = "profile-id"
    manager.appconfig_poll_interval_seconds = 60
    manager.appconfig_sync_enabled = True
    return manager


def _appconfig_client(*, next_interval, body=b"", version="v1") -> MagicMock:
    """A boto3 'appconfigdata' client mock returning the given poll response."""
    client = MagicMock()
    client.start_configuration_session.return_value = {
        "InitialConfigurationToken": "tok-0"
    }
    client.get_latest_configuration.return_value = {
        "NextPollConfigurationToken": "tok-1",
        "NextPollIntervalInSeconds": next_interval,
        "Configuration": io.BytesIO(body),
        "VersionLabel": version,
    }
    return client


class TestAppConfigPollIntervalHonored:
    """The server-returned NextPollIntervalInSeconds drives the next sleep."""

    def test_poll_captures_server_next_interval(self, tmp_path):
        manager = _make_appconfig_manager(tmp_path)
        client = _appconfig_client(next_interval=42)

        with _patched_boto3(client):
            manager._poll_appconfig_if_changed()

        assert manager._appconfig_next_poll_interval == 42

    def test_next_poll_delay_uses_server_value_not_static(self, tmp_path):
        manager = _make_appconfig_manager(tmp_path)
        client = _appconfig_client(next_interval=42)

        with _patched_boto3(client):
            manager._poll_appconfig_if_changed()

        # The loop must sleep the SERVER value (42), not the static 300.
        assert manager._next_poll_delay() == 42
        assert manager.sync_interval == 300

    def test_server_interval_clamped_to_floor(self, tmp_path):
        """A too-small server value is clamped to the floor, never a busy-loop."""
        manager = _make_appconfig_manager(tmp_path)
        client = _appconfig_client(next_interval=1)

        with _patched_boto3(client):
            manager._poll_appconfig_if_changed()

        assert (
            manager._appconfig_next_poll_interval
            == _APPCONFIG_MIN_POLL_INTERVAL_SECONDS
        )
        assert manager._next_poll_delay() == _APPCONFIG_MIN_POLL_INTERVAL_SECONDS

    def test_interval_captured_even_when_config_unchanged(self, tmp_path):
        """Empty body == unchanged, but the server interval is still honored."""
        manager = _make_appconfig_manager(tmp_path)
        client = _appconfig_client(next_interval=90, body=b"")

        with _patched_boto3(client):
            changed = manager._poll_appconfig_if_changed()

        assert changed is False  # empty body -> unchanged
        assert manager._appconfig_next_poll_interval == 90
        assert manager._next_poll_delay() == 90

    def test_next_poll_delay_falls_back_to_static_before_first_poll(self, tmp_path):
        """Before any AppConfig poll, the static interval is used."""
        manager = _make_appconfig_manager(tmp_path)

        assert manager._appconfig_next_poll_interval is None
        assert manager._next_poll_delay() == 300

    def test_next_poll_delay_static_when_appconfig_disabled(self, tmp_path):
        """S3/GCS-only (AppConfig off) always uses the static interval."""
        manager = ConfigSyncManager(
            local_config_path=str(tmp_path / "config.yaml"),
            sync_interval_seconds=120,
        )
        manager.appconfig_sync_enabled = False
        # Even a stale hint must be ignored when AppConfig is not the source.
        manager._appconfig_next_poll_interval = 5

        assert manager._next_poll_delay() == 120

    def test_missing_server_interval_keeps_static(self, tmp_path):
        """If the data plane omits the hint, fall back to the static interval."""
        manager = _make_appconfig_manager(tmp_path)
        client = _appconfig_client(next_interval=None)

        with _patched_boto3(client):
            manager._poll_appconfig_if_changed()

        assert manager._appconfig_next_poll_interval is None
        assert manager._next_poll_delay() == 300


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
