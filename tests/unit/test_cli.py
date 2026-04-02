"""
Tests for the RouteIQ CLI module.

Tests cover:
1. `routeiq version` output
2. `routeiq validate-config` with valid/invalid configs
3. `routeiq probe-services` (mocked)
4. Unknown/missing command shows help and exits
5. `routeiq start` delegates to startup
6. Argument parsing
"""

from __future__ import annotations

import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from litellm_llmrouter.cli import (
    main,
    _cmd_version,
    _cmd_validate_config,
    _cmd_probe_services,
)


# ============================================================================
# routeiq version
# ============================================================================


class TestVersion:
    """Test the version command."""

    def test_cmd_version_prints_version(self, capsys):
        """_cmd_version should print RouteIQ Gateway vX.Y.Z."""
        _cmd_version()
        captured = capsys.readouterr()
        assert "RouteIQ Gateway v" in captured.out

    def test_cmd_version_fallback(self, capsys):
        """When importlib.metadata.version fails, uses 0.0.0-dev."""
        with patch(
            "importlib.metadata.version",
            side_effect=Exception("not found"),
        ):
            _cmd_version()
        captured = capsys.readouterr()
        assert "v0.0.0-dev" in captured.out

    def test_version_via_main(self, capsys):
        """Invoking main() with 'version' should print version info."""
        with patch("sys.argv", ["routeiq", "version"]):
            main()
        captured = capsys.readouterr()
        assert "RouteIQ Gateway v" in captured.out


# ============================================================================
# routeiq validate-config
# ============================================================================


class TestValidateConfig:
    """Test the validate-config command."""

    def test_valid_yaml_config(self, capsys, tmp_path):
        """Valid YAML config should pass validation."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("model_list:\n  - model_name: gpt-4\n")

        import argparse

        args = argparse.Namespace(config=str(config_file))

        # Mock validate_environment to return no errors
        mock_result = MagicMock()
        mock_result.errors = []
        mock_result.warnings = []

        with patch(
            "litellm_llmrouter.cli.validate_environment",
            return_value=mock_result,
            create=True,
        ):
            with patch(
                "litellm_llmrouter.env_validation.validate_environment",
                return_value=mock_result,
            ):
                with pytest.raises(SystemExit) as exc_info:
                    _cmd_validate_config(args)
                assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert "YAML syntax: OK" in captured.out

    def test_missing_config_file(self, capsys, tmp_path):
        """Missing config file should print error and exit 1."""
        import argparse

        args = argparse.Namespace(config=str(tmp_path / "nonexistent.yaml"))

        with pytest.raises(SystemExit) as exc_info:
            _cmd_validate_config(args)
        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "ERROR" in captured.out

    def test_invalid_yaml_config(self, capsys, tmp_path):
        """Invalid YAML should exit with error."""
        config_file = tmp_path / "bad.yaml"
        config_file.write_text("{{invalid yaml!!")

        import argparse

        args = argparse.Namespace(config=str(config_file))

        with pytest.raises(SystemExit) as exc_info:
            _cmd_validate_config(args)
        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "ERROR" in captured.out

    def test_non_mapping_yaml(self, capsys, tmp_path):
        """YAML that parses but is not a dict should fail."""
        config_file = tmp_path / "list.yaml"
        config_file.write_text("- item1\n- item2\n")

        import argparse

        args = argparse.Namespace(config=str(config_file))

        with pytest.raises(SystemExit) as exc_info:
            _cmd_validate_config(args)
        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "not a valid YAML mapping" in captured.out

    def test_validate_config_via_main(self, capsys, tmp_path):
        """Test calling validate-config via main()."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("model_list:\n  - model_name: gpt-4\n")

        mock_result = MagicMock()
        mock_result.errors = []
        mock_result.warnings = []

        with patch(
            "sys.argv", ["routeiq", "validate-config", "--config", str(config_file)]
        ):
            with patch(
                "litellm_llmrouter.env_validation.validate_environment",
                return_value=mock_result,
            ):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 0


# ============================================================================
# routeiq probe-services
# ============================================================================


class TestProbeServices:
    """Test the probe-services command."""

    def test_probe_services_success(self, capsys):
        """Successful probe should print a table."""
        mock_table = "Service | Status\nRedis   | OK\n"

        # Patch asyncio.run at the top-level module since cli.py imports asyncio locally
        with patch("asyncio.run", return_value=mock_table):
            _cmd_probe_services()

        captured = capsys.readouterr()
        assert "Service" in captured.out or "Redis" in captured.out

    def test_probe_services_exception(self, capsys):
        """Exception during probing should exit 1."""
        with patch("asyncio.run", side_effect=RuntimeError("probe failed")):
            with pytest.raises(SystemExit) as exc_info:
                _cmd_probe_services()
            assert exc_info.value.code == 1


# ============================================================================
# No command / unknown command
# ============================================================================


class TestNoCommand:
    """Test behavior when no command is given."""

    def test_no_command_exits_with_error(self):
        """Running without a command should exit 1."""
        with patch("sys.argv", ["routeiq"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1


# ============================================================================
# routeiq start
# ============================================================================


class TestStartCommand:
    """Test the start command."""

    def test_start_delegates_to_startup(self):
        """routeiq start should call startup.main()."""
        with patch(
            "sys.argv", ["routeiq", "start", "--config", "test.yaml", "--port", "5000"]
        ):
            with patch("litellm_llmrouter.startup.main") as mock_startup:
                main()
                mock_startup.assert_called_once()

    def test_start_sets_env_var(self):
        """routeiq start should set LITELLM_CONFIG_PATH."""
        with patch("sys.argv", ["routeiq", "start", "--config", "my_config.yaml"]):
            with patch("litellm_llmrouter.startup.main"):
                original = os.environ.get("LITELLM_CONFIG_PATH")
                try:
                    main()
                    # Should be set (setdefault won't overwrite existing)
                    assert "LITELLM_CONFIG_PATH" in os.environ
                finally:
                    if original is not None:
                        os.environ["LITELLM_CONFIG_PATH"] = original
                    else:
                        os.environ.pop("LITELLM_CONFIG_PATH", None)

    def test_start_workers_argument(self):
        """routeiq start --workers 4 should set ROUTEIQ_WORKERS env var."""
        with patch("sys.argv", ["routeiq", "start", "--workers", "4"]):
            with patch("litellm_llmrouter.startup.main") as mock_startup:
                original = os.environ.get("ROUTEIQ_WORKERS")
                try:
                    main()
                    assert os.environ.get("ROUTEIQ_WORKERS") == "4"
                finally:
                    if original is not None:
                        os.environ["ROUTEIQ_WORKERS"] = original
                    else:
                        os.environ.pop("ROUTEIQ_WORKERS", None)


# ============================================================================
# Argument parsing
# ============================================================================


class TestArgParsing:
    """Test CLI argument parsing edge cases."""

    def test_start_default_port(self):
        """Default port should be 4000 (set via LITELLM_PORT env var)."""
        with patch("sys.argv", ["routeiq", "start"]):
            with patch("litellm_llmrouter.startup.main"):
                original = os.environ.get("LITELLM_PORT")
                try:
                    main()
                    assert os.environ.get("LITELLM_PORT") == "4000"
                finally:
                    if original is not None:
                        os.environ["LITELLM_PORT"] = original
                    else:
                        os.environ.pop("LITELLM_PORT", None)

    def test_start_default_config(self):
        """Default config should be config/config.yaml (set via LITELLM_CONFIG_PATH env var)."""
        with patch("sys.argv", ["routeiq", "start"]):
            with patch("litellm_llmrouter.startup.main"):
                original = os.environ.get("LITELLM_CONFIG_PATH")
                try:
                    # Remove any existing value so setdefault can set it
                    os.environ.pop("LITELLM_CONFIG_PATH", None)
                    main()
                    assert os.environ.get("LITELLM_CONFIG_PATH") == "config/config.yaml"
                finally:
                    if original is not None:
                        os.environ["LITELLM_CONFIG_PATH"] = original
                    else:
                        os.environ.pop("LITELLM_CONFIG_PATH", None)
