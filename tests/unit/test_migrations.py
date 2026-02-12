"""
Unit Tests for migrations.py
==============================

Tests for leader-election-based database migration support:
- Feature gate (ROUTEIQ_LEADER_MIGRATIONS env var)
- Leader path: schema discovery, prisma db push execution, error handling
- Non-leader path: waiting for leader signal, timeout behaviour
- reset_migration_state for test isolation
"""

import asyncio
import os
import subprocess
from unittest.mock import MagicMock, patch

from litellm_llmrouter.migrations import (
    _find_prisma_schema,
    _is_leader_migrations_enabled,
    _run_as_leader,
    _wait_as_follower,
    reset_migration_state,
    run_migrations_if_leader,
)


# =============================================================================
# Feature gate
# =============================================================================


class TestIsLeaderMigrationsEnabled:
    def test_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            assert _is_leader_migrations_enabled() is False

    def test_disabled_when_false(self):
        with patch.dict(os.environ, {"ROUTEIQ_LEADER_MIGRATIONS": "false"}):
            assert _is_leader_migrations_enabled() is False

    def test_enabled_when_true(self):
        with patch.dict(os.environ, {"ROUTEIQ_LEADER_MIGRATIONS": "true"}):
            assert _is_leader_migrations_enabled() is True

    def test_enabled_when_one(self):
        with patch.dict(os.environ, {"ROUTEIQ_LEADER_MIGRATIONS": "1"}):
            assert _is_leader_migrations_enabled() is True

    def test_enabled_when_yes(self):
        with patch.dict(os.environ, {"ROUTEIQ_LEADER_MIGRATIONS": "yes"}):
            assert _is_leader_migrations_enabled() is True

    def test_case_insensitive(self):
        with patch.dict(os.environ, {"ROUTEIQ_LEADER_MIGRATIONS": "TRUE"}):
            assert _is_leader_migrations_enabled() is True


# =============================================================================
# Schema discovery
# =============================================================================


class TestFindPrismaSchema:
    def test_finds_schema_when_exists(self, tmp_path):
        schema_file = tmp_path / "proxy" / "schema.prisma"
        schema_file.parent.mkdir(parents=True)
        schema_file.write_text("// prisma schema")

        mock_litellm = MagicMock()
        mock_litellm.__file__ = str(tmp_path / "__init__.py")

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            result = _find_prisma_schema()
            assert result == str(schema_file)

    def test_returns_none_when_missing(self, tmp_path):
        mock_litellm = MagicMock()
        mock_litellm.__file__ = str(tmp_path / "__init__.py")

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            result = _find_prisma_schema()
            assert result is None

    def test_returns_none_on_import_error(self):
        with patch.dict("sys.modules", {"litellm": None}):
            result = _find_prisma_schema()
            assert result is None


# =============================================================================
# run_migrations_if_leader (top-level)
# =============================================================================


class TestRunMigrationsIfLeader:
    async def test_noop_when_disabled(self):
        """Feature disabled -> returns True immediately."""
        with patch.dict(os.environ, {"ROUTEIQ_LEADER_MIGRATIONS": "false"}):
            result = await run_migrations_if_leader(is_leader=True)
            assert result is True

    async def test_noop_when_no_database_url(self):
        """Feature enabled but no DATABASE_URL -> returns True."""
        env = {"ROUTEIQ_LEADER_MIGRATIONS": "true"}
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("DATABASE_URL", None)
            result = await run_migrations_if_leader(is_leader=True)
            assert result is True

    async def test_leader_path_called_when_leader(self):
        env = {
            "ROUTEIQ_LEADER_MIGRATIONS": "true",
            "DATABASE_URL": "postgresql://localhost/test",
        }
        with patch.dict(os.environ, env):
            with patch(
                "litellm_llmrouter.migrations._run_as_leader",
                return_value=True,
            ) as mock_leader:
                result = await run_migrations_if_leader(is_leader=True)
                assert result is True
                mock_leader.assert_awaited_once()

    async def test_follower_path_called_when_not_leader(self):
        env = {
            "ROUTEIQ_LEADER_MIGRATIONS": "true",
            "DATABASE_URL": "postgresql://localhost/test",
        }
        with patch.dict(os.environ, env):
            with patch(
                "litellm_llmrouter.migrations._wait_as_follower",
                return_value=True,
            ) as mock_follower:
                result = await run_migrations_if_leader(
                    is_leader=False, timeout=5, poll_interval=1
                )
                assert result is True
                mock_follower.assert_awaited_once_with(timeout=5, poll_interval=1)


# =============================================================================
# Leader path
# =============================================================================


class TestRunAsLeader:
    async def test_success(self, tmp_path):
        schema = tmp_path / "proxy" / "schema.prisma"
        schema.parent.mkdir(parents=True)
        schema.write_text("// schema")

        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ok", stderr=""
        )
        with patch(
            "litellm_llmrouter.migrations._find_prisma_schema",
            return_value=str(schema),
        ):
            with patch(
                "litellm_llmrouter.migrations._run_prisma_db_push",
                return_value=completed,
            ):
                reset_migration_state()
                result = await _run_as_leader()
                assert result is True

    async def test_schema_not_found(self):
        with patch(
            "litellm_llmrouter.migrations._find_prisma_schema",
            return_value=None,
        ):
            reset_migration_state()
            result = await _run_as_leader()
            assert result is True  # graceful skip

    async def test_prisma_failure(self, tmp_path):
        schema = tmp_path / "proxy" / "schema.prisma"
        schema.parent.mkdir(parents=True)
        schema.write_text("// schema")

        with patch(
            "litellm_llmrouter.migrations._find_prisma_schema",
            return_value=str(schema),
        ):
            with patch(
                "litellm_llmrouter.migrations._run_prisma_db_push",
                side_effect=subprocess.CalledProcessError(1, "prisma", stderr="error"),
            ):
                reset_migration_state()
                result = await _run_as_leader()
                assert result is False

    async def test_prisma_timeout(self, tmp_path):
        schema = tmp_path / "proxy" / "schema.prisma"
        schema.parent.mkdir(parents=True)
        schema.write_text("// schema")

        with patch(
            "litellm_llmrouter.migrations._find_prisma_schema",
            return_value=str(schema),
        ):
            with patch(
                "litellm_llmrouter.migrations._run_prisma_db_push",
                side_effect=subprocess.TimeoutExpired("prisma", 90),
            ):
                reset_migration_state()
                result = await _run_as_leader()
                assert result is False

    async def test_prisma_not_installed(self, tmp_path):
        schema = tmp_path / "proxy" / "schema.prisma"
        schema.parent.mkdir(parents=True)
        schema.write_text("// schema")

        with patch(
            "litellm_llmrouter.migrations._find_prisma_schema",
            return_value=str(schema),
        ):
            with patch(
                "litellm_llmrouter.migrations._run_prisma_db_push",
                side_effect=FileNotFoundError("prisma"),
            ):
                reset_migration_state()
                result = await _run_as_leader()
                assert result is False


# =============================================================================
# Follower path
# =============================================================================


class TestWaitAsFollower:
    async def test_proceeds_when_leader_signals(self):
        reset_migration_state()
        from litellm_llmrouter import migrations

        # Simulate leader completing in background
        async def signal_later():
            await asyncio.sleep(0.05)
            migrations._migration_complete.set()

        asyncio.ensure_future(signal_later())
        result = await _wait_as_follower(timeout=5, poll_interval=1)
        assert result is True

    async def test_timeout_proceeds_anyway(self):
        reset_migration_state()
        result = await _wait_as_follower(timeout=0.1, poll_interval=0.05)
        assert result is True  # proceeds even on timeout


# =============================================================================
# reset_migration_state
# =============================================================================


class TestResetMigrationState:
    async def test_clears_event(self):
        from litellm_llmrouter import migrations

        migrations._migration_complete.set()
        reset_migration_state()
        assert not migrations._migration_complete.is_set()


# =============================================================================
# Startup integration (run_leader_migrations_if_enabled)
# =============================================================================


class TestRunLeaderMigrationsIfEnabled:
    def test_noop_when_disabled(self):
        """When ROUTEIQ_LEADER_MIGRATIONS is not set, does nothing."""
        from litellm_llmrouter.startup import run_leader_migrations_if_enabled

        with patch.dict(os.environ, {"ROUTEIQ_LEADER_MIGRATIONS": "false"}):
            with patch(
                "litellm_llmrouter.migrations.run_migrations_if_leader"
            ) as mock_run:
                run_leader_migrations_if_enabled()
                mock_run.assert_not_called()

    def test_runs_when_enabled(self):
        """When enabled, initialises leader election and runs migrations."""
        from litellm_llmrouter.startup import run_leader_migrations_if_enabled

        mock_election = MagicMock()
        mock_election.is_leader = True

        async def mock_init():
            return mock_election

        async def mock_migrate(is_leader):
            return True

        with patch.dict(os.environ, {"ROUTEIQ_LEADER_MIGRATIONS": "true"}):
            with patch(
                "litellm_llmrouter.leader_election.initialize_leader_election",
                side_effect=mock_init,
            ):
                with patch(
                    "litellm_llmrouter.migrations.run_migrations_if_leader",
                    side_effect=mock_migrate,
                ) as mock_run:
                    run_leader_migrations_if_enabled()
                    mock_run.assert_called_once_with(True)

    def test_handles_no_leader_election(self):
        """When leader election returns None, defaults to is_leader=True."""
        from litellm_llmrouter.startup import run_leader_migrations_if_enabled

        async def mock_init():
            return None

        async def mock_migrate(is_leader):
            return True

        with patch.dict(os.environ, {"ROUTEIQ_LEADER_MIGRATIONS": "true"}):
            with patch(
                "litellm_llmrouter.leader_election.initialize_leader_election",
                side_effect=mock_init,
            ):
                with patch(
                    "litellm_llmrouter.migrations.run_migrations_if_leader",
                    side_effect=mock_migrate,
                ) as mock_run:
                    run_leader_migrations_if_enabled()
                    mock_run.assert_called_once_with(True)

    def test_handles_import_error(self):
        """Import errors are caught gracefully."""
        from litellm_llmrouter.startup import run_leader_migrations_if_enabled

        with patch.dict(os.environ, {"ROUTEIQ_LEADER_MIGRATIONS": "true"}):
            with patch(
                "litellm_llmrouter.leader_election.initialize_leader_election",
                side_effect=ImportError("no asyncpg"),
            ):
                # Should not raise
                run_leader_migrations_if_enabled()

    def test_handles_runtime_error(self):
        """Runtime errors are caught gracefully."""
        from litellm_llmrouter.startup import run_leader_migrations_if_enabled

        with patch.dict(os.environ, {"ROUTEIQ_LEADER_MIGRATIONS": "true"}):
            with patch(
                "litellm_llmrouter.leader_election.initialize_leader_election",
                side_effect=RuntimeError("db unreachable"),
            ):
                # Should not raise
                run_leader_migrations_if_enabled()
