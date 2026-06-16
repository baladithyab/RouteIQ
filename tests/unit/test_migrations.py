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
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm_llmrouter.migrations import (
    _find_prisma_schema,
    _is_leader_migrations_enabled,
    _marker_exists,
    _run_as_leader,
    _wait_as_follower,
    _write_completion_marker,
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
        """Same-process fast-path: a co-located leader sets the Event."""
        reset_migration_state()
        from litellm_llmrouter import migrations

        # Simulate leader completing in background
        async def signal_later():
            await asyncio.sleep(0.05)
            migrations._migration_complete.set()

        asyncio.ensure_future(signal_later())
        # No durable marker available -> rely on the in-process Event fast-path.
        with patch(
            "litellm_llmrouter.migrations._marker_exists",
            new=AsyncMock(return_value=False),
        ):
            result = await _wait_as_follower(timeout=5, poll_interval=0.01)
        assert result is True

    async def test_timeout_raises(self):
        """Fail-closed: no Event, no durable marker -> raises TimeoutError."""
        reset_migration_state()
        with patch(
            "litellm_llmrouter.migrations._marker_exists",
            new=AsyncMock(return_value=False),
        ):
            with pytest.raises(TimeoutError):
                await _wait_as_follower(timeout=0.1, poll_interval=0.02)


# =============================================================================
# Durable cross-pod marker (RouteIQ-2166)
# =============================================================================
#
# These tests model a multi-pod deploy: a single shared "database" (the
# ``_FakeMarkerStore``) stands in for the cross-pod source of truth, while each
# pod has its OWN in-process ``asyncio.Event`` (the per-process fast-path that
# is NOT shared across pods). The bug being fixed: an ``asyncio.Event`` set on
# the leader pod is never visible to a non-leader pod, so the durable marker
# must be the thing the follower observes.


class _FakeConn:
    """Minimal asyncpg-connection stand-in backed by a shared dict store.

    Understands just enough SQL to emulate the marker table:
    - ``CREATE TABLE`` is a no-op (idempotent).
    - ``INSERT ... ON CONFLICT`` upserts ``version -> holder_id``.
    - ``SELECT version ... WHERE version = $1`` returns a row or ``None``.
    """

    def __init__(self, store: dict):
        self._store = store

    async def execute(self, sql: str, *args):
        if "INSERT INTO" in sql:
            version, holder = args[0], args[1]
            self._store[version] = holder
        # CREATE TABLE / anything else: no-op
        return "OK"

    async def fetchrow(self, sql: str, *args):
        if "SELECT version" in sql:
            version = args[0]
            if version in self._store:
                return {"version": version}
            return None
        return None


class _FakeAcquire:
    """Async context manager returned by ``pool.acquire()``."""

    def __init__(self, conn: "_FakeConn"):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class _FakeMarkerStore:
    """Stands in for the cross-pod database: one store shared by all pods."""

    def __init__(self):
        self.rows: dict[str, str | None] = {}

    def acquire(self):
        return _FakeAcquire(_FakeConn(self.rows))


def _patch_pool(store: _FakeMarkerStore):
    """Patch ``database.get_db_pool`` to return a pool over *store*.

    ``_write_completion_marker`` / ``_marker_exists`` both
    ``from .database import ... get_db_pool`` at call time, so patching the
    attribute on the ``database`` module is what takes effect.
    """
    return patch(
        "litellm_llmrouter.database.get_db_pool",
        new=AsyncMock(return_value=store),
    )


class TestWriteCompletionMarker:
    """Acceptance (a): leader writes the durable marker after migration."""

    async def test_writes_marker_row(self):
        store = _FakeMarkerStore()
        with _patch_pool(store):
            wrote = await _write_completion_marker("v1")
        assert wrote is True
        assert "v1" in store.rows

    async def test_returns_false_when_pool_unavailable(self):
        """No DB pool -> cannot write marker -> returns False (no crash)."""
        with patch(
            "litellm_llmrouter.database.get_db_pool",
            new=AsyncMock(return_value=None),
        ):
            wrote = await _write_completion_marker("v1")
        assert wrote is False

    async def test_iam_region_error_is_fail_loud(self):
        """Boot-critical IAM region misconfig must propagate, not be swallowed."""
        from litellm_llmrouter.database import IamRegionUnresolvedError

        with patch(
            "litellm_llmrouter.database.get_db_pool",
            new=AsyncMock(side_effect=IamRegionUnresolvedError("no region")),
        ):
            with pytest.raises(IamRegionUnresolvedError):
                await _write_completion_marker("v1")

    async def test_leader_writes_marker_after_migration(self, tmp_path):
        """End-to-end leader path: a successful migration writes the marker."""
        schema = tmp_path / "proxy" / "schema.prisma"
        schema.parent.mkdir(parents=True)
        schema.write_text("// schema")

        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ok", stderr=""
        )
        store = _FakeMarkerStore()
        with patch(
            "litellm_llmrouter.migrations._find_prisma_schema",
            return_value=str(schema),
        ):
            with patch(
                "litellm_llmrouter.migrations._run_prisma_db_push",
                return_value=completed,
            ):
                with _patch_pool(store):
                    reset_migration_state()
                    result = await _run_as_leader()
        assert result is True
        # Durable marker written for the default version.
        assert len(store.rows) == 1

    async def test_leader_does_not_write_marker_on_failure(self, tmp_path):
        """A failed migration must NOT leave a marker non-leaders could observe."""
        schema = tmp_path / "proxy" / "schema.prisma"
        schema.parent.mkdir(parents=True)
        schema.write_text("// schema")

        store = _FakeMarkerStore()
        with patch(
            "litellm_llmrouter.migrations._find_prisma_schema",
            return_value=str(schema),
        ):
            with patch(
                "litellm_llmrouter.migrations._run_prisma_db_push",
                side_effect=subprocess.CalledProcessError(1, "prisma", stderr="boom"),
            ):
                with _patch_pool(store):
                    reset_migration_state()
                    result = await _run_as_leader()
        assert result is False
        assert store.rows == {}  # no durable marker on failure


class TestMarkerExists:
    async def test_true_when_marker_present(self):
        store = _FakeMarkerStore()
        store.rows["v1"] = "leader-pod"
        with _patch_pool(store):
            assert await _marker_exists("v1") is True

    async def test_false_when_marker_absent(self):
        store = _FakeMarkerStore()
        with _patch_pool(store):
            assert await _marker_exists("v1") is False

    async def test_false_when_pool_unavailable(self):
        with patch(
            "litellm_llmrouter.database.get_db_pool",
            new=AsyncMock(return_value=None),
        ):
            assert await _marker_exists("v1") is False

    async def test_iam_region_error_is_fail_loud(self):
        from litellm_llmrouter.database import IamRegionUnresolvedError

        with patch(
            "litellm_llmrouter.database.get_db_pool",
            new=AsyncMock(side_effect=IamRegionUnresolvedError("no region")),
        ):
            with pytest.raises(IamRegionUnresolvedError):
                await _marker_exists("v1")


class TestCrossPodBarrier:
    """Acceptance (b): a NON-leader observes completion via the durable marker,
    with NO shared in-process Event between the two pods."""

    async def test_follower_observes_durable_marker_without_event(self):
        """The follower's own Event stays UNSET; it must proceed solely because
        the leader (on another 'pod') wrote the durable marker.

        This is the heart of RouteIQ-2166: an ``asyncio.Event`` is per-process,
        so the follower MUST be unblocked by the cross-pod marker, not an Event.
        """
        from litellm_llmrouter import migrations

        # Shared cross-pod store; this pod's own Event is cleared and never set.
        store = _FakeMarkerStore()
        reset_migration_state()
        assert not migrations._migration_complete.is_set()

        # Leader pod (separate code path, no shared Event) writes the marker
        # mid-poll. We simulate it by writing into the shared store after a beat.
        async def leader_writes_marker():
            await asyncio.sleep(0.05)
            with _patch_pool(store):
                await _write_completion_marker(migrations._marker_version())

        with _patch_pool(store):
            writer = asyncio.ensure_future(leader_writes_marker())
            result = await _wait_as_follower(timeout=5, poll_interval=0.01)
            await writer

        assert result is True
        # Proven cross-pod: the follower's in-process Event was never set.
        assert not migrations._migration_complete.is_set()
        # And the durable marker is what it observed.
        assert len(store.rows) == 1

    async def test_follower_observes_preexisting_marker(self):
        """Leader on another pod already finished before this pod started:
        the marker is present from the first poll; Event never involved."""
        from litellm_llmrouter import migrations

        store = _FakeMarkerStore()
        store.rows[migrations._marker_version()] = "other-leader-pod"
        reset_migration_state()

        with _patch_pool(store):
            result = await _wait_as_follower(timeout=5, poll_interval=0.01)

        assert result is True
        assert not migrations._migration_complete.is_set()

    async def test_follower_times_out_when_marker_never_written(self):
        """Acceptance (c): leader pod never writes the marker (e.g. it died) and
        this pod's Event is never set -> fail-closed TimeoutError."""
        from litellm_llmrouter import migrations

        store = _FakeMarkerStore()  # stays empty
        reset_migration_state()

        with _patch_pool(store):
            with pytest.raises(TimeoutError):
                await _wait_as_follower(timeout=0.1, poll_interval=0.02)
        assert not migrations._migration_complete.is_set()


class TestMarkerVersion:
    """The marker key is the migration/schema version so a schema change forces
    non-leaders to wait for a FRESH marker instead of observing a stale one."""

    def test_default_version(self):
        from litellm_llmrouter import migrations

        with patch.dict(os.environ, {}, clear=True):
            assert migrations._marker_version() == migrations._DEFAULT_MARKER_VERSION

    def test_override_via_env(self):
        from litellm_llmrouter import migrations

        with patch.dict(os.environ, {"ROUTEIQ_MIGRATION_VERSION": "schema-2026-06"}):
            assert migrations._marker_version() == "schema-2026-06"

    def test_blank_override_falls_back_to_default(self):
        from litellm_llmrouter import migrations

        with patch.dict(os.environ, {"ROUTEIQ_MIGRATION_VERSION": "   "}):
            assert migrations._marker_version() == migrations._DEFAULT_MARKER_VERSION

    async def test_follower_ignores_stale_version_marker(self):
        """A marker for a DIFFERENT version must not satisfy this follower."""
        store = _FakeMarkerStore()
        store.rows["old-schema-version"] = "old-leader"  # stale, different key
        reset_migration_state()

        with patch.dict(
            os.environ, {"ROUTEIQ_MIGRATION_VERSION": "new-schema-version"}
        ):
            with _patch_pool(store):
                with pytest.raises(TimeoutError):
                    await _wait_as_follower(timeout=0.1, poll_interval=0.02)


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
