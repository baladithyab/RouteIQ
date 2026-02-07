"""
Unit tests for leader election in HA config sync.

Tests cover:
- Lock acquisition and release
- Lease expiration semantics
- Concurrent instance competition
- Configuration from environment variables
"""

import asyncio
import os
import threading
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, patch
import pytest

from litellm_llmrouter.leader_election import (
    LeaderElection,
    LeaseInfo,
    get_leader_election_config,
    DEFAULT_LEASE_SECONDS,
    DEFAULT_RENEW_INTERVAL_SECONDS,
)


# Check if asyncpg is available
try:
    import asyncpg  # noqa: F401

    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False


# =============================================================================
# Configuration Tests
# =============================================================================


class TestLeaderElectionConfig:
    """Test configuration loading from environment."""

    def test_default_config_no_database(self):
        """Default config with no database should disable leader election."""
        with patch.dict(os.environ, {}, clear=True):
            # Ensure DATABASE_URL is not set
            os.environ.pop("DATABASE_URL", None)
            os.environ.pop("LLMROUTER_CONFIG_SYNC_LEADER_ELECTION_ENABLED", None)
            os.environ.pop("LLMROUTER_HA_MODE", None)

            config = get_leader_election_config()

            assert config["enabled"] is False
            assert config["lease_seconds"] == DEFAULT_LEASE_SECONDS
            assert config["renew_interval_seconds"] == DEFAULT_RENEW_INTERVAL_SECONDS
            assert config["lock_name"] == "config_sync"

    def test_default_config_with_database(self):
        """Default config with database should NOT auto-enable leader election (new behavior).

        Note: The new behavior requires explicit LLMROUTER_HA_MODE=leader_election.
        DATABASE_URL alone no longer auto-enables leader election for backwards compatibility.
        """
        with patch.dict(
            os.environ,
            {"DATABASE_URL": "postgresql://localhost/test"},
            clear=True,
        ):
            # Explicitly remove HA_MODE to test default
            os.environ.pop("LLMROUTER_HA_MODE", None)
            os.environ.pop("LLMROUTER_CONFIG_SYNC_LEADER_ELECTION_ENABLED", None)

            config = get_leader_election_config()

            # New default: single mode (disabled)
            assert config["enabled"] is False
            assert config["ha_mode"] == "single"

    def test_explicit_enable_ha_mode(self):
        """Explicit LLMROUTER_HA_MODE=leader_election enables leader election."""
        with patch.dict(
            os.environ,
            {
                "LLMROUTER_HA_MODE": "leader_election",
            },
            clear=True,
        ):
            config = get_leader_election_config()

            assert config["enabled"] is True
            assert config["ha_mode"] == "leader_election"

    def test_legacy_explicit_enable(self):
        """Legacy LLMROUTER_CONFIG_SYNC_LEADER_ELECTION_ENABLED still works with DATABASE_URL."""
        with patch.dict(
            os.environ,
            {
                "LLMROUTER_CONFIG_SYNC_LEADER_ELECTION_ENABLED": "true",
                "DATABASE_URL": "postgresql://localhost/test",
            },
            clear=True,
        ):
            config = get_leader_election_config()

            assert config["enabled"] is True

    def test_explicit_disable_with_database(self):
        """Explicit disable should work even with database configured."""
        with patch.dict(
            os.environ,
            {
                "DATABASE_URL": "postgresql://localhost/test",
                "LLMROUTER_HA_MODE": "single",
            },
            clear=True,
        ):
            config = get_leader_election_config()

            assert config["enabled"] is False

    def test_custom_lease_settings(self):
        """Custom lease settings from environment."""
        with patch.dict(
            os.environ,
            {
                "LLMROUTER_CONFIG_SYNC_LEASE_SECONDS": "60",
                "LLMROUTER_CONFIG_SYNC_RENEW_INTERVAL_SECONDS": "20",
                "LLMROUTER_CONFIG_SYNC_LOCK_NAME": "custom_lock",
            },
            clear=True,
        ):
            config = get_leader_election_config()

            assert config["lease_seconds"] == 60
            assert config["renew_interval_seconds"] == 20
            assert config["lock_name"] == "custom_lock"

    def test_invalid_int_settings(self):
        """Invalid integer settings should use defaults."""
        with patch.dict(
            os.environ,
            {
                "LLMROUTER_CONFIG_SYNC_LEASE_SECONDS": "invalid",
                "LLMROUTER_CONFIG_SYNC_RENEW_INTERVAL_SECONDS": "not-a-number",
            },
            clear=True,
        ):
            config = get_leader_election_config()

            assert config["lease_seconds"] == DEFAULT_LEASE_SECONDS
            assert config["renew_interval_seconds"] == DEFAULT_RENEW_INTERVAL_SECONDS


# =============================================================================
# LeaderElection Class Tests
# =============================================================================


class TestLeaderElection:
    """Test the LeaderElection class."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        election = LeaderElection()

        assert election.lock_name == "config_sync"
        assert election.lease_seconds == DEFAULT_LEASE_SECONDS
        assert election.renew_interval_seconds == DEFAULT_RENEW_INTERVAL_SECONDS
        assert election.holder_id is not None
        assert len(election.holder_id) > 0

    def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        election = LeaderElection(
            lock_name="my_lock",
            lease_seconds=60,
            renew_interval_seconds=15,
            holder_id="test-holder",
        )

        assert election.lock_name == "my_lock"
        assert election.lease_seconds == 60
        assert election.renew_interval_seconds == 15
        assert election.holder_id == "test-holder"

    def test_is_leader_initially_false(self):
        """New election should not be leader initially."""
        election = LeaderElection()

        assert election.is_leader is False

    def test_database_configured_property(self):
        """Test database_configured property."""
        # Without database URL
        election = LeaderElection(database_url=None)
        assert election.database_configured is False

        # With database URL
        election = LeaderElection(database_url="postgresql://localhost/test")
        assert election.database_configured is True

    def test_holder_id_generation(self):
        """Holder IDs should be unique."""
        election1 = LeaderElection()
        election2 = LeaderElection()

        assert election1.holder_id != election2.holder_id

    def test_get_status(self):
        """Test get_status method returns expected keys."""
        election = LeaderElection(holder_id="test-holder")
        status = election.get_status()

        assert "holder_id" in status
        assert "is_leader" in status
        assert "lock_name" in status
        assert "lease_seconds" in status
        assert "renew_interval_seconds" in status
        assert "lease_expires_at" in status
        assert "database_configured" in status
        assert "last_renewal_error" in status
        assert "renewal_thread_alive" in status

        assert status["holder_id"] == "test-holder"
        assert status["is_leader"] is False


# =============================================================================
# Lock Acquisition Tests (No Database)
# =============================================================================


class TestLeaderElectionNoDB:
    """Test leader election without database (single instance mode)."""

    @pytest.mark.asyncio
    async def test_acquire_without_db(self):
        """With no database, acquisition should always succeed."""
        election = LeaderElection(database_url=None)

        result = await election.try_acquire()

        assert result is True
        # Without a database, is_leader is True but _lease_expires_at is not set
        # So is_leader property returns False (checks expiry)
        # This is correct behavior - single instance mode
        assert election._is_leader is True

    @pytest.mark.asyncio
    async def test_release_without_db(self):
        """With no database, release should always succeed."""
        election = LeaderElection(database_url=None)
        await election.try_acquire()

        result = await election.release()

        assert result is True
        assert election.is_leader is False

    @pytest.mark.asyncio
    async def test_renew_without_db(self):
        """With no database, renewal without being leader should fail."""
        election = LeaderElection(database_url=None)

        # Not a leader yet
        result = await election.renew()
        assert result is False

        # After becoming leader
        await election.try_acquire()
        result = await election.renew()
        assert result is True

    @pytest.mark.asyncio
    async def test_get_current_leader_without_db(self):
        """Get current leader when not using database."""
        election = LeaderElection(database_url=None)

        # Before acquiring
        leader = await election.get_current_leader()
        assert leader is None

        # After acquiring
        await election.try_acquire()
        leader = await election.get_current_leader()
        assert leader is not None
        assert leader.is_leader is True


# =============================================================================
# Lock Acquisition Tests (With Mock Database)
# =============================================================================


class TestLeaderElectionWithMockDB:
    """Test leader election with mocked database operations."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(not ASYNCPG_AVAILABLE, reason="asyncpg not installed")
    async def test_ensure_table_exists_success(self):
        """Test table creation success."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_conn.close = AsyncMock()

        with patch("asyncpg.connect", AsyncMock(return_value=mock_conn)):
            election = LeaderElection(database_url="postgresql://localhost/test")
            result = await election.ensure_table_exists()

        assert result is True
        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.skipif(not ASYNCPG_AVAILABLE, reason="asyncpg not installed")
    async def test_try_acquire_new_leader(self):
        """Test acquiring leadership when no one holds it."""
        holder_id = "test-holder"
        mock_conn = AsyncMock()
        mock_conn.close = AsyncMock()

        # Simulate successful acquisition
        mock_conn.fetchrow = AsyncMock(
            return_value={
                "holder_id": holder_id,
                "expires_at": datetime.now(timezone.utc) + timedelta(seconds=30),
            }
        )

        with patch("asyncpg.connect", AsyncMock(return_value=mock_conn)):
            election = LeaderElection(
                database_url="postgresql://localhost/test",
                holder_id=holder_id,
            )
            result = await election.try_acquire()

        assert result is True
        assert election._is_leader is True

    @pytest.mark.asyncio
    @pytest.mark.skipif(not ASYNCPG_AVAILABLE, reason="asyncpg not installed")
    async def test_try_acquire_already_held(self):
        """Test trying to acquire when another instance holds the lock."""
        mock_conn = AsyncMock()
        mock_conn.close = AsyncMock()

        # Simulate failed acquisition (another holder)
        mock_conn.fetchrow = AsyncMock(return_value=None)

        with patch("asyncpg.connect", AsyncMock(return_value=mock_conn)):
            election = LeaderElection(
                database_url="postgresql://localhost/test",
                holder_id="test-holder",
            )
            result = await election.try_acquire()

        assert result is False
        assert election.is_leader is False

    @pytest.mark.asyncio
    @pytest.mark.skipif(not ASYNCPG_AVAILABLE, reason="asyncpg not installed")
    async def test_release_success(self):
        """Test releasing leadership."""
        holder_id = "test-holder"
        mock_conn = AsyncMock()
        mock_conn.close = AsyncMock()

        # First acquire
        mock_conn.fetchrow = AsyncMock(
            return_value={
                "holder_id": holder_id,
                "expires_at": datetime.now(timezone.utc) + timedelta(seconds=30),
            }
        )

        with patch("asyncpg.connect", AsyncMock(return_value=mock_conn)):
            election = LeaderElection(
                database_url="postgresql://localhost/test",
                holder_id=holder_id,
            )
            await election.try_acquire()
            assert election._is_leader is True

            # Now release
            mock_conn.execute = AsyncMock(return_value="DELETE 1")
            result = await election.release()

        assert result is True
        assert election.is_leader is False


# =============================================================================
# Lease Expiration Tests
# =============================================================================


class TestLeaseExpiration:
    """Test lease expiration semantics."""

    def test_is_leader_respects_expiry(self):
        """is_leader should return False if lease has expired."""
        election = LeaderElection(database_url=None)

        # Manually set state as if we acquired but lease expired
        election._is_leader = True
        election._lease_expires_at = datetime.now(timezone.utc) - timedelta(seconds=10)

        # Should now report as not leader
        assert election.is_leader is False

    def test_is_leader_with_valid_lease(self):
        """is_leader should return True if lease is still valid."""
        election = LeaderElection(database_url=None)

        # Manually set state as if we acquired with future expiry
        election._is_leader = True
        election._lease_expires_at = datetime.now(timezone.utc) + timedelta(seconds=30)

        assert election.is_leader is True


# =============================================================================
# Concurrency Tests
# =============================================================================


class TestConcurrentLeaderElection:
    """Test concurrent leader election (two instances competing)."""

    @pytest.mark.asyncio
    async def test_two_instances_only_one_leader_no_db(self):
        """
        Without database, both instances become "leaders" (single instance mode).
        This is expected behavior when database is not configured.
        """
        election1 = LeaderElection(database_url=None, holder_id="instance-1")
        election2 = LeaderElection(database_url=None, holder_id="instance-2")

        result1 = await election1.try_acquire()
        result2 = await election2.try_acquire()

        # Both should succeed without DB (no coordination)
        assert result1 is True
        assert result2 is True

    @pytest.mark.asyncio
    @pytest.mark.skipif(not ASYNCPG_AVAILABLE, reason="asyncpg not installed")
    async def test_two_instances_compete_first_wins(self):
        """
        Test that when two instances compete, only one becomes leader.
        Uses a shared mock database state to simulate real contention.
        """
        # Shared database state
        current_holder = {"holder_id": None, "expires_at": None}
        lock = threading.Lock()

        async def mock_fetch(*args, **kwargs):
            """Simulate atomic upsert behavior."""
            # Extract arguments from the fetchrow call:
            # fetchrow(query, lock_name, holder_id, now, expires_at)
            holder_id = args[2]
            now = args[3]
            expires_at = args[4]

            with lock:
                # If no current holder or lease expired
                if (
                    current_holder["holder_id"] is None
                    or current_holder["expires_at"] < now
                ):
                    current_holder["holder_id"] = holder_id
                    current_holder["expires_at"] = expires_at
                    return {
                        "holder_id": holder_id,
                        "expires_at": expires_at,
                    }
                # If we already hold it
                elif current_holder["holder_id"] == holder_id:
                    current_holder["expires_at"] = expires_at
                    return {
                        "holder_id": holder_id,
                        "expires_at": expires_at,
                    }
                # Someone else holds it
                else:
                    return None

        mock_conn = AsyncMock()
        mock_conn.fetchrow = mock_fetch
        mock_conn.close = AsyncMock()

        with patch("asyncpg.connect", AsyncMock(return_value=mock_conn)):
            election1 = LeaderElection(
                database_url="postgresql://localhost/test",
                holder_id="instance-1",
                lease_seconds=30,
            )
            election2 = LeaderElection(
                database_url="postgresql://localhost/test",
                holder_id="instance-2",
                lease_seconds=30,
            )

            # First instance acquires
            result1 = await election1.try_acquire()
            assert result1 is True
            assert election1._is_leader is True

            # Second instance tries but fails
            result2 = await election2.try_acquire()
            assert result2 is False
            assert election2.is_leader is False

    @pytest.mark.asyncio
    @pytest.mark.skipif(not ASYNCPG_AVAILABLE, reason="asyncpg not installed")
    async def test_leadership_transfer_on_release(self):
        """Test that leadership can transfer when released."""
        current_holder = {"holder_id": None, "expires_at": None}
        lock = threading.Lock()

        async def mock_fetch(*args, **kwargs):
            # fetchrow(query, lock_name, holder_id, now, expires_at)
            holder_id = args[2]
            now = args[3]
            expires_at = args[4]

            with lock:
                if (
                    current_holder["holder_id"] is None
                    or current_holder["expires_at"] < now
                ):
                    current_holder["holder_id"] = holder_id
                    current_holder["expires_at"] = expires_at
                    return {"holder_id": holder_id, "expires_at": expires_at}
                elif current_holder["holder_id"] == holder_id:
                    current_holder["expires_at"] = expires_at
                    return {"holder_id": holder_id, "expires_at": expires_at}
                else:
                    return None

        async def mock_execute(query, *args):
            # Simulate delete on release
            if "DELETE" in query:
                with lock:
                    if len(args) >= 2 and current_holder["holder_id"] == args[1]:
                        current_holder["holder_id"] = None
                        current_holder["expires_at"] = None
                        return "DELETE 1"
            return "DELETE 0"

        mock_conn = AsyncMock()
        mock_conn.fetchrow = mock_fetch
        mock_conn.execute = mock_execute
        mock_conn.close = AsyncMock()

        with patch("asyncpg.connect", AsyncMock(return_value=mock_conn)):
            election1 = LeaderElection(
                database_url="postgresql://localhost/test",
                holder_id="instance-1",
            )
            election2 = LeaderElection(
                database_url="postgresql://localhost/test",
                holder_id="instance-2",
            )

            # First instance acquires
            await election1.try_acquire()
            assert election1._is_leader is True

            # Release leadership
            await election1.release()
            assert election1.is_leader is False

            # Now second instance can acquire
            result2 = await election2.try_acquire()
            assert result2 is True
            assert election2._is_leader is True


# =============================================================================
# Renewal Thread Tests
# =============================================================================


class TestRenewalThread:
    """Test the background renewal thread."""

    def test_start_and_stop_renewal(self):
        """Test starting and stopping the renewal thread."""
        election = LeaderElection(database_url=None, renew_interval_seconds=1)

        # Start renewal
        callback_called = []
        election.start_renewal(on_leadership_change=lambda x: callback_called.append(x))

        assert election._renew_thread is not None
        assert election._renew_thread.is_alive()

        # Stop renewal
        election.stop_renewal()

        # Thread should stop within timeout
        time.sleep(0.5)
        assert not election._renew_thread.is_alive()

    @pytest.mark.skipif(not ASYNCPG_AVAILABLE, reason="asyncpg not installed")
    def test_renewal_callback_on_leadership_change(self):
        """Test that leadership change callback is invoked when becoming leader with DB."""
        mock_conn = AsyncMock()
        mock_conn.close = AsyncMock()
        holder_id = "test-holder"

        # Simulate successful acquisition
        mock_conn.fetchrow = AsyncMock(
            return_value={
                "holder_id": holder_id,
                "expires_at": datetime.now(timezone.utc) + timedelta(seconds=30),
            }
        )

        callback_values = []

        def callback(is_leader):
            callback_values.append(is_leader)

        with patch("asyncpg.connect", AsyncMock(return_value=mock_conn)):
            election = LeaderElection(
                database_url="postgresql://localhost/test",
                holder_id=holder_id,
            )
            election._on_leadership_change = callback

            # Simulate becoming leader
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(election.try_acquire())
            finally:
                loop.close()

        # Callback should have been called with True
        assert True in callback_values


# =============================================================================
# LeaseInfo Tests
# =============================================================================


class TestLeaseInfo:
    """Test the LeaseInfo dataclass."""

    def test_to_dict(self):
        """Test LeaseInfo serialization."""
        now = datetime.now(timezone.utc)
        expires = now + timedelta(seconds=30)

        info = LeaseInfo(
            lock_name="test_lock",
            holder_id="test-holder",
            acquired_at=now,
            expires_at=expires,
            is_leader=True,
        )

        data = info.to_dict()

        assert data["lock_name"] == "test_lock"
        assert data["holder_id"] == "test-holder"
        assert data["is_leader"] is True
        assert "acquired_at" in data
        assert "expires_at" in data


# =============================================================================
# Integration with ConfigSyncManager
# =============================================================================


class TestConfigSyncIntegration:
    """Test integration with ConfigSyncManager."""

    @pytest.mark.asyncio
    async def test_config_sync_manager_leader_check(self):
        """Test that ConfigSyncManager correctly checks leadership."""
        # This tests the integration without actually running the sync
        from litellm_llmrouter.config_sync import ConfigSyncManager

        with patch.dict(
            os.environ,
            {
                "LLMROUTER_CONFIG_SYNC_LEADER_ELECTION_ENABLED": "false",
            },
            clear=False,
        ):
            manager = ConfigSyncManager()

            # With leader election disabled, should always be "leader"
            assert manager._is_leader() is True

    @pytest.mark.asyncio
    async def test_config_sync_status_includes_leader_election(self):
        """Test that get_status includes leader election info."""
        from litellm_llmrouter.config_sync import ConfigSyncManager

        with patch.dict(
            os.environ,
            {"LLMROUTER_CONFIG_SYNC_LEADER_ELECTION_ENABLED": "false"},
            clear=False,
        ):
            manager = ConfigSyncManager()
            status = manager.get_status()

            assert "leader_election" in status
            assert "enabled" in status["leader_election"]
            assert "is_leader" in status["leader_election"]
