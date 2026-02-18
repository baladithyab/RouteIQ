"""
Leader Election for HA Config Sync
====================================

Provides a database-backed lease lock for coordinating config sync
across multiple replicas in High Availability deployments.

This module implements a simple leader election mechanism using a
single-row table in PostgreSQL. Only one replica can hold the lease
at a time, and the lease is automatically renewed periodically.

Design Principles:
- Lease-based with automatic renewal for crash recovery
- Database-backed (works with PostgreSQL, with SQLite fallback for testing)
- Optional and backwards compatible (disabled by default in non-HA mode)
- Does not hold lock during long I/O operations
"""

import asyncio
import os
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from litellm._logging import verbose_proxy_logger


# =============================================================================
# Configuration
# =============================================================================

# Default settings
DEFAULT_LEASE_SECONDS = 30
DEFAULT_RENEW_INTERVAL_SECONDS = 10
DEFAULT_LOCK_NAME = "config_sync"

# HA Mode settings
HA_MODE_SINGLE = "single"
HA_MODE_LEADER_ELECTION = "leader_election"
DEFAULT_HA_MODE = HA_MODE_SINGLE


def _get_env_bool(key: str, default: bool) -> bool:
    """Get boolean from environment variable."""
    value = os.getenv(key, "").lower()
    if value in ("true", "1", "yes"):
        return True
    if value in ("false", "0", "no"):
        return False
    return default


def _get_env_int(key: str, default: int) -> int:
    """Get integer from environment variable."""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def get_ha_mode() -> str:
    """
    Get the HA mode configuration.

    Returns:
        HA mode string: 'single' or 'leader_election'
    """
    mode = os.getenv("LLMROUTER_HA_MODE", "").lower().strip()
    if mode in (HA_MODE_SINGLE, HA_MODE_LEADER_ELECTION):
        return mode

    # Legacy support: auto-enable leader_election if DATABASE_URL is set
    # and the legacy env var is true
    legacy_enabled = _get_env_bool(
        "LLMROUTER_CONFIG_SYNC_LEADER_ELECTION_ENABLED",
        default=False,
    )
    if legacy_enabled and os.getenv("DATABASE_URL"):
        return HA_MODE_LEADER_ELECTION

    # Default: if DATABASE_URL is set and HA_MODE not explicitly set,
    # still default to 'single' for backwards compatibility
    return DEFAULT_HA_MODE


def get_leader_election_config() -> dict:
    """
    Get leader election configuration from environment.

    Returns:
        Dictionary with leader election configuration
    """
    ha_mode = get_ha_mode()
    enabled = ha_mode == HA_MODE_LEADER_ELECTION

    return {
        "enabled": enabled,
        "ha_mode": ha_mode,
        "lease_seconds": _get_env_int(
            "LLMROUTER_CONFIG_SYNC_LEASE_SECONDS",
            DEFAULT_LEASE_SECONDS,
        ),
        "renew_interval_seconds": _get_env_int(
            "LLMROUTER_CONFIG_SYNC_RENEW_INTERVAL_SECONDS",
            DEFAULT_RENEW_INTERVAL_SECONDS,
        ),
        "lock_name": os.getenv(
            "LLMROUTER_CONFIG_SYNC_LOCK_NAME",
            DEFAULT_LOCK_NAME,
        ),
    }


# =============================================================================
# Leader Election Lock Status
# =============================================================================


@dataclass
class LeaseInfo:
    """Information about a lease lock."""

    lock_name: str
    holder_id: str
    acquired_at: datetime
    expires_at: datetime
    is_leader: bool
    generation: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "lock_name": self.lock_name,
            "holder_id": self.holder_id,
            "acquired_at": self.acquired_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "is_leader": self.is_leader,
            "generation": self.generation,
        }


# =============================================================================
# SQL Schema for Leader Election
# =============================================================================


LEADER_ELECTION_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS config_sync_leader (
    lock_name VARCHAR(255) PRIMARY KEY,
    holder_id VARCHAR(255) NOT NULL,
    acquired_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    generation BIGINT NOT NULL DEFAULT 0,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_config_sync_leader_expires
    ON config_sync_leader(expires_at);
"""


# =============================================================================
# Database-backed Leader Election
# =============================================================================


class LeaderElection:
    """
    Database-backed leader election using lease locks.

    Uses a single-row table with optimistic locking to ensure
    only one replica can hold the leadership at a time.

    The leader is determined by whoever successfully acquires
    or renews the lease. Leases expire automatically if not renewed,
    allowing recovery from crashed leaders.
    """

    def __init__(
        self,
        lock_name: str = DEFAULT_LOCK_NAME,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
        renew_interval_seconds: int = DEFAULT_RENEW_INTERVAL_SECONDS,
        database_url: str | None = None,
        holder_id: str | None = None,
    ):
        """
        Initialize the leader election.

        Args:
            lock_name: Name of the lock (allows multiple independent locks)
            lease_seconds: How long a lease is valid before expiration
            renew_interval_seconds: How often to renew the lease
            database_url: PostgreSQL connection string (uses DATABASE_URL if not provided)
            holder_id: Unique ID for this instance (generated if not provided)
        """
        self.lock_name = lock_name
        self.lease_seconds = lease_seconds
        self.renew_interval_seconds = renew_interval_seconds
        self.holder_id = holder_id or self._generate_holder_id()
        self._db_url = database_url or os.getenv("DATABASE_URL")

        # State
        self._is_leader = False
        self._lease_expires_at: datetime | None = None
        self._generation: int = 0
        self._consecutive_renewal_failures: int = 0
        self._stop_event = threading.Event()
        self._renew_thread: threading.Thread | None = None
        self._on_leadership_change: Callable[[bool], None] | None = None
        self._last_renewal_error: str | None = None

    def _generate_holder_id(self) -> str:
        """Generate a unique holder ID for this instance."""
        # Combine hostname (if available) with UUID for debugging
        import socket

        try:
            hostname = socket.gethostname()[:20]
        except Exception:
            hostname = "unknown"

        short_uuid = str(uuid.uuid4())[:8]
        return f"{hostname}-{short_uuid}"

    @property
    def is_leader(self) -> bool:
        """Check if this instance is currently the leader."""
        if not self._is_leader:
            return False

        # Also check if lease is still valid
        if self._lease_expires_at is None:
            return False

        return datetime.now(timezone.utc) < self._lease_expires_at

    @property
    def database_configured(self) -> bool:
        """Check if database is configured."""
        return self._db_url is not None

    async def ensure_table_exists(self) -> bool:
        """
        Create the leader election table if it doesn't exist.

        Returns:
            True if table exists or was created, False on error
        """
        if not self._db_url:
            return False

        try:
            import asyncpg

            conn = await asyncpg.connect(self._db_url)
            try:
                await conn.execute(LEADER_ELECTION_TABLE_SQL)
                verbose_proxy_logger.debug("Leader election: Table created/verified")
                return True
            finally:
                await conn.close()
        except ImportError:
            verbose_proxy_logger.warning("Leader election: asyncpg not installed")
            return False
        except Exception as e:
            verbose_proxy_logger.error(f"Leader election: Error creating table: {e}")
            return False

    async def try_acquire(self) -> bool:
        """
        Try to acquire the leadership lease.

        Uses an atomic INSERT with ON CONFLICT to handle race conditions.
        If the existing lease has expired, takes over leadership.

        Returns:
            True if this instance is now the leader, False otherwise
        """
        if not self._db_url:
            # No database configured, assume single instance mode
            # Set a synthetic lease that never expires (far future)
            from datetime import timedelta

            self._is_leader = True
            self._lease_expires_at = datetime.now(timezone.utc) + timedelta(days=365)
            self._generation += 1
            self._consecutive_renewal_failures = 0
            return True

        try:
            import asyncpg
            from datetime import timedelta

            conn = await asyncpg.connect(self._db_url)
            try:
                now = datetime.now(timezone.utc)
                expires_at = now + timedelta(seconds=self.lease_seconds)

                # Atomic upsert that only succeeds if:
                # 1. No lock exists (INSERT)
                # 2. Lock exists but is expired (UPDATE with WHERE)
                # 3. Lock exists and we already hold it (UPDATE with WHERE)
                # Generation is incremented atomically on every acquisition.
                result = await conn.fetchrow(
                    """
                    INSERT INTO config_sync_leader (lock_name, holder_id, acquired_at, expires_at, generation)
                    VALUES ($1, $2, $3, $4, 1)
                    ON CONFLICT (lock_name) DO UPDATE SET
                        holder_id = EXCLUDED.holder_id,
                        acquired_at = EXCLUDED.acquired_at,
                        expires_at = EXCLUDED.expires_at,
                        generation = config_sync_leader.generation + 1
                    WHERE
                        config_sync_leader.expires_at < $3
                        OR config_sync_leader.holder_id = $2
                    RETURNING holder_id, expires_at, generation
                    """,
                    self.lock_name,
                    self.holder_id,
                    now,
                    expires_at,
                )

                if result and result["holder_id"] == self.holder_id:
                    was_leader = self._is_leader
                    self._is_leader = True
                    self._lease_expires_at = result["expires_at"]
                    self._generation = result["generation"]
                    self._consecutive_renewal_failures = 0
                    self._last_renewal_error = None

                    if not was_leader:
                        verbose_proxy_logger.info(
                            f"Leader election: Acquired leadership "
                            f"(holder={self.holder_id}, expires={expires_at})"
                        )
                        if self._on_leadership_change:
                            self._on_leadership_change(True)

                    return True
                else:
                    # Could not acquire - someone else holds a valid lease
                    was_leader = self._is_leader
                    self._is_leader = False
                    self._lease_expires_at = None

                    if was_leader:
                        verbose_proxy_logger.info(
                            f"Leader election: Lost leadership (holder={self.holder_id})"
                        )
                        if self._on_leadership_change:
                            self._on_leadership_change(False)

                    return False

            finally:
                await conn.close()

        except ImportError:
            verbose_proxy_logger.warning(
                "Leader election: asyncpg not installed, assuming single instance"
            )
            self._is_leader = True
            return True

        except Exception as e:
            self._last_renewal_error = str(e)
            verbose_proxy_logger.error(f"Leader election: Error acquiring lease: {e}")
            # On error, don't change leadership status (favour stability)
            return self._is_leader

    async def renew(self) -> bool:
        """
        Renew the leadership lease if we are the leader.

        Returns:
            True if lease was renewed, False otherwise
        """
        if not self._is_leader:
            return False

        # Renewal is just re-acquisition
        return await self.try_acquire()

    async def release(self) -> bool:
        """
        Release the leadership lease voluntarily.

        Returns:
            True if lease was released, False on error
        """
        if not self._is_leader:
            return True

        if not self._db_url:
            self._is_leader = False
            return True

        try:
            import asyncpg

            conn = await asyncpg.connect(self._db_url)
            try:
                # Only delete if we hold the lease
                result = await conn.execute(
                    """
                    DELETE FROM config_sync_leader
                    WHERE lock_name = $1 AND holder_id = $2
                    """,
                    self.lock_name,
                    self.holder_id,
                )

                self._is_leader = False
                self._lease_expires_at = None

                verbose_proxy_logger.info(
                    f"Leader election: Released leadership (holder={self.holder_id})"
                )

                if self._on_leadership_change:
                    self._on_leadership_change(False)

                return "DELETE 1" in result

            finally:
                await conn.close()

        except Exception as e:
            verbose_proxy_logger.error(f"Leader election: Error releasing lease: {e}")
            return False

    async def get_current_leader(self) -> LeaseInfo | None:
        """
        Get information about the current leader.

        Returns:
            LeaseInfo if there is a valid leader, None otherwise
        """
        if not self._db_url:
            if self._is_leader:
                return LeaseInfo(
                    lock_name=self.lock_name,
                    holder_id=self.holder_id,
                    acquired_at=datetime.now(timezone.utc),
                    expires_at=self._lease_expires_at or datetime.now(timezone.utc),
                    is_leader=True,
                )
            return None

        try:
            import asyncpg

            conn = await asyncpg.connect(self._db_url)
            try:
                now = datetime.now(timezone.utc)
                row = await conn.fetchrow(
                    """
                    SELECT lock_name, holder_id, acquired_at, expires_at
                    FROM config_sync_leader
                    WHERE lock_name = $1 AND expires_at > $2
                    """,
                    self.lock_name,
                    now,
                )

                if row:
                    return LeaseInfo(
                        lock_name=row["lock_name"],
                        holder_id=row["holder_id"],
                        acquired_at=row["acquired_at"],
                        expires_at=row["expires_at"],
                        is_leader=row["holder_id"] == self.holder_id,
                    )
                return None

            finally:
                await conn.close()

        except Exception as e:
            verbose_proxy_logger.error(f"Leader election: Error getting leader: {e}")
            return None

    def _renew_loop(self):
        """Background thread to periodically renew the lease."""
        verbose_proxy_logger.debug(
            f"Leader election: Renewal thread started "
            f"(interval={self.renew_interval_seconds}s)"
        )

        while not self._stop_event.is_set():
            try:
                # Run async renewal in sync context
                loop = asyncio.new_event_loop()
                try:
                    renewed = loop.run_until_complete(self.renew())
                finally:
                    loop.close()

                if renewed:
                    self._consecutive_renewal_failures = 0
                elif self._is_leader:
                    # Renewal returned False while we think we're leader
                    self._consecutive_renewal_failures += 1
                    verbose_proxy_logger.warning(
                        f"Leader election: Renewal failed "
                        f"(consecutive_failures="
                        f"{self._consecutive_renewal_failures})"
                    )

                    # Auto-demote after 2 consecutive failures
                    if self._consecutive_renewal_failures >= 2:
                        verbose_proxy_logger.error(
                            "Leader election: Auto-demoting after "
                            f"{self._consecutive_renewal_failures} "
                            "consecutive renewal failures"
                        )
                        self._is_leader = False
                        self._lease_expires_at = None
                        if self._on_leadership_change:
                            self._on_leadership_change(False)

            except Exception as e:
                verbose_proxy_logger.error(f"Leader election: Renewal error: {e}")
                if self._is_leader:
                    self._consecutive_renewal_failures += 1
                    verbose_proxy_logger.warning(
                        f"Leader election: Renewal exception "
                        f"(consecutive_failures="
                        f"{self._consecutive_renewal_failures})"
                    )
                    if self._consecutive_renewal_failures >= 2:
                        verbose_proxy_logger.error(
                            "Leader election: Auto-demoting after "
                            f"{self._consecutive_renewal_failures} "
                            "consecutive renewal failures (exception)"
                        )
                        self._is_leader = False
                        self._lease_expires_at = None
                        if self._on_leadership_change:
                            self._on_leadership_change(False)

            # Wait for next renewal interval
            self._stop_event.wait(self.renew_interval_seconds)

        verbose_proxy_logger.debug("Leader election: Renewal thread stopped")

    def start_renewal(self, on_leadership_change: Callable[[bool], None] | None = None):
        """
        Start the background lease renewal thread.

        Args:
            on_leadership_change: Callback when leadership status changes
        """
        self._on_leadership_change = on_leadership_change
        self._stop_event.clear()

        self._renew_thread = threading.Thread(
            target=self._renew_loop,
            daemon=True,
            name="leader-election-renewal",
        )
        self._renew_thread.start()

    def stop_renewal(self):
        """Stop the background lease renewal thread."""
        self._stop_event.set()
        if self._renew_thread and self._renew_thread.is_alive():
            self._renew_thread.join(timeout=5)

    def get_status(self) -> dict:
        """Get the current leader election status."""
        return {
            "holder_id": self.holder_id,
            "is_leader": self.is_leader,
            "lock_name": self.lock_name,
            "lease_seconds": self.lease_seconds,
            "renew_interval_seconds": self.renew_interval_seconds,
            "lease_expires_at": (
                self._lease_expires_at.isoformat() if self._lease_expires_at else None
            ),
            "generation": self._generation,
            "consecutive_renewal_failures": self._consecutive_renewal_failures,
            "database_configured": self.database_configured,
            "last_renewal_error": self._last_renewal_error,
            "renewal_thread_alive": self._renew_thread is not None
            and self._renew_thread.is_alive(),
        }

    def validate_fencing_token(self, token: int) -> bool:
        """Validate that a fencing token matches the current generation.

        Operations should include the current generation to ensure they're
        not executing with a stale leadership claim.

        Args:
            token: The fencing token (generation number) to validate.

        Returns:
            True if this instance is the leader and the token matches
            the current generation, False otherwise.
        """
        return self._is_leader and token == self._generation


# =============================================================================
# Singleton Instance
# =============================================================================


_leader_election: LeaderElection | None = None


def get_leader_election() -> LeaderElection | None:
    """
    Get the global leader election instance, if leader election is enabled.

    Returns:
        LeaderElection instance if enabled, None otherwise
    """
    global _leader_election

    config = get_leader_election_config()

    if not config["enabled"]:
        return None

    if _leader_election is None:
        _leader_election = LeaderElection(
            lock_name=config["lock_name"],
            lease_seconds=config["lease_seconds"],
            renew_interval_seconds=config["renew_interval_seconds"],
        )

    return _leader_election


def reset_leader_election() -> None:
    """Reset the leader election singleton. For testing only."""
    global _leader_election
    _leader_election = None


async def initialize_leader_election() -> LeaderElection | None:
    """
    Initialize leader election (create table and try initial acquisition).

    Returns:
        LeaderElection instance if enabled, None otherwise
    """
    election = get_leader_election()
    if election is None:
        verbose_proxy_logger.info("Leader election: Disabled (single instance mode)")
        return None

    # Ensure table exists
    await election.ensure_table_exists()

    # Try initial acquisition
    is_leader = await election.try_acquire()

    verbose_proxy_logger.info(
        f"Leader election: Initialized (holder={election.holder_id}, "
        f"is_leader={is_leader})"
    )

    return election


def shutdown_leader_election():
    """Shutdown leader election and release resources."""
    global _leader_election

    if _leader_election is not None:
        _leader_election.stop_renewal()

        # Try to release the lease (best effort)
        try:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(_leader_election.release())
            finally:
                loop.close()
        except Exception:
            pass

        _leader_election = None
