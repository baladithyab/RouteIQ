"""
Leader Election for HA Config Sync
====================================

Provides distributed leader election for coordinating config sync
across multiple replicas in High Availability deployments.

Supports multiple backends (see :class:`LeaderElectionBackend`):

- **Kubernetes Lease API** — preferred when running inside K8s pods.
  Uses ``coordination.k8s.io/v1`` Lease objects via the ``kubernetes``
  Python client (lazy-imported; not a core dependency).
- **Redis SETNX** — lightweight alternative for Docker Compose / bare-metal.
- **PostgreSQL** — original backend; works but adds unnecessary DB coupling.
- **None** — single-instance mode, no election needed.

Auto-detection (:func:`detect_leader_election_backend`) picks the best
backend based on environment signals.  An explicit override is available
via ``ROUTEIQ_LEADER_ELECTION_BACKEND``.

Design Principles:
- Lease-based with automatic renewal for crash recovery
- Pluggable backends with a common protocol
- Optional and backwards compatible (disabled by default in non-HA mode)
- Does not hold lock during long I/O operations
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Optional, Protocol

from litellm._logging import verbose_proxy_logger

logger = logging.getLogger(__name__)


# =============================================================================
# Backend Enum & Protocol
# =============================================================================


class LeaderElectionBackend(str, Enum):
    """Supported leader election backends."""

    POSTGRES = "postgres"
    REDIS = "redis"
    KUBERNETES = "kubernetes"
    NONE = "none"  # single-instance, no election needed


class LeaderElectionProtocol(Protocol):
    """Common interface implemented by all leader election backends."""

    async def try_acquire(self, identity: str, ttl_seconds: int = 30) -> bool: ...
    async def release(self, identity: str) -> None: ...
    async def get_current_leader(self) -> Optional[str]: ...


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
    # LLMROUTER_HA_MODE is NOT ROUTEIQ_-prefixed, so check env var first.
    mode = os.getenv("LLMROUTER_HA_MODE", "").lower().strip()
    if not mode:
        try:
            from litellm_llmrouter.settings import get_settings

            mode = get_settings().ha.mode.lower().strip()
        except Exception:
            pass
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
        "backend": detect_leader_election_backend(),
    }


def detect_leader_election_backend() -> LeaderElectionBackend:
    """Auto-detect the best leader election backend.

    Priority:
    1. ``ROUTEIQ_LEADER_ELECTION_BACKEND`` env var / settings (explicit override)
    2. Kubernetes if running in a pod (detected via ``KUBERNETES_SERVICE_HOST``)
    3. Redis if ``REDIS_HOST`` is configured
    4. PostgreSQL if ``DATABASE_URL`` is configured
    5. None (single-instance mode)
    """
    # 1. Explicit override — env var first, settings fallback
    explicit = os.getenv("ROUTEIQ_LEADER_ELECTION_BACKEND", "").strip().lower()
    if not explicit:
        try:
            from litellm_llmrouter.settings import get_settings

            explicit = get_settings().ha.leader_election_backend.strip().lower()
        except Exception:
            pass

    if explicit:
        try:
            return LeaderElectionBackend(explicit)
        except ValueError:
            logger.warning(
                "Unknown ROUTEIQ_LEADER_ELECTION_BACKEND=%r, falling back to auto-detect",
                explicit,
            )

    # 2. Kubernetes
    if os.getenv("KUBERNETES_SERVICE_HOST"):
        return LeaderElectionBackend.KUBERNETES

    # 3. Redis — check settings then env var
    redis_configured = False
    try:
        from litellm_llmrouter.settings import get_settings

        redis_configured = get_settings().redis_configured
    except Exception:
        redis_configured = bool(os.getenv("REDIS_HOST"))

    if redis_configured:
        return LeaderElectionBackend.REDIS

    # 4. PostgreSQL — check settings then env var
    pg_configured = False
    try:
        from litellm_llmrouter.settings import get_settings

        pg_configured = get_settings().postgres_configured
    except Exception:
        pg_configured = bool(os.getenv("DATABASE_URL"))

    if pg_configured:
        return LeaderElectionBackend.POSTGRES

    # 5. Single-instance
    return LeaderElectionBackend.NONE


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
# Kubernetes Lease API Backend
# =============================================================================


class K8sLeaseLeaderElection:
    """Leader election using K8s coordination.k8s.io/v1 Lease API.

    Uses the ``kubernetes`` Python client to create/renew a Lease object.
    The ``holderIdentity`` field identifies the current leader.

    Only works when running inside a K8s pod with appropriate RBAC::

        rules:
          - apiGroups: ["coordination.k8s.io"]
            resources: ["leases"]
            verbs: ["get", "create", "update"]

    The ``kubernetes`` package is imported lazily so it remains an optional
    dependency (included in the ``[cloud]`` extra).
    """

    def __init__(
        self,
        lease_name: str = "routeiq-leader",
        namespace: str | None = None,
    ):
        self.lease_name = lease_name
        self.namespace = namespace or self._detect_namespace()
        self._api: Any | None = None

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    @staticmethod
    def _detect_namespace() -> str:
        """Read the pod namespace from the mounted service-account token."""
        ns_path = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
        try:
            with open(ns_path) as fh:
                return fh.read().strip()
        except OSError:
            return os.getenv("POD_NAMESPACE", "default")

    def _get_api(self) -> Any:
        """Lazy-load the K8s CoordinationV1Api client."""
        if self._api is not None:
            return self._api
        try:
            from kubernetes import client as k8s_client
            from kubernetes import config as k8s_config

            # Try in-cluster first, fall back to kubeconfig for local dev
            try:
                k8s_config.load_incluster_config()
            except k8s_config.ConfigException:
                k8s_config.load_kube_config()

            self._api = k8s_client.CoordinationV1Api()
            return self._api
        except ImportError:
            raise ImportError(
                "The 'kubernetes' package is required for K8s leader election. "
                "Install it via: pip install routeiq[cloud] "
                "or: pip install kubernetes"
            )

    # -----------------------------------------------------------------
    # Public API (LeaderElectionProtocol)
    # -----------------------------------------------------------------

    async def try_acquire(self, identity: str, ttl_seconds: int = 30) -> bool:
        """Try to acquire leadership by creating/updating the Lease.

        Uses optimistic concurrency via ``resourceVersion``: if another pod
        updated the Lease between our read and write the API server returns
        409 Conflict and we return ``False``.
        """
        api = self._get_api()
        now = datetime.now(timezone.utc)

        try:
            from kubernetes import client as k8s_client
            from kubernetes.client.rest import ApiException
        except ImportError as exc:
            raise ImportError(
                "The 'kubernetes' package is required for K8s leader election."
            ) from exc

        try:
            # Try to read the existing Lease
            lease = await asyncio.to_thread(
                api.read_namespaced_lease,
                name=self.lease_name,
                namespace=self.namespace,
            )

            holder = lease.spec.holder_identity
            renew_time = lease.spec.renew_time
            duration = lease.spec.lease_duration_seconds or ttl_seconds

            # Check if the existing lease is still valid
            if holder and holder != identity and renew_time:
                expires_at = renew_time + timedelta(seconds=duration)
                if now < expires_at:
                    # Another pod holds a valid lease
                    return False

            # Lease is expired or we already hold it — update
            lease.spec.holder_identity = identity
            lease.spec.lease_duration_seconds = ttl_seconds
            lease.spec.renew_time = now
            lease.spec.acquire_time = lease.spec.acquire_time or now

            await asyncio.to_thread(
                api.replace_namespaced_lease,
                name=self.lease_name,
                namespace=self.namespace,
                body=lease,
            )
            verbose_proxy_logger.debug(
                "K8s leader election: acquired/renewed lease (holder=%s, namespace=%s)",
                identity,
                self.namespace,
            )
            return True

        except ApiException as exc:
            if exc.status == 404:
                # Lease does not exist — create it
                lease_body = k8s_client.V1Lease(
                    metadata=k8s_client.V1ObjectMeta(
                        name=self.lease_name,
                        namespace=self.namespace,
                    ),
                    spec=k8s_client.V1LeaseSpec(
                        holder_identity=identity,
                        lease_duration_seconds=ttl_seconds,
                        acquire_time=now,
                        renew_time=now,
                    ),
                )
                try:
                    await asyncio.to_thread(
                        api.create_namespaced_lease,
                        namespace=self.namespace,
                        body=lease_body,
                    )
                    verbose_proxy_logger.info(
                        "K8s leader election: created lease (holder=%s, namespace=%s)",
                        identity,
                        self.namespace,
                    )
                    return True
                except ApiException as create_exc:
                    if create_exc.status == 409:
                        # Race: another pod created it first
                        return False
                    verbose_proxy_logger.error(
                        "K8s leader election: error creating lease: %s",
                        create_exc,
                    )
                    return False
            elif exc.status == 409:
                # Optimistic concurrency conflict — another pod updated first
                return False
            else:
                verbose_proxy_logger.error("K8s leader election: API error: %s", exc)
                return False

        except Exception as exc:
            verbose_proxy_logger.error("K8s leader election: unexpected error: %s", exc)
            return False

    async def release(self, identity: str) -> None:
        """Release leadership by clearing the Lease holder."""
        api = self._get_api()

        try:
            from kubernetes.client.rest import ApiException
        except ImportError:
            return

        try:
            lease = await asyncio.to_thread(
                api.read_namespaced_lease,
                name=self.lease_name,
                namespace=self.namespace,
            )

            if lease.spec.holder_identity != identity:
                return  # We don't hold the lease

            lease.spec.holder_identity = None
            lease.spec.renew_time = None

            await asyncio.to_thread(
                api.replace_namespaced_lease,
                name=self.lease_name,
                namespace=self.namespace,
                body=lease,
            )
            verbose_proxy_logger.info(
                "K8s leader election: released lease (holder=%s)", identity
            )
        except ApiException as exc:
            verbose_proxy_logger.error(
                "K8s leader election: error releasing lease: %s", exc
            )
        except Exception as exc:
            verbose_proxy_logger.error(
                "K8s leader election: unexpected error on release: %s", exc
            )

    async def get_current_leader(self) -> Optional[str]:
        """Get the current leader from the Lease ``spec.holderIdentity``."""
        api = self._get_api()
        now = datetime.now(timezone.utc)

        try:
            from kubernetes.client.rest import ApiException
        except ImportError:
            return None

        try:
            lease = await asyncio.to_thread(
                api.read_namespaced_lease,
                name=self.lease_name,
                namespace=self.namespace,
            )

            holder = lease.spec.holder_identity
            renew_time = lease.spec.renew_time
            duration = lease.spec.lease_duration_seconds or DEFAULT_LEASE_SECONDS

            if not holder or not renew_time:
                return None

            expires_at = renew_time + timedelta(seconds=duration)
            if now >= expires_at:
                return None  # Expired

            return holder

        except ApiException as exc:
            if exc.status == 404:
                return None
            verbose_proxy_logger.error(
                "K8s leader election: error reading lease: %s", exc
            )
            return None
        except Exception as exc:
            verbose_proxy_logger.error(
                "K8s leader election: unexpected error reading leader: %s", exc
            )
            return None


# =============================================================================
# Redis SETNX Backend
# =============================================================================


class RedisLeaderElection:
    """Leader election using Redis ``SETNX`` with TTL.

    Atomic acquisition via ``SET key identity NX EX ttl``.  Release uses a
    Lua script to ensure only the holder can delete the key.

    Requires ``REDIS_HOST`` to be configured.  Uses the shared async Redis
    client from :mod:`litellm_llmrouter.redis_pool`.
    """

    # Lua script: delete key only if the value matches the caller's identity.
    # This prevents a released/expired lease from deleting a new holder's key.
    _RELEASE_SCRIPT = """
    if redis.call("GET", KEYS[1]) == ARGV[1] then
        return redis.call("DEL", KEYS[1])
    else
        return 0
    end
    """

    def __init__(self, key: str = "routeiq:leader"):
        self.key = key

    async def _get_client(self) -> Any:
        """Get the shared async Redis client."""
        from .redis_pool import get_async_redis_client

        client = await get_async_redis_client()
        if client is None:
            raise RuntimeError(
                "Redis is not configured (REDIS_HOST not set). "
                "Cannot use Redis leader election."
            )
        return client

    async def try_acquire(self, identity: str, ttl_seconds: int = 30) -> bool:
        """``SET key identity NX EX ttl_seconds``.

        Returns ``True`` if the key was set (we acquired leadership) or
        if we already hold the key (renewal).
        """
        try:
            client = await self._get_client()

            # Try SET NX (create if not exists)
            acquired = await client.set(self.key, identity, nx=True, ex=ttl_seconds)
            if acquired:
                verbose_proxy_logger.debug(
                    "Redis leader election: acquired (holder=%s, ttl=%ds)",
                    identity,
                    ttl_seconds,
                )
                return True

            # Key already exists — check if we hold it (renewal case)
            current = await client.get(self.key)
            if current == identity:
                # Refresh the TTL
                await client.expire(self.key, ttl_seconds)
                verbose_proxy_logger.debug(
                    "Redis leader election: renewed (holder=%s, ttl=%ds)",
                    identity,
                    ttl_seconds,
                )
                return True

            return False

        except Exception as exc:
            verbose_proxy_logger.error(
                "Redis leader election: error acquiring lease: %s", exc
            )
            return False

    async def release(self, identity: str) -> None:
        """Delete key only if we hold it (Lua script for atomicity)."""
        try:
            client = await self._get_client()
            await client.eval(self._RELEASE_SCRIPT, 1, self.key, identity)
            verbose_proxy_logger.info(
                "Redis leader election: released (holder=%s)", identity
            )
        except Exception as exc:
            verbose_proxy_logger.error(
                "Redis leader election: error releasing lease: %s", exc
            )

    async def get_current_leader(self) -> Optional[str]:
        """``GET key`` — returns the current holder identity or ``None``."""
        try:
            client = await self._get_client()
            return await client.get(self.key)
        except Exception as exc:
            verbose_proxy_logger.error(
                "Redis leader election: error reading leader: %s", exc
            )
            return None


# =============================================================================
# SQL Schema for Leader Election (PostgreSQL backend)
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
# Database-backed Leader Election (PostgreSQL backend — original)
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
            from .database import get_db_pool

            pool = await get_db_pool(self._db_url)
            if pool is None:
                verbose_proxy_logger.warning(
                    "Leader election: Database pool not available"
                )
                return False
            async with pool.acquire() as conn:
                await conn.execute(LEADER_ELECTION_TABLE_SQL)
                verbose_proxy_logger.debug("Leader election: Table created/verified")
                return True
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
            from datetime import timedelta
            from .database import get_db_pool

            pool = await get_db_pool(self._db_url)
            if pool is None:
                verbose_proxy_logger.warning(
                    "Leader election: Database pool not available, assuming single instance"
                )
                self._is_leader = True
                return True

            async with pool.acquire() as conn:
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
            from .database import get_db_pool

            pool = await get_db_pool(self._db_url)
            if pool is None:
                self._is_leader = False
                return True

            async with pool.acquire() as conn:
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
            from .database import get_db_pool

            pool = await get_db_pool(self._db_url)
            if pool is None:
                return None

            async with pool.acquire() as conn:
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
# Backend-Aware Leader Election Wrapper
# =============================================================================


class MultiBackendLeaderElection:
    """Wraps :class:`LeaderElection` (Postgres) with optional delegation to
    :class:`K8sLeaseLeaderElection` or :class:`RedisLeaderElection`.

    When ``backend`` is ``KUBERNETES`` or ``REDIS`` the corresponding
    specialised backend handles ``try_acquire``/``release``/
    ``get_current_leader`` and this wrapper only maintains the shared
    state (``is_leader``, generation, renewal thread, callbacks).

    When ``backend`` is ``POSTGRES`` (or ``NONE``) the existing
    :class:`LeaderElection` logic is used directly (backward-compatible).
    """

    def __init__(
        self,
        backend: LeaderElectionBackend,
        lock_name: str = DEFAULT_LOCK_NAME,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
        renew_interval_seconds: int = DEFAULT_RENEW_INTERVAL_SECONDS,
        database_url: str | None = None,
        holder_id: str | None = None,
    ):
        self.backend = backend
        self.lock_name = lock_name
        self.lease_seconds = lease_seconds
        self.renew_interval_seconds = renew_interval_seconds
        self.holder_id = holder_id or self._generate_holder_id()

        # Internal delegation target
        self._delegate: LeaderElectionProtocol | None = None

        # State
        self._is_leader = False
        self._lease_expires_at: datetime | None = None
        self._generation: int = 0
        self._consecutive_renewal_failures: int = 0
        self._stop_event = threading.Event()
        self._renew_thread: threading.Thread | None = None
        self._on_leadership_change: Callable[[bool], None] | None = None
        self._last_renewal_error: str | None = None

        # Initialise backend delegate
        if backend == LeaderElectionBackend.KUBERNETES:
            self._delegate = K8sLeaseLeaderElection(
                lease_name=lock_name,
                namespace=None,  # auto-detect
            )
        elif backend == LeaderElectionBackend.REDIS:
            self._delegate = RedisLeaderElection(
                key=f"routeiq:leader:{lock_name}",
            )
        elif backend == LeaderElectionBackend.POSTGRES:
            # Keep the original LeaderElection for Postgres
            self._postgres_election = LeaderElection(
                lock_name=lock_name,
                lease_seconds=lease_seconds,
                renew_interval_seconds=renew_interval_seconds,
                database_url=database_url,
                holder_id=self.holder_id,
            )
        # NONE backend — always leader, handled in try_acquire

    @staticmethod
    def _generate_holder_id() -> str:
        """Generate a unique holder ID for this instance."""
        try:
            hostname = socket.gethostname()[:20]
        except Exception:
            hostname = "unknown"
        # Use POD_NAME if available (K8s-friendly)
        pod_name = os.getenv("POD_NAME")
        if pod_name:
            return pod_name
        short_uuid = str(uuid.uuid4())[:8]
        return f"{hostname}-{short_uuid}"

    @property
    def is_leader(self) -> bool:
        """Check if this instance is currently the leader."""
        if self.backend == LeaderElectionBackend.POSTGRES:
            return self._postgres_election.is_leader
        if not self._is_leader:
            return False
        if self.backend == LeaderElectionBackend.NONE:
            return True
        if self._lease_expires_at is None:
            return False
        return datetime.now(timezone.utc) < self._lease_expires_at

    @property
    def database_configured(self) -> bool:
        """Check if database is configured (only relevant for Postgres backend)."""
        if self.backend == LeaderElectionBackend.POSTGRES:
            return self._postgres_election.database_configured
        return False

    async def ensure_table_exists(self) -> bool:
        """Create the leader election table if using Postgres backend."""
        if self.backend == LeaderElectionBackend.POSTGRES:
            return await self._postgres_election.ensure_table_exists()
        return True  # Not needed for K8s/Redis/None

    async def try_acquire(self) -> bool:
        """Try to acquire the leadership lease via the configured backend."""
        if self.backend == LeaderElectionBackend.NONE:
            self._is_leader = True
            self._lease_expires_at = datetime.now(timezone.utc) + timedelta(days=365)
            self._generation += 1
            self._consecutive_renewal_failures = 0
            return True

        if self.backend == LeaderElectionBackend.POSTGRES:
            return await self._postgres_election.try_acquire()

        # K8s or Redis delegate
        assert self._delegate is not None
        try:
            acquired = await self._delegate.try_acquire(
                identity=self.holder_id,
                ttl_seconds=self.lease_seconds,
            )
            was_leader = self._is_leader
            self._is_leader = acquired

            if acquired:
                self._lease_expires_at = datetime.now(timezone.utc) + timedelta(
                    seconds=self.lease_seconds
                )
                self._generation += 1
                self._consecutive_renewal_failures = 0
                self._last_renewal_error = None

                if not was_leader:
                    verbose_proxy_logger.info(
                        "Leader election [%s]: Acquired leadership (holder=%s)",
                        self.backend.value,
                        self.holder_id,
                    )
                    if self._on_leadership_change:
                        self._on_leadership_change(True)
            else:
                if was_leader:
                    self._lease_expires_at = None
                    verbose_proxy_logger.info(
                        "Leader election [%s]: Lost leadership (holder=%s)",
                        self.backend.value,
                        self.holder_id,
                    )
                    if self._on_leadership_change:
                        self._on_leadership_change(False)

            return acquired

        except Exception as exc:
            self._last_renewal_error = str(exc)
            verbose_proxy_logger.error(
                "Leader election [%s]: Error acquiring lease: %s",
                self.backend.value,
                exc,
            )
            return self._is_leader  # Favour stability

    async def renew(self) -> bool:
        """Renew the leadership lease if we are the leader."""
        if not self._is_leader:
            return False
        return await self.try_acquire()

    async def release(self) -> bool:
        """Release the leadership lease voluntarily."""
        if not self._is_leader:
            return True

        if self.backend == LeaderElectionBackend.POSTGRES:
            return await self._postgres_election.release()

        if self.backend == LeaderElectionBackend.NONE:
            self._is_leader = False
            return True

        assert self._delegate is not None
        try:
            await self._delegate.release(self.holder_id)
            self._is_leader = False
            self._lease_expires_at = None

            verbose_proxy_logger.info(
                "Leader election [%s]: Released leadership (holder=%s)",
                self.backend.value,
                self.holder_id,
            )
            if self._on_leadership_change:
                self._on_leadership_change(False)
            return True
        except Exception as exc:
            verbose_proxy_logger.error(
                "Leader election [%s]: Error releasing lease: %s",
                self.backend.value,
                exc,
            )
            return False

    async def get_current_leader(self) -> LeaseInfo | None:
        """Get information about the current leader."""
        if self.backend == LeaderElectionBackend.POSTGRES:
            return await self._postgres_election.get_current_leader()

        if self.backend == LeaderElectionBackend.NONE:
            if self._is_leader:
                return LeaseInfo(
                    lock_name=self.lock_name,
                    holder_id=self.holder_id,
                    acquired_at=datetime.now(timezone.utc),
                    expires_at=self._lease_expires_at or datetime.now(timezone.utc),
                    is_leader=True,
                )
            return None

        assert self._delegate is not None
        try:
            leader_id = await self._delegate.get_current_leader()
            if leader_id:
                return LeaseInfo(
                    lock_name=self.lock_name,
                    holder_id=leader_id,
                    acquired_at=datetime.now(timezone.utc),
                    expires_at=datetime.now(timezone.utc)
                    + timedelta(seconds=self.lease_seconds),
                    is_leader=(leader_id == self.holder_id),
                )
            return None
        except Exception as exc:
            verbose_proxy_logger.error(
                "Leader election [%s]: Error getting leader: %s",
                self.backend.value,
                exc,
            )
            return None

    def _renew_loop(self):
        """Background thread to periodically renew the lease."""
        verbose_proxy_logger.debug(
            "Leader election [%s]: Renewal thread started (interval=%ds)",
            self.backend.value,
            self.renew_interval_seconds,
        )

        while not self._stop_event.is_set():
            try:
                loop = asyncio.new_event_loop()
                try:
                    renewed = loop.run_until_complete(self.renew())
                finally:
                    loop.close()

                if renewed:
                    self._consecutive_renewal_failures = 0
                elif self._is_leader:
                    self._consecutive_renewal_failures += 1
                    verbose_proxy_logger.warning(
                        "Leader election [%s]: Renewal failed "
                        "(consecutive_failures=%d)",
                        self.backend.value,
                        self._consecutive_renewal_failures,
                    )

                    if self._consecutive_renewal_failures >= 2:
                        verbose_proxy_logger.error(
                            "Leader election [%s]: Auto-demoting after "
                            "%d consecutive renewal failures",
                            self.backend.value,
                            self._consecutive_renewal_failures,
                        )
                        self._is_leader = False
                        self._lease_expires_at = None
                        if self._on_leadership_change:
                            self._on_leadership_change(False)

            except Exception as exc:
                verbose_proxy_logger.error(
                    "Leader election [%s]: Renewal error: %s",
                    self.backend.value,
                    exc,
                )
                if self._is_leader:
                    self._consecutive_renewal_failures += 1
                    if self._consecutive_renewal_failures >= 2:
                        self._is_leader = False
                        self._lease_expires_at = None
                        if self._on_leadership_change:
                            self._on_leadership_change(False)

            self._stop_event.wait(self.renew_interval_seconds)

        verbose_proxy_logger.debug(
            "Leader election [%s]: Renewal thread stopped", self.backend.value
        )

    def start_renewal(self, on_leadership_change: Callable[[bool], None] | None = None):
        """Start the background lease renewal thread."""
        self._on_leadership_change = on_leadership_change

        if self.backend == LeaderElectionBackend.POSTGRES:
            self._postgres_election.start_renewal(on_leadership_change)
            return

        if self.backend == LeaderElectionBackend.NONE:
            return  # No renewal needed

        self._stop_event.clear()
        self._renew_thread = threading.Thread(
            target=self._renew_loop,
            daemon=True,
            name=f"leader-election-{self.backend.value}",
        )
        self._renew_thread.start()

    def stop_renewal(self):
        """Stop the background lease renewal thread."""
        if self.backend == LeaderElectionBackend.POSTGRES:
            self._postgres_election.stop_renewal()
            return

        self._stop_event.set()
        if self._renew_thread and self._renew_thread.is_alive():
            self._renew_thread.join(timeout=5)

    def get_status(self) -> dict:
        """Get the current leader election status."""
        if self.backend == LeaderElectionBackend.POSTGRES:
            status = self._postgres_election.get_status()
            status["backend"] = self.backend.value
            return status

        return {
            "backend": self.backend.value,
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
        """Validate that a fencing token matches the current generation."""
        if self.backend == LeaderElectionBackend.POSTGRES:
            return self._postgres_election.validate_fencing_token(token)
        return self._is_leader and token == self._generation


# =============================================================================
# Singleton Instance
# =============================================================================


_leader_election: MultiBackendLeaderElection | None = None


def get_leader_election() -> MultiBackendLeaderElection | None:
    """
    Get the global leader election instance, if leader election is enabled.

    Returns the :class:`MultiBackendLeaderElection` wrapper which
    auto-selects the best backend (K8s, Redis, Postgres, or None).

    Returns:
        MultiBackendLeaderElection instance if enabled, None otherwise
    """
    global _leader_election

    config = get_leader_election_config()

    if not config["enabled"]:
        return None

    if _leader_election is None:
        _leader_election = MultiBackendLeaderElection(
            backend=config["backend"],
            lock_name=config["lock_name"],
            lease_seconds=config["lease_seconds"],
            renew_interval_seconds=config["renew_interval_seconds"],
        )

    return _leader_election


def reset_leader_election() -> None:
    """Reset the leader election singleton. For testing only."""
    global _leader_election
    _leader_election = None


async def initialize_leader_election() -> MultiBackendLeaderElection | None:
    """
    Initialize leader election (create table and try initial acquisition).

    Returns:
        MultiBackendLeaderElection instance if enabled, None otherwise
    """
    election = get_leader_election()
    if election is None:
        verbose_proxy_logger.info("Leader election: Disabled (single instance mode)")
        return None

    # Ensure table exists (only relevant for Postgres backend)
    await election.ensure_table_exists()

    # Try initial acquisition
    is_leader = await election.try_acquire()

    verbose_proxy_logger.info(
        "Leader election: Initialized (backend=%s, holder=%s, is_leader=%s)",
        election.backend.value,
        election.holder_id,
        is_leader,
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
