"""
Leader-Election-Based Database Migrations
==========================================

Provides leader-election-aware database migration support for HA deployments.
When ``ROUTEIQ_LEADER_MIGRATIONS=true``, the elected leader runs
``prisma db push`` during startup while non-leaders wait for the migration
to complete before proceeding.

This replaces the shell-based ``LITELLM_RUN_DB_MIGRATIONS`` approach in
``docker/entrypoint.sh`` with an in-process, leader-aware alternative that
avoids race conditions across replicas.

Design:
- Opt-in via ``ROUTEIQ_LEADER_MIGRATIONS`` (default ``false``)
- Leader runs ``prisma db push --schema=<path> --accept-data-loss``
- After a successful run the leader writes a **durable completion marker**
  (a row in ``routeiq_migration_marker`` keyed to the migration/schema
  version) via the shared database pool
- Non-leaders **poll that durable marker** (up to *timeout* seconds) so the
  barrier coordinates ACROSS PODS, not just across coroutines in one process
- The in-process ``asyncio.Event`` is retained only as a same-process
  fast-path; the durable marker is the cross-pod source of truth
- Falls back gracefully: if Prisma or the schema cannot be found, logs a
  warning and continues startup

RouteIQ-2166 (cross-pod migration barrier): the previous implementation used a
module-level ``asyncio.Event`` as the leader -> non-leader signal. An
``asyncio.Event`` is per-PROCESS, so on a multi-pod Aurora-backed deploy a
non-leader pod's Event is never set by the leader pod -- the timeout-poll was
the only real barrier and non-leaders could proceed before migrations actually
completed. The leader now writes a durable marker row that non-leaders poll;
the Event is kept purely as a same-process optimisation.

RouteIQ-0921 (IAM region fail-loud, ADR-0028): the ``prisma db push`` step does
NOT build a RouteIQ-managed asyncpg pool. It shells out to the Prisma CLI, which
reads ``DATABASE_URL`` itself and never goes through ``database.get_db_pool`` /
``_resolve_db_iam_region``. So no ``IamRegionUnresolvedError`` can originate from
the push step. The durable marker, however, DOES go through
``database.get_db_pool`` (the same boot-critical pool builder as
``leader_election`` and ``database.run_migrations``). It therefore re-raises
``IamRegionUnresolvedError`` rather than swallowing it -- an unresolved region
under RDS/Aurora IAM auth is a fail-loud startup misconfiguration, consistent
with the other boot-critical DB callers.
"""

import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# How long non-leaders wait for the leader to finish migrations (seconds)
_DEFAULT_MIGRATION_TIMEOUT = 120
_DEFAULT_POLL_INTERVAL = 2

# Default marker version used when no explicit version is supplied. Operators
# may override via ``ROUTEIQ_MIGRATION_VERSION`` to force a re-coordination
# (e.g. after a schema change) so non-leaders do not observe a stale marker.
_DEFAULT_MARKER_VERSION = "litellm-prisma"

# Name of the durable marker table written by the leader and polled by
# non-leaders. ``CREATE TABLE IF NOT EXISTS`` makes it idempotent.
_MARKER_TABLE = "routeiq_migration_marker"

_MARKER_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {_MARKER_TABLE} (
    version VARCHAR(255) PRIMARY KEY,
    completed_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    holder_id VARCHAR(255)
);
"""

# Module-level Event: SAME-PROCESS fast-path ONLY. It is NOT the cross-pod
# source of truth (an asyncio.Event is per-process). The durable marker row is
# authoritative for coordination across pods. See RouteIQ-2166.
_migration_complete = asyncio.Event()


def _is_leader_migrations_enabled() -> bool:
    """Check if leader-based migrations are opt-ed in."""
    return os.getenv("ROUTEIQ_LEADER_MIGRATIONS", "false").lower() in (
        "true",
        "1",
        "yes",
    )


def _marker_version() -> str:
    """Resolve the migration/schema version used as the durable marker key.

    Operators can pin this via ``ROUTEIQ_MIGRATION_VERSION`` so a schema change
    forces non-leaders to wait for a fresh marker instead of observing a stale
    one from a previous deploy.
    """
    return os.getenv("ROUTEIQ_MIGRATION_VERSION", _DEFAULT_MARKER_VERSION).strip() or (
        _DEFAULT_MARKER_VERSION
    )


def _holder_id() -> str | None:
    """Best-effort identity for the leader that wrote the marker (debug only)."""
    return os.getenv("POD_NAME") or os.getenv("HOSTNAME")


def _find_prisma_schema() -> str | None:
    """Locate LiteLLM's ``schema.prisma`` file.

    Returns the path as a string, or ``None`` if not found.
    """
    try:
        import litellm

        schema = Path(litellm.__file__).parent / "proxy" / "schema.prisma"
        if schema.is_file():
            return str(schema)
    except Exception:
        pass
    return None


def _run_prisma_db_push(schema_path: str) -> subprocess.CompletedProcess:
    """Execute ``prisma db push`` synchronously.

    Raises ``subprocess.CalledProcessError`` on non-zero exit.
    """
    cmd = [
        sys.executable,
        "-m",
        "prisma",
        "db",
        "push",
        f"--schema={schema_path}",
        "--accept-data-loss",
    ]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=90,
        check=True,
    )


# =============================================================================
# Durable cross-pod marker (source of truth)
# =============================================================================


async def _write_completion_marker(version: str) -> bool:
    """Write the durable completion marker for *version* via the shared pool.

    The leader calls this AFTER migrations succeed so non-leaders on OTHER pods
    can observe completion. Idempotent: an existing marker row for the version
    is refreshed rather than duplicated.

    Returns ``True`` if the marker was written, ``False`` if the database pool
    is unavailable (e.g. asyncpg not installed). Re-raises
    ``IamRegionUnresolvedError`` (boot-critical fail-loud) so an unresolved IAM
    region is never folded into a soft "no marker" outcome.
    """
    from .database import IamRegionUnresolvedError, get_db_pool

    try:
        pool = await get_db_pool()
        if pool is None:
            logger.warning(
                "Leader migrations: DB pool unavailable, cannot write durable "
                "completion marker (non-leaders on other pods cannot observe "
                "completion via the marker)"
            )
            return False
        async with pool.acquire() as conn:
            await conn.execute(_MARKER_TABLE_SQL)
            await conn.execute(
                f"""
                INSERT INTO {_MARKER_TABLE} (version, completed_at, holder_id)
                VALUES ($1, NOW(), $2)
                ON CONFLICT (version) DO UPDATE SET
                    completed_at = NOW(),
                    holder_id = EXCLUDED.holder_id
                """,
                version,
                _holder_id(),
            )
        logger.info(
            f"Leader migrations: durable completion marker written (version={version})"
        )
        return True
    except IamRegionUnresolvedError:
        # Boot-critical: surface the IAM region misconfig, do not swallow.
        raise
    except Exception as exc:
        logger.error(
            f"Leader migrations: failed to write durable completion marker: {exc}"
        )
        return False


async def _marker_exists(version: str) -> bool:
    """Return ``True`` if the durable completion marker for *version* exists.

    Used by non-leaders to poll for cross-pod completion. Re-raises
    ``IamRegionUnresolvedError`` (boot-critical). Any other transient error
    (table not yet created, connection blip) returns ``False`` so the caller
    keeps polling until the timeout.
    """
    from .database import IamRegionUnresolvedError, get_db_pool

    try:
        pool = await get_db_pool()
        if pool is None:
            return False
        async with pool.acquire() as conn:
            # Ensure the table exists so a SELECT before the leader's first
            # write does not raise "relation does not exist" on every poll.
            await conn.execute(_MARKER_TABLE_SQL)
            row = await conn.fetchrow(
                f"SELECT version FROM {_MARKER_TABLE} WHERE version = $1",
                version,
            )
            return row is not None
    except IamRegionUnresolvedError:
        raise
    except Exception as exc:
        logger.debug(f"Leader migrations: marker poll error (will retry): {exc}")
        return False


async def run_migrations_if_leader(
    is_leader: bool,
    *,
    timeout: int | None = None,
    poll_interval: int | None = None,
) -> bool:
    """Run database migrations if this instance is the HA leader.

    * **Leader**: locates the Prisma schema and runs ``prisma db push``, then
      writes a **durable completion marker** so non-leaders -- including those
      on OTHER pods -- can observe completion. Also sets the in-process Event
      for same-process coroutines.
    * **Non-leader**: polls the durable marker (up to *timeout* seconds). The
      in-process Event is honoured as a fast-path for the same-process case.

    This function is a no-op when ``ROUTEIQ_LEADER_MIGRATIONS`` is not
    ``true``.

    Args:
        is_leader: Whether this instance currently holds the leader lease.
        timeout: Maximum seconds a non-leader will wait. Defaults to 120.
        poll_interval: Seconds between readiness checks. Defaults to 2.

    Returns:
        ``True`` if migrations ran (or were skipped because the feature is
        disabled), ``False`` on failure.
    """
    if not _is_leader_migrations_enabled():
        logger.debug("Leader migrations disabled (ROUTEIQ_LEADER_MIGRATIONS != true)")
        return True

    if not os.getenv("DATABASE_URL"):
        logger.debug("Leader migrations skipped (no DATABASE_URL)")
        return True

    if timeout is None:
        timeout = _DEFAULT_MIGRATION_TIMEOUT
    if poll_interval is None:
        poll_interval = _DEFAULT_POLL_INTERVAL

    if is_leader:
        return await _run_as_leader()
    else:
        return await _wait_as_follower(timeout=timeout, poll_interval=poll_interval)


async def _run_as_leader() -> bool:
    """Leader path: find schema, run prisma db push, write durable marker."""
    logger.info("Leader migrations: this instance is the leader, running migrations...")

    version = _marker_version()
    schema_path = _find_prisma_schema()
    if schema_path is None:
        logger.warning(
            "Leader migrations: Prisma schema not found, skipping migrations"
        )
        # Nothing to migrate -> mark complete so non-leaders proceed. Write the
        # durable marker first (best-effort) then the same-process Event.
        await _write_completion_marker(version)
        _migration_complete.set()
        return True

    logger.info(f"Leader migrations: running prisma db push (schema={schema_path})")

    try:
        result = await asyncio.to_thread(_run_prisma_db_push, schema_path)
        logger.info("Leader migrations: prisma db push completed successfully")
        if result.stdout:
            logger.debug(f"prisma db push stdout: {result.stdout}")
    except subprocess.CalledProcessError as exc:
        logger.error(
            f"Leader migrations: prisma db push failed (exit={exc.returncode})"
        )
        if exc.stderr:
            logger.error(f"prisma db push stderr: {exc.stderr}")
        # Do NOT write the durable marker on failure -- non-leaders must NOT
        # observe completion when migrations did not actually succeed. The
        # same-process Event is set only so co-located followers stop blocking;
        # the leader's False return signals failure to the caller.
        _migration_complete.set()
        return False
    except subprocess.TimeoutExpired:
        logger.error("Leader migrations: prisma db push timed out")
        _migration_complete.set()
        return False
    except FileNotFoundError:
        logger.error(
            "Leader migrations: prisma CLI not found. Install with: pip install prisma"
        )
        _migration_complete.set()
        return False

    # Migrations succeeded: write the DURABLE cross-pod marker first (source of
    # truth for non-leaders on other pods), then the same-process Event.
    await _write_completion_marker(version)
    _migration_complete.set()
    return True


async def _wait_as_follower(*, timeout: int, poll_interval: int) -> bool:
    """Non-leader path: poll the DURABLE cross-pod marker for completion.

    The durable marker row written by the leader is the source of truth across
    pods. The in-process ``asyncio.Event`` is honoured purely as a same-process
    fast-path (it is only ever set by a leader in THIS process).
    """
    logger.info(
        f"Leader migrations: not the leader, polling durable marker up to {timeout}s "
        "for leader to complete migrations..."
    )
    version = _marker_version()
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout

    while True:
        # Same-process fast-path: a co-located leader coroutine already signalled.
        if _migration_complete.is_set():
            logger.info(
                "Leader migrations: same-process leader signalled completion, "
                "proceeding"
            )
            return True

        # Cross-pod source of truth: durable marker written by the leader pod.
        if await _marker_exists(version):
            logger.info(
                "Leader migrations: durable completion marker observed "
                f"(version={version}), proceeding"
            )
            return True

        if loop.time() >= deadline:
            msg = (
                f"Leader migrations: timed out after {timeout}s waiting for the "
                "durable cross-pod completion marker; migrations may not have "
                "completed on the leader pod"
            )
            # Fail-closed (RouteIQ-2166): with the in-process Event no longer
            # the cross-pod source of truth, a timeout means we genuinely never
            # observed the durable marker. Raise so the caller surfaces the
            # un-migrated-DB risk rather than silently proceeding.
            logger.error(msg)
            raise TimeoutError(msg)

        await asyncio.sleep(poll_interval)


def reset_migration_state() -> None:
    """Reset module state. For testing only."""
    global _migration_complete
    _migration_complete = asyncio.Event()
