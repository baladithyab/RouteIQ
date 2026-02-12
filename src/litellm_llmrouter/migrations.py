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
- Non-leaders poll a readiness flag until the leader finishes or a timeout
  is reached
- Falls back gracefully: if Prisma or the schema cannot be found, logs a
  warning and continues startup
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

# Module-level flag shared between leader and non-leader coroutines
_migration_complete = asyncio.Event()


def _is_leader_migrations_enabled() -> bool:
    """Check if leader-based migrations are opt-ed in."""
    return os.getenv("ROUTEIQ_LEADER_MIGRATIONS", "false").lower() in (
        "true",
        "1",
        "yes",
    )


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


async def run_migrations_if_leader(
    is_leader: bool,
    *,
    timeout: int | None = None,
    poll_interval: int | None = None,
) -> bool:
    """Run database migrations if this instance is the HA leader.

    * **Leader**: locates the Prisma schema and runs ``prisma db push``, then
      signals completion so non-leaders can proceed.
    * **Non-leader**: waits (up to *timeout* seconds) for the leader to signal
      completion.

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
    """Leader path: find schema, run prisma db push, signal completion."""
    logger.info("Leader migrations: this instance is the leader, running migrations...")

    schema_path = _find_prisma_schema()
    if schema_path is None:
        logger.warning(
            "Leader migrations: Prisma schema not found, skipping migrations"
        )
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

    _migration_complete.set()
    return True


async def _wait_as_follower(*, timeout: int, poll_interval: int) -> bool:
    """Non-leader path: wait for the leader to finish migrations."""
    logger.info(
        f"Leader migrations: not the leader, waiting up to {timeout}s "
        "for leader to complete migrations..."
    )
    try:
        await asyncio.wait_for(_migration_complete.wait(), timeout=timeout)
        logger.info("Leader migrations: leader signalled completion, proceeding")
        return True
    except asyncio.TimeoutError:
        logger.warning(
            f"Leader migrations: timed out after {timeout}s waiting for leader. "
            "Proceeding anyway."
        )
        return True


def reset_migration_state() -> None:
    """Reset module state. For testing only."""
    global _migration_complete
    _migration_complete = asyncio.Event()
