"""
Audit Logging for RouteIQ Gateway Control-Plane
================================================

This module provides durable, Postgres-backed audit logging for control-plane actions.
Audit logs capture security-relevant operations on high-risk endpoints.

Captured Fields:
- timestamp: When the action occurred (UTC)
- request_id: Correlation ID for tracing
- actor: Who performed the action (team_id, user_id, api_key_subject)
- action: What action was performed (e.g., "mcp.server.create")
- resource_type: Type of resource affected (e.g., "mcp_server")
- resource_id: Identifier of the affected resource
- outcome: Result of the action ("success", "denied", "error")
- metadata: Additional context (minimal, not request body)

Graceful Degradation:
- AUDIT_LOG_ENABLED: Enable/disable audit logging (default: true)
- AUDIT_LOG_FAIL_MODE: "open" (default) or "closed"
  - fail-open: DB errors logged to app logger, request continues
  - fail-closed: DB errors cause 503 Service Unavailable

Usage:
    from litellm_llmrouter.audit import audit_log, AuditAction

    # In an endpoint handler:
    await audit_log(
        action=AuditAction.MCP_SERVER_CREATE,
        resource_type="mcp_server",
        resource_id=server_id,
        outcome="success",
        actor_info=rbac_info,  # From RBAC dependency
    )
"""

import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from litellm._logging import verbose_proxy_logger

from .auth import get_request_id

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


def is_audit_log_enabled() -> bool:
    """Check if audit logging is enabled."""
    env_val = os.getenv("AUDIT_LOG_ENABLED", "true").lower().strip()
    return env_val not in ("false", "0", "no", "off")


def is_audit_fail_closed() -> bool:
    """Check if audit logging uses fail-closed mode."""
    env_val = os.getenv("AUDIT_LOG_FAIL_MODE", "open").lower().strip()
    return env_val == "closed"


def get_database_url() -> str | None:
    """Get the database URL from environment."""
    return os.getenv("DATABASE_URL")


# =============================================================================
# Audit Actions (Control-Plane Operations)
# =============================================================================


class AuditAction(str, Enum):
    """Enumeration of auditable control-plane actions."""

    # MCP Server Management
    MCP_SERVER_CREATE = "mcp.server.create"
    MCP_SERVER_UPDATE = "mcp.server.update"
    MCP_SERVER_DELETE = "mcp.server.delete"

    # MCP Tool Management
    MCP_TOOL_REGISTER = "mcp.tool.register"
    MCP_TOOL_CALL = "mcp.tool.call"

    # Config Reload/Sync
    CONFIG_RELOAD = "system.config.reload"
    CONFIG_SYNC = "system.config.sync"

    # A2A Agent Management
    A2A_AGENT_CREATE = "a2a.agent.create"
    A2A_AGENT_DELETE = "a2a.agent.delete"

    # Management Operations (v0.2.0 — catch-all for LiteLLM passthrough)
    MANAGEMENT_KEY_GENERATE = "management.key.generate"
    MANAGEMENT_KEY_DELETE = "management.key.delete"
    MANAGEMENT_TEAM_CREATE = "management.team.create"
    MANAGEMENT_TEAM_DELETE = "management.team.delete"
    MANAGEMENT_MODEL_ADD = "management.model.add"
    MANAGEMENT_MODEL_DELETE = "management.model.delete"
    MANAGEMENT_BUDGET_CREATE = "management.budget.create"
    MANAGEMENT_CREDENTIAL_CREATE = "management.credential.create"
    MANAGEMENT_CREDENTIAL_DELETE = "management.credential.delete"
    MANAGEMENT_GUARDRAIL_CREATE = "management.guardrail.create"
    MANAGEMENT_CONFIG_RELOAD = "management.config.reload"


class AuditOutcome(str, Enum):
    """Outcome of an audited action."""

    SUCCESS = "success"
    DENIED = "denied"
    ERROR = "error"


# =============================================================================
# Audit Log Database Model
# =============================================================================


@dataclass
class AuditLogEntry:
    """A single audit log entry."""

    # Identifiers
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    request_id: str | None = None

    # Actor (who)
    actor_type: str = "unknown"  # "admin_key", "api_key", "user"
    actor_id: str | None = None  # team_id, user_id, or masked key

    # Action (what)
    action: str = ""

    # Resource (on what)
    resource_type: str = ""
    resource_id: str | None = None

    # Outcome
    outcome: str = "success"  # "success", "denied", "error"
    outcome_reason: str | None = None  # Error message or denial reason

    # Context (minimal metadata, no request body)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "request_id": self.request_id,
            "actor_type": self.actor_type,
            "actor_id": self.actor_id,
            "action": self.action,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "outcome": self.outcome,
            "outcome_reason": self.outcome_reason,
            "metadata": self.metadata,
        }


# =============================================================================
# Audit Log Repository
# =============================================================================


class AuditLogRepository:
    """
    Repository for persistent audit log storage.

    Uses PostgreSQL when DATABASE_URL is configured.
    Provides graceful degradation based on fail mode configuration.
    """

    def __init__(self):
        self._db_url = get_database_url()
        self._enabled = is_audit_log_enabled()
        self._fail_closed = is_audit_fail_closed()

    @property
    def is_enabled(self) -> bool:
        """Check if audit logging is enabled and configured."""
        return self._enabled and self._db_url is not None

    @property
    def is_fail_closed(self) -> bool:
        """Check if fail-closed mode is enabled."""
        return self._fail_closed

    async def write(self, entry: AuditLogEntry) -> bool:
        """
        Write an audit log entry to the database.

        Args:
            entry: The audit log entry to write

        Returns:
            True if the entry was written successfully

        Raises:
            AuditWriteError: If fail-closed mode is enabled and write fails
        """
        if not self._enabled:
            return True  # Audit logging disabled, silently succeed

        if not self._db_url:
            # No database configured - log fallback and continue
            self._log_fallback(entry, "No DATABASE_URL configured")
            return True

        try:
            await self._persist_to_db(entry)
            verbose_proxy_logger.debug(
                f"Audit: Logged {entry.action} on {entry.resource_type}/{entry.resource_id}"
            )
            return True
        except ImportError:
            self._log_fallback(entry, "asyncpg not installed")
            if self._fail_closed:
                raise AuditWriteError("asyncpg not installed")
            return True
        except Exception as e:
            self._log_fallback(entry, str(e))
            if self._fail_closed:
                raise AuditWriteError(f"Database write failed: {e}")
            return True

    def _log_fallback(self, entry: AuditLogEntry, reason: str) -> None:
        """Log audit entry to application logger as fallback."""
        logger.warning(
            f"AUDIT_FALLBACK: action={entry.action} resource={entry.resource_type}/{entry.resource_id} "
            f"outcome={entry.outcome} actor={entry.actor_type}/{entry.actor_id} "
            f"request_id={entry.request_id} reason={reason}",
            extra={
                "audit_fallback": True,
                "audit_entry": entry.to_dict(),
                "fallback_reason": reason,
            },
        )

    async def _persist_to_db(self, entry: AuditLogEntry) -> None:
        """Persist audit log entry to PostgreSQL database."""
        import asyncpg
        import json

        conn = await asyncpg.connect(self._db_url)
        try:
            await conn.execute(
                """
                INSERT INTO audit_logs (
                    id, timestamp, request_id,
                    actor_type, actor_id,
                    action, resource_type, resource_id,
                    outcome, outcome_reason, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """,
                entry.id,
                entry.timestamp,
                entry.request_id,
                entry.actor_type,
                entry.actor_id,
                entry.action,
                entry.resource_type,
                entry.resource_id,
                entry.outcome,
                entry.outcome_reason,
                json.dumps(entry.metadata) if entry.metadata else "{}",
            )
        finally:
            await conn.close()

    async def query(
        self,
        action: str | None = None,
        resource_type: str | None = None,
        outcome: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditLogEntry]:
        """
        Query audit logs with optional filters.

        This is primarily for administrative/debugging use.
        """
        if not self._db_url:
            return []

        try:
            import asyncpg
            import json

            conn = await asyncpg.connect(self._db_url)
            try:
                # Build query with optional filters
                query = "SELECT * FROM audit_logs WHERE 1=1"
                params = []
                param_idx = 1

                if action:
                    query += f" AND action = ${param_idx}"
                    params.append(action)
                    param_idx += 1

                if resource_type:
                    query += f" AND resource_type = ${param_idx}"
                    params.append(resource_type)
                    param_idx += 1

                if outcome:
                    query += f" AND outcome = ${param_idx}"
                    params.append(outcome)
                    param_idx += 1

                if since:
                    query += f" AND timestamp >= ${param_idx}"
                    params.append(since)
                    param_idx += 1

                if until:
                    query += f" AND timestamp <= ${param_idx}"
                    params.append(until)
                    param_idx += 1

                query += f" ORDER BY timestamp DESC LIMIT ${param_idx}"
                params.append(limit)

                rows = await conn.fetch(query, *params)

                entries = []
                for row in rows:
                    entries.append(
                        AuditLogEntry(
                            id=row["id"],
                            timestamp=row["timestamp"],
                            request_id=row["request_id"],
                            actor_type=row["actor_type"],
                            actor_id=row["actor_id"],
                            action=row["action"],
                            resource_type=row["resource_type"],
                            resource_id=row["resource_id"],
                            outcome=row["outcome"],
                            outcome_reason=row["outcome_reason"],
                            metadata=json.loads(row["metadata"])
                            if row["metadata"]
                            else {},
                        )
                    )
                return entries
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Audit query failed: {e}")
            return []


class AuditWriteError(Exception):
    """Raised when audit log write fails in fail-closed mode."""

    pass


# =============================================================================
# SQL Migration
# =============================================================================


AUDIT_LOGS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS audit_logs (
    id VARCHAR(36) PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    request_id VARCHAR(255),
    actor_type VARCHAR(50) NOT NULL DEFAULT 'unknown',
    actor_id VARCHAR(255),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100) NOT NULL,
    resource_id VARCHAR(255),
    outcome VARCHAR(20) NOT NULL DEFAULT 'success',
    outcome_reason TEXT,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action);
CREATE INDEX IF NOT EXISTS idx_audit_logs_resource ON audit_logs(resource_type, resource_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_actor ON audit_logs(actor_type, actor_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_outcome ON audit_logs(outcome);
CREATE INDEX IF NOT EXISTS idx_audit_logs_request_id ON audit_logs(request_id);
"""


async def run_audit_migrations() -> None:
    """Run audit log table migrations."""
    db_url = get_database_url()
    if not db_url:
        verbose_proxy_logger.info(
            "No DATABASE_URL configured, skipping audit migrations"
        )
        return

    if not is_audit_log_enabled():
        verbose_proxy_logger.info("Audit logging disabled, skipping migrations")
        return

    try:
        import asyncpg

        conn = await asyncpg.connect(db_url)
        try:
            await conn.execute(AUDIT_LOGS_TABLE_SQL)
            verbose_proxy_logger.info("Audit: Migrations completed successfully")
        finally:
            await conn.close()
    except ImportError:
        verbose_proxy_logger.warning("asyncpg not installed, skipping audit migrations")
    except Exception as e:
        verbose_proxy_logger.error(f"Audit: Error running migrations: {e}")


# =============================================================================
# Singleton Repository Instance
# =============================================================================


_audit_repository: AuditLogRepository | None = None


def get_audit_repository() -> AuditLogRepository:
    """Get the global audit log repository instance."""
    global _audit_repository
    if _audit_repository is None:
        _audit_repository = AuditLogRepository()
    return _audit_repository


def reset_audit_repository() -> None:
    """Reset the audit repository singleton (for testing)."""
    global _audit_repository
    _audit_repository = None


# =============================================================================
# High-Level Audit Log API
# =============================================================================


def extract_actor_info(rbac_info: dict[str, Any] | None) -> tuple[str, str | None]:
    """
    Extract actor type and ID from RBAC info.

    Args:
        rbac_info: RBAC information from requires_permission dependency

    Returns:
        (actor_type, actor_id) tuple
    """
    if rbac_info is None:
        return ("unknown", None)

    if rbac_info.get("is_admin"):
        # Admin key authentication
        admin_key = rbac_info.get("admin_key", "")
        # Mask the key for security (show last 4 chars only)
        if admin_key and len(admin_key) > 4:
            masked = "..." + admin_key[-4:]
        else:
            masked = "***"
        return ("admin_key", masked)

    # User authentication - extract from user_info
    user_info = rbac_info.get("user_info", {})

    # Try to get user/team identifiers
    user_id = user_info.get("user_id")
    team_id = user_info.get("team_id")

    if user_id:
        return ("user", user_id)
    if team_id:
        return ("team", team_id)

    # Fall back to API key token (masked)
    token = user_info.get("token", "")
    if token:
        masked = "..." + token[-4:] if len(token) > 4 else "***"
        return ("api_key", masked)

    return ("unknown", None)


async def audit_log(
    action: AuditAction | str,
    resource_type: str,
    resource_id: str | None = None,
    outcome: AuditOutcome | str = AuditOutcome.SUCCESS,
    outcome_reason: str | None = None,
    actor_info: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """
    Log an audit event for a control-plane action.

    This is the primary API for logging audit events. Call this from
    endpoint handlers after the action completes (success or failure).

    Args:
        action: The action being performed (use AuditAction enum)
        resource_type: Type of resource (e.g., "mcp_server")
        resource_id: Identifier of the resource
        outcome: Result of the action (success/denied/error)
        outcome_reason: Reason for denied/error outcomes
        actor_info: RBAC info dict from requires_permission dependency
        metadata: Additional context (minimal, no request body)

    Returns:
        True if logged successfully (or fail-open mode)

    Raises:
        AuditWriteError: If fail-closed mode and write fails
    """
    repo = get_audit_repository()

    if not repo.is_enabled:
        return True

    # Get request ID from context
    request_id = get_request_id() or str(uuid.uuid4())

    # Extract actor info
    actor_type, actor_id = extract_actor_info(actor_info)

    # Normalize action and outcome to strings
    action_str = action.value if isinstance(action, AuditAction) else str(action)
    outcome_str = outcome.value if isinstance(outcome, AuditOutcome) else str(outcome)

    entry = AuditLogEntry(
        request_id=request_id,
        actor_type=actor_type,
        actor_id=actor_id,
        action=action_str,
        resource_type=resource_type,
        resource_id=resource_id,
        outcome=outcome_str,
        outcome_reason=outcome_reason,
        metadata=metadata or {},
    )

    return await repo.write(entry)


async def audit_success(
    action: AuditAction | str,
    resource_type: str,
    resource_id: str | None = None,
    actor_info: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Convenience function to log a successful action."""
    return await audit_log(
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        outcome=AuditOutcome.SUCCESS,
        actor_info=actor_info,
        metadata=metadata,
    )


async def audit_denied(
    action: AuditAction | str,
    resource_type: str,
    resource_id: str | None = None,
    reason: str | None = None,
    actor_info: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Convenience function to log a denied action."""
    return await audit_log(
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        outcome=AuditOutcome.DENIED,
        outcome_reason=reason,
        actor_info=actor_info,
        metadata=metadata,
    )


async def audit_error(
    action: AuditAction | str,
    resource_type: str,
    resource_id: str | None = None,
    error: str | None = None,
    actor_info: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Convenience function to log an action that resulted in error."""
    return await audit_log(
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        outcome=AuditOutcome.ERROR,
        outcome_reason=error,
        actor_info=actor_info,
        metadata=metadata,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Config
    "is_audit_log_enabled",
    "is_audit_fail_closed",
    # Enums
    "AuditAction",
    "AuditOutcome",
    # Model
    "AuditLogEntry",
    # Repository
    "AuditLogRepository",
    "AuditWriteError",
    "get_audit_repository",
    "reset_audit_repository",
    # Migrations
    "run_audit_migrations",
    "AUDIT_LOGS_TABLE_SQL",
    # High-level API
    "audit_log",
    "audit_success",
    "audit_denied",
    "audit_error",
    "extract_actor_info",
]
