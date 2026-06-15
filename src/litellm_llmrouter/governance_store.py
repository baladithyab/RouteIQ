"""Durable (Aurora-backed) governance / usage-policy / spend store.

This module is the **optional durable layer** for the RouteIQ control plane
(P4).  It is additive and DB-optional, mirroring :mod:`audit.py`:

* When ``DATABASE_URL`` is unset the store is *disabled* and every method is a
  no-op returning a sentinel -- callers degrade to the in-memory dicts + JSON
  file that ``governance.py`` / ``usage_policies.py`` already ship.
* When ``DATABASE_URL`` is set the store owns its own SQL + asyncpg access (so
  ``governance.py`` stays free of asyncpg), persists CRUD mutations, and serves
  the durable monthly spend rollup that is the budget system-of-record.

Tables (all ``CREATE ... IF NOT EXISTS``, idempotent, additive):
  * ``governance_orgs``        -- first-class organizations (Org -> WS -> Key top)
  * ``governance_workspaces``  -- mirror of ``WorkspaceConfig``
  * ``governance_keys``        -- mirror of ``KeyGovernance``
  * ``usage_policies``         -- mirror of ``UsagePolicy``
  * ``governance_spend``       -- durable budget rollup (the NEW write path)

Soft FKs only (no PG ``REFERENCES``) to preserve the additive contract and
tolerate hydration ordering on a partial restore.

NOTE: this is the asyncpg (raw SQL) path reached through ``database.run_migrations``
-- NOT the leader-gated Prisma path in ``migrations.py``.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING

from litellm._logging import verbose_proxy_logger

from .database import get_database_url, get_db_pool

if TYPE_CHECKING:  # avoid import cycle at runtime; only needed for typing
    from .governance import KeyGovernance, OrgConfig, WorkspaceConfig
    from .usage_policies import UsagePolicy


# =============================================================================
# SQL Migration
# =============================================================================


GOVERNANCE_TABLES_SQL = """
-- Organizations (first-class entity; pre-P4 org was only a string field)
CREATE TABLE IF NOT EXISTS governance_orgs (
    org_id       VARCHAR(255) PRIMARY KEY,
    name         VARCHAR(255) NOT NULL,
    metadata     JSONB NOT NULL DEFAULT '{}',
    created_at   TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Workspaces (mirror of WorkspaceConfig)
CREATE TABLE IF NOT EXISTS governance_workspaces (
    workspace_id            VARCHAR(255) PRIMARY KEY,
    name                    VARCHAR(255) NOT NULL,
    org_id                  VARCHAR(255),
    allowed_models          JSONB NOT NULL DEFAULT '[]',
    blocked_models          JSONB NOT NULL DEFAULT '[]',
    max_budget_usd          DOUBLE PRECISION,
    budget_alert_threshold  DOUBLE PRECISION NOT NULL DEFAULT 0.8,
    max_rpm                 INTEGER,
    max_tpm                 INTEGER,
    enforced_guardrails     JSONB NOT NULL DEFAULT '[]',
    default_routing_profile VARCHAR(255),
    config_override_allowed BOOLEAN NOT NULL DEFAULT TRUE,
    metadata                JSONB NOT NULL DEFAULT '{}',
    created_at              TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_gov_ws_org ON governance_workspaces(org_id);

-- Per-key governance (mirror of KeyGovernance)
CREATE TABLE IF NOT EXISTS governance_keys (
    key_id                  VARCHAR(255) PRIMARY KEY,
    workspace_id            VARCHAR(255),
    scopes                  JSONB NOT NULL DEFAULT '["completions.write"]',
    max_budget_usd          DOUBLE PRECISION,
    budget_period           VARCHAR(20) NOT NULL DEFAULT 'monthly',
    max_rpm                 INTEGER,
    max_tpm                 INTEGER,
    allowed_models          JSONB NOT NULL DEFAULT '[]',
    enforced_config         JSONB,
    config_override_allowed BOOLEAN NOT NULL DEFAULT TRUE,
    metadata                JSONB NOT NULL DEFAULT '{}',
    created_at              TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_gov_keys_ws ON governance_keys(workspace_id);

-- Usage policy definitions (mirror of UsagePolicy; enum-as-VARCHAR not PG enum)
CREATE TABLE IF NOT EXISTS usage_policies (
    policy_id        VARCHAR(255) PRIMARY KEY,
    name             VARCHAR(255) NOT NULL,
    description      TEXT NOT NULL DEFAULT '',
    enabled          BOOLEAN NOT NULL DEFAULT TRUE,
    conditions       JSONB NOT NULL DEFAULT '{}',
    exclusions       JSONB NOT NULL DEFAULT '{}',
    group_by         JSONB NOT NULL DEFAULT '[]',
    limit_type       VARCHAR(20) NOT NULL DEFAULT 'requests',
    limit_value      DOUBLE PRECISION NOT NULL DEFAULT 100,
    limit_period     VARCHAR(20) NOT NULL DEFAULT 'minute',
    action           VARCHAR(20) NOT NULL DEFAULT 'deny',
    alert_threshold  DOUBLE PRECISION NOT NULL DEFAULT 0.8,
    priority         INTEGER NOT NULL DEFAULT 100,
    workspace_id     VARCHAR(255),
    created_at       TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_usage_policies_ws ON usage_policies(workspace_id);

-- Durable spend ledger (budget system-of-record; the NEW write path).
-- One row per (scope, period, period_start).  period_start is epoch-aligned to
-- the Redis spend bucket (NOT calendar months) so it stays byte-compatible.
CREATE TABLE IF NOT EXISTS governance_spend (
    scope            VARCHAR(255) NOT NULL,
    scope_type       VARCHAR(20)  NOT NULL,
    period_start     TIMESTAMP WITH TIME ZONE NOT NULL,
    period           VARCHAR(20)  NOT NULL DEFAULT 'monthly',
    spend_usd        DOUBLE PRECISION NOT NULL DEFAULT 0,
    request_count    BIGINT NOT NULL DEFAULT 0,
    total_tokens     BIGINT NOT NULL DEFAULT 0,
    updated_at       TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    PRIMARY KEY (scope, period, period_start)
);
CREATE INDEX IF NOT EXISTS idx_gov_spend_scope ON governance_spend(scope);
CREATE INDEX IF NOT EXISTS idx_gov_spend_period ON governance_spend(period_start);
"""


async def run_governance_migrations() -> None:
    """Create the governance/spend tables (idempotent, additive).

    Wired into ``database.run_migrations`` so all RouteIQ-native tables are
    created in one place.  Early-returns (no raise) when there is no
    ``DATABASE_URL`` or the pool is unavailable -- matching ``run_audit_migrations``.
    """
    db_url = get_database_url()
    if not db_url:
        verbose_proxy_logger.info(
            "No DATABASE_URL configured, skipping governance migrations"
        )
        return
    try:
        pool = await get_db_pool(db_url)
        if pool is None:
            verbose_proxy_logger.warning(
                "Database pool not available, skipping governance migrations"
            )
            return
        async with pool.acquire() as conn:
            await conn.execute(GOVERNANCE_TABLES_SQL)
        verbose_proxy_logger.info("Governance: migrations completed successfully")
    except Exception as e:
        verbose_proxy_logger.error(f"Governance: Error running migrations: {e}")


# =============================================================================
# Serialization helpers (pure -- no DB; testable for model/schema drift)
# =============================================================================


def _org_columns(org: "OrgConfig") -> dict:
    """Map an ``OrgConfig`` to its ``governance_orgs`` column dict."""
    return {
        "org_id": org.org_id,
        "name": org.name,
        "metadata": json.dumps(org.metadata or {}),
    }


def _workspace_columns(ws: "WorkspaceConfig") -> dict:
    """Map a ``WorkspaceConfig`` to its ``governance_workspaces`` column dict."""
    return {
        "workspace_id": ws.workspace_id,
        "name": ws.name,
        "org_id": ws.org_id,
        "allowed_models": json.dumps(ws.allowed_models or []),
        "blocked_models": json.dumps(ws.blocked_models or []),
        "max_budget_usd": ws.max_budget_usd,
        "budget_alert_threshold": ws.budget_alert_threshold,
        "max_rpm": ws.max_rpm,
        "max_tpm": ws.max_tpm,
        "enforced_guardrails": json.dumps(ws.enforced_guardrails or []),
        "default_routing_profile": ws.default_routing_profile,
        "config_override_allowed": ws.config_override_allowed,
        "metadata": json.dumps(ws.metadata or {}),
    }


def _key_columns(kg: "KeyGovernance") -> dict:
    """Map a ``KeyGovernance`` to its ``governance_keys`` column dict.

    ``scopes`` is a ``Set[str]`` in the model -- stored as a sorted JSON array
    (deterministic) and coerced back to ``set`` on load.
    """
    return {
        "key_id": kg.key_id,
        "workspace_id": kg.workspace_id,
        "scopes": json.dumps(sorted(kg.scopes)),
        "max_budget_usd": kg.max_budget_usd,
        "budget_period": kg.budget_period,
        "max_rpm": kg.max_rpm,
        "max_tpm": kg.max_tpm,
        "allowed_models": json.dumps(kg.allowed_models or []),
        "enforced_config": (
            json.dumps(kg.enforced_config) if kg.enforced_config is not None else None
        ),
        "config_override_allowed": kg.config_override_allowed,
        "metadata": json.dumps(kg.metadata or {}),
    }


def _policy_columns(p: "UsagePolicy") -> dict:
    """Map a ``UsagePolicy`` to its ``usage_policies`` column dict.

    Enum fields are stored as their ``.value`` strings (VARCHAR, not PG enum).
    """
    return {
        "policy_id": p.policy_id,
        "name": p.name,
        "description": p.description,
        "enabled": p.enabled,
        "conditions": json.dumps(p.conditions or {}),
        "exclusions": json.dumps(p.exclusions or {}),
        "group_by": json.dumps(p.group_by or []),
        "limit_type": p.limit_type.value,
        "limit_value": p.limit_value,
        "limit_period": p.limit_period.value,
        "action": p.action.value,
        "alert_threshold": p.alert_threshold,
        "priority": p.priority,
        "workspace_id": p.workspace_id,
    }


# =============================================================================
# Governance Store (Aurora-backed; no-op when DATABASE_URL unset)
# =============================================================================


class GovernanceStore:
    """Durable store for orgs / workspaces / keys / policies / spend.

    Reads ``DATABASE_URL`` at construction (like ``AuditLogRepository``).  When
    not configured, :attr:`enabled` is ``False`` and every method is a fail-open
    no-op so callers transparently degrade to in-memory + JSON.
    """

    def __init__(self) -> None:
        self._db_url = get_database_url()

    @property
    def enabled(self) -> bool:
        """True when a database is configured (DATABASE_URL present)."""
        return self._db_url is not None

    # -- internal -----------------------------------------------------------

    async def _execute(self, sql: str, *args) -> None:
        """Run a write statement, fail-open (log, never raise)."""
        try:
            pool = await get_db_pool(self._db_url)
            if pool is None:
                return
            async with pool.acquire() as conn:
                await conn.execute(sql, *args)
        except Exception as e:
            verbose_proxy_logger.error(f"GovernanceStore: write failed: {e}")

    async def _fetch(self, sql: str, *args):
        """Run a read query, fail-open (log, return [])."""
        try:
            pool = await get_db_pool(self._db_url)
            if pool is None:
                return []
            async with pool.acquire() as conn:
                return await conn.fetch(sql, *args)
        except Exception as e:
            verbose_proxy_logger.error(f"GovernanceStore: read failed: {e}")
            return []

    # -- Org ops ------------------------------------------------------------

    async def upsert_org(self, org: "OrgConfig") -> None:
        if not self.enabled:
            return
        c = _org_columns(org)
        await self._execute(
            """
            INSERT INTO governance_orgs (org_id, name, metadata, updated_at)
            VALUES ($1, $2, $3, NOW())
            ON CONFLICT (org_id) DO UPDATE SET
                name = EXCLUDED.name,
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
            """,
            c["org_id"],
            c["name"],
            c["metadata"],
        )

    async def delete_org(self, org_id: str) -> None:
        if not self.enabled:
            return
        await self._execute("DELETE FROM governance_orgs WHERE org_id = $1", org_id)

    async def load_all_orgs(self) -> list["OrgConfig"]:
        if not self.enabled:
            return []
        from .governance import OrgConfig

        rows = await self._fetch("SELECT * FROM governance_orgs")
        out: list[OrgConfig] = []
        for r in rows:
            out.append(
                OrgConfig(
                    org_id=r["org_id"],
                    name=r["name"],
                    metadata=json.loads(r["metadata"]) if r["metadata"] else {},
                )
            )
        return out

    # -- Workspace ops ------------------------------------------------------

    async def upsert_workspace(self, ws: "WorkspaceConfig") -> None:
        if not self.enabled:
            return
        c = _workspace_columns(ws)
        await self._execute(
            """
            INSERT INTO governance_workspaces (
                workspace_id, name, org_id, allowed_models, blocked_models,
                max_budget_usd, budget_alert_threshold, max_rpm, max_tpm,
                enforced_guardrails, default_routing_profile,
                config_override_allowed, metadata, updated_at
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13, NOW())
            ON CONFLICT (workspace_id) DO UPDATE SET
                name = EXCLUDED.name,
                org_id = EXCLUDED.org_id,
                allowed_models = EXCLUDED.allowed_models,
                blocked_models = EXCLUDED.blocked_models,
                max_budget_usd = EXCLUDED.max_budget_usd,
                budget_alert_threshold = EXCLUDED.budget_alert_threshold,
                max_rpm = EXCLUDED.max_rpm,
                max_tpm = EXCLUDED.max_tpm,
                enforced_guardrails = EXCLUDED.enforced_guardrails,
                default_routing_profile = EXCLUDED.default_routing_profile,
                config_override_allowed = EXCLUDED.config_override_allowed,
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
            """,
            c["workspace_id"],
            c["name"],
            c["org_id"],
            c["allowed_models"],
            c["blocked_models"],
            c["max_budget_usd"],
            c["budget_alert_threshold"],
            c["max_rpm"],
            c["max_tpm"],
            c["enforced_guardrails"],
            c["default_routing_profile"],
            c["config_override_allowed"],
            c["metadata"],
        )

    async def delete_workspace(self, workspace_id: str) -> None:
        if not self.enabled:
            return
        await self._execute(
            "DELETE FROM governance_workspaces WHERE workspace_id = $1", workspace_id
        )

    async def load_all_workspaces(self) -> list["WorkspaceConfig"]:
        if not self.enabled:
            return []
        from .governance import WorkspaceConfig

        rows = await self._fetch("SELECT * FROM governance_workspaces")
        out: list[WorkspaceConfig] = []
        for r in rows:
            out.append(
                WorkspaceConfig(
                    workspace_id=r["workspace_id"],
                    name=r["name"],
                    org_id=r["org_id"],
                    allowed_models=json.loads(r["allowed_models"])
                    if r["allowed_models"]
                    else [],
                    blocked_models=json.loads(r["blocked_models"])
                    if r["blocked_models"]
                    else [],
                    max_budget_usd=r["max_budget_usd"],
                    budget_alert_threshold=r["budget_alert_threshold"],
                    max_rpm=r["max_rpm"],
                    max_tpm=r["max_tpm"],
                    enforced_guardrails=json.loads(r["enforced_guardrails"])
                    if r["enforced_guardrails"]
                    else [],
                    default_routing_profile=r["default_routing_profile"],
                    config_override_allowed=r["config_override_allowed"],
                    metadata=json.loads(r["metadata"]) if r["metadata"] else {},
                )
            )
        return out

    # -- Key ops ------------------------------------------------------------

    async def upsert_key(self, kg: "KeyGovernance") -> None:
        if not self.enabled:
            return
        c = _key_columns(kg)
        await self._execute(
            """
            INSERT INTO governance_keys (
                key_id, workspace_id, scopes, max_budget_usd, budget_period,
                max_rpm, max_tpm, allowed_models, enforced_config,
                config_override_allowed, metadata, updated_at
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11, NOW())
            ON CONFLICT (key_id) DO UPDATE SET
                workspace_id = EXCLUDED.workspace_id,
                scopes = EXCLUDED.scopes,
                max_budget_usd = EXCLUDED.max_budget_usd,
                budget_period = EXCLUDED.budget_period,
                max_rpm = EXCLUDED.max_rpm,
                max_tpm = EXCLUDED.max_tpm,
                allowed_models = EXCLUDED.allowed_models,
                enforced_config = EXCLUDED.enforced_config,
                config_override_allowed = EXCLUDED.config_override_allowed,
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
            """,
            c["key_id"],
            c["workspace_id"],
            c["scopes"],
            c["max_budget_usd"],
            c["budget_period"],
            c["max_rpm"],
            c["max_tpm"],
            c["allowed_models"],
            c["enforced_config"],
            c["config_override_allowed"],
            c["metadata"],
        )

    async def delete_key(self, key_id: str) -> None:
        if not self.enabled:
            return
        await self._execute("DELETE FROM governance_keys WHERE key_id = $1", key_id)

    async def load_all_keys(self) -> list["KeyGovernance"]:
        if not self.enabled:
            return []
        from .governance import KeyGovernance

        rows = await self._fetch("SELECT * FROM governance_keys")
        out: list[KeyGovernance] = []
        for r in rows:
            out.append(
                KeyGovernance(
                    key_id=r["key_id"],
                    workspace_id=r["workspace_id"],
                    scopes=set(json.loads(r["scopes"])) if r["scopes"] else set(),
                    max_budget_usd=r["max_budget_usd"],
                    budget_period=r["budget_period"],
                    max_rpm=r["max_rpm"],
                    max_tpm=r["max_tpm"],
                    allowed_models=json.loads(r["allowed_models"])
                    if r["allowed_models"]
                    else [],
                    enforced_config=json.loads(r["enforced_config"])
                    if r["enforced_config"]
                    else None,
                    config_override_allowed=r["config_override_allowed"],
                    metadata=json.loads(r["metadata"]) if r["metadata"] else {},
                )
            )
        return out

    # -- Policy ops ---------------------------------------------------------

    async def upsert_policy(self, p: "UsagePolicy") -> None:
        if not self.enabled:
            return
        c = _policy_columns(p)
        await self._execute(
            """
            INSERT INTO usage_policies (
                policy_id, name, description, enabled, conditions, exclusions,
                group_by, limit_type, limit_value, limit_period, action,
                alert_threshold, priority, workspace_id, updated_at
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14, NOW())
            ON CONFLICT (policy_id) DO UPDATE SET
                name = EXCLUDED.name,
                description = EXCLUDED.description,
                enabled = EXCLUDED.enabled,
                conditions = EXCLUDED.conditions,
                exclusions = EXCLUDED.exclusions,
                group_by = EXCLUDED.group_by,
                limit_type = EXCLUDED.limit_type,
                limit_value = EXCLUDED.limit_value,
                limit_period = EXCLUDED.limit_period,
                action = EXCLUDED.action,
                alert_threshold = EXCLUDED.alert_threshold,
                priority = EXCLUDED.priority,
                workspace_id = EXCLUDED.workspace_id,
                updated_at = NOW()
            """,
            c["policy_id"],
            c["name"],
            c["description"],
            c["enabled"],
            c["conditions"],
            c["exclusions"],
            c["group_by"],
            c["limit_type"],
            c["limit_value"],
            c["limit_period"],
            c["action"],
            c["alert_threshold"],
            c["priority"],
            c["workspace_id"],
        )

    async def delete_policy(self, policy_id: str) -> None:
        if not self.enabled:
            return
        await self._execute(
            "DELETE FROM usage_policies WHERE policy_id = $1", policy_id
        )

    async def load_all_policies(self) -> list["UsagePolicy"]:
        if not self.enabled:
            return []
        from .usage_policies import LimitPeriod, LimitType, PolicyAction, UsagePolicy

        rows = await self._fetch("SELECT * FROM usage_policies")
        out: list[UsagePolicy] = []
        for r in rows:
            out.append(
                UsagePolicy(
                    policy_id=r["policy_id"],
                    name=r["name"],
                    description=r["description"] or "",
                    enabled=r["enabled"],
                    conditions=json.loads(r["conditions"]) if r["conditions"] else {},
                    exclusions=json.loads(r["exclusions"]) if r["exclusions"] else {},
                    group_by=json.loads(r["group_by"]) if r["group_by"] else [],
                    limit_type=LimitType(r["limit_type"]),
                    limit_value=r["limit_value"],
                    limit_period=LimitPeriod(r["limit_period"]),
                    action=PolicyAction(r["action"]),
                    alert_threshold=r["alert_threshold"],
                    priority=r["priority"],
                    workspace_id=r["workspace_id"],
                )
            )
        return out

    # -- Spend ops (the durable budget system-of-record) --------------------

    async def record_spend(
        self,
        scope: str,
        scope_type: str,
        period: str,
        period_start: datetime,
        *,
        cost: float = 0.0,
        tokens: int = 0,
        requests: int = 1,
    ) -> None:
        """Atomically accumulate spend/tokens/requests for a scope+period.

        No-op when the store is disabled.  Uses ``ON CONFLICT DO UPDATE`` so the
        accumulation is atomic in Postgres (no read-modify-write race).
        """
        if not self.enabled:
            return
        await self._execute(
            """
            INSERT INTO governance_spend (
                scope, scope_type, period_start, period,
                spend_usd, request_count, total_tokens, updated_at
            ) VALUES ($1,$2,$3,$4,$5,$6,$7, NOW())
            ON CONFLICT (scope, period, period_start) DO UPDATE SET
                spend_usd = governance_spend.spend_usd + EXCLUDED.spend_usd,
                request_count = governance_spend.request_count
                    + EXCLUDED.request_count,
                total_tokens = governance_spend.total_tokens
                    + EXCLUDED.total_tokens,
                updated_at = NOW()
            """,
            scope,
            scope_type,
            period_start,
            period,
            float(cost or 0.0),
            int(requests or 0),
            int(tokens or 0),
        )

    async def get_spend(self, scope: str, period: str, period_start: datetime) -> float:
        """Read durable spend for a scope+period.  Returns 0.0 when disabled."""
        if not self.enabled:
            return 0.0
        rows = await self._fetch(
            """
            SELECT spend_usd FROM governance_spend
            WHERE scope = $1 AND period = $2 AND period_start = $3
            """,
            scope,
            period,
            period_start,
        )
        if rows:
            return float(rows[0]["spend_usd"])
        return 0.0


# =============================================================================
# Singleton
# =============================================================================

_governance_store: GovernanceStore | None = None


def get_governance_store() -> GovernanceStore:
    """Get the global governance store singleton."""
    global _governance_store
    if _governance_store is None:
        _governance_store = GovernanceStore()
    return _governance_store


def reset_governance_store() -> None:
    """Reset the governance store singleton (for testing)."""
    global _governance_store
    _governance_store = None


__all__ = [
    "GOVERNANCE_TABLES_SQL",
    "run_governance_migrations",
    "GovernanceStore",
    "get_governance_store",
    "reset_governance_store",
]
