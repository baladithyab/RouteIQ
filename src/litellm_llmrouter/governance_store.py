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
  * ``guardrail_policies``     -- mirror of ``GuardrailPolicy`` (14-check engine)
  * ``prompts``                -- mirror of ``PromptDefinition`` (versioning/A-B)

The ``guardrail_policies`` / ``prompts`` rows store the full Pydantic model as a
single JSONB ``payload`` column (plus a few indexed scalar columns) so the load
side is a drift-free ``model_validate`` round-trip -- the policy/prompt schemas
are nested (enums, version maps, A/B weights) and a column-per-field mirror
would be brittle.

Soft FKs only (no PG ``REFERENCES``) to preserve the additive contract and
tolerate hydration ordering on a partial restore.

NOTE: this is the asyncpg (raw SQL) path reached through ``database.run_migrations``
-- NOT the leader-gated Prisma path in ``migrations.py``.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

from litellm._logging import verbose_proxy_logger

from .database import get_database_url, get_db_pool

if TYPE_CHECKING:  # avoid import cycle at runtime; only needed for typing
    from .governance import KeyGovernance, OrgConfig, WorkspaceConfig
    from .guardrail_policies import GuardrailPolicy
    from .prompt_management import PromptDefinition
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

-- Durable per-model spend rollup (the chargeback breakdown system-of-record).
-- The scope-level governance_spend table has no model dimension, so this side
-- table carries the (model x scope x period) cost/token aggregate the spend
-- report renders as per-model breakdowns.  Same epoch-aligned period_start as
-- governance_spend so the two tables roll up on the same bucket boundaries.
CREATE TABLE IF NOT EXISTS governance_spend_model (
    model            VARCHAR(255) NOT NULL,
    scope            VARCHAR(255) NOT NULL DEFAULT 'global',
    scope_type       VARCHAR(20)  NOT NULL DEFAULT 'global',
    period_start     TIMESTAMP WITH TIME ZONE NOT NULL,
    period           VARCHAR(20)  NOT NULL DEFAULT 'monthly',
    spend_usd        DOUBLE PRECISION NOT NULL DEFAULT 0,
    request_count    BIGINT NOT NULL DEFAULT 0,
    input_tokens     BIGINT NOT NULL DEFAULT 0,
    output_tokens    BIGINT NOT NULL DEFAULT 0,
    updated_at       TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    PRIMARY KEY (model, scope, period, period_start)
);
CREATE INDEX IF NOT EXISTS idx_gov_spend_model_model
    ON governance_spend_model(model);
CREATE INDEX IF NOT EXISTS idx_gov_spend_model_period
    ON governance_spend_model(period_start);

-- Guardrail policy definitions (mirror of GuardrailPolicy, the 14-check engine).
-- The full model is stored as JSONB ``payload`` (nested enums) + indexed scalars.
CREATE TABLE IF NOT EXISTS guardrail_policies (
    guardrail_id  VARCHAR(255) PRIMARY KEY,
    name          VARCHAR(255) NOT NULL,
    workspace_id  VARCHAR(255),
    payload       JSONB NOT NULL,
    updated_at    TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_guardrail_policies_ws
    ON guardrail_policies(workspace_id);

-- Prompt definitions (mirror of PromptDefinition: versions / A-B / rollback).
-- ``storage_key`` is the manager's composite key ("workspace::name" or "name").
CREATE TABLE IF NOT EXISTS prompts (
    storage_key   VARCHAR(512) PRIMARY KEY,
    name          VARCHAR(255) NOT NULL,
    workspace_id  VARCHAR(255),
    payload       JSONB NOT NULL,
    updated_at    TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_prompts_ws ON prompts(workspace_id);
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


def _guardrail_columns(g: "GuardrailPolicy") -> dict:
    """Map a ``GuardrailPolicy`` to its ``guardrail_policies`` column dict.

    The whole model is serialized to JSONB ``payload`` (Pydantic ``model_dump``
    with ``mode="json"`` so the nested enums become their ``.value`` strings).
    ``guardrail_id`` / ``name`` / ``workspace_id`` are duplicated as indexed
    scalar columns for the PK + scope filtering.
    """
    return {
        "guardrail_id": g.guardrail_id,
        "name": g.name,
        "workspace_id": g.workspace_id,
        "payload": json.dumps(g.model_dump(mode="json")),
    }


def _prompt_columns(storage_key: str, p: "PromptDefinition") -> dict:
    """Map a ``PromptDefinition`` to its ``prompts`` column dict.

    ``storage_key`` is the manager's composite key (``workspace::name`` or
    ``name``) and serves as the PK so workspace-scoped prompts stay distinct.
    The full model (versions / A-B weights / metadata) is the JSONB ``payload``.
    """
    return {
        "storage_key": storage_key,
        "name": p.name,
        "workspace_id": p.workspace_id,
        "payload": json.dumps(p.model_dump(mode="json")),
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

    # -- Guardrail policy ops -----------------------------------------------

    async def upsert_guardrail(self, g: "GuardrailPolicy") -> None:
        if not self.enabled:
            return
        c = _guardrail_columns(g)
        await self._execute(
            """
            INSERT INTO guardrail_policies (
                guardrail_id, name, workspace_id, payload, updated_at
            ) VALUES ($1,$2,$3,$4, NOW())
            ON CONFLICT (guardrail_id) DO UPDATE SET
                name = EXCLUDED.name,
                workspace_id = EXCLUDED.workspace_id,
                payload = EXCLUDED.payload,
                updated_at = NOW()
            """,
            c["guardrail_id"],
            c["name"],
            c["workspace_id"],
            c["payload"],
        )

    async def delete_guardrail(self, guardrail_id: str) -> None:
        if not self.enabled:
            return
        await self._execute(
            "DELETE FROM guardrail_policies WHERE guardrail_id = $1", guardrail_id
        )

    async def load_all_guardrails(self) -> list["GuardrailPolicy"]:
        if not self.enabled:
            return []
        from .guardrail_policies import GuardrailPolicy

        rows = await self._fetch("SELECT payload FROM guardrail_policies")
        out: list[GuardrailPolicy] = []
        for r in rows:
            payload = r["payload"]
            data = json.loads(payload) if isinstance(payload, str) else payload
            out.append(GuardrailPolicy.model_validate(data))
        return out

    # -- Prompt ops ---------------------------------------------------------

    async def upsert_prompt(self, storage_key: str, p: "PromptDefinition") -> None:
        if not self.enabled:
            return
        c = _prompt_columns(storage_key, p)
        await self._execute(
            """
            INSERT INTO prompts (
                storage_key, name, workspace_id, payload, updated_at
            ) VALUES ($1,$2,$3,$4, NOW())
            ON CONFLICT (storage_key) DO UPDATE SET
                name = EXCLUDED.name,
                workspace_id = EXCLUDED.workspace_id,
                payload = EXCLUDED.payload,
                updated_at = NOW()
            """,
            c["storage_key"],
            c["name"],
            c["workspace_id"],
            c["payload"],
        )

    async def delete_prompt(self, storage_key: str) -> None:
        if not self.enabled:
            return
        await self._execute("DELETE FROM prompts WHERE storage_key = $1", storage_key)

    async def load_all_prompts(self) -> list[tuple[str, "PromptDefinition"]]:
        """Load all prompts as ``(storage_key, PromptDefinition)`` pairs.

        The storage key is returned alongside the model so the caller can hydrate
        ``PromptManager._prompts`` (keyed by the composite ``workspace::name``)
        without re-deriving the key.
        """
        if not self.enabled:
            return []
        from .prompt_management import PromptDefinition

        rows = await self._fetch("SELECT storage_key, payload FROM prompts")
        out: list[tuple[str, PromptDefinition]] = []
        for r in rows:
            payload = r["payload"]
            data = json.loads(payload) if isinstance(payload, str) else payload
            out.append((r["storage_key"], PromptDefinition.model_validate(data)))
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

    async def record_model_spend(
        self,
        model: str,
        scope: str,
        scope_type: str,
        period: str,
        period_start: datetime,
        *,
        cost: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        requests: int = 1,
    ) -> None:
        """Atomically accumulate per-model spend for the chargeback breakdown.

        The companion of :meth:`record_spend` carrying the model dimension the
        scope-level table lacks.  No-op when disabled.  ``ON CONFLICT DO UPDATE``
        keeps the accumulation atomic (no read-modify-write race).
        """
        if not self.enabled:
            return
        await self._execute(
            """
            INSERT INTO governance_spend_model (
                model, scope, scope_type, period_start, period,
                spend_usd, request_count, input_tokens, output_tokens, updated_at
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9, NOW())
            ON CONFLICT (model, scope, period, period_start) DO UPDATE SET
                spend_usd = governance_spend_model.spend_usd + EXCLUDED.spend_usd,
                request_count = governance_spend_model.request_count
                    + EXCLUDED.request_count,
                input_tokens = governance_spend_model.input_tokens
                    + EXCLUDED.input_tokens,
                output_tokens = governance_spend_model.output_tokens
                    + EXCLUDED.output_tokens,
                updated_at = NOW()
            """,
            model,
            scope,
            scope_type,
            period_start,
            period,
            float(cost or 0.0),
            int(requests or 0),
            int(input_tokens or 0),
            int(output_tokens or 0),
        )

    async def aggregate_spend_report(self, *, since: datetime | None = None) -> dict:
        """Aggregate the durable spend ledger into a chargeback report.

        Returns real totals plus per-scope (key/team/workspace) and per-model
        breakdowns and daily rollups, computed entirely in SQL from the
        ``governance_spend`` / ``governance_spend_model`` rollup tables.  An
        empty (zero) report when disabled or when no spend has been recorded --
        the caller (``get_spend_report``) renders it either way.

        Args:
            since: Optional lower bound on ``period_start`` -- when provided only
                buckets at/after this instant are summed.
        """
        empty = {
            "total_cost_usd": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_requests": 0,
            "by_scope": [],
            "by_model": [],
            "daily": [],
        }
        if not self.enabled:
            return empty

        # -- Totals + per-scope breakdown (from the scope-level ledger). --------
        scope_where = "WHERE period_start >= $1" if since is not None else ""
        scope_args: tuple = (since,) if since is not None else ()
        scope_rows = await self._fetch(
            f"""
            SELECT scope, scope_type,
                   COALESCE(SUM(spend_usd), 0)     AS spend_usd,
                   COALESCE(SUM(request_count), 0) AS request_count,
                   COALESCE(SUM(total_tokens), 0)  AS total_tokens
            FROM governance_spend
            {scope_where}
            GROUP BY scope, scope_type
            ORDER BY spend_usd DESC
            """,
            *scope_args,
        )

        # -- Per-model breakdown (from the model-level ledger). ----------------
        model_where = "WHERE period_start >= $1" if since is not None else ""
        model_rows = await self._fetch(
            f"""
            SELECT model,
                   COALESCE(SUM(spend_usd), 0)     AS spend_usd,
                   COALESCE(SUM(request_count), 0) AS request_count,
                   COALESCE(SUM(input_tokens), 0)  AS input_tokens,
                   COALESCE(SUM(output_tokens), 0) AS output_tokens
            FROM governance_spend_model
            {model_where}
            GROUP BY model
            ORDER BY spend_usd DESC
            """,
            *scope_args,
        )

        # -- Daily rollup (calendar-day truncation of the bucket period_start). -
        daily_rows = await self._fetch(
            f"""
            SELECT date_trunc('day', period_start) AS day,
                   COALESCE(SUM(spend_usd), 0)     AS spend_usd,
                   COALESCE(SUM(request_count), 0) AS request_count
            FROM governance_spend
            {scope_where}
            GROUP BY day
            ORDER BY day
            """,
            *scope_args,
        )

        by_scope = [
            {
                "scope": r["scope"],
                "scope_type": r["scope_type"],
                "spend_usd": float(r["spend_usd"]),
                "request_count": int(r["request_count"]),
                "total_tokens": int(r["total_tokens"]),
            }
            for r in scope_rows
        ]
        by_model = [
            {
                "model": r["model"],
                "spend_usd": float(r["spend_usd"]),
                "request_count": int(r["request_count"]),
                "input_tokens": int(r["input_tokens"]),
                "output_tokens": int(r["output_tokens"]),
            }
            for r in model_rows
        ]
        daily = [
            {
                "date": r["day"].date().isoformat()
                if hasattr(r["day"], "date")
                else str(r["day"]),
                "spend_usd": float(r["spend_usd"]),
                "request_count": int(r["request_count"]),
            }
            for r in daily_rows
        ]

        return {
            "total_cost_usd": round(sum(s["spend_usd"] for s in by_scope), 6),
            "total_input_tokens": sum(m["input_tokens"] for m in by_model),
            "total_output_tokens": sum(m["output_tokens"] for m in by_model),
            "total_requests": sum(s["request_count"] for s in by_scope),
            "by_scope": by_scope,
            "by_model": by_model,
            "daily": daily,
        }


# =============================================================================
# One-time legacy self-service key backfill (RouteIQ-a433)
# =============================================================================


def _mask_legacy_secret(secret: str) -> str:
    """Non-secret preview of a plaintext key (prefix + last-4).

    Mirrors the user-portal ``_mask_secret`` so a backfilled row's ``masked``
    preview is byte-identical to one a freshly-created key would carry.
    """
    last4 = secret[-4:] if len(secret) >= 4 else ""
    return f"sk-rq-...{last4}"


def backfill_self_service_key_metadata(engine: Any) -> int:
    """Backfill legacy self-service keys to the hashed-secret schema.

    Self-service keys created before the hashed-secret migration stored the raw
    api_key as the governance ``key_id`` and carried no ``public_id`` /
    ``secret_hash`` / ``masked`` in ``metadata``. The user portal falls back to
    the raw ``key_id`` as the public id for those rows -- which LEAKS the secret
    as the addressable id. This one-time, idempotent backfill stamps the three
    missing fields so legacy rows match the current schema:

      * ``public_id``   -- a fresh non-secret ``kid_<token>`` id
      * ``secret_hash`` -- SHA-256 of the raw secret (the legacy ``key_id``)
      * ``masked``      -- the ``sk-rq-...<last4>`` preview

    A row is a backfill candidate when ``metadata.self_service`` is truthy and
    ``metadata.public_id`` is absent. Rows already carrying a ``public_id`` (new
    schema) and non-self-service rows are left untouched, so running this twice
    is a no-op (idempotent). Operates purely on the in-memory engine state; the
    caller persists via the engine's normal save path (file/durable store).

    Returns the number of rows backfilled (0 when nothing matched).
    """
    import hashlib
    import secrets as _secrets

    key_governance = getattr(engine, "_key_governance", None)
    if not key_governance:
        return 0

    count = 0
    for kg in list(key_governance.values()):
        metadata = getattr(kg, "metadata", None)
        if not isinstance(metadata, dict):
            continue
        if not metadata.get("self_service"):
            continue
        if metadata.get("public_id"):
            continue  # already migrated -> idempotent skip

        raw_secret = str(getattr(kg, "key_id", "") or "")
        metadata["public_id"] = f"kid_{_secrets.token_urlsafe(12)}"
        metadata["secret_hash"] = hashlib.sha256(raw_secret.encode("utf-8")).hexdigest()
        metadata["masked"] = _mask_legacy_secret(raw_secret)
        count += 1
        verbose_proxy_logger.info(
            "Backfilled legacy self-service key %s -> public_id=%s",
            raw_secret[:6] + "..." if raw_secret else "?",
            metadata["public_id"],
        )

    if count:
        verbose_proxy_logger.info(
            "Self-service key backfill: migrated %d legacy row(s)", count
        )
    return count


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
    "backfill_self_service_key_metadata",
]
