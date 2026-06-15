"""Unit tests for the durable governance store (Aurora-backed, DB-optional).

These tests run with NO database (DATABASE_URL unset) and assert the
disabled-path contract: every method is a fail-open no-op, the migration SQL is
idempotent-shaped and covers all five tables, and the pure serialize helpers map
the Pydantic models 1:1 (a guard against schema/model drift).
"""

import json

import pytest

from litellm_llmrouter.governance import KeyGovernance, OrgConfig, WorkspaceConfig
from litellm_llmrouter.governance_store import (
    GOVERNANCE_TABLES_SQL,
    GovernanceStore,
    _key_columns,
    _org_columns,
    _policy_columns,
    _workspace_columns,
    get_governance_store,
    reset_governance_store,
    run_governance_migrations,
)
from litellm_llmrouter.usage_policies import (
    LimitPeriod,
    LimitType,
    PolicyAction,
    UsagePolicy,
)


# -- Disabled-path contract --------------------------------------------------


def test_store_disabled_when_no_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    reset_governance_store()
    store = get_governance_store()
    assert store.enabled is False


def test_store_enabled_when_database_url_set(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u@h:5432/db")
    reset_governance_store()
    store = get_governance_store()
    assert store.enabled is True


@pytest.mark.asyncio
async def test_disabled_crud_are_noops(monkeypatch):
    """With the store disabled every CRUD method is a no-op that does not raise."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    reset_governance_store()
    store = get_governance_store()
    assert store.enabled is False

    ws = WorkspaceConfig(workspace_id="ws-1", name="WS One")
    kg = KeyGovernance(key_id="key-1", workspace_id="ws-1")
    org = OrgConfig(org_id="org-1", name="Org One")
    policy = UsagePolicy(policy_id="p-1", name="P One")

    # Writes/deletes: no-op, no raise.
    await store.upsert_org(org)
    await store.upsert_workspace(ws)
    await store.upsert_key(kg)
    await store.upsert_policy(policy)
    await store.delete_org("org-1")
    await store.delete_workspace("ws-1")
    await store.delete_key("key-1")
    await store.delete_policy("p-1")

    # Loads: return empty lists.
    assert await store.load_all_orgs() == []
    assert await store.load_all_workspaces() == []
    assert await store.load_all_keys() == []
    assert await store.load_all_policies() == []


@pytest.mark.asyncio
async def test_disabled_spend_ops_are_noops(monkeypatch):
    from datetime import datetime, timezone

    monkeypatch.delenv("DATABASE_URL", raising=False)
    reset_governance_store()
    store = get_governance_store()

    ps = datetime.fromtimestamp(0, tz=timezone.utc)
    # record_spend no-ops, get_spend returns 0.0
    await store.record_spend("ws-1", "workspace", "monthly", ps, cost=1.5, requests=1)
    assert await store.get_spend("ws-1", "monthly", ps) == 0.0


@pytest.mark.asyncio
async def test_run_migrations_noop_without_db(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    # Early-return path: must not raise.
    await run_governance_migrations()


def test_reset_clears_singleton(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    reset_governance_store()
    first = get_governance_store()
    second = get_governance_store()
    assert first is second
    reset_governance_store()
    third = get_governance_store()
    assert third is not first


# -- Migration SQL shape -----------------------------------------------------


def test_migration_sql_covers_all_tables():
    for table in (
        "governance_orgs",
        "governance_workspaces",
        "governance_keys",
        "usage_policies",
        "governance_spend",
    ):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in GOVERNANCE_TABLES_SQL


def test_migration_sql_is_idempotent_shaped():
    # Every CREATE TABLE / CREATE INDEX must be IF NOT EXISTS.
    for stmt in GOVERNANCE_TABLES_SQL.split(";"):
        s = stmt.strip().upper()
        if s.startswith("CREATE TABLE"):
            assert "IF NOT EXISTS" in s, stmt
        if s.startswith("CREATE INDEX"):
            assert "IF NOT EXISTS" in s, stmt


def test_spend_table_has_composite_pk():
    assert "PRIMARY KEY (scope, period, period_start)" in GOVERNANCE_TABLES_SQL


# -- Serialize-helper round-trip (model/schema drift guard, no DB) ----------


def test_workspace_columns_cover_model_fields():
    ws = WorkspaceConfig(
        workspace_id="ws-1",
        name="WS One",
        org_id="org-1",
        allowed_models=["gpt-4o*"],
        blocked_models=["bad-*"],
        max_budget_usd=100.0,
        max_rpm=60,
        enforced_guardrails=["g1"],
        metadata={"k": "v"},
    )
    cols = _workspace_columns(ws)
    assert cols["workspace_id"] == "ws-1"
    assert cols["org_id"] == "org-1"
    # JSONB columns are json-serialized strings.
    assert json.loads(cols["allowed_models"]) == ["gpt-4o*"]
    assert json.loads(cols["blocked_models"]) == ["bad-*"]
    assert json.loads(cols["enforced_guardrails"]) == ["g1"]
    assert json.loads(cols["metadata"]) == {"k": "v"}
    assert cols["max_budget_usd"] == 100.0
    assert cols["max_rpm"] == 60


def test_key_columns_serialize_scopes_set_deterministically():
    kg = KeyGovernance(
        key_id="key-1",
        workspace_id="ws-1",
        scopes={"completions.write", "embeddings.read"},
        allowed_models=["gpt-4o"],
        enforced_config={"routing_profile": "fast"},
    )
    cols = _key_columns(kg)
    # scopes stored as a SORTED json array (deterministic) -> coerced to set on load.
    assert json.loads(cols["scopes"]) == sorted(
        ["completions.write", "embeddings.read"]
    )
    assert json.loads(cols["allowed_models"]) == ["gpt-4o"]
    assert json.loads(cols["enforced_config"]) == {"routing_profile": "fast"}


def test_key_columns_null_enforced_config():
    kg = KeyGovernance(key_id="key-2", workspace_id=None)
    cols = _key_columns(kg)
    assert cols["enforced_config"] is None


def test_policy_columns_serialize_enums_as_values():
    p = UsagePolicy(
        policy_id="p-1",
        name="P One",
        conditions={"model": "gpt-4o*"},
        group_by=["metadata._user"],
        limit_type=LimitType.COST,
        limit_value=50.0,
        limit_period=LimitPeriod.MONTH,
        action=PolicyAction.DENY,
        workspace_id="ws-1",
    )
    cols = _policy_columns(p)
    assert cols["limit_type"] == "cost"
    assert cols["limit_period"] == "month"
    assert cols["action"] == "deny"
    assert json.loads(cols["conditions"]) == {"model": "gpt-4o*"}
    assert json.loads(cols["group_by"]) == ["metadata._user"]
    assert cols["workspace_id"] == "ws-1"


def test_org_columns():
    org = OrgConfig(org_id="org-1", name="Org One", metadata={"tier": "gold"})
    cols = _org_columns(org)
    assert cols["org_id"] == "org-1"
    assert cols["name"] == "Org One"
    assert json.loads(cols["metadata"]) == {"tier": "gold"}


# -- Construction reads env at init (matches AuditLogRepository) -------------


def test_construction_reads_env_at_init(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u@h:5432/db")
    s1 = GovernanceStore()
    assert s1.enabled is True
    monkeypatch.delenv("DATABASE_URL", raising=False)
    s2 = GovernanceStore()
    assert s2.enabled is False
