"""Unit tests for the durable (Aurora-backed) governance store -- ENABLED path.

The sibling ``test_governance_store.py`` exercises the *disabled* (no
``DATABASE_URL``) contract: every method is a fail-open no-op.  This module
covers the complementary *enabled* path with a **mocked asyncpg pool** (no real
database, no AWS creds) and asserts:

  * each CRUD/spend method issues the EXPECTED SQL against an acquired
    connection (durable WRITE actually happens, not a silent skip),
  * ``load_all_*`` round-trips rows back into the Pydantic models with the
    correct types (set-coercion for scopes, enum-coercion for policies, JSON
    decode for JSONB columns) -- a guard against the load-side / schema drift,
  * ``record_spend`` uses the atomic ``ON CONFLICT DO UPDATE`` accumulator and
    ``get_spend`` reads the durable rollup,
  * a pool that is ``None`` (asyncpg unavailable) stays fail-open even when the
    store is *enabled* (read returns 0.0/[], write does not raise),
  * the GovernanceEngine durable-spend fallback (Redis miss -> Aurora rollup)
    reaches ``store.get_spend`` when the store is enabled.

All SQL is asserted on the statement text the store passes to ``conn.execute`` /
``conn.fetch``; the fake connection records args so the test can prove the
column values + bind params without a live Postgres.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from litellm_llmrouter.governance import (
    GovernanceContext,
    GovernanceEngine,
    KeyGovernance,
    OrgConfig,
    WorkspaceConfig,
)
from litellm_llmrouter.governance_store import (
    GovernanceStore,
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

_DB_URL = "postgresql://u@h:5432/db"


# ---------------------------------------------------------------------------
# Fake asyncpg pool / connection (records execute() + serves fetch())
# ---------------------------------------------------------------------------


class _FakeConn:
    """Records every execute() and answers fetch() from a queued result list."""

    def __init__(self, fetch_results: list | None = None) -> None:
        self.executed: list[tuple[str, tuple]] = []
        self.fetched: list[tuple[str, tuple]] = []
        self._fetch_results = list(fetch_results or [])

    async def execute(self, sql: str, *args) -> None:
        self.executed.append((sql, args))

    async def fetch(self, sql: str, *args):
        self.fetched.append((sql, args))
        if self._fetch_results:
            return self._fetch_results.pop(0)
        return []


class _FakePool:
    """Minimal asyncpg.Pool stand-in: ``async with pool.acquire() as conn``."""

    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    def acquire(self):
        conn = self._conn

        @asynccontextmanager
        async def _cm():
            yield conn

        return _cm()


def _patch_pool(conn: _FakeConn):
    """Patch ``governance_store.get_db_pool`` to hand back a fake pool."""
    return patch(
        "litellm_llmrouter.governance_store.get_db_pool",
        AsyncMock(return_value=_FakePool(conn)),
    )


@pytest.fixture
def enabled_store(monkeypatch):
    """A GovernanceStore that believes a DB is configured (enabled=True)."""
    monkeypatch.setenv("DATABASE_URL", _DB_URL)
    reset_governance_store()
    store = get_governance_store()
    assert store.enabled is True
    return store


# ---------------------------------------------------------------------------
# Durable WRITE: each CRUD method issues the expected SQL
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_org_writes_durably(enabled_store):
    conn = _FakeConn()
    with _patch_pool(conn):
        await enabled_store.upsert_org(
            OrgConfig(org_id="org-1", name="Org One", metadata={"tier": "gold"})
        )
    assert len(conn.executed) == 1
    sql, args = conn.executed[0]
    assert "INSERT INTO governance_orgs" in sql
    assert "ON CONFLICT (org_id) DO UPDATE" in sql
    assert args[0] == "org-1"
    assert args[1] == "Org One"


@pytest.mark.asyncio
async def test_upsert_workspace_writes_all_columns(enabled_store):
    conn = _FakeConn()
    ws = WorkspaceConfig(
        workspace_id="ws-1",
        name="WS One",
        org_id="org-1",
        allowed_models=["gpt-4o*"],
        max_budget_usd=100.0,
        max_rpm=60,
    )
    with _patch_pool(conn):
        await enabled_store.upsert_workspace(ws)
    sql, args = conn.executed[0]
    assert "INSERT INTO governance_workspaces" in sql
    assert "ON CONFLICT (workspace_id) DO UPDATE" in sql
    # workspace_id, name, org_id are the first three positional binds.
    assert args[0] == "ws-1"
    assert args[1] == "WS One"
    assert args[2] == "org-1"


@pytest.mark.asyncio
async def test_upsert_key_writes_durably(enabled_store):
    conn = _FakeConn()
    kg = KeyGovernance(
        key_id="key-1",
        workspace_id="ws-1",
        scopes={"completions.write", "embeddings.read"},
    )
    with _patch_pool(conn):
        await enabled_store.upsert_key(kg)
    sql, args = conn.executed[0]
    assert "INSERT INTO governance_keys" in sql
    assert "ON CONFLICT (key_id) DO UPDATE" in sql
    assert args[0] == "key-1"
    assert args[1] == "ws-1"


@pytest.mark.asyncio
async def test_upsert_policy_writes_durably(enabled_store):
    conn = _FakeConn()
    p = UsagePolicy(
        policy_id="p-1",
        name="P One",
        limit_type=LimitType.COST,
        limit_period=LimitPeriod.MONTH,
        action=PolicyAction.DENY,
        workspace_id="ws-1",
    )
    with _patch_pool(conn):
        await enabled_store.upsert_policy(p)
    sql, args = conn.executed[0]
    assert "INSERT INTO usage_policies" in sql
    assert "ON CONFLICT (policy_id) DO UPDATE" in sql
    assert args[0] == "p-1"
    # enums serialize to their .value strings (VARCHAR, not PG enum).
    assert "cost" in args
    assert "month" in args
    assert "deny" in args


@pytest.mark.asyncio
async def test_delete_methods_issue_delete_sql(enabled_store):
    conn = _FakeConn()
    with _patch_pool(conn):
        await enabled_store.delete_org("org-1")
        await enabled_store.delete_workspace("ws-1")
        await enabled_store.delete_key("key-1")
        await enabled_store.delete_policy("p-1")
    stmts = [s for s, _ in conn.executed]
    assert any("DELETE FROM governance_orgs" in s for s in stmts)
    assert any("DELETE FROM governance_workspaces" in s for s in stmts)
    assert any("DELETE FROM governance_keys" in s for s in stmts)
    assert any("DELETE FROM usage_policies" in s for s in stmts)
    # The deleted id is the sole bind on each delete.
    assert all(len(a) == 1 for _, a in conn.executed)


# ---------------------------------------------------------------------------
# Durable READ: load_all_* round-trips rows into the Pydantic models
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_all_orgs_round_trip(enabled_store):
    rows = [
        {"org_id": "org-1", "name": "Org One", "metadata": '{"tier": "gold"}'},
    ]
    conn = _FakeConn(fetch_results=[rows])
    with _patch_pool(conn):
        orgs = await enabled_store.load_all_orgs()
    assert len(orgs) == 1
    assert isinstance(orgs[0], OrgConfig)
    assert orgs[0].org_id == "org-1"
    assert orgs[0].metadata == {"tier": "gold"}
    assert "SELECT * FROM governance_orgs" in conn.fetched[0][0]


@pytest.mark.asyncio
async def test_load_all_workspaces_decodes_jsonb(enabled_store):
    rows = [
        {
            "workspace_id": "ws-1",
            "name": "WS One",
            "org_id": "org-1",
            "allowed_models": '["gpt-4o*"]',
            "blocked_models": "[]",
            "max_budget_usd": 100.0,
            "budget_alert_threshold": 0.8,
            "max_rpm": 60,
            "max_tpm": None,
            "enforced_guardrails": '["g1"]',
            "default_routing_profile": None,
            "config_override_allowed": True,
            "metadata": '{"k": "v"}',
        }
    ]
    conn = _FakeConn(fetch_results=[rows])
    with _patch_pool(conn):
        wss = await enabled_store.load_all_workspaces()
    assert len(wss) == 1
    ws = wss[0]
    assert isinstance(ws, WorkspaceConfig)
    assert ws.allowed_models == ["gpt-4o*"]
    assert ws.enforced_guardrails == ["g1"]
    assert ws.metadata == {"k": "v"}
    assert ws.max_budget_usd == 100.0


@pytest.mark.asyncio
async def test_load_all_keys_coerces_scopes_to_set(enabled_store):
    rows = [
        {
            "key_id": "key-1",
            "workspace_id": "ws-1",
            "scopes": '["completions.write", "embeddings.read"]',
            "max_budget_usd": None,
            "budget_period": "monthly",
            "max_rpm": None,
            "max_tpm": None,
            "allowed_models": '["gpt-4o"]',
            "enforced_config": '{"routing_profile": "fast"}',
            "config_override_allowed": True,
            "metadata": "{}",
        }
    ]
    conn = _FakeConn(fetch_results=[rows])
    with _patch_pool(conn):
        keys = await enabled_store.load_all_keys()
    assert len(keys) == 1
    kg = keys[0]
    assert isinstance(kg, KeyGovernance)
    # scopes is a Set[str] on the model; JSON array must coerce back to set.
    assert kg.scopes == {"completions.write", "embeddings.read"}
    assert isinstance(kg.scopes, set)
    assert kg.allowed_models == ["gpt-4o"]
    assert kg.enforced_config == {"routing_profile": "fast"}


@pytest.mark.asyncio
async def test_load_all_policies_coerces_enums(enabled_store):
    rows = [
        {
            "policy_id": "p-1",
            "name": "P One",
            "description": "",
            "enabled": True,
            "conditions": '{"model": "gpt-4o*"}',
            "exclusions": "{}",
            "group_by": '["metadata._user"]',
            "limit_type": "cost",
            "limit_value": 50.0,
            "limit_period": "month",
            "action": "deny",
            "alert_threshold": 0.8,
            "priority": 100,
            "workspace_id": "ws-1",
        }
    ]
    conn = _FakeConn(fetch_results=[rows])
    with _patch_pool(conn):
        policies = await enabled_store.load_all_policies()
    assert len(policies) == 1
    p = policies[0]
    assert isinstance(p, UsagePolicy)
    # VARCHAR -> enum coercion on the load side.
    assert p.limit_type is LimitType.COST
    assert p.limit_period is LimitPeriod.MONTH
    assert p.action is PolicyAction.DENY
    assert p.conditions == {"model": "gpt-4o*"}
    assert p.group_by == ["metadata._user"]


# ---------------------------------------------------------------------------
# Guardrail policy ops (RouteIQ-4f30): durable WRITE + round-trip READ
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_guardrail_writes_durably(enabled_store):
    from litellm_llmrouter.guardrail_policies import (
        GuardrailAction,
        GuardrailPhase,
        GuardrailPolicy,
        GuardrailType,
    )

    conn = _FakeConn()
    g = GuardrailPolicy(
        guardrail_id="g-1",
        name="Deny Secrets",
        check_type=GuardrailType.REGEX_DENY,
        action=GuardrailAction.DENY,
        phase=GuardrailPhase.INPUT,
        parameters={"patterns": ["(?i)password"]},
        workspace_id="ws-1",
    )
    with _patch_pool(conn):
        await enabled_store.upsert_guardrail(g)
    sql, args = conn.executed[0]
    assert "INSERT INTO guardrail_policies" in sql
    assert "ON CONFLICT (guardrail_id) DO UPDATE" in sql
    assert args[0] == "g-1"
    assert args[1] == "Deny Secrets"
    assert args[2] == "ws-1"
    # The full model is serialized to the JSONB payload (enum -> .value string).
    import json as _json

    payload = _json.loads(args[3])
    assert payload["check_type"] == "regex_deny"
    assert payload["action"] == "deny"
    assert payload["parameters"]["patterns"] == ["(?i)password"]


@pytest.mark.asyncio
async def test_delete_guardrail_issues_delete_sql(enabled_store):
    conn = _FakeConn()
    with _patch_pool(conn):
        await enabled_store.delete_guardrail("g-1")
    sql, args = conn.executed[0]
    assert "DELETE FROM guardrail_policies" in sql
    assert args == ("g-1",)


@pytest.mark.asyncio
async def test_load_all_guardrails_round_trip(enabled_store):
    from litellm_llmrouter.guardrail_policies import (
        GuardrailAction,
        GuardrailPhase,
        GuardrailPolicy,
        GuardrailType,
    )

    rows = [
        {
            "payload": (
                '{"guardrail_id": "g-1", "name": "Deny Secrets", "enabled": true, '
                '"version": 3, "phase": "output", "check_type": "regex_deny", '
                '"parameters": {"patterns": ["x"]}, "action": "alert", '
                '"workspace_id": "ws-1", "org_id": null, "async_execution": false, '
                '"priority": 50, "created_at": 1.0, "updated_at": 2.0, '
                '"description": ""}'
            )
        }
    ]
    conn = _FakeConn(fetch_results=[rows])
    with _patch_pool(conn):
        guards = await enabled_store.load_all_guardrails()
    assert len(guards) == 1
    g = guards[0]
    assert isinstance(g, GuardrailPolicy)
    assert g.guardrail_id == "g-1"
    # enum coercion on the load side
    assert g.check_type is GuardrailType.REGEX_DENY
    assert g.action is GuardrailAction.ALERT
    assert g.phase is GuardrailPhase.OUTPUT
    assert g.version == 3
    assert g.priority == 50
    assert "SELECT payload FROM guardrail_policies" in conn.fetched[0][0]


@pytest.mark.asyncio
async def test_load_all_guardrails_accepts_dict_payload(enabled_store):
    """asyncpg may decode JSONB to a dict directly (not a str) -- handle both."""
    from litellm_llmrouter.guardrail_policies import GuardrailPolicy

    rows = [
        {
            "payload": {
                "guardrail_id": "g-2",
                "name": "G2",
                "check_type": "max_tokens",
                "parameters": {"max_tokens_limit": 100},
            }
        }
    ]
    conn = _FakeConn(fetch_results=[rows])
    with _patch_pool(conn):
        guards = await enabled_store.load_all_guardrails()
    assert len(guards) == 1
    assert isinstance(guards[0], GuardrailPolicy)
    assert guards[0].guardrail_id == "g-2"


# ---------------------------------------------------------------------------
# Prompt ops (RouteIQ-c2af): durable WRITE + round-trip READ (versions / A-B)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_prompt_writes_durably(enabled_store):
    from litellm_llmrouter.prompt_management import PromptDefinition, PromptVersion

    conn = _FakeConn()
    p = PromptDefinition(
        name="code-review",
        workspace_id="ws-1",
        active_version=2,
        ab_enabled=True,
        ab_versions={1: 0.9, 2: 0.1},
        versions={
            1: PromptVersion(version=1, template="v1"),
            2: PromptVersion(version=2, template="v2"),
        },
    )
    with _patch_pool(conn):
        await enabled_store.upsert_prompt("ws-1::code-review", p)
    sql, args = conn.executed[0]
    assert "INSERT INTO prompts" in sql
    assert "ON CONFLICT (storage_key) DO UPDATE" in sql
    assert args[0] == "ws-1::code-review"
    assert args[1] == "code-review"
    assert args[2] == "ws-1"
    import json as _json

    payload = _json.loads(args[3])
    assert payload["active_version"] == 2
    assert payload["ab_enabled"] is True
    # JSON serializes int dict keys to strings; Pydantic coerces them back on load.
    assert set(payload["versions"].keys()) == {"1", "2"}


@pytest.mark.asyncio
async def test_delete_prompt_issues_delete_sql(enabled_store):
    conn = _FakeConn()
    with _patch_pool(conn):
        await enabled_store.delete_prompt("ws-1::code-review")
    sql, args = conn.executed[0]
    assert "DELETE FROM prompts" in sql
    assert args == ("ws-1::code-review",)


@pytest.mark.asyncio
async def test_load_all_prompts_round_trips_versions_and_ab(enabled_store):
    """The (storage_key, PromptDefinition) pairs round-trip int version keys and
    A/B weights through JSON -- the rollback/A-B state that pod churn was losing."""
    from litellm_llmrouter.prompt_management import PromptDefinition, PromptVersion

    src = PromptDefinition(
        name="code-review",
        workspace_id="ws-1",
        active_version=2,
        ab_enabled=True,
        ab_versions={1: 0.7, 2: 0.3},
        versions={
            1: PromptVersion(version=1, template="v1", change_note="init"),
            2: PromptVersion(version=2, template="v2", change_note="tweak"),
        },
    )
    rows = [{"storage_key": "ws-1::code-review", "payload": src.model_dump_json()}]
    conn = _FakeConn(fetch_results=[rows])
    with _patch_pool(conn):
        pairs = await enabled_store.load_all_prompts()
    assert len(pairs) == 1
    storage_key, p = pairs[0]
    assert storage_key == "ws-1::code-review"
    assert isinstance(p, PromptDefinition)
    # int version keys survive the JSON round-trip (Pydantic Dict[int, ...]).
    assert set(p.versions.keys()) == {1, 2}
    assert all(isinstance(k, int) for k in p.versions)
    assert p.versions[2].template == "v2"
    assert p.active_version == 2
    assert p.ab_enabled is True
    assert p.ab_versions == {1: 0.7, 2: 0.3}
    assert "SELECT storage_key, payload FROM prompts" in conn.fetched[0][0]


@pytest.mark.asyncio
async def test_guardrail_prompt_disabled_store_is_noop(monkeypatch):
    """Store disabled (no DATABASE_URL): guardrail/prompt ops are fail-open no-ops
    and never acquire a pool."""
    from litellm_llmrouter.guardrail_policies import GuardrailPolicy, GuardrailType
    from litellm_llmrouter.prompt_management import PromptDefinition

    monkeypatch.delenv("DATABASE_URL", raising=False)
    reset_governance_store()
    store = get_governance_store()
    assert store.enabled is False

    conn = _FakeConn()
    with _patch_pool(conn):
        await store.upsert_guardrail(
            GuardrailPolicy(
                guardrail_id="g", name="g", check_type=GuardrailType.REGEX_DENY
            )
        )
        await store.delete_guardrail("g")
        assert await store.load_all_guardrails() == []
        await store.upsert_prompt("k", PromptDefinition(name="p"))
        await store.delete_prompt("k")
        assert await store.load_all_prompts() == []
    # No DB call at all when disabled.
    assert conn.executed == []
    assert conn.fetched == []


# ---------------------------------------------------------------------------
# Spend ops: atomic accumulator write + durable read
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_spend_uses_atomic_upsert(enabled_store):
    conn = _FakeConn()
    ps = datetime.fromtimestamp(2_592_000, tz=timezone.utc)
    with _patch_pool(conn):
        await enabled_store.record_spend(
            "ws-1", "workspace", "monthly", ps, cost=2.5, tokens=100, requests=1
        )
    sql, args = conn.executed[0]
    assert "INSERT INTO governance_spend" in sql
    # The accumulation must be atomic (no read-modify-write race).
    assert "ON CONFLICT (scope, period, period_start) DO UPDATE" in sql
    assert "governance_spend.spend_usd + EXCLUDED.spend_usd" in sql
    # Bind order: scope, scope_type, period_start, period, cost, requests, tokens.
    assert args[0] == "ws-1"
    assert args[1] == "workspace"
    assert args[2] == ps
    assert args[3] == "monthly"
    assert args[4] == 2.5
    assert args[5] == 1  # requests
    assert args[6] == 100  # tokens


@pytest.mark.asyncio
async def test_get_spend_reads_durable_rollup(enabled_store):
    ps = datetime.fromtimestamp(2_592_000, tz=timezone.utc)
    conn = _FakeConn(fetch_results=[[{"spend_usd": 42.5}]])
    with _patch_pool(conn):
        spend = await enabled_store.get_spend("ws-1", "monthly", ps)
    assert spend == 42.5
    sql, args = conn.fetched[0]
    assert "SELECT spend_usd FROM governance_spend" in sql
    assert args == ("ws-1", "monthly", ps)


@pytest.mark.asyncio
async def test_get_spend_returns_zero_when_no_row(enabled_store):
    ps = datetime.fromtimestamp(0, tz=timezone.utc)
    conn = _FakeConn(fetch_results=[[]])  # empty -> no rollup yet
    with _patch_pool(conn):
        assert await enabled_store.get_spend("ws-new", "monthly", ps) == 0.0


# ---------------------------------------------------------------------------
# Enabled-but-pool-None: still fail-open (asyncpg unavailable at runtime)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enabled_but_pool_none_is_failopen(enabled_store):
    """Store says enabled (DATABASE_URL set) but the pool is None (asyncpg
    missing / DB unreachable): writes must not raise, reads return empty/0.0."""
    with patch(
        "litellm_llmrouter.governance_store.get_db_pool",
        AsyncMock(return_value=None),
    ):
        # Writes: no-op, no raise.
        await enabled_store.upsert_org(OrgConfig(org_id="o", name="n"))
        await enabled_store.record_spend(
            "ws",
            "workspace",
            "monthly",
            datetime.fromtimestamp(0, tz=timezone.utc),
            cost=1.0,
        )
        # Reads: empty / 0.0.
        assert await enabled_store.load_all_orgs() == []
        assert await enabled_store.load_all_workspaces() == []
        assert (
            await enabled_store.get_spend(
                "ws", "monthly", datetime.fromtimestamp(0, tz=timezone.utc)
            )
            == 0.0
        )


@pytest.mark.asyncio
async def test_write_swallows_connection_error(enabled_store):
    """A DB error on write is logged + swallowed (fail-open) -- never breaks the
    CRUD response path."""

    class _BoomConn:
        async def execute(self, sql, *args):
            raise RuntimeError("connection reset")

    boom_pool = _FakePool(_BoomConn())  # type: ignore[arg-type]
    with patch(
        "litellm_llmrouter.governance_store.get_db_pool",
        AsyncMock(return_value=boom_pool),
    ):
        # Must not raise despite the underlying execute() blowing up.
        await enabled_store.upsert_org(OrgConfig(org_id="o", name="n"))


# ---------------------------------------------------------------------------
# Migrations (enabled): the table DDL is executed on an acquired connection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_migrations_executes_ddl_when_enabled(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", _DB_URL)
    reset_governance_store()
    conn = _FakeConn()
    with _patch_pool(conn):
        await run_governance_migrations()
    assert len(conn.executed) == 1
    sql = conn.executed[0][0]
    # All durable tables created in one DDL block (incl. guardrails + prompts).
    for table in (
        "governance_orgs",
        "governance_workspaces",
        "governance_keys",
        "usage_policies",
        "governance_spend",
        "guardrail_policies",
        "prompts",
    ):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in sql


# ---------------------------------------------------------------------------
# GovernanceEngine durable fallback: Redis miss -> Aurora rollup (enabled)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_engine_durable_spend_fallback_hits_store(monkeypatch):
    """When Redis returns no hot counter, the engine falls back to the durable
    Aurora rollup via store.get_spend (only when the store is enabled)."""
    monkeypatch.setenv("DATABASE_URL", _DB_URL)
    reset_governance_store()

    # Redis present but returns None for the spend key (cache miss / flush).
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)

    durable = AsyncMock(return_value=7.75)
    with (
        patch(
            "litellm_llmrouter.redis_pool.get_async_redis_client",
            AsyncMock(return_value=redis),
        ),
        patch.object(GovernanceStore, "get_spend", durable),
    ):
        engine = GovernanceEngine()
        ctx = GovernanceContext(workspace_id="ws-7")
        spend = await engine._get_current_spend(ctx)

    assert spend == 7.75
    durable.assert_awaited_once()


@pytest.mark.asyncio
async def test_engine_durable_fallback_skipped_when_store_disabled(monkeypatch):
    """No DATABASE_URL -> the durable rollup is skipped; engine returns 0.0 on a
    Redis miss (never reaches the store)."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    reset_governance_store()

    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)

    with patch(
        "litellm_llmrouter.redis_pool.get_async_redis_client",
        AsyncMock(return_value=redis),
    ):
        engine = GovernanceEngine()
        ctx = GovernanceContext(workspace_id="ws-x")
        spend = await engine._get_current_spend(ctx)

    assert spend == 0.0
