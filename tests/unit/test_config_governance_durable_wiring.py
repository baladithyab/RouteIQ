"""Unit tests for the governance CRUD-route -> Aurora durable-write wiring (P4).

The P4 edit threads ``get_governance_store()`` into every governance /
usage-policy CRUD handler in ``routes/config.py`` so that, when the store is
enabled (DATABASE_URL set), each in-memory mutation is mirrored to a durable
Aurora row.  When the store is disabled the handler degrades to the existing
in-memory + JSON-file path with NO database call.

These tests call the route handler *functions* directly (passing a stub
``rbac_info`` so the auth Depends is bypassed) with a mocked asyncpg pool, and
assert:

  * the store WRITE fires with the correct SQL on create/update/delete (durable),
  * the in-memory engine is updated regardless (fallback survives),
  * with the store disabled, NO pool is acquired (no DB call) yet the in-memory
    CRUD still succeeds.

No real database, no AWS creds.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest

from litellm_llmrouter.governance import (
    KeyGovernance,
    WorkspaceConfig,
    get_governance_engine,
)
from litellm_llmrouter.governance_store import (
    get_governance_store,
    reset_governance_store,
)
from litellm_llmrouter.routes import config as config_routes
from litellm_llmrouter.usage_policies import UsagePolicy, get_usage_policy_engine

_DB_URL = "postgresql://u@h:5432/db"
_RBAC = {"actor": "admin", "permissions": ["system.config.reload"]}


class _FakeConn:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple]] = []

    async def execute(self, sql: str, *args) -> None:
        self.executed.append((sql, args))

    async def fetch(self, sql: str, *args):
        return []


class _FakePool:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn
        self.acquire_count = 0

    def acquire(self):
        self.acquire_count += 1
        conn = self._conn

        @asynccontextmanager
        async def _cm():
            yield conn

        return _cm()


def _patch_pool(conn: _FakeConn, pool_holder: list | None = None):
    pool = _FakePool(conn)
    if pool_holder is not None:
        pool_holder.append(pool)
    return patch(
        "litellm_llmrouter.governance_store.get_db_pool",
        AsyncMock(return_value=pool),
    )


# ---------------------------------------------------------------------------
# Enabled path: CRUD route -> durable Aurora write
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_workspace_writes_durable_row(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", _DB_URL)
    reset_governance_store()
    assert get_governance_store().enabled is True

    conn = _FakeConn()
    ws = WorkspaceConfig(workspace_id="ws-1", name="WS One", org_id="org-1")
    with _patch_pool(conn):
        resp = await config_routes.create_workspace(ws, rbac_info=_RBAC)

    # In-memory engine updated.
    assert get_governance_engine().get_workspace("ws-1") is not None
    assert resp["created"] is True
    # Durable write issued.
    assert any("INSERT INTO governance_workspaces" in s for s, _ in conn.executed)


@pytest.mark.asyncio
async def test_delete_workspace_writes_durable_delete(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", _DB_URL)
    reset_governance_store()

    get_governance_engine().register_workspace(
        WorkspaceConfig(workspace_id="ws-del", name="To Delete")
    )
    conn = _FakeConn()
    with _patch_pool(conn):
        await config_routes.delete_workspace("ws-del", rbac_info=_RBAC)

    assert get_governance_engine().get_workspace("ws-del") is None
    assert any(
        "DELETE FROM governance_workspaces" in s and a == ("ws-del",)
        for s, a in conn.executed
    )


@pytest.mark.asyncio
async def test_update_key_governance_writes_durable_row(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", _DB_URL)
    reset_governance_store()

    # The handler validates the workspace reference, so register it first.
    get_governance_engine().register_workspace(
        WorkspaceConfig(workspace_id="ws-1", name="WS One")
    )
    conn = _FakeConn()
    kg = KeyGovernance(key_id="key-1", workspace_id="ws-1")
    with _patch_pool(conn):
        await config_routes.update_key_governance("key-1", kg, rbac_info=_RBAC)

    assert get_governance_engine().get_key_governance("key-1") is not None
    assert any("INSERT INTO governance_keys" in s for s, _ in conn.executed)


@pytest.mark.asyncio
async def test_create_usage_policy_writes_durable_row(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", _DB_URL)
    reset_governance_store()

    conn = _FakeConn()
    policy = UsagePolicy(policy_id="p-1", name="P One")
    with _patch_pool(conn):
        await config_routes.create_usage_policy(policy, rbac_info=_RBAC)

    assert get_usage_policy_engine().get_policy("p-1") is not None
    assert any("INSERT INTO usage_policies" in s for s, _ in conn.executed)


# ---------------------------------------------------------------------------
# Disabled path: in-memory CRUD succeeds, NO database call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_workspace_disabled_no_db_call(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    reset_governance_store()
    assert get_governance_store().enabled is False

    holder: list = []
    ws = WorkspaceConfig(workspace_id="ws-mem", name="In Memory")
    with _patch_pool(_FakeConn(), pool_holder=holder):
        resp = await config_routes.create_workspace(ws, rbac_info=_RBAC)

    # In-memory path works.
    assert get_governance_engine().get_workspace("ws-mem") is not None
    assert resp["created"] is True
    # The disabled store short-circuits BEFORE acquiring the pool: no DB call.
    assert holder and holder[0].acquire_count == 0


@pytest.mark.asyncio
async def test_delete_usage_policy_disabled_no_db_call(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    reset_governance_store()

    get_usage_policy_engine().add_policy(UsagePolicy(policy_id="p-mem", name="Mem"))
    holder: list = []
    with _patch_pool(_FakeConn(), pool_holder=holder):
        await config_routes.delete_usage_policy("p-mem", rbac_info=_RBAC)

    assert get_usage_policy_engine().get_policy("p-mem") is None
    assert holder and holder[0].acquire_count == 0


# ---------------------------------------------------------------------------
# Guardrail policy CRUD -> Aurora durable write (RouteIQ-4f30)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_guardrail_writes_durable_row(monkeypatch):
    from litellm_llmrouter.guardrail_policies import (
        GuardrailPolicy,
        GuardrailType,
        get_guardrail_policy_engine,
    )

    monkeypatch.setenv("DATABASE_URL", _DB_URL)
    reset_governance_store()

    conn = _FakeConn()
    g = GuardrailPolicy(
        guardrail_id="g-1", name="G One", check_type=GuardrailType.REGEX_DENY
    )
    with _patch_pool(conn):
        resp = await config_routes.create_guardrail_policy(g, rbac_info=_RBAC)

    assert get_guardrail_policy_engine().get_policy("g-1") is not None
    assert resp["created"] is True
    assert any("INSERT INTO guardrail_policies" in s for s, _ in conn.executed)


@pytest.mark.asyncio
async def test_delete_guardrail_writes_durable_delete(monkeypatch):
    from litellm_llmrouter.guardrail_policies import (
        GuardrailPolicy,
        GuardrailType,
        get_guardrail_policy_engine,
    )

    monkeypatch.setenv("DATABASE_URL", _DB_URL)
    reset_governance_store()

    get_guardrail_policy_engine().add_policy(
        GuardrailPolicy(
            guardrail_id="g-del", name="To Delete", check_type=GuardrailType.MAX_TOKENS
        )
    )
    conn = _FakeConn()
    with _patch_pool(conn):
        await config_routes.delete_guardrail_policy("g-del", rbac_info=_RBAC)

    assert get_guardrail_policy_engine().get_policy("g-del") is None
    assert any(
        "DELETE FROM guardrail_policies" in s and a == ("g-del",)
        for s, a in conn.executed
    )


@pytest.mark.asyncio
async def test_create_guardrail_disabled_no_db_call(monkeypatch):
    from litellm_llmrouter.guardrail_policies import (
        GuardrailPolicy,
        GuardrailType,
        get_guardrail_policy_engine,
    )

    monkeypatch.delenv("DATABASE_URL", raising=False)
    reset_governance_store()
    assert get_governance_store().enabled is False

    holder: list = []
    g = GuardrailPolicy(
        guardrail_id="g-mem", name="In Memory", check_type=GuardrailType.REGEX_DENY
    )
    with _patch_pool(_FakeConn(), pool_holder=holder):
        resp = await config_routes.create_guardrail_policy(g, rbac_info=_RBAC)

    assert get_guardrail_policy_engine().get_policy("g-mem") is not None
    assert resp["created"] is True
    assert holder and holder[0].acquire_count == 0


# ---------------------------------------------------------------------------
# Prompt CRUD -> Aurora durable write (RouteIQ-c2af)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_prompt_writes_durable_row(monkeypatch):
    from litellm_llmrouter.prompt_management import (
        CreatePromptRequest,
        get_prompt_manager,
    )

    monkeypatch.setenv("DATABASE_URL", _DB_URL)
    monkeypatch.setattr(config_routes, "is_prompt_management_enabled", lambda: True)
    reset_governance_store()

    conn = _FakeConn()
    body = CreatePromptRequest(
        name="code-review", template="Review {{code}}", workspace_id="ws-1"
    )
    with _patch_pool(conn):
        await config_routes.create_prompt(body, rbac_info=_RBAC)

    assert (
        get_prompt_manager().get_prompt("code-review", workspace_id="ws-1") is not None
    )
    durable = [(s, a) for s, a in conn.executed if "INSERT INTO prompts" in s]
    assert durable
    assert durable[0][1][0] == "ws-1::code-review"


@pytest.mark.asyncio
async def test_delete_prompt_writes_durable_delete(monkeypatch):
    from litellm_llmrouter.prompt_management import get_prompt_manager

    monkeypatch.setenv("DATABASE_URL", _DB_URL)
    monkeypatch.setattr(config_routes, "is_prompt_management_enabled", lambda: True)
    reset_governance_store()

    get_prompt_manager().create_prompt(
        name="to-delete", template="x", workspace_id="ws-1"
    )
    conn = _FakeConn()
    with _patch_pool(conn):
        await config_routes.delete_prompt(
            "to-delete", workspace_id="ws-1", rbac_info=_RBAC
        )

    assert get_prompt_manager().get_prompt("to-delete", workspace_id="ws-1") is None
    assert any(
        "DELETE FROM prompts" in s and a == ("ws-1::to-delete",)
        for s, a in conn.executed
    )


@pytest.mark.asyncio
async def test_rollback_prompt_writes_durable_row(monkeypatch):
    from litellm_llmrouter.prompt_management import (
        RollbackRequest,
        get_prompt_manager,
    )

    monkeypatch.setenv("DATABASE_URL", _DB_URL)
    monkeypatch.setattr(config_routes, "is_prompt_management_enabled", lambda: True)
    reset_governance_store()

    mgr = get_prompt_manager()
    mgr.create_prompt(name="p-roll", template="v1")
    mgr.update_prompt(name="p-roll", template="v2")  # active_version -> 2
    conn = _FakeConn()
    with _patch_pool(conn):
        await config_routes.rollback_prompt(
            "p-roll", RollbackRequest(version=1), rbac_info=_RBAC
        )

    # The rollback set active_version back to 1; the durable payload reflects it.
    inserts = [a for s, a in conn.executed if "INSERT INTO prompts" in s]
    assert inserts
    import json as _json

    payload = _json.loads(inserts[0][3])
    assert payload["active_version"] == 1


@pytest.mark.asyncio
async def test_create_prompt_disabled_no_db_call(monkeypatch):
    from litellm_llmrouter.prompt_management import (
        CreatePromptRequest,
        get_prompt_manager,
    )

    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(config_routes, "is_prompt_management_enabled", lambda: True)
    reset_governance_store()
    assert get_governance_store().enabled is False

    holder: list = []
    body = CreatePromptRequest(name="p-mem", template="x")
    with _patch_pool(_FakeConn(), pool_holder=holder):
        await config_routes.create_prompt(body, rbac_info=_RBAC)

    assert get_prompt_manager().get_prompt("p-mem") is not None
    assert holder and holder[0].acquire_count == 0
