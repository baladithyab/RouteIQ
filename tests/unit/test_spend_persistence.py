"""Tests for persisted spend / chargeback aggregation (RouteIQ-67fe).

Before this change ``get_spend_report`` returned ``total_cost_usd`` hard-0.0 --
the cost tracker was OTel-span-only with no DB persistence.  Now:

* ``GovernanceStore.record_model_spend`` accumulates a per-model rollup
  (``governance_spend_model``) atomically (``ON CONFLICT DO UPDATE``),
* ``GovernanceStore.aggregate_spend_report`` aggregates the durable
  ``governance_spend`` / ``governance_spend_model`` ledgers into real totals +
  per-scope + per-model + daily breakdowns, and
* the ``/llmrouter/spend/report`` endpoint overlays those real numbers when the
  store is enabled, while staying a zero/no-op report when no DB is configured.

All tests use a MOCKED asyncpg pool (no real DB, no AWS creds) and assert the
aggregation reflects the recorded costs.  The fake connection records execute()
and serves queued fetch() results so the report-shaping is provable without a
live Postgres.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from litellm_llmrouter.governance_store import (
    get_governance_store,
    reset_governance_store,
)

_DB_URL = "postgresql://u@h:5432/db"
_PERIOD_START = datetime(2026, 6, 1, tzinfo=timezone.utc)


class _FakeConn:
    """Records execute() and answers fetch() from a queued result list."""

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
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    def acquire(self):
        conn = self._conn

        @asynccontextmanager
        async def _cm():
            yield conn

        return _cm()


def _patch_pool(conn: _FakeConn):
    return patch(
        "litellm_llmrouter.governance_store.get_db_pool",
        AsyncMock(return_value=_FakePool(conn)),
    )


@pytest.fixture
def enabled_store(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", _DB_URL)
    reset_governance_store()
    store = get_governance_store()
    assert store.enabled is True
    return store


# ---------------------------------------------------------------------------
# record_model_spend: atomic per-model accumulator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_model_spend_writes_atomically(enabled_store):
    conn = _FakeConn()
    with _patch_pool(conn):
        await enabled_store.record_model_spend(
            "gpt-4o",
            "key-1",
            "key",
            "monthly",
            _PERIOD_START,
            cost=0.42,
            input_tokens=100,
            output_tokens=50,
            requests=1,
        )
    assert len(conn.executed) == 1
    sql, args = conn.executed[0]
    assert "INSERT INTO governance_spend_model" in sql
    assert "ON CONFLICT (model, scope, period, period_start) DO UPDATE" in sql
    # Accumulator (not overwrite): adds EXCLUDED to the existing row.
    assert "governance_spend_model.spend_usd + EXCLUDED.spend_usd" in sql
    assert args[0] == "gpt-4o"
    assert args[1] == "key-1"
    assert args[5] == 0.42  # cost
    assert args[7] == 100  # input_tokens
    assert args[8] == 50  # output_tokens


@pytest.mark.asyncio
async def test_record_model_spend_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    reset_governance_store()
    store = get_governance_store()
    assert store.enabled is False
    conn = _FakeConn()
    with _patch_pool(conn):
        await store.record_model_spend(
            "gpt-4o", "k", "key", "monthly", _PERIOD_START, cost=1.0
        )
    assert conn.executed == []  # disabled -> no write


# ---------------------------------------------------------------------------
# aggregate_spend_report: recorded costs roll up into the report
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aggregate_spend_report_sums_recorded_costs(enabled_store):
    # Queue fetch results in the order aggregate_spend_report issues them:
    # 1) per-scope, 2) per-model, 3) daily.
    scope_rows = [
        {
            "scope": "team-a",
            "scope_type": "workspace",
            "spend_usd": 3.0,
            "request_count": 10,
            "total_tokens": 1500,
        },
        {
            "scope": "key-1",
            "scope_type": "key",
            "spend_usd": 1.5,
            "request_count": 5,
            "total_tokens": 600,
        },
    ]
    model_rows = [
        {
            "model": "gpt-4o",
            "spend_usd": 3.5,
            "request_count": 12,
            "input_tokens": 1200,
            "output_tokens": 400,
        },
        {
            "model": "claude-sonnet-4",
            "spend_usd": 1.0,
            "request_count": 3,
            "input_tokens": 300,
            "output_tokens": 200,
        },
    ]
    daily_rows = [
        {"day": _PERIOD_START, "spend_usd": 4.5, "request_count": 15},
    ]
    conn = _FakeConn([scope_rows, model_rows, daily_rows])

    with _patch_pool(conn):
        report = await enabled_store.aggregate_spend_report()

    # Real totals -- NOT hard-0.0.
    assert report["total_cost_usd"] == pytest.approx(4.5)  # 3.0 + 1.5
    assert report["total_requests"] == 15  # 10 + 5
    assert report["total_input_tokens"] == 1500  # 1200 + 300
    assert report["total_output_tokens"] == 600  # 400 + 200

    # Per-scope (key/team/workspace) breakdown.
    assert report["by_scope"][0]["scope"] == "team-a"
    assert report["by_scope"][0]["spend_usd"] == pytest.approx(3.0)

    # Per-model breakdown.
    assert report["by_model"][0]["model"] == "gpt-4o"
    assert report["by_model"][0]["spend_usd"] == pytest.approx(3.5)

    # Daily rollup.
    assert report["daily"][0]["date"] == "2026-06-01"
    assert report["daily"][0]["spend_usd"] == pytest.approx(4.5)


@pytest.mark.asyncio
async def test_aggregate_spend_report_zero_when_disabled(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    reset_governance_store()
    store = get_governance_store()
    report = await store.aggregate_spend_report()
    assert report["total_cost_usd"] == 0.0
    assert report["by_scope"] == []
    assert report["by_model"] == []
    assert report["daily"] == []


# ---------------------------------------------------------------------------
# get_spend_report endpoint wiring: real totals when the store is enabled
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_spend_report_reflects_durable_aggregate(monkeypatch):
    """The endpoint overlays the real durable aggregate over the model catalog."""
    from litellm_llmrouter.routes import config as config_routes

    # A fake store that is enabled and returns a real aggregate.
    class _FakeStore:
        enabled = True

        async def aggregate_spend_report(self):
            return {
                "total_cost_usd": 7.25,
                "total_input_tokens": 2000,
                "total_output_tokens": 800,
                "total_requests": 30,
                "by_scope": [
                    {
                        "scope": "team-a",
                        "scope_type": "workspace",
                        "spend_usd": 7.25,
                        "request_count": 30,
                        "total_tokens": 2800,
                    }
                ],
                "by_model": [
                    {
                        "model": "gpt-4o",
                        "spend_usd": 7.25,
                        "request_count": 30,
                        "input_tokens": 2000,
                        "output_tokens": 800,
                    }
                ],
                "daily": [
                    {"date": "2026-06-01", "spend_usd": 7.25, "request_count": 30}
                ],
            }

    monkeypatch.setattr(config_routes, "get_governance_store", lambda: _FakeStore())
    # Force the llm_router branch to the no_router early-return so the test does
    # not depend on a live LiteLLM router -- the durable overlay already ran.
    import sys
    import types as _types

    stub = _types.ModuleType("litellm.proxy.proxy_server")
    stub.llm_router = None  # type: ignore[attr-defined]
    with patch.dict(sys.modules, {"litellm.proxy.proxy_server": stub}):
        report = await config_routes.get_spend_report(rbac_info={"is_admin": True})

    assert report["persisted"] is True
    assert report["total_cost_usd"] == pytest.approx(7.25)
    assert report["total_requests"] == 30
    assert report["by_model"][0]["model"] == "gpt-4o"
    assert report["by_scope"][0]["scope"] == "team-a"
    assert report["daily"][0]["date"] == "2026-06-01"


@pytest.mark.asyncio
async def test_get_spend_report_degrades_when_no_db(monkeypatch):
    """No DB -> persisted False, total_cost_usd stays 0.0 (legacy contract)."""
    from litellm_llmrouter.routes import config as config_routes

    class _DisabledStore:
        enabled = False

        async def aggregate_spend_report(self):  # pragma: no cover - not called
            raise AssertionError("must not aggregate when disabled")

    monkeypatch.setattr(config_routes, "get_governance_store", lambda: _DisabledStore())
    import sys
    import types as _types

    stub = _types.ModuleType("litellm.proxy.proxy_server")
    stub.llm_router = None  # type: ignore[attr-defined]
    with patch.dict(sys.modules, {"litellm.proxy.proxy_server": stub}):
        report = await config_routes.get_spend_report(rbac_info={"is_admin": True})

    assert report["persisted"] is False
    assert report["total_cost_usd"] == 0.0
