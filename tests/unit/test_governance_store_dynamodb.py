"""Tests for the DynamoDB governance backend (RouteIQ-a865).

Cred-free: a fake DynamoDB Table records put/delete/query so the CRUD + load
round-trip is asserted without boto3 / AWS. Covers the default-off (file)
backend selection and the enabled DynamoDB path.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

import inspect
from datetime import datetime, timezone

from litellm_llmrouter.governance import KeyGovernance, OrgConfig, WorkspaceConfig
from litellm_llmrouter.governance_store import GovernanceStore
from litellm_llmrouter.governance_store_dynamodb import (
    PK_KEY,
    PK_ORG,
    PK_WORKSPACE,
    DynamoDBGovernanceStore,
    dynamodb_backend_enabled,
    governance_backend,
    reset_dynamodb_governance_store,
)


class _FakeTable:
    """Records items keyed by (pk, sk); serves Key('pk').eq() queries.

    Also models the subset of ``update_item`` (ADD counters / SET attrs) and
    ``get_item`` the spend ops use, cred-free (no boto3 / AWS).
    """

    def __init__(self) -> None:
        self.items: dict[tuple[str, str], dict] = {}

    def put_item(self, Item: dict) -> None:  # noqa: N803 - boto3 kwarg name
        self.items[(Item["pk"], Item["sk"])] = Item

    def delete_item(self, Key: dict) -> None:  # noqa: N803
        self.items.pop((Key["pk"], Key["sk"]), None)

    def get_item(self, Key: dict) -> dict:  # noqa: N803
        item = self.items.get((Key["pk"], Key["sk"]))
        return {"Item": item} if item is not None else {}

    def update_item(  # noqa: N803
        self,
        Key: dict,
        UpdateExpression: str,
        ExpressionAttributeNames: dict,
        ExpressionAttributeValues: dict,
    ) -> None:
        key = (Key["pk"], Key["sk"])
        item = self.items.setdefault(key, {"pk": Key["pk"], "sk": Key["sk"]})
        verb, _, body = UpdateExpression.partition(" ")
        for clause in body.split(","):
            clause = clause.strip()
            if not clause:
                continue
            if verb == "ADD":
                name_ph, val_ph = clause.split()
                attr = ExpressionAttributeNames[name_ph]
                item[attr] = item.get(attr, 0) + ExpressionAttributeValues[val_ph]
            elif verb == "SET":
                name_ph, _, val_ph = clause.partition(" = ")
                attr = ExpressionAttributeNames[name_ph]
                item[attr] = ExpressionAttributeValues[val_ph]

    def query(self, KeyConditionExpression):  # noqa: N803
        # The store's _pk_condition is patched to return the raw pk string in
        # tests (boto3 is not installed), so KeyConditionExpression IS the pk.
        pk = KeyConditionExpression
        return {"Items": [v for (p, _), v in self.items.items() if p == pk]}


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_dynamodb_governance_store()
    yield
    reset_dynamodb_governance_store()


def test_default_backend_is_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ROUTEIQ_GOVERNANCE_BACKEND", raising=False)
    assert governance_backend() == "file"
    assert dynamodb_backend_enabled() is False
    store = DynamoDBGovernanceStore()
    assert store.enabled is False


async def test_disabled_store_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ROUTEIQ_GOVERNANCE_BACKEND", raising=False)
    store = DynamoDBGovernanceStore()
    # No-ops; load returns empty without touching boto3.
    await store.upsert_org(OrgConfig(org_id="o1", name="Org"))
    assert await store.load_all_orgs() == []


async def test_enabled_crud_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROUTEIQ_GOVERNANCE_BACKEND", "dynamodb")
    table = _FakeTable()
    store = DynamoDBGovernanceStore()
    assert store.enabled is True
    # _pk_condition is patched to return the raw pk (boto3 not installed).
    with (
        patch.object(store, "_pk_condition", side_effect=lambda pk: pk),
        patch.object(store, "_get_table", return_value=table),
    ):
        await store.upsert_org(OrgConfig(org_id="o1", name="Acme"))
        await store.upsert_workspace(
            WorkspaceConfig(workspace_id="w1", name="WS", org_id="o1")
        )
        await store.upsert_key(KeyGovernance(key_id="k1", workspace_id="w1"))

        orgs = await store.load_all_orgs()
        workspaces = await store.load_all_workspaces()
        keys = await store.load_all_keys()

    assert [o.org_id for o in orgs] == ["o1"]
    assert orgs[0].name == "Acme"
    assert [w.workspace_id for w in workspaces] == ["w1"]
    assert [k.key_id for k in keys] == ["k1"]
    # the item carries the JSON payload + pk/sk shape
    assert (PK_ORG, "o1") in table.items
    assert (PK_WORKSPACE, "w1") in table.items
    assert (PK_KEY, "k1") in table.items
    assert json.loads(table.items[(PK_ORG, "o1")]["payload"])["name"] == "Acme"


async def test_enabled_delete(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROUTEIQ_GOVERNANCE_BACKEND", "dynamodb")
    table = _FakeTable()
    store = DynamoDBGovernanceStore()
    with (
        patch.object(store, "_pk_condition", side_effect=lambda pk: pk),
        patch.object(store, "_get_table", return_value=table),
    ):
        await store.upsert_org(OrgConfig(org_id="o1", name="Acme"))
        await store.delete_org("o1")
        assert await store.load_all_orgs() == []


async def test_failopen_on_table_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROUTEIQ_GOVERNANCE_BACKEND", "dynamodb")
    store = DynamoDBGovernanceStore()

    class _BoomTable:
        def put_item(self, **_):
            raise RuntimeError("throttled")

    with patch.object(store, "_get_table", return_value=_BoomTable()):
        # write swallows the error (fail-open, never raises)
        await store.upsert_org(OrgConfig(org_id="o1", name="X"))


# ---------------------------------------------------------------------------
# Spend ops (RouteIQ-40d5): mirror the Aurora interface, fail-open, cred-free.
# ---------------------------------------------------------------------------

_SPEND_METHODS = (
    "record_spend",
    "get_spend",
    "record_model_spend",
    "aggregate_spend_report",
)


def test_spend_methods_exist_and_match_aurora_interface() -> None:
    """The DynamoDB store mirrors the Aurora spend interface 1:1.

    With ``ROUTEIQ_GOVERNANCE_BACKEND=dynamodb`` the spend callsites see
    ``enabled is True`` and INVOKE these methods; if they were absent the store
    would raise ``AttributeError`` and break (not degrade) the spend path. Assert
    each method exists and carries the exact same signature as the Aurora store.
    """
    for name in _SPEND_METHODS:
        ddb = getattr(DynamoDBGovernanceStore, name, None)
        aurora = getattr(GovernanceStore, name, None)
        assert callable(ddb), f"DynamoDBGovernanceStore.{name} missing"
        assert callable(aurora), f"GovernanceStore.{name} missing (test baseline)"
        assert inspect.iscoroutinefunction(ddb), f"{name} must be async"
        # Compare the CALLER-VISIBLE shape: parameter names, kinds, and defaults.
        # (Raw annotation strings differ only because the DDB store imports
        # ``datetime`` under TYPE_CHECKING, so its stringized annotation is extra-
        # quoted -- a cosmetic artifact, not an interface difference.)

        def _shape(fn):
            return [
                (p.name, p.kind, p.default)
                for p in inspect.signature(fn).parameters.values()
            ]

        assert _shape(ddb) == _shape(aurora), (
            f"{name} parameter shape drifted from the Aurora store"
        )


async def test_disabled_spend_is_safe_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the backend is not selected, spend ops are fail-open no-ops."""
    monkeypatch.delenv("ROUTEIQ_GOVERNANCE_BACKEND", raising=False)
    store = DynamoDBGovernanceStore()
    assert store.enabled is False
    ps = datetime(2026, 6, 1, tzinfo=timezone.utc)
    # No boto3 touched; no raise.
    await store.record_spend("ws1", "workspace", "monthly", ps, cost=1.0, requests=1)
    await store.record_model_spend(
        "gpt", "ws1", "workspace", "monthly", ps, cost=1.0, requests=1
    )
    assert await store.get_spend("ws1", "monthly", ps) == 0.0
    report = await store.aggregate_spend_report()
    assert report["total_cost_usd"] == 0.0
    assert report["by_scope"] == []
    assert report["by_model"] == []
    assert report["daily"] == []


async def test_enabled_spend_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Accumulate scope + model spend, then read it back via get/aggregate."""
    monkeypatch.setenv("ROUTEIQ_GOVERNANCE_BACKEND", "dynamodb")
    table = _FakeTable()
    store = DynamoDBGovernanceStore()
    assert store.enabled is True
    ps = datetime(2026, 6, 1, tzinfo=timezone.utc)
    with (
        patch.object(store, "_pk_condition", side_effect=lambda pk: pk),
        patch.object(store, "_get_table", return_value=table),
    ):
        # Two writes to the same scope+period accumulate atomically (ADD).
        await store.record_spend(
            "ws1", "workspace", "monthly", ps, cost=1.5, tokens=10, requests=1
        )
        await store.record_spend(
            "ws1", "workspace", "monthly", ps, cost=2.5, tokens=20, requests=1
        )
        await store.record_model_spend(
            "gpt-4o",
            "ws1",
            "workspace",
            "monthly",
            ps,
            cost=3.0,
            input_tokens=5,
            output_tokens=7,
            requests=2,
        )

        assert await store.get_spend("ws1", "monthly", ps) == 4.0
        report = await store.aggregate_spend_report()

    assert report["total_cost_usd"] == 4.0
    assert report["total_requests"] == 2
    assert report["total_input_tokens"] == 5
    assert report["total_output_tokens"] == 7
    assert report["by_scope"] == [
        {
            "scope": "ws1",
            "scope_type": "workspace",
            "spend_usd": 4.0,
            "request_count": 2,
            "total_tokens": 30,
        }
    ]
    assert report["by_model"] == [
        {
            "model": "gpt-4o",
            "spend_usd": 3.0,
            "request_count": 2,
            "input_tokens": 5,
            "output_tokens": 7,
        }
    ]
    assert report["daily"] == [
        {"date": "2026-06-01", "spend_usd": 4.0, "request_count": 2}
    ]


async def test_aggregate_since_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ``since`` lower bound excludes older period buckets."""
    monkeypatch.setenv("ROUTEIQ_GOVERNANCE_BACKEND", "dynamodb")
    table = _FakeTable()
    store = DynamoDBGovernanceStore()
    old = datetime(2026, 1, 1, tzinfo=timezone.utc)
    new = datetime(2026, 6, 1, tzinfo=timezone.utc)
    with (
        patch.object(store, "_pk_condition", side_effect=lambda pk: pk),
        patch.object(store, "_get_table", return_value=table),
    ):
        await store.record_spend("ws1", "workspace", "monthly", old, cost=10.0)
        await store.record_spend("ws1", "workspace", "monthly", new, cost=5.0)
        report = await store.aggregate_spend_report(since=new)
    assert report["total_cost_usd"] == 5.0


async def test_spend_fail_open_on_client_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every spend method swallows a client error (never raises)."""
    monkeypatch.setenv("ROUTEIQ_GOVERNANCE_BACKEND", "dynamodb")
    store = DynamoDBGovernanceStore()
    ps = datetime(2026, 6, 1, tzinfo=timezone.utc)

    class _BoomTable:
        def update_item(self, **_):
            raise RuntimeError("throttled")

        def get_item(self, **_):
            raise RuntimeError("throttled")

        def query(self, **_):
            raise RuntimeError("throttled")

    with (
        patch.object(store, "_pk_condition", side_effect=lambda pk: pk),
        patch.object(store, "_get_table", return_value=_BoomTable()),
    ):
        # writes/reads all swallow -> safe empties, never raise
        await store.record_spend("ws1", "workspace", "monthly", ps, cost=1.0)
        await store.record_model_spend(
            "gpt", "ws1", "workspace", "monthly", ps, cost=1.0
        )
        assert await store.get_spend("ws1", "monthly", ps) == 0.0
        report = await store.aggregate_spend_report()
    assert report["total_cost_usd"] == 0.0
    assert report["by_scope"] == []
    assert report["by_model"] == []
