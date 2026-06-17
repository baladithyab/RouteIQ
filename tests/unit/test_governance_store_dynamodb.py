"""Tests for the DynamoDB governance backend (RouteIQ-a865).

Cred-free: a fake DynamoDB Table records put/delete/query so the CRUD + load
round-trip is asserted without boto3 / AWS. Covers the default-off (file)
backend selection and the enabled DynamoDB path.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from litellm_llmrouter.governance import KeyGovernance, OrgConfig, WorkspaceConfig
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
    """Records items keyed by (pk, sk); serves Key('pk').eq() queries."""

    def __init__(self) -> None:
        self.items: dict[tuple[str, str], dict] = {}

    def put_item(self, Item: dict) -> None:  # noqa: N803 - boto3 kwarg name
        self.items[(Item["pk"], Item["sk"])] = Item

    def delete_item(self, Key: dict) -> None:  # noqa: N803
        self.items.pop((Key["pk"], Key["sk"]), None)

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
