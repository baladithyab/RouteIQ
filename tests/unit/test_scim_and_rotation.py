"""Tests for SCIM v2 provisioning + scheduled key rotation (RouteIQ-b8a2).

Cred-free: SCIM endpoints are driven via a FastAPI TestClient over the in-memory
governance engine; key rotation is pure-Python over the engine. Covers the
default-off gates, bearer-token fail-closed auth, User/Group provisioning +
de-provisioning, and age-based rotation.
"""

from __future__ import annotations

import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm_llmrouter.governance import (
    KeyGovernance,
    reset_governance_engine,
)
from litellm_llmrouter.scim import (
    create_scim_router,
    key_rotation_enabled,
    rotate_stale_keys,
    scim_enabled,
)


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_governance_engine()
    yield
    reset_governance_engine()


@pytest.fixture
def _client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("ROUTEIQ_SCIM_ENABLED", "true")
    monkeypatch.setenv("ROUTEIQ_SCIM_BEARER_TOKEN", "tok-123")
    app = FastAPI()
    app.include_router(create_scim_router())
    return TestClient(app)


_AUTH = {"Authorization": "Bearer tok-123"}


# -- gates -----------------------------------------------------------------


def test_scim_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ROUTEIQ_SCIM_ENABLED", raising=False)
    assert scim_enabled() is False


def test_rotation_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ROUTEIQ_KEY_ROTATION_ENABLED", raising=False)
    assert key_rotation_enabled() is False


# -- SCIM auth -------------------------------------------------------------


def test_scim_requires_bearer(_client: TestClient) -> None:
    assert _client.get("/scim/v2/Users").status_code == 401
    assert (
        _client.get(
            "/scim/v2/Users", headers={"Authorization": "Bearer wrong"}
        ).status_code
        == 401
    )


def test_scim_failclosed_when_no_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROUTEIQ_SCIM_ENABLED", "true")
    monkeypatch.delenv("ROUTEIQ_SCIM_BEARER_TOKEN", raising=False)
    app = FastAPI()
    app.include_router(create_scim_router())
    client = TestClient(app)
    assert client.get("/scim/v2/Users", headers=_AUTH).status_code == 401


# -- SCIM Users ------------------------------------------------------------


def test_provision_and_get_user(_client: TestClient) -> None:
    resp = _client.post(
        "/scim/v2/Users",
        json={"userName": "alice@corp.com", "displayName": "Alice", "active": True},
        headers=_AUTH,
    )
    assert resp.status_code == 201
    user_id = resp.json()["id"]
    assert resp.json()["userName"] == "alice@corp.com"

    got = _client.get(f"/scim/v2/Users/{user_id}", headers=_AUTH)
    assert got.status_code == 200
    assert got.json()["userName"] == "alice@corp.com"

    listed = _client.get("/scim/v2/Users", headers=_AUTH)
    assert listed.json()["totalResults"] == 1


def test_deprovision_user(_client: TestClient) -> None:
    user_id = _client.post(
        "/scim/v2/Users", json={"userName": "bob"}, headers=_AUTH
    ).json()["id"]
    assert _client.delete(f"/scim/v2/Users/{user_id}", headers=_AUTH).status_code == 204
    assert _client.get(f"/scim/v2/Users/{user_id}", headers=_AUTH).status_code == 404


def test_create_user_requires_username(_client: TestClient) -> None:
    assert _client.post("/scim/v2/Users", json={}, headers=_AUTH).status_code == 400


# -- SCIM Groups -----------------------------------------------------------


def test_provision_and_delete_group(_client: TestClient) -> None:
    resp = _client.post(
        "/scim/v2/Groups", json={"displayName": "engineering"}, headers=_AUTH
    )
    assert resp.status_code == 201
    group_id = resp.json()["id"]
    assert resp.json()["displayName"] == "engineering"

    listed = _client.get("/scim/v2/Groups", headers=_AUTH)
    assert any(g["displayName"] == "engineering" for g in listed.json()["Resources"])

    assert (
        _client.delete(f"/scim/v2/Groups/{group_id}", headers=_AUTH).status_code == 204
    )


# -- Key rotation ----------------------------------------------------------


def test_rotation_noop_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ROUTEIQ_KEY_ROTATION_ENABLED", raising=False)
    from litellm_llmrouter.governance import get_governance_engine

    engine = get_governance_engine()
    engine.register_key_governance(
        KeyGovernance(key_id="k1", metadata={"auto_rotate": True, "created_at": 0})
    )
    assert rotate_stale_keys(engine) == []


def test_rotation_rotates_stale_auto_rotate_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROUTEIQ_KEY_ROTATION_ENABLED", "true")
    from litellm_llmrouter.governance import get_governance_engine

    engine = get_governance_engine()
    now = time.time()
    # stale + auto_rotate -> rotated
    engine.register_key_governance(
        KeyGovernance(
            key_id="sk-old",
            metadata={"auto_rotate": True, "created_at": now - 200 * 86400},
        )
    )
    # fresh -> not rotated
    engine.register_key_governance(
        KeyGovernance(
            key_id="sk-fresh",
            metadata={"auto_rotate": True, "created_at": now - 1 * 86400},
        )
    )
    # no auto_rotate -> not rotated
    engine.register_key_governance(
        KeyGovernance(
            key_id="sk-manual",
            metadata={"created_at": now - 999 * 86400},
        )
    )

    rotated = rotate_stale_keys(engine, max_age_seconds=90 * 86400, now=now)
    assert [r["key_id"] for r in rotated] == ["sk-old"]
    kg = engine.get_key_governance("sk-old")
    assert kg.metadata["public_id"].startswith("kid_")
    assert "secret_hash" in kg.metadata
    assert kg.metadata["masked"].startswith("sk-rq-...")

    # idempotent within window: rotated_at now reset -> nothing more to rotate
    assert rotate_stale_keys(engine, max_age_seconds=90 * 86400, now=now) == []
