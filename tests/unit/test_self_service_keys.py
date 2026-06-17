"""Tests for self-service key management (RouteIQ-3215).

A logged-in user manages THEIR OWN API keys via the user-tier endpoints
(``/api/v1/routeiq/me/keys``) -- admin auth is NOT required.  Ownership is taken
from the authenticated caller's identity, never from request input, so:

* a user can create a key scoped to their own identity (secret returned once),
* a user lists ONLY their own keys,
* a user CANNOT see or revoke another user's keys (scope isolation), and
* the endpoints are on the user tier (admin auth not required).

All offline / credential-free: ``user_api_key_auth`` is overridden per caller and
the governance store stays in-memory (no DATABASE_URL).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from litellm_llmrouter.gateway import create_standalone_app


def _client_for_caller(caller_key: str) -> TestClient:
    """A TestClient whose user_api_key_auth resolves to *caller_key*."""
    app = create_standalone_app(enable_plugins=False, enable_resilience=False)

    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    app.dependency_overrides[user_api_key_auth] = lambda: {"api_key": caller_key}
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _reset_governance():
    """Reset governance singletons so keys don't leak across tests."""
    from litellm_llmrouter.governance import reset_governance_engine
    from litellm_llmrouter.governance_store import reset_governance_store

    reset_governance_engine()
    reset_governance_store()
    yield
    reset_governance_engine()
    reset_governance_store()


class TestSelfServiceKeyCreate:
    def test_user_creates_key_scoped_to_their_identity(self):
        """A user creates a key; it is owned by THEM and the secret is returned."""
        client = _client_for_caller("alice")

        resp = client.post(
            "/api/v1/routeiq/me/keys",
            json={"name": "ci-bot", "max_budget_usd": 25.0},
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()

        assert body["owner_id"] == "alice"
        assert body["name"] == "ci-bot"
        assert body["max_budget_usd"] == 25.0
        # Plaintext secret returned exactly once, at creation.
        secret = body["key"]
        assert secret and secret.startswith("sk-rq-")
        # The public id is NOT the secret -- it is a separate, non-secret handle.
        assert body["key_id"] != secret
        assert body["key_id"].startswith("kid_")
        # A masked, non-recoverable preview accompanies the public id.
        assert body["masked"] and body["masked"].endswith(secret[-4:])
        assert secret not in body["masked"]

        # The governance row is keyed on the raw secret (spend-scope contract),
        # but the public id stored in metadata is what the API surfaces address.
        from litellm_llmrouter.governance import get_governance_engine

        engine = get_governance_engine()
        kg = engine.get_key_governance(secret)
        assert kg is not None
        assert kg.metadata["owner_id"] == "alice"
        assert kg.metadata["public_id"] == body["key_id"]
        # Only a hash of the secret is retained -- never the plaintext itself.
        import hashlib

        assert (
            kg.metadata["secret_hash"]
            == hashlib.sha256(secret.encode("utf-8")).hexdigest()
        )
        assert "secret" not in kg.metadata or kg.metadata.get("secret") != secret

    def test_plaintext_secret_is_not_recoverable_after_creation(self):
        """The plaintext secret is returned ONCE and never recoverable later.

        Neither the list path nor the governance store exposes a field that
        contains (or can be used to reconstruct) the plaintext secret.
        """
        client = _client_for_caller("alice")
        created = client.post("/api/v1/routeiq/me/keys", json={"name": "k"})
        secret = created.json()["key"]
        assert secret.startswith("sk-rq-")

        # The list path NEVER carries the plaintext secret in any field.
        listing = client.get("/api/v1/routeiq/me/keys").json()
        assert listing["count"] == 1
        entry = listing["keys"][0]
        assert entry["key"] is None
        assert secret not in entry.values()
        # The masked preview is present and does not reveal the secret.
        assert entry["masked"] and secret not in entry["masked"]
        # The serialized list response, in full, never contains the secret.
        import json as _json

        assert secret not in _json.dumps(listing)

        # The governance metadata stores only a hash, never the plaintext.
        from litellm_llmrouter.governance import get_governance_engine

        kg = get_governance_engine().get_key_governance(secret)
        assert secret not in _json.dumps(kg.metadata)

    def test_create_rejects_anonymous_caller(self):
        """A caller with no resolvable identity cannot mint keys."""
        app = create_standalone_app(enable_plugins=False, enable_resilience=False)
        from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

        # No identity fields at all -> _resolve_caller_key_id == "anonymous".
        app.dependency_overrides[user_api_key_auth] = lambda: {}
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/api/v1/routeiq/me/keys", json={"name": "x"})
        assert resp.status_code == 401


class TestSelfServiceKeyList:
    def test_user_lists_only_their_own_keys(self):
        """List returns ONLY the caller's keys, not other users' keys."""
        alice = _client_for_caller("alice")
        bob = _client_for_caller("bob")

        alice.post("/api/v1/routeiq/me/keys", json={"name": "a1"})
        alice.post("/api/v1/routeiq/me/keys", json={"name": "a2"})
        bob.post("/api/v1/routeiq/me/keys", json={"name": "b1"})

        a_list = alice.get("/api/v1/routeiq/me/keys")
        assert a_list.status_code == 200
        a_body = a_list.json()
        assert a_body["count"] == 2
        assert {k["name"] for k in a_body["keys"]} == {"a1", "a2"}
        assert all(k["owner_id"] == "alice" for k in a_body["keys"])
        # The list path never leaks the secret -- only non-secret public ids
        # (kid_...) and masked previews are returned.
        assert all(k["key"] is None for k in a_body["keys"])
        assert all(k["key_id"].startswith("kid_") for k in a_body["keys"])
        assert all(not k["key_id"].startswith("sk-rq-") for k in a_body["keys"])

        b_list = bob.get("/api/v1/routeiq/me/keys")
        assert b_list.json()["count"] == 1
        assert b_list.json()["keys"][0]["name"] == "b1"


class TestSelfServiceKeyScopeIsolation:
    def test_user_cannot_revoke_another_users_key(self):
        """Revoking someone else's key returns 404 (no ownership leak)."""
        alice = _client_for_caller("alice")
        bob = _client_for_caller("bob")

        created = alice.post("/api/v1/routeiq/me/keys", json={"name": "secret"})
        created_body = created.json()
        alice_public_id = created_body["key_id"]
        alice_secret = created_body["key"]
        # The path param is the NON-SECRET public id, never the plaintext secret.
        assert alice_public_id.startswith("kid_")

        # Bob tries to revoke Alice's key by its public id -> 404 (indistinguishable
        # from missing).
        resp = bob.delete(f"/api/v1/routeiq/me/keys/{alice_public_id}")
        assert resp.status_code == 404

        # Bob also cannot revoke it by guessing the raw secret as the path param
        # (the secret is not an addressable handle for ANY caller).
        resp_secret = bob.delete(f"/api/v1/routeiq/me/keys/{alice_secret}")
        assert resp_secret.status_code == 404

        # Alice's key still exists and is functional (looked up by the raw key).
        from litellm_llmrouter.governance import get_governance_engine

        assert get_governance_engine().get_key_governance(alice_secret) is not None

    def test_user_cannot_see_another_users_key_in_list(self):
        """A user's list never contains another user's key id."""
        alice = _client_for_caller("alice")
        bob = _client_for_caller("bob")

        created = alice.post("/api/v1/routeiq/me/keys", json={"name": "a"})
        alice_key_id = created.json()["key_id"]

        b_list = bob.get("/api/v1/routeiq/me/keys").json()
        assert all(k["key_id"] != alice_key_id for k in b_list["keys"])
        assert b_list["count"] == 0

    def test_user_revokes_their_own_key(self):
        """A user can revoke a key they own, addressing it by public id."""
        alice = _client_for_caller("alice")
        created = alice.post("/api/v1/routeiq/me/keys", json={"name": "a"})
        public_id = created.json()["key_id"]
        secret = created.json()["key"]
        assert public_id.startswith("kid_")

        resp = alice.delete(f"/api/v1/routeiq/me/keys/{public_id}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True
        # The response echoes only the non-secret public id, never the secret.
        assert resp.json()["key_id"] == public_id
        assert secret not in resp.text

        # Gone from the engine (keyed on the raw secret) and the caller's list.
        from litellm_llmrouter.governance import get_governance_engine

        assert get_governance_engine().get_key_governance(secret) is None
        assert alice.get("/api/v1/routeiq/me/keys").json()["count"] == 0


class TestSelfServiceKeyUserTier:
    def test_endpoints_do_not_require_admin_auth(self):
        """The key endpoints are user-tier: no admin key is presented and they
        still succeed (they live on llmrouter_router, not admin_router)."""
        client = _client_for_caller("carol")

        # No X-Admin-API-Key header anywhere -- pure user-tier.
        create = client.post("/api/v1/routeiq/me/keys", json={"name": "k"})
        assert create.status_code == 201

        listing = client.get("/api/v1/routeiq/me/keys")
        assert listing.status_code == 200

        key_id = create.json()["key_id"]
        revoke = client.delete(f"/api/v1/routeiq/me/keys/{key_id}")
        assert revoke.status_code == 200
