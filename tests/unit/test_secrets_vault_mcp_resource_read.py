"""Live-callsite wiring test: MCPGateway.proxy_resource_read routes its
upstream auth credential through the secrets vault (RouteIQ-ea5b).

This mirrors ``test_secrets_vault_provider_keys.py`` (which proves the FIRST
migrated call-site, ``mcp_jsonrpc_client._build_headers``) for the SECOND
migrated call-site: the REST resource-read proxy at
``mcp_gateway.MCPGateway.proxy_resource_read`` (~L955-963). The resolver under
test is ``secrets_vault.resolve_provider_value``.

Unlike a direct resolver call, this drives the REAL ``proxy_resource_read``
coroutine end-to-end -- the SSRF check and the outbound HTTP client are mocked
so no network/DNS I/O happens, but the production header-building code path
(which calls ``resolve_provider_value``) actually executes and the headers it
hands to the upstream POST are captured and asserted on.

Cred-free: boto3 is mocked via ``SecretsVault._get_client``. Contract:

* vault disabled (default) -> the literal ``auth_token`` / ``api_key`` flows
  through byte-stable (identical to pre-migration ``os.getenv`` behaviour);
* vault enabled + mock     -> an ``aws-secrets://<id>[#key]`` reference is
  dereferenced and the vault-resolved value is what reaches the wire.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm_llmrouter.mcp_gateway as mcp_gw_mod
from litellm_llmrouter.mcp_gateway import (
    MCPGateway,
    MCPServer,
    reset_mcp_gateway,
)
from litellm_llmrouter.secrets_vault import (
    get_secrets_vault,
    reset_secrets_vault,
)


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_mcp_gateway()
    reset_secrets_vault()
    yield
    reset_mcp_gateway()
    reset_secrets_vault()
    mcp_gw_mod._mcp_gateway = None


_RESOURCE_URI = "file:///secure/doc.txt"


def _install_gateway(*, auth_type: str, metadata: dict) -> MCPGateway:
    """Install a singleton MCP gateway owning one resource on an auth'd server."""
    gw = MCPGateway()
    gw.enabled = True
    gw._tool_invocation_enabled = True
    gw.register_server(
        MCPServer(
            server_id="srv-res",
            name="res",
            url="https://upstream.example.com/mcp",
            resources=[_RESOURCE_URI],
            auth_type=auth_type,
            metadata=metadata,
        )
    )
    mcp_gw_mod._mcp_gateway = gw
    return gw


@asynccontextmanager
async def _fake_client_cm(captured: dict):
    """Stand-in for ``get_client_for_request`` -> yields a client whose ``post``
    records the outbound headers and returns a valid MCP ``contents`` payload."""

    async def _post(url, json=None, headers=None, timeout=None):  # noqa: A002
        captured["url"] = url
        captured["headers"] = headers
        resp = MagicMock()
        resp.status_code = 200
        resp.json = MagicMock(
            return_value={"contents": [{"uri": _RESOURCE_URI, "text": "ok"}]}
        )
        return resp

    client = MagicMock()
    client.post = AsyncMock(side_effect=_post)
    yield client


async def _drive_proxy_resource_read(gw: MCPGateway, captured: dict) -> dict:
    """Run the REAL proxy_resource_read with SSRF + HTTP transport mocked.

    The header-building under test (resolve_provider_value -> Authorization /
    X-API-Key) runs in production code BEFORE the outbound HTTP call. The SSRF
    DNS check and the client are stubbed so no I/O happens. ``httpx.Timeout`` is
    also stubbed because the installed httpx rejects the production 3-arg
    ``Timeout(connect=, read=, write=)`` construction (a pre-existing,
    resolver-unrelated incompatibility filed as a seed) -- stubbing it keeps the
    real header path reachable instead of masking the wiring with a ValueError.
    """

    def _cm(*args, **kwargs):
        return _fake_client_cm(captured)

    with (
        patch(
            "litellm_llmrouter.mcp_gateway.validate_outbound_url_async",
            new=AsyncMock(return_value="https://upstream.example.com/mcp"),
        ),
        patch("litellm_llmrouter.mcp_gateway.get_client_for_request", new=_cm),
        patch(
            "litellm_llmrouter.mcp_gateway.httpx.Timeout",
            new=MagicMock(return_value="timeout-sentinel"),
        ),
    ):
        result = await gw.proxy_resource_read(_RESOURCE_URI)
    return result


# ---------------------------------------------------------------------------
# Vault DISABLED (default) -> literal pass-through, byte-stable
# ---------------------------------------------------------------------------


async def test_resource_read_bearer_disabled_is_bytestable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vault OFF => the bearer header is byte-identical to pre-migration."""
    monkeypatch.delenv("ROUTEIQ_SECRETS_VAULT_ENABLED", raising=False)
    gw = _install_gateway(
        auth_type="bearer_token", metadata={"auth_token": "tok-literal"}
    )
    captured: dict = {}
    result = await _drive_proxy_resource_read(gw, captured)

    assert "error" not in result
    assert captured["headers"]["Authorization"] == "Bearer tok-literal"
    # An aws-secrets:// reference is ALSO returned verbatim when disabled
    # (no AWS call) -- identical to the pre-migration literal behaviour.
    gw2 = _install_gateway(
        auth_type="bearer_token",
        metadata={"auth_token": "aws-secrets://prod/mcp#token"},
    )
    captured2: dict = {}
    await _drive_proxy_resource_read(gw2, captured2)
    assert (
        captured2["headers"]["Authorization"] == "Bearer aws-secrets://prod/mcp#token"
    )


async def test_resource_read_apikey_disabled_is_bytestable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ROUTEIQ_SECRETS_VAULT_ENABLED", raising=False)
    gw = _install_gateway(auth_type="api_key", metadata={"api_key": "key-literal"})
    captured: dict = {}
    result = await _drive_proxy_resource_read(gw, captured)

    assert "error" not in result
    assert captured["headers"]["X-API-Key"] == "key-literal"


# ---------------------------------------------------------------------------
# Vault ENABLED + mock -> the resolver is WIRED into the live call-site
# ---------------------------------------------------------------------------


async def test_resource_read_bearer_resolves_vault_ref_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vault ON => an aws-secrets:// auth_token is dereferenced before the wire.

    Proof the resolver is wired into proxy_resource_read: the upstream MCP
    credential is fetched from the (mocked) vault, not used raw.
    """
    monkeypatch.setenv("ROUTEIQ_SECRETS_VAULT_ENABLED", "true")
    client = MagicMock()
    client.get_secret_value.return_value = {
        "SecretString": json.dumps({"token": "tok-from-vault"})
    }
    vault = get_secrets_vault()
    gw = _install_gateway(
        auth_type="bearer_token",
        metadata={"auth_token": "aws-secrets://prod/mcp#token"},
    )
    captured: dict = {}
    with patch.object(vault, "_get_client", return_value=client):
        result = await _drive_proxy_resource_read(gw, captured)

    assert "error" not in result
    assert captured["headers"]["Authorization"] == "Bearer tok-from-vault"
    client.get_secret_value.assert_called_once_with(SecretId="prod/mcp")


async def test_resource_read_apikey_resolves_vault_ref_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROUTEIQ_SECRETS_VAULT_ENABLED", "true")
    client = MagicMock()
    client.get_secret_value.return_value = {"SecretString": "key-from-vault"}
    vault = get_secrets_vault()
    gw = _install_gateway(
        auth_type="api_key", metadata={"api_key": "aws-secrets://prod/mcp-key"}
    )
    captured: dict = {}
    with patch.object(vault, "_get_client", return_value=client):
        result = await _drive_proxy_resource_read(gw, captured)

    assert "error" not in result
    assert captured["headers"]["X-API-Key"] == "key-from-vault"
