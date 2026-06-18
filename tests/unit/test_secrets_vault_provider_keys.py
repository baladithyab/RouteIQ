"""Phase-2 provider-key resolver tests (RouteIQ-1786).

Covers the single resolver helper that scattered provider/model API-key
call-sites migrate to, plus proof that a REAL migrated call-site (the MCP
JSON-RPC client's outbound auth-header builder) now routes its upstream
credential through the vault.

Cred-free: boto3 is mocked via ``SecretsVault._get_client``. The contract:

* vault disabled (default) -> resolver returns exactly ``os.getenv`` / the
  in-hand value (byte-stable);
* vault enabled + mock     -> an ``aws-secrets://`` reference is dereferenced;
* missing both             -> ``None`` (or the supplied default), as before.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from litellm_llmrouter import secrets_vault
from litellm_llmrouter.mcp_jsonrpc_client import _build_headers
from litellm_llmrouter.secrets_vault import (
    get_secrets_vault,
    reset_secrets_vault,
    resolve_provider_key,
    resolve_provider_value,
)


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_secrets_vault()
    yield
    reset_secrets_vault()


# ---------------------------------------------------------------------------
# resolve_provider_key (env-name form): vault-first, os.getenv fallback
# ---------------------------------------------------------------------------


def test_provider_key_disabled_is_os_getenv_bytestable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vault OFF => resolve_provider_key == os.getenv (byte-stable)."""
    monkeypatch.delenv("ROUTEIQ_SECRETS_VAULT_ENABLED", raising=False)
    monkeypatch.setenv("MY_PROVIDER_KEY", "sk-plain-env-value")
    assert resolve_provider_key("MY_PROVIDER_KEY") == "sk-plain-env-value"
    # Even an aws-secrets:// reference is returned verbatim when disabled
    # (no AWS call) -- identical to the pre-migration os.getenv behaviour.
    monkeypatch.setenv("MY_PROVIDER_KEY", "aws-secrets://prod/openai")
    assert resolve_provider_key("MY_PROVIDER_KEY") == "aws-secrets://prod/openai"


def test_provider_key_missing_both_returns_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing env var => None (or the supplied default), as before."""
    monkeypatch.delenv("ROUTEIQ_SECRETS_VAULT_ENABLED", raising=False)
    monkeypatch.delenv("ABSENT_KEY", raising=False)
    assert resolve_provider_key("ABSENT_KEY") is None
    assert resolve_provider_key("ABSENT_KEY", default="fallback") == "fallback"


def test_provider_key_enabled_dereferences_vault_ref(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vault ON + mock => env value of aws-secrets://id#key is dereferenced."""
    monkeypatch.setenv("ROUTEIQ_SECRETS_VAULT_ENABLED", "true")
    monkeypatch.setenv("MY_PROVIDER_KEY", "aws-secrets://prod/llm#openai_key")
    client = MagicMock()
    client.get_secret_value.return_value = {
        "SecretString": json.dumps({"openai_key": "sk-from-vault"})
    }
    vault = get_secrets_vault()
    with patch.object(vault, "_get_client", return_value=client):
        assert resolve_provider_key("MY_PROVIDER_KEY") == "sk-from-vault"
    client.get_secret_value.assert_called_once_with(SecretId="prod/llm")


def test_provider_key_enabled_plain_env_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vault ON but a plain (non-reference) env value is returned unchanged."""
    monkeypatch.setenv("ROUTEIQ_SECRETS_VAULT_ENABLED", "true")
    monkeypatch.setenv("MY_PROVIDER_KEY", "sk-literal")
    # No client needed: a non-reference short-circuits before any AWS call.
    assert resolve_provider_key("MY_PROVIDER_KEY") == "sk-literal"


# ---------------------------------------------------------------------------
# resolve_provider_value (in-hand value form)
# ---------------------------------------------------------------------------


def test_provider_value_disabled_passthrough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ROUTEIQ_SECRETS_VAULT_ENABLED", raising=False)
    assert resolve_provider_value("sk-in-hand") == "sk-in-hand"
    assert resolve_provider_value("aws-secrets://x") == "aws-secrets://x"
    assert resolve_provider_value(None) is None


def test_provider_value_enabled_dereferences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROUTEIQ_SECRETS_VAULT_ENABLED", "true")
    client = MagicMock()
    client.get_secret_value.return_value = {"SecretString": "sk-resolved"}
    vault = get_secrets_vault()
    with patch.object(vault, "_get_client", return_value=client):
        assert resolve_provider_value("aws-secrets://prod/mcp") == "sk-resolved"


# ---------------------------------------------------------------------------
# REAL migrated call-site: mcp_jsonrpc_client._build_headers
# Proves the live outbound-auth path now routes the upstream credential
# through the resolver (not just a direct resolver call).
# ---------------------------------------------------------------------------


def test_build_headers_bearer_disabled_is_bytestable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vault OFF => the bearer header is byte-identical to pre-migration."""
    monkeypatch.delenv("ROUTEIQ_SECRETS_VAULT_ENABLED", raising=False)
    headers = _build_headers("bearer_token", {"auth_token": "tok-literal"})
    assert headers["Authorization"] == "Bearer tok-literal"


def test_build_headers_apikey_disabled_is_bytestable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ROUTEIQ_SECRETS_VAULT_ENABLED", raising=False)
    headers = _build_headers("api_key", {"api_key": "key-literal"})
    assert headers["X-API-Key"] == "key-literal"


def test_build_headers_bearer_resolves_vault_ref_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vault ON => an aws-secrets:// auth_token is dereferenced in the header.

    This is the proof the resolver is WIRED into the live call-site: the
    upstream MCP credential is fetched from the (mocked) vault, not used raw.
    """
    monkeypatch.setenv("ROUTEIQ_SECRETS_VAULT_ENABLED", "true")
    client = MagicMock()
    client.get_secret_value.return_value = {
        "SecretString": json.dumps({"token": "tok-from-vault"})
    }
    vault = get_secrets_vault()
    with patch.object(vault, "_get_client", return_value=client):
        headers = _build_headers(
            "bearer_token", {"auth_token": "aws-secrets://prod/mcp#token"}
        )
    assert headers["Authorization"] == "Bearer tok-from-vault"
    client.get_secret_value.assert_called_once_with(SecretId="prod/mcp")


def test_build_headers_apikey_resolves_vault_ref_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROUTEIQ_SECRETS_VAULT_ENABLED", "true")
    client = MagicMock()
    client.get_secret_value.return_value = {"SecretString": "key-from-vault"}
    vault = get_secrets_vault()
    with patch.object(vault, "_get_client", return_value=client):
        headers = _build_headers("api_key", {"api_key": "aws-secrets://prod/mcp-key"})
    assert headers["X-API-Key"] == "key-from-vault"


def test_build_headers_vault_resolution_failure_omits_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vault ON but resolution fails => header omitted (no 'Bearer None')."""
    monkeypatch.setenv("ROUTEIQ_SECRETS_VAULT_ENABLED", "true")
    client = MagicMock()
    client.get_secret_value.side_effect = RuntimeError("AccessDenied")
    vault = get_secrets_vault()
    with patch.object(vault, "_get_client", return_value=client):
        headers = _build_headers("bearer_token", {"auth_token": "aws-secrets://denied"})
    assert "Authorization" not in headers


def test_exports_include_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    assert "resolve_provider_key" in secrets_vault.__all__
    assert "resolve_provider_value" in secrets_vault.__all__
