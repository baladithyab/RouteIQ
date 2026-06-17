"""Tests for the AWS Secrets Manager credential vault (RouteIQ-3d33).

Cred-free: the boto3 client is mocked. Covers the default-off pass-through, the
``aws-secrets://`` reference resolution (whole secret + JSON-key form), caching,
invalidation, and fail-safe behaviour.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from litellm_llmrouter.secrets_vault import (
    SecretsVault,
    is_vault_reference,
    reset_secrets_vault,
    resolve_secret,
)


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_secrets_vault()
    yield
    reset_secrets_vault()


def test_disabled_by_default_is_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ROUTEIQ_SECRETS_VAULT_ENABLED", raising=False)
    vault = SecretsVault()
    assert vault.enabled is False
    # Even a reference is returned unchanged when disabled (no AWS call).
    assert vault.resolve("aws-secrets://prod/openai") == "aws-secrets://prod/openai"
    assert vault.resolve("plain-key") == "plain-key"


def test_is_vault_reference() -> None:
    assert is_vault_reference("aws-secrets://id") is True
    assert is_vault_reference("aws-secrets://id#key") is True
    assert is_vault_reference("plain") is False
    assert is_vault_reference(None) is False


def test_non_reference_returned_unchanged_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROUTEIQ_SECRETS_VAULT_ENABLED", "true")
    vault = SecretsVault()
    assert vault.enabled is True
    assert vault.resolve("sk-plain-literal") == "sk-plain-literal"


def test_resolve_whole_secret_string(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROUTEIQ_SECRETS_VAULT_ENABLED", "true")
    client = MagicMock()
    client.get_secret_value.return_value = {"SecretString": "sk-resolved-value"}
    vault = SecretsVault()
    with patch.object(vault, "_get_client", return_value=client):
        assert vault.resolve("aws-secrets://prod/openai") == "sk-resolved-value"
    client.get_secret_value.assert_called_once_with(SecretId="prod/openai")


def test_resolve_json_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROUTEIQ_SECRETS_VAULT_ENABLED", "true")
    client = MagicMock()
    client.get_secret_value.return_value = {
        "SecretString": json.dumps({"openai_key": "sk-x", "anthropic_key": "sk-y"})
    }
    vault = SecretsVault()
    with patch.object(vault, "_get_client", return_value=client):
        assert vault.resolve("aws-secrets://prod/llm#openai_key") == "sk-x"
        # cached -> a second resolve does not re-fetch
        assert vault.resolve("aws-secrets://prod/llm#anthropic_key") == "sk-y"
    client.get_secret_value.assert_called_once()


def test_missing_json_key_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROUTEIQ_SECRETS_VAULT_ENABLED", "true")
    client = MagicMock()
    client.get_secret_value.return_value = {"SecretString": json.dumps({"a": "1"})}
    vault = SecretsVault()
    with patch.object(vault, "_get_client", return_value=client):
        assert vault.resolve("aws-secrets://s#nope") is None


def test_fetch_error_is_failsafe_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROUTEIQ_SECRETS_VAULT_ENABLED", "true")
    client = MagicMock()
    client.get_secret_value.side_effect = RuntimeError("AccessDenied")
    vault = SecretsVault()
    with patch.object(vault, "_get_client", return_value=client):
        assert vault.resolve("aws-secrets://denied") is None


def test_invalidate_clears_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROUTEIQ_SECRETS_VAULT_ENABLED", "true")
    client = MagicMock()
    client.get_secret_value.return_value = {"SecretString": "v1"}
    vault = SecretsVault()
    with patch.object(vault, "_get_client", return_value=client):
        assert vault.resolve("aws-secrets://rot") == "v1"
        vault.invalidate("rot")
        client.get_secret_value.return_value = {"SecretString": "v2"}
        assert vault.resolve("aws-secrets://rot") == "v2"
    assert client.get_secret_value.call_count == 2


def test_module_level_resolve_uses_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ROUTEIQ_SECRETS_VAULT_ENABLED", raising=False)
    assert resolve_secret("plain") == "plain"
