"""AWS Secrets Manager credential vault backend for provider keys (RouteIQ-3d33).

Resolves provider API keys (and any other gateway secret) from AWS Secrets
Manager instead of plaintext environment variables, so production deployments
keep credentials in a managed vault with rotation + audit instead of baking them
into the process environment.

Design (mirrors the rest of the cluster -- additive, gated, cred-free testable):

* DEFAULT-OFF. The vault is disabled unless ``ROUTEIQ_SECRETS_VAULT_ENABLED=true``.
  When disabled, :meth:`SecretsVault.resolve` is a transparent pass-through that
  returns the value already in the environment, so a default deployment behaves
  byte-for-byte as it does today (no boto3 import, no AWS call).

* Reference syntax. A provider key whose value is the placeholder
  ``aws-secrets://<secret-id>[#<json-key>]`` is resolved at lookup time:
    - ``aws-secrets://prod/openai`` -> the whole ``SecretString``
    - ``aws-secrets://prod/llm#openai_key`` -> the ``openai_key`` field of the
      JSON ``SecretString``
  A plain (non-reference) value is returned unchanged even when the vault is on.

* Caching. Resolved secrets are cached in-process (the value rarely changes
  within a process lifetime); :meth:`invalidate` clears the cache for rotation.

* Fail-safe. A resolution error logs and returns ``None`` (the caller decides
  whether a missing key is fatal) -- it never raises into the hot path.

The boto3 client is created lazily on first real resolution so importing this
module has no AWS dependency and unit tests can mock the client without creds.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger("litellm_llmrouter.secrets_vault")

#: Reference scheme an operator uses to point a config value at a vault secret.
VAULT_REF_PREFIX = "aws-secrets://"


def _vault_enabled() -> bool:
    """Whether the Secrets Manager vault backend is active (default OFF)."""
    return os.getenv("ROUTEIQ_SECRETS_VAULT_ENABLED", "false").lower() == "true"


def is_vault_reference(value: Optional[str]) -> bool:
    """True when ``value`` is an ``aws-secrets://...`` vault reference."""
    return isinstance(value, str) and value.startswith(VAULT_REF_PREFIX)


def _parse_reference(value: str) -> tuple[str, Optional[str]]:
    """Split ``aws-secrets://<secret-id>[#<json-key>]`` into (secret_id, key)."""
    body = value[len(VAULT_REF_PREFIX) :]
    if "#" in body:
        secret_id, json_key = body.split("#", 1)
        return secret_id, (json_key or None)
    return body, None


class SecretsVault:
    """Resolves provider keys from AWS Secrets Manager (gated, cached).

    Reads enablement + region from the environment at construction. When
    disabled, :meth:`resolve` is a pass-through, so a caller can unconditionally
    route every secret through the vault and pay nothing when it is off.
    """

    def __init__(self, region: Optional[str] = None) -> None:
        self._enabled = _vault_enabled()
        self._region = region or os.getenv("AWS_REGION", "us-east-1")
        self._client: Any = None
        self._cache: dict[str, str] = {}

    @property
    def enabled(self) -> bool:
        """True when the vault backend is turned on."""
        return self._enabled

    def _get_client(self) -> Any:
        """Lazily build the Secrets Manager boto3 client (no creds at import)."""
        if self._client is None:
            import boto3  # local import: optional dependency

            self._client = boto3.client("secretsmanager", region_name=self._region)
        return self._client

    def _fetch_secret_string(self, secret_id: str) -> Optional[str]:
        """Fetch a secret's raw ``SecretString`` (cached, fail-safe)."""
        if secret_id in self._cache:
            return self._cache[secret_id]
        try:
            resp = self._get_client().get_secret_value(SecretId=secret_id)
        except Exception as exc:  # pragma: no cover - exercised via mock raise
            logger.warning("SecretsVault: failed to fetch %s: %s", secret_id, exc)
            return None
        secret_string = resp.get("SecretString")
        if secret_string is None:
            logger.warning("SecretsVault: secret %s has no SecretString", secret_id)
            return None
        self._cache[secret_id] = secret_string
        return secret_string

    def resolve(self, value: Optional[str]) -> Optional[str]:
        """Resolve a config value, dereferencing a vault reference if present.

        * Vault OFF -> returns ``value`` unchanged (pass-through).
        * Non-reference value -> returned unchanged (even when vault is on).
        * ``aws-secrets://id`` -> the whole ``SecretString``.
        * ``aws-secrets://id#field`` -> the ``field`` of the JSON ``SecretString``.

        Returns ``None`` on any resolution failure (the caller decides fatality).
        """
        if not value:
            return value
        if not self._enabled or not is_vault_reference(value):
            return value

        secret_id, json_key = _parse_reference(value)
        secret_string = self._fetch_secret_string(secret_id)
        if secret_string is None:
            return None
        if json_key is None:
            return secret_string
        try:
            data = json.loads(secret_string)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning(
                "SecretsVault: secret %s is not JSON but a key '%s' was requested: %s",
                secret_id,
                json_key,
                exc,
            )
            return None
        if json_key not in data:
            logger.warning(
                "SecretsVault: key '%s' not in secret %s", json_key, secret_id
            )
            return None
        return str(data[json_key])

    def invalidate(self, secret_id: Optional[str] = None) -> None:
        """Drop cached secret(s) so the next resolve re-fetches (rotation)."""
        if secret_id is None:
            self._cache.clear()
        else:
            self._cache.pop(secret_id, None)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_vault: Optional[SecretsVault] = None


def get_secrets_vault() -> SecretsVault:
    """Get or create the global secrets vault singleton."""
    global _vault
    if _vault is None:
        _vault = SecretsVault()
    return _vault


def reset_secrets_vault() -> None:
    """Reset the secrets vault singleton (for testing)."""
    global _vault
    _vault = None


def resolve_secret(value: Optional[str]) -> Optional[str]:
    """Module-level convenience: resolve through the singleton vault."""
    return get_secrets_vault().resolve(value)


__all__ = [
    "VAULT_REF_PREFIX",
    "SecretsVault",
    "is_vault_reference",
    "get_secrets_vault",
    "reset_secrets_vault",
    "resolve_secret",
]
