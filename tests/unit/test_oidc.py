"""
Unit tests for oidc.py — OIDC / SSO Identity Integration
=========================================================

Tests cover:
1. OIDCConfig model validation (defaults, HTTPS enforcement, bounds)
2. OIDCDiscovery (discovery doc fetch/cache, JWKS fetch/cache, error handling)
3. JWTValidator (token validation, claim extraction, role mapping)
4. OIDC Router endpoints (SSO login, callback, token exchange, userinfo)
5. resolve_identity() (disabled, valid JWT, invalid token, exchanged keys)
6. Singleton lifecycle (setup_oidc, reset_oidc, get_oidc_config)
7. Internal helpers (state pruning, hash_token, email domain enforcement)
"""

from __future__ import annotations

import base64
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from starlette.requests import Request

from litellm_llmrouter.oidc import (
    JWTValidator,
    OIDCAuthError,
    OIDCConfig,
    OIDCDiscovery,
    OIDCDiscoveryError,
    OIDCIdentity,
    OIDCProviderMetadata,
    TokenExchangeRequest,
    TokenExchangeResponse,
    _exchanged_keys,
    _generate_state,
    _hash_token,
    _identity_cache,
    _pending_auth_states,
    _prune_expired_identities,
    _prune_expired_states,
    create_oidc_router,
    get_oidc_config,
    reset_oidc,
    resolve_identity,
    setup_oidc,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _reset_oidc_state():
    """Reset all OIDC module state between tests."""
    reset_oidc()
    yield
    reset_oidc()


SAMPLE_DISCOVERY = {
    "issuer": "https://auth.example.com/realms/main",
    "authorization_endpoint": "https://auth.example.com/realms/main/protocol/openid-connect/auth",
    "token_endpoint": "https://auth.example.com/realms/main/protocol/openid-connect/token",
    "userinfo_endpoint": "https://auth.example.com/realms/main/protocol/openid-connect/userinfo",
    "jwks_uri": "https://auth.example.com/realms/main/protocol/openid-connect/certs",
    "scopes_supported": ["openid", "email", "profile"],
    "response_types_supported": ["code"],
    "end_session_endpoint": "https://auth.example.com/realms/main/protocol/openid-connect/logout",
}

SAMPLE_JWKS = {
    "keys": [
        {
            "kty": "RSA",
            "kid": "test-key-1",
            "use": "sig",
            "n": "0vx7agoebGcQSuuPiLJXZptN9nndrQmbXEps2aiAFbWhM78LhWx4cbbfAAtVT86zwu1RK7aPFFxuhDR1L6tSoc_BJECPebWKRXjBZCiFV4n3oknjhMstn64tZ_2W-5JsGY4Hc5n9yBXArwl93lqt7_RN5w6Cf0h4QyQ5v-65YGjQR0_FDW2QvzqY368QQMicAtaSqzs8KJZgnYb9c7d0zgdAZHzu6qMQvRL5hajrn1n91CbOpbISD08qNLyrdkt-bFTWhAI4vMQFh6WeZu0fM4lFd2NcRwr3XPksINHaQ-G_xBniIqbw0Ls1jF44-csFCur-kEgU8awapJzKnqDKgw",
            "e": "AQAB",
        }
    ]
}


def _make_config(**overrides) -> OIDCConfig:
    """Helper to create an OIDCConfig with sensible defaults for testing."""
    defaults = {
        "enabled": True,
        "issuer_url": "https://auth.example.com/realms/main",
        "client_id": "routeiq-gateway",
        "client_secret": "test-secret",
        "session_ttl": 1800,
    }
    defaults.update(overrides)
    return OIDCConfig(**defaults)


# =============================================================================
# 1. OIDCConfig Model Tests
# =============================================================================


class TestOIDCConfig:
    """Tests for OIDCConfig Pydantic model validation."""

    def test_default_values(self):
        """All defaults match the documented values."""
        config = OIDCConfig()
        assert config.enabled is False
        assert config.issuer_url is None
        assert config.client_id is None
        assert config.client_secret is None
        assert config.user_id_claim == "sub"
        assert config.email_claim == "email"
        assert config.display_name_claim == "name"
        assert config.team_claim is None
        assert config.org_claim is None
        assert config.role_claim is None
        assert config.role_mapping == {}
        assert config.default_role == "internal_user"
        assert config.allowed_email_domains == []
        assert config.auto_provision_users is True
        assert config.session_ttl == 1800
        assert config.token_exchange_enabled is True
        assert config.max_key_ttl_days == 365
        assert config.default_key_ttl_days == 90

    def test_issuer_url_must_be_https(self):
        """Non-HTTPS, non-localhost issuer_url is rejected."""
        with pytest.raises(ValidationError, match="HTTPS"):
            OIDCConfig(issuer_url="http://evil.example.com/realms/main")

    def test_issuer_url_allows_https(self):
        config = OIDCConfig(issuer_url="https://auth.example.com/realms/main")
        assert config.issuer_url == "https://auth.example.com/realms/main"

    def test_issuer_url_allows_localhost_http(self):
        """HTTP is allowed for localhost development."""
        config = OIDCConfig(issuer_url="http://localhost:8080/realms/main")
        assert config.issuer_url == "http://localhost:8080/realms/main"

    def test_issuer_url_allows_127_0_0_1_http(self):
        config = OIDCConfig(issuer_url="http://127.0.0.1:8080/realms/main")
        assert config.issuer_url == "http://127.0.0.1:8080/realms/main"

    def test_issuer_url_strips_trailing_slash(self):
        config = OIDCConfig(issuer_url="https://auth.example.com/realms/main/")
        assert config.issuer_url == "https://auth.example.com/realms/main"

    def test_issuer_url_none_is_valid(self):
        config = OIDCConfig(issuer_url=None)
        assert config.issuer_url is None

    def test_session_ttl_minimum_bound(self):
        """session_ttl below 60 seconds is rejected."""
        with pytest.raises(ValidationError, match="at least 60"):
            OIDCConfig(session_ttl=30)

    def test_session_ttl_maximum_bound(self):
        """session_ttl above 86400 seconds is rejected."""
        with pytest.raises(ValidationError, match="86400"):
            OIDCConfig(session_ttl=100_000)

    def test_session_ttl_boundary_values(self):
        assert OIDCConfig(session_ttl=60).session_ttl == 60
        assert OIDCConfig(session_ttl=86400).session_ttl == 86400

    def test_max_key_ttl_days_minimum_bound(self):
        with pytest.raises(ValidationError, match="at least 1"):
            OIDCConfig(max_key_ttl_days=0)

    def test_max_key_ttl_days_maximum_bound(self):
        with pytest.raises(ValidationError, match="3650"):
            OIDCConfig(max_key_ttl_days=5000)

    def test_max_key_ttl_days_boundary_values(self):
        assert OIDCConfig(max_key_ttl_days=1).max_key_ttl_days == 1
        assert OIDCConfig(max_key_ttl_days=3650).max_key_ttl_days == 3650

    def test_role_mapping_parsed_correctly(self):
        mapping = {"admin": ["proxy_admin"], "dev": ["internal_user", "proxy_viewer"]}
        config = OIDCConfig(role_mapping=mapping)
        assert config.role_mapping == mapping


# =============================================================================
# 2. OIDCDiscovery Tests
# =============================================================================


class TestOIDCDiscovery:
    """Tests for OIDC discovery and JWKS fetching."""

    @pytest.mark.asyncio
    async def test_discover_fetches_and_caches(self):
        """discover() fetches .well-known and caches the result."""
        discovery = OIDCDiscovery(cache_ttl=300)
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_DISCOVERY
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            # First call — fetches
            metadata = await discovery.discover("https://auth.example.com/realms/main")
            assert metadata.issuer == "https://auth.example.com/realms/main"
            assert metadata.jwks_uri == SAMPLE_DISCOVERY["jwks_uri"]
            assert mock_client.get.call_count == 1

            # Second call — from cache (no HTTP)
            metadata2 = await discovery.discover("https://auth.example.com/realms/main")
            assert metadata2.issuer == metadata.issuer
            assert mock_client.get.call_count == 1  # no additional call

    @pytest.mark.asyncio
    async def test_discover_handles_http_error(self):
        """discover() raises OIDCDiscoveryError on HTTP errors."""
        discovery = OIDCDiscovery(cache_ttl=300)

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error", request=MagicMock(), response=mock_response
        )

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with pytest.raises(OIDCDiscoveryError, match="HTTP 500"):
                await discovery.discover("https://auth.example.com/realms/main")

    @pytest.mark.asyncio
    async def test_discover_handles_network_error(self):
        """discover() raises OIDCDiscoveryError on network errors."""
        discovery = OIDCDiscovery(cache_ttl=300)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(
                side_effect=httpx.ConnectError("Connection refused")
            )
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with pytest.raises(OIDCDiscoveryError, match="network error"):
                await discovery.discover("https://auth.example.com/realms/main")

    @pytest.mark.asyncio
    async def test_discover_respects_cache_ttl(self):
        """discover() refetches after cache TTL expires."""
        discovery = OIDCDiscovery(cache_ttl=10)
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_DISCOVERY
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            # First call
            await discovery.discover("https://auth.example.com/realms/main")
            assert mock_client.get.call_count == 1

            # Expire the cache by manipulating the stored timestamp
            issuer = "https://auth.example.com/realms/main"
            meta, _ = discovery._metadata_cache[issuer]
            discovery._metadata_cache[issuer] = (meta, time.monotonic() - 20)

            # Second call — refetches due to expired TTL
            await discovery.discover(issuer)
            assert mock_client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_get_jwks_fetches_and_caches(self):
        """get_jwks() fetches JWKS and caches the result."""
        discovery = OIDCDiscovery(cache_ttl=300)
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_JWKS
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            jwks_uri = "https://auth.example.com/certs"
            jwks = await discovery.get_jwks(jwks_uri)
            assert "keys" in jwks
            assert mock_client.get.call_count == 1

            # Cached — no extra request
            await discovery.get_jwks(jwks_uri)
            assert mock_client.get.call_count == 1

    @pytest.mark.asyncio
    async def test_get_jwks_force_refresh(self):
        """get_jwks(force_refresh=True) bypasses cache."""
        discovery = OIDCDiscovery(cache_ttl=300)
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_JWKS
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            jwks_uri = "https://auth.example.com/certs"
            await discovery.get_jwks(jwks_uri)
            await discovery.get_jwks(jwks_uri, force_refresh=True)
            assert mock_client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_get_jwks_handles_http_error(self):
        """get_jwks() raises OIDCDiscoveryError on HTTP errors."""
        discovery = OIDCDiscovery(cache_ttl=300)
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found", request=MagicMock(), response=mock_response
        )

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with pytest.raises(OIDCDiscoveryError, match="HTTP 404"):
                await discovery.get_jwks("https://auth.example.com/certs")

    @pytest.mark.asyncio
    async def test_get_jwks_handles_invalid_json(self):
        """get_jwks() raises OIDCDiscoveryError on malformed JSON."""
        discovery = OIDCDiscovery(cache_ttl=300)
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.side_effect = json.JSONDecodeError("bad", "", 0)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with pytest.raises(OIDCDiscoveryError, match="not valid JSON"):
                await discovery.get_jwks("https://auth.example.com/certs")

    def test_invalidate_clears_caches(self):
        """invalidate() empties both caches."""
        discovery = OIDCDiscovery()
        metadata = OIDCProviderMetadata(**SAMPLE_DISCOVERY)
        discovery._metadata_cache["test"] = (metadata, time.monotonic())
        discovery._jwks_cache["test"] = (SAMPLE_JWKS, time.monotonic())

        discovery.invalidate()
        assert len(discovery._metadata_cache) == 0
        assert len(discovery._jwks_cache) == 0


# =============================================================================
# 3. JWTValidator Tests
# =============================================================================


class TestJWTValidatorExtractClaims:
    """Tests for JWTValidator.extract_claims() and role mapping."""

    def _validator(self, config: OIDCConfig | None = None) -> JWTValidator:
        if config is None:
            config = _make_config()
        return JWTValidator(config, OIDCDiscovery())

    def test_extract_claims_basic(self):
        """Standard claims are mapped correctly."""
        config = _make_config()
        v = self._validator(config)
        token_data = {
            "sub": "user-123",
            "email": "alice@example.com",
            "name": "Alice Smith",
        }
        identity = v.extract_claims(token_data, config)
        assert identity.user_id == "user-123"
        assert identity.email == "alice@example.com"
        assert identity.display_name == "Alice Smith"
        assert identity.roles == ["internal_user"]

    def test_extract_claims_custom_claim_mapping(self):
        """Custom claim mapping extracts from different fields."""
        config = _make_config(
            user_id_claim="preferred_username",
            email_claim="corporate_email",
            display_name_claim="full_name",
        )
        v = self._validator(config)
        token_data = {
            "preferred_username": "asmith",
            "corporate_email": "alice@corp.com",
            "full_name": "Alice Smith",
        }
        identity = v.extract_claims(token_data, config)
        assert identity.user_id == "asmith"
        assert identity.email == "alice@corp.com"
        assert identity.display_name == "Alice Smith"

    def test_extract_claims_missing_user_id_raises(self):
        """Missing required user_id claim raises OIDCAuthError."""
        config = _make_config()
        v = self._validator(config)
        with pytest.raises(OIDCAuthError, match="'sub' missing"):
            v.extract_claims({"email": "a@b.com"}, config)

    def test_extract_claims_missing_email_raises(self):
        """Missing required email claim raises OIDCAuthError."""
        config = _make_config()
        v = self._validator(config)
        with pytest.raises(OIDCAuthError, match="'email' missing"):
            v.extract_claims({"sub": "user-1"}, config)

    def test_extract_claims_missing_optional_display_name(self):
        """Missing display_name is None, not an error."""
        config = _make_config()
        v = self._validator(config)
        token_data = {"sub": "user-1", "email": "a@b.com"}
        identity = v.extract_claims(token_data, config)
        assert identity.display_name is None

    def test_extract_claims_team_as_string(self):
        """Single-value team claim is wrapped in a list."""
        config = _make_config(team_claim="team")
        v = self._validator(config)
        token_data = {"sub": "u1", "email": "a@b.com", "team": "engineering"}
        identity = v.extract_claims(token_data, config)
        assert identity.team_ids == ["engineering"]
        assert identity.team_id == "engineering"

    def test_extract_claims_team_as_list(self):
        """Multi-value team claim is preserved as a list."""
        config = _make_config(team_claim="teams")
        v = self._validator(config)
        token_data = {"sub": "u1", "email": "a@b.com", "teams": ["eng", "platform"]}
        identity = v.extract_claims(token_data, config)
        assert identity.team_ids == ["eng", "platform"]
        assert identity.team_id == "eng"

    def test_extract_claims_org_id(self):
        config = _make_config(org_claim="org")
        v = self._validator(config)
        token_data = {"sub": "u1", "email": "a@b.com", "org": "acme-corp"}
        identity = v.extract_claims(token_data, config)
        assert identity.org_id == "acme-corp"

    def test_extract_claims_email_domain_allowed(self):
        """Email in allowed domain list passes."""
        config = _make_config(allowed_email_domains=["example.com", "corp.com"])
        v = self._validator(config)
        token_data = {"sub": "u1", "email": "alice@example.com"}
        identity = v.extract_claims(token_data, config)
        assert identity.email == "alice@example.com"

    def test_extract_claims_email_domain_rejected(self):
        """Email outside allowed domain list raises OIDCAuthError."""
        config = _make_config(allowed_email_domains=["corp.com"])
        v = self._validator(config)
        token_data = {"sub": "u1", "email": "alice@evil.com"}
        with pytest.raises(OIDCAuthError, match="not allowed"):
            v.extract_claims(token_data, config)

    def test_extract_claims_email_domain_case_insensitive(self):
        """Email domain check is case-insensitive."""
        config = _make_config(allowed_email_domains=["EXAMPLE.COM"])
        v = self._validator(config)
        token_data = {"sub": "u1", "email": "alice@example.com"}
        identity = v.extract_claims(token_data, config)
        assert identity.email == "alice@example.com"

    def test_extract_claims_raw_claims_preserved(self):
        """raw_claims includes the full token payload."""
        config = _make_config()
        v = self._validator(config)
        token_data = {"sub": "u1", "email": "a@b.com", "custom": "value"}
        identity = v.extract_claims(token_data, config)
        assert identity.raw_claims == token_data


class TestRoleMapping:
    """Tests for JWTValidator._resolve_roles() static method."""

    def test_no_role_claim_returns_default(self):
        """When role_claim is not configured, default_role is used."""
        config = _make_config(role_claim=None)
        roles = JWTValidator._resolve_roles({"sub": "u1"}, config)
        assert roles == ["internal_user"]

    def test_role_mapping_maps_roles(self):
        """IdP roles are mapped to RouteIQ roles via role_mapping."""
        config = _make_config(
            role_claim="roles",
            role_mapping={"admin": ["proxy_admin"], "viewer": ["proxy_viewer"]},
        )
        token_data = {"roles": ["admin", "viewer"]}
        roles = JWTValidator._resolve_roles(token_data, config)
        assert roles == ["proxy_admin", "proxy_viewer"]

    def test_role_mapping_deduplicates(self):
        """Duplicate roles after mapping are removed, order preserved."""
        config = _make_config(
            role_claim="roles",
            role_mapping={
                "admin": ["proxy_admin", "internal_user"],
                "dev": ["internal_user"],
            },
        )
        token_data = {"roles": ["admin", "dev"]}
        roles = JWTValidator._resolve_roles(token_data, config)
        assert roles == ["proxy_admin", "internal_user"]

    def test_no_matching_mapping_returns_default(self):
        """When no role_mapping matches, default_role is used."""
        config = _make_config(
            role_claim="roles",
            role_mapping={"admin": ["proxy_admin"]},
            default_role="viewer",
        )
        token_data = {"roles": ["unknown-role"]}
        roles = JWTValidator._resolve_roles(token_data, config)
        assert roles == ["viewer"]

    def test_single_role_as_string(self):
        """Single role string (not list) is handled."""
        config = _make_config(
            role_claim="role",
            role_mapping={"admin": ["proxy_admin"]},
        )
        token_data = {"role": "admin"}
        roles = JWTValidator._resolve_roles(token_data, config)
        assert roles == ["proxy_admin"]

    def test_empty_roles_list_returns_default(self):
        """Empty roles list returns default_role."""
        config = _make_config(role_claim="roles")
        token_data = {"roles": []}
        roles = JWTValidator._resolve_roles(token_data, config)
        assert roles == ["internal_user"]

    def test_missing_role_claim_returns_default(self):
        """When the role claim is absent from the token, default_role is used."""
        config = _make_config(role_claim="roles")
        token_data = {}
        roles = JWTValidator._resolve_roles(token_data, config)
        assert roles == ["internal_user"]


class TestJWTValidatorValidateToken:
    """Tests for JWTValidator.validate_token() — mocking authlib."""

    @pytest.mark.asyncio
    async def test_validate_token_rejects_expired(self):
        """Expired token raises OIDCAuthError."""
        config = _make_config()
        discovery = OIDCDiscovery()
        validator = JWTValidator(config, discovery)

        mock_metadata = OIDCProviderMetadata(**SAMPLE_DISCOVERY)
        discovery.discover = AsyncMock(return_value=mock_metadata)
        discovery.get_jwks = AsyncMock(return_value=SAMPLE_JWKS)

        with (
            patch("litellm_llmrouter.oidc.JsonWebToken") as mock_jwt_cls,
            patch("litellm_llmrouter.oidc.JsonWebKey") as mock_jwk,
        ):
            from litellm_llmrouter.oidc import ExpiredTokenError as ETE

            mock_jwk.import_key_set.return_value = MagicMock()
            mock_jwt = MagicMock()
            mock_jwt_cls.return_value = mock_jwt
            mock_token_data = MagicMock()
            mock_jwt.decode.return_value = mock_token_data
            mock_token_data.validate.side_effect = ETE("Token is expired")

            with pytest.raises(OIDCAuthError, match="expired"):
                await validator.validate_token("expired.jwt.token")

    @pytest.mark.asyncio
    async def test_validate_token_rejects_bad_signature(self):
        """Invalid signature raises OIDCAuthError after JWKS refresh attempt."""
        config = _make_config()
        discovery = OIDCDiscovery()
        validator = JWTValidator(config, discovery)

        mock_metadata = OIDCProviderMetadata(**SAMPLE_DISCOVERY)
        discovery.discover = AsyncMock(return_value=mock_metadata)
        discovery.get_jwks = AsyncMock(return_value=SAMPLE_JWKS)

        with (
            patch("litellm_llmrouter.oidc.JsonWebToken") as mock_jwt_cls,
            patch("litellm_llmrouter.oidc.JsonWebKey") as mock_jwk,
        ):
            from litellm_llmrouter.oidc import BadSignatureError as BSE

            mock_jwk.import_key_set.return_value = MagicMock()
            mock_jwt = MagicMock()
            mock_jwt_cls.return_value = mock_jwt
            # First decode raises BadSignatureError, retry also fails
            mock_jwt.decode.side_effect = BSE("bad sig")

            with pytest.raises(OIDCAuthError, match="Invalid token signature"):
                await validator.validate_token("bad.jwt.token")

            # Verify JWKS refresh was attempted
            assert discovery.get_jwks.call_count == 2  # initial + force_refresh

    @pytest.mark.asyncio
    async def test_validate_token_rejects_wrong_issuer(self):
        """Wrong issuer in claims_options causes InvalidClaimError."""
        config = _make_config()
        discovery = OIDCDiscovery()
        validator = JWTValidator(config, discovery)

        mock_metadata = OIDCProviderMetadata(**SAMPLE_DISCOVERY)
        discovery.discover = AsyncMock(return_value=mock_metadata)
        discovery.get_jwks = AsyncMock(return_value=SAMPLE_JWKS)

        with (
            patch("litellm_llmrouter.oidc.JsonWebToken") as mock_jwt_cls,
            patch("litellm_llmrouter.oidc.JsonWebKey") as mock_jwk,
        ):
            from litellm_llmrouter.oidc import InvalidClaimError as ICE

            mock_jwk.import_key_set.return_value = MagicMock()
            mock_jwt = MagicMock()
            mock_jwt_cls.return_value = mock_jwt
            mock_token_data = MagicMock()
            mock_jwt.decode.return_value = mock_token_data
            mock_token_data.validate.side_effect = ICE("iss")

            with pytest.raises(OIDCAuthError, match="validation failed"):
                await validator.validate_token("wrong.iss.token")

    @pytest.mark.asyncio
    async def test_validate_token_no_issuer_url_configured(self):
        """Missing issuer_url in config raises OIDCAuthError."""
        config = _make_config(issuer_url=None)
        discovery = OIDCDiscovery()
        validator = JWTValidator(config, discovery)

        with pytest.raises(OIDCAuthError, match="issuer_url is not configured"):
            await validator.validate_token("any.jwt.token")

    @pytest.mark.asyncio
    async def test_validate_token_discovery_failure(self):
        """Discovery failure is wrapped in OIDCAuthError."""
        config = _make_config()
        discovery = OIDCDiscovery()
        discovery.discover = AsyncMock(
            side_effect=OIDCDiscoveryError("Connection refused")
        )
        validator = JWTValidator(config, discovery)

        with pytest.raises(OIDCAuthError, match="Provider discovery failed"):
            await validator.validate_token("any.jwt.token")

    @pytest.mark.asyncio
    async def test_validate_token_success(self):
        """Successful validation returns OIDCIdentity."""
        config = _make_config()
        discovery = OIDCDiscovery()
        validator = JWTValidator(config, discovery)

        mock_metadata = OIDCProviderMetadata(**SAMPLE_DISCOVERY)
        discovery.discover = AsyncMock(return_value=mock_metadata)
        discovery.get_jwks = AsyncMock(return_value=SAMPLE_JWKS)

        token_data_dict = {
            "sub": "user-42",
            "email": "bob@example.com",
            "name": "Bob",
            "iss": "https://auth.example.com/realms/main",
            "aud": "routeiq-gateway",
            "exp": int(time.time()) + 3600,
        }

        with (
            patch("litellm_llmrouter.oidc.JsonWebToken") as mock_jwt_cls,
            patch("litellm_llmrouter.oidc.JsonWebKey") as mock_jwk,
        ):
            mock_jwk.import_key_set.return_value = MagicMock()
            mock_jwt = MagicMock()
            mock_jwt_cls.return_value = mock_jwt
            # Make the decoded object behave like a dict
            mock_token_data = MagicMock()
            mock_jwt.decode.return_value = mock_token_data
            mock_token_data.validate.return_value = None
            mock_token_data.__iter__ = MagicMock(
                return_value=iter(token_data_dict.keys())
            )
            mock_token_data.__getitem__ = lambda s, k: token_data_dict[k]
            mock_token_data.keys = lambda: token_data_dict.keys()

            # Patch dict() conversion
            with patch.object(validator, "extract_claims") as mock_extract:
                expected_identity = OIDCIdentity(
                    user_id="user-42",
                    email="bob@example.com",
                    display_name="Bob",
                    roles=["internal_user"],
                )
                mock_extract.return_value = expected_identity

                identity = await validator.validate_token("good.jwt.token")
                assert identity.user_id == "user-42"
                assert identity.email == "bob@example.com"


# =============================================================================
# 4. OIDC Router Endpoint Tests
# =============================================================================


class TestOIDCRouterEndpoints:
    """Tests for the FastAPI router endpoints created by create_oidc_router()."""

    def _make_app(self, config: OIDCConfig | None = None) -> FastAPI:
        if config is None:
            config = _make_config()
        app = FastAPI()
        router = create_oidc_router(config)
        app.include_router(router)
        return app

    def test_sso_login_redirects_to_idp(self):
        """GET /sso/login returns a 302 redirect to the IdP."""
        config = _make_config()
        app = self._make_app(config)

        with patch.object(
            OIDCDiscovery, "discover", new_callable=AsyncMock
        ) as mock_discover:
            mock_discover.return_value = OIDCProviderMetadata(**SAMPLE_DISCOVERY)
            client = TestClient(app, raise_server_exceptions=False)
            response = client.get("/sso/login", follow_redirects=False)
            assert response.status_code == 302
            location = response.headers["location"]
            assert "auth.example.com" in location
            assert "response_type=code" in location
            assert "client_id=routeiq-gateway" in location
            assert "state=" in location

    def test_sso_login_missing_config(self):
        """GET /sso/login returns 500 when config is incomplete."""
        config = _make_config(issuer_url=None, client_id=None)
        app = self._make_app(config)
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/sso/login")
        assert response.status_code == 500

    def test_sso_callback_rejects_invalid_state(self):
        """GET /sso/callback rejects unknown state (CSRF protection)."""
        app = self._make_app()
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get(
            "/sso/callback",
            params={"code": "auth-code-123", "state": "invalid-state"},
        )
        assert response.status_code == 400
        assert "state" in response.json()["detail"].lower()

    def test_sso_callback_missing_code(self):
        """GET /sso/callback rejects missing authorization code."""
        app = self._make_app()
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/sso/callback", params={"state": "some-state"})
        assert response.status_code == 400
        assert "code" in response.json()["detail"].lower()

    def test_sso_callback_idp_error(self):
        """GET /sso/callback handles IdP-reported errors."""
        app = self._make_app()
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get(
            "/sso/callback",
            params={"error": "access_denied", "error_description": "User cancelled"},
        )
        assert response.status_code == 400
        assert "access_denied" in response.json()["detail"]

    def test_token_exchange_requires_bearer(self):
        """POST /auth/token-exchange rejects missing Bearer token."""
        app = self._make_app()
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post("/auth/token-exchange", json={})
        assert response.status_code == 401
        assert "Bearer" in response.json()["detail"]

    def test_token_exchange_disabled(self):
        """POST /auth/token-exchange returns 403 when exchange is disabled."""
        config = _make_config(token_exchange_enabled=False)
        app = self._make_app(config)
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/auth/token-exchange",
            json={},
            headers={"Authorization": "Bearer some-jwt"},
        )
        assert response.status_code == 403

    def test_token_exchange_with_invalid_jwt(self):
        """POST /auth/token-exchange with invalid JWT returns 401."""
        config = _make_config()
        app = self._make_app(config)

        with patch.object(
            JWTValidator, "validate_token", new_callable=AsyncMock
        ) as mock_validate:
            mock_validate.side_effect = OIDCAuthError("Token validation failed")
            client = TestClient(app, raise_server_exceptions=False)
            response = client.post(
                "/auth/token-exchange",
                json={},
                headers={"Authorization": "Bearer invalid-token"},
            )
            assert response.status_code == 401

    def test_token_exchange_success(self):
        """POST /auth/token-exchange returns API key on success."""
        config = _make_config()
        app = self._make_app(config)

        identity = OIDCIdentity(
            user_id="user-42",
            email="alice@example.com",
            roles=["internal_user"],
        )

        with patch.object(
            JWTValidator, "validate_token", new_callable=AsyncMock
        ) as mock_validate:
            mock_validate.return_value = identity
            client = TestClient(app, raise_server_exceptions=False)
            response = client.post(
                "/auth/token-exchange",
                json={"ttl_days": 30, "key_alias": "my-key"},
                headers={"Authorization": "Bearer valid-token"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["api_key"].startswith("sk-oidc-")
            assert data["user_id"] == "user-42"
            assert data["key_alias"] == "my-key"
            assert data["expires_at"] > int(time.time())

    def test_token_exchange_clamps_ttl(self):
        """Token exchange clamps ttl_days to max_key_ttl_days."""
        config = _make_config(max_key_ttl_days=7)
        app = self._make_app(config)

        identity = OIDCIdentity(user_id="u1", email="a@b.com", roles=["internal_user"])

        with patch.object(
            JWTValidator, "validate_token", new_callable=AsyncMock
        ) as mock_validate:
            mock_validate.return_value = identity
            client = TestClient(app, raise_server_exceptions=False)
            response = client.post(
                "/auth/token-exchange",
                json={"ttl_days": 365},
                headers={"Authorization": "Bearer valid-token"},
            )
            assert response.status_code == 200
            data = response.json()
            # Max TTL is 7 days, so expires_at should be ~7 days from now
            max_expected = int(time.time()) + (7 * 86400) + 60
            assert data["expires_at"] <= max_expected

    def test_userinfo_requires_bearer(self):
        """GET /auth/userinfo rejects missing Bearer token."""
        app = self._make_app()
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/auth/userinfo")
        assert response.status_code == 401

    def test_userinfo_returns_identity(self):
        """GET /auth/userinfo returns user identity for valid JWT."""
        config = _make_config()
        app = self._make_app(config)

        identity = OIDCIdentity(
            user_id="user-42",
            email="alice@example.com",
            display_name="Alice",
            roles=["internal_user"],
        )

        with patch.object(
            JWTValidator, "validate_token", new_callable=AsyncMock
        ) as mock_validate:
            mock_validate.return_value = identity
            client = TestClient(app, raise_server_exceptions=False)
            response = client.get(
                "/auth/userinfo",
                headers={"Authorization": "Bearer valid-token"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["user_id"] == "user-42"
            assert data["email"] == "alice@example.com"
            assert "raw_claims" not in data  # excluded from response

    def test_userinfo_uses_identity_cache(self):
        """GET /auth/userinfo uses cached identity on second call."""
        config = _make_config()
        app = self._make_app(config)

        identity = OIDCIdentity(
            user_id="user-42",
            email="alice@example.com",
            roles=["internal_user"],
        )

        with patch.object(
            JWTValidator, "validate_token", new_callable=AsyncMock
        ) as mock_validate:
            mock_validate.return_value = identity
            client = TestClient(app, raise_server_exceptions=False)

            # First call — validates token
            resp1 = client.get(
                "/auth/userinfo",
                headers={"Authorization": "Bearer cached-token"},
            )
            assert resp1.status_code == 200
            assert mock_validate.call_count == 1

            # Second call — uses cache
            resp2 = client.get(
                "/auth/userinfo",
                headers={"Authorization": "Bearer cached-token"},
            )
            assert resp2.status_code == 200
            assert mock_validate.call_count == 1  # NOT called again


# =============================================================================
# 5. resolve_identity() Tests
# =============================================================================


class TestResolveIdentity:
    """Tests for the unified resolve_identity() function."""

    @pytest.mark.asyncio
    async def test_returns_none_when_disabled(self):
        """Returns None when OIDC is disabled."""
        setup_oidc(OIDCConfig(enabled=False))
        request = MagicMock(spec=Request)
        request.headers = {}
        result = await resolve_identity(request)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_not_setup(self):
        """Returns None when setup_oidc() hasn't been called."""
        # _oidc_config is None after reset_oidc()
        request = MagicMock(spec=Request)
        request.headers = {}
        result = await resolve_identity(request)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_auth_header(self):
        """Returns None when no Authorization or x-api-key header."""
        setup_oidc(_make_config())
        request = MagicMock(spec=Request)
        request.headers = {}
        result = await resolve_identity(request)
        assert result is None

    @pytest.mark.asyncio
    async def test_resolves_exchanged_api_key(self):
        """resolve_identity() finds a valid exchanged OIDC API key."""
        config = _make_config()
        setup_oidc(config)

        identity = OIDCIdentity(user_id="u1", email="a@b.com", roles=["internal_user"])
        api_key = "sk-oidc-test123456"
        key_hash = _hash_token(api_key)
        _exchanged_keys[key_hash] = (identity, int(time.time()) + 3600)

        request = MagicMock(spec=Request)
        request.headers = {"x-api-key": api_key}
        result = await resolve_identity(request)
        assert result is not None
        assert result.user_id == "u1"

    @pytest.mark.asyncio
    async def test_rejects_expired_api_key(self):
        """resolve_identity() returns None for expired OIDC API key."""
        config = _make_config()
        setup_oidc(config)

        identity = OIDCIdentity(user_id="u1", email="a@b.com", roles=["internal_user"])
        api_key = "sk-oidc-expired"
        key_hash = _hash_token(api_key)
        _exchanged_keys[key_hash] = (identity, int(time.time()) - 100)

        request = MagicMock(spec=Request)
        request.headers = {"x-api-key": api_key}
        result = await resolve_identity(request)
        assert result is None
        # Also cleans up expired key
        assert key_hash not in _exchanged_keys

    @pytest.mark.asyncio
    async def test_resolves_jwt_via_validator(self):
        """resolve_identity() validates JWT via JWTValidator."""
        config = _make_config()
        setup_oidc(config)

        identity = OIDCIdentity(
            user_id="jwt-user", email="jwt@example.com", roles=["internal_user"]
        )

        import litellm_llmrouter.oidc as oidc_mod

        mock_validator = MagicMock()
        mock_validator.validate_token = AsyncMock(return_value=identity)
        oidc_mod._jwt_validator = mock_validator

        request = MagicMock(spec=Request)
        request.headers = {"Authorization": "Bearer my.jwt.token"}
        result = await resolve_identity(request)
        assert result is not None
        assert result.user_id == "jwt-user"
        mock_validator.validate_token.assert_called_once_with("my.jwt.token")

    @pytest.mark.asyncio
    async def test_returns_none_for_invalid_jwt(self):
        """resolve_identity() returns None (no crash) on invalid JWT."""
        config = _make_config()
        setup_oidc(config)

        import litellm_llmrouter.oidc as oidc_mod

        mock_validator = MagicMock()
        mock_validator.validate_token = AsyncMock(
            side_effect=OIDCAuthError("bad token")
        )
        oidc_mod._jwt_validator = mock_validator

        request = MagicMock(spec=Request)
        request.headers = {"Authorization": "Bearer invalid.jwt"}
        result = await resolve_identity(request)
        assert result is None

    @pytest.mark.asyncio
    async def test_uses_identity_cache(self):
        """resolve_identity() uses cached identity instead of re-validating."""
        config = _make_config()
        setup_oidc(config)

        identity = OIDCIdentity(
            user_id="cached", email="c@b.com", roles=["internal_user"]
        )
        token = "my.cached.jwt"
        token_hash = _hash_token(token)
        _identity_cache[token_hash] = (identity, time.monotonic() + 600)

        import litellm_llmrouter.oidc as oidc_mod

        mock_validator = MagicMock()
        mock_validator.validate_token = AsyncMock()
        oidc_mod._jwt_validator = mock_validator

        request = MagicMock(spec=Request)
        request.headers = {"Authorization": f"Bearer {token}"}
        result = await resolve_identity(request)
        assert result is not None
        assert result.user_id == "cached"
        mock_validator.validate_token.assert_not_called()


# =============================================================================
# 6. Singleton Lifecycle Tests
# =============================================================================


class TestSingletonLifecycle:
    """Tests for setup_oidc, reset_oidc, get_oidc_config."""

    def test_setup_oidc_disabled(self):
        """setup_oidc() with disabled config sets _oidc_config but not validator."""
        import litellm_llmrouter.oidc as oidc_mod

        config = setup_oidc(OIDCConfig(enabled=False))
        assert config is not None
        assert config.enabled is False
        assert oidc_mod._oidc_config is not None
        assert oidc_mod._jwt_validator is None

    def test_setup_oidc_enabled(self):
        """setup_oidc() with enabled config creates validator."""
        import litellm_llmrouter.oidc as oidc_mod

        config = setup_oidc(_make_config())
        assert config is not None
        assert config.enabled is True
        assert oidc_mod._jwt_validator is not None

    def test_setup_oidc_missing_issuer_url(self):
        """setup_oidc() with enabled but no issuer sets config, no validator."""
        import litellm_llmrouter.oidc as oidc_mod

        config = setup_oidc(OIDCConfig(enabled=True, issuer_url=None))
        assert config.enabled is True
        assert oidc_mod._jwt_validator is None

    def test_setup_oidc_loads_from_env(self, monkeypatch):
        """setup_oidc(None) loads from environment."""
        monkeypatch.setenv("ROUTEIQ_OIDC_ENABLED", "true")
        monkeypatch.setenv(
            "ROUTEIQ_OIDC_ISSUER_URL", "https://auth.example.com/realms/main"
        )
        monkeypatch.setenv("ROUTEIQ_OIDC_CLIENT_ID", "my-client")
        config = setup_oidc()
        assert config is not None
        assert config.enabled is True
        assert config.client_id == "my-client"

    def test_reset_oidc_clears_everything(self):
        """reset_oidc() clears singletons and all caches."""
        import litellm_llmrouter.oidc as oidc_mod

        setup_oidc(_make_config())
        _pending_auth_states["test"] = (time.time(), "url", "verifier")
        _exchanged_keys["hash"] = (
            OIDCIdentity(user_id="u", email="e@x.com"),
            int(time.time()),
        )
        _identity_cache["hash"] = (
            OIDCIdentity(user_id="u", email="e@x.com"),
            time.monotonic(),
        )

        reset_oidc()
        assert oidc_mod._oidc_config is None
        assert oidc_mod._jwt_validator is None
        assert len(_pending_auth_states) == 0
        assert len(_exchanged_keys) == 0
        assert len(_identity_cache) == 0

    def test_get_oidc_config_defaults_disabled(self):
        """get_oidc_config() with no env vars returns disabled config."""
        config = get_oidc_config()
        assert config.enabled is False
        assert config.issuer_url is None

    def test_get_oidc_config_from_env(self, monkeypatch):
        """get_oidc_config() reads all ROUTEIQ_OIDC_* env vars."""
        monkeypatch.setenv("ROUTEIQ_OIDC_ENABLED", "true")
        monkeypatch.setenv(
            "ROUTEIQ_OIDC_ISSUER_URL", "https://auth.example.com/realms/main"
        )
        monkeypatch.setenv("ROUTEIQ_OIDC_CLIENT_ID", "test-client")
        monkeypatch.setenv("ROUTEIQ_OIDC_CLIENT_SECRET", "test-secret")
        monkeypatch.setenv("ROUTEIQ_OIDC_USER_ID_CLAIM", "preferred_username")
        monkeypatch.setenv("ROUTEIQ_OIDC_DEFAULT_ROLE", "viewer")
        monkeypatch.setenv(
            "ROUTEIQ_OIDC_ALLOWED_EMAIL_DOMAINS", "example.com, corp.com"
        )
        monkeypatch.setenv("ROUTEIQ_OIDC_SESSION_TTL", "3600")
        monkeypatch.setenv("ROUTEIQ_OIDC_MAX_KEY_TTL_DAYS", "30")
        monkeypatch.setenv("ROUTEIQ_OIDC_DEFAULT_KEY_TTL_DAYS", "7")
        monkeypatch.setenv("ROUTEIQ_OIDC_AUTO_PROVISION_USERS", "false")
        monkeypatch.setenv("ROUTEIQ_OIDC_TOKEN_EXCHANGE_ENABLED", "false")

        config = get_oidc_config()
        assert config.enabled is True
        assert config.issuer_url == "https://auth.example.com/realms/main"
        assert config.client_id == "test-client"
        assert config.client_secret == "test-secret"
        assert config.user_id_claim == "preferred_username"
        assert config.default_role == "viewer"
        assert config.allowed_email_domains == ["example.com", "corp.com"]
        assert config.session_ttl == 3600
        assert config.max_key_ttl_days == 30
        assert config.default_key_ttl_days == 7
        assert config.auto_provision_users is False
        assert config.token_exchange_enabled is False

    def test_get_oidc_config_role_mapping_json(self, monkeypatch):
        """get_oidc_config() parses ROUTEIQ_OIDC_ROLE_MAPPING as JSON."""
        monkeypatch.setenv(
            "ROUTEIQ_OIDC_ROLE_MAPPING",
            '{"admin": ["proxy_admin"], "dev": "internal_user"}',
        )
        config = get_oidc_config()
        assert config.role_mapping == {
            "admin": ["proxy_admin"],
            "dev": ["internal_user"],  # string value wrapped in list
        }

    def test_get_oidc_config_invalid_role_mapping_json(self, monkeypatch):
        """get_oidc_config() handles invalid JSON in ROLE_MAPPING gracefully."""
        monkeypatch.setenv("ROUTEIQ_OIDC_ROLE_MAPPING", "not-valid-json")
        config = get_oidc_config()
        assert config.role_mapping == {}

    def test_get_oidc_config_non_dict_role_mapping(self, monkeypatch):
        """get_oidc_config() handles non-dict JSON in ROLE_MAPPING."""
        monkeypatch.setenv("ROUTEIQ_OIDC_ROLE_MAPPING", '["admin", "dev"]')
        config = get_oidc_config()
        assert config.role_mapping == {}

    def test_get_oidc_config_invalid_int_env(self, monkeypatch):
        """get_oidc_config() falls back to default on invalid int env vars."""
        monkeypatch.setenv("ROUTEIQ_OIDC_SESSION_TTL", "not-a-number")
        config = get_oidc_config()
        assert config.session_ttl == 1800  # default

    def test_get_oidc_config_bool_env_parsing(self, monkeypatch):
        """get_oidc_config() parses various bool formats."""
        for truthy in ("true", "1", "yes", "True", "YES"):
            monkeypatch.setenv("ROUTEIQ_OIDC_ENABLED", truthy)
            assert get_oidc_config().enabled is True

        for falsy in ("false", "0", "no", "anything-else"):
            monkeypatch.setenv("ROUTEIQ_OIDC_ENABLED", falsy)
            assert get_oidc_config().enabled is False


# =============================================================================
# 7. Internal Helper Tests
# =============================================================================


class TestInternalHelpers:
    """Tests for internal helper functions."""

    def test_generate_state_is_unique(self):
        """_generate_state() produces unique values."""
        states = {_generate_state() for _ in range(100)}
        assert len(states) == 100

    def test_hash_token_deterministic(self):
        """_hash_token() produces consistent SHA-256 hashes."""
        h1 = _hash_token("my-secret-token")
        h2 = _hash_token("my-secret-token")
        assert h1 == h2
        assert len(h1) == 64  # hex SHA-256

    def test_hash_token_different_inputs(self):
        assert _hash_token("token-a") != _hash_token("token-b")

    def test_prune_expired_states(self):
        """_prune_expired_states() removes old entries."""
        old_time = time.time() - 700
        _pending_auth_states["old"] = (old_time, "url", "verifier")
        _pending_auth_states["new"] = (time.time(), "url", "verifier")

        _prune_expired_states(max_age=600)
        assert "old" not in _pending_auth_states
        assert "new" in _pending_auth_states

    def test_prune_expired_identities(self):
        """_prune_expired_identities() removes expired entries."""
        identity = OIDCIdentity(user_id="u", email="e@x.com")
        _identity_cache["expired"] = (identity, time.monotonic() - 10)
        _identity_cache["valid"] = (identity, time.monotonic() + 600)

        _prune_expired_identities()
        assert "expired" not in _identity_cache
        assert "valid" in _identity_cache


# =============================================================================
# 8. Data Model Tests
# =============================================================================


class TestDataModels:
    """Tests for supporting Pydantic models."""

    def test_oidc_identity_defaults(self):
        identity = OIDCIdentity(user_id="u1", email="a@b.com")
        assert identity.display_name is None
        assert identity.team_id is None
        assert identity.team_ids == []
        assert identity.org_id is None
        assert identity.roles == []
        assert identity.raw_claims == {}

    def test_token_exchange_request_defaults(self):
        req = TokenExchangeRequest()
        assert req.ttl_days is None
        assert req.key_alias is None
        assert req.scopes == []

    def test_token_exchange_response(self):
        resp = TokenExchangeResponse(
            api_key="sk-oidc-abc",
            expires_at=1234567890,
            user_id="u1",
            key_alias="my-key",
        )
        assert resp.api_key == "sk-oidc-abc"
        assert resp.expires_at == 1234567890

    def test_oidc_provider_metadata_defaults(self):
        meta = OIDCProviderMetadata(
            issuer="https://x.com",
            authorization_endpoint="https://x.com/auth",
            token_endpoint="https://x.com/token",
            jwks_uri="https://x.com/certs",
        )
        assert meta.userinfo_endpoint is None
        assert meta.end_session_endpoint is None
        assert "openid" in meta.scopes_supported


# =============================================================================
# 9. Error Class Tests
# =============================================================================


class TestErrors:
    """Tests for OIDC error hierarchy."""

    def test_oidc_error_base(self):
        from litellm_llmrouter.oidc import OIDCError

        err = OIDCError("base error")
        assert str(err) == "base error"
        assert isinstance(err, Exception)

    def test_discovery_error_inherits(self):
        from litellm_llmrouter.oidc import OIDCError

        err = OIDCDiscoveryError("disc failed")
        assert isinstance(err, OIDCError)

    def test_auth_error_inherits(self):
        from litellm_llmrouter.oidc import OIDCError

        err = OIDCAuthError("auth failed")
        assert isinstance(err, OIDCError)


# =============================================================================
# 10. Authlib Not Available Tests
# =============================================================================


class TestAuthLibNotAvailable:
    """Tests for behaviour when authlib is not installed."""

    def test_router_returns_501_when_authlib_missing(self):
        """When authlib_available is False, endpoints return 501."""
        config = _make_config()

        with patch("litellm_llmrouter.oidc.authlib_available", False):
            app = FastAPI()
            router = create_oidc_router(config)
            app.include_router(router)
            client = TestClient(app, raise_server_exceptions=False)

            for path in ["/sso/login", "/sso/callback", "/auth/userinfo"]:
                resp = client.get(path)
                assert resp.status_code == 501
                assert "authlib" in resp.json()["message"]

            resp = client.post("/auth/token-exchange", json={})
            assert resp.status_code == 501
