"""
OIDC / SSO Identity Integration for RouteIQ Gateway
=====================================================

This module implements OpenID Connect (OIDC) and Single Sign-On (SSO) integration
for RouteIQ, enabling organizations to authenticate users via external identity
providers such as Keycloak, Auth0, Okta, Azure AD, and Google Workspace.

Features:
- OIDC Discovery: Auto-fetches provider metadata from .well-known/openid-configuration
- JWT Validation: Validates ID tokens and access tokens using provider JWKS
- Claim Mapping: Configurable extraction of user identity, team, org, and roles
- Role Mapping: Maps IdP roles/groups to RouteIQ internal roles
- Token Exchange: Exchanges OIDC tokens for RouteIQ API keys
- User Provisioning: Auto-creates users on first login (configurable)
- Email Domain Restriction: Allows only approved email domains

Usage:
    from litellm_llmrouter.oidc import (
        get_oidc_config,
        create_oidc_router,
        resolve_identity,
    )

    config = get_oidc_config()
    if config.enabled:
        router = create_oidc_router(config)
        app.include_router(router)

Configuration (Environment Variables):
    - ROUTEIQ_OIDC_ENABLED: Enable OIDC integration (default: false)
    - ROUTEIQ_OIDC_ISSUER_URL: OIDC issuer URL (e.g., https://auth.example.com/realms/main)
    - ROUTEIQ_OIDC_CLIENT_ID: OIDC client ID
    - ROUTEIQ_OIDC_CLIENT_SECRET: OIDC client secret
    - ROUTEIQ_OIDC_USER_ID_CLAIM: Claim for user ID (default: sub)
    - ROUTEIQ_OIDC_EMAIL_CLAIM: Claim for email (default: email)
    - ROUTEIQ_OIDC_DISPLAY_NAME_CLAIM: Claim for display name (default: name)
    - ROUTEIQ_OIDC_TEAM_CLAIM: Claim for team assignment
    - ROUTEIQ_OIDC_ORG_CLAIM: Claim for organization assignment
    - ROUTEIQ_OIDC_ROLE_CLAIM: Claim for role(s)
    - ROUTEIQ_OIDC_ROLE_MAPPING: JSON dict mapping IdP roles to RouteIQ roles
    - ROUTEIQ_OIDC_DEFAULT_ROLE: Default role for SSO users (default: internal_user)
    - ROUTEIQ_OIDC_ALLOWED_EMAIL_DOMAINS: Comma-separated allowed email domains
    - ROUTEIQ_OIDC_AUTO_PROVISION_USERS: Auto-create users on first login (default: true)
    - ROUTEIQ_OIDC_SESSION_TTL: Session TTL in seconds (default: 1800)
    - ROUTEIQ_OIDC_TOKEN_EXCHANGE_ENABLED: Enable token exchange (default: true)
    - ROUTEIQ_OIDC_MAX_KEY_TTL_DAYS: Max API key TTL in days (default: 365)
    - ROUTEIQ_OIDC_DEFAULT_KEY_TTL_DAYS: Default API key TTL in days (default: 90)

Dependencies:
    Requires the ``authlib`` package. Install via ``pip install routeiq[oidc]``.
    When authlib is not installed, OIDC endpoints return a 501 explaining the
    dependency requirement.

Note:
    Importing this module does NOT produce side effects. Configuration is loaded
    explicitly via ``get_oidc_config()`` and routes are created via
    ``create_oidc_router()``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
import threading
import time
from typing import Any, Optional
from urllib.parse import urlencode, urljoin

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("litellm_llmrouter.oidc")

# =============================================================================
# Optional dependency guard
# =============================================================================

try:
    from authlib.integrations.httpx_client import AsyncOAuth2Client
    from authlib.jose import JsonWebKey, JsonWebToken
    from authlib.jose.errors import (
        BadSignatureError,
        DecodeError,
        ExpiredTokenError,
        InvalidClaimError,
    )

    authlib_available = True
except ImportError:
    authlib_available = False
    # Provide type stubs so the module is importable without authlib.
    AsyncOAuth2Client = None  # type: ignore[assignment, misc]
    JsonWebKey = None  # type: ignore[assignment, misc]
    JsonWebToken = None  # type: ignore[assignment, misc]
    BadSignatureError = Exception  # type: ignore[assignment, misc]
    DecodeError = Exception  # type: ignore[assignment, misc]
    ExpiredTokenError = Exception  # type: ignore[assignment, misc]
    InvalidClaimError = Exception  # type: ignore[assignment, misc]


# =============================================================================
# Data Models
# =============================================================================


class OIDCConfig(BaseModel):
    """Configuration for OIDC / SSO integration.

    Controls all aspects of the OIDC authentication flow including provider
    connection details, claim mapping, role mapping, email domain restrictions,
    user provisioning behaviour, and token exchange settings.

    Attributes:
        enabled: Whether OIDC authentication is active.
        issuer_url: The OIDC issuer URL (must support .well-known/openid-configuration).
        client_id: OAuth2 client ID registered with the IdP.
        client_secret: OAuth2 client secret.
        user_id_claim: JWT claim containing the unique user identifier.
        email_claim: JWT claim containing the user's email address.
        display_name_claim: JWT claim containing the user's display name.
        team_claim: JWT claim containing the user's team (single value or list).
        org_claim: JWT claim containing the user's organization.
        role_claim: JWT claim containing the user's roles.
        role_mapping: Maps IdP role names to lists of RouteIQ internal roles.
        default_role: Role assigned when no role mapping matches.
        allowed_email_domains: When non-empty, only emails from these domains
            are accepted. Empty list means all domains are allowed.
        auto_provision_users: Whether to auto-create user records on first SSO login.
        session_ttl: How long (seconds) a validated identity is cached.
        token_exchange_enabled: Whether the ``/auth/token-exchange`` endpoint is active.
        max_key_ttl_days: Maximum allowed TTL for exchanged API keys.
        default_key_ttl_days: Default TTL for exchanged API keys when not specified.
    """

    enabled: bool = False
    issuer_url: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None

    # Claim mapping
    user_id_claim: str = "sub"
    email_claim: str = "email"
    display_name_claim: str = "name"
    team_claim: Optional[str] = None
    org_claim: Optional[str] = None
    role_claim: Optional[str] = None

    # Role mapping: IdP role -> list of RouteIQ roles
    role_mapping: dict[str, list[str]] = Field(default_factory=dict)
    default_role: str = "internal_user"

    # Security constraints
    allowed_email_domains: list[str] = Field(default_factory=list)
    auto_provision_users: bool = True
    session_ttl: int = 1800  # 30 minutes

    # Token exchange settings
    token_exchange_enabled: bool = True
    max_key_ttl_days: int = 365
    default_key_ttl_days: int = 90

    @field_validator("issuer_url")
    @classmethod
    def _validate_issuer_url(cls, v: Optional[str]) -> Optional[str]:
        """Ensure issuer URL uses HTTPS (allow HTTP only for localhost dev)."""
        if v is None:
            return v
        v = v.rstrip("/")
        if not v.startswith(("https://", "http://localhost", "http://127.0.0.1")):
            raise ValueError(
                "OIDC issuer_url must use HTTPS (HTTP allowed only for localhost)"
            )
        return v

    @field_validator("session_ttl")
    @classmethod
    def _validate_session_ttl(cls, v: int) -> int:
        if v < 60:
            raise ValueError("session_ttl must be at least 60 seconds")
        if v > 86400:
            raise ValueError("session_ttl must not exceed 86400 seconds (24 hours)")
        return v

    @field_validator("max_key_ttl_days")
    @classmethod
    def _validate_max_key_ttl(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_key_ttl_days must be at least 1")
        if v > 3650:
            raise ValueError("max_key_ttl_days must not exceed 3650 (10 years)")
        return v


class OIDCProviderMetadata(BaseModel):
    """Cached OIDC provider metadata from .well-known/openid-configuration.

    Attributes:
        issuer: The provider's issuer identifier.
        authorization_endpoint: URL for the authorization flow.
        token_endpoint: URL for exchanging authorization codes.
        userinfo_endpoint: URL for retrieving user profile info.
        jwks_uri: URL for the provider's JSON Web Key Set.
        scopes_supported: Scopes the provider supports.
        response_types_supported: OAuth2 response types supported.
        end_session_endpoint: URL for RP-initiated logout (optional).
    """

    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    userinfo_endpoint: Optional[str] = None
    jwks_uri: str
    scopes_supported: list[str] = Field(default_factory=lambda: ["openid"])
    response_types_supported: list[str] = Field(default_factory=list)
    end_session_endpoint: Optional[str] = None


class OIDCIdentity(BaseModel):
    """Resolved identity from an OIDC token.

    Represents the normalized user identity extracted from JWT claims after
    validation and claim mapping. This is the canonical identity object used
    throughout the gateway for authorization decisions.

    Attributes:
        user_id: Unique user identifier (from ``sub`` claim by default).
        email: User's email address.
        display_name: Human-readable display name.
        team_id: Primary team assignment (first element of team_ids).
        team_ids: All team assignments (from team claim, may be list).
        org_id: Organization identifier.
        roles: Resolved RouteIQ roles (after role mapping).
        raw_claims: Complete set of original JWT claims for audit/debugging.
    """

    user_id: str
    email: str
    display_name: Optional[str] = None
    team_id: Optional[str] = None
    team_ids: list[str] = Field(default_factory=list)
    org_id: Optional[str] = None
    roles: list[str] = Field(default_factory=list)
    raw_claims: dict[str, Any] = Field(default_factory=dict)


class TokenExchangeRequest(BaseModel):
    """Request body for the token exchange endpoint.

    Attributes:
        ttl_days: Requested API key TTL in days. Clamped to max_key_ttl_days.
        key_alias: Optional human-readable alias for the API key.
        scopes: Optional list of scopes/permissions for the exchanged key.
    """

    ttl_days: Optional[int] = None
    key_alias: Optional[str] = None
    scopes: list[str] = Field(default_factory=list)


class TokenExchangeResponse(BaseModel):
    """Response from the token exchange endpoint.

    Attributes:
        api_key: The generated API key.
        expires_at: Unix timestamp when the key expires.
        user_id: The OIDC user ID the key is bound to.
        key_alias: The alias assigned to the key.
    """

    api_key: str
    expires_at: int
    user_id: str
    key_alias: Optional[str] = None


# =============================================================================
# OIDC Discovery
# =============================================================================


class OIDCDiscovery:
    """Fetches and caches OIDC provider metadata from .well-known/openid-configuration.

    The discovery document is cached for the configured session TTL to avoid
    hitting the provider on every request. JWKS keys are cached separately and
    refreshed when key ID lookup fails (key rotation support).

    Args:
        cache_ttl: How long (seconds) to cache discovery metadata and JWKS.

    Example::

        discovery = OIDCDiscovery(cache_ttl=3600)
        metadata = await discovery.discover("https://auth.example.com/realms/main")
        jwks = await discovery.get_jwks(metadata.jwks_uri)
    """

    def __init__(self, cache_ttl: int = 1800) -> None:
        self._cache_ttl = cache_ttl
        self._metadata_cache: dict[str, tuple[OIDCProviderMetadata, float]] = {}
        self._jwks_cache: dict[str, tuple[dict[str, Any], float]] = {}

    async def discover(self, issuer_url: str) -> OIDCProviderMetadata:
        """Fetch OIDC provider metadata, using cache when available.

        Args:
            issuer_url: The OIDC issuer URL. The ``.well-known/openid-configuration``
                path is appended automatically.

        Returns:
            Parsed provider metadata.

        Raises:
            OIDCDiscoveryError: If the discovery document cannot be fetched or parsed.
        """
        now = time.monotonic()
        cached = self._metadata_cache.get(issuer_url)
        if cached is not None:
            metadata, fetched_at = cached
            if now - fetched_at < self._cache_ttl:
                return metadata

        import httpx

        well_known_url = urljoin(
            issuer_url.rstrip("/") + "/",
            ".well-known/openid-configuration",
        )
        logger.info(
            "Fetching OIDC discovery document from %s",
            well_known_url,
        )

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(well_known_url)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "OIDC discovery failed: HTTP %d from %s",
                exc.response.status_code,
                well_known_url,
            )
            raise OIDCDiscoveryError(
                f"OIDC discovery returned HTTP {exc.response.status_code}"
            ) from exc
        except httpx.RequestError as exc:
            logger.error(
                "OIDC discovery failed: network error fetching %s: %s",
                well_known_url,
                exc,
            )
            raise OIDCDiscoveryError(f"OIDC discovery network error: {exc}") from exc
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error("OIDC discovery returned invalid JSON: %s", exc)
            raise OIDCDiscoveryError("OIDC discovery returned invalid JSON") from exc

        try:
            metadata = OIDCProviderMetadata(**data)
        except Exception as exc:
            logger.error("OIDC discovery metadata validation failed: %s", exc)
            raise OIDCDiscoveryError(f"OIDC discovery metadata invalid: {exc}") from exc

        self._metadata_cache[issuer_url] = (metadata, now)
        logger.info("OIDC discovery successful for issuer: %s", issuer_url)
        return metadata

    async def get_jwks(
        self, jwks_uri: str, *, force_refresh: bool = False
    ) -> dict[str, Any]:
        """Fetch the provider's JSON Web Key Set.

        Args:
            jwks_uri: The JWKS endpoint URL from the discovery document.
            force_refresh: When True, bypass the cache. Used when a key ID
                is not found in the cached keyset (key rotation scenario).

        Returns:
            The JWKS document as a dict.

        Raises:
            OIDCDiscoveryError: If the JWKS cannot be fetched or parsed.
        """
        now = time.monotonic()
        if not force_refresh:
            cached = self._jwks_cache.get(jwks_uri)
            if cached is not None:
                jwks, fetched_at = cached
                if now - fetched_at < self._cache_ttl:
                    return jwks

        import httpx

        logger.debug(
            "Fetching JWKS from %s (force_refresh=%s)", jwks_uri, force_refresh
        )

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(jwks_uri)
                response.raise_for_status()
                jwks = response.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "JWKS fetch failed: HTTP %d from %s",
                exc.response.status_code,
                jwks_uri,
            )
            raise OIDCDiscoveryError(
                f"JWKS fetch returned HTTP {exc.response.status_code}"
            ) from exc
        except httpx.RequestError as exc:
            logger.error("JWKS fetch failed: network error: %s", exc)
            raise OIDCDiscoveryError(f"JWKS fetch network error: {exc}") from exc
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error("JWKS response is not valid JSON: %s", exc)
            raise OIDCDiscoveryError("JWKS response is not valid JSON") from exc

        self._jwks_cache[jwks_uri] = (jwks, now)
        return jwks

    def invalidate(self) -> None:
        """Clear all cached discovery data and JWKS."""
        self._metadata_cache.clear()
        self._jwks_cache.clear()
        logger.debug("OIDC discovery cache invalidated")


# =============================================================================
# JWT Validator
# =============================================================================


class JWTValidator:
    """Validates JWT tokens issued by OIDC providers.

    Performs full validation including signature verification against the
    provider's JWKS, audience checks, issuer checks, and expiration checks.
    Supports automatic key rotation handling by refreshing the JWKS when a
    key ID is not found in the cached keyset.

    Args:
        config: The OIDC configuration with claim mapping and provider details.
        discovery: An OIDCDiscovery instance for fetching JWKS.

    Example::

        validator = JWTValidator(config, discovery)
        identity = await validator.validate_token(bearer_token)
    """

    def __init__(self, config: OIDCConfig, discovery: OIDCDiscovery) -> None:
        self._config = config
        self._discovery = discovery

    async def validate_token(self, token: str) -> OIDCIdentity:
        """Validate a JWT token and extract the user identity.

        Performs the full OIDC token validation flow:
        1. Fetch provider metadata (cached).
        2. Fetch JWKS (cached, with rotation support).
        3. Decode and verify the JWT signature.
        4. Validate standard claims (iss, aud, exp).
        5. Extract mapped claims into an OIDCIdentity.

        Args:
            token: The raw JWT string (without ``Bearer `` prefix).

        Returns:
            The resolved OIDCIdentity.

        Raises:
            OIDCAuthError: If validation fails for any reason.
        """
        if not authlib_available:
            raise OIDCAuthError("authlib is not installed; cannot validate OIDC tokens")

        config = self._config
        if not config.issuer_url:
            raise OIDCAuthError("OIDC issuer_url is not configured")

        # 1. Discover provider metadata
        try:
            metadata = await self._discovery.discover(config.issuer_url)
        except OIDCDiscoveryError as exc:
            raise OIDCAuthError(f"Provider discovery failed: {exc}") from exc

        # 2. Fetch JWKS
        try:
            jwks_data = await self._discovery.get_jwks(metadata.jwks_uri)
        except OIDCDiscoveryError as exc:
            raise OIDCAuthError(f"JWKS fetch failed: {exc}") from exc

        # 3. Decode and verify JWT
        jwt = JsonWebToken(["RS256", "RS384", "RS512", "ES256", "ES384", "ES512"])
        claims_options = {
            "iss": {"essential": True, "value": metadata.issuer},
            "exp": {"essential": True},
        }
        # Only enforce audience if client_id is configured
        if config.client_id:
            claims_options["aud"] = {"essential": True, "value": config.client_id}

        try:
            key_set = JsonWebKey.import_key_set(jwks_data)
            token_data = jwt.decode(token, key_set, claims_options=claims_options)
            token_data.validate()
        except ExpiredTokenError as exc:
            logger.info("OIDC token expired for issuer %s", config.issuer_url)
            raise OIDCAuthError("Token has expired") from exc
        except BadSignatureError:
            logger.warning(
                "OIDC token signature invalid for issuer %s — attempting JWKS refresh",
                config.issuer_url,
            )
            # Attempt JWKS refresh for key rotation
            try:
                jwks_data = await self._discovery.get_jwks(
                    metadata.jwks_uri, force_refresh=True
                )
                key_set = JsonWebKey.import_key_set(jwks_data)
                token_data = jwt.decode(token, key_set, claims_options=claims_options)
                token_data.validate()
            except (BadSignatureError, DecodeError, InvalidClaimError) as retry_exc:
                logger.warning(
                    "OIDC token still invalid after JWKS refresh: %s", retry_exc
                )
                raise OIDCAuthError("Invalid token signature") from retry_exc
        except (DecodeError, InvalidClaimError) as exc:
            logger.warning("OIDC token validation failed: %s", exc)
            raise OIDCAuthError(f"Token validation failed: {exc}") from exc
        except Exception as exc:
            logger.error("Unexpected error during JWT validation: %s", exc)
            raise OIDCAuthError(f"Token validation error: {exc}") from exc

        # 4. Extract mapped claims
        return self.extract_claims(dict(token_data), config)

    def extract_claims(
        self, token_data: dict[str, Any], config: OIDCConfig
    ) -> OIDCIdentity:
        """Extract and map JWT claims into an OIDCIdentity.

        Applies the configured claim mapping to transform raw JWT claims
        into a normalized identity representation. Performs email domain
        validation when ``allowed_email_domains`` is configured.

        Args:
            token_data: Decoded JWT payload as a dictionary.
            config: OIDC configuration with claim mapping rules.

        Returns:
            The mapped OIDCIdentity.

        Raises:
            OIDCAuthError: If required claims are missing or email domain is
                not allowed.
        """
        # Required claims
        user_id = token_data.get(config.user_id_claim)
        if not user_id:
            raise OIDCAuthError(
                f"Required claim '{config.user_id_claim}' missing from token"
            )

        email = token_data.get(config.email_claim, "")
        if not email:
            raise OIDCAuthError(
                f"Required claim '{config.email_claim}' missing from token"
            )

        # Email domain restriction
        if config.allowed_email_domains:
            email_domain = email.rsplit("@", 1)[-1].lower() if "@" in email else ""
            allowed_lower = [d.lower() for d in config.allowed_email_domains]
            if email_domain not in allowed_lower:
                logger.warning(
                    "OIDC login rejected: email domain '%s' not in allowed domains "
                    "(user_id=%s)",
                    email_domain,
                    user_id,
                )
                raise OIDCAuthError(f"Email domain '{email_domain}' is not allowed")

        # Optional claims
        display_name = token_data.get(config.display_name_claim)
        org_id = token_data.get(config.org_claim) if config.org_claim else None

        # Team claim: may be a string or list
        team_ids: list[str] = []
        if config.team_claim:
            raw_team = token_data.get(config.team_claim)
            if isinstance(raw_team, str):
                team_ids = [raw_team]
            elif isinstance(raw_team, list):
                team_ids = [str(t) for t in raw_team]

        # Role mapping: extract IdP roles, map to RouteIQ roles
        roles = self._resolve_roles(token_data, config)

        return OIDCIdentity(
            user_id=str(user_id),
            email=str(email),
            display_name=str(display_name) if display_name else None,
            team_id=team_ids[0] if team_ids else None,
            team_ids=team_ids,
            org_id=str(org_id) if org_id else None,
            roles=roles,
            raw_claims=token_data,
        )

    @staticmethod
    def _resolve_roles(token_data: dict[str, Any], config: OIDCConfig) -> list[str]:
        """Resolve RouteIQ roles from IdP claims and role mapping.

        The resolution flow:
        1. Extract raw roles from the configured ``role_claim``.
        2. For each raw role, look up the ``role_mapping`` dict.
        3. If a mapping exists, use the mapped RouteIQ roles.
        4. If no mapping matches, use ``default_role``.

        Args:
            token_data: Decoded JWT payload.
            config: OIDC configuration with role mapping.

        Returns:
            Deduplicated list of resolved RouteIQ roles.
        """
        if not config.role_claim:
            return [config.default_role]

        raw_roles = token_data.get(config.role_claim, [])
        if isinstance(raw_roles, str):
            raw_roles = [raw_roles]
        elif not isinstance(raw_roles, list):
            raw_roles = []

        if not raw_roles:
            return [config.default_role]

        resolved: list[str] = []
        for raw_role in raw_roles:
            raw_role_str = str(raw_role)
            mapped = config.role_mapping.get(raw_role_str)
            if mapped:
                resolved.extend(mapped)
            # If no explicit mapping, don't add the raw role — use default below

        if not resolved:
            resolved = [config.default_role]

        # Deduplicate while preserving order
        seen: set[str] = set()
        deduped: list[str] = []
        for role in resolved:
            if role not in seen:
                seen.add(role)
                deduped.append(role)
        return deduped


# =============================================================================
# Errors
# =============================================================================


class OIDCError(Exception):
    """Base exception for OIDC-related errors."""


class OIDCDiscoveryError(OIDCError):
    """Raised when OIDC provider discovery fails."""


class OIDCAuthError(OIDCError):
    """Raised when OIDC authentication or token validation fails."""


# =============================================================================
# In-memory session & state store (bounded, thread-safe)
# =============================================================================

_MAX_CACHE_SIZE = 10_000


class _BoundedTTLDict:
    """Thread-safe, bounded dict with TTL expiry."""

    def __init__(self, max_size: int = _MAX_CACHE_SIZE):
        self._data: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._max_size = max_size

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if len(self._data) >= self._max_size:
                # Evict oldest 10%
                to_remove = list(self._data.keys())[: self._max_size // 10]
                for k in to_remove:
                    del self._data[k]
            self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def pop(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.pop(key, default)

    def prune(self, predicate) -> int:
        """Remove entries where predicate(key, value) is True."""
        with self._lock:
            to_remove = [k for k, v in self._data.items() if predicate(k, v)]
            for k in to_remove:
                del self._data[k]
            return len(to_remove)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        return len(self._data)


# Maps state nonce -> (timestamp, redirect_uri, code_verifier) for CSRF protection
_pending_auth_states: _BoundedTTLDict = _BoundedTTLDict()

# Maps api_key_hash -> (OIDCIdentity, expires_at) for exchanged tokens
_exchanged_keys: _BoundedTTLDict = _BoundedTTLDict()

# Validated identity cache: token_hash -> (OIDCIdentity, expires_at)
_identity_cache: _BoundedTTLDict = _BoundedTTLDict()


def _generate_state() -> str:
    """Generate a cryptographically random state parameter for OAuth2 CSRF."""
    return secrets.token_urlsafe(32)


def _hash_token(token: str) -> str:
    """Create a SHA-256 hash of a token for safe cache keying.

    Avoids storing raw tokens in memory.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _prune_expired_states(max_age: float = 600.0) -> None:
    """Remove auth states older than max_age seconds.

    Called lazily during state lookup. Prevents unbounded memory growth from
    abandoned auth flows.
    """
    now = time.time()
    _pending_auth_states.prune(lambda k, v: now - v[0] > max_age)


def _prune_expired_identities() -> None:
    """Remove expired entries from the identity cache."""
    now = time.monotonic()
    _identity_cache.prune(lambda k, v: now > v[1])


# =============================================================================
# FastAPI Router Factory
# =============================================================================


def create_oidc_router(config: OIDCConfig) -> APIRouter:
    """Create an APIRouter with OIDC/SSO endpoints.

    When ``authlib`` is not installed, returns a minimal router with a single
    endpoint that explains the dependency requirement.

    Endpoints created:
    - ``GET /sso/login`` — Redirect the user to the IdP's authorization endpoint.
    - ``GET /sso/callback`` — Handle the IdP callback after authentication.
    - ``POST /auth/token-exchange`` — Exchange a validated OIDC JWT for a RouteIQ API key.
    - ``GET /auth/userinfo`` — Return the current user's identity from their JWT.

    Args:
        config: The OIDC configuration.

    Returns:
        A FastAPI ``APIRouter`` ready to be mounted on the application.
    """
    router = APIRouter(tags=["oidc"])

    if not authlib_available:
        logger.warning(
            "OIDC module loaded but authlib is not installed. "
            "OIDC endpoints will return 501. Install with: pip install routeiq[oidc]"
        )

        @router.get("/sso/login")
        @router.get("/sso/callback")
        @router.post("/auth/token-exchange")
        @router.get("/auth/userinfo")
        async def _oidc_not_available() -> JSONResponse:
            """Placeholder when authlib is not installed."""
            return JSONResponse(
                status_code=501,
                content={
                    "error": "oidc_dependency_missing",
                    "message": (
                        "OIDC support requires the authlib package. "
                        "Install it with: pip install routeiq[oidc]"
                    ),
                },
            )

        return router

    discovery = OIDCDiscovery(cache_ttl=config.session_ttl)
    validator = JWTValidator(config, discovery)

    # -----------------------------------------------------------------
    # GET /sso/login — Initiate SSO login
    # -----------------------------------------------------------------

    @router.get("/sso/login", summary="Initiate OIDC SSO login")
    async def sso_login(
        request: Request,
        redirect_uri: Optional[str] = Query(
            default=None,
            description="URL to redirect to after successful login. "
            "Defaults to the /sso/callback URL.",
        ),
    ) -> RedirectResponse:
        """Redirect the user to the OIDC provider's authorization endpoint.

        Generates a cryptographic state parameter for CSRF protection and
        stores it in memory for validation during the callback.

        Args:
            request: The incoming FastAPI request.
            redirect_uri: Optional override for the OAuth2 redirect URI.

        Returns:
            A 302 redirect to the IdP's authorization page.

        Raises:
            HTTPException: 500 if OIDC is misconfigured or discovery fails.
        """
        if not config.issuer_url or not config.client_id:
            raise HTTPException(
                status_code=500,
                detail="OIDC is not fully configured (missing issuer_url or client_id)",
            )

        try:
            metadata = await discovery.discover(config.issuer_url)
        except OIDCDiscoveryError as exc:
            logger.error("OIDC discovery failed during login: %s", exc)
            raise HTTPException(
                status_code=502,
                detail="Failed to discover OIDC provider metadata",
            ) from exc

        # Determine callback URL
        callback_url = redirect_uri or str(request.url_for("sso_callback"))

        # Generate state for CSRF protection
        state = _generate_state()
        code_verifier = secrets.token_urlsafe(64)
        _pending_auth_states.set(state, (time.time(), callback_url, code_verifier))

        # Build PKCE code_challenge from code_verifier (S256)
        code_challenge_digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
        code_challenge = (
            base64.urlsafe_b64encode(code_challenge_digest).rstrip(b"=").decode("ascii")
        )

        # Build authorization URL
        params = {
            "response_type": "code",
            "client_id": config.client_id,
            "redirect_uri": callback_url,
            "scope": "openid email profile",
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }

        auth_url = f"{metadata.authorization_endpoint}?{urlencode(params)}"
        logger.info("Redirecting user to OIDC provider for authentication")
        return RedirectResponse(url=auth_url, status_code=302)

    # -----------------------------------------------------------------
    # GET /sso/callback — Handle IdP callback
    # -----------------------------------------------------------------

    @router.get("/sso/callback", summary="OIDC SSO callback")
    async def sso_callback(
        request: Request,
        code: Optional[str] = Query(default=None, description="Authorization code"),
        state: Optional[str] = Query(default=None, description="CSRF state parameter"),
        error: Optional[str] = Query(default=None, description="OAuth2 error code"),
        error_description: Optional[str] = Query(
            default=None, description="OAuth2 error description"
        ),
    ) -> JSONResponse:
        """Handle the OIDC provider's callback after user authentication.

        Validates the state parameter, exchanges the authorization code for
        tokens, validates the ID token, and returns the user identity.

        Args:
            request: The incoming FastAPI request.
            code: The authorization code from the IdP.
            state: The state parameter for CSRF validation.
            error: OAuth2 error code (if authentication failed at the IdP).
            error_description: Human-readable error description.

        Returns:
            JSON response with the user identity and session information.

        Raises:
            HTTPException: 400 if the callback is invalid, 401 if token
                validation fails, 502 if token exchange fails.
        """
        # Handle IdP-side errors
        if error:
            logger.warning(
                "OIDC callback received error: %s — %s", error, error_description
            )
            raise HTTPException(
                status_code=400,
                detail=f"OIDC authentication failed: {error} — {error_description or 'no details'}",
            )

        if not code or not state:
            raise HTTPException(
                status_code=400,
                detail="Missing authorization code or state parameter",
            )

        # Validate state (CSRF protection)
        _prune_expired_states()
        state_entry = _pending_auth_states.pop(state)
        if state_entry is None:
            logger.warning("OIDC callback with invalid or expired state parameter")
            raise HTTPException(
                status_code=400,
                detail="Invalid or expired state parameter. Please restart the login flow.",
            )

        _, callback_url, code_verifier = state_entry

        if not config.issuer_url or not config.client_id:
            raise HTTPException(
                status_code=500,
                detail="OIDC is not fully configured",
            )

        # Discover metadata for token endpoint
        try:
            metadata = await discovery.discover(config.issuer_url)
        except OIDCDiscoveryError as exc:
            logger.error("OIDC discovery failed during callback: %s", exc)
            raise HTTPException(
                status_code=502,
                detail="Failed to discover OIDC provider metadata",
            ) from exc

        # Exchange authorization code for tokens
        import httpx

        token_payload = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": callback_url,
            "client_id": config.client_id,
            "code_verifier": code_verifier,
        }
        if config.client_secret:
            token_payload["client_secret"] = config.client_secret

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                token_response = await client.post(
                    metadata.token_endpoint,
                    data=token_payload,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                token_response.raise_for_status()
                token_data = token_response.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "OIDC token exchange failed: HTTP %d",
                exc.response.status_code,
            )
            raise HTTPException(
                status_code=502,
                detail="Failed to exchange authorization code for tokens",
            ) from exc
        except httpx.RequestError as exc:
            logger.error("OIDC token exchange network error: %s", exc)
            raise HTTPException(
                status_code=502,
                detail="Network error during token exchange",
            ) from exc

        # Extract and validate the ID token
        id_token = token_data.get("id_token")
        if not id_token:
            logger.error("OIDC token response missing id_token")
            raise HTTPException(
                status_code=502,
                detail="OIDC provider did not return an id_token",
            )

        try:
            identity = await validator.validate_token(id_token)
        except OIDCAuthError as exc:
            logger.warning("OIDC ID token validation failed: %s", exc)
            raise HTTPException(
                status_code=401,
                detail=f"ID token validation failed: {exc}",
            ) from exc

        # Cache the identity
        token_hash = _hash_token(id_token)
        _identity_cache.set(
            token_hash,
            (
                identity,
                time.monotonic() + config.session_ttl,
            ),
        )

        logger.info(
            "OIDC SSO login successful: user_id=%s email=%s roles=%s",
            identity.user_id,
            identity.email,
            identity.roles,
        )

        return JSONResponse(
            status_code=200,
            content={
                "status": "authenticated",
                "user_id": identity.user_id,
                "email": identity.email,
                "display_name": identity.display_name,
                "team_ids": identity.team_ids,
                "org_id": identity.org_id,
                "roles": identity.roles,
                "session_ttl": config.session_ttl,
                "token_exchange_available": config.token_exchange_enabled,
            },
        )

    # -----------------------------------------------------------------
    # POST /auth/token-exchange — Exchange JWT for API key
    # -----------------------------------------------------------------

    @router.post(
        "/auth/token-exchange",
        summary="Exchange OIDC JWT for RouteIQ API key",
        response_model=TokenExchangeResponse,
    )
    async def token_exchange(
        request: Request,
        body: TokenExchangeRequest,
    ) -> TokenExchangeResponse:
        """Exchange a validated OIDC JWT for a RouteIQ API key.

        The caller must present a valid OIDC Bearer token in the
        ``Authorization`` header. The token is validated and an API key is
        generated bound to the identity. The key TTL is clamped to the
        configured maximum.

        Args:
            request: The incoming FastAPI request.
            body: The token exchange request parameters.

        Returns:
            A response containing the API key and expiration info.

        Raises:
            HTTPException: 403 if token exchange is disabled, 401 if the
                Bearer token is missing or invalid.
        """
        if not config.token_exchange_enabled:
            raise HTTPException(
                status_code=403,
                detail="Token exchange is disabled",
            )

        # Extract Bearer token
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.lower().startswith("bearer "):
            raise HTTPException(
                status_code=401,
                detail="Authorization header with Bearer token required",
            )
        bearer_token = auth_header[7:]  # len("Bearer ") == 7

        # Validate the token
        try:
            identity = await validator.validate_token(bearer_token)
        except OIDCAuthError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

        # Calculate TTL
        ttl_days = body.ttl_days or config.default_key_ttl_days
        ttl_days = min(ttl_days, config.max_key_ttl_days)
        ttl_days = max(ttl_days, 1)
        expires_at = int(time.time()) + (ttl_days * 86400)

        # Generate API key
        api_key = f"sk-oidc-{secrets.token_urlsafe(32)}"
        key_hash = _hash_token(api_key)

        # Store the key mapping
        _exchanged_keys.set(key_hash, (identity, expires_at))

        alias = body.key_alias or f"oidc-{identity.user_id[:8]}"

        logger.info(
            "OIDC token exchange: generated API key for user_id=%s "
            "alias=%s ttl_days=%d",
            identity.user_id,
            alias,
            ttl_days,
        )

        return TokenExchangeResponse(
            api_key=api_key,
            expires_at=expires_at,
            user_id=identity.user_id,
            key_alias=alias,
        )

    # -----------------------------------------------------------------
    # GET /auth/userinfo — Return current user identity
    # -----------------------------------------------------------------

    @router.get("/auth/userinfo", summary="Get current user identity")
    async def userinfo(request: Request) -> JSONResponse:
        """Return the current user's identity from their OIDC JWT.

        Looks up the Bearer token in the identity cache first. If not
        cached, performs full JWT validation.

        Args:
            request: The incoming FastAPI request.

        Returns:
            JSON response with the user's identity information.

        Raises:
            HTTPException: 401 if no valid Bearer token is present.
        """
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.lower().startswith("bearer "):
            raise HTTPException(
                status_code=401,
                detail="Authorization header with Bearer token required",
            )
        bearer_token = auth_header[7:]

        # Check identity cache
        token_hash = _hash_token(bearer_token)
        _prune_expired_identities()
        cached = _identity_cache.get(token_hash)
        if cached is not None:
            identity, _ = cached
            return JSONResponse(
                status_code=200,
                content=identity.model_dump(exclude={"raw_claims"}),
            )

        # Full validation
        try:
            identity = await validator.validate_token(bearer_token)
        except OIDCAuthError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

        # Cache the result
        _identity_cache.set(
            token_hash,
            (
                identity,
                time.monotonic() + config.session_ttl,
            ),
        )

        return JSONResponse(
            status_code=200,
            content=identity.model_dump(exclude={"raw_claims"}),
        )

    return router


# =============================================================================
# Unified Identity Resolver
# =============================================================================

# Module-level singleton instances (set during setup_oidc)
_oidc_config: Optional[OIDCConfig] = None
_jwt_validator: Optional[JWTValidator] = None


async def resolve_identity(request: Request) -> Optional[OIDCIdentity]:
    """Resolve the caller's identity from a JWT Bearer token or exchanged API key.

    This is the primary entry point for other RouteIQ modules to obtain the
    OIDC identity associated with a request. It checks in order:

    1. Exchanged API keys (from token exchange, stored in memory).
    2. Bearer JWT tokens (validated against the OIDC provider).
    3. Returns ``None`` if no OIDC credentials are present.

    This function is safe to call even when OIDC is disabled — it will return
    ``None`` immediately.

    Args:
        request: The incoming FastAPI request.

    Returns:
        The resolved ``OIDCIdentity``, or ``None`` if no OIDC credentials
        are present or OIDC is not configured.
    """
    if _oidc_config is None or not _oidc_config.enabled:
        return None

    # 1. Check for exchanged API key (custom header or Authorization)
    api_key = request.headers.get("x-api-key") or request.headers.get("X-Api-Key") or ""
    if api_key.startswith("sk-oidc-"):
        key_hash = _hash_token(api_key)
        entry = _exchanged_keys.get(key_hash)
        if entry is not None:
            identity, expires_at = entry
            if time.time() < expires_at:
                return identity
            else:
                # Expired — clean up
                _exchanged_keys.pop(key_hash)
                logger.debug(
                    "Expired OIDC-exchanged key removed for user %s", identity.user_id
                )
        return None  # Invalid or expired OIDC key — don't fall through to JWT

    # 2. Check for Bearer JWT
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.lower().startswith("bearer "):
        return None

    bearer_token = auth_header[7:]
    if not bearer_token:
        return None

    # Check identity cache
    token_hash = _hash_token(bearer_token)
    _prune_expired_identities()
    cached = _identity_cache.get(token_hash)
    if cached is not None:
        identity, _ = cached
        return identity

    # Validate token
    if _jwt_validator is None:
        return None

    try:
        identity = await _jwt_validator.validate_token(bearer_token)
    except OIDCAuthError as exc:
        logger.debug("OIDC identity resolution failed: %s", exc)
        return None

    # Cache the result
    _identity_cache.set(
        token_hash,
        (
            identity,
            time.monotonic() + _oidc_config.session_ttl,
        ),
    )
    return identity


# =============================================================================
# Configuration Loader
# =============================================================================


def get_oidc_config() -> OIDCConfig:
    """Load OIDC configuration from environment variables.

    Reads ``ROUTEIQ_OIDC_*`` environment variables and constructs an
    ``OIDCConfig`` instance. This function is idempotent and safe to call
    multiple times.

    Environment Variable Mapping:
        - ``ROUTEIQ_OIDC_ENABLED`` → ``enabled`` (bool)
        - ``ROUTEIQ_OIDC_ISSUER_URL`` → ``issuer_url``
        - ``ROUTEIQ_OIDC_CLIENT_ID`` → ``client_id``
        - ``ROUTEIQ_OIDC_CLIENT_SECRET`` → ``client_secret``
        - ``ROUTEIQ_OIDC_USER_ID_CLAIM`` → ``user_id_claim``
        - ``ROUTEIQ_OIDC_EMAIL_CLAIM`` → ``email_claim``
        - ``ROUTEIQ_OIDC_DISPLAY_NAME_CLAIM`` → ``display_name_claim``
        - ``ROUTEIQ_OIDC_TEAM_CLAIM`` → ``team_claim``
        - ``ROUTEIQ_OIDC_ORG_CLAIM`` → ``org_claim``
        - ``ROUTEIQ_OIDC_ROLE_CLAIM`` → ``role_claim``
        - ``ROUTEIQ_OIDC_ROLE_MAPPING`` → ``role_mapping`` (JSON string)
        - ``ROUTEIQ_OIDC_DEFAULT_ROLE`` → ``default_role``
        - ``ROUTEIQ_OIDC_ALLOWED_EMAIL_DOMAINS`` → ``allowed_email_domains`` (comma-sep)
        - ``ROUTEIQ_OIDC_AUTO_PROVISION_USERS`` → ``auto_provision_users`` (bool)
        - ``ROUTEIQ_OIDC_SESSION_TTL`` → ``session_ttl`` (int)
        - ``ROUTEIQ_OIDC_TOKEN_EXCHANGE_ENABLED`` → ``token_exchange_enabled`` (bool)
        - ``ROUTEIQ_OIDC_MAX_KEY_TTL_DAYS`` → ``max_key_ttl_days`` (int)
        - ``ROUTEIQ_OIDC_DEFAULT_KEY_TTL_DAYS`` → ``default_key_ttl_days`` (int)

    Returns:
        An ``OIDCConfig`` instance populated from the environment.
    """

    def _bool_env(key: str, default: bool) -> bool:
        val = os.getenv(key, "").strip().lower()
        if not val:
            return default
        return val in ("true", "1", "yes")

    def _int_env(key: str, default: int) -> int:
        val = os.getenv(key, "").strip()
        if not val:
            return default
        try:
            return int(val)
        except ValueError:
            logger.warning(
                "Invalid integer value for %s: '%s', using default %d",
                key,
                val,
                default,
            )
            return default

    def _optional_str(key: str) -> Optional[str]:
        val = os.getenv(key, "").strip()
        return val if val else None

    # Parse role mapping JSON
    role_mapping: dict[str, list[str]] = {}
    role_mapping_raw = os.getenv("ROUTEIQ_OIDC_ROLE_MAPPING", "").strip()
    if role_mapping_raw:
        try:
            parsed = json.loads(role_mapping_raw)
            if isinstance(parsed, dict):
                role_mapping = {
                    str(k): (v if isinstance(v, list) else [str(v)])
                    for k, v in parsed.items()
                }
            else:
                logger.warning(
                    "ROUTEIQ_OIDC_ROLE_MAPPING must be a JSON object, got %s",
                    type(parsed).__name__,
                )
        except json.JSONDecodeError as exc:
            logger.warning("ROUTEIQ_OIDC_ROLE_MAPPING is not valid JSON: %s", exc)

    # Parse allowed email domains
    allowed_domains_raw = os.getenv("ROUTEIQ_OIDC_ALLOWED_EMAIL_DOMAINS", "").strip()
    allowed_domains = (
        [d.strip() for d in allowed_domains_raw.split(",") if d.strip()]
        if allowed_domains_raw
        else []
    )

    return OIDCConfig(
        enabled=_bool_env("ROUTEIQ_OIDC_ENABLED", False),
        issuer_url=_optional_str("ROUTEIQ_OIDC_ISSUER_URL"),
        client_id=_optional_str("ROUTEIQ_OIDC_CLIENT_ID"),
        client_secret=_optional_str("ROUTEIQ_OIDC_CLIENT_SECRET"),
        user_id_claim=os.getenv("ROUTEIQ_OIDC_USER_ID_CLAIM", "sub").strip(),
        email_claim=os.getenv("ROUTEIQ_OIDC_EMAIL_CLAIM", "email").strip(),
        display_name_claim=os.getenv("ROUTEIQ_OIDC_DISPLAY_NAME_CLAIM", "name").strip(),
        team_claim=_optional_str("ROUTEIQ_OIDC_TEAM_CLAIM"),
        org_claim=_optional_str("ROUTEIQ_OIDC_ORG_CLAIM"),
        role_claim=_optional_str("ROUTEIQ_OIDC_ROLE_CLAIM"),
        role_mapping=role_mapping,
        default_role=os.getenv("ROUTEIQ_OIDC_DEFAULT_ROLE", "internal_user").strip(),
        allowed_email_domains=allowed_domains,
        auto_provision_users=_bool_env("ROUTEIQ_OIDC_AUTO_PROVISION_USERS", True),
        session_ttl=_int_env("ROUTEIQ_OIDC_SESSION_TTL", 1800),
        token_exchange_enabled=_bool_env("ROUTEIQ_OIDC_TOKEN_EXCHANGE_ENABLED", True),
        max_key_ttl_days=_int_env("ROUTEIQ_OIDC_MAX_KEY_TTL_DAYS", 365),
        default_key_ttl_days=_int_env("ROUTEIQ_OIDC_DEFAULT_KEY_TTL_DAYS", 90),
    )


# =============================================================================
# Module Setup & Reset
# =============================================================================


def setup_oidc(config: Optional[OIDCConfig] = None) -> Optional[OIDCConfig]:
    """Initialize the OIDC module singletons.

    Called during application startup to set up the module-level OIDC config
    and JWT validator. If OIDC is disabled, this is a no-op.

    Args:
        config: OIDC configuration. If ``None``, loads from environment.

    Returns:
        The active ``OIDCConfig``, or ``None`` if OIDC is disabled.
    """
    global _oidc_config, _jwt_validator

    if config is None:
        config = get_oidc_config()

    _oidc_config = config

    if not config.enabled:
        logger.info("OIDC integration is disabled")
        _jwt_validator = None
        return config

    if not authlib_available:
        logger.warning(
            "OIDC is enabled but authlib is not installed. "
            "OIDC authentication will not work. "
            "Install with: pip install routeiq[oidc]"
        )
        _jwt_validator = None
        return config

    if not config.issuer_url:
        logger.warning("OIDC is enabled but ROUTEIQ_OIDC_ISSUER_URL is not set")
        _jwt_validator = None
        return config

    discovery = OIDCDiscovery(cache_ttl=config.session_ttl)
    _jwt_validator = JWTValidator(config, discovery)

    logger.info(
        "OIDC integration initialized: issuer=%s client_id=%s "
        "auto_provision=%s token_exchange=%s session_ttl=%ds",
        config.issuer_url,
        config.client_id,
        config.auto_provision_users,
        config.token_exchange_enabled,
        config.session_ttl,
    )
    return config


def reset_oidc() -> None:
    """Reset all OIDC module state.

    Clears the singleton config, validator, and all caches. This MUST be
    called in test fixtures (``autouse=True``) to prevent cross-test
    contamination.
    """
    global _oidc_config, _jwt_validator
    _oidc_config = None
    _jwt_validator = None
    _pending_auth_states.clear()
    _exchanged_keys.clear()
    _identity_cache.clear()
    logger.debug("OIDC module state reset")
