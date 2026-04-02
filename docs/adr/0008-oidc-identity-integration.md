# ADR-0008: OIDC/SSO for Enterprise Identity Management

**Status**: Accepted
**Date**: 2026-04-02
**Decision Makers**: RouteIQ Core Team

## Context

### Enterprise Identity Requirements

Enterprise deployments of RouteIQ require integration with corporate identity
providers for:

1. **Single Sign-On (SSO)**: Engineers and administrators authenticate via
   their corporate identity provider rather than managing separate credentials
   for the gateway.

2. **API key self-service**: Teams need to create, rotate, and revoke API keys
   without admin intervention. This requires authenticated sessions with
   appropriate permissions.

3. **Team/org hierarchy**: Large organizations need multi-level access control
   (org > team > user) with budget and quota inheritance.

4. **Audit compliance**: Regulatory requirements (SOC 2, HIPAA, GDPR) demand
   that all administrative actions are tied to verified identities, not just
   API keys.

### Existing Capabilities

LiteLLM already provides substantial identity infrastructure:

- **User/Team/Org tables**: `LiteLLM_UserTable`, `LiteLLM_TeamTable`,
  `LiteLLM_OrganizationTable` with budgets, metadata, and model access.
- **Virtual keys**: `LiteLLM_VerificationToken` with 44+ fields including
  spend tracking, team assignment, TPM/RPM limits, and TTL.
- **SSO endpoints**: `fastapi-sso` integration for Google, Microsoft, Okta,
  and generic OIDC providers.
- **JWT validation**: Built-in JWT token verification for API access.

RouteIQ adds:

- **Admin API key auth** (`auth.py`): Two-tier auth for control-plane vs
  data-plane operations.
- **RBAC** (`rbac.py`): 5-permission role-based access control.
- **Policy engine** (`policy_engine.py`): Subject-based policy evaluation.

### Gap Analysis

The gap between existing capabilities and enterprise requirements:

1. No OIDC Authorization Code + PKCE flow for web UI authentication
2. No token exchange endpoint (OIDC token -> RouteIQ API key)
3. No automatic team/org mapping from OIDC claims
4. No API key self-service UI
5. RouteIQ's RBAC is independent of LiteLLM's team permissions

## Decision

Implement OIDC/SSO integration that bridges RouteIQ's security layer with
LiteLLM's identity infrastructure.

### Architecture

```
[Corporate IdP]  <--OIDC-->  [RouteIQ Gateway]  <--DB-->  [LiteLLM Tables]
 (Keycloak,                    /auth/oidc/*                  Users, Teams,
  Auth0, Okta,                 /auth/token                   Orgs, Keys
  Azure AD,                    /auth/keys/*
  Google)
```

### OIDC Authorization Code + PKCE Flow

For web UI authentication (admin dashboard, playground):

1. Browser redirects to `/auth/oidc/login`
2. RouteIQ generates PKCE code verifier + challenge
3. Redirect to IdP authorization endpoint with PKCE
4. User authenticates at IdP
5. IdP redirects back with authorization code
6. RouteIQ exchanges code + verifier for tokens at IdP token endpoint
7. RouteIQ validates ID token, extracts claims
8. Creates or updates user in LiteLLM's `LiteLLM_UserTable`
9. Issues RouteIQ session token (JWT, short-lived)

### JWT Validation for API Access

For programmatic API access:

1. Client includes `Authorization: Bearer <jwt>` header
2. RouteIQ validates JWT signature against IdP's JWKS endpoint
3. Extracts `sub`, `email`, `groups`, `org` claims
4. Maps to LiteLLM user/team/org
5. Applies RBAC permissions from mapped roles

### Token Exchange Endpoint

`POST /auth/token/exchange` converts an OIDC access token into a RouteIQ
API key:

```json
{
  "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
  "subject_token": "<oidc_access_token>",
  "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
  "requested_token_type": "routeiq:api_key",
  "scope": "chat completions",
  "ttl": 3600
}
```

Response:
```json
{
  "access_token": "sk-riq-...",
  "token_type": "api_key",
  "expires_in": 3600
}
```

### Claim-to-Role Mapping

OIDC claims are mapped to LiteLLM's hierarchy via configuration:

```yaml
oidc:
  provider: keycloak
  issuer: https://keycloak.example.com/realms/routeiq
  client_id: routeiq-gateway
  claim_mapping:
    org: "custom:organization"
    team: "custom:team"
    role: "custom:routeiq_role"
  role_mapping:
    admin: ["routeiq-admin", "platform-team"]
    developer: ["engineering"]
    viewer: ["readonly"]
```

### Supported Providers

| Provider | Protocol | Notes |
|----------|----------|-------|
| Keycloak | OIDC | Full support, recommended for self-hosted |
| Auth0 | OIDC | Full support, custom claims via Actions |
| Okta | OIDC | Full support, custom claims via authorization servers |
| Azure AD | OIDC | Full support, uses `tid` for org mapping |
| Google Workspace | OIDC | Full support, uses `hd` (hosted domain) for org |

### Multi-Tenancy

Multi-tenancy is implemented via `org_id` scoping in LiteLLM's existing
tables. Each OIDC-authenticated user is assigned to an organization based
on their IdP claims, and all resource access (keys, budgets, models) is
scoped to that organization.

## Consequences

### Positive

- **Enterprise SSO**: Administrators authenticate via corporate IdP without
  managing separate credentials. Meets SOC 2 and compliance requirements.

- **Self-service key management**: Teams can create/rotate API keys through
  the web UI without admin intervention.

- **Automatic provisioning**: New users are auto-provisioned in the correct
  org/team based on OIDC claims. No manual user creation needed.

- **Leverages LiteLLM tables**: No new database tables. Users, teams, orgs,
  and keys all use LiteLLM's existing schema.

- **Standard protocol**: OIDC is widely adopted. Any identity provider that
  supports OIDC can integrate with RouteIQ.

### Negative

- **External dependency**: Requires a running OIDC identity provider.
  Deployments without an IdP must use static API keys (backward compatible).

- **Configuration complexity**: OIDC setup requires configuring client IDs,
  secrets, redirect URIs, and claim mappings. This is inherent to OIDC, not
  specific to RouteIQ.

- **Token management**: Short-lived JWTs require periodic refresh. Clients
  must handle token expiration and renewal.

- **Optional dependency**: The `authlib` library is required but only installed
  with `[oidc]` extra (see [ADR-0007](0007-dependency-tiering.md)).

## Alternatives Considered

### Alternative A: Custom Auth System

Build a custom authentication system with email/password, MFA, etc.

- **Pros**: No external IdP dependency; full control.
- **Cons**: Massive security liability; duplicates what every IdP already
  does; requires storing and protecting passwords; no SSO capability.
- **Rejected**: Building auth from scratch is almost always wrong.

### Alternative B: API Key Only

Keep using only static API keys for all authentication.

- **Pros**: Simplest; no external dependencies; works everywhere.
- **Cons**: No SSO; no self-service; no audit trail tied to identities;
  key rotation requires admin intervention; doesn't meet enterprise
  compliance requirements.
- **Rejected**: Insufficient for enterprise deployments. Retained as
  fallback for simple deployments.

### Alternative C: OAuth 2.0 (Without OIDC)

Use plain OAuth 2.0 without the OpenID Connect identity layer.

- **Pros**: Simpler protocol; fewer claims to validate.
- **Cons**: OAuth 2.0 alone doesn't provide identity information (who the
  user is). Would need additional endpoints to fetch user profile. OIDC
  standardizes this via ID tokens.
- **Rejected**: OIDC is the standard identity protocol built on OAuth 2.0.
  Using plain OAuth 2.0 would require reimplementing what OIDC provides.

## References

- `src/litellm_llmrouter/auth.py` — Admin auth integration point
- `src/litellm_llmrouter/rbac.py` — Role-based access control
- `pyproject.toml` — `[oidc]` optional extra (authlib)
- [ADR-0007: Dependency Tiering](0007-dependency-tiering.md)
- OpenID Connect Core: https://openid.net/specs/openid-connect-core-1_0.html
- RFC 8693: OAuth 2.0 Token Exchange
- LiteLLM SSO docs: `reference/litellm/docs/my-website/docs/proxy/ui.md`
