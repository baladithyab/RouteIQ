# Identity (OIDC/SSO)

RouteIQ supports OIDC/SSO for enterprise identity management.

## Overview

OIDC integration enables:

- Single Sign-On via corporate identity providers (Okta, Azure AD, Auth0)
- JWT-based authentication for API requests
- User identity mapping to teams and workspaces
- Automatic role assignment based on OIDC claims

## Enabling OIDC

```bash
ROUTEIQ_OIDC_ENABLED=true
ROUTEIQ_OIDC_ISSUER_URL=https://your-idp.example.com
ROUTEIQ_OIDC_CLIENT_ID=your-client-id
ROUTEIQ_OIDC_CLIENT_SECRET=your-client-secret
```

## Configuration

| Variable | Description |
|----------|-------------|
| `ROUTEIQ_OIDC_ENABLED` | Enable OIDC authentication |
| `ROUTEIQ_OIDC_ISSUER_URL` | OIDC provider discovery URL |
| `ROUTEIQ_OIDC_CLIENT_ID` | Client ID |
| `ROUTEIQ_OIDC_CLIENT_SECRET` | Client secret |

Requires the `oidc` extra: `uv add routeiq[oidc]`

## Authentication Flow

1. Client obtains JWT from identity provider
2. Client sends request with `Authorization: Bearer <jwt>`
3. RouteIQ validates JWT against OIDC issuer
4. User identity extracted from claims
5. Team/workspace resolved from user mapping
