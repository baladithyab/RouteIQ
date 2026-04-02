# Security

Security is a core design principle of RouteIQ Gateway. This guide outlines the
security features and best practices for securing your deployment.

## Authentication

### Two-Tier Model

RouteIQ implements a two-tier authentication model to separate user traffic from
administrative operations:

- **User API Keys**: Standard LiteLLM API key authentication (`Authorization: Bearer sk-xxx`)
  for inference endpoints, read-only monitoring, and MCP/A2A listing endpoints
- **Admin API Keys**: Separate admin API key for control-plane operations

Admin keys are provided via:

- **Preferred**: `X-Admin-API-Key: <key>` header
- **Fallback**: `Authorization: Bearer <key>` (checked against admin key list)

### Admin Auth Configuration

```bash
# .env - Generate with: openssl rand -hex 32
ADMIN_API_KEYS=key1,key2,key3

# Or single key (legacy)
ADMIN_API_KEY=your-admin-key

# Disable admin auth (NOT recommended for production)
# ADMIN_AUTH_ENABLED=false
```

**Fail-closed behavior**: If `ADMIN_API_KEYS` is not configured and `ADMIN_AUTH_ENABLED`
is not explicitly `false`, control-plane endpoints return `403 Forbidden` with error code
`control_plane_not_configured`.

**Protected endpoints (require admin auth):**

- `POST /router/reload` — Hot reload routing strategies
- `POST /config/reload` — Reload configuration
- `POST/PUT/DELETE /llmrouter/mcp/servers/*` — MCP server management
- `POST /llmrouter/mcp/servers/{id}/tools` — MCP tool registration
- `POST /llmrouter/mcp/tools/call` — MCP tool invocation
- `POST/DELETE /a2a/agents` — A2A agent registration

**Unauthenticated endpoints** (for Kubernetes compatibility):

- `/_health/live` — Liveness probe
- `/_health/ready` — Readiness probe

## Role-Based Access Control (RBAC)

RBAC assigns roles (e.g., `admin`, `user`, `viewer`) to API keys and enforces access
policies based on those roles. It integrates with the management middleware for
endpoint-level authorization, ensuring that only appropriately privileged users can
access sensitive operations.

Key features:

- **Role assignment**: API keys are mapped to roles via configuration
- **Endpoint-level authorization**: Each endpoint declares its required role
- **Integration with management middleware**: Classifies LiteLLM management endpoints
  so middleware can enforce RBAC policies on upstream operations
- **Fail-closed by default**: Requests without a valid role assignment are denied

## Policy Engine

RouteIQ includes an OPA-style policy engine that evaluates pre-request policies to
allow or deny traffic at the ASGI middleware layer.

```bash
POLICY_ENGINE_ENABLED=true
POLICY_CONFIG_PATH=config/policy.example.yaml
```

**Evaluation context** — the policy engine evaluates each request against:

- **Subject**: The authenticated user/key identity
- **Route**: The target API endpoint path
- **Model**: The requested model (for inference endpoints)
- **Headers**: Request headers
- **Source IP**: The client's IP address

**Policy modes:**

- **Fail-open** (default): Policy evaluation errors allow the request through
- **Fail-closed**: Any policy evaluation error denies the request

## Request Correlation

All requests are assigned a correlation ID for tracing and debugging:

- **Header**: `X-Request-ID`
- **Passthrough**: If provided in the request, the same ID is used
- **Generated**: UUID v4 generated if not provided
- **Response**: Returned in `X-Request-ID` response header
- **Error bodies**: All error responses include `request_id` field

## Error Response Sanitization

Client-facing error responses are sanitized to prevent information leakage:

- **No stack traces** in responses
- **No internal exception messages** (e.g., database connection strings)
- **Generic messages** with machine-readable error codes
- **Request ID** for correlation with server logs

| HTTP Status | Error Code | When |
|------------|------------|------|
| 401 | `admin_key_required` | Missing admin API key |
| 401 | `invalid_admin_key` | Invalid admin API key |
| 403 | `control_plane_not_configured` | Admin keys not set up |
| 500 | `internal_error` | Unexpected server error |
| 503 | (readiness detail) | Dependency health check failed |

## Secret Scrubbing

All error logs are scrubbed of sensitive values via `auth._scrub_secrets()`:

- API keys and bearer tokens
- Database connection strings
- Cloud provider credentials
- Any values matching known secret patterns

## Audit Logging

Structured audit logging for tracking security-relevant operations:

- Authentication attempts (success and failure)
- Control-plane operations (config changes, model management)
- Policy engine decisions (allow/deny with context)
- MCP server and A2A agent registration/deregistration

Supports file-based logging and structured event objects for SIEM/observability forwarding.

## SSRF Protection

Server-Side Request Forgery protection with a **secure-by-default (fail-closed)** policy.

### Default Behavior

| Target | Always Blocked | Reason |
|--------|---------------|--------|
| Loopback (127.0.0.0/8, ::1) | Yes | Cannot be overridden |
| Link-local (169.254.0.0/16) | Yes | Includes cloud metadata endpoints |
| Localhost hostnames | Yes | Cannot be overridden |
| Private IPs (10.x, 172.16.x, 192.168.x) | By default | Can be allowed via allowlist |

### Configuration

```bash
# Allow ALL private IP ranges (NOT recommended for production)
LLMROUTER_ALLOW_PRIVATE_IPS=false

# Allowlist specific hosts/domains (comma-separated)
# Supports exact match and suffix match (prefix with ".")
LLMROUTER_SSRF_ALLOWLIST_HOSTS=mcp.internal,.trusted-corp.com

# Allowlist specific IP ranges in CIDR notation
LLMROUTER_SSRF_ALLOWLIST_CIDRS=10.100.0.0/16,192.168.50.0/24
```

### Where SSRF Protection is Enforced

1. MCP Server Registration (`POST /llmrouter/mcp/servers`)
2. MCP Server Update (`PUT /llmrouter/mcp/servers/{id}`)
3. MCP Tool Invocation (when calling registered servers)
4. A2A Agent Registration (`POST /a2a/agents`)
5. A2A Agent Invocation (when forwarding requests to agents)

!!! warning "Dual Validation"
    SSRF validation happens twice: at registration time (no DNS) and at invocation
    time (with DNS resolution) to catch DNS rebinding attacks.

## Security Plugins

| Plugin | Protection |
|--------|------------|
| Content Filter | Harmful content blocking |
| PII Guard | PII detection and redaction |
| Prompt Injection Guard | Injection pattern detection |
| LlamaGuard | Safety classification |
| Bedrock Guardrails | AWS-managed content moderation |

## Model Artifact Security

RouteIQ uses ML models for routing. To prevent arbitrary code execution from
malicious model files:

- **Pickle Disabled**: Loading via Python's `pickle` is **disabled by default**
- **Safe Formats**: Recommend `safetensors` or ONNX for model weights
- **Opt-in Only**: Enable with `LLMROUTER_ALLOW_PICKLE_MODELS=true`

### Manifest-Based Verification

When pickle loading is enabled, cryptographic verification ensures only authorized
model files can be loaded:

```bash
LLMROUTER_ALLOW_PICKLE_MODELS=true
LLMROUTER_MODEL_MANIFEST_PATH=/app/models/manifest.json
LLMROUTER_MODEL_PUBLIC_KEY_B64=<base64-encoded-32-byte-public-key>
```

Supported signature types: `ed25519` (recommended), `hmac-sha256`, `none` (hash only).

### Safe Activation with Rollback

Hot-reloaded models use a safe activation pattern:

1. Load new model into a temporary instance
2. Verify against manifest before activating
3. Atomic swap to new model only if verification succeeds
4. Rollback: if loading/verification fails, keep old model active

## Key Management

- **Environment Variables**: Use env vars (e.g., `OPENAI_API_KEY`) injected at runtime
- **Secret Managers**: Use AWS Secrets Manager, HashiCorp Vault, or K8s Secrets in production

## Kubernetes Security Context

```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  runAsGroup: 3000
  fsGroup: 2000
  readOnlyRootFilesystem: true
  allowPrivilegeEscalation: false
  capabilities:
    drop:
      - ALL
```

## Network Policies

Restrict traffic using Kubernetes Network Policies:

- **Ingress**: Allow traffic only from application services or ingress controller
- **Egress**: Allow traffic only to LLM provider APIs and internal dependencies (Redis, Postgres)
