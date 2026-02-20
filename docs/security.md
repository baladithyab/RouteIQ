# Security Guide

> **Attribution**:
> RouteIQ is built on top of upstream [LiteLLM](https://github.com/BerriAI/litellm) for proxy/API compatibility and [LLMRouter](https://github.com/ulab-uiuc/LLMRouter) for ML routing.

Security is a core design principle of RouteIQ Gateway. This guide outlines the security features and best practices for securing your deployment.

## Authentication Boundaries

RouteIQ Gateway implements a two-tier authentication model to separate user traffic from administrative operations:

### User API Key Authentication

Standard LiteLLM API key authentication (`Authorization: Bearer sk-xxx`) is used for:
- All inference endpoints (`/chat/completions`, `/completions`, etc.)
- Read-only monitoring endpoints (`/router/info`, `/config/sync/status`)
- Read-only MCP/A2A listing endpoints

### Admin API Key Authentication

Control-plane operations require a separate admin API key to prevent accidental or malicious configuration changes. Admin keys are provided via:
- **Preferred**: `X-Admin-API-Key: <key>` header
- **Fallback**: `Authorization: Bearer <key>` (checked against admin key list)

**Protected endpoints (require admin auth):**
- `POST /router/reload` - Hot reload routing strategies
- `POST /config/reload` - Reload configuration
- `POST/PUT/DELETE /llmrouter/mcp/servers/*` - MCP server management
- `POST /llmrouter/mcp/servers/{id}/tools` - MCP tool registration
- `POST /llmrouter/mcp/tools/call` - MCP tool invocation
- `POST/DELETE /a2a/agents` - A2A agent registration

**Configuration:**

```bash
# .env - Generate with: openssl rand -hex 32
ADMIN_API_KEYS=key1,key2,key3

# Or single key (legacy)
ADMIN_API_KEY=your-admin-key

# Disable admin auth (NOT recommended for production)
# ADMIN_AUTH_ENABLED=false
```

**Fail-closed behavior:** If `ADMIN_API_KEYS` is not configured and `ADMIN_AUTH_ENABLED` is not explicitly `false`, control-plane endpoints return `403 Forbidden` with error code `control_plane_not_configured`.

### Unauthenticated Endpoints

Health probe endpoints remain unauthenticated for Kubernetes compatibility:
- `/_health/live` - Liveness probe
- `/_health/ready` - Readiness probe

## Request Correlation

All requests are assigned a correlation ID for tracing and debugging:

- **Header**: `X-Request-ID`
- **Passthrough**: If provided in the request, the same ID is used
- **Generated**: UUID v4 is generated if not provided
- **Response**: ID is returned in `X-Request-ID` response header
- **Error bodies**: All error responses include `request_id` field

Example error response:
```json
{
  "error": "invalid_admin_key",
  "message": "Invalid admin API key.",
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

Server-side logs include the request ID for correlation:
```
ERROR request_id=550e8400-e29b-41d4-a716-446655440000 Invalid admin API key in request
```

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

## Role-Based Access Control (RBAC)

RouteIQ Gateway provides role-based access control via [`rbac.py`](../src/litellm_llmrouter/rbac.py) for fine-grained authorization of API endpoints.

RBAC assigns roles (e.g., `admin`, `user`, `viewer`) to API keys and enforces access policies based on those roles. It integrates with the management middleware ([`management_middleware.py`](../src/litellm_llmrouter/management_middleware.py)) for endpoint-level authorization, ensuring that only appropriately privileged users can access sensitive operations like configuration changes, model management, and agent registration.

Key features:
- **Role assignment**: API keys are mapped to roles via configuration
- **Endpoint-level authorization**: Each endpoint declares its required role
- **Integration with management middleware**: The [`management_classifier.py`](../src/litellm_llmrouter/management_classifier.py) classifies LiteLLM management endpoints so the middleware can enforce RBAC policies on upstream operations
- **Fail-closed by default**: Requests without a valid role assignment are denied

## Policy Engine

RouteIQ Gateway includes an OPA-style policy engine via [`policy_engine.py`](../src/litellm_llmrouter/policy_engine.py) that evaluates pre-request policies to allow or deny traffic at the ASGI middleware layer.

**Configuration:**

```bash
# Enable the policy engine
POLICY_ENGINE_ENABLED=true

# Path to the policy configuration file
POLICY_CONFIG_PATH=config/policy.example.yaml
```

**Evaluation context** — the policy engine evaluates each request against:
- **Subject**: The authenticated user/key identity
- **Route**: The target API endpoint path
- **Model**: The requested model (for inference endpoints)
- **Headers**: Request headers
- **Source IP**: The client's IP address

**Policy modes:**
- **Fail-open** (default): If the policy engine encounters an error during evaluation, the request is allowed through. This prevents policy misconfigurations from causing outages.
- **Fail-closed**: If enabled, any policy evaluation error results in the request being denied. Use this in high-security environments where safety is preferred over availability.

**Example policy rules** are provided in [`config/policy.example.yaml`](../config/policy.example.yaml).

## Audit Logging

RouteIQ Gateway provides structured audit logging via [`audit.py`](../src/litellm_llmrouter/audit.py) for tracking security-relevant operations.

Audit events are emitted for:
- Authentication attempts (success and failure)
- Control-plane operations (config changes, model management)
- Policy engine decisions (allow/deny with context)
- MCP server and A2A agent registration/deregistration

Audit logs support two output modes:
- **File-based logging**: Structured audit events written to a log file for offline analysis
- **Structured event logging**: Machine-readable event objects that can be forwarded to SIEM systems or observability platforms

Each audit event includes the request ID, timestamp, actor identity, action performed, and outcome for full traceability.

## Secret Scrubbing

All error logs in RouteIQ Gateway are scrubbed of sensitive values before being emitted to prevent accidental secret leakage. The [`auth._scrub_secrets()`](../src/litellm_llmrouter/auth.py) function automatically redacts:

- API keys and bearer tokens
- Database connection strings
- Cloud provider credentials
- Any values matching known secret patterns

This applies to all log output, including error responses logged server-side, middleware error handling, and plugin error paths. Combined with the [Error Response Sanitization](#error-response-sanitization) for client-facing responses, this ensures secrets are never exposed in either logs or API responses.

## SSRF Protection

Server-Side Request Forgery (SSRF) is a major risk for gateways that proxy requests. RouteIQ Gateway includes built-in SSRF guards with a **secure-by-default (fail-closed)** policy.

### Default Behavior (Secure-by-Default)

By default, the gateway blocks requests to:

| Target | Always Blocked | Reason |
|--------|---------------|--------|
| Loopback (127.0.0.0/8, ::1) | ✅ Yes | Cannot be overridden |
| Link-local (169.254.0.0/16) | ✅ Yes | Includes cloud metadata endpoints |
| Localhost hostnames | ✅ Yes | Cannot be overridden |
| Private IPs (10.x, 172.16.x, 192.168.x) | By default | Can be allowed via allowlist |

**Important:** Loopback and link-local addresses are **always blocked** and cannot be allowed via allowlist configuration. This protects against cloud metadata endpoint attacks.

### Configuration

Configure SSRF protection via environment variables:

```bash
# .env

# Allow ALL private IP ranges (NOT recommended for production)
# Default: false (blocked)
LLMROUTER_ALLOW_PRIVATE_IPS=false

# Allowlist specific hosts/domains (comma-separated)
# Supports exact match and suffix match (prefix with ".")
LLMROUTER_SSRF_ALLOWLIST_HOSTS=mcp.internal,.trusted-corp.com

# Allowlist specific IP ranges in CIDR notation (comma-separated)
LLMROUTER_SSRF_ALLOWLIST_CIDRS=10.100.0.0/16,192.168.50.0/24
```

### Host Allowlist Patterns

The `LLMROUTER_SSRF_ALLOWLIST_HOSTS` setting supports two patterns:

| Pattern | Example | Matches |
|---------|---------|---------|
| Exact match | `mcp.internal` | Only `mcp.internal` |
| Suffix match | `.trusted.com` | `api.trusted.com`, `mcp.trusted.com`, `trusted.com` |

### CIDR Allowlist

The `LLMROUTER_SSRF_ALLOWLIST_CIDRS` setting allows specific IP ranges:

```bash
# Allow a specific /16 for internal MCP servers
LLMROUTER_SSRF_ALLOWLIST_CIDRS=10.100.0.0/16

# Multiple ranges
LLMROUTER_SSRF_ALLOWLIST_CIDRS=10.100.0.0/16,192.168.50.0/24
```

**Note:** CIDR allowlists cannot override the loopback/link-local blocks.

### Where SSRF Protection is Enforced

SSRF protection is enforced at:

1. **MCP Server Registration** (`POST /llmrouter/mcp/servers`)
2. **MCP Server Update** (`PUT /llmrouter/mcp/servers/{id}`)
3. **MCP Tool Invocation** (when calling registered servers)
4. **A2A Agent Registration** (`POST /a2a/agents`)
5. **A2A Agent Invocation** (when forwarding requests to agents)

### Error Responses

When a URL is blocked, the API returns:

```json
{
  "error": "ssrf_blocked",
  "message": "Server URL blocked for security reasons: private IP 10.0.0.1 is blocked",
  "request_id": "abc123"
}
```

### Production Recommendations

1. **Keep `LLMROUTER_ALLOW_PRIVATE_IPS=false`** (the default)
2. Use explicit CIDR allowlists for internal services
3. Prefer hostname allowlists over broad IP ranges
4. Monitor blocked requests via logs for security auditing

## Artifact Safety

RouteIQ Gateway uses machine learning models for routing. To prevent arbitrary code execution from malicious model files:

- **Pickle Disabled**: Loading models via Python's `pickle` module is **disabled by default**.
- **Safe Formats**: We recommend using `safetensors` or ONNX for model weights.
- **Opt-in Only**: If you must use pickle (e.g., for legacy Scikit-Learn models), you must explicitly enable it via environment variable: `LLMROUTER_ALLOW_PICKLE_MODELS=true`.

### Manifest-Based Verification

When pickle loading is enabled, RouteIQ provides cryptographic verification of model artifacts against a signed manifest. This ensures only authorized model files can be loaded.

**Configuration:**

```bash
# .env

# Enable pickle loading (required for sklearn models)
LLMROUTER_ALLOW_PICKLE_MODELS=true

# Path to the model manifest (JSON or YAML)
LLMROUTER_MODEL_MANIFEST_PATH=/app/models/manifest.json

# Enforce manifest verification (automatic when pickle is enabled)
# Set to "false" to bypass (NOT recommended)
LLMROUTER_ENFORCE_SIGNED_MODELS=true

# For Ed25519 signature verification (recommended)
LLMROUTER_MODEL_PUBLIC_KEY_B64=<base64-encoded-32-byte-public-key>
# Or use a file path
LLMROUTER_MODEL_PUBLIC_KEY_PATH=/app/secrets/model-signing-key.pub

# For HMAC-SHA256 verification (alternative)
LLMROUTER_MODEL_HMAC_SECRET=<shared-secret>
```

### Manifest Format

The manifest lists allowed model artifacts with their SHA256 hashes:

```json
{
  "version": "1.0",
  "created_at": "2026-01-28T10:00:00Z",
  "signature_type": "ed25519",
  "signature": "<base64-encoded-signature>",
  "artifacts": [
    {
      "path": "models/knn_router.pkl",
      "sha256": "a1b2c3d4e5f6...",
      "description": "Production KNN routing model v1.2",
      "tags": ["production", "v1.2"]
    }
  ]
}
```

See [`models/manifest.example.json`](../models/manifest.example.json) for a complete example.

### Supported Signature Types

| Type | Environment Variable | Description |
|------|---------------------|-------------|
| `ed25519` | `LLMROUTER_MODEL_PUBLIC_KEY_B64` or `_PATH` | **Recommended**. Asymmetric signing with Ed25519. |
| `hmac-sha256` | `LLMROUTER_MODEL_HMAC_SECRET` | Symmetric signing with HMAC-SHA256. |
| `none` | N/A | Unsigned manifest (hash verification only). |

### Verification Flow

1. **Load Manifest**: Gateway loads manifest from `LLMROUTER_MODEL_MANIFEST_PATH`
2. **Verify Signature**: If `signature_type` is not `none`, verify the manifest signature
3. **Artifact Check**: Before loading any pickle file:
   - Check if artifact exists in manifest
   - Compute SHA256 hash of the file
   - Compare with manifest hash
4. **Reject on Mismatch**: If hash doesn't match, loading is blocked

### Safe Activation with Rollback

When hot-reloading models, RouteIQ uses a safe activation pattern:

1. **Load New Model**: Load the new model into a temporary instance
2. **Verify**: Check against manifest before activating
3. **Atomic Swap**: Only swap to the new model if verification succeeds
4. **Rollback**: If loading or verification fails, keep the old model active
5. **Observability**: Log and track active model versions with SHA256 hashes

This ensures the gateway never enters an invalid state during reloads:

```python
# Internal behavior (pseudocode)
def reload_model(self):
    old_model = self.active_model
    try:
        new_model = load_and_verify(self.model_path)
        self.active_model = new_model  # Only on success
        return True
    except (VerificationError, LoadError):
        # Keep old model active
        log.error("Reload failed, keeping old model")
        return False
```

### Recommended Signing Workflow

**1. Generate Ed25519 Keypair (one-time):**

```python
from litellm_llmrouter.model_artifacts import generate_ed25519_keypair

private_key_b64, public_key_b64 = generate_ed25519_keypair()
print(f"Private key (keep secret): {private_key_b64}")
print(f"Public key (distribute):   {public_key_b64}")
```

Or via OpenSSL:
```bash
# Generate private key
openssl genpkey -algorithm ED25519 -out model-signing-key.pem

# Extract public key
openssl pkey -in model-signing-key.pem -pubout -out model-signing-key.pub
```

**2. Create Signed Manifest (in CI/CD):**

```python
from litellm_llmrouter.model_artifacts import ManifestSigner, SignatureType

signer = ManifestSigner(private_key_path="/secrets/model-signing-key.pem")

# Create entries for each model file
entry = signer.create_artifact_entry(
    "models/knn_router.pkl",
    description="KNN routing model v1.2",
    tags=["production", "v1.2"],
)

# Create and sign manifest
manifest = signer.create_manifest(
    artifacts=[entry],
    signature_type=SignatureType.ED25519,
    version="1.0",
)

# Save to file
signer.save_manifest(manifest, "models/manifest.json")
```

**3. Deploy with Public Key:**

```yaml
# Kubernetes ConfigMap or Secret
apiVersion: v1
kind: Secret
metadata:
  name: model-signing-key
data:
  public-key: <base64-encoded-public-key>
```

```yaml
# Deployment environment
env:
  - name: LLMROUTER_ALLOW_PICKLE_MODELS
    value: "true"
  - name: LLMROUTER_MODEL_MANIFEST_PATH
    value: "/app/models/manifest.json"
  - name: LLMROUTER_MODEL_PUBLIC_KEY_B64
    valueFrom:
      secretKeyRef:
        name: model-signing-key
        key: public-key
```

### Observability

Active model versions are tracked for observability:

- **SHA256 hash** of the active model
- **Manifest path** used for verification
- **Load timestamp**
- **Tags** from manifest (e.g., `["production", "v1.2"]`)

This metadata is available via the routing span attributes and can be queried in your observability platform.

## Key Management

Never store API keys in your configuration files or code.

- **Environment Variables**: Use environment variables (e.g., `OPENAI_API_KEY`) which are injected into the container at runtime.
- **Secret Managers**: In production, use a secret manager (AWS Secrets Manager, HashiCorp Vault, Kubernetes Secrets) to inject these variables.

## Kubernetes Security Context

When deploying to Kubernetes, we recommend the following security context to minimize the attack surface:

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

Restrict traffic to and from the gateway using Kubernetes Network Policies.

- **Ingress**: Allow traffic only from your application services or ingress controller.
- **Egress**: Allow traffic only to the necessary LLM provider APIs and internal dependencies (Redis, Postgres).
