# Security

Security is a core design principle of RouteIQ Gateway.

## Authentication

### Two-Tier Model

- **User API Keys**: For inference endpoints (`Authorization: Bearer sk-xxx`)
- **Admin API Keys**: For control-plane operations (`X-Admin-API-Key: <key>`)

### Admin Auth

```bash
ADMIN_API_KEYS=key1,key2,key3
```

**Fail-closed**: If `ADMIN_API_KEYS` is not configured, control-plane returns 403.

## SSRF Protection

All external URLs are validated via `url_security.py`:

- Validation at registration time (no DNS resolution)
- Validation at invocation time (with DNS resolution)
- Catches DNS rebinding attacks

## Request Correlation

All requests get a `X-Request-ID` header:

- Passed through if provided, or UUID v4 generated
- Included in all error responses
- Logged server-side for correlation

## Error Sanitization

Client-facing errors are sanitized to prevent information leakage.
Internal stack traces are never exposed to clients.

## Secret Scrubbing

All error logs are scrubbed via `auth._scrub_secrets()` to prevent
API keys from appearing in log output.

## Security Plugins

| Plugin | Protection |
|--------|------------|
| Content Filter | Harmful content blocking |
| PII Guard | PII detection and redaction |
| Prompt Injection Guard | Injection pattern detection |
| LlamaGuard | Safety classification |
| Bedrock Guardrails | AWS-managed content moderation |

## Model Artifact Security

- Pickle loading disabled by default
- Enable with `LLMROUTER_ALLOW_PICKLE_MODELS=true`
- Manifest verification: `LLMROUTER_ENFORCE_SIGNED_MODELS=true`
