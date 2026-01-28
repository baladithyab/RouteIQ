# Security Guide

Security is a core design principle of RouteIQ Gateway. This guide outlines the security features and best practices for securing your deployment.

## SSRF Protection

Server-Side Request Forgery (SSRF) is a major risk for gateways that proxy requests. RouteIQ Gateway includes built-in SSRF guards.

- **Private Address Blocking**: By default, the gateway blocks requests to private IP ranges (e.g., `127.0.0.1`, `10.0.0.0/8`, `192.168.0.0/16`) unless explicitly allowed.
- **Allowlist**: You can configure an allowlist of domains or IPs that the gateway is permitted to contact.

```yaml
# config.yaml
general_settings:
  ssrf_protection: true
  allowed_domains:
    - "api.openai.com"
    - "api.anthropic.com"
```

## Artifact Safety

RouteIQ Gateway uses machine learning models for routing. To prevent arbitrary code execution from malicious model files:

- **Pickle Disabled**: Loading models via Python's `pickle` module is **disabled by default**.
- **Safe Formats**: We recommend using `safetensors` or ONNX for model weights.
- **Opt-in Only**: If you must use pickle (e.g., for legacy Scikit-Learn models), you must explicitly enable it via environment variable: `LLMROUTER_ALLOW_PICKLE_MODELS=true`.

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
