# ADR-0006: Secure-by-Default Security Posture

**Status**: Accepted
**Date**: 2026-04-02
**Decision Makers**: RouteIQ Core Team

## Context

### Security Audit Findings

A comprehensive security review of RouteIQ identified multiple gaps in the
default security posture. While RouteIQ had strong security features (SSRF
protection, policy engine, model artifact verification), several implementation
defaults were insecure:

1. **Timing attack on API key comparison**: Admin API key validation used
   Python's `in` operator (`candidate in valid_keys`) which short-circuits
   on the first matching character. An attacker could brute-force API keys
   one character at a time by measuring response time differences.

2. **Log injection via request IDs**: The `RequestIDMiddleware` accepted
   arbitrary client-provided `X-Request-ID` headers without validation.
   An attacker could inject newline characters, ANSI escape codes, or
   log-format strings into request IDs, corrupting log parsers or enabling
   log injection attacks.

3. **Policy name leakage**: When the policy engine denied a request, the
   error response included the internal policy name (e.g.,
   `"denied by policy: block-gpt4-for-free-tier"`). This leaked internal
   policy structure to external callers.

4. **Unsigned pickle models by default**: ML routing models (KNN, SVM, MLP)
   were loaded from pickle files without signature verification. A malicious
   pickle file could execute arbitrary code during deserialization.
   `LLMROUTER_ENFORCE_SIGNED_MODELS` defaulted to `false`.

5. **Admin auth bypass in production**: When no `ADMIN_API_KEYS` were
   configured, the admin auth check silently allowed all requests. This
   fail-open behavior meant a production deployment without explicit admin
   key configuration had no control-plane protection.

6. **Pickle loading enabled by default**: `LLMROUTER_ALLOW_PICKLE_MODELS`
   was implicitly `true`, allowing pickle deserialization even in environments
   where it wasn't needed.

## Decision

Implement six specific security hardening measures, all enabled by default:

### 1. Constant-Time API Key Comparison

Replace `candidate in valid_keys` with a constant-time comparison that
iterates ALL valid keys using `hmac.compare_digest()`:

```python
def _constant_time_contains(candidate: str, valid_keys: set[str]) -> bool:
    found = False
    for key in valid_keys:
        if _hmac_module.compare_digest(candidate.encode(), key.encode()):
            found = True
    return found
```

The function iterates all keys regardless of whether a match is found,
ensuring the execution time is proportional to the number of valid keys,
not to the position of the matching key.

### 2. Request ID Validation

Validate client-provided `X-Request-ID` headers against a strict regex:

```python
_VALID_REQUEST_ID = re.compile(r"^[a-zA-Z0-9\-_.]{1,128}$")
```

Request IDs that fail validation are silently replaced with a server-generated
UUID. This prevents log injection while maintaining backward compatibility for
legitimate request ID formats.

### 3. Policy Name Redaction

Policy denial responses no longer include the internal policy name. Instead:

```python
# Before: "denied by policy: block-gpt4-for-free-tier"
# After:  "Request denied by gateway policy"
```

The internal policy name is logged server-side at `INFO` level for debugging
but never included in client-facing error responses.

### 4. Enforce Signed Models by Default

The model artifact verification system (`model_artifacts.py`) now defaults to
requiring manifest verification:

- `LLMROUTER_ENFORCE_SIGNED_MODELS` defaults to `true` in production
- ML model loading verifies SHA-256 hash against a signed manifest
- Manifest must be signed with a trusted key
- Unsigned models are rejected unless explicitly allowed

### 5. Admin Auth Fail-Closed

When no `ADMIN_API_KEYS` are configured and `ADMIN_AUTH_ENABLED` is not
explicitly set to `"false"`, control-plane endpoints **deny all requests**
rather than allowing them:

```python
# No keys configured + auth not explicitly disabled = deny all
if not admin_keys and admin_auth_enabled:
    raise HTTPException(status_code=403, detail="No admin keys configured")
```

This fail-closed behavior forces operators to explicitly configure admin
keys or explicitly opt out of admin auth (for development environments only).

### 6. Pickle Security Environment Validation

At startup, `env_validation.py` checks for dangerous pickle-related
configuration:

- If `LLMROUTER_ALLOW_PICKLE_MODELS=true` and
  `LLMROUTER_ENFORCE_SIGNED_MODELS` is not `true`, emit a `WARNING`:
  "Pickle models enabled without signature enforcement. This allows
  arbitrary code execution from untrusted model files."

- `LLMROUTER_ALLOW_PICKLE_MODELS` defaults to `false`. Users must
  explicitly opt in to pickle loading.

## Consequences

### Positive

- **Timing attack resistance**: API key brute-forcing via timing analysis
  is no longer feasible. The constant-time comparison makes response times
  independent of key content.

- **Log integrity**: Request ID validation prevents log injection, ANSI
  escape injection, and log format string attacks. Log parsers can trust
  request IDs to be safe alphanumeric strings.

- **Information hiding**: Policy names are no longer leaked to external
  callers, preventing attackers from enumerating or reverse-engineering
  the policy structure.

- **Model supply chain security**: Enforced signature verification prevents
  malicious model files from executing code during deserialization.

- **Fail-closed by default**: Production deployments without explicit admin
  key configuration are protected rather than exposed. The explicit opt-out
  for development is intentional and documented.

- **Defense in depth**: The pickle warning ensures operators are aware of
  the risk when enabling pickle loading.

### Negative

- **Backward compatibility**: Existing deployments that relied on the
  fail-open admin auth behavior must now configure `ADMIN_API_KEYS` or
  explicitly set `ADMIN_AUTH_ENABLED=false`.

- **Development friction**: The fail-closed admin auth and signed model
  requirement add configuration steps for new development environments.
  Mitigated by documenting the development setup.

- **Performance overhead**: Constant-time key comparison iterates all
  valid keys for every auth check, adding O(n) cost where n is the number
  of configured admin keys. In practice, n < 10, so the overhead is
  negligible (~microseconds).

- **Request ID compatibility**: Clients sending request IDs with special
  characters (e.g., spaces, colons) will have their IDs silently replaced.
  This could break request correlation for clients using non-standard ID
  formats.

## Alternatives Considered

### Alternative A: Keep Fail-Open Admin Auth

Maintain the existing behavior where missing admin keys = allow all.

- **Pros**: Zero friction for new deployments; backward compatible.
- **Cons**: Silent security vulnerability in production; violates
  principle of least privilege; creates a class of deployment where the
  control plane is completely unprotected.
- **Rejected**: The security risk is too high. A single misconfigured
  deployment could expose the entire gateway control plane.

### Alternative B: Strict Request ID Rejection

Reject requests with invalid `X-Request-ID` headers (return 400).

- **Pros**: Forces clients to use valid request IDs; no silent replacement.
- **Cons**: Breaking change for clients sending non-standard IDs; adds
  friction; request ID is an operational concern, not a security boundary.
- **Rejected**: Silent replacement is less disruptive while still preventing
  log injection. The request ID is for correlation, not authentication.

### Alternative C: External Key Vault for Admin Keys

Store admin keys in AWS Secrets Manager / HashiCorp Vault instead of
environment variables.

- **Pros**: Better secret management; automatic rotation; audit trails.
- **Cons**: Adds external dependency; increases deployment complexity;
  doesn't address the other five security issues.
- **Rejected**: Orthogonal improvement. Can be added later. The current
  hardening addresses the immediate security gaps.

## References

- `src/litellm_llmrouter/auth.py` — Constant-time comparison, request ID validation
- `src/litellm_llmrouter/policy_engine.py` — Policy name redaction
- `src/litellm_llmrouter/model_artifacts.py` — Signed model verification
- `src/litellm_llmrouter/env_validation.py` — Startup security validation
- OWASP Timing Attack: https://owasp.org/www-community/attacks/Timing_attack
- CWE-208: Observable Timing Discrepancy
