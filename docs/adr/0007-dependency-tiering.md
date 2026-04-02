# ADR-0007: Tier Dependencies into Optional Extras

**Status**: Accepted
**Date**: 2026-04-02
**Decision Makers**: RouteIQ Core Team

## Context

### Problem: Monolithic Dependency Tree

RouteIQ v0.1.0 installed all dependencies unconditionally via a single
`dependencies` list in `pyproject.toml`. This included:

- **boto3** (~30MB installed): Required only for S3 config sync, but installed
  on every deployment regardless of whether S3 was used.
- **a2a-sdk**: Required only when A2A gateway is enabled
  (`A2A_GATEWAY_ENABLED=true`), which defaults to `false`.
- **sentence-transformers** + **PyTorch** (~1.5GB): Required only for KNN
  routing strategy, but installed on every deployment.
- **watchdog**: Required only for hot-reload (`CONFIG_HOT_RELOAD=true`).
- **authlib**: Required only for OIDC authentication.
- **langfuse**: Required only for Langfuse callback integration.
- **asyncpg** + **prisma**: Required only for database persistence.

The monolithic dependency tree caused:

1. **Bloated Docker images**: The production image was 2-4GB due to PyTorch
   and sentence-transformers, even for deployments that only used centroid
   routing (which needs no ML dependencies).

2. **Slow installs**: `uv sync` took 3-5 minutes due to PyTorch wheel
   downloads, even for developers not working on ML routing.

3. **Dependency conflicts**: boto3's strict pinning conflicted with other
   AWS SDK users. PyTorch's CUDA variants added complexity for GPU vs CPU
   deployments.

4. **Attack surface**: Every installed dependency is a potential vulnerability.
   Installing 30MB of boto3 when S3 isn't used adds unnecessary CVE exposure.

## Decision

Tier dependencies into a minimal core plus optional extras in `pyproject.toml`.

### Core Dependencies (Always Installed)

The `dependencies` list contains only what's needed for the basic gateway
to start and serve requests:

```toml
dependencies = [
    # Core framework
    "fastapi>=0.109.0",
    "pydantic>=2.5.0",
    "httpx>=0.26.0",
    # LiteLLM core
    "litellm>=1.81.3,<1.82.0",
    # Required by LiteLLM proxy
    "apscheduler>=3.10.0",
    "email-validator>=2.0.0",
    "fastapi-sso>=0.16.0",
    "websockets>=15.0.0",
    "backoff>=2.0.0",
    "redis>=5.0.0",
    # Configuration
    "pyyaml>=6.0",
    "aiofiles>=23.0.0",
    # Observability core
    "prometheus-client>=0.24.1",
    "opentelemetry-api>=1.22.0",
    "opentelemetry-sdk>=1.22.0",
    "opentelemetry-exporter-otlp>=1.22.0",
    "opentelemetry-instrumentation>=0.43b0",
    "opentelemetry-instrumentation-logging>=0.43b0",
    "python-multipart>=0.0.22",
]
```

### Optional Extras

| Extra | Dependencies | Use Case |
|-------|-------------|----------|
| `[db]` | asyncpg, prisma | PostgreSQL persistence |
| `[otel]` | OTLP gRPC/HTTP exporters, FastAPI/httpx instrumentation | Full observability pipeline |
| `[cloud]` | boto3, google-cloud-aiplatform, azure-identity | Cloud provider SDKs |
| `[a2a]` | a2a-sdk | Agent-to-Agent protocol |
| `[hotreload]` | watchdog | Config file watching |
| `[oidc]` | authlib | OIDC/SSO authentication |
| `[callbacks]` | langfuse | Callback integrations |
| `[knn]` | sentence-transformers, scikit-learn | KNN routing inference |
| `[dev]` | pytest, pytest-asyncio, hypothesis, ruff, mypy | Development tools |
| `[prod]` | All of the above (except dev) | Production meta-extra |
| `[all]` | Everything including dev | Complete installation |

### Graceful Degradation Pattern

Modules that depend on optional extras use try/except imports with
fallback behavior:

```python
# centroid_routing.py
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# custom_routing_strategy.py
try:
    from litellm_llmrouter.centroid_routing import (
        get_centroid_strategy,
        warmup_centroid_classifier,
    )
    CENTROID_AVAILABLE = True
except ImportError:
    CENTROID_AVAILABLE = False
```

When an optional dependency is missing, the feature is silently disabled
rather than crashing at import time. This allows the gateway to start
with reduced functionality.

### Docker Build Integration

The `docker/Dockerfile` uses build ARGs to control which extras are
installed:

```dockerfile
ARG ROUTEIQ_EXTRAS="prod"
RUN uv sync --extra ${ROUTEIQ_EXTRAS}
```

This enables multi-tier images (see [ADR-0009](0009-multi-tier-docker-images.md)).

## Consequences

### Positive

- **Slim images possible**: A deployment using only centroid routing can
  skip `[knn]` (saves ~1.5GB of PyTorch/sentence-transformers), `[cloud]`
  (saves ~30MB of boto3), etc. Slim image target: ~500MB.

- **Faster development iteration**: `uv sync` with just core deps completes
  in <30s. Adding `--extra dev` for testing is still fast.

- **Reduced attack surface**: Each skipped extra removes potential CVEs.
  A deployment without `[cloud]` has no boto3 vulnerabilities to track.

- **Clear feature boundaries**: The extras make it obvious which features
  require which dependencies. Documentation can reference extras directly.

- **Dependency conflict isolation**: boto3 pinning conflicts only affect
  deployments that install `[cloud]`. Other users are unaffected.

### Negative

- **Runtime errors for missing deps**: If a user enables a feature (e.g.,
  `CONFIG_S3_BUCKET`) without installing the corresponding extra (`[cloud]`),
  they get an import error at runtime rather than a helpful message.
  Mitigated by `env_validation.py` checking for missing extras at startup.

- **Documentation complexity**: Users must understand which extras they need.
  The `[prod]` meta-extra simplifies this for production deployments.

- **Testing matrix expansion**: CI must test with different extra combinations
  to ensure graceful degradation works correctly. Currently tested: core-only,
  `[dev]`, `[prod]`, `[all]`.

- **pyproject.toml complexity**: The optional-dependencies section adds
  maintenance burden when adding new features that need new dependencies.

## Alternatives Considered

### Alternative A: Monolithic Dependencies (Status Quo)

Keep all dependencies in the core `dependencies` list.

- **Pros**: Simplest; no user confusion about extras; guaranteed all
  features work.
- **Cons**: 2-4GB Docker images; slow installs; unnecessary attack surface;
  dependency conflicts.
- **Rejected**: The image size and install time are unacceptable for
  production use.

### Alternative B: Separate Packages

Split RouteIQ into separate pip packages (`routeiq-core`, `routeiq-routing`,
`routeiq-security`, etc.) with their own dependency trees.

- **Pros**: Cleanest separation; users install only what they need;
  independent versioning.
- **Cons**: Premature for current project maturity; adds release complexity;
  requires careful API design across packages.
- **Deferred**: This is the long-term goal (see [ADR-0014](0014-plugin-extraction.md)).
  Optional extras are the intermediate step.

### Alternative C: Conditional Requirements via Markers

Use PEP 508 environment markers to conditionally install dependencies.

- **Pros**: Automatic; no user action needed.
- **Cons**: Markers are based on platform/Python version, not feature flags.
  There's no way to express "install boto3 only if the user wants S3 support"
  via markers.
- **Rejected**: PEP 508 markers don't support feature-based conditioning.

## References

- `pyproject.toml` — Dependency definitions
- `docker/Dockerfile` — Build ARG integration
- `src/litellm_llmrouter/env_validation.py` — Missing extra detection
- [ADR-0009: Multi-Tier Docker Images](0009-multi-tier-docker-images.md)
- [ADR-0014: Plugin Extraction](0014-plugin-extraction.md)
- PEP 508: https://peps.python.org/pep-0508/
