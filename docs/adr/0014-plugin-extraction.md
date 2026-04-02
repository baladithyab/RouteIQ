# ADR-0014: Extract RouteIQ Value into Independently Installable Packages

**Status**: Accepted
**Date**: 2026-04-02
**Decision Makers**: RouteIQ Core Team

## Context

### Problem: All-or-Nothing Distribution

RouteIQ currently ships as a single package (`routeiq`) that includes all
functionality: ML routing, security, observability, plugins, resilience,
and the gateway composition root. Users who only want ML routing must
install the entire gateway.

This limits adoption scenarios:

1. **LiteLLM users who want better routing**: Must deploy the full RouteIQ
   gateway instead of just adding routing to their existing LiteLLM setup.
2. **Users who want security features**: Must install ML dependencies even
   if they only need the policy engine and SSRF protection.
3. **SDK users**: No Python SDK for interacting with RouteIQ programmatically.

### Current Package Structure

```
routeiq (single package)
+-- Routing: strategies.py, strategy_registry.py, centroid_routing.py, custom_routing_strategy.py
+-- Security: policy_engine.py, url_security.py, auth.py, rbac.py
+-- Observability: observability.py, telemetry_contracts.py, metrics.py
+-- Resilience: resilience.py, http_client_pool.py
+-- Plugins: gateway/plugins/*.py
+-- Gateway: gateway/app.py, startup.py, routes/*.py
```

## Decision

Extract RouteIQ's value into independently installable packages, each
working standalone with vanilla LiteLLM.

### Proposed Package Structure

| Package | Contents | Dependencies |
|---------|----------|--------------|
| `routeiq-routing` | ML strategies, centroid routing, A/B testing, routing profiles | litellm, numpy |
| `routeiq-security` | Policy engine, SSRF protection, admin auth, RBAC | fastapi, pydantic |
| `routeiq-observability` | OTel integration, telemetry contracts, metrics | opentelemetry-api |
| `routeiq-evaluator` | LLM-as-judge evaluation, skills gateway | litellm |
| `routeiq` | Gateway meta-package (all of the above + composition root) | All sub-packages |

### Usage Examples

```python
# Just ML routing with existing LiteLLM
pip install routeiq-routing

from routeiq_routing import install_routeiq_strategy
from litellm import Router

router = Router(model_list=[...])
install_routeiq_strategy(router, strategy_name="llmrouter-knn")
```

```python
# Just security with existing LiteLLM
pip install routeiq-security

from routeiq_security import PolicyMiddleware
app.add_middleware(PolicyMiddleware)
```

```python
# Full gateway (backwards compatible)
pip install routeiq[prod]
```

### Extraction Criteria

A module is eligible for extraction when:

1. It has no imports from other RouteIQ sub-packages (or only from its
   own sub-package)
2. It integrates with LiteLLM via official extension points
3. It has independent tests that don't require the full gateway
4. It provides value standalone

### Timeline

This is a phased extraction:

1. **Phase 1**: `routeiq-routing` (highest standalone value)
2. **Phase 2**: `routeiq-security` (clear boundary)
3. **Phase 3**: `routeiq-observability` (smallest, cleanest)
4. **Phase 4**: `routeiq-evaluator` (depends on routing telemetry)

## Consequences

### Positive

- **Wider adoption**: LiteLLM users can add RouteIQ features incrementally.
- **Smaller install footprint**: Users install only what they need.
- **Independent versioning**: Routing bugs can be fixed without releasing
  the entire gateway.
- **Clearer API boundaries**: Package extraction forces clean interfaces.

### Negative

- **Release complexity**: Multiple packages to version, test, and publish.
- **Cross-package changes**: Features spanning packages require coordinated
  releases.
- **Monorepo vs multi-repo**: Need to decide on repository structure.

## Alternatives Considered

### Alternative A: Keep Single Package with Extras

- **Pros**: Simpler release process; current approach works.
- **Cons**: Users can't add just routing to their LiteLLM setup.
- **Status quo**: See [ADR-0007](0007-dependency-tiering.md). Extras are
  the current intermediate step.

### Alternative B: Publish as LiteLLM Plugin

- **Pros**: Leverages LiteLLM's plugin ecosystem.
- **Cons**: LiteLLM doesn't have a formal plugin registry; distribution
  is still via pip.
- **Complementary**: The packages can be advertised as LiteLLM plugins.

## References

- `src/litellm_llmrouter/` — Current monolithic source
- [ADR-0001: Three-Layer Architecture](0001-three-layer-architecture.md)
- [ADR-0007: Dependency Tiering](0007-dependency-tiering.md)
- [TG3 Rearchitecture Proposal](../architecture/tg3-rearchitecture-proposal.md)
