# ADR-0016: Developer Experience Features (Dashboard, Playground, SDK)

**Status**: Proposed
**Date**: 2026-04-02
**Decision Makers**: RouteIQ Core Team

## Context

### Problem: No Self-Service Developer Experience

RouteIQ provides powerful backend capabilities (ML routing, policy engine,
observability) but lacks developer-facing features that competitors offer:

| Feature | RouteIQ | Portkey | Helicone | OpenRouter |
|---------|:-------:|:-------:|:--------:|:----------:|
| Analytics dashboard | None | Yes | Yes | Yes |
| Playground / chat UI | None | Yes | Yes | Yes |
| Python SDK | None | Yes | Yes | Yes |
| Prompt management | None | Yes | Yes | No |
| Token-aware rate limiting | Partial | Yes | No | Yes |
| Webhook events | None | Yes | Yes | No |

This gap means:

1. **No visibility**: Users can't see routing decisions, cost trends, or
   latency distributions without querying OTel directly.
2. **No experimentation**: Testing different models/prompts requires writing
   curl commands against the API.
3. **No programmatic integration**: Users interact via raw HTTP, not a
   typed SDK with error handling and retry logic.

### LiteLLM's Admin UI

LiteLLM ships a Next.js admin dashboard that provides user/team/key
management, spend tracking, and model configuration. RouteIQ currently
does not expose this UI (`ROUTEIQ_ADMIN_UI_ENABLED` defaults to `false`).

## Decision

Implement developer experience features in phases, building on LiteLLM's
existing UI and adding RouteIQ-specific capabilities.

### Phase 1: Admin Dashboard

Enable and extend LiteLLM's admin UI:

- Expose at `/ui/` when `ROUTEIQ_ADMIN_UI_ENABLED=true`
- Add RouteIQ-specific pages:
  - **Routing dashboard**: Active strategy, A/B test weights, centroid
    classification distribution, routing profile
  - **Policy viewer**: Active policies, recent denials, policy evaluation
    metrics
  - **Plugin status**: Loaded plugins, health, quarantined plugins

### Phase 2: Playground / Chat UI

Interactive chat interface for testing models:

- Model selection dropdown (filtered by routing profile)
- Side-by-side model comparison
- Routing decision visualization (which model was selected and why)
- Cost/latency display per message
- Conversation history with model annotations

### Phase 3: Python SDK

```python
from routeiq import RouteIQClient

client = RouteIQClient(
    base_url="https://gateway.example.com",
    api_key="sk-riq-...",
)

# Chat completion (routed by RouteIQ)
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Hello"}],
    routing_profile="eco",  # RouteIQ-specific
)

# Routing decision metadata
print(response.routeiq.strategy)        # "centroid"
print(response.routeiq.model_selected)  # "gpt-4o-mini"
print(response.routeiq.tier)            # 2
print(response.routeiq.cost)            # 0.00015
```

The SDK is OpenAI-compatible with RouteIQ extensions in a `.routeiq`
namespace.

### Phase 4: Prompt Management

- Named prompt templates stored in database
- Version history with diff view
- A/B testing prompts across models
- Evaluation scores per prompt version

### Phase 5: Token-Aware Rate Limiting

- Pre-estimate token count before routing
- Enforce TPM (tokens per minute) limits
- Burst allowance with sliding window
- Cost-based limiting ($/minute per team)

### Phase 6: Webhook Events

```yaml
webhooks:
  - url: https://slack.example.com/webhook
    events: [budget_exceeded, model_degraded, policy_denied]
  - url: https://pagerduty.example.com/webhook
    events: [circuit_breaker_opened, leader_changed]
```

## Consequences

### Positive

- **Competitive parity**: Matches the feature set of commercial gateways.
- **Self-service**: Teams can manage keys, test models, and monitor costs
  without admin intervention.
- **Adoption driver**: Rich UX reduces barrier to evaluation and adoption.
- **Typed integration**: SDK provides better DX than raw HTTP.

### Negative

- **Frontend maintenance**: Dashboard and playground require JavaScript/
  TypeScript maintenance alongside the Python backend.
- **Scope expansion**: Significant engineering effort across 6 phases.
- **Security surface**: Web UI adds XSS, CSRF, and session management
  concerns.

## Alternatives Considered

### Alternative A: No UI, API Only

- **Pros**: Simpler; less maintenance; pure infrastructure product.
- **Cons**: Poor developer experience; can't compete with commercial
  gateways; harder to evaluate.
- **Rejected**: Developer experience is a key differentiator.

### Alternative B: Grafana Dashboards Only

- **Pros**: Leverages existing OTel data; no custom UI code.
- **Cons**: Requires Grafana deployment; no interactive features
  (playground, key management); read-only.
- **Complementary**: Grafana dashboards are valuable alongside the
  built-in UI, not as a replacement.

## References

- `src/litellm_llmrouter/gateway/app.py` — Admin UI static file serving
- [ADR-0008: OIDC Identity Integration](0008-oidc-identity-integration.md)
- [TG3 Admin UI Design](../architecture/tg3-admin-ui-design.md)
- LiteLLM Admin UI: `reference/litellm/ui/`
