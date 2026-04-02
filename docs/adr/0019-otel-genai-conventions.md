# ADR-0019: Adopt OpenTelemetry GenAI Semantic Conventions

**Status**: Accepted
**Date**: 2026-04-02
**Decision Makers**: RouteIQ Core Team

## Context

OpenTelemetry has published experimental semantic conventions for Generative AI
(gen_ai.*) that standardize how AI applications emit telemetry. Major competitors
(Portkey, Datadog, Arize, W&B Weave) are adopting these conventions. Non-compliance
will make RouteIQ traces incompatible with standard observability tooling.

Key convention namespaces:
- gen_ai.system — provider identification
- gen_ai.request.* — model, temperature, max_tokens
- gen_ai.response.* — finish_reason, id
- gen_ai.usage.* — input_tokens, output_tokens
- gen_ai.token.* — token-level events
- gen_ai.agent.* — agent name, description
- gen_ai.tool.* — tool invocations (including MCP)

## Decision

Align RouteIQ's telemetry emission (observability.py, telemetry_contracts.py,
router_decision_callback.py, mcp_tracing.py, a2a_tracing.py) with the OTel
GenAI semantic conventions.

## Consequences

### Positive
- RouteIQ traces work natively with Datadog, Grafana, Jaeger GenAI views
- Standard attribute names reduce learning curve
- MCP and A2A traces use standard conventions

### Negative
- Migration effort across telemetry modules
- Must maintain backward compatibility during transition period

## References
- https://opentelemetry.io/docs/specs/semconv/gen-ai/
- https://opentelemetry.io/blog/2025/ai-agent-observability/
