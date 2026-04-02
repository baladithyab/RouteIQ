# ADR-0017: Leverage LiteLLM Upstream Capabilities Rather Than Reimplement

**Status**: Accepted
**Date**: 2026-04-02
**Decision Makers**: RouteIQ Core Team

## Context

### Problem: Redundant Reimplementation

Over the course of RouteIQ's development, several subsystems were built that
duplicate capabilities LiteLLM already provides natively. This happened for
understandable reasons — LiteLLM didn't always have these features, or the
features weren't discovered during initial development — but the result is a
codebase where ~61% of code is either redundant with upstream or infrastructure
scaffolding that provides no marginal value.

The [Deep Architecture Review v0.2.0](../architecture/deep-review-v0.2.0.md)
mapped capabilities across LiteLLM, RouteIQ, and industry best practices,
revealing significant overlap.

### Redundancy Map

| Capability | LiteLLM Provides | RouteIQ Reimplemented | Lines Wasted |
|------------|:-:|:-:|------:|
| MCP Gateway (JSON-RPC, SSE, REST, OAuth) | Yes (v1.81+) | Yes | ~3,379 |
| A2A Gateway (agent registry, cards) | Yes | Yes (thin wrapper) | ~1,620 |
| Auth (virtual keys, JWT, SSO) | Yes | Yes (parallel system) | ~534 |
| Team/Org/Key management | Yes (full CRUD + UI) | No (uses LiteLLM's) | 0 |
| Budget management | Yes (per-key, per-team) | Yes (quota.py) | ~800 |
| Caching | Yes (11 backends) | Yes (cache_plugin.py) | ~400 |
| Rate limiting | Yes (TPM/RPM per key) | Yes (quota.py) | (shared) |
| Callback integrations | Yes (60+ providers) | Yes (plugin bridge) | ~300 |
| Guardrails | Yes (CustomGuardrail base) | Yes (5 plugins) | ~1,200 |
| Cost tracking | Yes (SpendLogs + aggregation) | Yes (cost_tracker.py) | ~300 |

### What RouteIQ Genuinely Adds (~14,000 lines)

Not everything is redundant. RouteIQ provides genuine value in these areas:

| Capability | Lines | Upstream Alternative |
|------------|------:|---------------------|
| ML routing (18 strategies) | ~5,500 | None — LiteLLM has basic routing only |
| Centroid zero-config routing | ~1,113 | None |
| Plugin system (13 plugins, lifecycle) | ~1,242 | LiteLLM has CustomLogger callbacks |
| SSRF protection (dual validation) | ~600 | None |
| OPA-style policy engine | ~993 | None |
| Backpressure + drain manager | ~800 | None |
| Model artifact verification | ~1,484 | None |
| Versioned telemetry contracts | ~500 | None |
| Router decision OTel attributes | ~400 | None |
| Evaluator (LLM-as-judge) | ~600 | None |

These are the capabilities worth maintaining and investing in.

## Decision

Adopt a clear principle: **Use LiteLLM for capabilities it provides natively.
RouteIQ invests only in genuine differentiators.**

### What LiteLLM Handles (RouteIQ Defers)

| Domain | LiteLLM Feature | RouteIQ Role |
|--------|----------------|-------------|
| **MCP Protocol** | Native JSON-RPC, SSE, REST, OAuth, management CRUD | SSRF validation wrapper, OTel instrumentation, audit logging |
| **A2A Protocol** | `global_agent_registry`, agent cards, task management | OTel instrumentation only |
| **Auth & Identity** | Virtual keys, JWT, SSO, team/org/user management | Admin key hardening, RBAC extensions, OIDC bridge |
| **Budget Management** | Per-key/per-team budgets, spend tracking, 7 aggregation tables | Cost-aware routing strategy (reads LiteLLM's data) |
| **Caching** | 11 cache backends (Redis, S3, Qdrant, etc.) | Cache plugin wraps LiteLLM's backend selection |
| **Rate Limiting** | TPM/RPM per key, global limits | Token-aware rate limiting extensions (proposed) |
| **Callbacks** | 60+ integrations (Langfuse, Datadog, Prometheus, etc.) | Plugin callback bridge for RouteIQ-specific hooks |
| **Guardrails** | `CustomGuardrail` base class, guardrails table | PII, content filter, prompt injection as RouteIQ plugins |
| **Admin UI** | Next.js dashboard (user/team/key/model management) | RouteIQ routing dashboard, policy viewer (extensions) |

### What RouteIQ Keeps and Invests In

| Domain | RouteIQ Feature | Why Not Upstream |
|--------|----------------|------------------|
| **ML Routing** | 18 strategies, A/B testing, centroid, profiles | Core differentiator. LiteLLM has only basic routing. |
| **Security** | SSRF dual-validation, policy engine | Not available upstream. Enterprise requirement. |
| **Resilience** | Backpressure, drain manager, circuit breakers | LiteLLM has basic retry/fallback only. |
| **Observability** | Versioned telemetry contracts, decision spans | Structured telemetry beyond LiteLLM's ad-hoc spans. |
| **Model Artifacts** | Hash + signature verification | ML model supply chain security. |
| **Evaluator** | LLM-as-judge, skills gateway | Quality evaluation pipeline. |
| **Plugin System** | 11 capabilities, dependency resolution, security | More powerful than LiteLLM's CustomLogger. |

### Integration Pattern

When RouteIQ enhances a LiteLLM-native capability, it does so via **wrapping**,
not **reimplementation**:

```python
# WRONG: Reimplement MCP JSON-RPC handler
@router.post("/mcp")
async def handle_mcp_jsonrpc(request: Request):
    # 868 lines of JSON-RPC handling...

# RIGHT: Wrap LiteLLM's MCP with SSRF + audit
@router.post("/llmrouter/mcp/servers")
async def register_mcp_server(request: Request):
    # Validate URL (SSRF)
    validated_url = await validate_outbound_url_async(server.url)
    # Delegate to LiteLLM's native registration
    await litellm_mcp_register(validated_url)
    # Audit log
    await audit_log("mcp_server_registered", ...)
```

### Deletion Decisions

Modules that were fully redundant have been or will be deleted:

- `mcp_parity.py` (1,072 lines) — Deleted ([ADR-0003](0003-delete-redundant-mcp.md))
- `mcp_jsonrpc.py` (868 lines) — Deleted ([ADR-0003](0003-delete-redundant-mcp.md))
- `mcp_sse_transport.py` (1,439 lines) — Deleted ([ADR-0003](0003-delete-redundant-mcp.md))
- `a2a_gateway.py` thin wrappers — Scheduled for simplification
- `quota.py` budget tracking — Defer to LiteLLM's budget tables

## Consequences

### Positive

- **Reduced maintenance**: ~7,578+ lines of redundant code removed or scheduled
  for removal. Each removed line is one fewer line to maintain, test, and secure.

- **Automatic upstream improvements**: When LiteLLM improves MCP, A2A, auth,
  caching, or callbacks, RouteIQ benefits automatically without code changes.

- **Faster feature velocity**: Engineering effort focuses on ML routing, security,
  and observability instead of reimplementing commodity features.

- **Smaller attack surface**: Less custom code means fewer potential
  vulnerabilities unique to RouteIQ.

- **Clearer value proposition**: "RouteIQ adds intelligent routing, enterprise
  security, and advanced observability to LiteLLM" is a clearer pitch than
  "RouteIQ is another LLM gateway."

### Negative

- **Upstream dependency**: RouteIQ's capabilities are bounded by what LiteLLM
  provides. If LiteLLM regresses or changes APIs, RouteIQ is affected.
  Mitigated by pinning LiteLLM versions (`>=1.81.3,<1.82.0`).

- **Feature request latency**: If users need a capability that LiteLLM doesn't
  provide, RouteIQ must either wait for upstream or build it. This is preferable
  to building everything from scratch.

- **Testing complexity**: Integration tests must verify that RouteIQ's wrappers
  work correctly with specific LiteLLM versions. Version pin changes require
  re-validation.

- **Breaking changes on upgrade**: LiteLLM's internal APIs may change between
  minor versions. The version pin mitigates this but requires active tracking
  of upstream releases.

## Alternatives Considered

### Alternative A: Build Everything

Continue building all capabilities in RouteIQ, regardless of upstream overlap.

- **Pros**: Full control; no upstream dependency; can innovate faster in
  specific areas.
- **Cons**: Unsustainable maintenance cost; 50-100 upstream commits/week to
  track; perpetuates the 3.05/5 goal alignment score; diverts engineering
  from genuine differentiators.
- **Rejected**: The maintenance cost of reimplementing commodity features
  (MCP, A2A, caching, auth) far exceeds the value. LiteLLM is well-funded,
  actively maintained, and improving rapidly.

### Alternative B: Fork LiteLLM

Fork LiteLLM and merge RouteIQ features directly into the fork.

- **Pros**: Full control; no integration layer needed.
- **Cons**: 50-100 upstream commits/week to merge; inevitable divergence;
  loss of community contributions; massive ongoing cost.
- **Rejected**: See [ADR-0001](0001-three-layer-architecture.md) —
  Fork scored 2.95/5 in the TG3 evaluation.

### Alternative C: Selective Reimplementation

Keep RouteIQ's implementations where they're "better" than LiteLLM's.

- **Pros**: Best-of-both-worlds for each feature.
- **Cons**: "Better" is subjective and temporary. LiteLLM improves rapidly.
  Maintaining parallel implementations creates confusion about which endpoint
  to use (see MCP URL proliferation in [ADR-0003](0003-delete-redundant-mcp.md)).
- **Rejected**: The maintenance cost of parallel implementations exceeds
  any marginal quality improvement.

## References

- [Deep Architecture Review v0.2.0](../architecture/deep-review-v0.2.0.md)
- [TG3 Alternative Patterns Analysis](../architecture/tg3-alternative-patterns.md)
- [TG3 Rearchitecture Proposal](../architecture/tg3-rearchitecture-proposal.md)
- [ADR-0001: Three-Layer Architecture](0001-three-layer-architecture.md)
- [ADR-0003: Delete Redundant MCP](0003-delete-redundant-mcp.md)
- LiteLLM repository: https://github.com/BerriAI/litellm
