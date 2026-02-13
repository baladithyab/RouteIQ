# RouteIQ Deep Architecture Review — v0.2.0 Planning

**Date:** 2026-02-13
**Scope:** LiteLLM capabilities, RouteIQ enhancements, industry patterns, plugin extensibility

---

## 1. Capability Mapping: LiteLLM vs RouteIQ vs Industry

### Storage & Persistence

| Capability | LiteLLM (inherited) | RouteIQ (added) | Industry Best Practice |
|-----------|---------------------|-----------------|----------------------|
| Model config | YAML + DB (`LiteLLM_ProxyModelTable`) | S3/GCS config sync with ETag caching | Config in S3/GCS with GitOps (Kong, Portkey) |
| Key management | 44+ field `VerificationToken` table, rotation, budgets, TTL | Admin key auth layer + RBAC permissions | Virtual keys with team/org hierarchy (Azure APIM, Portkey) |
| Spend tracking | Per-request `SpendLogs` + 7 daily aggregation tables | Cost-aware routing strategy, cost-per-token OTel metric | Token-based metering with budget alerts (Cloudflare, Azure) |
| Credential storage | `CredentialsTable` with encryption, Vault/AWS SM integration | Not yet leveraged | Secrets Manager with rotation (Kong, Azure) |
| MCP servers | `MCPServerTable` in DB with health checks | 5 MCP surfaces (JSON-RPC, SSE, REST, parity, proxy) | DB-backed with health monitoring (LiteLLM native) |
| A2A agents | `AgentsTable` in DB with card_params | A2A gateway wrapper, OTel tracing | Emerging standard |
| Guardrails | `GuardrailsTable` in DB, content filter hooks | Plugin-based guardrails (content filter, PII, LlamaGuard, Bedrock, prompt injection) | Content safety + DLP scanning (Cloudflare, Azure, Kong) |
| Audit trail | `AuditLog` table (who changed what) | `audit.py` with fail-closed mode | Audit logging with tamper-proof storage |

**Gap:** RouteIQ does not fully leverage LiteLLM's DB-backed credential management. Provider API keys are in config YAML, not in the encrypted `CredentialsTable`. This should be the v0.2.0 priority.

### Observability

| Capability | LiteLLM | RouteIQ | Industry |
|-----------|---------|---------|----------|
| Tracing | OTel spans for LLM calls | Router decision spans (TG4.1) with 12 attributes, MCP tracing, A2A tracing | OTel GenAI semantic conventions (CNCF) |
| Metrics | Prometheus counters/histograms, request latency | Config sync age gauge, cost-per-token histogram, model health summary | Token usage, cost, latency p50/p95/p99 (Cloudflare, Portkey) |
| Logging | Structured logging with `_service_logger`, `ErrorLogs` table | Secret scrubbing in error logs | Structured JSON with trace correlation |
| Telemetry contracts | None (ad hoc span attributes) | Versioned `routeiq.router_decision.v1` contract with GenAI semantic conventions | Versioned telemetry schemas (Portkey, Datadog) |
| Callback system | 20+ integrations (Langfuse, Datadog, Weave, Arize, Prometheus) | Plugin callback bridge (`on_llm_pre_call`, `on_llm_success`, `on_llm_failure`) | Webhook-based callbacks (Kong, Cloudflare) |

**Gap:** Missing per-model latency histograms (TTFB, streaming, total), and a CloudWatch/Grafana dashboard template.

### Resilience

| Capability | LiteLLM | RouteIQ | Industry |
|-----------|---------|---------|----------|
| Load balancing | Router with retry, fallback, cooldown | 18+ ML strategies + cost-aware + A/B testing | Weighted round-robin + semantic routing (Kong) |
| Circuit breakers | Per-provider cooldown | Per-provider CB (CLOSED/OPEN/HALF_OPEN) + per-service CB (DB, Redis) | Per-backend circuit breakers (Envoy, Kong) |
| Backpressure | None built-in | ASGI-level concurrency limiter with 503 shedding | Concurrency limits + queue depth (Envoy) |
| Graceful shutdown | Basic signal handling | Drain manager with active request tracking + timeout | Connection draining (ALB, Envoy, K8s) |
| Rate limiting | Per-key TPM/RPM via DB | Redis-backed quota enforcement (request, token, spend) with Lua scripts | Token-based rate limiting (Azure APIM, Cloudflare) |

**Strength:** RouteIQ's resilience layer is significantly more advanced than base LiteLLM and competitive with commercial gateways.

### Security

| Capability | LiteLLM | RouteIQ | Industry |
|-----------|---------|---------|----------|
| Auth | Virtual keys + JWT + OAuth2 + SSO | Two-tier (admin + user) + RBAC with 5 permissions | OIDC + RBAC + API keys (Kong, Azure) |
| SSRF protection | None | URL validation at registration + invocation (DNS rebinding defense) | SSRF prevention (OWASP) |
| Policy engine | None | OPA-style pre-request evaluation (ASGI layer, fail-open/fail-closed) | OPA/Rego policies (Kong, Envoy) |
| Secret scrubbing | Basic | Error response sanitization | Standard practice |
| Model artifact verification | None | Hash + signature verification for ML models | Model provenance verification (emerging) |

**Strength:** RouteIQ's security posture (SSRF, policy engine, model verification) exceeds LiteLLM and most commercial alternatives.

---

## 2. Plugin Architecture Assessment

### Current State (v0.1.0)

RouteIQ has a **production-grade plugin system** with:

- **11 capabilities**: ROUTES, MIDDLEWARE, ROUTING_STRATEGY, EVALUATOR, OBSERVABILITY_EXPORTER, TOOL_RUNTIME, AUTH_PROVIDER, STORAGE_BACKEND, GUARDRAIL, CACHE, COST_TRACKER
- **12 lifecycle hooks**: startup, shutdown, health_check, on_request, on_response, on_llm_pre_call, on_llm_success, on_llm_failure, on_config_reload, on_route_register, on_model_health_change, on_circuit_breaker_change
- **Security policies**: Allowlist, capability restrictions, SSRF prevention
- **Dependency resolution**: Topological sort with priority tie-breaking
- **Failure modes**: continue, abort, quarantine
- **12 built-in plugins**: evaluator, skills_discovery, upskill_evaluator, content_filter, pii_guard, prompt_injection_guard, bedrock_guardrails, llamaguard, cache, cost_tracker, guardrails_base

### What's Needed for a Bedrock AgentCore MCP Connector

1. **MCP server registration hook** — No `register_mcp_server()` in PluginContext. The MCP gateway has its own server registry not exposed to plugins.
2. **Credential passthrough** — Plugin needs IAM roles for AgentCore. LiteLLM's `CredentialsTable` exists but PluginContext doesn't expose credential retrieval.
3. **Health check integration** — Plugin-registered MCP servers aren't in the health check loop.

### What's Needed for a Custom MCP Gateway Connector

1. **Transport abstraction** — Plugins need to register MCP servers with different transports without reimplementing the transport layer.
2. **Tool forwarding** — When tool calls come through RouteIQ's MCP surface, plugin needs to intercept and forward. Currently requires middleware interception.
3. **Schema aggregation** — Multiple connectors need tool schemas merged into `/mcp/tools/list`.

### Recommended PluginContext Extensions (v0.2.0)

```python
class PluginContext:
    # Existing
    logger: Logger
    settings: dict
    validate_outbound_url: Callable

    # NEW: MCP server registration
    register_mcp_server: Callable[[str, MCPServerConfig], None]
    unregister_mcp_server: Callable[[str], None]

    # NEW: Credential access (from LiteLLM's CredentialsTable)
    get_credential: Callable[[str], Optional[dict]]

    # NEW: Model registry access (read-only)
    list_models: Callable[[], list[dict]]

    # NEW: Routing strategy registration
    register_strategy: Callable[[str, RoutingStrategy], None]
```

---

## 3. Recommendations for v0.2.0

### Priority 1: Leverage LiteLLM's DB-Backed Features

| Feature | Status | Action |
|---------|--------|--------|
| `store_model_in_db` | Works (tested) | Document, make default for production |
| Credential management (`CredentialsTable`) | Not used | Wire into config loader, add credential rotation |
| Daily spend aggregations | Available via LiteLLM | Expose via `/admin/spend/daily` endpoint |
| Team management | Available via LiteLLM | Document team-based access control |
| MCP server DB storage | Available via LiteLLM | Verify RouteIQ MCP surfaces use DB-backed storage |
| Guardrails DB storage | Available via LiteLLM | Verify RouteIQ guardrail plugins sync with DB |

### Priority 2: Expose Plugin Extension Points for MCP/A2A

- `register_mcp_server()` / `unregister_mcp_server()` in PluginContext
- `get_credential()` for accessing provider credentials
- `register_strategy()` for custom routing strategies
- `list_models()` for model registry access

### Priority 3: Observability Improvements

- Align with OTel GenAI semantic conventions (`gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.*`)
- Add per-model latency histograms (TTFB, total)
- Create CloudWatch dashboard template
- Add cost attribution dashboards (per-team, per-model, per-key)

### Priority 4: Storage Backend Flexibility

Allow users to choose storage backends via config:

```yaml
storage:
  config_backend: s3  # or: db, gcs, local
  credential_backend: secrets_manager  # or: db, vault, env
  spend_backend: db  # or: cloudwatch, datadog
  cache_backend: redis  # or: elasticache, memcached, in_memory
```

### Priority 5: Semantic Caching

RouteIQ has a `semantic_cache.py` module. Industry leaders (Azure APIM, Kong, Cloudflare) all offer semantic caching. Verify and harden the existing implementation.

---

## 4. Industry Comparison Matrix

| Feature | RouteIQ | LiteLLM | Kong AI | Cloudflare AI | Azure APIM | Portkey |
|---------|---------|---------|---------|--------------|------------|---------|
| ML routing | 18+ strategies | simple-shuffle | semantic routing | round-robin | weighted | latency-based |
| Plugin system | 11 capabilities | callbacks only | Lua plugins | none | XML policies | none |
| MCP support | 5 surfaces | native /mcp | none | none | none | none |
| A2A support | gateway + tracing | DB-backed | none | none | none | none |
| Circuit breakers | per-provider + per-service | per-provider cooldown | per-backend | auto-retry | per-backend | auto-retry |
| Policy engine | OPA-style ASGI | none | OPA plugin | none | XML policies | none |
| Cost tracking | per-request + daily agg | per-request | per-route | per-request | per-subscription | per-request |
| Quota enforcement | Redis-backed, multi-dim | DB-backed TPM/RPM | plugin-based | per-gateway | per-subscription | per-workspace |
| Guardrails | 5 plugin types | content filter | prompt guard | DLP | content safety | guardrails |
| Secret management | YAML (gap) | Vault/SM | Kong Vault | Secret Store | Key Vault | Vault |
| Open source | yes | yes (core) | yes (core) | no | no | yes |

**RouteIQ's unique strengths:** ML-based routing (18+ strategies), MCP/A2A protocol support, plugin extensibility (11 capabilities, 12 hooks), OPA-style policy engine, model artifact verification. These are genuine differentiators vs. all competitors.

**RouteIQ's gaps vs. industry:** Credential management (should use LiteLLM's existing DB-backed system), visual dashboard (no UI beyond LiteLLM's built-in), and PluginContext extension points for MCP/A2A connectors.
