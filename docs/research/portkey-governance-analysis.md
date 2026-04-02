# Portkey AI Governance System: Deep Research Analysis

**Date:** 2026-04-02
**Purpose:** Competitive intelligence for RouteIQ governance layer design
**Sources:** Portkey docs, blog posts, press releases, third-party comparisons (NO source code reviewed)

---

## Executive Summary

On March 24, 2026, Portkey fully open-sourced their "Production Gateway" — previously
a SaaS-only product — including governance, observability, authentication, cost controls,
and their MCP Gateway. This is a significant strategic move: everything that previously
required a paid subscription is now available as open source (MIT license).

Portkey processes 2T+ tokens/day, manages $180M+ in annualized AI spend across 24,000+
organizations. They raised a $15M Series A in Feb 2026 led by Elevation Capital with
Lightspeed participation.

**"Governance" in Portkey's context means:** A unified control plane that enforces
who can use which models, under what conditions, with what safeguards, at what cost,
and with full audit visibility — spanning both LLM requests and MCP tool invocations.

---

## 1. What is Portkey's "Governance" Specifically?

Portkey's governance is NOT a single feature — it's a **constellation of 7 interlocking
systems** that together form an enterprise control plane:

| System | Purpose |
|--------|---------|
| **RBAC** | Who can access what (roles, workspaces, API key scoping) |
| **Budget/Cost Controls** | How much can be spent (per-user, per-team, per-model limits) |
| **Rate Limiting** | How fast can resources be consumed (RPM/RPH/RPD) |
| **Guardrails** | What content is allowed in/out (PII, injection, content moderation) |
| **Model Catalog** | Which models are available and at what cost |
| **MCP Governance** | Which tools/servers agents can access |
| **Audit Logging** | What happened, when, and by whom |

The key insight is that Portkey treats governance as **middleware in the request path** —
every AI request flows through the gateway, and governance policies are evaluated at
ingress before the model is ever called.

---

## 2. RBAC Model (Detailed)

### Hierarchy: Organization > Workspace > API Key

Portkey uses a **two-level hierarchy**:

#### Organization Level Roles
| Role | Capabilities |
|------|-------------|
| **Owner** | Full control: billing, org settings, create/delete workspaces, manage admin API keys, edit user roles, invite users, configure access permissions, access all workspaces |
| **Admin** | Everything except billing management |
| **Member** | No org-level capabilities without workspace assignment |

#### Workspace Level Roles
| Role | Capabilities |
|------|-------------|
| **Admin** | Invite members, assign roles (including admin), create API keys, CRUD resources, view resources |
| **Manager** | Invite members, create API keys, CRUD resources, view resources |
| **Member** | View resources only |

### API Key Scoping

Two key types:
1. **Organization (Admin) API Keys** — `type="organisation"`, access to Admin APIs for org-wide management
2. **Workspace API Keys** — `type="workspace"`, scoped to a single workspace

Sub-types: `service` (machine-to-machine) or `user` (user-specific, requires user_id)

Each API key supports:
- **Scopes** — List of permission strings (e.g., `workspace_service_api_keys.create`)
- **Rate limits** — Per-key rate limiting
- **Usage limits** — Per-key budget limits
- **Alert emails** — Notifications for usage thresholds

### Granular Permission Configuration
Workspaces have configurable access for:
- Logs access (view logs, view metadata — separately for managers vs members)
- Analytics access
- Provider/virtual key access
- API key management
- Prompt access
- Data visibility settings

### Terraform Support
Portkey provides a **Terraform provider** (`Portkey-AI/portkey`) for infrastructure-as-code management of API keys, workspaces, and permissions.

---

## 3. Budget/Cost Controls (Detailed)

Portkey has the most sophisticated budget system in the AI gateway space. It operates
at **three levels**:

### Level 1: Provider/Virtual Key Budget Limits
- Set cost-based limits ($) or token-based limits on individual AI provider keys
- Alert thresholds (e.g., alert at $400 when limit is $500)
- Periodic reset: weekly, monthly, or custom days (1-365)

### Level 2: API Key Budget Limits
- Per-key budget limits (cost or tokens)
- Per-key rate limits (RPM/RPH/RPD)
- Alert thresholds with email notifications

### Level 3: Workspace Budget Limits
- Workspace-wide budget caps
- Workspace-wide rate limits
- API-driven configuration:
```python
workspace = portkey.admin.workspaces.update(
    usage_limits=[{
        "credit_limit": 500,        # dollars
        "alert_threshold": 400,     # alert at $400
        "periodic_reset": "monthly" # auto-reset monthly
    }],
    rate_limits=[{
        "value": 1000,
        "type": "requests",
        "unit": "rpm"
    }]
)
```

### Level 4: Usage & Rate Limit Policies (the most powerful layer)

This is Portkey's most sophisticated governance feature — a **policy engine** with
condition-based rules and group-by dimensions:

**Policy Structure:**
```json
{
    "type": "usage_limits",
    "policy": {
        "conditions": [
            {"key": "metadata._user", "value": "*"}
        ],
        "group_by": [
            {"key": "metadata._user"},
            {"key": "model"}
        ],
        "credit_limit": 10,
        "type": "cost",
        "periodic_reset": "monthly",
        "status": "active"
    }
}
```

**Supported Condition Keys:**
| Key | Description | Example |
|-----|-------------|---------|
| `api_key` | Match by API key ID | UUID |
| `metadata.*` | Match by request metadata | `metadata._user`, `metadata._team` |
| `virtual_key` | Match by provider key | Provider slug |
| `provider` | Match by provider | `"openai"`, `"anthropic"` |
| `config` | Match by config slug | Config identifier |
| `prompt` | Match by prompt slug | Prompt identifier |
| `model` | Match by model (with wildcards) | `"@openai/gpt-4o"`, `"@anthropic/*"` |

**Key Use Cases Demonstrated:**
1. Global workspace rate limit
2. Per-user rate limit (via metadata)
3. Per-user monthly spend budget
4. Provider-specific rate limit
5. Model-specific token rate limit
6. Wildcard provider limits (`@anthropic/*`)
7. Per-provider weekly budget
8. Config-specific rate limit
9. Prompt-specific usage budget
10. Multiple allowed values (OR logic)
11. Exclude specific models
12. Per-user, per-model budget (multi-dimension group_by)
13. Team-based provider quota
14. Exclude internal API keys from limits
15. Combined conditions (premium users on specific models)

**Rate limit units:** `rpm` (per minute), `rph` (per hour), `rpd` (per day)
**Budget types:** `cost` (dollars) or `tokens`
**Reset periods:** `weekly`, `monthly`, or custom days (1-365)

**Full CRUD API:** Create, List, Retrieve, Update, Delete for both usage and rate limit policies.
**Per-entity reset:** Can list entities affected by a policy and reset individual entity usage.

---

## 4. Guardrails Governance (Detailed)

### Architecture

Guardrails are **named entities** that are created, configured, then attached to configs
via guardrail IDs. They operate as hooks:
- `input_guardrails` — Run before sending to LLM
- `output_guardrails` — Run after receiving from LLM

### Execution Modes

| Setting | Description |
|---------|-------------|
| **Async: TRUE** (default) | Run alongside LLM request, no latency impact, log-only |
| **Async: FALSE** | Run BEFORE request/response is forwarded, adds latency |
| **Deny: TRUE** | If checks fail, kill request with 446 status |
| **Deny: FALSE** (default) | If checks fail, continue but return 246 status |
| **Sequential: TRUE** | Run checks one after another |
| **Sequential: FALSE** (default) | Run checks in parallel |
| **On Success/Failure** | Append custom feedback for building eval datasets |

Custom status codes: **246** (guardrail failed, request continues) and **446** (guardrail
failed, request denied).

### Guardrail Types

#### BASIC (Deterministic, in OSS):

**Text & Format:**
- Regex Match, Sentence Count, Word Count, Character Count
- Uppercase/Lowercase Detection, Ends With

**Data & Structure:**
- JSON Schema validation, JSON Keys check
- Valid URLs, Contains Code, Not Null, Contains

**Security & Auth:**
- JWT Token Validator (JWKS, claim validation)
- Request Parameters Check (tool/param allowlists)

**Routing & Control:**
- Model Whitelist, Model Rules (metadata-driven)
- Allowed Request Types (endpoint allowlist/blocklist)
- Required Metadata Keys

**Extensibility:**
- Webhook (custom guardrails via HTTP)
- Log (send to external log URL)

#### PRO (LLM-based, paid):
- Content Moderation (category-based)
- Language Detection
- PII Detection (category-based)
- Gibberish Detection

#### Partner Guardrails (17+ integrations):
Acuvity, Aporia, AWS Bedrock, Azure, Javelin/Highflame, Lasso Security,
Mistral, Pangea, Palo Alto Prisma AIRS, Patronus, Pillar, Prompt Security,
Qualifire, CrowdStrike AIDR, Exa, F5, Walled AI, Zscaler, Akto

### Guardrail Governance Levels
- **Config-level:** Attached to specific routing configs
- **Workspace-level:** Enforced on ALL requests in a workspace
- **Organization-level:** Enforced across ALL workspaces

### Bring Your Own Guardrails
Custom guardrails can be defined as JSON or via webhook integration.

---

## 5. MCP Gateway Governance (Detailed)

### Two-Layer Authentication Architecture

| Layer | Direction | Methods |
|-------|-----------|---------|
| **Gateway Auth** | Agent -> Portkey | API Key, Portkey OAuth 2.1, External IdP |
| **Server Auth** | Portkey -> MCP Server | OAuth Auto, Client Credentials, Custom Headers |

### MCP-Specific Governance Features
- **MCP Registry:** Track, version, and manage MCP servers centrally
- **OAuth 2.1 for MCP:** PKCE flow (no more hardcoded API keys)
- **Per-user OAuth:** When MCP server accesses user-specific data
- **Identity Forwarding:** Pass user identity claims to MCP servers
- **Header Forwarding:** Pass custom headers (x-request-id, x-trace-id, x-tenant-id)
- **Credential Isolation:** Agents never see MCP server credentials

### Access Control Model
- Per-workspace MCP server registration
- Tool-level access control (which tools each team can invoke)
- All MCP requests go through the same guardrails, rate limits, and budget controls
  as LLM requests

---

## 6. Audit/Compliance (Detailed)

### Audit Logs
| Field | Description |
|-------|-------------|
| Timestamp | When the action occurred |
| User | Who performed the action |
| Workspace | Context of the action |
| Action | Operation type (create, update, delete) |
| Resource | What was affected |
| Response Status | HTTP status code |
| Client IP | Origin IP |
| Country | Geographic location |

### Filtering: By user, workspace, action type, date range, status code

### Compliance Certifications
- SOC 2
- ISO 27001
- GDPR
- HIPAA (with BAA signing)

### Enterprise Security Features
- SSO (OIDC with Okta)
- SCIM provisioning
- JWT-based authentication
- Bring Your Own Key (BYOK) for encryption
- VPC managed hosting
- Private tenancy
- Configurable retention periods
- Configurable exports to data lakes
- IP/Geo restrictions

---

## 7. Open-Source Architecture

### Tech Stack
- **Language:** TypeScript (96.1%), HTML (3.5%), JavaScript (0.4%)
- **Runtime:** Cloudflare Workers (edge computing, V8 engine)
- **Package:** `@portkey-ai/gateway` on npm (MIT license)
- **GitHub:** 11,157 stars, 970 forks
- **Latency:** Sub-10ms (single-digit globally on edge network)

### Why TypeScript over Python/Rust
- Edge computing platforms (Cloudflare Workers) limited to JS/TS/Rust
- Team expertise was in TypeScript
- V8 JIT compiler optimizations
- Async/await alignment with concurrent request handling
- Accessibility for open-source contributors

### Open Source vs. Paid (Open Core Model)

Despite the "fully open source" messaging, Portkey operates an **open core** model:

**Open Source (Gateway):**
- Universal API (250+ LLMs)
- Automatic fallbacks, load balancing, retries
- Conditional routing, request timeouts
- Config management, LLM key management
- Circuit breakers, semantic cache
- Budget limits, model catalog, metadata governance
- Usage policies
- Real-time metrics
- MCP registry, OAuth 2.1 for MCP
- Deterministic guardrails (BASIC)
- Logs, traces, feedback, custom metadata

**Paid (SaaS / Enterprise):**
- Advanced RBAC, team management, multi-workspace
- Audit logs
- Admin APIs (full control plane + data plane)
- SCIM provisioning, JWT authentication
- BYOK encryption
- Org-level metadata enforcement
- Org-level guardrails enforcement
- SSO (Okta)
- SOC2/ISO27001/GDPR/HIPAA compliance certificates
- BAA signing
- VPC managed hosting, private tenancy
- Configurable retention, data lake exports
- LLM-based guardrails (PII, content moderation)
- Partner guardrails
- Prompt management (advanced)
- Autonomous fine-tuning
- gRPC support
- Log retention beyond 3 days (Dev: 3 days, Pro: 30 days, Enterprise: custom)

**Pricing:**
- Open Source: Free, no request limits
- Dev: Free forever, 10K requests/month
- Pro: $49/month, 100K requests/month
- Enterprise: Custom

---

## 8. Portkey vs. LiteLLM Comparison

| Feature | Portkey | LiteLLM |
|---------|---------|---------|
| **Language** | TypeScript | Python |
| **Model Support** | 250+ LLMs | 100+ LLMs |
| **Latency** | Sub-10ms (edge) | Higher (not edge-deployed) |
| **RBAC** | Full org/workspace hierarchy | Basic admin auth |
| **Budget Controls** | Multi-level with policy engine | Per-team/per-key quotas |
| **Guardrails** | 30+ built-in + 17 partners | Plugin-based |
| **MCP Gateway** | Full MCP governance + OAuth 2.1 | Native MCP via upstream |
| **Audit Logs** | Full with filtering | Basic structured logging |
| **Prompt Management** | Full studio with versioning | Basic templates |
| **Observability** | Built-in traces, logs, metrics | OTEL integration |
| **Open Source** | TypeScript gateway (MIT) | Python proxy (MIT) |
| **Edge Deployment** | Native (Cloudflare Workers) | Not designed for edge |
| **ML Routing** | Not a focus | Advanced (18+ strategies) |

---

## 9. What RouteIQ Already Has That Matches

| Portkey Feature | RouteIQ Equivalent | Status |
|----------------|-------------------|--------|
| Budget limits at provider level | `quota.py` — per-team/per-key quota enforcement | **Partial** |
| Admin authentication | `auth.py` — admin auth, fail-closed pattern | **Present** |
| Basic RBAC | `rbac.py` — role-based access control | **Present** |
| Policy engine | `policy_engine.py` — OPA-style pre-request policy evaluation | **Present** |
| Audit logging | `audit.py` — file + structured event logging | **Present** |
| Guardrails (content filter) | `gateway/plugins/content_filter.py` | **Present** |
| Guardrails (PII) | `gateway/plugins/pii_guard.py` | **Present** |
| Guardrails (prompt injection) | `gateway/plugins/prompt_injection_guard.py` | **Present** |
| Guardrails (LlamaGuard) | `gateway/plugins/llamaguard_plugin.py` | **Present** |
| Guardrails (Bedrock) | `gateway/plugins/bedrock_guardrails.py` | **Present** |
| MCP Gateway | `mcp_gateway.py` + `mcp_tracing.py` | **Present** |
| SSRF Protection | `url_security.py` | **Present** |
| Circuit breakers | `resilience.py` | **Present** |
| OIDC/SSO | `oidc.py` | **Present** |
| Observability (OTEL) | `observability.py`, `metrics.py`, `telemetry_contracts.py` | **Present** |
| Config management | `config_loader.py`, `hot_reload.py`, `config_sync.py` | **Present** |
| Semantic cache | `semantic_cache.py` | **Present** |
| ML routing | `strategies.py` (18+ algorithms) | **Ahead** |
| Management middleware | `management_middleware.py` + `management_classifier.py` | **Present** |

---

## 10. What RouteIQ is Missing

### Critical Gaps (High Priority)

1. **Multi-level Budget Policy Engine**
   - Portkey's policy engine with conditions + group_by is far more sophisticated than
     RouteIQ's current `quota.py`. They support 15+ use cases combining user/team/model/
     provider/config dimensions with wildcards and exclusions.
   - RouteIQ needs: Condition-based policy definitions, multi-dimension group_by,
     per-entity tracking and reset, CRUD API for policies.

2. **Organization/Workspace Hierarchy**
   - Portkey has a full Org > Workspace > API Key hierarchy with different roles at each level.
   - RouteIQ has basic RBAC but lacks the multi-tenant workspace model.
   - RouteIQ needs: Workspace abstraction, workspace-scoped API keys, workspace-level
     resource isolation.

3. **Guardrails as First-Class CRUD Entities**
   - Portkey guardrails are named, versioned entities with IDs that are created via UI/API
     then attached to configs. They can be enforced at workspace or org level.
   - RouteIQ has guardrails as plugins but lacks: guardrail CRUD API, guardrail IDs,
     workspace/org-level enforcement, the 246/446 status code pattern.

4. **Model Catalog**
   - Portkey maintains a continuously updated registry of models across providers with
     pricing data.
   - RouteIQ has `llm_candidates.json` but no model catalog API or pricing data.

### Important Gaps (Medium Priority)

5. **MCP OAuth 2.1 Authentication**
   - Portkey's two-layer auth (gateway auth + server auth) with OAuth 2.1 PKCE for MCP
     is more sophisticated than RouteIQ's current MCP auth.
   - RouteIQ needs: Per-user OAuth for MCP servers, identity forwarding, credential isolation.

6. **Terraform/IaC Integration**
   - Portkey has a Terraform provider for managing API keys, workspaces, policies.
   - RouteIQ has no IaC integration.

7. **Budget Alerts & Notifications**
   - Portkey has alert thresholds with email notifications, exhaustion alerts.
   - RouteIQ's quota system lacks notification capabilities.

8. **Admin API (Full Control Plane)**
   - Portkey exposes a comprehensive Admin API for programmatic management of all
     governance entities.
   - RouteIQ's `routes/config.py` is limited compared to what Portkey offers.

### Nice-to-Have Gaps (Lower Priority)

9. **Deterministic Guardrail Suite**
   - Portkey's BASIC guardrails (regex, JSON schema, JWT validation, model whitelist,
     request type allowlist, required metadata) are built into the OSS gateway.
   - RouteIQ should expand its guardrail plugin library with similar deterministic checks.

10. **Partner Guardrail Integrations**
    - Portkey has 17+ partner guardrail integrations (CrowdStrike, Zscaler, Palo Alto, etc.)
    - RouteIQ has Bedrock Guardrails and LlamaGuard.

11. **Custom Status Codes (246/446)**
    - Portkey's guardrail-specific status codes enable smart retry/fallback behavior.
    - RouteIQ could adopt a similar pattern in its plugin middleware.

---

## 11. Architectural Recommendations for RouteIQ

### Principle: Build governance as a composable middleware stack, not a monolith

RouteIQ has an architectural advantage over Portkey: it's Python-based with a plugin
system, runs on FastAPI, and inherits LiteLLM's massive provider ecosystem. The
governance layer should leverage these strengths.

### Recommendation 1: Governance Policy Engine (`governance/policy_engine.py`)

Replace/augment the current `quota.py` with a full policy engine inspired by (but
architecturally distinct from) Portkey's approach:

```
Policy = {
    type: "usage_limit" | "rate_limit",
    conditions: [{key, value, excludes}],  # Multi-dimension matching
    group_by: [{key}],                      # Aggregation dimensions
    limits: {credit_limit, alert_threshold, periodic_reset},
    status: "active" | "paused"
}
```

**Key design decisions:**
- Store policies in Postgres (not just config YAML)
- Use Redis for real-time counter tracking (already have `redis_pool.py`)
- Expose full CRUD API via new `routes/governance.py`
- Evaluate policies at the ASGI middleware layer (like `policy_engine.py` already does)
- Support the same condition keys: `api_key`, `metadata.*`, `model`, `provider`, `config`

### Recommendation 2: Workspace Abstraction (`governance/workspaces.py`)

Add a workspace layer to RouteIQ's existing RBAC:

```
Organization (existing)
  └── Workspace (NEW)
       ├── API Keys (scoped)
       ├── Configs (scoped)
       ├── Budget Policies (scoped)
       ├── Guardrails (scoped, inherits org-level)
       └── MCP Server Registrations (scoped)
```

- Workspaces are the unit of resource isolation
- API keys are always scoped to a workspace
- Budget/rate limits can be set at workspace level
- Guardrails can be enforced at workspace level (mandatory for all requests)

### Recommendation 3: Guardrails as CRUD Entities (`governance/guardrails.py`)

Refactor the current plugin-based guardrails into named, versioned entities:

```python
class GuardrailDefinition(BaseModel):
    id: str
    name: str
    checks: list[GuardrailCheck]
    actions: GuardrailActions  # async, deny, sequential, on_success/failure
    scope: "input" | "output" | "both"
    workspace_id: str | None  # None = org-level
```

- CRUD API for guardrail management
- Guardrail IDs referenced in configs
- Workspace-level and org-level enforcement
- Support the 246/446 status code pattern for smart retry/fallback
- Keep the plugin system for complex guardrails (LlamaGuard, Bedrock)
- Add deterministic checks: regex, JSON schema, model whitelist, required metadata

### Recommendation 4: Model Catalog Service (`governance/model_catalog.py`)

- Maintain a registry of available models with pricing data
- Enable workspace-level model access control (which models each team can use)
- Track model versions and deprecation status
- Integrate with the routing strategies for cost-aware routing

### Recommendation 5: Enhanced MCP Governance

Build on the existing `mcp_gateway.py`:
- Add OAuth 2.1 PKCE flow for MCP server authentication
- Per-workspace MCP server registration
- Tool-level access control (which tools each workspace can invoke)
- Identity forwarding to MCP servers
- Apply the same budget/rate limit policies to MCP tool invocations

### Recommendation 6: Admin API Expansion

Expand `routes/config.py` into a comprehensive Admin API:
- Workspace CRUD
- API Key management (with scopes, limits, alerts)
- Policy CRUD (usage limits, rate limits)
- Guardrail CRUD
- Model catalog management
- MCP server management
- Audit log querying

### Implementation Priority Order

| Phase | Feature | Effort | Impact |
|-------|---------|--------|--------|
| **Phase 1** | Governance Policy Engine (budget + rate limit policies) | 2-3 weeks | Critical |
| **Phase 2** | Workspace Abstraction | 2 weeks | High |
| **Phase 3** | Guardrails as CRUD Entities | 2 weeks | High |
| **Phase 4** | Admin API Expansion | 1-2 weeks | High |
| **Phase 5** | Model Catalog | 1 week | Medium |
| **Phase 6** | MCP OAuth 2.1 + workspace-scoped MCP | 2 weeks | Medium |
| **Phase 7** | Budget Alerts & Notifications | 1 week | Medium |
| **Phase 8** | Terraform Provider | 2-3 weeks | Nice-to-have |

---

## 12. RouteIQ's Competitive Advantages

Despite the gaps, RouteIQ has significant advantages over Portkey:

1. **ML-Based Routing (18+ strategies)** — Portkey has no equivalent. This is RouteIQ's
   unique value proposition and cannot be replicated easily.

2. **Python Ecosystem** — Direct integration with data science and ML tooling.
   Portkey's TypeScript gateway cannot do ML model inference natively.

3. **LiteLLM Inheritance** — 100+ provider support, battle-tested proxy, massive
   community (30K+ GitHub stars on upstream).

4. **Plugin Architecture** — RouteIQ's plugin system with lifecycle management is
   more extensible than Portkey's guardrail-as-config approach.

5. **OPA-Style Policy Engine** — RouteIQ already has a policy engine pattern;
   Portkey's is simpler (conditions + group_by) but doesn't support arbitrary
   policy logic like OPA.

6. **A2A Protocol** — RouteIQ has Agent-to-Agent support; Portkey does not.

7. **Self-Hosted First** — RouteIQ is designed for self-hosting; Portkey's edge
   architecture makes true self-hosting more complex.

---

## Appendix: Key Portkey URLs for Reference

- Feature comparison: https://docs.portkey.ai/docs/product/product-feature-comparison
- RBAC: https://docs.portkey.ai/docs/product/enterprise-offering/org-management/user-roles-and-permissions
- API Keys: https://docs.portkey.ai/docs/product/enterprise-offering/org-management/api-keys-authn-and-authz
- Budget policies: https://docs.portkey.ai/docs/product/enterprise-offering/budget-policies
- Workspace budgets: https://docs.portkey.ai/docs/product/administration/enforce-workspace-budget-limts-and-rate-limits
- Guardrails: https://docs.portkey.ai/docs/product/guardrails
- Guardrail checks: https://docs.portkey.ai/docs/product/guardrails/list-of-guardrail-checks
- Workspace guardrails: https://docs.portkey.ai/docs/product/administration/enforce-workspace-level-guardials
- Org guardrails: https://docs.portkey.ai/docs/product/administration/enforce-orgnization-level-guardrails
- MCP Gateway auth: https://docs.portkey.ai/docs/product/mcp-gateway/authentication
- Audit logs: https://docs.portkey.ai/docs/product/enterprise-offering/audit-logs
- Security: https://docs.portkey.ai/docs/product/enterprise-offering/security-portkey
- Admin API: https://docs.portkey.ai/docs/api-reference/admin-api/introduction
- Gateway 2.0 blog: https://portkey.ai/blog/gateway-2-0
- Open source announcement: https://www.globenewswire.com/news-release/2026/03/24/3261574/0/en/
- Terraform provider: https://registry.terraform.io/providers/Portkey-AI/portkey/latest
- GitHub: https://github.com/Portkey-AI/gateway
