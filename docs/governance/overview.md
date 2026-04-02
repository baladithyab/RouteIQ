# Governance Overview

RouteIQ provides enterprise governance features for managing AI usage across
teams, enforcing policies, and maintaining compliance.

## Key Capabilities

- **Workspaces** — Isolated environments with per-team budgets and model access
- **Usage Policies** — Rate limits, quotas, and spend caps per team/key
- **Guardrails** — Content filtering, PII detection, prompt injection defense
- **Identity** — OIDC/SSO integration for enterprise authentication
- **Audit Logging** — Structured audit trail for all operations
- **Policy Engine** — OPA-style pre-request policy evaluation

## Architecture

Governance operates at the ASGI middleware layer, evaluating policies before
requests reach the routing layer. The hierarchy flows from organization to
workspace to API key, with enforcement computing the intersection of
constraints at each level.

```mermaid
graph TB
    subgraph "Organization"
        ORG[Org: Acme Corp<br/>org_id: acme]
        
        subgraph "Workspace: Engineering"
            WS1[Workspace Config<br/>budget: $500/mo<br/>models: gpt-4o*, claude-*<br/>rpm: 1000]
            KEY1[API Key: sk-eng-1<br/>budget: $100/mo<br/>scopes: completions.write]
            KEY2[API Key: sk-eng-2<br/>budget: $50/mo<br/>models: gpt-4o-mini only]
        end
        
        subgraph "Workspace: Research"
            WS2[Workspace Config<br/>budget: $2000/mo<br/>models: *<br/>rpm: 5000]
            KEY3[API Key: sk-res-1<br/>budget: $500/mo<br/>scopes: completions.write, embeddings.write]
        end
    end
    
    ORG --> WS1
    ORG --> WS2
    WS1 --> KEY1
    WS1 --> KEY2
    WS2 --> KEY3
    
    KEY1 --> ENF[GovernanceEngine.enforce]
    KEY2 --> ENF
    KEY3 --> ENF
    ENF --> |model access| MA[Model Allowlist<br/>Intersection of<br/>workspace + key]
    ENF --> |budget| BUD[Budget Check<br/>min workspace, key<br/>Redis atomic]
    ENF --> |rate limit| RL[Rate Limit<br/>Redis sliding window]
```

## Getting Started

- [Workspaces](workspaces.md) — Set up isolated team environments
- [Usage Policies](usage-policies.md) — Configure rate limits and budgets
- [Guardrails](guardrails.md) — Enable content safety plugins
- [Identity](identity.md) — Configure OIDC/SSO
