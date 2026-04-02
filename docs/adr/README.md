# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records for the RouteIQ Gateway project.
ADRs document significant architectural decisions, their context, and consequences.

## Format

Each ADR follows the standard format:

- **Status**: Accepted, Proposed, Superseded, or Deprecated
- **Date**: When the decision was made or proposed
- **Context**: Why the decision was needed
- **Decision**: What was decided
- **Consequences**: Positive and negative outcomes
- **Alternatives Considered**: What else was evaluated

## Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [0001](0001-three-layer-architecture.md) | Adopt Three-Layer Architecture | Accepted | 2026-04-02 |
| [0002](0002-plugin-routing-strategy.md) | Use CustomRoutingStrategyBase Instead of Monkey-Patching | Accepted | 2026-04-02 |
| [0003](0003-delete-redundant-mcp.md) | Delete Redundant MCP Transport Code | Accepted | 2026-04-02 |
| [0004](0004-asyncpg-connection-pooling.md) | Implement asyncpg Connection Pooling | Accepted | 2026-04-02 |
| [0005](0005-redis-singleton.md) | Redis Singleton Client with Lifecycle Management | Accepted | 2026-04-02 |
| [0006](0006-security-hardening-defaults.md) | Secure-by-Default Security Posture | Accepted | 2026-04-02 |
| [0007](0007-dependency-tiering.md) | Tier Dependencies into Optional Extras | Accepted | 2026-04-02 |
| [0008](0008-oidc-identity-integration.md) | OIDC/SSO for Enterprise Identity Management | Accepted | 2026-04-02 |
| [0009](0009-multi-tier-docker-images.md) | Multi-Tier Docker Images (Slim/Full/GPU) | Accepted | 2026-04-02 |
| [0010](0010-centroid-zero-config-routing.md) | Centroid-Based Zero-Config Routing | Accepted | 2026-04-02 |
| [0011](0011-pluggable-external-services.md) | Pluggable External Services | Accepted | 2026-04-02 |
| [0012](0012-own-fastapi-app.md) | RouteIQ Owns Its FastAPI Application | Accepted | 2026-04-02 |
| [0013](0013-pydantic-settings.md) | Pydantic Settings for Typed Configuration | Accepted | 2026-04-02 |
| [0014](0014-plugin-extraction.md) | Extract RouteIQ into Independently Installable Packages | Accepted | 2026-04-02 |
| [0015](0015-k8s-native-leader-election.md) | K8s-Native Leader Election via Lease API | Accepted | 2026-04-02 |
| [0016](0016-developer-experience-features.md) | Developer Experience Features | Accepted | 2026-04-02 |
| [0017](0017-leverage-litellm-upstream.md) | Leverage LiteLLM Upstream Capabilities | Accepted | 2026-04-02 |
| [0018](0018-disaggregated-ui.md) | Support Disaggregated UI Deployment | Accepted | 2026-04-02 |
| [0019](0019-otel-genai-conventions.md) | Adopt OpenTelemetry GenAI Semantic Conventions | Accepted | 2026-04-02 |
| [0020](0020-governance-layer.md) | Governance Layer — Workspace Isolation & Dynamic Policies | Accepted | 2026-04-02 |
| [0021](0021-externalized-state.md) | Externalize In-Process State to Redis for Multi-Worker Safety | Accepted | 2026-04-02 |

## Lifecycle

- **Proposed**: Under discussion, not yet implemented
- **Accepted**: Decision made and implemented (or in progress)
- **Superseded**: Replaced by a newer ADR (link to successor)
- **Deprecated**: No longer relevant

## Creating a New ADR

1. Copy the template below
2. Assign the next sequential number
3. Fill in all sections
4. Submit for review
5. Update this index

### Template

```markdown
# ADR-XXXX: Title

**Status**: Proposed
**Date**: YYYY-MM-DD
**Decision Makers**: RouteIQ Core Team

## Context

Why this decision is needed.

## Decision

What was decided.

## Consequences

### Positive
- ...

### Negative
- ...

## Alternatives Considered

### Alternative A
- Pros: ...
- Cons: ...

### Alternative B
- Pros: ...
- Cons: ...
```
