# Architecture Decision Records

ADRs document significant architectural decisions for the RouteIQ Gateway project.

## Index

| ADR | Title | Status |
|-----|-------|--------|
| [0001](../../docs/adr/0001-three-layer-architecture.md) | Adopt Three-Layer Architecture | Accepted |
| [0002](../../docs/adr/0002-plugin-routing-strategy.md) | Use CustomRoutingStrategyBase Instead of Monkey-Patching | Accepted |
| [0003](../../docs/adr/0003-delete-redundant-mcp.md) | Delete Redundant MCP Transport Code | Accepted |
| [0004](../../docs/adr/0004-asyncpg-connection-pooling.md) | Implement asyncpg Connection Pooling | Accepted |
| [0005](../../docs/adr/0005-redis-singleton.md) | Redis Singleton Client with Lifecycle Management | Accepted |
| [0006](../../docs/adr/0006-security-hardening-defaults.md) | Secure-by-Default Security Posture | Accepted |
| [0007](../../docs/adr/0007-dependency-tiering.md) | Tier Dependencies into Optional Extras | Accepted |
| [0008](../../docs/adr/0008-oidc-identity-integration.md) | OIDC/SSO for Enterprise Identity Management | Accepted |
| [0009](../../docs/adr/0009-multi-tier-docker-images.md) | Multi-Tier Docker Images (Slim/Full/GPU) | Accepted |
| [0010](../../docs/adr/0010-centroid-zero-config-routing.md) | Centroid-Based Zero-Config Routing | Accepted |
| [0011](../../docs/adr/0011-pluggable-external-services.md) | Pluggable External Services | Accepted |
| [0012](../../docs/adr/0012-own-fastapi-app.md) | RouteIQ Owns Its FastAPI Application | Accepted |
| [0013](../../docs/adr/0013-pydantic-settings.md) | Pydantic Settings for Typed Configuration | Accepted |
| [0014](../../docs/adr/0014-plugin-extraction.md) | Extract RouteIQ into Independently Installable Packages | Proposed |
| [0015](../../docs/adr/0015-k8s-native-leader-election.md) | K8s-Native Leader Election via Lease API | Accepted |
| [0016](../../docs/adr/0016-developer-experience-features.md) | Developer Experience Features | Accepted |
| [0017](../../docs/adr/0017-leverage-litellm-upstream.md) | Leverage LiteLLM Upstream Capabilities | Accepted |
| [0018](../../docs/adr/0018-disaggregated-ui.md) | Support Disaggregated UI Deployment | Accepted |
| [0019](../../docs/adr/0019-otel-genai-conventions.md) | Adopt OpenTelemetry GenAI Semantic Conventions | Proposed |
| [0020](../../docs/adr/0020-governance-layer.md) | Governance Layer - Workspace Isolation & Dynamic Policies | Accepted |
| [0021](../../docs/adr/0021-externalized-state.md) | Externalize In-Process State to Redis for Multi-Worker Safety | Accepted |

## ADR Format

Each ADR includes:

- **Status**: Accepted, Proposed, Superseded, or Deprecated
- **Date**: When the decision was made
- **Context**: Why the decision was needed
- **Decision**: What was decided
- **Consequences**: Positive and negative outcomes
- **Alternatives Considered**: What else was evaluated
