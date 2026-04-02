# Architecture Overview

RouteIQ Gateway is a high-performance, cloud-native AI Gateway built on
[LiteLLM](https://github.com/BerriAI/litellm) and [LLMRouter](https://github.com/ulab-uiuc/LLMRouter).

## Three-Layer Architecture

RouteIQ is composed of three distinct layers. Layer 1 (LiteLLM Proxy) provides
the OpenAI-compatible API surface and protocol support. Layer 2 (RouteIQ
Intelligence) adds ML-based routing, governance, guardrails, evaluation, and
context optimization. Layer 3 (RouteIQ Gateway) is the composition root —
FastAPI app factory, middleware stack, route registration, and plugin lifecycle.

```mermaid
graph TB
    subgraph "Layer 3: RouteIQ Gateway"
        APP[FastAPI App<br/>create_gateway_app]
        MW[Middleware Stack<br/>Backpressure → CORS → RequestID<br/>→ Policy → Governance → Guardrails]
        ROUTES[Routes<br/>Health · Config · Models<br/>Admin UI · Governance API]
        PLUGINS[Plugin Manager<br/>14 plugins with lifecycle]
    end
    
    subgraph "Layer 2: RouteIQ Intelligence"
        ROUTING[Routing Engine<br/>18+ ML strategies<br/>Centroid · Personalized · R1]
        GOV[Governance Engine<br/>Workspaces · Budgets<br/>Rate Limits · Policies]
        GUARD[Guardrail Engine<br/>14 check types<br/>Input/Output phases]
        EVAL[Eval Pipeline<br/>LLM-as-judge<br/>Quality feedback]
        CTX[Context Optimizer<br/>6 transforms<br/>30-70% savings]
    end
    
    subgraph "Layer 1: LiteLLM Proxy"
        LITELLM[LiteLLM v1.82.3<br/>100+ providers<br/>OpenAI-compatible API]
        MCP[MCP Gateway<br/>Native JSON-RPC · SSE]
        A2A[A2A Protocol<br/>Agent registry]
        CACHE[Caching<br/>11 backends]
    end
    
    APP --> MW --> ROUTES
    APP --> PLUGINS
    ROUTES --> ROUTING --> LITELLM
    ROUTES --> GOV
    ROUTES --> GUARD
    MW --> GOV
    MW --> GUARD
    PLUGINS --> CTX
    PLUGINS --> EVAL
    LITELLM --> MCP
    LITELLM --> A2A
    LITELLM --> CACHE
```

## Request Flow

Every request traverses the full middleware stack before reaching the LLM
provider. The middleware order is critical — backpressure is innermost (registered
first), ensuring concurrency limits apply before any other processing.

```mermaid
sequenceDiagram
    participant C as Client
    participant BP as Backpressure
    participant CORS as CORS
    participant RID as RequestID
    participant POL as Policy Engine
    participant GOV as Governance
    participant GR as Guardrails (Input)
    participant CTX as Context Optimizer
    participant RT as Routing Strategy
    participant PR as Personalized Router
    participant LLM as LiteLLM → Provider
    participant GRO as Guardrails (Output)
    participant TEL as Telemetry
    
    C->>BP: POST /chat/completions
    BP->>CORS: Check concurrency limit
    CORS->>RID: Add X-Request-ID
    RID->>POL: Evaluate allow/deny rules
    POL->>GOV: Enforce workspace limits
    GOV->>GR: Evaluate input guardrails
    GR->>CTX: Optimize context (30-70% savings)
    CTX->>RT: Route request (centroid/ML/strategy)
    RT->>PR: Re-rank by user preference
    PR->>LLM: Forward to selected model
    LLM-->>GRO: Response
    GRO-->>TEL: Log telemetry + eval sample
    TEL-->>C: Response + X-RouteIQ-* headers
```

## Routing Decision Flow

The routing engine handles capability detection, session affinity, profile-based
sorting, vision content swaps, context window validation, and personalized
re-ranking before selecting a deployment.

```mermaid
flowchart TD
    REQ[Request arrives] --> GOV{Governance<br/>check}
    GOV -->|Denied| R403[403 Forbidden]
    GOV -->|Allowed| GUARD{Input<br/>guardrails}
    GUARD -->|Denied| R446[446 Guardrail Denied]
    GUARD -->|Passed| CAP[Detect required<br/>capabilities]
    CAP --> FILTER[Filter deployments<br/>by capabilities]
    FILTER --> SESSION{Session<br/>cache hit?}
    SESSION -->|Hit| CACHED[Return cached model]
    SESSION -->|Miss| PROFILE{Routing<br/>profile?}
    PROFILE -->|eco| COST[Sort by cost<br/>cheapest first]
    PROFILE -->|premium| QUALITY[Sort by quality<br/>best first]
    PROFILE -->|auto| CENTROID[Centroid<br/>classify tier]
    CENTROID --> VISION{Vision<br/>content?}
    VISION -->|Yes, non-vision model| SWAP[Swap to vision model]
    VISION -->|No or already vision| CTX{Context<br/>window OK?}
    SWAP --> CTX
    CTX -->|Exceeds| UPGRADE[Try larger model]
    CTX -->|Fits| PERS{Personalized<br/>routing enabled?}
    UPGRADE --> PERS
    PERS -->|Yes + user_id| RERANK[Re-rank by<br/>user preference]
    PERS -->|No| SELECT[Select deployment]
    RERANK --> SELECT
    COST --> SELECT
    QUALITY --> SELECT
    SELECT --> CACHE_STORE[Store in session cache]
    CACHE_STORE --> LLM[Forward to LLM]
```

## Governance Architecture

Governance enforces organization → workspace → API key hierarchy. Each level
can set budgets, model allowlists, and rate limits. Enforcement computes the
intersection of workspace and key constraints.

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

## Plugin Lifecycle

Plugins are registered, dependency-resolved, started during app lifespan, and
shut down on termination. Failed plugins enter a quarantined state with retry.

```mermaid
stateDiagram-v2
    [*] --> Registered: register_plugin()
    Registered --> DepsResolved: resolve_dependencies()
    DepsResolved --> Starting: startup(app)
    Starting --> Active: success
    Starting --> Quarantined: failure
    Active --> Active: on_llm_pre_call / on_llm_post_call
    Active --> Stopping: shutdown(app)
    Quarantined --> Starting: retry
    Stopping --> [*]
```

## Deployment Topology

Production deployments use Kubernetes with HPA, backed by PostgreSQL, Redis,
an OTel Collector, and an OIDC provider. Leader election (via Redis or K8s
Lease API) ensures only one pod runs config sync.

```mermaid
graph TB
    subgraph "Edge / CDN"
        UI[Admin UI<br/>S3 + CloudFront<br/>or embedded]
    end
    
    subgraph "K8s Cluster"
        subgraph "RouteIQ Pods (HPA)"
            POD1[RouteIQ Pod 1<br/>Leader]
            POD2[RouteIQ Pod 2]
            POD3[RouteIQ Pod 3]
        end
        
        SVC[Service<br/>LoadBalancer / Ingress]
        SM[ServiceMonitor<br/>→ Prometheus]
    end
    
    subgraph "External Services"
        PG[(PostgreSQL<br/>RDS / CloudSQL)]
        REDIS[(Redis<br/>ElastiCache / Memorystore)]
        OTEL[OTel Collector<br/>→ Jaeger / Datadog]
        OIDC[OIDC Provider<br/>Keycloak / Auth0 / Okta]
    end
    
    UI --> SVC
    SVC --> POD1 & POD2 & POD3
    POD1 & POD2 & POD3 --> PG
    POD1 & POD2 & POD3 --> REDIS
    POD1 & POD2 & POD3 --> OTEL
    POD1 & POD2 & POD3 --> OIDC
    SM --> POD1 & POD2 & POD3
    
    POD1 -.->|Leader Election<br/>K8s Lease API| REDIS
```

## Key Components

### Data Plane

- **Unified API**: OpenAI-compatible proxy (inherited from LiteLLM)
- **Protocol Translation**: Bedrock, Vertex AI, Azure, etc.
- **Gateway Surfaces**: MCP, A2A, Skills endpoints
- **Plugin System**: 14 built-in plugins with lifecycle management

### Routing Intelligence Layer

- **Static Strategies**: round-robin, fallback (LiteLLM-native)
- **ML Strategies**: 18+ `llmrouter-*` strategies
- **Centroid Routing**: Zero-config ~2ms classification
- **A/B Testing**: Runtime strategy hot-swapping

### Control Plane

- **Configuration Management**: YAML-based, hot-reloadable
- **Artifact Registry**: S3/MinIO for trained routing models
- **Rollout Delivery**: Rolling deploys or sync sidecars

### Closed-Loop MLOps

- **Collect**: OTel traces/logs from data plane
- **Train**: Offline jobs produce routing artifacts
- **Deploy**: New artifacts rolled out, routing layer reloads

## Middleware Stack

Request processing order (innermost to outermost):

1. **Backpressure** - Concurrent request limiting
2. **CORS** - Cross-origin resource sharing
3. **RequestID** - Correlation ID assignment
4. **Policy Engine** - OPA-style allow/deny evaluation
5. **Governance** - Workspace and budget enforcement
6. **Guardrails** - Input/output content safety
7. **Management RBAC** - Admin endpoint protection
8. **Plugin Middleware** - Plugin-injected middleware
9. **Router Decision** - Telemetry span attributes
