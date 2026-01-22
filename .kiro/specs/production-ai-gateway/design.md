# Design Document: Production AI Gateway with LiteLLM + LLMRouter

## Overview

This design document describes a production-ready AI Gateway that serves as an **enhancement layer on top of the LiteLLM Proxy**. The system integrates LiteLLM (unified LLM API gateway) with LLMRouter (ML-based intelligent routing) into a single, production-hardened container.

**Enhancement Layer Philosophy:** We inherit all LiteLLM capabilities (authentication, caching, rate limiting, 100+ provider integrations, Skills endpoints) and extend them with ML routing, hot-reload, standardized HA protocols (HTTP/SSE/OTLP/Postgres/Redis), and agentic gateway support (MCP, A2A).

The system provides a single OpenAI-compatible API endpoint that can route requests to 100+ LLM providers using 18+ intelligent routing strategies, with enterprise features including high availability, persistence, observability, hot reload, and support for modern protocols (A2A, MCP, Skills).

### Key Design Principles

1. **Enhancement, Not Replacement**: Extend LiteLLM without forking core functionality
2. **Zero-Downtime Updates**: Hot reload for routing models and configurations without service restart
3. **Horizontal Scalability**: Stateless design with Redis for distributed state and PostgreSQL for persistence
4. **Observability First**: Comprehensive tracing, metrics, and logging for production debugging
5. **Protocol Extensibility**: Support for emerging standards (A2A, MCP, Skills) alongside OpenAI compatibility
6. **ML-Powered Intelligence**: Use trained routing models to optimize for cost, latency, and quality
7. **Moat-Mode Ready**: Support air-gapped and controlled-egress deployments with standardized protocols

### Enhancement Backlog

This design is complemented by a prioritized backlog of cloud-native enhancements (HA, Security, Observability) detailed in **[`docs/litellm-cloud-native-enhancements.md`](../../../docs/litellm-cloud-native-enhancements.md)**. This backlog tracks the implementation of advanced features like hot-reload sync, streaming-aware shutdown, and durable audit exports.

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Client Applications                              │
│                    (OpenAI SDK, HTTP Clients, etc.)                      │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 │ HTTP/HTTPS
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          Load Balancer (Nginx)                           │
│                      (Optional for HA deployment)                        │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                ┌────────────────┴────────────────┐
                │                                 │
                ▼                                 ▼
┌───────────────────────────┐       ┌───────────────────────────┐
│   Gateway Instance 1      │       │   Gateway Instance N      │
│  ┌─────────────────────┐  │       │  ┌─────────────────────┐  │
│  │  LiteLLM Proxy      │  │       │  │  LiteLLM Proxy      │  │
│  │  (FastAPI Server)   │  │       │  │  (FastAPI Server)   │  │
│  └──────────┬──────────┘  │       │  └──────────┬──────────┘  │
│             │              │       │             │              │
│  ┌──────────▼──────────┐  │       │  ┌──────────▼──────────┐  │
│  │  LLMRouter          │  │       │  │  LLMRouter          │  │
│  │  Strategy Family    │  │       │  │  Strategy Family    │  │
│  │  (18+ Strategies)   │  │       │  │  (18+ Strategies)   │  │
│  └──────────┬──────────┘  │       │  └──────────┬──────────┘  │
│             │              │       │             │              │
│  ┌──────────▼──────────┐  │       │  ┌──────────▼──────────┐  │
│  │  Extension Routes   │  │       │  │  Extension Routes   │  │
│  │  • A2A Gateway      │  │       │  │  • A2A Gateway      │  │
│  │  • MCP Gateway      │  │       │  │  • MCP Gateway      │  │
│  │  • Hot Reload API   │  │       │  │  • Hot Reload API   │  │
│  └─────────────────────┘  │       │  └─────────────────────┘  │
└───────────┬───────────────┘       └───────────┬───────────────┘
            │                                   │
            └───────────────┬───────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Redis      │    │ PostgreSQL   │    │  S3 / GCS    │
│  (Cache &    │    │ (Persistence │    │  (Config &   │
│   State)     │    │  & Logs)     │    │   Models)    │
└──────────────┘    └──────────────┘    └──────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │   LLM Providers       │
                │  • OpenAI             │
                │  • Anthropic          │
                │  • AWS Bedrock        │
                │  • Azure OpenAI       │
                │  • 100+ others        │
                └───────────────────────┘
```

### Request Flow

1. **Client Request**: Client sends OpenAI-compatible request to Gateway
2. **Authentication**: Gateway validates API key against master_key or database
3. **Routing Decision**: LLMRouter strategy selects optimal model based on query
4. **Cache Check**: Gateway checks Redis cache for identical previous request
5. **LLM Invocation**: LiteLLM forwards request to selected provider
6. **Response Processing**: Gateway processes response, updates cache, logs metrics
7. **Client Response**: Gateway returns OpenAI-compatible response to client

### Data Flow for Hot Reload

```
┌─────────────────────────────────────────────────────────────────┐
│                    Config Sync Manager                          │
│                   (Background Thread)                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ Periodic Check (ETag-based)
                             ▼
                    ┌────────────────┐
                    │   S3 / GCS     │
                    │  (Remote       │
                    │   Storage)     │
                    └────────┬───────┘
                             │
                             │ Download if ETag changed
                             ▼
                    ┌────────────────┐
                    │  Local Config  │
                    │  /app/config/  │
                    └────────┬───────┘
                             │
                             │ SIGHUP signal
                             ▼
                    ┌────────────────┐
                    │  LiteLLM Proxy │
                    │  (Reload)      │
                    └────────────────┘
```

## Components and Interfaces

### 1. LiteLLM Proxy Server

**Responsibility**: Unified API gateway providing OpenAI-compatible interface to 100+ LLM providers.

**Key Interfaces**:
- `/v1/chat/completions` - Chat completion endpoint (OpenAI-compatible)
- `/v1/embeddings` - Text embeddings endpoint
- `/v1/images/generations` - Image generation endpoint
- `/v1/audio/transcriptions` - Audio transcription endpoint
- `/v1/audio/speech` - Text-to-speech endpoint
- `/v1/moderations` - Content moderation endpoint
- `/v1/batches` - Batch processing endpoint
- `/v1/rerank` - Document reranking endpoint
- `/health/liveliness` - Health check for orchestrators
- `/health/readiness` - Readiness check for traffic routing

**Configuration**:
```yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: os.environ/OPENAI_API_KEY

litellm_settings:
  cache: true
  cache_params:
    type: redis
    host: os.environ/REDIS_HOST
    port: 6379
    ttl: 3600
```

### 2. LLMRouter Strategy Family

**Responsibility**: ML-based routing strategies for selecting optimal LLM based on query characteristics.

**Class**: `LLMRouterStrategyFamily`

**Key Methods**:
- `__init__(strategy_name, model_path, llm_data_path, hot_reload, reload_interval, **kwargs)` - Initialize strategy with configuration
- `router` (property) - Get router instance, loading/reloading as needed
- `_load_router()` - Load the appropriate LLMRouter model based on strategy name
- `_should_reload()` - Check if model should be reloaded based on file mtime and interval

**Supported Strategies**:
- **Single-round**: `llmrouter-knn`, `llmrouter-svm`, `llmrouter-mlp`, `llmrouter-mf`, `llmrouter-elo`, `llmrouter-routerdc`, `llmrouter-hybrid`, `llmrouter-causallm`, `llmrouter-graph`, `llmrouter-automix`
- **Multi-round**: `llmrouter-r1`
- **Personalized**: `llmrouter-gmt`
- **Agentic**: `llmrouter-knn-multiround`, `llmrouter-llm-multiround`
- **Baseline**: `llmrouter-smallest`, `llmrouter-largest`
- **Custom**: `llmrouter-custom`

**Configuration**:
```yaml
router_settings:
  routing_strategy: llmrouter-knn
  routing_strategy_args:
    model_path: /app/models/knn_router
    llm_data_path: /app/config/llm_candidates.json
    hot_reload: true
    reload_interval: 300
```

**Thread Safety**: Uses `threading.RLock()` for thread-safe model access and reloading.

### 3. A2A Gateway

**Responsibility**: Agent-to-Agent protocol support for agent communication with full LiteLLM parity.

**Class**: `A2AGateway`

**Key Methods**:
- `register_agent(agent: A2AAgent)` - Register a new A2A agent
- `unregister_agent(agent_id: str)` - Remove an agent
- `get_agent(agent_id: str)` - Retrieve agent by ID
- `update_agent(agent_id: str, agent: A2AAgent)` - Full update of agent
- `patch_agent(agent_id: str, updates: dict)` - Partial update of agent
- `discover_agents(capability: str, user_id: str, team_id: str)` - Find agents by capability with permission filtering
- `get_agent_card(agent_id: str)` - Get A2A protocol agent card
- `invoke_agent(agent_id: str, message: JSONRPCRequest)` - Invoke agent via JSON-RPC 2.0
- `stream_agent_response(agent_id: str, message: JSONRPCRequest)` - Stream agent response via SSE
- `get_daily_activity(agent_id: str, start_date: date, end_date: date)` - Get agent usage analytics

**Data Model**:
```python
@dataclass
class A2AAgent:
    agent_id: str
    name: str
    description: str
    url: str
    capabilities: list[str]
    metadata: dict[str, Any]
    team_id: str | None = None
    user_id: str | None = None
    is_public: bool = False
    created_at: datetime = None
    updated_at: datetime = None

@dataclass
class JSONRPCRequest:
    jsonrpc: str = "2.0"
    method: str  # "message/send" or "message/stream"
    params: dict[str, Any]
    id: str | int

@dataclass
class A2AMessage:
    role: str  # "user" or "agent"
    parts: list[A2AMessagePart]

@dataclass
class A2AMessagePart:
    type: str  # "text", "file", "data"
    content: str | bytes
    mime_type: str | None = None
```

**API Endpoints**:
- `GET /v1/agents` - List all agents (with permission filtering)
- `POST /v1/agents` - Register new agent (DB persistence)
- `GET /v1/agents/{agent_id}` - Get agent details
- `PUT /v1/agents/{agent_id}` - Full update agent
- `PATCH /v1/agents/{agent_id}` - Partial update agent
- `DELETE /v1/agents/{agent_id}` - Unregister agent
- `POST /v1/agents/{agent_id}/make_public` - Make agent public
- `POST /a2a/{agent_id}` - Invoke agent (JSON-RPC 2.0)
- `GET /v1/agents/{agent_id}/.well-known/agent-card.json` - Get agent card
- `GET /agent/daily/activity` - Agent analytics

**Database Schema**:
```sql
CREATE TABLE a2a_agents (
    agent_id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    url VARCHAR(1024) NOT NULL,
    capabilities JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    team_id VARCHAR(255),
    user_id VARCHAR(255),
    is_public BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_a2a_agents_team ON a2a_agents(team_id);
CREATE INDEX idx_a2a_agents_user ON a2a_agents(user_id);
CREATE INDEX idx_a2a_agents_public ON a2a_agents(is_public);
```

### 4. MCP Gateway

**Responsibility**: Model Context Protocol support for tool and context integration with full LiteLLM parity.

**Class**: `MCPGateway`

**Key Methods**:
- `register_server(server: MCPServer)` - Register a new MCP server
- `unregister_server(server_id: str)` - Remove a server
- `get_server(server_id: str)` - Retrieve server by ID
- `update_server(server_id: str, server: MCPServer)` - Full update of server
- `list_tools()` - List all available tools across servers
- `list_resources()` - List all available resources across servers
- `call_tool(server_id: str, tool_name: str, arguments: dict)` - Invoke an MCP tool
- `get_registry()` - Get MCP registry for discovery
- `check_server_health(server_id: str)` - Check MCP server health
- `get_access_groups()` - List MCP access groups
- `create_oauth_session(server_id: str)` - Create temporary OAuth session
- `authorize_oauth(server_id: str, redirect_uri: str)` - OAuth authorization
- `exchange_oauth_token(server_id: str, code: str)` - OAuth token exchange

**Data Model**:
```python
@dataclass
class MCPServer:
    server_id: str
    name: str
    url: str
    transport: MCPTransport  # streamable_http, sse, stdio
    tools: list[MCPTool]
    resources: list[MCPResource]
    auth_type: str  # none, api_key, bearer_token, oauth2
    oauth_config: MCPOAuthConfig | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    team_id: str | None = None
    created_at: datetime = None
    updated_at: datetime = None

@dataclass
class MCPTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    server_id: str

@dataclass
class MCPResource:
    uri: str
    name: str
    description: str
    mime_type: str
    server_id: str

@dataclass
class MCPOAuthConfig:
    client_id: str
    client_secret: str
    authorization_url: str
    token_url: str
    scopes: list[str]

@dataclass
class MCPToolCallRequest:
    server_id: str
    tool_name: str
    arguments: dict[str, Any]

@dataclass
class MCPToolCallResponse:
    content: list[dict[str, Any]]
    is_error: bool = False
```

**API Endpoints**:
- `GET /v1/mcp/server` - List all servers
- `POST /v1/mcp/server` - Register new server (DB persistence)
- `GET /v1/mcp/server/{server_id}` - Get server details
- `PUT /v1/mcp/server/{server_id}` - Full update server
- `DELETE /v1/mcp/server/{server_id}` - Unregister server
- `GET /v1/mcp/server/health` - Health check all servers
- `GET /v1/mcp/tools` - List all tools
- `GET /mcp/tools/list` - List tools (REST API)
- `POST /mcp/tools/call` - Call tool (REST API)
- `GET /v1/mcp/registry.json` - MCP registry for discovery
- `GET /v1/mcp/access_groups` - List access groups
- `POST /v1/mcp/server/oauth/session` - Create OAuth session
- `GET /v1/mcp/server/oauth/{server_id}/authorize` - OAuth authorize
- `POST /v1/mcp/server/oauth/{server_id}/token` - OAuth token exchange
- `GET /.well-known/oauth-authorization-server` - OAuth discovery

**Database Schema**:
```sql
CREATE TABLE mcp_servers (
    server_id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    url VARCHAR(1024) NOT NULL,
    transport VARCHAR(50) NOT NULL,
    auth_type VARCHAR(50) DEFAULT 'none',
    oauth_config JSONB,
    metadata JSONB DEFAULT '{}',
    team_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE mcp_tools (
    tool_id UUID PRIMARY KEY,
    server_id UUID REFERENCES mcp_servers(server_id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    input_schema JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE mcp_resources (
    resource_id UUID PRIMARY KEY,
    server_id UUID REFERENCES mcp_servers(server_id) ON DELETE CASCADE,
    uri VARCHAR(1024) NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    mime_type VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE mcp_access_groups (
    group_id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    server_ids JSONB DEFAULT '[]',
    team_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_mcp_servers_team ON mcp_servers(team_id);
CREATE INDEX idx_mcp_tools_server ON mcp_tools(server_id);
CREATE INDEX idx_mcp_resources_server ON mcp_resources(server_id);
```

### 5. Skills Gateway (Anthropic Skills)

**Responsibility**: Expose Anthropic's "Skills" (Computer Use, Bash, Text Editor) endpoints with operational enhancements for observability and database backing.

**Implementation**: Direct passthrough to LiteLLM's native `/v1/skills` endpoints with added instrumentation.

**Key Design Decisions**:
- **No Custom Implementation**: Skills endpoints are inherited from LiteLLM without modification
- **Observability Layer**: Emit OpenTelemetry spans for Skills invocations
- **Database Backing**: Use LiteLLM's PostgreSQL persistence for skill registrations (when `database_url` configured)
- **Multi-Account Routing**: Support routing to different Anthropic accounts via `model_list` configuration

**API Endpoints** (LiteLLM native):
- `POST /v1/skills` - Create/invoke skill
- `GET /v1/skills` - List available skills
- `GET /v1/skills/{id}` - Get skill details
- `DELETE /v1/skills/{id}` - Remove skill

**Configuration**:
```yaml
model_list:
  - model_name: claude-3-5-sonnet
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY
```

**Relationship to MCP**:
- **Skills**: Anthropic-specific agentic capabilities (Computer Use, precursor to MCP)
- **MCP**: Open standard for connecting LLMs to tools/data (broader ecosystem)
- **Gateway supports both** simultaneously without conflict

**Observability**:
```python
# Pseudocode: Emit span for Skills invocation
with tracer.start_as_current_span("skills.invoke") as span:
    span.set_attribute("skill.id", skill_id)
    span.set_attribute("skill.type", "computer_use")
    # ... invoke LiteLLM endpoint
    span.set_attribute("skill.duration_ms", duration)
```

**Moat-Mode Support**:
- Skills work in air-gapped environments if Anthropic endpoints are reachable via VPC-private links
- Database-backed persistence ensures skills survive container restarts

### 6. Hot Reload Manager

**Responsibility**: Manage hot reload operations for routing models and configurations.

**Class**: `HotReloadManager`

**Key Methods**:
- `register_router_reload_callback(strategy_name, callback)` - Register reload callback for a strategy
- `reload_router(strategy: str | None)` - Reload specific or all strategies
- `reload_config(force_sync: bool)` - Reload configuration, optionally syncing from remote
- `get_router_info()` - Get information about current routing configuration

**API Endpoints**:
- `POST /router/reload` - Trigger router reload
- `POST /config/reload` - Trigger config reload
- `GET /router/info` - Get router information

### 6. Config Sync Manager

**Responsibility**: Background synchronization of configuration from S3/GCS with ETag-based change detection.

**Class**: `ConfigSyncManager`

**Key Methods**:
- `start()` - Start background sync thread
- `stop()` - Stop background sync
- `force_sync()` - Force immediate sync from remote
- `get_status()` - Get current sync status
- `_get_s3_etag()` - Get S3 object ETag without downloading
- `_download_from_s3_if_changed()` - Download only if ETag changed
- `_trigger_reload()` - Trigger config reload via SIGHUP signal

**Configuration**:
```bash
CONFIG_S3_BUCKET=my-bucket
CONFIG_S3_KEY=config/config.yaml
CONFIG_HOT_RELOAD=true
CONFIG_SYNC_INTERVAL=60
```

**ETag-Based Optimization**: Uses S3 ETag to avoid unnecessary downloads, only downloading when remote config has actually changed.

**API Endpoints**:
- `GET /config/sync/status` - Get sync status

### 7. OpenTelemetry Observability

**Responsibility**: Unified observability via OpenTelemetry for traces, logs, and metrics.

**Components**:
- **Tracer**: Emits spans for routing decisions, LLM calls, cache operations
- **Logger**: Structured logging with trace correlation
- **Meter**: Metrics for request count, latency, errors, costs

**Configuration**:
```yaml
litellm_settings:
  success_callback: ["prometheus", "otel"]
  otel_config:
    endpoint: "http://otel-collector:4317"
    service_name: "litellm-gateway"
    traces_exporter: "otlp"
    logs_exporter: "otlp"
    metrics_exporter: "prometheus"
```

**Semantic Conventions**: Uses OpenTelemetry semantic conventions for:
- HTTP spans: `http.method`, `http.status_code`, `http.route`
- LLM spans: `gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`
- Custom spans: `llm.routing.strategy`, `llm.routing.selected_model`

**Trace Correlation**: All logs include `trace_id` and `span_id` for correlation with distributed traces.

**Exporters**:
- **OTLP**: OpenTelemetry Protocol for traces and logs to collectors (Jaeger, Tempo, CloudWatch)
- **Prometheus**: Pull-based metrics at `/metrics` endpoint
- **Langfuse**: LLM-specific observability platform

### 8. Startup Module

**Responsibility**: Entry point that wires all components together and starts LiteLLM proxy.

**Key Functions**:
- `register_routes_with_litellm()` - Register extension routes with LiteLLM's FastAPI app
- `register_strategies()` - Register LLMRouter strategies with LiteLLM
- `start_config_sync_if_enabled()` - Start background config sync if enabled
- `main()` - Main entry point that orchestrates startup

**Startup Sequence**:
1. Parse command-line arguments
2. Register LLMRouter strategies
3. Start config sync (if enabled)
4. Build litellm command with arguments
5. Execute litellm proxy via `os.execvp()`

### 9. MLOps Loop (OTEL Traces → Training → Hot Reload)

**Responsibility**: Enable continuous improvement of routing models based on production traffic.

**MLOps Pipeline Flow**:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Production Gateway                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Request → Route → LLM → Response                        │   │
│  │  (Emit OTLP spans with routing_decision, model_selected) │   │
│  └────────────────────────┬─────────────────────────────────┘   │
└───────────────────────────┼─────────────────────────────────────┘
                            │ OTLP/gRPC (port 4317)
                            ▼
                   ┌────────────────────┐
                   │ OTEL Collector     │
                   │ (Jaeger/Tempo)     │
                   └────────┬───────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MLOps Training Pipeline                      │
│  1. Extract traces: examples/mlops/scripts/extract_jaeger_traces.py  │
│  2. Convert format: examples/mlops/scripts/convert_traces_to_llmrouter.py  │
│  3. Train model: examples/mlops/scripts/train_router.py        │
│  4. Deploy artifact: examples/mlops/scripts/deploy_model.py    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                    ┌────────────────────┐
                    │ S3 / MinIO / NFS   │
                    │ /app/models/       │
                    └────────┬───────────┘
                             │ ETag change detected
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│               Gateway (Hot Reload Manager)                      │
│  1. _should_reload() detects model file change                 │
│  2. _load_router() loads new model artifact                    │
│  3. Routing uses updated strategy (zero downtime)              │
└─────────────────────────────────────────────────────────────────┘
```

**Inference-Only Runtime Requirement**:
- **Production Container**: Only includes inference dependencies (PyTorch for model loading, NumPy)
- **Training Container**: Separate image with full training stack (scikit-learn, MLflow, training libraries)
- **Model Artifacts**: Pre-trained `.pt` files (pickled models or PyTorch state dicts)
- **Why**: Reduces production image size, attack surface, and deployment complexity

**Training Data Format**:
```json
{
  "query": "What is machine learning?",
  "candidates": ["gpt-4", "claude-3-opus", "llama-3-70b"],
  "selected_model": "gpt-4",
  "outcome": {
    "latency_ms": 850,
    "cost_usd": 0.06,
    "user_rating": 5
  }
}
```

**Docker Compose Setup**:
```yaml
# examples/mlops/docker-compose.mlops.yml
services:
  trainer:
    build:
      context: .
      dockerfile: Dockerfile.trainer
    volumes:
      - ./data:/data
      - ./models:/models
    command: python train_router.py --strategy knn --input /data/traces.json

  deployer:
    build:
      context: .
      dockerfile: Dockerfile.deployer
    volumes:
      - ./models:/models
    environment:
      - S3_BUCKET=llm-models
    command: python deploy_model.py --model /models/knn_router.pt
```

### 10. Moat-Mode Design Patterns

**Responsibility**: Support air-gapped and controlled-egress deployments with standardized protocols.

**Reference**: See [`docs/moat-mode.md`](../../../docs/moat-mode.md) for complete runbook.

**Key Design Constraints**:

1. **No Cloud Dependencies**:
   - ❌ AWS Secrets Manager → ✅ Kubernetes Secrets, HashiCorp Vault (self-hosted)
   - ❌ AWS RDS → ✅ On-premises PostgreSQL with streaming replication
   - ❌ ElastiCache → ✅ Redis Sentinel or on-premises Redis cluster
   - ❌ CloudWatch/X-Ray → ✅ Self-hosted Jaeger/Tempo + Prometheus + Loki

2. **TLS Certificate Management**:
   - **Option A**: TLS termination at load balancer (Nginx) with internal CA certs
   - **Option B**: End-to-end mTLS with client certificates for zero-trust environments

3. **Network Segmentation**:
   ```
   ┌────────────────────────────────────────────────────┐
   │              Moat Perimeter (No Internet)          │
   │                                                    │
   │  ┌────────┐    ┌────────┐    ┌──────────┐        │
   │  │ Nginx  │───▶│Gateway │───▶│PostgreSQL│        │
   │  │  :443  │    │ :4000  │    │  :5432   │        │
   │  └────────┘    └────┬───┘    └──────────┘        │
   │                     │                             │
   │                     ▼                             │
   │              ┌──────────────┐                     │
   │              │ Redis :6379  │                     │
   │              └──────────────┘                     │
   │                     │                             │
   │                     ▼                             │
   │              ┌──────────────────┐                 │
   │              │ OTEL Collector   │                 │
   │              │ :4317 (gRPC)     │                 │
   │              └────────┬─────────┘                 │
   │                       │                           │
   │          ┌────────────┴────────────┐              │
   │          ▼                         ▼              │
   │   ┌──────────┐               ┌──────────┐        │
   │   │  Jaeger  │               │Prometheus│        │
   │   └──────────┘               └──────────┘        │
   └────────────────────────────────────────────────────┘
   ```

4. **Firewall Rules** (example):
   ```bash
   # Allow internal VPC traffic only
   iptables -A INPUT -s 10.0.0.0/8 -j ACCEPT
   iptables -A OUTPUT -d 10.0.0.0/8 -j ACCEPT

   # Allow on-premises LLM provider (via VPC endpoint)
   iptables -A OUTPUT -d <BEDROCK_VPC_ENDPOINT_IP> --dport 443 -j ACCEPT

   # Deny all other egress
   iptables -A OUTPUT -j DROP
   ```

5. **Offline Container Build**:
   ```dockerfile
   # Vendored dependencies (no PyPI access)
   FROM python:3.11-slim
   COPY vendor/python /tmp/vendor/python
   RUN pip install --no-index --find-links=/tmp/vendor/python -r requirements.txt
   ```

**Protocol Compliance**:
| Protocol | Port | Purpose | Moat-Mode Status |
|----------|------|---------|------------------|
| HTTP/HTTPS | 4000 | API endpoint | ✅ Internal TLS |
| SSE | 4000 | Streaming responses | ✅ Same port as HTTP |
| OTLP/gRPC | 4317 | Trace/metrics export | ✅ Internal collector |
| OTLP/HTTP | 4318 | Trace/metrics (alt) | ✅ Internal collector |
| PostgreSQL | 5432 | State persistence | ✅ mTLS supported |
| Redis | 6379 | Caching/rate limit | ✅ AUTH + TLS |
| Prometheus | 9090 | Metrics scrape | ✅ Internal Prometheus |

**Backup/Restore Strategy**:
```bash
# Daily Postgres backup (cron job)
0 2 * * * docker exec postgres pg_dump -U litellm litellm | gzip > /backup/litellm_$(date +\%Y\%m\%d).sql.gz

# Redis AOF persistence (automatic)
redis-server --appendonly yes --appendfsync everysec

# Model artifacts versioning (S3/MinIO)
mc version enable myminio/llm-models
```

**Split-Brain Prevention**:
- Use database-backed leases for critical operations (config reload)
- Circuit breaker pattern: Gateway fails closed if Postgres unreachable
- Redis Sentinel for automatic failover (future enhancement)

## Data Models

### Configuration Schema

```yaml
# Model List - LLM provider configurations
model_list:
  - model_name: string          # Logical model name
    litellm_params:
      model: string              # Provider/model format (e.g., "openai/gpt-4")
      api_key: string            # API key (supports os.environ/ prefix)
      api_base: string           # Optional custom endpoint
      rpm: integer               # Requests per minute limit
      timeout: integer           # Request timeout in seconds
      stream_timeout: integer    # Stream timeout in seconds
    model_info:
      id: string                 # Optional model ID
      mode: string               # Optional mode (e.g., "embedding")

# Router Settings - Routing strategy configuration
router_settings:
  routing_strategy: string       # Strategy name (e.g., "llmrouter-knn")
  routing_strategy_args:
    model_path: string           # Path to trained model
    llm_data_path: string        # Path to LLM candidates JSON
    hot_reload: boolean          # Enable hot reload
    reload_interval: integer     # Reload check interval (seconds)
    model_s3_bucket: string      # Optional S3 bucket for models
    model_s3_key: string         # Optional S3 key for models
  num_retries: integer           # Number of retries on failure
  retry_after: integer           # Seconds to wait before retry
  timeout: integer               # Global timeout
  cache_responses: boolean       # Enable response caching
  redis_host: string             # Redis host for distributed state
  redis_password: string         # Redis password
  redis_port: integer            # Redis port

# General Settings - Gateway configuration
general_settings:
  master_key: string             # Master API key for admin access
  database_url: string           # PostgreSQL connection string
  store_model_in_db: boolean     # Store models in database
  proxy_budget_rescheduler_min_time: integer
  proxy_budget_rescheduler_max_time: integer
  proxy_batch_write_at: integer
  database_connection_pool_limit: integer

# LiteLLM Settings - LiteLLM-specific configuration
litellm_settings:
  cache: boolean                 # Enable response caching
  cache_params:
    type: string                 # Cache backend (e.g., "redis")
    host: string                 # Cache host
    port: integer                # Cache port
    ttl: integer                 # Cache TTL in seconds
  set_verbose: boolean           # Enable verbose logging
  success_callback: list[string] # Success callbacks (e.g., ["prometheus", "langfuse"])
  failure_callback: list[string] # Failure callbacks
  num_retries: integer           # Number of retries
  request_timeout: integer       # Request timeout
  telemetry: boolean             # Enable telemetry
  context_window_fallbacks: list # Fallback models for context limits
  default_team_settings: list    # Per-team settings
  drop_params: boolean           # Drop unsupported parameters

# MCP Servers - Model Context Protocol servers
mcp_servers:
  server_name:
    url: string                  # Server URL
    transport: string            # Transport type (streamable_http, sse, stdio)
    spec_path: string            # Optional OpenAPI spec path
    auth_type: string            # Authentication type (none, api_key, bearer_token, oauth2)
    created_at: datetime
    updated_at: datetime
```

### LLM Candidates JSON

```json
{
  "models": [
    {
      "model_name": "gpt-4",
      "provider": "openai",
      "cost_per_1k_tokens": 0.03,
      "latency_p50_ms": 500,
      "latency_p95_ms": 1200,
      "context_window": 8192,
      "capabilities": ["chat", "function_calling"]
    }
  ]
}
```

### Database Schema

**Virtual Keys Table**:
```sql
CREATE TABLE virtual_keys (
    key_id UUID PRIMARY KEY,
    key_hash VARCHAR(255) UNIQUE NOT NULL,
    key_alias VARCHAR(255),
    team_id VARCHAR(255),
    user_id VARCHAR(255),
    max_budget DECIMAL(10, 2),
    budget_duration VARCHAR(50),
    budget_reset_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP,
    metadata JSONB
);
```

**Request Logs Table**:
```sql
CREATE TABLE request_logs (
    request_id UUID PRIMARY KEY,
    key_id UUID REFERENCES virtual_keys(key_id),
    model_name VARCHAR(255),
    provider VARCHAR(255),
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    cost DECIMAL(10, 6),
    latency_ms INTEGER,
    status VARCHAR(50),
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB
);
```

## Error Handling

### Error Categories

1. **Authentication Errors** (401)
   - Invalid API key
   - Expired API key
   - Missing Authorization header

2. **Authorization Errors** (403)
   - Budget exceeded
   - Model access denied
   - Rate limit exceeded

3. **Validation Errors** (400)
   - Invalid request format
   - Missing required parameters
   - Unsupported model

4. **Provider Errors** (502, 503, 504)
   - Provider API unavailable
   - Provider timeout
   - Provider rate limit

5. **Internal Errors** (500)
   - Routing strategy failure
   - Database connection failure
   - Cache connection failure

### Retry Strategy

```python
# Exponential backoff with jitter
def calculate_retry_delay(attempt: int, base_delay: int = 5) -> int:
    """Calculate retry delay with exponential backoff and jitter."""
    max_delay = base_delay * (2 ** attempt)
    jitter = random.uniform(0, 0.1 * max_delay)
    return min(max_delay + jitter, 60)  # Cap at 60 seconds
```

### Fallback Strategy

```yaml
litellm_settings:
  context_window_fallbacks:
    - gpt-3.5-turbo: [gpt-3.5-turbo-16k]
    - gpt-4: [gpt-4-32k, claude-3-opus]
```

When a request exceeds the context window, the Gateway automatically falls back to the next model in the list.

## Testing Strategy

### Unit Testing

**Scope**: Individual components and functions

**Framework**: pytest

**Coverage Areas**:
- LLMRouter strategy loading and initialization
- Config sync ETag-based change detection
- Hot reload manager callback registration
- A2A agent registration and discovery
- MCP server registration and tool listing
- Authentication and authorization logic
- Cost calculation and budget tracking

**Example Test**:
```python
def test_llmrouter_strategy_initialization():
    """Test that LLMRouter strategies initialize correctly."""
    strategy = LLMRouterStrategyFamily(
        strategy_name="llmrouter-knn",
        model_path="/app/models/knn_router",
        llm_data_path="/app/config/llm_candidates.json",
        hot_reload=True,
        reload_interval=300
    )
    assert strategy.strategy_name == "llmrouter-knn"
    assert strategy.hot_reload is True
    assert strategy.reload_interval == 300
```

### Integration Testing

**Scope**: Component interactions and end-to-end flows

**Framework**: pytest with Docker Compose

**Coverage Areas**:
- LiteLLM + LLMRouter integration
- Redis caching integration
- PostgreSQL persistence integration
- S3 config sync integration
- A2A protocol compliance
- MCP protocol compliance
- Hot reload end-to-end flow

**Example Test**:
```python
@pytest.mark.integration
async def test_routing_with_cache():
    """Test that routing decisions are cached correctly."""
    # First request - cache miss
    response1 = await client.post("/v1/chat/completions", json={
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}]
    })
    assert response1.status_code == 200

    # Second identical request - cache hit
    response2 = await client.post("/v1/chat/completions", json={
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}]
    })
    assert response2.status_code == 200
    assert response2.headers.get("X-Cache-Hit") == "true"
```

### Load Testing

**Scope**: Performance and scalability validation

**Framework**: Locust or k6

**Metrics**:
- Requests per second (RPS)
- P50, P95, P99 latency
- Error rate
- Cache hit rate
- Database connection pool utilization

**Target Performance**:
- 1000 RPS sustained
- P95 latency < 100ms (excluding LLM call)
- Error rate < 0.1%
- Cache hit rate > 80% for repeated queries

### Property-Based Testing

**Scope**: Universal properties that should hold across all inputs

**Framework**: Hypothesis (Python)

**Note**: Property-based testing will be defined in the Correctness Properties section below.


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Request Forwarding Correctness

*For any* valid client request to a supported endpoint with a configured LLM provider, the Gateway should successfully forward the request to the provider using LiteLLM's unified interface and return a response in the expected format.

**Validates: Requirements 1.2**

### Property 2: Authentication Enforcement

*For any* request to the Gateway when master_key is configured, the request should be accepted if and only if it includes a valid API key (either master_key or a valid virtual key from the database).*
