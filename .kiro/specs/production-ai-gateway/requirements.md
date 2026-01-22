# Requirements Document: Production AI Gateway with LiteLLM + LLMRouter

## Introduction

This document specifies the requirements for a production-ready AI Gateway that serves as an **enhancement layer on top of the LiteLLM Proxy**. The system integrates LiteLLM (unified LLM API gateway supporting 100+ providers) with LLMRouter (ML-based intelligent routing with 18+ strategies) into a single, production-hardened container.

**Project Positioning:** This gateway enhances—not replaces—the LiteLLM Proxy. We inherit all LiteLLM capabilities (authentication, caching, rate limiting, provider integrations) and extend them with ML routing, hot-reload, standardized HA protocols, and agentic gateway support.

The system provides enterprise-grade features including high availability, persistence, observability, MLOps training pipelines, and support for modern protocols (A2A, MCP, Skills).

## Glossary

- **LiteLLM**: Unified LLM API gateway providing OpenAI-compatible interface to 100+ LLM providers
- **LLMRouter**: ML-based routing library with 18+ intelligent routing strategies
- **Gateway**: The integrated system combining LiteLLM and LLMRouter as an enhancement layer
- **Enhancement Layer**: The philosophy of extending (not replacing) LiteLLM core functionality
- **A2A**: Agent-to-Agent protocol (Google's standard for agent communication)
- **MCP**: Model Context Protocol (open standard for connecting LLMs to tools/data sources)
- **Skills**: Anthropic-specific agentic capabilities (Computer Use, Bash, Text Editor) exposed via `/v1/skills` endpoints
- **Router_Strategy**: Algorithm for selecting optimal LLM based on query characteristics
- **Hot_Reload**: Dynamic configuration/model updates without service restart
- **Config_Sync**: Background process synchronizing configuration from remote storage (S3/MinIO/GCS)
- **MLOps_Pipeline**: Training and evaluation system for routing models based on OTLP traces
- **MLOps_Loop**: OTEL traces → training → model deployment → hot reload cycle
- **Inference-Only_Runtime**: Production container with only model loading dependencies (no training libraries)
- **HA_Stack**: High availability infrastructure (Redis, PostgreSQL, load balancer)
- **Moat-Mode**: Air-gapped or controlled-egress deployment with no external internet dependencies
- **OTLP**: OpenTelemetry Protocol for exporting traces, metrics, and logs
- **ETag**: HTTP entity tag used for detecting S3 object changes without downloading

## Requirements

### Requirement 1: Core LiteLLM Integration

**User Story:** As a developer, I want to access 100+ LLM providers through a single OpenAI-compatible API, so that I can switch providers without changing my application code.

#### Acceptance Criteria

1. THE Gateway SHALL expose all LiteLLM proxy endpoints including `/chat/completions`, `/embeddings`, `/images/generations`, `/audio/transcriptions`, `/audio/speech`, `/moderations`, `/batches`, `/rerank`, `/a2a`, and `/messages`
2. WHEN a client makes a request to any supported endpoint, THE Gateway SHALL forward it to the configured LLM provider using LiteLLM's unified interface
3. THE Gateway SHALL support authentication via API keys configured in the master_key setting
4. THE Gateway SHALL support all LiteLLM provider formats including `openai/`, `anthropic/`, `bedrock/`, `azure/`, `vertex_ai/`, and 100+ other providers
5. THE Gateway SHALL load model configurations from a YAML config file specifying model_list, litellm_params, and provider credentials

### Requirement 2: LLMRouter Strategy Integration

**User Story:** As a platform engineer, I want to use ML-based routing strategies to select the optimal LLM for each request, so that I can optimize for cost, latency, and quality.

#### Acceptance Criteria

1. THE Gateway SHALL register all 18+ LLMRouter strategies as available routing options including `llmrouter-knn`, `llmrouter-svm`, `llmrouter-mlp`, `llmrouter-mf`, `llmrouter-elo`, `llmrouter-hybrid`, `llmrouter-graph`, `llmrouter-automix`, `llmrouter-causallm`, `llmrouter-routerdc`, and multi-round/personalized variants
2. WHEN router_settings.routing_strategy is set to a `llmrouter-*` value, THE Gateway SHALL use the corresponding LLMRouter strategy for model selection
3. THE Gateway SHALL load routing models from the path specified in routing_strategy_args.model_path
4. THE Gateway SHALL load LLM candidate metadata from routing_strategy_args.llm_data_path
5. WHEN a routing strategy is configured, THE Gateway SHALL invoke it with the user query and return the selected model name
6. THE Gateway SHALL fall back to LiteLLM's built-in strategies (`simple-shuffle`, `least-busy`, `latency-based-routing`, `cost-based-routing`, `usage-based-routing`) when routing_strategy does not start with `llmrouter-`

### Requirement 3: Hot Reload and Configuration Sync

**User Story:** As a DevOps engineer, I want to update routing strategies and configurations without restarting the service, so that I can deploy changes with zero downtime.

#### Acceptance Criteria

1. WHEN routing_strategy_args.hot_reload is true, THE Gateway SHALL periodically check for model updates at the interval specified by routing_strategy_args.reload_interval
2. WHEN a model file's modification time or ETag changes, THE Gateway SHALL reload the routing strategy with the new model
3. WHEN CONFIG_HOT_RELOAD environment variable is true, THE Config_Sync SHALL start a background thread monitoring configuration changes
4. WHEN routing_strategy_args.model_s3_bucket and model_s3_key are configured, THE Gateway SHALL download models from S3 and monitor for updates via ETag
5. WHEN a configuration file changes on S3/GCS, THE Config_Sync SHALL download the new config and trigger a Gateway reload
6. THE Gateway SHALL expose a `/hot-reload/status` endpoint returning the current reload state and last update timestamp

### Requirement 4: High Availability Infrastructure

**User Story:** As a platform architect, I want to deploy the gateway with Redis and PostgreSQL for distributed state and persistence, so that the system can scale horizontally and survive restarts.

#### Acceptance Criteria

1. WHEN database_url is configured, THE Gateway SHALL persist all virtual keys, user data, and request logs to PostgreSQL
2. WHEN cache_params.type is `redis`, THE Gateway SHALL use Redis for response caching with the configured TTL
3. WHEN router_settings.redis_host is configured, THE Gateway SHALL use Redis for distributed rate limiting and routing state
4. THE Gateway SHALL support connection pooling with database_connection_pool_limit setting
5. WHEN multiple Gateway instances are deployed, THE HA_Stack SHALL distribute requests via load balancer (Nginx) and share state via Redis
6. THE Gateway SHALL expose `/health/liveliness` and `/health/readiness` endpoints for orchestrator health checks

### Requirement 5: MLOps Training Pipeline (OTEL Traces → Training → Hot Reload)

**User Story:** As a machine learning engineer, I want to train and evaluate custom routing models on my workload data from OpenTelemetry traces, so that I can optimize routing decisions for my specific use case and hot-reload models without downtime.

**MLOps Loop:**
1. **Trace Collection**: Gateway emits OTLP traces with routing decisions and outcomes
2. **Training**: Extract traces → convert to LLMRouter format → train models
3. **Deployment**: Upload model artifact to S3/local storage
4. **Hot Reload**: Gateway detects new model via ETag/mtime and reloads without restart

#### Acceptance Criteria

1. THE MLOps_Pipeline SHALL provide a Docker Compose setup in `examples/mlops` for training routing models
2. THE MLOps_Pipeline SHALL support training all LLMRouter strategy types (KNN, SVM, MLP, Matrix Factorization, ELO, Hybrid, Graph, etc.)
3. WHEN training data is provided in the LLMRouter format, THE MLOps_Pipeline SHALL train a routing model and save it to the configured output path
4. THE MLOps_Pipeline SHALL support evaluation metrics including accuracy, latency, cost, and quality scores
5. THE MLOps_Pipeline SHALL generate model artifacts compatible with the Gateway's hot reload mechanism
6. THE MLOps_Pipeline SHALL support extraction of training data from OpenTelemetry traces (Jaeger, Tempo, CloudWatch X-Ray)
7. THE MLOps_Pipeline SHALL include scripts for converting OTLP trace data to LLMRouter training format
8. THE Gateway SHALL support **inference-only runtime requirements**: Model artifacts must be pre-trained and deployable without training dependencies (scikit-learn, torch for training) in the production container
9. THE Gateway production image SHALL NOT include training dependencies (PyTorch, scikit-learn model training libraries), only inference dependencies
10. WHEN routing_strategy_args.model_s3_bucket is configured, THE MLOps_Pipeline MAY upload trained models to S3/MinIO for Gateway hot-reload

### Requirement 6: Observability and Tracing

**User Story:** As a site reliability engineer, I want distributed tracing, metrics collection, and structured logging via OpenTelemetry, so that I can monitor performance and debug issues across the gateway with a unified observability stack.

#### Acceptance Criteria

1. WHEN litellm_settings.success_callback includes `prometheus`, THE Gateway SHALL expose Prometheus metrics at `/metrics`
2. THE Gateway SHALL support OpenTelemetry tracing with OTLP exporters for Jaeger, Tempo, and CloudWatch X-Ray
3. THE Gateway SHALL support OpenTelemetry logging with OTLP exporters for centralized log aggregation
4. WHEN observability is configured, THE Gateway SHALL emit spans for routing decisions, LLM calls, and cache hits/misses
5. WHEN observability is configured, THE Gateway SHALL emit structured logs via OpenTelemetry with correlation IDs linking logs to traces
6. THE Gateway SHALL track metrics including request count, latency percentiles (P50, P95, P99), error rates, and cost per request
7. WHEN litellm_settings.success_callback includes `langfuse`, THE Gateway SHALL send request traces to Langfuse for analysis
8. THE Gateway SHALL support per-team observability settings via default_team_settings configuration
9. THE Gateway SHALL use OpenTelemetry semantic conventions for spans, logs, and metrics to ensure compatibility with observability backends

### Requirement 7: A2A Agent Gateway

**User Story:** As an agent developer, I want to expose my agents via the A2A protocol, so that they can communicate with other A2A-compatible agents.

#### Acceptance Criteria

1. WHEN A2A_GATEWAY_ENABLED is true, THE Gateway SHALL expose A2A protocol endpoints at `/a2a/{agent_id}`
2. THE Gateway SHALL support A2A agent registration via configuration or API at `/v1/agents` endpoint
3. WHEN an A2A SendMessageRequest is received, THE Gateway SHALL forward it to the configured agent backend
4. THE Gateway SHALL return A2A-compliant responses including agent cards, message responses, and streaming updates
5. THE Gateway SHALL support A2A authentication via virtual keys in the Authorization header
6. THE Gateway SHALL expose `/a2a/{agent_id}/.well-known/agent-card.json` endpoint returning the agent's capability card in A2A protocol format
7. WHEN database_url is configured, THE Gateway SHALL persist A2A agent registrations to PostgreSQL
8. WHEN a POST request is made to `/a2a/{agent_id}`, THE Gateway SHALL process JSON-RPC 2.0 messages and invoke the agent
9. WHEN the JSON-RPC method is `message/send`, THE Gateway SHALL forward the message to the agent and return the response
10. WHEN the JSON-RPC method is `message/stream`, THE Gateway SHALL forward the message and stream the response using Server-Sent Events
11. THE Gateway SHALL support PUT requests to `/v1/agents/{agent_id}` for full agent updates
12. THE Gateway SHALL support PATCH requests to `/v1/agents/{agent_id}` for partial agent updates
13. WHEN listing agents via GET `/v1/agents`, THE Gateway SHALL filter results based on user permissions and team membership
14. THE Gateway SHALL expose `/agent/daily/activity` endpoint for agent usage analytics when database_url is configured

### Requirement 8: MCP Server Gateway

**User Story:** As a tool developer, I want to expose MCP servers through the gateway, so that LLMs can access external tools and context.

**Context:** This requirement covers the **Model Context Protocol** (an open standard for connecting LLMs to data sources/tools). This is distinct from Anthropic Skills (see Requirement 9).

#### Acceptance Criteria

1. WHEN MCP_GATEWAY_ENABLED is true, THE Gateway SHALL expose MCP protocol endpoints at `/v1/mcp/server` and `/mcp/{server_name}`
2. THE Gateway SHALL support MCP server registration via mcp_servers configuration section or POST to `/v1/mcp/server`
3. WHEN mcp_servers are configured, THE Gateway SHALL load tool definitions and make them available to LLMs
4. WHEN a `/chat/completions` request includes `tools` with type `mcp`, THE Gateway SHALL invoke the specified MCP server
5. THE Gateway SHALL support MCP transports including `streamable_http`, `sse`, and `stdio`
6. THE Gateway SHALL support OpenAPI-to-MCP conversion for REST APIs via spec_path configuration
7. WHEN database_url is configured, THE Gateway SHALL persist MCP server registrations to PostgreSQL
8. THE Gateway SHALL expose `/mcp/tools/call` POST endpoint for invoking MCP tools directly
9. THE Gateway SHALL expose `/mcp/tools/list` GET endpoint for listing available tools
10. THE Gateway SHALL support OAuth 2.0 authentication for MCP servers via `/v1/mcp/server/oauth/{server_id}/authorize` and `/v1/mcp/server/oauth/{server_id}/token` endpoints
11. THE Gateway SHALL expose `/.well-known/oauth-authorization-server` endpoint for OAuth discovery
12. THE Gateway SHALL expose `/v1/mcp/registry.json` endpoint for MCP server discovery
13. THE Gateway SHALL expose `/v1/mcp/server/health` GET endpoint for checking MCP server health status
14. THE Gateway SHALL support PUT requests to `/v1/mcp/server/{server_id}` for full server updates
15. THE Gateway SHALL support MCP access groups via `/v1/mcp/access_groups` endpoint for permission management

### Requirement 9: Anthropic Skills Endpoint Support

**User Story:** As an agent developer, I want to use Anthropic's "Skills" (Computer Use, Bash, Text Editor) natively, so that I can leverage agentic capabilities without custom integration work.

**Context:** Skills are Anthropic-specific agentic capabilities that predate MCP. The gateway inherits these endpoints directly from LiteLLM and provides operational enhancements (observability, database backing, routing).

#### Acceptance Criteria

1. THE Gateway SHALL expose `/v1/skills` endpoints inherited from LiteLLM Proxy without modification
2. THE Gateway SHALL support POST `/v1/skills` for creating/invoking skills
3. THE Gateway SHALL support GET `/v1/skills` for listing available skills
4. THE Gateway SHALL support GET `/v1/skills/{id}` for retrieving skill details
5. THE Gateway SHALL support DELETE `/v1/skills/{id}` for removing skills
6. WHEN database_url is configured, THE Gateway SHALL persist skill registrations to PostgreSQL (LiteLLM functionality)
7. THE Gateway SHALL support routing Skills requests to multiple Anthropic accounts via model_list configuration
8. THE Gateway SHALL emit OpenTelemetry spans for Skills invocations with attributes including skill_id, duration, and status
9. WHEN moat-mode is enabled, THE Gateway SHALL support Skills operations without external internet access (using configured Anthropic endpoints)
10. THE Gateway documentation SHALL clearly distinguish Skills (Anthropic-specific) from MCP (open standard) and explain when to use each

### Requirement 9: Multi-Architecture Docker Support

**User Story:** As a deployment engineer, I want to run the gateway on both x86 and ARM architectures, so that I can deploy on various cloud platforms and edge devices.

#### Acceptance Criteria

1. THE Gateway SHALL provide Docker images for both `linux/amd64` and `linux/arm64` platforms
2. THE Gateway SHALL use multi-stage builds to minimize image size
3. THE Gateway SHALL run as a non-root user (UID 1000) for security
4. THE Gateway SHALL use tini as init process for proper signal handling and zombie reaping
5. THE Gateway SHALL include health check configuration in the Dockerfile
6. THE Gateway SHALL publish images to GitHub Container Registry with version tags and `latest` tag

### Requirement 10: AWS Cloud Integration

**User Story:** As a cloud architect, I want to deploy the gateway on AWS with native integrations, so that I can leverage AWS services for security, storage, and observability.

#### Acceptance Criteria

1. THE Gateway SHALL support AWS Bedrock models via `bedrock/` provider prefix
2. WHEN deployed on AWS, THE Gateway SHALL support IAM role-based authentication for Bedrock
3. THE Gateway SHALL support S3 for configuration and model storage with automatic sync
4. THE Gateway SHALL support AWS Secrets Manager for API key management via `os.environ/` prefix
5. THE Gateway SHALL support CloudWatch X-Ray tracing via ADOT sidecar integration
6. THE Gateway SHALL support deployment on ECS Fargate, EKS, and App Runner with provided configuration examples

### Requirement 11: Security and Authentication

**User Story:** As a security engineer, I want robust authentication and authorization, so that only authorized users can access the gateway.

#### Acceptance Criteria

1. WHEN general_settings.master_key is configured, THE Gateway SHALL require valid API keys for all requests
2. THE Gateway SHALL support virtual keys with per-key budgets, rate limits, and model access controls
3. THE Gateway SHALL support team-based access control with per-team settings
4. THE Gateway SHALL validate API keys against the database when database_url is configured
5. THE Gateway SHALL support budget tracking with max_budget and budget_duration settings
6. THE Gateway SHALL reject requests exceeding rate limits with appropriate HTTP 429 responses

### Requirement 12: Cost Tracking and Management

**User Story:** As a finance manager, I want to track LLM costs per user, team, and model, so that I can manage spending and charge back to business units.

#### Acceptance Criteria

1. THE Gateway SHALL calculate costs for each request based on model pricing and token usage
2. WHEN database_url is configured, THE Gateway SHALL persist cost data per request, user, and team
3. THE Gateway SHALL support budget limits per virtual key with automatic enforcement
4. THE Gateway SHALL expose cost metrics via Prometheus including total spend, spend per model, and spend per team
5. WHEN litellm_settings.success_callback includes cost tracking services, THE Gateway SHALL send cost data to external systems
6. THE Gateway SHALL support budget reset intervals via budget_duration setting (e.g., `30d`, `1m`)

### Requirement 13: Caching and Performance

**User Story:** As a performance engineer, I want response caching to reduce latency and costs, so that identical requests return instantly without calling the LLM.

#### Acceptance Criteria

1. WHEN litellm_settings.cache is true, THE Gateway SHALL cache LLM responses based on request parameters
2. WHEN cache_params.type is `redis`, THE Gateway SHALL use Redis as the cache backend
3. THE Gateway SHALL respect cache_params.ttl for cache expiration (in seconds)
4. WHEN enable_caching_on_provider_specific_optional_params is true, THE Gateway SHALL include provider-specific parameters in cache keys
5. THE Gateway SHALL emit cache hit/miss metrics to observability systems
6. THE Gateway SHALL support cache invalidation via API or TTL expiration

### Requirement 14: Error Handling and Retries

**User Story:** As a reliability engineer, I want automatic retries and fallbacks, so that transient failures don't impact user experience.

#### Acceptance Criteria

1. WHEN router_settings.num_retries is configured, THE Gateway SHALL retry failed requests up to the specified count
2. WHEN a retry is triggered, THE Gateway SHALL wait router_settings.retry_after seconds before retrying
3. WHEN litellm_settings.context_window_fallbacks is configured, THE Gateway SHALL fall back to alternative models when context limits are exceeded
4. THE Gateway SHALL respect timeout settings at both global (request_timeout) and per-model (timeout) levels
5. WHEN all retries are exhausted, THE Gateway SHALL return an appropriate error response with details
6. THE Gateway SHALL emit retry and fallback metrics to observability systems

### Requirement 15: Logging and Debugging

**User Story:** As a developer, I want detailed structured logging via OpenTelemetry for debugging, so that I can troubleshoot issues quickly with correlated logs and traces.

#### Acceptance Criteria

1. WHEN litellm_settings.set_verbose is true, THE Gateway SHALL emit detailed debug logs via OpenTelemetry
2. THE Gateway SHALL log all routing decisions including strategy used and model selected with OpenTelemetry structured logging
3. THE Gateway SHALL log request/response metadata without logging sensitive prompt content when store_model_in_db is false
4. THE Gateway SHALL support OpenTelemetry structured logging with OTLP exporters for centralized log aggregation
5. THE Gateway SHALL include trace context (trace_id, span_id) in all log entries for correlation with distributed traces
6. THE Gateway SHALL emit startup logs confirming registered strategies, routes, and configuration
7. THE Gateway SHALL log errors with stack traces and context for debugging, including trace correlation IDs

---

## Non-Goals / Out of Scope

This section clarifies what the gateway does **NOT** aim to provide, to avoid duplicating LiteLLM core features and to maintain a clear boundary as an enhancement layer.

### Not Replacing LiteLLM Core Features

1. **Provider Integrations**: We do NOT fork or modify LiteLLM's provider implementations. All provider compatibility (OpenAI, Anthropic, Bedrock, Azure, 100+ others) is inherited directly from upstream LiteLLM.
2. **Authentication Mechanisms**: We use LiteLLM's existing authentication (master_key, virtual keys, team-based access control) without modification.
3. **Rate Limiting & Budget Enforcement**: We rely on LiteLLM's built-in rate limiting and budget tracking, not custom implementations.
4. **Caching Logic**: Response caching is handled by LiteLLM's cache layer (Redis, in-memory). We do not implement custom caching.
5. **Request Validation**: Parameter validation and OpenAI compatibility are maintained by LiteLLM.

### Not Providing LLM Training/Hosting

1. **LLM Model Training**: This gateway does NOT train foundation models (GPT, Claude, etc.). It only trains **routing models** to select among pre-existing LLMs.
2. **LLM Model Hosting**: We do not host or serve LLMs. The gateway routes requests to external providers or on-premises endpoints.
3. **Fine-Tuning LLMs**: We support routing to fine-tuned models (e.g., `ft:gpt-4:org:model:id`) but do not perform the fine-tuning.

### Not Replacing Observability Backends

1. **Trace Storage**: We export OTLP traces but do not store them. Use Jaeger, Tempo, or CloudWatch X-Ray.
2. **Metrics Storage**: We expose Prometheus metrics but do not store them. Use Prometheus, Grafana Cloud, or CloudWatch.
3. **Log Aggregation**: We export structured logs but do not aggregate them. Use Loki, CloudWatch Logs, or Elasticsearch.

### Not Providing Infrastructure Management

1. **Database Provisioning**: We require PostgreSQL but do not provision it. Use RDS, managed Postgres, or your own cluster.
2. **Redis Provisioning**: We require Redis but do not manage it. Use ElastiCache, Redis Sentinel, or your own cluster.
3. **Load Balancer Configuration**: We document deployment behind Nginx/HAProxy/ALB but do not provide the load balancer itself.
4. **Container Orchestration**: We provide Docker images and Helm charts but do not manage Kubernetes/ECS clusters.

### Not Implementing Custom Protocols

1. **gRPC Gateway**: We support HTTP/REST and SSE, not gRPC endpoints (LiteLLM uses HTTP).
2. **GraphQL**: We do not provide a GraphQL interface to the gateway.
3. **WebSocket**: Streaming uses SSE, not WebSockets.

### Not Supporting Legacy LiteLLM Versions

1. **LiteLLM Version Guarantee**: We track a specific LiteLLM version (via git submodule) and test against it. Arbitrary LiteLLM versions are not supported.
2. **Backward Compatibility**: We follow LiteLLM's breaking changes. If LiteLLM makes a breaking change, we may also introduce breaking changes.

### Not Providing Managed Services

1. **SaaS Offering**: This is a self-hosted solution, not a managed cloud service.
2. **Support/SLA**: This is an open-source project without commercial support or uptime guarantees.
3. **Compliance Certifications**: We document security features and moat-mode deployment but do not provide SOC 2, HIPAA, or FedRAMP certifications.

### Explicitly Out of Scope for Initial Release

1. **Multi-Region Active-Active**: Initial release supports single-region HA. Multi-region is future work.
2. **OIDC/SAML Authentication**: Documented as future roadmap (Q2 2026). Use API keys or external proxy (oauth2-proxy) in the interim.
3. **Built-in PKI/CA**: Certificate management relies on external CA (HashiCorp Vault, EJBCA, on-prem CA). We do not provide a built-in CA.
4. **Automated Certificate Rotation**: Must be implemented externally (cert-manager for K8s, AWS ACM for cloud). We support manual cert updates.
5. **Web UI for Management**: Configuration is via YAML and REST API. No web dashboard is provided (use LiteLLM's UI if needed).

---

## Related Documentation

- **[`docs/moat-mode.md`](../../../docs/moat-mode.md)**: Complete runbook for air-gapped deployments, network security, and compliance
- **[`docs/skills-gateway.md`](../../../docs/skills-gateway.md)**: Anthropic Skills endpoint usage and configuration
- **[`docs/mcp-gateway.md`](../../../docs/mcp-gateway.md)**: Model Context Protocol server registration and tool invocation
- **[`docs/a2a-gateway.md`](../../../docs/a2a-gateway.md)**: Agent-to-Agent protocol for multi-agent systems
- **[`docs/mlops-training.md`](../../../docs/mlops-training.md)**: Training routing models from OTLP traces
- **[`docs/observability.md`](../../../docs/observability.md)**: OpenTelemetry configuration and semantic conventions
- **[`docs/high-availability.md`](../../../docs/high-availability.md)**: HA deployment patterns (Redis Sentinel, Postgres replication, load balancing)
- **[`docs/litellm-cloud-native-enhancements.md`](../../../docs/litellm-cloud-native-enhancements.md)**: Prioritized backlog of cloud-native enhancements (HA, Security, Observability)

### Requirement 16: Moat-Mode / Air-Gapped Deployment Support

**User Story:** As a security or compliance engineer, I want to deploy the gateway in air-gapped or controlled-egress environments, so that I can meet data residency, zero-trust, and regulatory requirements.

**Context:** "Moat-mode" refers to deployments with no or limited external internet access. See [`docs/moat-mode.md`](../../../docs/moat-mode.md) for complete runbook.

#### Acceptance Criteria

1. THE Gateway SHALL operate without external internet access when all dependencies are sourced from internal networks
2. THE Gateway SHALL support on-premises LLM providers via VPC-private endpoints (e.g., AWS Bedrock PrivateLink, self-hosted vLLM)
3. THE Gateway SHALL support TLS termination at load balancer OR end-to-end mTLS with certificates from internal CA
4. THE Gateway SHALL support authentication via local mechanisms: API keys stored in PostgreSQL, Kubernetes Secrets, or self-hosted Vault
5. THE Gateway SHALL support state persistence via on-premises PostgreSQL and Redis with NO cloud-managed services (no AWS RDS, ElastiCache, etc.)
6. THE Gateway SHALL support observability via self-hosted OpenTelemetry collector with OTLP export to Jaeger/Tempo/Prometheus, with NO cloud telemetry (CloudWatch, X-Ray, Datadog SaaS)
7. THE Gateway SHALL support configuration and model storage via self-hosted S3-compatible storage (MinIO, Ceph) or NFS mounts
8. THE Gateway SHALL support air-gapped container builds via vendored dependencies (no PyPI access during build)
9. THE Gateway SHALL support database migrations and schema updates compatible with zero-downtime rolling deployments
10. THE Gateway documentation SHALL include firewall rules, network port requirements, and certificate management procedures for moat-mode deployments

### Requirement 17: Protocol and Interface Compatibility

**User Story:** As a platform integrator, I want standardized protocol support and predictable interfaces, so that I can integrate the gateway with existing infrastructure (load balancers, observability backends, databases).

#### Acceptance Criteria

1. THE Gateway SHALL support HTTP/1.1 and HTTP/2 for all API endpoints
2. THE Gateway SHALL support Server-Sent Events (SSE) for streaming responses at `/chat/completions` (with `stream: true`) and A2A streaming endpoints
3. THE Gateway SHALL support OpenTelemetry Protocol (OTLP) for traces, metrics, and logs via gRPC (port 4317) and HTTP (port 4318)
4. THE Gateway SHALL support PostgreSQL 13+ for state persistence with connection pooling via pgbouncer or native pooling
5. THE Gateway SHALL support Redis 6+ for caching and rate limiting with Sentinel support for HA deployments
6. THE Gateway SHALL expose Prometheus-compatible metrics at `/metrics` endpoint
7. THE Gateway SHALL support S3-compatible object storage (AWS S3, MinIO, Ceph S3 Gateway) for configuration and model artifacts
8. THE Gateway SHALL support reverse proxy deployment behind Nginx, HAProxy, or cloud load balancers (ALB, NLB) with health checks at `/health/liveliness` and `/health/readiness`
9. THE Gateway SHALL support container orchestrators (Docker Compose, Kubernetes, ECS, EKS) with standard OCI container interfaces
10. WHEN deployed in Kubernetes, THE Gateway SHALL support HPA (Horizontal Pod Autoscaling) based on CPU/memory/custom metrics
