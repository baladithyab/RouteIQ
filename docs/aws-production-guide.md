# RouteIQ: AWS Production Deployment Guide

> A complete guide to deploying RouteIQ as a production AI gateway on AWS using cloud-native services.

## What Is RouteIQ?

RouteIQ is a **production-grade AI gateway** that sits between your applications and LLM providers (Amazon Bedrock, OpenAI, Anthropic, Azure OpenAI, etc.). It provides a single OpenAI-compatible API endpoint while adding intelligent routing, cost controls, security, observability, and multi-protocol support.

Built on [LiteLLM](https://github.com/BerriAI/litellm) (200+ model proxy) and [LLMRouter](https://github.com/ulab-uiuc/LLMRouter) (ML-based routing), RouteIQ extends both with enterprise features for running AI workloads at scale.

```
┌──────────────┐     ┌─────────────────────────────────────────┐     ┌──────────────────┐
│  Your Apps   │     │             RouteIQ Gateway              │     │   LLM Providers  │
│              │     │                                         │     │                  │
│  Backend     │────>│  Auth ─> Policy ─> Route ─> Guardrails  │────>│  Amazon Bedrock  │
│  Frontend    │     │                                         │     │  OpenAI          │
│  Agents      │<────│  Cache <─ Observe <─ Stream <─ Respond  │<────│  Anthropic       │
│  MCP Clients │     │                                         │     │  Azure OpenAI    │
│  A2A Agents  │     └─────────────────────────────────────────┘     │  SageMaker       │
└──────────────┘                                                     └──────────────────┘
```

---

## Feature Overview

### 1. Unified LLM API (OpenAI-Compatible)

**What it does**: Expose a single `/v1/chat/completions` endpoint that routes to 200+ models across providers. Applications code to one API format regardless of which provider serves the request.

**Use cases**:
- Avoid vendor lock-in across Bedrock, OpenAI, Anthropic, Azure
- Switch providers without changing application code
- Run the same prompt against multiple providers for comparison

**AWS integration**: Gateway runs on ECS Fargate behind an ALB. Models served by Amazon Bedrock use IAM roles (no API keys needed).

### 2. ML-Powered Intelligent Routing (18+ Strategies)

**What it does**: Route each request to the optimal model based on ML predictions of quality, cost, and latency. Strategies include KNN, SVM, MLP, Matrix Factorization, ELO, Graph Neural Networks, and hybrid approaches.

**Use cases**:
- Route simple queries to cheaper models (Nova Micro) and complex ones to powerful models (Claude Opus)
- A/B test routing strategies with traffic splitting
- Hot-swap models at runtime without restarts
- Cost-aware routing: cheapest model above a quality threshold

**AWS integration**: ML model artifacts stored in S3. Hot-reload watches S3 for updated artifacts. SageMaker or self-hosted MLflow for experiment tracking.

**Strategies available**:

| Strategy | Type | Best For |
|----------|------|----------|
| `llmrouter-knn` | K-Nearest Neighbors | General routing, inference-only (no training) |
| `llmrouter-svm` | Support Vector Machine | Binary quality classification |
| `llmrouter-mlp` | Multi-Layer Perceptron | Complex quality prediction |
| `llmrouter-mf` | Matrix Factorization | Collaborative filtering patterns |
| `llmrouter-elo` | ELO Rating | Competitive model ranking |
| `llmrouter-hybrid` | Probabilistic Hybrid | Ensemble of multiple signals |
| `llmrouter-causallm` | Transformer-based | Sequence-aware routing |
| `llmrouter-cost-aware` | Cost-Quality Pareto | Budget-constrained environments |
| `llmrouter-custom` | User-defined | Domain-specific routing logic |
| `simple-shuffle` | Random | Load balancing baseline |
| `least-busy` | Load-based | Spread active requests |
| `latency-based-routing` | Latency-based | Minimize response time |

### 3. MCP Gateway (Model Context Protocol)

**What it does**: Acts as a centralized MCP server registry and proxy. Claude Desktop, IDEs, and other MCP clients connect to one gateway endpoint that federates requests across multiple upstream MCP servers.

**Use cases**:
- Centralize tool access for Claude Desktop across your organization
- Register internal tools (database queries, APIs, code search) as MCP servers
- Apply auth, rate limiting, and audit logging to all tool calls
- Expose tools via JSON-RPC, SSE, or REST depending on client capabilities

**Surfaces**:

| Surface | Endpoint | Protocol | Client |
|---------|----------|----------|--------|
| JSON-RPC | `POST /mcp` | MCP JSON-RPC 2.0 | Claude Desktop, IDEs |
| SSE | `GET /mcp/sse` | Server-Sent Events | Streaming MCP clients |
| REST | `GET/POST /llmrouter/mcp/*` | HTTP REST | Custom integrations |
| Parity | `GET/POST /v1/mcp/*` | HTTP REST | Upstream-compatible |
| Proxy | `POST /mcp-proxy/*` | Protocol proxy | Alternative transports |

**AWS integration**: MCP servers can be ECS services, Lambda functions, or any HTTP endpoint in your VPC.

### 4. A2A Gateway (Agent-to-Agent Protocol)

**What it does**: Implements the [A2A protocol](https://google.github.io/a2a/) for multi-agent communication. Register agents, discover capabilities via Agent Cards, and invoke agents with streaming support.

**Use cases**:
- Build multi-agent systems where specialized agents collaborate
- Agent discovery via `/.well-known/agent.json`
- Streaming agent responses with SSE
- Centralized agent registry with auth and audit

**AWS integration**: Agents can be ECS tasks, Lambda functions, or SageMaker endpoints. Agent metadata stored in RDS.

### 5. Security & Access Control

**What it does**: Multi-layer security: admin auth, user API key auth, RBAC, policy engine, audit logging, SSRF protection, and content guardrails.

| Layer | Purpose | Implementation |
|-------|---------|---------------|
| Admin Auth | Control-plane protection | `X-Admin-API-Key` header, fail-closed |
| User Auth | Data-plane protection | LiteLLM API key auth (`Authorization: Bearer`) |
| RBAC | Fine-grained permissions | Per-role permission sets |
| Policy Engine | OPA-style pre-request rules | YAML-configured allow/deny rules |
| Audit Logging | Compliance trail | PostgreSQL-backed, fail-closed option |
| SSRF Protection | Block internal network access | URL validation, DNS rebind prevention |
| Secret Scrubbing | Prevent key leakage | Redacts API keys, AWS creds in error logs |
| Guardrails | Content safety | Regex, Bedrock Guardrails, LlamaGuard plugins |

**AWS integration**: API keys in Secrets Manager. Audit logs in RDS. Bedrock Guardrails for content moderation. WAF on ALB for rate limiting.

### 6. Observability (OpenTelemetry)

**What it does**: Full distributed tracing, metrics, and structured logging following OpenTelemetry GenAI semantic conventions.

**Metrics collected**:
- Request latency (P50/P90/P99)
- Time-to-first-token (TTFT) for streaming
- Tokens per second (TPS)
- Token usage (input/output) by model
- Cache hit/miss rates
- Circuit breaker state changes
- Routing decision telemetry (strategy, score, model selected, latency)

**AWS integration**: ADOT collector sidecar exports to X-Ray (traces) and CloudWatch (metrics/logs). Or use Amazon Managed Grafana + Prometheus for dashboards.

### 7. Resilience & Reliability

**What it does**: Production hardening for high-throughput AI workloads.

| Feature | Purpose |
|---------|---------|
| Backpressure | Reject requests at capacity (503) to prevent cascade failure |
| Graceful drain | Stop accepting new requests, finish in-flight before shutdown |
| Circuit breakers | Per-provider automatic failover when a provider is down |
| Request quotas | Per-team/per-key limits on requests, tokens, and spend |
| Cost reconciliation | Post-call adjustment for accurate spend tracking |
| Leader election | Single-leader operations in multi-replica deployments |
| Conversation affinity | Route follow-up requests to the same provider |

**AWS integration**: Circuit breakers protect Bedrock/external API calls. Quotas enforced via Redis (ElastiCache). Leader election via PostgreSQL (RDS).

### 8. Caching

**What it does**: Two-tier semantic caching to reduce costs and latency for repeated queries.

| Tier | Backend | Latency | Capacity |
|------|---------|---------|----------|
| L1 | In-process LRU | <1ms | ~10K entries per instance |
| L2 | Redis | ~1ms | Configurable, shared across instances |
| Semantic | Vector similarity | ~5ms | Fuzzy matching for similar queries |

**Cache control headers**:
- `x-routeiq-cache: HIT|MISS` — whether response was cached
- `x-routeiq-cache-tier: l1|l2|semantic` — which tier served it
- `x-routeiq-cache-age` — seconds since cached
- `x-routeiq-cache-control: no-cache` — bypass cache per request

**AWS integration**: L2 cache on ElastiCache Redis/Valkey. Admin endpoints for cache management (`/admin/cache/stats`, `/admin/cache/flush`).

### 9. Configuration Management

**What it does**: Dynamic configuration without restarts.

| Feature | How It Works |
|---------|-------------|
| S3 config sync | Background loop checks S3 ETag, reloads on change |
| Hot reload | Filesystem watcher triggers reload on config file changes |
| Model hot-swap | ML routing models can be updated at runtime |
| A/B experiments | Change traffic split between strategies via API |
| Strategy staging | Stage new strategies, validate, then promote |

**AWS integration**: Config YAML in S3. ML model artifacts in S3. ETag-based change detection for efficient polling. Config sync runs only on HA leader (via RDS leader election).

### 10. Plugin System

**What it does**: Extensible plugin architecture for custom logic at every stage of the request lifecycle.

**Built-in plugins**:

| Plugin | Purpose |
|--------|---------|
| Cache Plugin | Semantic + exact-match response caching |
| Cost Tracker | Per-model and per-team cost tracking with reconciliation |
| Evaluator | Model quality evaluation framework |
| Skills Discovery | Agent capability extraction and mapping |
| Bedrock Guardrails | AWS Bedrock content moderation |
| LlamaGuard | Self-hosted content safety classification |
| PII Guard | Personally identifiable information detection |
| Prompt Injection Guard | Prompt injection attack detection |
| Content Filter | Generic content filtering rules |

**Plugin hooks**: `on_request`, `on_response`, `on_llm_success`, `on_llm_failure`, `startup`, `shutdown`.

---

## AWS Cloud-Native Architecture

### Reference Architecture

```
                        ┌─────────────────────────────────────────────────────────┐
                        │                     AWS Cloud                            │
Internet ──────>  ┌─────┼─────────────────────────────────────────────────────┐    │
                  │     │  Route 53 (DNS) ──> CloudFront (CDN, optional)      │    │
                  │     │         │                                            │    │
                  │     │         ▼                                            │    │
                  │     │  ┌──────────────┐                                    │    │
                  │     │  │     ALB      │  ACM Certificate (TLS)             │    │
                  │     │  │  + AWS WAF   │  Rate limiting, IP filtering       │    │
                  │     │  └──────┬───────┘                                    │    │
                  │     │         │                                            │    │
                  │  ┌──┼─────────┼────────────────────────────────────┐       │    │
                  │  │  │  Private Subnets                             │       │    │
                  │  │  │         ▼                                    │       │    │
                  │  │  │  ┌──────────────┐   ┌──────────────┐        │       │    │
                  │  │  │  │ ECS Fargate  │   │ ECS Fargate  │        │       │    │
                  │  │  │  │  Gateway-1   │   │  Gateway-2   │  ...   │       │    │
                  │  │  │  │  (Leader)    │   │  (Replica)   │        │       │    │
                  │  │  │  └──────┬───────┘   └──────┬───────┘        │       │    │
                  │  │  │         │                   │                │       │    │
                  │  │  │         ▼                   ▼                │       │    │
                  │  │  │  ┌──────────────────────────────────┐       │       │    │
                  │  │  │  │     ElastiCache (Redis/Valkey)    │       │       │    │
                  │  │  │  │  - Response cache (L2)            │       │       │    │
                  │  │  │  │  - Rate limiting                  │       │       │    │
                  │  │  │  │  - Distributed state              │       │       │    │
                  │  │  │  │  - Conversation affinity          │       │       │    │
                  │  │  │  └──────────────────────────────────┘       │       │    │
                  │  │  │         │                                    │       │    │
                  │  │  │         ▼                                    │       │    │
                  │  │  │  ┌──────────────────────────────────┐       │       │    │
                  │  │  │  │    RDS Aurora PostgreSQL           │       │       │    │
                  │  │  │  │  - API key management             │       │       │    │
                  │  │  │  │  - Spend tracking                 │       │       │    │
                  │  │  │  │  - Team management                │       │       │    │
                  │  │  │  │  - Audit logs                     │       │       │    │
                  │  │  │  │  - Leader election                │       │       │    │
                  │  │  │  │  - A2A agent registry             │       │       │    │
                  │  │  │  └──────────────────────────────────┘       │       │    │
                  │  │  │                                              │       │    │
                  │  └──┼──────────────────────────────────────────────┘       │    │
                  │     │                                                      │    │
                  │     │  ┌───────────────────────────────────────────────┐    │    │
                  │     │  │              AWS Managed Services              │    │    │
                  │     │  │                                               │    │    │
                  │     │  │  ┌─────────────┐  ┌──────────────────────┐    │    │    │
                  │     │  │  │   Amazon    │  │      Amazon S3       │    │    │    │
                  │     │  │  │   Bedrock   │  │  - config.yaml       │    │    │    │
                  │     │  │  │  (Claude,   │  │  - ML model artifacts│    │    │    │
                  │     │  │  │   Nova,     │  │  - Policy files      │    │    │    │
                  │     │  │  │   Titan)    │  │                      │    │    │    │
                  │     │  │  └─────────────┘  └──────────────────────┘    │    │    │
                  │     │  │                                               │    │    │
                  │     │  │  ┌─────────────┐  ┌──────────────────────┐    │    │    │
                  │     │  │  │ CloudWatch  │  │   Secrets Manager    │    │    │    │
                  │     │  │  │  + X-Ray    │  │  - LITELLM_MASTER_KEY│    │    │    │
                  │     │  │  │  (via ADOT) │  │  - DATABASE_URL      │    │    │    │
                  │     │  │  │             │  │  - REDIS_PASSWORD     │    │    │    │
                  │     │  │  └─────────────┘  └──────────────────────┘    │    │    │
                  │     │  │                                               │    │    │
                  │     │  │  ┌─────────────┐  ┌──────────────────────┐    │    │    │
                  │     │  │  │  Bedrock    │  │   SageMaker /        │    │    │    │
                  │     │  │  │  Guardrails │  │   MLflow on ECS      │    │    │    │
                  │     │  │  │  (Content   │  │  (ML experiment      │    │    │    │
                  │     │  │  │   safety)   │  │   tracking)          │    │    │    │
                  │     │  │  └─────────────┘  └──────────────────────┘    │    │    │
                  │     │  └───────────────────────────────────────────────┘    │    │
                  │     │                                                      │    │
                  └─────┼──────────────────────────────────────────────────────┘    │
                        └─────────────────────────────────────────────────────────┘
```

### AWS Service Mapping

| RouteIQ Component | AWS Service | Purpose | Required? |
|-------------------|-------------|---------|-----------|
| Gateway compute | **ECS Fargate** | Run RouteIQ containers | Yes |
| Load balancer | **ALB** + **WAF** | HTTPS termination, rate limiting | Yes (production) |
| DNS | **Route 53** | Custom domain | Optional |
| TLS | **ACM** | Free SSL certificates | Yes (production) |
| Database | **Aurora PostgreSQL Serverless v2** | API keys, audit, teams, leader election | Recommended |
| Cache | **ElastiCache Redis/Valkey** | Response cache, rate limiting, quotas | Recommended |
| Config storage | **S3** | Config YAML, ML model artifacts | Recommended |
| LLM providers | **Amazon Bedrock** | Claude, Nova, Titan, Llama models | Yes |
| Secrets | **Secrets Manager** | API keys, database URLs | Yes (production) |
| Observability | **CloudWatch** + **X-Ray** via ADOT | Logs, metrics, traces | Recommended |
| Content safety | **Bedrock Guardrails** | Content moderation | Optional |
| ML tracking | **SageMaker** or MLflow on ECS | Routing model experiments | Optional |
| Container registry | **ECR** | Store gateway container images | Yes |
| CI/CD | **CodePipeline** or GitHub Actions | Build and deploy | Optional |

---

## Deployment Guide

### Prerequisites

- AWS account with Bedrock model access enabled
- AWS CLI v2 configured
- Container build tool (Docker or Finch)

### Step 1: Build and Push Container Image

```bash
# Build the image
finch build -f docker/Dockerfile -t routeiq-gateway:0.0.3 .

# Create ECR repository
aws ecr create-repository --repository-name routeiq-gateway

# Tag and push
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGION=us-west-2

aws ecr get-login-password --region $REGION | \
  finch login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$REGION.amazonaws.com

finch tag routeiq-gateway:0.0.3 $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/routeiq-gateway:0.0.3
finch push $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/routeiq-gateway:0.0.3
```

### Step 2: Create Supporting Infrastructure

#### Secrets Manager

```bash
# Master API key for admin access
aws secretsmanager create-secret \
  --name routeiq/master-key \
  --secret-string "sk-your-production-key-here"

# Database URL (after creating RDS)
aws secretsmanager create-secret \
  --name routeiq/database-url \
  --secret-string "postgresql://routeiq:<password>@<aurora-endpoint>:5432/routeiq"
```

#### Aurora PostgreSQL Serverless v2

```bash
# Create cluster (scales to zero when idle)
aws rds create-db-cluster \
  --db-cluster-identifier routeiq-db \
  --engine aurora-postgresql \
  --engine-version 15.4 \
  --master-username routeiq \
  --master-user-password <secure-password> \
  --serverless-v2-scaling-configuration MinCapacity=0.5,MaxCapacity=4 \
  --storage-encrypted \
  --vpc-security-group-ids sg-xxx \
  --db-subnet-group-name private-subnets

# Create instance in the cluster
aws rds create-db-instance \
  --db-instance-identifier routeiq-db-1 \
  --db-cluster-identifier routeiq-db \
  --db-instance-class db.serverless \
  --engine aurora-postgresql
```

#### ElastiCache Redis (or Valkey)

```bash
# Redis for caching, rate limiting, and distributed state
aws elasticache create-replication-group \
  --replication-group-id routeiq-cache \
  --replication-group-description "RouteIQ response cache" \
  --engine redis \
  --cache-node-type cache.t4g.medium \
  --num-node-groups 1 \
  --replicas-per-node-group 1 \
  --automatic-failover-enabled \
  --at-rest-encryption-enabled \
  --transit-encryption-enabled \
  --security-group-ids sg-xxx \
  --cache-subnet-group-name private-subnets

# Or use Valkey (AWS's open-source Redis fork)
aws elasticache create-replication-group \
  --replication-group-id routeiq-cache \
  --engine valkey \
  --cache-node-type cache.t4g.medium \
  # ... same options as above
```

#### S3 Config Bucket

```bash
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)

aws s3 mb s3://routeiq-config-$ACCOUNT

# Upload gateway config
aws s3 cp config/config.yaml s3://routeiq-config-$ACCOUNT/config/config.yaml

# Upload ML model artifacts (if using ML-based routing)
aws s3 sync models/ s3://routeiq-config-$ACCOUNT/models/
```

### Step 3: Create IAM Roles

#### Task Execution Role (ECS pulls images, reads secrets)

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:GetAuthorizationToken",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": "secretsmanager:GetSecretValue",
      "Resource": "arn:aws:secretsmanager:*:*:secret:routeiq/*"
    }
  ]
}
```

#### Task Role (what the gateway can access at runtime)

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "BedrockModels",
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": "arn:aws:bedrock:*:*:foundation-model/*"
    },
    {
      "Sid": "BedrockGuardrails",
      "Effect": "Allow",
      "Action": "bedrock:ApplyGuardrail",
      "Resource": "arn:aws:bedrock:*:*:guardrail/*"
    },
    {
      "Sid": "S3Config",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::routeiq-config-*",
        "arn:aws:s3:::routeiq-config-*/*"
      ]
    },
    {
      "Sid": "CloudWatchLogs",
      "Effect": "Allow",
      "Action": ["logs:CreateLogStream", "logs:PutLogEvents"],
      "Resource": "arn:aws:logs:*:*:log-group:/ecs/routeiq-*"
    },
    {
      "Sid": "XRayTracing",
      "Effect": "Allow",
      "Action": ["xray:PutTraceSegments", "xray:PutTelemetryRecords"],
      "Resource": "*"
    }
  ]
}
```

### Step 4: ECS Task Definition

```json
{
  "family": "routeiq-gateway",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::<account>:role/routeiq-execution-role",
  "taskRoleArn": "arn:aws:iam::<account>:role/routeiq-task-role",
  "containerDefinitions": [
    {
      "name": "routeiq-gateway",
      "image": "<account>.dkr.ecr.<region>.amazonaws.com/routeiq-gateway:0.0.3",
      "essential": true,
      "portMappings": [{"containerPort": 4000, "protocol": "tcp"}],
      "environment": [
        {"name": "AWS_DEFAULT_REGION", "value": "us-west-2"},
        {"name": "CONFIG_S3_BUCKET", "value": "routeiq-config-<account>"},
        {"name": "CONFIG_S3_KEY", "value": "config/config.yaml"},
        {"name": "CONFIG_HOT_RELOAD", "value": "true"},
        {"name": "CONFIG_SYNC_INTERVAL", "value": "60"},
        {"name": "REDIS_HOST", "value": "<elasticache-endpoint>"},
        {"name": "REDIS_PORT", "value": "6379"},
        {"name": "OTEL_EXPORTER_OTLP_ENDPOINT", "value": "http://localhost:4317"},
        {"name": "OTEL_SERVICE_NAME", "value": "routeiq-gateway"},
        {"name": "OTEL_TRACES_EXPORTER", "value": "otlp"},
        {"name": "OTEL_METRICS_EXPORTER", "value": "otlp"},
        {"name": "MCP_GATEWAY_ENABLED", "value": "true"},
        {"name": "A2A_GATEWAY_ENABLED", "value": "true"},
        {"name": "ROUTEIQ_MAX_CONCURRENT_REQUESTS", "value": "100"},
        {"name": "ROUTEIQ_PROVIDER_CB_ENABLED", "value": "true"}
      ],
      "secrets": [
        {
          "name": "LITELLM_MASTER_KEY",
          "valueFrom": "arn:aws:secretsmanager:<region>:<account>:secret:routeiq/master-key"
        },
        {
          "name": "DATABASE_URL",
          "valueFrom": "arn:aws:secretsmanager:<region>:<account>:secret:routeiq/database-url"
        }
      ],
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:4000/_health/live || exit 1"],
        "interval": 30,
        "timeout": 10,
        "retries": 3,
        "startPeriod": 60
      },
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/routeiq-gateway",
          "awslogs-region": "<region>",
          "awslogs-stream-prefix": "gateway"
        }
      }
    },
    {
      "name": "adot-collector",
      "image": "public.ecr.aws/aws-observability/aws-otel-collector:latest",
      "essential": false,
      "command": ["--config=/etc/ecs/ecs-xray.yaml"],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/routeiq-otel",
          "awslogs-region": "<region>",
          "awslogs-stream-prefix": "otel"
        }
      }
    }
  ]
}
```

### Step 5: Create ECS Service

```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name routeiq

# Enable Container Insights
aws ecs update-cluster-settings \
  --cluster routeiq \
  --settings name=containerInsights,value=enhanced

# Create service
aws ecs create-service \
  --cluster routeiq \
  --service-name routeiq-gateway \
  --task-definition routeiq-gateway \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-private-1,subnet-private-2],securityGroups=[sg-gateway],assignPublicIp=DISABLED}" \
  --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:...,containerName=routeiq-gateway,containerPort=4000" \
  --health-check-grace-period-seconds 120
```

### Step 6: ALB + WAF

```bash
# Create ALB
aws elbv2 create-load-balancer \
  --name routeiq-alb \
  --subnets subnet-public-1 subnet-public-2 \
  --security-groups sg-alb \
  --scheme internet-facing

# Create target group
aws elbv2 create-target-group \
  --name routeiq-tg \
  --protocol HTTP \
  --port 4000 \
  --vpc-id vpc-xxx \
  --target-type ip \
  --health-check-path /_health/live \
  --health-check-interval-seconds 30 \
  --healthy-threshold-count 2

# Create HTTPS listener (requires ACM certificate)
aws elbv2 create-listener \
  --load-balancer-arn <alb-arn> \
  --protocol HTTPS \
  --port 443 \
  --certificates CertificateArn=<acm-cert-arn> \
  --default-actions Type=forward,TargetGroupArn=<target-group-arn>
```

### Step 7: Auto Scaling

```bash
# Register scalable target
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/routeiq/routeiq-gateway \
  --min-capacity 2 \
  --max-capacity 10

# Scale on CPU
aws application-autoscaling put-scaling-policy \
  --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/routeiq/routeiq-gateway \
  --policy-name routeiq-cpu-scaling \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration \
    "TargetValue=70,PredefinedMetricSpecification={PredefinedMetricType=ECSServiceAverageCPUUtilization}"

# Scale on request count
aws application-autoscaling put-scaling-policy \
  --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/routeiq/routeiq-gateway \
  --policy-name routeiq-request-scaling \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration \
    "TargetValue=1000,PredefinedMetricSpecification={PredefinedMetricType=ALBRequestCountPerTarget},ScaleOutCooldown=60,ScaleInCooldown=300"
```

---

## Gateway Configuration (config.yaml)

This is the config file uploaded to S3. It defines which models are available and how requests are routed.

```yaml
# config/config.yaml — RouteIQ Gateway Configuration
model_list:
  # Amazon Bedrock models (IAM auth, no API keys)
  - model_name: claude-sonnet
    litellm_params:
      model: bedrock/anthropic.claude-sonnet-4-20250514-v1:0
      aws_region_name: us-west-2
  - model_name: claude-haiku
    litellm_params:
      model: bedrock/anthropic.claude-3-5-haiku-20241022-v1:0
      aws_region_name: us-west-2
  - model_name: nova-pro
    litellm_params:
      model: bedrock/amazon.nova-pro-v1:0
      aws_region_name: us-west-2
  - model_name: nova-micro
    litellm_params:
      model: bedrock/amazon.nova-micro-v1:0
      aws_region_name: us-west-2

  # OpenAI models (API key auth)
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY

router_settings:
  # ML-based routing: send simple queries to cheap models
  routing_strategy: llmrouter-knn
  # Or use simple load balancing:
  # routing_strategy: simple-shuffle

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  database_url: os.environ/DATABASE_URL

litellm_settings:
  cache: true
  cache_params:
    type: redis
    host: os.environ/REDIS_HOST
    port: os.environ/REDIS_PORT
  callbacks:
    - otel
  success_callback:
    - otel
  failure_callback:
    - otel
```

---

## Operational Playbook

### Health Checks

```bash
# Liveness (is the process running?)
curl https://gateway.example.com/_health/live

# Readiness (is it ready to serve traffic?)
curl https://gateway.example.com/_health/ready

# Routing info
curl -H "Authorization: Bearer sk-your-key" \
  https://gateway.example.com/router/info
```

### Cache Management

```bash
# View cache stats
curl -H "X-Admin-API-Key: sk-admin-key" \
  https://gateway.example.com/admin/cache/stats

# Flush cache
curl -X POST -H "X-Admin-API-Key: sk-admin-key" \
  https://gateway.example.com/admin/cache/flush

# List cached entries
curl -H "X-Admin-API-Key: sk-admin-key" \
  https://gateway.example.com/admin/cache/entries?limit=50
```

### Config Reload

```bash
# Trigger config reload (pulls latest from S3)
curl -X POST -H "X-Admin-API-Key: sk-admin-key" \
  -H "Content-Type: application/json" \
  -d '{"force_sync": true}' \
  https://gateway.example.com/config/reload

# Check sync status
curl -H "Authorization: Bearer sk-your-key" \
  https://gateway.example.com/config/sync/status
```

### MCP Server Registration

```bash
# Register an MCP server
curl -X POST -H "X-Admin-API-Key: sk-admin-key" \
  -H "Content-Type: application/json" \
  -d '{
    "server_id": "my-tools",
    "name": "Internal Tools",
    "url": "https://tools.internal.example.com/mcp",
    "transport": "streamable-http"
  }' \
  https://gateway.example.com/llmrouter/mcp/servers

# List available tools
curl -H "Authorization: Bearer sk-your-key" \
  https://gateway.example.com/llmrouter/mcp/tools
```

### A2A Agent Registration

```bash
# Discover gateway agent card
curl https://gateway.example.com/.well-known/agent.json

# List registered agents
curl -H "Authorization: Bearer sk-your-key" \
  https://gateway.example.com/a2a/agents
```

### CloudWatch Alarms

```bash
# High P99 latency alarm
aws cloudwatch put-metric-alarm \
  --alarm-name routeiq-high-latency \
  --metric-name TargetResponseTime \
  --namespace AWS/ApplicationELB \
  --statistic p99 \
  --period 60 \
  --threshold 5.0 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 3 \
  --alarm-actions arn:aws:sns:<region>:<account>:ops-alerts

# Error rate alarm
aws cloudwatch put-metric-alarm \
  --alarm-name routeiq-error-rate \
  --metric-name HTTPCode_Target_5XX_Count \
  --namespace AWS/ApplicationELB \
  --statistic Sum \
  --period 60 \
  --threshold 10 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2 \
  --alarm-actions arn:aws:sns:<region>:<account>:ops-alerts
```

---

## Cost Estimation

Monthly cost for a mid-size deployment (2 Fargate tasks, Aurora Serverless, ElastiCache):

| Component | Sizing | Estimated Monthly Cost |
|-----------|--------|----------------------|
| ECS Fargate (2 tasks) | 2 vCPU, 4 GB each | ~$120 |
| ALB | ~1M requests/month | ~$25 |
| Aurora Serverless v2 | 0.5-4 ACU | ~$50-200 |
| ElastiCache Redis | cache.t4g.medium | ~$50 |
| S3 | <1 GB config/models | ~$1 |
| CloudWatch + X-Ray | Logs + traces | ~$20-50 |
| Secrets Manager | 3-5 secrets | ~$3 |
| ECR | Container images | ~$1 |
| **Total (infrastructure)** | | **~$270-450/mo** |
| Amazon Bedrock | Per-token pricing | Variable (usage-based) |

> Bedrock model costs are separate and usage-based. Infrastructure costs scale with replica count and database utilization.

---

## Deployment Tiers

### Minimal (Development/POC)

Single container, no database, no cache. Environment variables only.

```bash
# Just the gateway + Bedrock
aws ecs create-service --desired-count 1
```

**Enabled**: LLM routing, OpenAI API, basic health checks
**Disabled**: Caching, spend tracking, audit, quotas, HA

### Standard (Production)

2 replicas, Aurora, ElastiCache, ADOT, WAF.

```bash
aws ecs create-service --desired-count 2
```

**Adds**: Response caching, spend tracking, API key management, audit logs, auto-scaling, observability

### Enterprise (High Availability)

3+ replicas, multi-AZ Aurora, ElastiCache replication, leader election, full policy engine.

```bash
aws ecs create-service --desired-count 3
```

**Adds**: Leader election, config sync from S3, policy engine, Bedrock Guardrails, MCP + A2A protocols, circuit breakers, quotas

---

## Next Steps

- [Detailed AWS deployment steps](deployment/aws.md)
- [High availability setup](high-availability.md)
- [Observability configuration](observability.md)
- [Routing strategy guide](routing-strategies.md)
- [MCP gateway setup](mcp-gateway.md)
- [A2A gateway setup](a2a-gateway.md)
- [Plugin development](plugins.md)
- [Security hardening](security.md)
