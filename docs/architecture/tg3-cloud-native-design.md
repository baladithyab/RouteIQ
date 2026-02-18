# TG3: Cloud-Native & Self-Hostable Deployment Architecture

> **Version**: 1.0
> **Date**: 2026-02-18
> **Status**: Proposal — awaiting review
> **Scope**: Cloud-native deployment architecture for RouteIQ v1.0

---

## Executive Summary

RouteIQ v1.0 fundamentally restructures the gateway from a 36,000-line monolith with
monkey-patching into a ~5,000-line `pip install routeiq-router` package that integrates
with LiteLLM via [`CustomRoutingStrategyBase`](../../reference/litellm/litellm/types/router.py:671).
This architectural shift — eliminating the `workers=1` constraint, removing 16 module-level
singletons, and shedding ~7,578 lines of redundant MCP/A2A code — has profound implications
for how RouteIQ is deployed, scaled, and operated.

This document designs the complete cloud-native deployment architecture for v1.0, covering:

1. **12-Factor compliance** audit and remediation plan
2. **State externalization** from in-process singletons to external stores
3. **Horizontal scaling** design (multi-worker, multi-replica, auto-scaling)
4. **Four self-hosting tiers** from `docker compose up` to multi-region HA
5. **Serverless and edge** deployment feasibility assessment
6. **Air-gapped deployment** updated for v1.0 architecture
7. **Configuration management** consolidation (124 env vars → ~30 core vars)
8. **Secrets management** across all deployment tiers

**Key design decisions:**

- RouteIQ v1.0 is a **stateless routing library** that runs inside LiteLLM's process
- All mutable state moves to **Redis** (routing config, circuit breakers, session affinity)
  or **PostgreSQL** (persistent config, audit logs, team management via LiteLLM)
- ML models are loaded **per-worker** from filesystem/S3/GCS with an in-memory LRU cache
- A/B testing accuracy uses **Redis atomic counters** for cross-replica consistency
- The existing Helm chart at [`deploy/charts/routeiq-gateway/`](../../deploy/charts/routeiq-gateway/)
  is updated, not replaced

**Referenced existing infrastructure:**

| Artifact | Location |
|----------|----------|
| Helm chart | [`deploy/charts/routeiq-gateway/`](../../deploy/charts/routeiq-gateway/) |
| Basic Compose | [`docker-compose.yml`](../../docker-compose.yml) |
| HA Compose | [`docker-compose.ha.yml`](../../docker-compose.ha.yml) |
| OTel Compose | [`docker-compose.otel.yml`](../../docker-compose.otel.yml) |
| HA+OTel Compose | [`docker-compose.ha-otel.yml`](../../docker-compose.ha-otel.yml) |
| OTel Collector | [`config/otel-collector-config.yaml`](../../config/otel-collector-config.yaml) |
| Nginx | [`config/nginx.conf`](../../config/nginx.conf) |
| Air-gapped guide | [`docs/deployment/air-gapped.md`](../deployment/air-gapped.md) |
| AWS guide | [`docs/deployment/aws.md`](../deployment/aws.md) |
| Env vars | [`.env.example`](../../.env.example) |

---

## 12-Factor Compliance

### Current State Assessment

| # | Factor | Current State | Gap | Target State | Effort |
|---|--------|--------------|-----|-------------|--------|
| 1 | **Codebase** | Single repo, git-tracked, one deployable | None | No change | 0 pw |
| 2 | **Dependencies** | [`pyproject.toml`](../../pyproject.toml) + `uv.lock`; vendored submodule at `reference/LLMRouter/` | Minor: submodule ref leaks into image | `routeiq-router` as pip package; LiteLLM as declared dependency | 1 pw |
| 3 | **Config** | 124 env vars in [`.env.example`](../../.env.example); only 54 documented; YAML config at [`config/config.yaml`](../../config/config.yaml) | Major: 70 undocumented vars; mixed env + file config; no schema validation | ~30 core vars via Pydantic Settings; full schema validation; env-only for secrets, YAML for model config | 2 pw |
| 4 | **Backing services** | PostgreSQL, Redis, S3, OTel Collector treated as attached resources via env vars | Minor: some singletons hold connection state that does not survive reconnection | All connections via env-injected URLs; connection pools with reconnect logic | 1 pw |
| 5 | **Build, release, run** | Multi-stage [`Dockerfile`](../../docker/Dockerfile); tagged images; compose for orchestration | Minor: no immutable release artifact beyond Docker image | `routeiq-router` on PyPI + Docker image with digest pinning | 1 pw |
| 6 | **Processes** | **BLOCKER**: `workers=1` hardcoded; 16 module-level singletons hold in-memory state | Critical: all routing state, circuit breakers, config cached in-process | Stateless workers; all mutable state in Redis/PostgreSQL | 3 pw |
| 7 | **Port binding** | FastAPI on port 4000 via uvicorn; nginx on 80/443 | None | No change; LiteLLM handles port binding | 0 pw |
| 8 | **Concurrency** | **BLOCKER**: Single worker; scale-out only via replicas (each with `workers=1`) | Critical: cannot use CPU-level parallelism within a pod | Multi-worker uvicorn (2-4 workers/pod); horizontal replicas via HPA | 1 pw |
| 9 | **Disposability** | Drain manager exists in [`resilience.py`](../../src/litellm_llmrouter/resilience.py); graceful shutdown | Minor: ML model loading is slow (~5-10s for KNN with sentence-transformers) | Pre-warm model in startup probe; startup probe timeout = 60s | 0.5 pw |
| 10 | **Dev/prod parity** | [`docker-compose.local-test.yml`](../../docker-compose.local-test.yml) mirrors prod | Minor: local uses SQLite or no DB; prod uses PostgreSQL | Tier 1 uses SQLite; all other tiers use PostgreSQL | 0.5 pw |
| 11 | **Logs** | Structured JSON logging via Python `logging`; OTel log exporter available | Minor: some modules use `print()` or unstructured logs | All logs as structured JSON events to stdout; OTel log bridge | 0.5 pw |
| 12 | **Admin processes** | No formal admin CLI; DB migrations via env var flag | Gap: migrations run on all replicas in HA; no one-off task runner | Init container for migrations; `routeiq admin` CLI for one-off tasks | 1 pw |

**Total remediation effort: ~11.5 person-weeks**

### Target State Design

#### Factor 3: Config Consolidation

The current 124 env vars will be consolidated into categories:

**Core vars (~30, always documented):**

```
# Identity & Auth
LITELLM_MASTER_KEY          # Required: admin access key
LITELLM_CONFIG_PATH          # Default: /app/config/config.yaml

# Backing Services
DATABASE_URL                 # PostgreSQL connection string
REDIS_URL                    # Redis connection (replaces REDIS_HOST/PORT/PASSWORD)

# Routing
ROUTEIQ_STRATEGY             # Default routing strategy (e.g., knn, mlp, round-robin)
ROUTEIQ_MODELS_PATH          # Path to ML model artifacts
ROUTEIQ_MODEL_S3_URI         # S3/GCS URI for model sync (s3://bucket/key)

# Observability
OTEL_EXPORTER_OTLP_ENDPOINT  # OTel collector endpoint
OTEL_SERVICE_NAME            # Service name for traces/metrics

# Feature Flags
ROUTEIQ_POLICY_ENABLED       # Enable policy engine (default: false)
ROUTEIQ_AB_TESTING_ENABLED   # Enable A/B testing (default: false)
```

**Provider vars (pass-through to LiteLLM):**
```
OPENAI_API_KEY, ANTHROPIC_API_KEY, AZURE_API_KEY, etc.
```

**Advanced vars (documented but rarely changed):**
```
ROUTEIQ_MAX_CONCURRENT_REQUESTS, ROUTEIQ_DRAIN_TIMEOUT_SECONDS,
ROUTEIQ_CB_FAILURE_THRESHOLD, OTEL_TRACES_SAMPLER, etc.
```

All vars validated at startup via Pydantic Settings with clear error messages for
missing required values.

#### Factor 6: Stateless Processes

The 16 singletons identified in the codebase and their disposition:

| Singleton | Module | v1.0 Disposition |
|-----------|--------|-----------------|
| `_mcp_gateway` | [`mcp_gateway.py`](../../src/litellm_llmrouter/mcp_gateway.py:1267) | **Deleted** — LiteLLM native |
| `_mcp_repository` | [`database.py`](../../src/litellm_llmrouter/database.py:961) | **Deleted** — LiteLLM native |
| `_a2a_repository` | [`database.py`](../../src/litellm_llmrouter/database.py:402) | **Deleted** — LiteLLM native |
| `_policy_engine` | [`policy_engine.py`](../../src/litellm_llmrouter/policy_engine.py:880) | **Keep** — per-worker; config from Redis/env |
| `_quota_enforcer` | [`quota.py`](../../src/litellm_llmrouter/quota.py:971) | **Deleted** — LiteLLM native quota |
| `_hot_reload_manager` | [`hot_reload.py`](../../src/litellm_llmrouter/hot_reload.py:499) | **Keep** — per-worker file watcher |
| `_sync_manager` | [`config_sync.py`](../../src/litellm_llmrouter/config_sync.py:447) | **Simplify** — LiteLLM's DB config store |
| `_leader_election` | [`leader_election.py`](../../src/litellm_llmrouter/leader_election.py:617) | **Deleted** — LiteLLM handles HA |
| `_gateway_metrics` | [`metrics.py`](../../src/litellm_llmrouter/metrics.py:366) | **Keep** — per-worker OTel meters |
| `_audit_repository` | [`audit.py`](../../src/litellm_llmrouter/audit.py:436) | **Simplify** — emit to OTel logs |
| `_plugin_manager` | [`plugin_manager.py`](../../src/litellm_llmrouter/gateway/plugin_manager.py:1220) | **Keep** — per-worker plugin registry |
| `_callback_bridge` | [`plugin_callback_bridge.py`](../../src/litellm_llmrouter/gateway/plugin_callback_bridge.py:191) | **Keep** — per-worker bridge |
| `_plugin_middleware` | [`plugin_middleware.py`](../../src/litellm_llmrouter/gateway/plugin_middleware.py:302) | **Keep** — per-worker middleware |
| `_tracker` | [`conversation_affinity.py`](../../src/litellm_llmrouter/conversation_affinity.py:326) | **Externalize** — Redis-backed |
| `_http_client` | [`http_client_pool.py`](../../src/litellm_llmrouter/http_client_pool.py:163) | **Deleted** — LiteLLM's httpx pool |
| `_cache_manager` | [`semantic_cache.py`](../../src/litellm_llmrouter/semantic_cache.py:705) | **Keep** — per-worker with Redis L2 |

**Result: 16 singletons → 8 kept (per-worker, stateless), 1 externalized (Redis), 7 deleted**

---

## State Externalization

### State Inventory

```
+------------------------------------------------------------------+
|                     CURRENT STATE MAP (v0.2)                      |
+------------------------------------------------------------------+
|                                                                    |
|  IN-PROCESS (per-worker, lost on restart):                        |
|  +-------------------------------------------------------------+ |
|  | Routing Strategy Config    | strategy_registry.py            | |
|  | A/B Test Weights           | strategy_registry.py            | |
|  | Traffic Counters           | strategy_registry.py            | |
|  | Circuit Breaker State      | resilience.py                   | |
|  | Conversation Affinity Map  | conversation_affinity.py        | |
|  | Policy Engine Rules        | policy_engine.py                | |
|  | Plugin Registry            | plugin_manager.py               | |
|  | ML Model Artifacts         | strategies.py (loaded in RAM)   | |
|  | HTTP Client Pool           | http_client_pool.py             | |
|  | Semantic Cache L1          | semantic_cache.py               | |
|  | MCP Server Registry        | mcp_gateway.py (DELETED)        | |
|  | A2A Agent Registry         | a2a_gateway.py (DELETED)        | |
|  +-------------------------------------------------------------+ |
|                                                                    |
|  EXTERNAL (survives restarts):                                    |
|  +-------------------------------------------------------------+ |
|  | PostgreSQL: API keys, teams, budgets, request logs (LiteLLM) | |
|  | Redis: response cache, rate limit counters (LiteLLM)         | |
|  | S3/GCS: config.yaml, ML model files                          | |
|  +-------------------------------------------------------------+ |
+------------------------------------------------------------------+
```

### Migration Plan

| State | Current Location | Target Location | Migration Strategy | Consistency |
|-------|-----------------|----------------|-------------------|-------------|
| **Active routing strategy** | `strategy_registry._active_strategy` (in-memory) | Redis key `routeiq:strategy:active` | Write-through: admin API writes Redis, workers read on each request | Eventual (< 1s) |
| **A/B test weights** | `strategy_registry._weights` (dict) | Redis hash `routeiq:ab:weights` | Workers read weights from Redis; admin API writes | Eventual (< 1s) |
| **A/B traffic counters** | `strategy_registry._counters` (dict) | Redis hash `routeiq:ab:counters` with HINCRBY | Atomic Redis counters; workers increment per-request | Strong (atomic) |
| **Circuit breaker state** | `resilience._breakers` (dict of CircuitBreaker) | Redis hash per breaker `routeiq:cb:{name}` | Workers read/write state via Redis; state = CLOSED/OPEN/HALF_OPEN + failure count + opened_at | Eventual (< 1s) |
| **Conversation affinity** | `_tracker._affinity_map` (LRU dict) | Redis hash `routeiq:affinity:{conv_id}` with TTL | Already supports Redis via `redis_url` param; make Redis the default in Tier 2+ | Eventual (< 1s) |
| **Policy engine rules** | `_policy_engine._config` (loaded from YAML file) | YAML file on filesystem (ConfigMap) or Redis key `routeiq:policy:rules` | Workers load from file at startup + watch for changes | Eventual (reload interval) |
| **ML model artifacts** | Loaded into memory from `models/` directory | S3/GCS/filesystem → downloaded to local cache → loaded into memory per-worker | Each worker downloads on startup; model hash verification via [`model_artifacts.py`](../../src/litellm_llmrouter/model_artifacts.py) | Eventual (model sync interval) |
| **Plugin registry** | `_plugin_manager._plugins` (list) | Per-worker; loaded from config at startup | Plugins are code (not runtime state); config-driven | N/A (per-worker) |
| **Semantic cache L1** | In-memory LRU dict | Per-worker L1 (memory) + shared L2 (Redis) | Already implemented; L2 Redis is the cross-replica cache | L1: per-worker; L2: shared |
| **HTTP client pool** | `_http_client` (httpx.AsyncClient) | Per-worker pool (managed by LiteLLM) | Deleted; LiteLLM manages its own HTTP clients | N/A |
| **Gateway metrics** | OTel meter instruments (per-worker) | Per-worker → OTel Collector → backend | Each worker emits independently; OTel Collector aggregates | N/A (additive) |

### Consistency Model

```
+-----------------------------------------------------------------+
|                    CONSISTENCY TIERS                              |
+-----------------------------------------------------------------+
|                                                                   |
|  STRONG (Redis atomic operations):                               |
|  - A/B test traffic counters (HINCRBY)                           |
|  - Rate limit counters (LiteLLM via Redis)                       |
|                                                                   |
|  EVENTUAL (< 1 second):                                          |
|  - Active routing strategy (Redis pub/sub notification)          |
|  - A/B test weights (read per-request from Redis)                |
|  - Circuit breaker state (Redis TTL-based)                       |
|  - Conversation affinity (Redis with TTL)                        |
|                                                                   |
|  EVENTUAL (reload interval, 5-300 seconds):                      |
|  - Policy engine rules (file watch or Redis)                     |
|  - ML model artifacts (S3 sync interval)                         |
|  - Plugin configuration (restart required)                       |
|                                                                   |
|  PER-WORKER (no cross-replica sharing needed):                   |
|  - L1 semantic cache (memory-local, Redis L2 shared)             |
|  - OTel metric instruments (additive, collector aggregates)      |
|  - HTTP client pools (per-worker connection management)          |
+-----------------------------------------------------------------+
```

---

## Horizontal Scaling

### Multi-Worker Design

With monkey-patching eliminated, RouteIQ v1.0 runs inside LiteLLM's uvicorn process with
multiple workers:

```
+-------------------------------------------------------+
|                     Pod / Container                     |
|                                                         |
|  uvicorn --workers N --host 0.0.0.0 --port 4000       |
|                                                         |
|  +----------+  +----------+  +----------+  +----+      |
|  | Worker 0 |  | Worker 1 |  | Worker 2 |  | .. |      |
|  |          |  |          |  |          |  |    |      |
|  | LiteLLM  |  | LiteLLM  |  | LiteLLM  |  |    |      |
|  | + RouteIQ|  | + RouteIQ|  | + RouteIQ|  |    |      |
|  | Strategy |  | Strategy |  | Strategy |  |    |      |
|  |          |  |          |  |          |  |    |      |
|  | ML Model |  | ML Model |  | ML Model |  |    |      |
|  | (in RAM) |  | (in RAM) |  | (in RAM) |  |    |      |
|  +----------+  +----------+  +----------+  +----+      |
|                                                         |
|  Shared: OS file cache, network stack                   |
+-------------------------------------------------------+
```

**Worker count guidance:**

| Routing Strategy | CPU Profile | Recommended Workers | RAM per Worker |
|-----------------|------------|-------------------|---------------|
| Simple (round-robin, random, least-busy) | I/O-bound | `2 * CPU_COUNT + 1` | ~256 MB |
| ELO, MF (matrix factorization) | Light CPU | `CPU_COUNT + 1` | ~512 MB |
| MLP, SVM | Moderate CPU | `CPU_COUNT` | ~512 MB |
| KNN (sentence-transformers) | Heavy CPU + RAM | `max(1, CPU_COUNT / 2)` | ~1.5 GB |

**KNN memory analysis:**

The KNN strategy loads a sentence-transformer model (e.g., `all-MiniLM-L6-v2`, ~80 MB)
plus pre-computed embeddings for the model catalog. With 50 models in the catalog:

```
Per-worker memory = model (~80 MB) + embeddings (~5 MB) + overhead (~15 MB) ≈ 100 MB
4 workers × 100 MB = 400 MB for routing alone
+ LiteLLM base (~200 MB) + Python runtime (~100 MB)
= ~700 MB per pod with 4 KNN workers
```

**Recommendation: 2 workers per CPU core for I/O strategies, 1 worker per 2 cores for KNN.**

### Multi-Replica Design

```
                    +---------------------+
                    |    Ingress / LB      |
                    |  (nginx / ALB / NLB) |
                    +----------+----------+
                               |
              +----------------+----------------+
              |                |                |
     +--------v------+  +-----v---------+  +---v-----------+
     |   Pod 1       |  |   Pod 2       |  |   Pod 3       |
     |               |  |               |  |               |
     | LiteLLM       |  | LiteLLM       |  | LiteLLM       |
     | + RouteIQ     |  | + RouteIQ     |  | + RouteIQ     |
     | (2-4 workers) |  | (2-4 workers) |  | (2-4 workers) |
     +-------+-------+  +-------+-------+  +-------+-------+
             |                   |                   |
     +-------v-------------------v-------------------v-------+
     |                    Backing Services                     |
     |                                                         |
     |  +-------------+  +----------+  +---------+             |
     |  | PostgreSQL  |  |  Redis   |  | S3/GCS  |             |
     |  | (keys,teams |  | (cache,  |  | (models |             |
     |  |  budgets,   |  |  routing |  |  config)|             |
     |  |  config)    |  |  state)  |  |         |             |
     |  +-------------+  +----------+  +---------+             |
     +---------------------------------------------------------+
```

**Key properties:**
- All pods are **identical** — no leader/follower distinction for request handling
- **Session affinity** not required at LB level (all state in Redis/PostgreSQL)
- LiteLLM's built-in DB config store (`STORE_MODEL_IN_DB=true`) ensures config consistency
- ML models are loaded per-pod from shared filesystem or S3/GCS

### Auto-Scaling Strategy

**HPA configuration (updated from existing [`hpa.yaml`](../../deploy/charts/routeiq-gateway/templates/hpa.yaml)):**

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: routeiq-gateway
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: routeiq-gateway
  minReplicas: 2
  maxReplicas: 20
  metrics:
    # Primary: CPU utilization (ML routing is CPU-intensive)
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    # Secondary: Request rate via custom metrics
    - type: Pods
      pods:
        metric:
          name: gateway_requests_per_second
        target:
          type: AverageValue
          averageValue: "50"
    # Tertiary: P95 latency via custom metrics
    - type: Pods
      pods:
        metric:
          name: gateway_p95_latency_ms
        target:
          type: AverageValue
          averageValue: "500"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300  # 5 min cooldown
      policies:
        - type: Percent
          value: 25
          periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 30
      policies:
        - type: Percent
          value: 100
          periodSeconds: 60
        - type: Pods
          value: 4
          periodSeconds: 60
      selectPolicy: Max
```

**Custom metrics pipeline:**
```
Pod (Prometheus /metrics) --> Prometheus Adapter --> HPA
     or
Pod (OTel) --> OTel Collector --> Prometheus --> Prometheus Adapter --> HPA
     or
Pod (OTel) --> OTel Collector --> KEDA ScaledObject (for queue-based scaling)
```

### ML Model Memory Planning

| Scenario | Strategy | Workers/Pod | RAM/Pod | Pods | Total Cluster RAM |
|----------|----------|------------|---------|------|------------------|
| Small (eval) | round-robin | 2 | 512 Mi | 1 | 512 Mi |
| Small (ML) | KNN | 1 | 1 Gi | 1 | 1 Gi |
| Medium | KNN + A/B | 2 | 2 Gi | 3 | 6 Gi |
| Large | KNN + A/B | 2 | 2 Gi | 10 | 20 Gi |
| Enterprise | KNN + A/B | 4 | 4 Gi | 20 | 80 Gi |

**Model sharing optimization (future):**

For KNN strategies with large embedding models, consider:
1. **Memory-mapped files**: Load model via `mmap()` so OS shares physical pages across workers
2. **Model server sidecar**: Dedicated process serving embeddings via Unix socket
3. **Pre-computed embeddings**: Cache prompt→embedding in Redis to avoid per-request computation

---

## Self-Hosting Tiers

### Tier 1: Minimal — `docker compose up`

**Target audience:** Local development, small teams (< 10 users), evaluation.

```
+-------------------------------------------+
|              Docker Host                   |
|                                            |
|  +--------------------------------------+ |
|  |        routeiq-gateway               | |
|  |                                      | |
|  |  LiteLLM + routeiq-router            | |
|  |  + embedded Admin UI (static assets) | |
|  |  + SQLite (or no DB)                 | |
|  |  + in-memory cache (no Redis)        | |
|  |                                      | |
|  |  Port 4000 ──────────────── :4000    | |
|  +--------------------------------------+ |
+-------------------------------------------+
```

**docker-compose.tier1.yml:**
```yaml
services:
  routeiq:
    image: ghcr.io/routeiq/routeiq-gateway:latest
    ports:
      - "4000:4000"
    volumes:
      - ./config:/app/config:ro
      - ./models:/app/models:ro
    environment:
      - LITELLM_MASTER_KEY=${LITELLM_MASTER_KEY:?Required}
      - LITELLM_CONFIG_PATH=/app/config/config.yaml
      - OPENAI_API_KEY=${OPENAI_API_KEY:-}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:4000/_health/live"]
      interval: 30s
      timeout: 5s
      retries: 3
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: "1.0"
    restart: unless-stopped
```

| Property | Value |
|----------|-------|
| Containers | 1 |
| Resources | 1 CPU, 2 GB RAM |
| Workers | 2 (I/O strategies) or 1 (KNN) |
| Database | SQLite or none |
| Cache | In-memory only |
| HA | None |
| Throughput | ~20-50 req/s |
| Startup time | ~5-15s |
| Operational complexity | ★☆☆☆☆ |

**Limitations:**
- No HA — single point of failure
- No session sharing across restarts
- Volatile routing state (A/B counters reset on restart)
- No persistent audit logs
- Admin UI limited (no team management without DB)

---

### Tier 2: Standard — Docker Compose + PostgreSQL + Redis

**Target audience:** Small-medium production (10-100 users), teams needing persistence.

```
+------------------------------------------------------------+
|                     Docker Host                             |
|                                                              |
|  +----------+  +----------+  +-----------+  +------------+  |
|  | routeiq  |  | routeiq  |  | PostgreSQL|  |   Redis    |  |
|  | gateway-1|  | gateway-2|  |  16-alpine|  |  7-alpine  |  |
|  | :4000    |  | :4000    |  |  :5432    |  |  :6379     |  |
|  +----+-----+  +----+-----+  +-----------+  +------------+  |
|       |              |                                       |
|  +----v--------------v-----+                                 |
|  |      Nginx LB           |                                 |
|  |      :8080 --> :80      |                                 |
|  +--------------------------+                                |
|                                                              |
|  Optional:                                                   |
|  +------------------+  +-----------+                         |
|  | OTel Collector   |  |  Jaeger   |                         |
|  | :4317            |  |  :16686   |                         |
|  +------------------+  +-----------+                         |
+------------------------------------------------------------+
```

**docker-compose.tier2.yml:**
```yaml
services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-litellm}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:?Required}
      POSTGRES_DB: ${POSTGRES_DB:-litellm}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-litellm}"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped
    networks:
      - routeiq

  redis:
    image: redis:7-alpine
    command: >
      redis-server
      --appendonly yes
      --maxmemory 256mb
      --maxmemory-policy allkeys-lru
      --requirepass ${REDIS_PASSWORD:-changeme}
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "-a", "${REDIS_PASSWORD:-changeme}", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped
    networks:
      - routeiq

  routeiq-gateway-1:
    image: ghcr.io/routeiq/routeiq-gateway:latest
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./config:/app/config:ro
      - ./models:/app/models:ro
    environment:
      - LITELLM_MASTER_KEY=${LITELLM_MASTER_KEY:?Required}
      - LITELLM_CONFIG_PATH=/app/config/config.yaml
      - DATABASE_URL=postgresql://${POSTGRES_USER:-litellm}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB:-litellm}
      - REDIS_URL=redis://:${REDIS_PASSWORD:-changeme}@redis:6379/0
      - STORE_MODEL_IN_DB=true
      - OPENAI_API_KEY=${OPENAI_API_KEY:-}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:4000/_health/live"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 15s
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: "2.0"
    restart: unless-stopped
    networks:
      - routeiq

  routeiq-gateway-2:
    image: ghcr.io/routeiq/routeiq-gateway:latest
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./config:/app/config:ro
      - ./models:/app/models:ro
    environment:
      - LITELLM_MASTER_KEY=${LITELLM_MASTER_KEY:?Required}
      - LITELLM_CONFIG_PATH=/app/config/config.yaml
      - DATABASE_URL=postgresql://${POSTGRES_USER:-litellm}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB:-litellm}
      - REDIS_URL=redis://:${REDIS_PASSWORD:-changeme}@redis:6379/0
      - STORE_MODEL_IN_DB=true
      - OPENAI_API_KEY=${OPENAI_API_KEY:-}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:4000/_health/live"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 15s
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: "2.0"
    restart: unless-stopped
    networks:
      - routeiq

  nginx:
    image: nginx:alpine
    depends_on:
      - routeiq-gateway-1
      - routeiq-gateway-2
    ports:
      - "8080:80"
    volumes:
      - ./config/nginx.conf:/etc/nginx/nginx.conf:ro
    healthcheck:
      test: ["CMD", "wget", "-q", "--spider", "http://localhost/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
    networks:
      - routeiq

volumes:
  postgres_data:
  redis_data:

networks:
  routeiq:
    driver: bridge
```

| Property | Value |
|----------|-------|
| Containers | 4-6 (gateway ×2, postgres, redis, nginx, optional otel) |
| Resources | 4 CPU, 8 GB RAM total |
| Workers/pod | 2-4 |
| Database | PostgreSQL 16 |
| Cache | Redis 7 (256 MB) |
| HA | Active-active behind nginx |
| Throughput | ~100-200 req/s |
| Startup time | ~15-30s (wait for DB/Redis) |
| Operational complexity | ★★☆☆☆ |

---

### Tier 3: Production — Kubernetes + Helm

**Target audience:** Medium-large production, teams with K8s expertise.

```
+--------------------------------------------------------------------+
|                        Kubernetes Cluster                           |
|                                                                      |
|  +---------------------------+  +-------------------------------+   |
|  |     Namespace: routeiq     |  |   Namespace: observability    |   |
|  |                            |  |                               |   |
|  | +--------+ +--------+     |  |  +------------------+         |   |
|  | | Pod 1  | | Pod 2  | ... |  |  | OTel Collector   |         |   |
|  | |gateway | |gateway |     |  |  | DaemonSet        |         |   |
|  | |2 wkrs  | |2 wkrs  |     |  |  +------------------+         |   |
|  | +---+----+ +---+----+     |  |                               |   |
|  |     |          |          |  |  +----------+ +-----------+   |   |
|  | +---v----------v---+      |  |  | Jaeger   | | Prometheus|   |   |
|  | |    Service        |      |  |  | or Tempo | | or Thanos |   |   |
|  | |    ClusterIP      |      |  |  +----------+ +-----------+   |   |
|  | +--------+----------+      |  +-------------------------------+   |
|  |          |                 |                                       |
|  | +--------v----------+      |  +-------------------------------+   |
|  | |    Ingress         |      |  |   External Backing Services   |   |
|  | |    nginx/ALB       |      |  |                               |   |
|  | +-------------------+      |  |  PostgreSQL (RDS/CloudSQL)    |   |
|  |                            |  |  Redis (ElastiCache/Memstore) |   |
|  | +-------------------+      |  |  S3/GCS (model storage)       |   |
|  | | NetworkPolicy     |      |  +-------------------------------+   |
|  | | PDB (minAvail: 1) |      |                                       |
|  | | HPA (2-20 pods)   |      |                                       |
|  | +-------------------+      |                                       |
|  +---------------------------+                                       |
+--------------------------------------------------------------------+
```

**Helm values (production overlay) — `values-production.yaml`:**
```yaml
replicaCount: 3

image:
  repository: ghcr.io/routeiq/routeiq-gateway
  tag: "1.0.0"
  pullPolicy: IfNotPresent

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300

podDisruptionBudget:
  enabled: true
  minAvailable: 2

resources:
  requests:
    memory: "1Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "4000m"

probes:
  startup:
    enabled: true
    path: /_health/live
    initialDelaySeconds: 0
    periodSeconds: 5
    failureThreshold: 24  # 2 min max startup (for KNN model loading)

gateway:
  features:
    mcpGatewayEnabled: false  # LiteLLM native
    a2aGatewayEnabled: false  # LiteLLM native
  otel:
    enabled: true
    serviceName: routeiq-gateway
    tracesExporter: otlp
    metricsExporter: otlp
    endpoint: http://otel-collector.observability:4317

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "600"
    nginx.ingress.kubernetes.io/proxy-buffering: "off"
  hosts:
    - host: routeiq.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: routeiq-tls
      hosts:
        - routeiq.example.com

networkPolicy:
  enabled: true
  ingress:
    fromNamespaceSelector:
      kubernetes.io/metadata.name: ingress-nginx
  egress:
    allowDns: true
    allowHttpsExternal: true
    to:
      - podSelector:
          matchLabels:
            app: postgres
        ports:
          - port: 5432
      - podSelector:
          matchLabels:
            app: redis
        ports:
          - port: 6379

externalSecrets:
  enabled: true
  refreshInterval: "1h"
  secretStoreRef:
    name: aws-secrets-manager
    kind: ClusterSecretStore
  data:
    - secretKey: LITELLM_MASTER_KEY
      remoteRef:
        key: routeiq/production/master-key
    - secretKey: DATABASE_URL
      remoteRef:
        key: routeiq/production/database-url
    - secretKey: OPENAI_API_KEY
      remoteRef:
        key: routeiq/production/openai-key

podAntiAffinity:
  enabled: true
  type: soft
  topologyKey: topology.kubernetes.io/zone

topologySpreadConstraints:
  - maxSkew: 1
    topologyKey: topology.kubernetes.io/zone
    whenUnsatisfiable: ScheduleAnyway
    labelSelector:
      matchLabels:
        app.kubernetes.io/name: routeiq-gateway
```

**Deploy:**
```bash
helm install routeiq-gateway ./deploy/charts/routeiq-gateway \
  -f values-production.yaml \
  --namespace routeiq \
  --create-namespace
```

| Property | Value |
|----------|-------|
| Pods | 3-20 (HPA managed) |
| Resources per pod | 1-4 CPU, 1-4 GB RAM |
| Workers/pod | 2-4 |
| Database | External PostgreSQL (RDS/CloudSQL) |
| Cache | External Redis (ElastiCache/Memorystore) |
| HA | Multi-AZ spread, PDB, HPA |
| Throughput | ~500-2,000 req/s |
| Startup time | ~30-60s (model loading + DB connection) |
| Operational complexity | ★★★☆☆ |

---

### Tier 4: Enterprise — Multi-Region HA

**Target audience:** Enterprise, regulated industries, global deployments.

```
+----------------------------------------------------------------------+
|                         Global Infrastructure                         |
|                                                                        |
|  +---------------------------+    +---------------------------+       |
|  |    Region: us-east-1       |    |    Region: eu-west-1      |       |
|  |                            |    |                            |       |
|  |  +--------------------+   |    |  +--------------------+    |       |
|  |  | K8s Cluster        |   |    |  | K8s Cluster        |    |       |
|  |  |                    |   |    |  |                    |    |       |
|  |  | RouteIQ Pods (3-10)|   |    |  | RouteIQ Pods (3-10)|    |       |
|  |  | + Istio Sidecar    |   |    |  | + Istio Sidecar    |    |       |
|  |  +--------------------+   |    |  +--------------------+    |       |
|  |                            |    |                            |       |
|  |  +----------+ +--------+  |    |  +----------+ +--------+  |       |
|  |  | RDS      | | Elasti-|  |    |  | RDS      | | Elasti-|  |       |
|  |  | Primary  | | Cache  |  |    |  | Read     | | Cache  |  |       |
|  |  |          | | Primary|  |    |  | Replica  | | Replica|  |       |
|  |  +-----+----+ +---+----+  |    |  +-----+----+ +---+----+  |       |
|  |        |           |       |    |        |           |       |       |
|  +--------|-----------|-------+    +--------|-----------|-------+       |
|            |           |                     |           |               |
|  +---------v-----------v---------------------v-----------v-----------+  |
|  |              Cross-Region Replication                              |  |
|  |                                                                    |  |
|  |  PostgreSQL: Aurora Global Database (< 1s replication lag)        |  |
|  |  Redis: Global Datastore (ElastiCache) or app-level sync         |  |
|  |  S3: Cross-Region Replication (CRR) for model artifacts           |  |
|  +--------------------------------------------------------------------+  |
|                                                                        |
|  +--------------------------------------------------------------------+  |
|  |              Global Load Balancing                                 |  |
|  |                                                                    |  |
|  |  Route 53 (latency-based) --> CloudFront --> ALB per region       |  |
|  |  or: Global Accelerator --> NLB per region                        |  |
|  +--------------------------------------------------------------------+  |
+----------------------------------------------------------------------+
```

**Key architectural decisions:**

1. **Database**: Aurora Global Database with writer in primary region, read replicas in
   secondary regions. Failover RPO < 1 second, RTO ~1 minute.

2. **Redis**: ElastiCache Global Datastore for cross-region replication of routing state,
   A/B counters, and conversation affinity. Replication lag < 1 second.

3. **Model artifacts**: S3 Cross-Region Replication ensures ML models are available in
   all regions. Each pod downloads from the nearest S3 bucket.

4. **Service mesh**: Istio for mTLS between all services, traffic management, and
   observability.

5. **DNS failover**: Route 53 health checks detect regional failures and automatically
   route traffic to healthy regions.

**Helm values (enterprise overlay) — extending Tier 3:**
```yaml
# values-enterprise.yaml (layered on top of values-production.yaml)

replicaCount: 5

autoscaling:
  minReplicas: 5
  maxReplicas: 50

resources:
  requests:
    memory: "2Gi"
    cpu: "2000m"
  limits:
    memory: "8Gi"
    cpu: "8000m"

podDisruptionBudget:
  minAvailable: 3

# Istio sidecar injection
podLabels:
  sidecar.istio.io/inject: "true"

podAnnotations:
  traffic.sidecar.istio.io/excludeOutboundPorts: "5432,6379"

# Multi-AZ + multi-region spread
topologySpreadConstraints:
  - maxSkew: 1
    topologyKey: topology.kubernetes.io/zone
    whenUnsatisfiable: DoNotSchedule
    labelSelector:
      matchLabels:
        app.kubernetes.io/name: routeiq-gateway
```

| Property | Value |
|----------|-------|
| Pods per region | 5-50 (HPA managed) |
| Regions | 2-4 |
| Resources per pod | 2-8 CPU, 2-8 GB RAM |
| Database | Aurora Global Database |
| Cache | ElastiCache Global Datastore |
| HA | Multi-region active-active |
| Throughput | ~5,000-20,000 req/s (global) |
| DR | RPO < 5 min, RTO < 15 min |
| Operational complexity | ★★★★★ |

---

## Serverless & Edge Deployment

### Serverless Gateway (Lambda / Cloud Run / Azure Container Apps)

**Feasibility assessment:**

| Factor | Lambda | Cloud Run | Azure Container Apps |
|--------|--------|-----------|---------------------|
| **Cold start** | 15-30s (ML model loading) | 5-15s (container boot) | 5-15s (container boot) |
| **Max execution time** | 15 min | 60 min | Unlimited |
| **Memory** | Up to 10 GB | Up to 32 GB | Up to 4 GB |
| **Persistent connections** | No (stateless) | Yes (min instances) | Yes (min instances) |
| **SSE/Streaming** | Response streaming OK | Native support | Native support |
| **Cost at scale** | High (per-invocation) | Moderate (per-request) | Moderate (per-vCPU-s) |

**Cold start is the blocker.** KNN routing requires loading a sentence-transformer model
(~80 MB) which takes ~10-15 seconds on Lambda. Mitigation options:

1. **Provisioned concurrency** (Lambda): Pre-warm N instances — eliminates cold start but
   increases cost to near-EC2 levels, defeating the purpose.
2. **Min instances** (Cloud Run): Keep 1+ instances warm — same cost tradeoff.
3. **Simple strategies only**: Use round-robin/random routing in serverless (no ML model
   loading) — viable but loses RouteIQ's differentiator.

**Hybrid serverless pattern (recommended for serverless use cases):**

```
                Client
                  |
                  v
        +------------------+
        | CloudFront / CDN |
        +--------+---------+
                 |
        +--------v---------+
        | API Gateway      |
        | (routing rules)  |
        +--------+---------+
                 |
      +----------+----------+
      |                     |
+-----v------+    +---------v--------+
| Lambda     |    | ECS/Cloud Run    |
| (simple    |    | (ML routing      |
|  routing)  |    |  + full gateway) |
+------------+    +------------------+
```

- Simple routing requests (round-robin, random) → Lambda (instant, cheap)
- ML routing requests → ECS/Cloud Run (warm instances, full model)

**Verdict: Serverless is viable for simple routing strategies only. ML routing requires
persistent compute.**

### Edge Routing (Workers / Lambda@Edge)

**Concept:** Pre-classify requests at the edge to reduce latency to the central gateway.

```
                User (Tokyo)
                     |
                     v
           +-------------------+
           | Edge Function     |
           | (Cloudflare/L@E)  |
           |                   |
           | Classify: "This   |
           | is a simple query |
           | -> route to cheap |
           | model"            |
           +--------+----------+
                    |
                    | X-RouteIQ-Hint: gpt-4o-mini
                    v
           +-------------------+
           | Central Gateway   |
           | (us-east-1)       |
           | (respects hint or |
           |  overrides)       |
           +-------------------+
```

**Feasible edge tasks:**
- Request classification (simple regex/keyword matching → model hint)
- Provider geo-routing (Tokyo user → ap-northeast-1 Bedrock endpoint)
- Rate limiting / authentication (moved to edge)
- Static response caching for identical prompts

**Not feasible at edge:**
- ML-based routing (model too large for Workers/Lambda@Edge)
- A/B testing with accurate traffic splits (requires centralized counters)
- Session affinity (requires Redis state)

**Verdict: Edge deployment is useful for pre-classification and geo-routing only. Core ML
routing must remain centralized.**

### Feasibility Summary

| Deployment Mode | Simple Routing | ML Routing | Full Gateway | Recommendation |
|----------------|:-:|:-:|:-:|---|
| **Lambda** | ✅ Viable | ❌ Cold start | ❌ Cold start | Simple routing only |
| **Cloud Run** | ✅ Viable | ⚠️ Min instances | ✅ With min instances | Good for variable load |
| **Azure Container Apps** | ✅ Viable | ⚠️ Min instances | ✅ With min instances | Good for variable load |
| **Cloudflare Workers** | ✅ Pre-classify | ❌ No ML models | ❌ No Python | Hints/pre-routing only |
| **Lambda@Edge** | ✅ Pre-classify | ❌ No ML models | ❌ Size limit | Hints/pre-routing only |
| **ECS Fargate** | ✅ | ✅ | ✅ | Full gateway, best serverless option |

---

## Air-Gapped Deployment

> This section updates the existing [`docs/deployment/air-gapped.md`](../deployment/air-gapped.md)
> for v1.0 architecture. The existing document remains the authoritative reference for
> v0.2 deployments.

### v1.0 Changes for Air-Gapped Environments

1. **Package installation**: `pip install routeiq-router` replaces the monolithic build.
   In air-gapped mode, vendor the wheel:
   ```bash
   # On internet-connected build host
   pip download routeiq-router litellm -d vendor/python/
   
   # In air-gapped Dockerfile
   RUN pip install --no-index --find-links=/tmp/vendor/python routeiq-router litellm
   ```

2. **ML model artifacts**: Models must be vendored into the Docker image or pre-loaded
   onto a shared filesystem:
   ```bash
   # Vendor during build
   COPY models/ /app/models/
   
   # Or mount at runtime
   volumes:
     - /nfs/routeiq-models:/app/models:ro
   ```

3. **Sentence-transformers offline**: For KNN strategies, pre-download the embedding model:
   ```bash
   # On internet-connected host
   python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2').save('/tmp/st-model')"
   
   # Copy to air-gapped image
   COPY vendor/st-model /app/models/sentence-transformers/
   ```
   Set `SENTENCE_TRANSFORMERS_HOME=/app/models/sentence-transformers/` in the environment.

4. **Admin UI assets**: The React+TypeScript SPA is bundled as static files in the Docker
   image during build. No CDN or external asset loading at runtime:
   ```dockerfile
   # Build UI
   FROM node:20-alpine AS ui-builder
   COPY ui/ /app/ui/
   RUN cd /app/ui && npm ci --offline && npm run build
   
   # Copy built assets
   FROM python:3.14-slim
   COPY --from=ui-builder /app/ui/dist /app/static/
   ```

5. **Container registry mirror**: For Kubernetes deployments, use an internal registry:
   ```yaml
   # values-airgap.yaml
   image:
     repository: internal-registry.corp.local/routeiq/routeiq-gateway
     tag: "1.0.0"
     pullPolicy: Never  # Image must be pre-loaded
   ```

6. **No PyPI/npm access**: All dependencies pinned with hashes in `uv.lock` and verified
   via [`model_artifacts.py`](../../src/litellm_llmrouter/model_artifacts.py) signature
   verification for ML models.

The existing [`docs/deployment/air-gapped.md`](../deployment/air-gapped.md) sections on
network requirements (Section 1), TLS termination (Section 1.2), identity/auth (Section 2),
backup/restore (Section 3), and observability (Section 5) remain valid for v1.0.

---

## Configuration Management

### GitOps Strategy

```
+-----------------------------------------------------------+
|                    GitOps Pipeline                          |
|                                                             |
|  Git Repository                                            |
|  +-------------------------------------------------------+ |
|  | deploy/                                                | |
|  |   charts/routeiq-gateway/                             | |
|  |     values.yaml           (base defaults)             | |
|  |   envs/                                               | |
|  |     staging/                                          | |
|  |       values.yaml         (staging overrides)         | |
|  |       secrets.yaml        (sealed/encrypted)          | |
|  |     production/                                       | |
|  |       values.yaml         (prod overrides)            | |
|  |       secrets.yaml        (sealed/encrypted)          | |
|  |     enterprise/                                       | |
|  |       us-east-1.yaml      (region-specific)           | |
|  |       eu-west-1.yaml      (region-specific)           | |
|  +---+----+----------------------------------------------+ |
|      |    |                                                 |
|      |    +---> ArgoCD / Flux                               |
|      |              |                                       |
|      |              v                                       |
|      |         Kubernetes Cluster                           |
|      |         (auto-sync, drift detection)                 |
|      |                                                      |
|      +--------> CI/CD Pipeline                              |
|                 (validate, lint, diff)                       |
+-----------------------------------------------------------+
```

**Helm values hierarchy** (merged left-to-right, later overrides earlier):
```bash
helm install routeiq-gateway ./deploy/charts/routeiq-gateway \
  -f values.yaml \                          # Base defaults
  -f envs/production/values.yaml \          # Environment overrides
  --set image.tag=1.0.0                     # Release-specific
```

**ArgoCD Application:**
```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: routeiq-gateway
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/org/routeiq-deploy.git
    targetRevision: main
    path: deploy/charts/routeiq-gateway
    helm:
      valueFiles:
        - values.yaml
        - envs/production/values.yaml
  destination:
    server: https://kubernetes.default.svc
    namespace: routeiq
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
      - CreateNamespace=true
```

### Environment Variable Consolidation

**From 124 vars to ~30 core vars:**

| Category | v0.2 Count | v1.0 Count | Strategy |
|----------|-----------|-----------|----------|
| **Required** | 1 (LITELLM_MASTER_KEY) | 1 | No change |
| **Provider keys** | 8 | 8 | Pass-through to LiteLLM |
| **Database/Redis** | 7 | 2 (DATABASE_URL, REDIS_URL) | Consolidate into connection strings |
| **Config sync** | 8 | 1 (ROUTEIQ_MODEL_S3_URI) | Simplify; LiteLLM handles config DB |
| **Feature flags** | 6 | 3 | Remove MCP/A2A (LiteLLM native) |
| **SSRF** | 8 | 3 | Consolidate canonical names |
| **Resilience** | 12 | 4 | Group into ROUTEIQ_CB_* prefix |
| **OTel** | 12 | 4 (standard OTEL_* vars) | Use standard OTEL env vars only |
| **Routing** | 10 | 3 | ROUTEIQ_STRATEGY, ROUTEIQ_MODELS_PATH, ROUTEIQ_AB_ENABLED |
| **MLOps/Dev** | 8 | 0 | Move to MLOps compose only |
| **Leader election** | 5 | 0 | Deleted (LiteLLM handles HA) |
| **Cache** | 8 | 2 | ROUTEIQ_CACHE_ENABLED, ROUTEIQ_CACHE_REDIS_URL |
| **Legacy/undocumented** | 31 | 0 | Deleted |
| **Total** | 124 | ~30 | 76% reduction |

**Pydantic Settings model:**
```python
from pydantic_settings import BaseSettings

class RouteIQSettings(BaseSettings):
    # Required
    strategy: str = "round-robin"
    models_path: str = "/app/models"
    
    # Backing services (optional for Tier 1)
    redis_url: str | None = None
    model_s3_uri: str | None = None
    
    # Feature flags
    policy_enabled: bool = False
    ab_testing_enabled: bool = False
    cache_enabled: bool = False
    
    # Resilience
    max_concurrent_requests: int = 0  # 0 = disabled
    drain_timeout_seconds: int = 30
    cb_failure_threshold: int = 5
    cb_timeout_seconds: int = 30
    
    model_config = {"env_prefix": "ROUTEIQ_"}
```

### Feature Flags

| Flag | Default | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
|------|---------|--------|--------|--------|--------|
| `ROUTEIQ_POLICY_ENABLED` | false | false | false | true | true |
| `ROUTEIQ_AB_TESTING_ENABLED` | false | false | true | true | true |
| `ROUTEIQ_CACHE_ENABLED` | false | false | true | true | true |
| LiteLLM `MCP_GATEWAY_ENABLED` | false | false | false | optional | optional |
| LiteLLM `A2A_GATEWAY_ENABLED` | false | false | false | optional | optional |
| LiteLLM `STORE_MODEL_IN_DB` | false | false | true | true | true |

---

## Secrets Management

### By Deployment Tier

| Tier | Secret Storage | Rotation Method | Encryption |
|------|---------------|----------------|------------|
| **Tier 1** | `.env` file | Manual | None (dev only) |
| **Tier 2** | Docker Compose `.env` or Docker Secrets | Manual with restart | At-rest on host |
| **Tier 3** | ExternalSecrets → Vault/AWS SM/GCP SM | Automatic (ESO refresh) | Vault seal / KMS |
| **Tier 4** | ExternalSecrets + Istio mTLS | Automatic + cert rotation | KMS + mTLS |

### Secret Categories

```
+-------------------------------------------------------------+
|                    SECRET CLASSIFICATION                      |
+-------------------------------------------------------------+
|                                                               |
|  CRITICAL (rotate every 30 days):                            |
|  - LITELLM_MASTER_KEY (admin access)                         |
|  - DATABASE_URL (contains password)                          |
|                                                               |
|  HIGH (rotate every 90 days):                                |
|  - LLM provider API keys (OPENAI_API_KEY, etc.)             |
|  - REDIS_URL (contains password)                             |
|                                                               |
|  MEDIUM (rotate annually):                                   |
|  - TLS certificates (cert-manager auto-rotation)             |
|  - Docker registry credentials                               |
|                                                               |
|  LOW (static):                                               |
|  - OTEL service name, config paths                           |
|  - Feature flags                                              |
+-------------------------------------------------------------+
```

### Zero-Downtime Rotation Pattern

**For K8s with ExternalSecrets Operator:**

```
1. Rotate secret in Vault/AWS SM
         |
         v
2. ESO detects change (refreshInterval: 1h)
         |
         v
3. ESO updates K8s Secret
         |
         v
4. Pod restarts (checksum/secret annotation triggers rolling update)
         |
         v
5. New pods start with new secret; old pods drain
```

**For database password rotation specifically:**

```
1. Create new DB user with new password
2. Update secret in Vault to point to new user
3. ESO updates K8s Secret --> rolling restart
4. Verify all pods using new credentials
5. Drop old DB user
```

### mTLS Between Services

For Tier 4 (enterprise) deployments with Istio:

```yaml
# PeerAuthentication: require mTLS for all routeiq namespace traffic
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: routeiq-strict-mtls
  namespace: routeiq
spec:
  mtls:
    mode: STRICT

# AuthorizationPolicy: only allow traffic from known services
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: routeiq-gateway-authz
  namespace: routeiq
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: routeiq-gateway
  rules:
    - from:
        - source:
            namespaces: ["ingress-nginx", "routeiq"]
      to:
        - operation:
            ports: ["4000"]
```

---

## Observability Architecture

### Metrics Pipeline

```
+---------------------------------------------------------------+
|                    Metrics Flow                                |
|                                                                 |
|  RouteIQ Pod                                                   |
|  +-----------------------------------------------------------+ |
|  | Worker 0:  OTel Meter --> OTLP push --> Collector          | |
|  | Worker 1:  OTel Meter --> OTLP push --> Collector          | |
|  | Worker N:  OTel Meter --> OTLP push --> Collector          | |
|  +-----------------------------------------------------------+ |
|                                                                 |
|  OTel Collector (DaemonSet or Sidecar)                         |
|  +-----------------------------------------------------------+ |
|  | Receive (OTLP) --> Batch --> Export to:                     | |
|  |   - Prometheus Remote Write (self-hosted / Thanos / Mimir) | |
|  |   - AWS CloudWatch EMF (via ADOT)                          | |
|  |   - Datadog / Grafana Cloud (SaaS)                         | |
|  +-----------------------------------------------------------+ |
|                                                                 |
|  Visualization                                                  |
|  +-----------------------------------------------------------+ |
|  | Grafana dashboard with:                                    | |
|  |   - Request rate by model/provider                         | |
|  |   - P50/P95/P99 latency                                   | |
|  |   - Routing strategy distribution                          | |
|  |   - A/B test traffic splits                                | |
|  |   - Circuit breaker state                                  | |
|  |   - Token usage and cost                                   | |
|  +-----------------------------------------------------------+ |
+---------------------------------------------------------------+
```

**Key metrics (emitted by RouteIQ):**

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `routeiq.routing.decision.duration` | Histogram | strategy, model | Time to make routing decision |
| `routeiq.routing.strategy.usage` | Counter | strategy, outcome | Which strategy was selected |
| `routeiq.ab.traffic.split` | Counter | experiment, variant | A/B test traffic distribution |
| `routeiq.circuit_breaker.state` | Gauge | name, state | Circuit breaker state per provider |
| `routeiq.model.load.duration` | Histogram | strategy | Time to load ML model |
| `routeiq.policy.evaluation.duration` | Histogram | action | Policy engine evaluation time |

LiteLLM emits its own metrics for request rate, latency, token usage, and cost — these
do not need to be duplicated.

### Distributed Tracing

**Trace structure (v1.0):**

```
[gateway.request]                              (root span, LiteLLM)
  |
  +-- [routeiq.routing.decision]               (RouteIQ routing)
  |     |-- strategy: knn
  |     |-- model_selected: claude-3-opus
  |     |-- candidates_count: 5
  |     |-- ab_experiment: pricing-v2
  |     |-- ab_variant: candidate
  |     +-- duration_ms: 12
  |
  +-- [routeiq.policy.evaluate]                (policy engine, if enabled)
  |     |-- action: allow
  |     +-- rules_evaluated: 3
  |
  +-- [litellm.completion]                     (LiteLLM provider call)
        |-- model: claude-3-opus-20240229
        |-- provider: anthropic
        |-- tokens.input: 150
        |-- tokens.output: 500
        +-- duration_ms: 2340
```

### Log Aggregation

All logs emitted as structured JSON to stdout:

```json
{
  "timestamp": "2026-02-18T17:00:00.000Z",
  "level": "INFO",
  "service": "routeiq-gateway",
  "trace_id": "abc123...",
  "span_id": "def456...",
  "message": "routing decision completed",
  "strategy": "knn",
  "model": "claude-3-opus",
  "duration_ms": 12,
  "worker_id": 2
}
```

**Collection pipeline:**
- **Tier 1-2:** `docker logs` or Docker logging driver → host file
- **Tier 3:** stdout → K8s log agent (FluentBit DaemonSet) → Loki/Elasticsearch
- **Tier 4:** Same + cross-region log aggregation via Grafana Cloud or S3 archive

### Alerting

**Critical alerts (PagerDuty/OpsGenie):**

| Alert | Condition | Severity |
|-------|-----------|----------|
| High error rate | `error_rate > 5%` for 5 min | P1 |
| All circuit breakers open | All providers OPEN for 2 min | P1 |
| Database unreachable | `postgres_up == 0` for 1 min | P1 |
| Pod crash loop | Restart count > 3 in 5 min | P2 |
| High latency | `p95 > 5s` for 10 min | P2 |
| Redis unreachable | `redis_up == 0` for 5 min | P3 |
| A/B test skew | Traffic split deviates > 10% from config | P3 |

---

## Implementation Roadmap

### Phase 1: Foundation (3 person-weeks)

| Task | Effort | Dependencies |
|------|--------|-------------|
| Implement `RouteIQRoutingStrategy(CustomRoutingStrategyBase)` | 1 pw | None |
| Implement Pydantic Settings for ~30 core env vars | 0.5 pw | None |
| Implement Redis-backed state for A/B counters and circuit breakers | 1 pw | None |
| Validate multi-worker support with `workers=4` | 0.5 pw | Strategy impl |

### Phase 2: Code Shed & Docker (2 person-weeks)

| Task | Effort | Dependencies |
|------|--------|-------------|
| Delete MCP/A2A/redundant modules (~7,578 lines) | 0.5 pw | Phase 1 |
| Update Dockerfile for `routeiq-router` pip install | 0.5 pw | Phase 1 |
| Create Tier 1 and Tier 2 docker-compose files | 0.5 pw | Dockerfile |
| Air-gapped Dockerfile updates (vendor routeiq-router wheel) | 0.5 pw | Dockerfile |

### Phase 3: Kubernetes & Helm (2 person-weeks)

| Task | Effort | Dependencies |
|------|--------|-------------|
| Update Helm chart for v1.0 (values, deployment, HPA) | 1 pw | Phase 2 |
| Add startup probe for ML model loading | 0.25 pw | Helm update |
| Add custom metrics HPA (request rate, p95 latency) | 0.5 pw | Helm update |
| ExternalSecrets integration (Vault, AWS SM, GCP SM) | 0.25 pw | Helm update |

### Phase 4: Observability & Admin UI (2 person-weeks)

| Task | Effort | Dependencies |
|------|--------|-------------|
| RouteIQ-specific OTel metrics and traces | 0.5 pw | Phase 1 |
| Grafana dashboard templates | 0.5 pw | Metrics |
| Admin UI static asset embedding in Docker image | 0.5 pw | UI build |
| Alerting rules (Prometheus + Alertmanager) | 0.5 pw | Metrics |

### Phase 5: Enterprise & Documentation (2.5 person-weeks)

| Task | Effort | Dependencies |
|------|--------|-------------|
| Multi-region Helm values and ArgoCD config | 0.5 pw | Phase 3 |
| Istio mTLS configuration | 0.5 pw | Phase 3 |
| Updated air-gapped deployment guide | 0.5 pw | Phase 2 |
| Configuration migration guide (v0.2 → v1.0) | 0.5 pw | Phase 2 |
| Comprehensive deployment documentation | 0.5 pw | All phases |

**Total: ~11.5 person-weeks**

```
Week:  1    2    3    4    5    6    7    8    9   10   11   12
       |----Phase 1----|----Phase 2----|---Phase 3---|
                                       |---Phase 4---|
                                                     |--Phase 5--|
```

---

## Appendix A: Environment Variable Migration Reference

| v0.2 Variable | v1.0 Variable | Action |
|--------------|--------------|--------|
| `LITELLM_MASTER_KEY` | `LITELLM_MASTER_KEY` | No change |
| `LITELLM_CONFIG_PATH` | `LITELLM_CONFIG_PATH` | No change |
| `DATABASE_URL` | `DATABASE_URL` | No change |
| `REDIS_HOST` + `REDIS_PORT` + `REDIS_PASSWORD` | `REDIS_URL` | Consolidate |
| `MCP_GATEWAY_ENABLED` | Removed | LiteLLM native |
| `A2A_GATEWAY_ENABLED` | Removed | LiteLLM native |
| `MCP_HA_SYNC_ENABLED` | Removed | LiteLLM native |
| `LLMROUTER_ENABLE_MCP_TOOL_INVOCATION` | Removed | LiteLLM native |
| `LLMROUTER_HA_MODE` | Removed | LiteLLM handles HA |
| `LLMROUTER_CONFIG_SYNC_*` (5 vars) | Removed | LiteLLM DB config |
| `LLMROUTER_MODELS_PATH` | `ROUTEIQ_MODELS_PATH` | Rename |
| `LLMROUTER_HOT_RELOAD` | `ROUTEIQ_HOT_RELOAD` | Rename |
| `LLMROUTER_RELOAD_INTERVAL` | `ROUTEIQ_RELOAD_INTERVAL` | Rename |
| `LLMROUTER_ALLOW_PRIVATE_IPS` | `ROUTEIQ_SSRF_ALLOW_PRIVATE` | Consolidate |
| `LLMROUTER_SSRF_ALLOWLIST_HOSTS` | `ROUTEIQ_SSRF_HOST_ALLOWLIST` | Consolidate |
| `LLMROUTER_SSRF_ALLOWLIST_CIDRS` | `ROUTEIQ_SSRF_CIDR_ALLOWLIST` | Consolidate |
| `ROUTEIQ_MAX_CONCURRENT_REQUESTS` | `ROUTEIQ_MAX_CONCURRENT_REQUESTS` | No change |
| `ROUTEIQ_DRAIN_TIMEOUT_SECONDS` | `ROUTEIQ_DRAIN_TIMEOUT_SECONDS` | No change |
| `LLMROUTER_ROUTER_CALLBACK_ENABLED` | Removed | Always enabled in v1.0 |
| `STORE_MODEL_IN_DB` | `STORE_MODEL_IN_DB` | No change (LiteLLM var) |
| `OTEL_*` (standard) | `OTEL_*` (standard) | No change |
| `LLMROUTER_OTEL_SAMPLE_RATE` | Removed | Use standard `OTEL_TRACES_SAMPLER` |

## Appendix B: Resource Sizing Quick Reference

| Tier | CPU | RAM | Disk | Network | Monthly Cost Estimate |
|------|-----|-----|------|---------|----------------------|
| **Tier 1** (Minimal) | 1 vCPU | 2 GB | 10 GB | 100 Mbps | ~$0 (local) |
| **Tier 2** (Standard) | 4 vCPU | 8 GB | 50 GB | 1 Gbps | ~$150 (cloud VMs) |
| **Tier 3** (Production) | 12 vCPU | 24 GB | 100 GB | 1 Gbps | ~$800 (K8s + managed DB) |
| **Tier 4** (Enterprise) | 48+ vCPU | 96+ GB | 500+ GB | 10 Gbps | ~$5,000+ (multi-region) |

*Cost estimates assume AWS us-east-1 pricing as of 2026. Actual costs vary by provider,
region, and usage patterns.*
