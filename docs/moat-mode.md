# Gate 9: Moat-Mode Validation Plan + Cloud-Agnostic HA Readiness

**Document Version:** 1.0
**Date:** 2026-01-22
**Status:** Draft for Review
**Scope:** Air-gapped / Controlled-Egress Deployment Standards

---

## Executive Summary

This document establishes validation criteria and deployment protocols for operating the LiteLLM + LLMRouter gateway in **"moat-mode"** environments:

- **Air-gapped networks** with no external internet access
- **Controlled-egress** environments with explicit allowlists for outbound traffic
- **Standardized interfaces** requiring predictable network requirements
- **No cloud-managed dependencies** (e.g., no reliance on AWS Secrets Manager, Azure Key Vault)

**Key Characteristics:**
- ✅ All dependencies can be sourced offline
- ✅ Observability via self-hosted collectors (no cloud telemetry)
- ✅ Authentication via local mechanisms (API keys, optional OIDC)
- ✅ State persistence via on-premises databases (PostgreSQL, Redis)
- ✅ Zero-trust network architecture with explicit firewall rules

**Related Documentation:**
- [`docs/high-availability.md`](high-availability.md) - HA architecture and scaling patterns
- [`docs/observability.md`](observability.md) - OpenTelemetry integration options
- [`docs/hot-reloading.md`](hot-reloading.md) - Zero-downtime model updates
- [`docs/configuration.md`](configuration.md) - Configuration management
- [`GATE7_SECURITY_VALIDATION_REPORT.md`](../GATE7_SECURITY_VALIDATION_REPORT.md) - Security controls validation
- [`GATE8_OBSERVABILITY_VALIDATION_REPORT.md`](../GATE8_OBSERVABILITY_VALIDATION_REPORT.md) - Observability validation

---

## 1. Network Requirements & Firewall Rules

### 1.1 Required Network Ports

All network traffic flows **WITHIN** the moat perimeter. No external internet access required for runtime operation.

#### Inbound Ports (Load Balancer → Gateway)

| Port | Protocol | Purpose | TLS Required | Source |
|------|----------|---------|--------------|--------|
| **4000** | HTTP/HTTPS | LiteLLM API (v1/chat/completions, etc.) | ✅ Yes | Load balancer only |
| **4000** | SSE | Streaming responses (A2A message/stream) | ✅ Yes | Load balancer only |

**Firewall Rule Example:**
```bash
# Allow load balancer → gateway traffic on port 4000
iptables -A INPUT -p tcp -s <LOAD_BALANCER_CIDR> --dport 4000 -j ACCEPT
```

#### Outbound Ports (Gateway → External LLM Providers)

| Port | Protocol | Purpose | Destination | Moat-Mode Status |
|------|----------|---------|-------------|------------------|
| **443** | HTTPS | AWS Bedrock API | bedrock.{region}.amazonaws.com | ⚠️ Requires egress allow-list |
| **443** | HTTPS | OpenAI API | api.openai.com | ⚠️ Requires egress allow-list |
| **443** | HTTPS | Anthropic API | api.anthropic.com | ⚠️ Requires egress allow-list |

**Moat-Mode Constraint:** In air-gapped deployments, use **on-premises LLM providers** or **VPC-private endpoints** (e.g., AWS PrivateLink for Bedrock).

**Firewall Rule Example (Controlled-Egress):**
```bash
# Allow gateway → Bedrock via PrivateLink (VPC endpoint)
iptables -A OUTPUT -p tcp -d <VPC_ENDPOINT_IP> --dport 443 -j ACCEPT

# Deny all other outbound HTTPS (default-deny posture)
iptables -A OUTPUT -p tcp --dport 443 -j DROP
```

#### Internal Services Ports (Within Moat)

| Service | Port | Protocol | Purpose | TLS Required |
|---------|------|----------|---------|--------------|
| **PostgreSQL** | 5432 | TCP | State persistence (keys, budgets, logs) | ✅ Recommended (mTLS) |
| **Redis** | 6379 | TCP | Caching, rate limiting | ⚠️ Optional (internal network) |
| **OTEL Collector** | 4317 | gRPC | Trace export (OTLP) | ✅ Yes |
| **OTEL Collector** | 4318 | HTTP | Trace export (OTLP/HTTP) | ✅ Yes |
| **Nginx** | 80/443 | HTTP/S | Load balancer front-end | ✅ Yes (443) |

**Principle:** All inter-service communication **STAYS WITHIN** the moat perimeter (same VPC, same Kubernetes namespace, or dedicated VLAN).

---

### 1.2 TLS Termination Patterns

#### Option A: TLS Termination at Load Balancer (Recommended for Moat-Mode)

```
┌─────────────────┐
│   External      │
│   Client        │  HTTPS (TLS 1.3)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Nginx / ALB    │  ← TLS termination here
│  (Load Balancer)│
└────────┬────────┘
         │ HTTP (cleartext within VPC)
         ▼
┌─────────────────┐
│  Gateway 1/2    │
└─────────────────┘
```

**Configuration:**
```nginx
# config/nginx.conf
server {
    listen 443 ssl http2;
    ssl_certificate /etc/nginx/certs/tls.crt;
    ssl_certificate_key /etc/nginx/certs/tls.key;
    ssl_protocols TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-CHACHA20-POLY1305;
    ssl_prefer_server_ciphers on;

    location / {
        proxy_pass http://litellm_backend;  # HTTP within VPC
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

**Advantages:**
- ✅ Single TLS certificate management point
- ✅ Gateway containers remain simple (no cert rotation)
- ✅ Better performance (TLS offload)

**Certificate Source (Moat-Mode):**
- On-premises PKI/CA (e.g., HashiCorp Vault, EJBCA)
- **NOT** AWS ACM or cloud-managed certificates

---

#### Option B: End-to-End TLS (mTLS)

For **zero-trust** environments requiring encryption at every hop:

```
Client → [TLS] → Nginx → [TLS] → Gateway → [TLS] → Postgres
                                           → [TLS] → Redis
```

**Gateway Configuration:**
```yaml
# docker-compose.ha.yml (excerpt)
litellm-gateway-1:
  environment:
    - LITELLM_SSL_CERT=/app/certs/gateway.crt
    - LITELLM_SSL_KEY=/app/certs/gateway.key
```

**PostgreSQL mTLS:**
```bash
# postgresql.conf
ssl = on
ssl_cert_file = '/etc/ssl/certs/server.crt'
ssl_key_file = '/etc/ssl/private/server.key'
ssl_ca_file = '/etc/ssl/certs/ca.crt'
ssl_crl_file = ''
```

**Moat-Mode Constraint:** All certificates **MUST** be issued by internal CA. External CA (Let's Encrypt, DigiCert) requires internet access.

---

### 1.3 mTLS Considerations

Mutual TLS (mTLS) provides **client certificate authentication**:

1. **Gateway ↔ LLM Provider (AWS Bedrock PrivateLink)**
   - Bedrock VPC endpoints support TLS by default
   - Client certificates optional but recommended in zero-trust

2. **Gateway ↔ PostgreSQL**
   - Use `sslmode=verify-full` in connection string
   - Client cert for gateway service account

3. **Gateway ↔ Redis**
   - Redis 6.0+ supports TLS/SSL
   - Optional ACL-based authentication

**Example DATABASE_URL with mTLS:**
```bash
DATABASE_URL="postgresql://litellm:password@postgres:5432/litellm?sslmode=verify-full&sslcert=/app/certs/client.crt&sslkey=/app/certs/client.key&sslrootcert=/app/certs/ca.crt"
```

---

## 2. Identity & Authentication

### 2.1 API Key Authentication (Primary Method)

**Validation Status:** ✅ Verified in [`GATE7_SECURITY_VALIDATION_REPORT.md`](../GATE7_SECURITY_VALIDATION_REPORT.md)

#### Master Key Management (Moat-Mode)

**NOT cloud-managed secrets** (AWS Secrets Manager, Azure Key Vault). Use:

1. **Kubernetes Secrets** (for K8s deployments):
   ```yaml
   apiVersion: v1
   kind: Secret
   metadata:
     name: litellm-master-key
   type: Opaque
   stringData:
     LITELLM_MASTER_KEY: "sk-master-production-abc123..."
   ```

2. **Docker Secrets** (for Docker Swarm):
   ```bash
   echo "sk-master-key" | docker secret create litellm_master_key -
   ```

3. **Environment Variables** (on-premises VMs):
   ```bash
   # /etc/litellm/environment (restricted to root:litellm)
   LITELLM_MASTER_KEY=sk-master-key-change-me
   ```

4. **HashiCorp Vault** (self-hosted KV store):
   ```bash
   vault kv put secret/litellm master_key="sk-..."
   ```

**Security Requirements:**
- ✅ Key rotation every 90 days
- ✅ Audit logging for `/key/generate` API calls
- ✅ Master key redacted from logs (verified in Gate 7)
- ✅ Keys stored in PostgreSQL with encryption at rest

---

### 2.2 Virtual Keys / Team Isolation

**Reference:** [`GATE7_SECURITY_VALIDATION_REPORT.md` Section 2-4](../GATE7_SECURITY_VALIDATION_REPORT.md)

Virtual keys provide multi-tenancy with budget enforcement:

```bash
# Create team-scoped key
curl -X POST http://localhost:8080/key/generate \
  -H "Authorization: Bearer ${LITELLM_MASTER_KEY}" \
  -d '{
    "team_id": "engineering-team",
    "max_budget": 100.0,
    "duration": "30d",
    "models": ["claude-haiku", "gpt-4"]
  }'
```

**Moat-Mode Requirements:**
- ✅ Keys persisted in PostgreSQL (no cloud database)
- ✅ Budget tracking in local database
- ✅ Rate limiting via Redis (in-memory, ephemeral state acceptable)

---

### 2.3 OIDC/SSO Integration (Optional)

For organizations with existing identity providers (Keycloak, Okta on-premises):

**Not Yet Implemented** - Future roadmap item. Current workaround:

1. External OIDC proxy (e.g., oauth2-proxy) in front of Nginx
2. Proxy validates JWT → sets `X-User-ID` header
3. LiteLLM validates master key or virtual key as usual

**Requirements for Moat-Mode:**
- ✅ OIDC provider must be **on-premises** (not auth0.com or okta.com SaaS)
- ✅ Token validation via local JWKS endpoint
- ✅ No external HTTP requests during auth flow

---

## 3. Data Persistence & Backup/Restore

### 3.1 PostgreSQL State Management

**Purpose:** Persistent storage for:
- Virtual API keys and budgets
- Team configurations
- Request logs (if `STORE_MODEL_IN_DB=true`)
- User/organization mappings

#### Backup Strategy

**Daily Automated Backups:**
```bash
#!/bin/bash
# /opt/litellm/backup-postgres.sh
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR=/backup/postgres
docker exec litellm-postgres pg_dump -U litellm litellm | \
  gzip > ${BACKUP_DIR}/litellm_${TIMESTAMP}.sql.gz

# Retain last 30 days
find ${BACKUP_DIR} -name "litellm_*.sql.gz" -mtime +30 -delete
```

**Cron Schedule:**
```cron
0 2 * * * /opt/litellm/backup-postgres.sh
```

#### Point-in-Time Recovery (PITR)

For critical deployments, enable WAL archiving:

```bash
# postgresql.conf
wal_level = replica
archive_mode = on
archive_command = 'cp %p /backup/postgres/wal_archive/%f'
```

**Restore Procedure:**
```bash
# 1. Stop gateway instances
docker compose stop litellm-gateway-1 litellm-gateway-2

# 2. Restore database
gunzip -c /backup/postgres/litellm_20260122_020000.sql.gz | \
  docker exec -i litellm-postgres psql -U litellm litellm

# 3. Restart gateways
docker compose start litellm-gateway-1 litellm-gateway-2
```

---

### 3.2 Redis State Management

**Purpose:** Ephemeral caching and rate limiting:
- Response caching (LRU eviction policy)
- Rate limit counters (short-lived)
- Session data (optional)

#### Backup Strategy

**Redis persistence options:**

1. **AOF (Append-Only File)** - Recommended for moat-mode:
   ```bash
   # docker-compose.ha.yml
   redis:
     command: redis-server --appendonly yes --appendfsync everysec
   ```

2. **RDB Snapshots** - For point-in-time recovery:
   ```bash
   # Save snapshot every hour if 1000+ keys changed
   redis:
     command: redis-server --save 3600 1000 --dir /data
   ```

**Backup Script:**
```bash
#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
docker exec litellm-redis redis-cli SAVE
docker cp litellm-redis:/data/dump.rdb /backup/redis/dump_${TIMESTAMP}.rdb
```

**Restore Procedure:**
```bash
docker compose stop redis
docker cp /backup/redis/dump_<timestamp>.rdb litellm-redis:/data/dump.rdb
docker compose start redis
```

**Moat-Mode Consideration:** Redis backup is **optional**. If Redis data is lost (e.g., node failure), the gateway will rebuild cache from database + upstream LLM providers.

---

### 3.3 Model Artifacts (LLMRouter)

**Purpose:** Trained routing models (`.pt` files for KNN, SVM, MLP strategies).

**Storage Options:**

1. **Volume Mounts** (HA-compatible with shared NFS):
   ```yaml
   volumes:
     - /mnt/nfs/llmrouter/models:/app/models:ro
   ```

2. **S3-Compatible Storage** (MinIO, Ceph):
   ```bash
   # Environment variables
   LLMROUTER_MODEL_S3_BUCKET=llm-models
   LLMROUTER_MODEL_S3_KEY=knn_router_v2.pt
   # Use MinIO endpoint instead of AWS S3
   AWS_ENDPOINT_URL=http://minio:9000
   ```

**Backup:**
```bash
# Version models in S3/MinIO with object versioning enabled
mc version enable myminio/llm-models
```

**Disaster Recovery:**
```bash
# Restore specific model version
mc cp myminio/llm-models/knn_router_v2.pt?versionId=<version> ./models/
```

---

## 4. Upgrade Strategy

### 4.1 Rolling Deployment (Zero-Downtime)

**Prerequisites:**
- ✅ Nginx load balancer with health checks
- ✅ Shared PostgreSQL/Redis (state synchronization)
- ✅ Database migrations tested in staging

**Procedure:**

```bash
# 1. Pull new image
docker pull ghcr.io/baladithyab/litellm-llm-router:v2.0.0

# 2. Update gateway-1 (Nginx routes traffic to gateway-2)
docker compose stop litellm-gateway-1
docker compose up -d litellm-gateway-1

# Wait for health check
while ! curl -f http://localhost:4000/health/readiness; do
  sleep 2
done

# 3. Update gateway-2
docker compose stop litellm-gateway-2
docker compose up -d litellm-gateway-2
```

**Nginx Health Check Configuration:**
```nginx
upstream litellm_backend {
    server litellm-gateway-1:4000 max_fails=3 fail_timeout=30s;
    server litellm-gateway-2:4000 max_fails=3 fail_timeout=30s;
}
```

**Kubernetes Rolling Update:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: litellm-gateway
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  template:
    spec:
      containers:
      - name: gateway
        image: ghcr.io/baladithyab/litellm-llm-router:v2.0.0
        readinessProbe:
          httpGet:
            path: /health/readiness
            port: 4000
          initialDelaySeconds: 15
          periodSeconds: 10
```

---

### 4.2 Database Schema Migrations

**Challenge:** Postgres schema changes must be backward-compatible during rolling deploys.

**Solution:** Blue-Green Database Migration

```mermaid
graph LR
    A[Old Gateway v1.0] -->|Reads/Writes| B[(DB Schema v1)]
    C[New Gateway v2.0] -->|Reads/Writes| B
    B -->|Migrate| D[(DB Schema v2)]
    C -->|Switch to| D
    A -->|Shutdown| E[Terminated]
```

**Migration Script (Idempotent):**
```sql
-- migrations/v2_add_budget_alerts.sql
BEGIN;

-- Add column if not exists (backward-compatible)
ALTER TABLE litellm_verificationtoken
ADD COLUMN IF NOT EXISTS budget_alert_threshold DECIMAL(10, 2) DEFAULT 0.8;

-- Create index (non-blocking in Postgres 11+)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_budget_alerts
ON litellm_verificationtoken(budget_alert_threshold);

COMMIT;
```

**Execution:**
```bash
# Run before deploying new gateway version
docker exec litellm-postgres psql -U litellm litellm < migrations/v2_add_budget_alerts.sql
```

**Rollback Plan:**
```sql
-- Backward-compatible: old gateways ignore new column
-- If rollback needed:
ALTER TABLE litellm_verificationtoken DROP COLUMN IF EXISTS budget_alert_threshold;
```

---

### 4.3 Split-Brain Prevention

**Risk:** In HA deployments, two gateway instances might diverge if:
- Redis connection lost (rate limit state)
- Postgres connection lost (key validation state)

**Mitigation 1: Circuit Breaker for Database**

```python
# Pseudocode: observability.py
if not postgres_connection.is_alive():
    # Stop accepting new requests
    return 503, "Database unavailable - circuit breaker open"
```

**Mitigation 2: Shared Redis Sentinel (HA Redis)**

```yaml
# docker-compose.ha.yml (future improvement)
redis-sentinel-1:
  image: redis:7-alpine
  command: redis-sentinel /etc/redis/sentinel.conf

redis-sentinel-2:
  image: redis:7-alpine
  command: redis-sentinel /etc/redis/sentinel.conf
```

**Sentinel Configuration:**
```conf
# sentinel.conf
sentinel monitor mymaster redis 6379 2
sentinel down-after-milliseconds mymaster 5000
sentinel failover-timeout mymaster 10000
```

**Mitigation 3: Lease-Based Leadership (Future)**

For critical operations (config reload), use database-backed leases:

```sql
-- Only one gateway instance holds lease at a time
INSERT INTO system_locks (lock_name, holder_id, expires_at)
VALUES ('config_reload', 'gateway-1', NOW() + INTERVAL '30 seconds')
ON CONFLICT (lock_name) DO NOTHING;
```

---

## 5. Observability Requirements for Restricted Networks

**Reference:** [`GATE8_OBSERVABILITY_VALIDATION_REPORT.md`](../GATE8_OBSERVABILITY_VALIDATION_REPORT.md)

### 5.1 OpenTelemetry Collector Topology

**Moat-Mode Constraint:** No cloud telemetry (CloudWatch, X-Ray, Datadog SaaS).

**Solution:** Self-hosted OTEL collector + storage backend.

```
┌─────────────┐
│  Gateway 1  │──┐
└─────────────┘  │
                 │ OTLP/gRPC (4317)
┌─────────────┐  │
│  Gateway 2  │──┼──► ┌────────────────┐
└─────────────┘  │    │ OTEL Collector │
                 │    │  (on-premises) │
┌─────────────┐  │    └────────┬───────┘
│  MCP Server │──┘             │
└─────────────┘                │
                              ▼
              ┌────────────────────────────┐
              │  Storage Backend           │
              │  - Jaeger (traces)         │
              │  - Prometheus (metrics)    │
              │  - Loki (logs)             │
              └────────────────────────────┘
```

**OTEL Collector Configuration:**
```yaml
# config/otel-collector-moat.yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:
    timeout: 10s
    send_batch_size: 1024

exporters:
  # Local Jaeger instance (no internet required)
  jaeger:
    endpoint: jaeger:14250
    tls:
      insecure: false
      cert_file: /etc/otel/certs/tls.crt
      key_file: /etc/otel/certs/tls.key

  # Prometheus remote write (on-premises Prometheus)
  prometheusremotewrite:
    endpoint: http://prometheus:9090/api/v1/write
    tls:
      insecure: true  # Within VPC

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [jaeger]
    metrics:
      receivers: [otlp]
      processors: [batch]
      exporters: [prometheusremotewrite]
```

**Gateway Configuration:**
```yaml
# docker-compose.ha.yml
litellm-gateway-1:
  environment:
    - OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
    - OTEL_EXPORTER_OTLP_PROTOCOL=grpc
    - OTEL_SERVICE_NAME=litellm-gateway-1
    - OTEL_TRACES_EXPORTER=otlp
    - OTEL_METRICS_EXPORTER=otlp
```

**Storage Backend Options:**

1. **Jaeger + Cassandra** (for high-volume traces):
   ```yaml
   jaeger:
     environment:
       - SPAN_STORAGE_TYPE=cassandra
       - CASSANDRA_SERVERS=cassandra:9042
   ```

2. **Grafana Tempo + S3-compatible** (MinIO):
   ```yaml
   tempo:
     volumes:
       - ./tempo-config.yaml:/etc/tempo.yaml
     environment:
       - S3_ENDPOINT=http://minio:9000
   ```

**Moat-Mode Validation:** All OTEL components must be **containerized** and **air-gapped compatible** (no external DNS lookups).

---

### 5.2 Log Aggregation (No Cloud Logging)

**Avoid:** AWS CloudWatch Logs, Azure Monitor Logs, Datadog

**Use:** Self-hosted solutions:

1. **Fluent Bit → Loki** (recommended for moat-mode):
   ```yaml
   fluent-bit:
     image: fluent/fluent-bit:2.0
     volumes:
       - ./fluent-bit.conf:/fluent-bit/etc/fluent-bit.conf
     environment:
       - LOKI_URL=http://loki:3100
   ```

2. **Syslog → Local File** (simplest):
   ```yaml
   litellm-gateway-1:
     logging:
       driver: syslog
       options:
         syslog-address: "tcp://syslog-server:514"
   ```

**Log Retention Policy:**
```bash
# Rotate logs every 7 days, keep 30 days
/var/log/litellm/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
}
```

---

### 5.3 Metrics Endpoint Accessibility

**Prometheus Scrape Configuration:**
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'litellm-gateway'
    static_configs:
      - targets: ['litellm-gateway-1:4000', 'litellm-gateway-2:4000']
    metrics_path: '/metrics'
    scheme: http  # Within VPC
```

**Key Metrics:**
- `litellm_requests_total` - Total API requests
- `litellm_latency_seconds` - Request latency histogram
- `llmrouter_routing_decision` - Routing strategy distribution
- `redis_cache_hit_rate` - Cache effectiveness

**Alert Rules (moat-mode):**
```yaml
# prometheus-alerts.yml
groups:
  - name: litellm
    rules:
      - alert: HighErrorRate
        expr: rate(litellm_requests_total{status="error"}[5m]) > 0.05
        annotations:
          summary: "High error rate detected"

      - alert: DatabaseConnectionLost
        expr: up{job="postgres"} == 0
        for: 1m
        annotations:
          summary: "PostgreSQL unreachable - circuit breaker may trip"
```

---

## 6. Supply-Chain & Offline Build Considerations

### 6.1 Container Image Strategy

**Problem:** Default Docker builds pull dependencies from internet (PyPI, npm, etc.).

**Solution:** Air-gapped container build process.

#### Step 1: Vendor Dependencies

```bash
# vendor-deps.sh
#!/bin/bash
set -e

# Download Python dependencies (no external network during build)
mkdir -p vendor/python
pip download -r requirements.txt -d vendor/python

# Download Node.js dependencies (if UI exists)
mkdir -p vendor/node
npm pack --pack-destination=vendor/node

# Download base Docker images
docker pull python:3.11-slim
docker save python:3.11-slim -o vendor/images/python.tar

docker pull postgres:16-alpine
docker save postgres:16-alpine -o vendor/images/postgres.tar
```

#### Step 2: Offline Dockerfile

```dockerfile
# docker/Dockerfile.airgap
FROM python:3.11-slim

# Copy vendored Python packages (NO pip install from PyPI)
COPY vendor/python /tmp/vendor/python
RUN pip install --no-index --find-links=/tmp/vendor/python -r requirements.txt

# Copy application code
COPY src /app/src
COPY config /app/config

WORKDIR /app
CMD ["python", "-m", "uvicorn", "litellm.proxy.proxy_server:app", "--host", "0.0.0.0", "--port", "4000"]
```

#### Step 3: Build in Air-Gapped Environment

```bash
# Load vendored base images
docker load -i vendor/images/python.tar

# Build without internet
docker build -f docker/Dockerfile.airgap -t litellm-llm-router:v2.0.0 .
```

**Export for Transfer to Moat:**
```bash
docker save litellm-llm-router:v2.0.0 | gzip > litellm-v2.0.0.tar.gz

# Transfer via sneakernet (USB drive, etc.) to air-gapped environment
# Then load:
docker load -i litellm-v2.0.0.tar.gz
```

---

### 6.2 Dependency Pinning

**Requirements:**
- ✅ All Python dependencies pinned to exact versions
- ✅ No `*` or `>=` in `requirements.txt`
- ✅ Hashes verified for supply-chain security

**requirements.lock (generated via pip-tools):**
```txt
# requirements.lock
litellm==1.50.5 \
    --hash=sha256:abc123...
fastapi==0.109.0 \
    --hash=sha256:def456...
pydantic==2.5.3 \
    --hash=sha256:ghi789...
```

**Build Command:**
```bash
pip install --require-hashes -r requirements.lock
```

**Security Scanning (in CI/CD, before air-gap transfer):**
```bash
# Scan for vulnerabilities
docker run --rm -v $(pwd):/workspace aquasec/trivy image litellm-llm-router:v2.0.0

# Fail build if HIGH/CRITICAL CVEs found
trivy image --severity HIGH,CRITICAL --exit-code 1 litellm-llm-router:v2.0.0
```

---

### 6.3 Git Submodules & Vendored Code

**Issue:** LiteLLM references external GitHub repos (e.g., model configs).

**Solution:** Vendor submodules into image.

```bash
# .gitmodules
[submodule "reference/litellm"]
    path = reference/litellm
    url = https://github.com/BerriAI/litellm.git
    branch = main
```

**Dockerfile Integration:**
```dockerfile
# Copy LiteLLM reference code (no git clone during build)
COPY reference/litellm /app/reference/litellm
```

**Update Procedure (before air-gap transfer):**
```bash
# On internet-connected build server
git submodule update --remote
git add reference/litellm
git commit -m "chore: update litellm submodule to v1.50.5"
```

---

## 7. Acceptance Tests for Moat-Mode Compliance

### 7.1 Network Isolation Test

**Objective:** Verify gateway operates without internet access.

```bash
#!/bin/bash
# tests/moat-mode/test_network_isolation.sh

# 1. Block all outbound traffic except internal services
iptables -A OUTPUT -d 10.0.0.0/8 -j ACCEPT    # Allow VPC
iptables -A OUTPUT -d 172.16.0.0/12 -j ACCEPT # Allow Docker network
iptables -A OUTPUT -j DROP                    # Deny all else

# 2. Start gateway stack
docker compose -f docker-compose.ha.yml up -d

# 3. Wait for health checks
sleep 30

# 4. Test API endpoint (should work with on-premises LLM)
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" \
  http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-master-key" \
  -d '{"model": "on-prem-llm", "messages": [{"role": "user", "content": "test"}]}')

if [ "$RESPONSE" == "200" ]; then
  echo "✅ PASS: Gateway operational without internet"
else
  echo "❌ FAIL: Gateway requires external access (HTTP $RESPONSE)"
  exit 1
fi

# 5. Restore firewall rules
iptables -F OUTPUT
```

**Expected Result:** Gateway serves requests using only internal resources.

---

### 7.2 TLS Certificate Validation

**Objective:** Ensure all TLS certs are from internal CA.

```python
# tests/moat-mode/test_tls_internal_ca.py
import ssl
import socket

def test_nginx_tls_cert():
    context = ssl.create_default_context()
    # Load internal CA cert
    context.load_verify_locations('/etc/pki/tls/certs/internal-ca.crt')

    with socket.create_connection(('nginx', 443)) as sock:
        with context.wrap_socket(sock, server_hostname='nginx') as ssock:
            cert = ssock.getpeercert()
            issuer = dict(x[0] for x in cert['issuer'])

            assert 'Internal CA' in issuer['organizationName'], \
                f"Certificate not from internal CA: {issuer}"

            print("✅ PASS: TLS certificate issued by internal CA")

if __name__ == '__main__':
    test_nginx_tls_cert()
```

**Expected Result:** All certificates trace to on-premises CA root.

---

### 7.3 Database Backup/Restore Validation

**Objective:** Verify zero data loss during disaster recovery.

```bash
#!/bin/bash
# tests/moat-mode/test_backup_restore.sh

# 1. Create test data
curl -X POST http://localhost:8080/key/generate \
  -H "Authorization: Bearer sk-master-key" \
  -d '{"metadata": {"test": "backup-restore"}}' | tee /tmp/test_key.json

TEST_KEY=$(jq -r .key /tmp/test_key.json)

# 2. Perform backup
docker exec litellm-postgres pg_dump -U litellm litellm > /tmp/backup.sql

# 3. Simulate disaster (drop database)
docker exec litellm-postgres psql -U litellm -c "DROP DATABASE litellm;"
docker exec litellm-postgres psql -U litellm -c "CREATE DATABASE litellm;"

# 4. Restore from backup
cat /tmp/backup.sql | docker exec -i litellm-postgres psql -U litellm litellm

# 5. Verify test key still works
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" \
  http://localhost:8080/key/info \
  -H "Authorization: Bearer $TEST_KEY")

if [ "$RESPONSE" == "200" ]; then
  echo "✅ PASS: Backup/restore successful, key recovered"
else
  echo "❌ FAIL: Key lost after restore (HTTP $RESPONSE)"
  exit 1
fi
```

**Expected Result:** All virtual keys functional after restore.

---

### 7.4 Observability Self-Containment Test

**Objective:** Traces/metrics stored locally, no cloud export.

```python
# tests/moat-mode/test_otel_local_export.py
import requests
import time

def test_traces_in_jaeger():
    # 1. Generate traced request
    resp = requests.post(
        'http://localhost:8080/v1/chat/completions',
        headers={'Authorization': 'Bearer sk-master-key'},
        json={'model': 'test-model', 'messages': [{'role': 'user', 'content': 'test'}]}
    )

    # 2. Wait for trace export
    time.sleep(5)

    # 3. Query Jaeger API (local instance)
    traces = requests.get('http://localhost:16686/api/traces?service=litellm-gateway-1').json()

    assert len(traces['data']) > 0, "No traces found in local Jaeger"
    print(f"✅ PASS: {len(traces['data'])} traces exported to local Jaeger")

    # 4. Verify no external HTTP requests in trace spans
    for trace in traces['data']:
        for span in trace['spans']:
            for tag in span['tags']:
                if tag['key'] == 'http.url':
                    url = tag['value']
                    # Fail if any span calls external services
                    assert not any(domain in url for domain in ['amazonaws.com', 'openai.com']), \
                        f"External URL detected in span: {url} (moat-mode violation)"

    print("✅ PASS: All spans local, no external telemetry")

if __name__ == '__main__':
    test_traces_in_jaeger()
```

**Expected Result:** All telemetry stays within moat perimeter.

---

### 7.5 Dependency Integrity Test

**Objective:** Verify offline install works (no PyPI access).

```bash
#!/bin/bash
# tests/moat-mode/test_offline_build.sh

# 1. Block outbound traffic to PyPI
iptables -A OUTPUT -d 151.101.0.0/16 -j DROP  # pypi.org
iptables -A OUTPUT -d 199.232.0.0/16 -j DROP  # files.pythonhosted.org

# 2. Attempt to build Docker image (should succeed via vendor/ dir)
docker build -f docker/Dockerfile.airgap -t litellm-test-offline .

BUILD_STATUS=$?

# 3. Restore network
iptables -D OUTPUT -d 151.101.0.0/16 -j DROP
iptables -D OUTPUT -d 199.232.0.0/16 -j DROP

if [ $BUILD_STATUS -eq 0 ]; then
  echo "✅ PASS: Offline build successful"
else
  echo "❌ FAIL: Build requires internet access"
  exit 1
fi
```

**Expected Result:** Docker build completes without external network.

---

## 8. Cloud-Agnostic HA Readiness Checklist

### 8.1 Infrastructure Requirements

| Component | Requirement | AWS Example | On-Premises Example | Status |
|-----------|-------------|-------------|---------------------|--------|
| **Compute** | 3+ gateway instances | EC2 t3.xlarge | VMware VMs (8 vCPU, 16GB RAM) | ⬜ |
| **Load Balancer** | L7 with health checks | ALB | Nginx, HAProxy | ⬜ |
| **Database** | PostgreSQL 13+ with HA | RDS Multi-AZ | Patroni cluster | ⬜ |
| **Cache** | Redis 6+ with persistence | ElastiCache | Redis Sentinel | ⬜ |
| **Storage** | Shared FS or object store | EFS + S3 | NFS + MinIO | ⬜ |
| **Observability** | OTEL collector + storage | CloudWatch | Jaeger + Prometheus | ⬜ |

---

### 8.2 Network Security Checklist

| Control | Description | Validation | Status |
|---------|-------------|------------|--------|
| **TLS Termination** | All external traffic uses TLS 1.3 | `openssl s_client -connect nginx:443 -tls1_3` | ⬜ |
| **mTLS (Optional)** | Client certs for DB/Redis | `psql "sslmode=verify-full"` connection test | ⬜ |
| **Firewall Rules** | Only required ports open (4000, 5432, 6379, 4317) | `nmap -p 1-65535 <gateway-ip>` | ⬜ |
| **No Internet Egress** | Block outbound except to on-prem LLM | Network isolation test (Section 7.1) | ⬜ |
| **Internal DNS** | All hostnames resolvable via internal DNS | `nslookup postgres` from gateway | ⬜ |

---

### 8.3 Data Persistence Checklist

| Task | Recovery Time Objective | Validation | Status |
|------|------------------------|------------|--------|
| **Daily Postgres Backup** | RPO: 24 hours | Automated cron job running | ⬜ |
| **Postgres PITR (Optional)** | RPO: < 5 minutes | WAL archiving enabled | ⬜ |
| **Redis AOF Persistence** | RPO: 1 second | `appendfsync everysec` configured | ⬜ |
| **Backup Restore Test** | RTO: < 1 hour | Section 7.3 test passed | ⬜ |
| **Off-Site Backup** | Copies stored in separate location | rsync to secondary datacenter | ⬜ |

---

### 8.4 High Availability Checklist

| Feature | Implementation | Test Procedure | Status |
|---------|---------------|----------------|--------|
| **Active-Active Gateways** | 2+ replicas behind LB | Kill one instance, verify requests still processed | ⬜ |
| **Database Failover** | Postgres streaming replication | Simulate primary failure, check standby promotion | ⬜ |
| **Redis Failover** | Sentinel-based HA | `redis-cli -p 26379 sentinel failover mymaster` | ⬜ |
| **Config Sync** | Hot reload without restart | `curl -X POST /config/reload` | ⬜ |
| **Split-Brain Prevention** | Shared state in Postgres | Disconnect Redis, verify circuit breaker | ⬜ |

---

### 8.5 Deployment Validation Checklist

| Phase | Action | Success Criteria | Status |
|-------|--------|------------------|--------|
| **Pre-Deployment** | Run moat-mode acceptance tests (Section 7) | All tests pass | ⬜ |
| **Staging Deploy** | Deploy to staging environment | No errors in logs, health checks pass | ⬜ |
| **Production Deploy** | Rolling update (Section 4.1) | Zero downtime, all replicas healthy | ⬜ |
| **Post-Deploy** | Verify observability | Traces visible in Jaeger within 5 min | ⬜ |
| **Rollback Plan** | Document rollback procedure | Previous version image available locally | ⬜ |

---

### 8.6 Security Hardening Checklist

| Control | Requirement | Validation | Status |
|---------|-------------|------------|--------|
| **Master Key Rotation** | Rotate every 90 days | Key creation date tracked in database | ⬜ |
| **Virtual Key Expiry** | All keys have TTL | No keys in DB with `expires_at IS NULL` | ⬜ |
| **Audit Logging** | All `/key/*` API calls logged | Query audit table for test period | ⬜ |
| **Secrets Redaction** | No sensitive data in logs | Section 7 Gate 7 validation passed | ⬜ |
| **Container Security** | Images scanned for CVEs | Trivy scan shows 0 CRITICAL | ⬜ |
| **Least Privilege** | Services run as non-root | `docker exec gateway whoami` returns `litellm` | ⬜ |

---

### 8.7 Observability Readiness Checklist

| Metric | SLO | Dashboard | Alert | Status |
|--------|-----|-----------|-------|--------|
| **API Latency (P95)** | < 2 seconds | Grafana panel: `histogram_quantile(0.95, litellm_latency_seconds)` | > 5s for 5m | ⬜ |
| **Error Rate** | < 1% | `rate(litellm_requests_total{status="error"}[5m])` | > 5% for 5m | ⬜ |
| **Cache Hit Rate** | > 80% | `redis_cache_hit_rate` | < 50% for 10m | ⬜ |
| **Database Connections** | < 80% of max | `pg_stat_database` | > 90% for 5m | ⬜ |
| **Trace Sampling** | 100% (moat-mode) | All requests in Jaeger | Missing traces | ⬜ |

---

### 8.8 Disaster Recovery Checklist

| Scenario | RTO | RPO | Procedure | Status |
|----------|-----|-----|-----------|--------|
| **Single Gateway Failure** | 0 seconds (LB failover) | 0 (stateless) | LB automatically routes to healthy replica | ⬜ |
| **Postgres Failure** | 5 minutes | < 5 min (PITR) | Promote standby, update gateway env vars | ⬜ |
| **Redis Failure** | 30 seconds | Acceptable (cache can rebuild) | Start new Redis, gateway reconnects automatically | ⬜ |
| **Complete Cluster Loss** | 2 hours | 24 hours (daily backup) | Restore from backup (Section 3.1), redeploy | ⬜ |
| **Data Center Failure** | 4 hours | 24 hours | Failover to secondary DC, restore from off-site backup | ⬜ |

---

## 9. Deployment Patterns for Moat Environments

### 9.1 On-Premises VM Deployment

**Infrastructure:**
- 3x VMs (gateway replicas)
- 1x Nginx LB VM
- 1x Postgres primary + 1x Postgres standby
- 1x Redis Sentinel cluster (3 nodes)
- 1x OTEL collector VM

**Example Terraform (VMware vSphere):**
```hcl
resource "vsphere_virtual_machine" "litellm_gateway" {
  count = 3
  name  = "litellm-gateway-${count.index + 1}"

  resource_pool_id = data.vsphere_resource_pool.pool.id
  datastore_id     = data.vsphere_datastore.datastore.id

  num_cpus = 8
  memory   = 16384

  network_interface {
    network_id = data.vsphere_network.internal.id
  }

  disk {
    label = "disk0"
    size  = 100
  }
}
```

---

### 9.2 Kubernetes Deployment (Air-Gapped)

**Helm Chart Structure:**
```
charts/litellm-llmrouter/
├── Chart.yaml
├── values.yaml
├── templates/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── configmap.yaml
│   └── secret.yaml
```

**Key Differences for Moat-Mode:**
```yaml
# values-moat.yaml
image:
  repository: internal-registry.company.com/litellm-llm-router
  pullPolicy: IfNotPresent  # Never pull from internet
  tag: "v2.0.0"

service:
  type: ClusterIP  # No public LoadBalancer

ingress:
  enabled: true
  className: nginx
  tls:
    - secretName: litellm-tls-internal
      hosts:
        - litellm.internal.company.com

postgresql:
  enabled: false  # Use external Postgres cluster
  external:
    host: postgres-cluster.internal
    port: 5432

redis:
  enabled: false  # Use external Redis Sentinel
  external:
    host: redis-sentinel.internal
    port: 26379

observability:
  otelCollector:
    endpoint: http://otel-collector.observability.svc.cluster.local:4317
```

**Deploy:**
```bash
# Add Helm repo from internal chart museum
helm repo add litellm https://charts.internal.company.com

# Install in air-gapped cluster
helm install litellm litellm/litellm-llmrouter \
  -f values-moat.yaml \
  --namespace litellm \
  --create-namespace
```

---

### 9.3 Docker Swarm Deployment

**Stack File:**
```yaml
# docker-stack-moat.yml
version: '3.8'

services:
  litellm-gateway:
    image: internal-registry:5000/litellm-llm-router:v2.0.0
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 30s
        order: start-first
      restart_policy:
        condition: on-failure
    environment:
      - DATABASE_URL=postgresql://litellm:pass@postgres:5432/litellm
    secrets:
      - litellm_master_key
    networks:
      - litellm-internal

  nginx:
    image: nginx:alpine
    deploy:
      replicas: 2
    ports:
      - "443:443"
    configs:
      - source: nginx_config
        target: /etc/nginx/nginx.conf
    networks:
      - litellm-internal

secrets:
  litellm_master_key:
    external: true

configs:
  nginx_config:
    file: ./config/nginx.conf

networks:
  litellm-internal:
    driver: overlay
    encrypted: true
```

**Deploy:**
```bash
docker stack deploy -c docker-stack-moat.yml litellm
```

---

## 10. Compliance & Audit

### 10.1 Audit Trail Requirements

**Database Audit Table:**
```sql
CREATE TABLE audit_log (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    user_id VARCHAR(255),
    action VARCHAR(50),  -- 'key.generate', 'key.delete', 'config.reload'
    resource_id VARCHAR(255),
    source_ip INET,
    success BOOLEAN,
    metadata JSONB
);

-- Index for queries
CREATE INDEX idx_audit_timestamp ON audit_log(timestamp DESC);
CREATE INDEX idx_audit_user ON audit_log(user_id);
```

**Automatic Logging (FastAPI middleware):**
```python
# src/litellm_llmrouter/audit.py
@app.middleware("http")
async def audit_middleware(request: Request, call_next):
    if request.url.path.startswith("/key/"):
        # Log request details to database
        await db.execute(
            "INSERT INTO audit_log (user_id, action, source_ip) VALUES ($1, $2, $3)",
            request.headers.get("x-user-id"),
            f"{request.method} {request.url.path}",
            request.client.host
        )

    response = await call_next(request)
    return response
```

**Query Audit Logs:**
```sql
-- All key generation events in last 30 days
SELECT * FROM audit_log
WHERE action = 'key.generate' AND timestamp > NOW() - INTERVAL '30 days'
ORDER BY timestamp DESC;

-- Failed authentication attempts
SELECT source_ip, COUNT(*) as attempts
FROM audit_log
WHERE action LIKE 'POST /key/%' AND success = FALSE
GROUP BY source_ip
HAVING COUNT(*) > 10;
```

---

### 10.2 Regulatory Compliance (GDPR, HIPAA, FedRAMP)

**Data Residency:**
- ✅ All data stored on-premises (no cloud egress)
- ✅ Geographic boundary enforced by network (moat perimeter)

**Encryption Requirements:**
- ✅ TLS 1.3 for data in transit
- ✅ Postgres encryption at rest (pgcrypto or LUKS disk encryption)
- ✅ API keys hashed in database (bcrypt)

**Right to Deletion:**
```sql
-- Delete user and all associated data
DELETE FROM litellm_verificationtoken WHERE user_id = 'user-123';
DELETE FROM audit_log WHERE user_id = 'user-123';
```

**Data Retention:**
```sql
-- Automatically purge logs older than 1 year (GDPR compliance)
DELETE FROM audit_log WHERE timestamp < NOW() - INTERVAL '1 year';
```

---

## 11. Known Limitations & Future Work

### 11.1 Current Limitations

1. **No OIDC/SAML Support** - Future roadmap item (Q2 2026)
2. **Nginx Not Instrumented with OTEL** - Manual correlation required
3. **No Multi-Region Active-Active** - Single datacenter deployment only
4. **LLM Provider Must Be Reachable** - Cannot operate in complete air-gap (needs on-prem LLM)

---

### 11.2 Roadmap for Moat-Mode Enhancements

| Feature | Priority | Target |
|---------|----------|--------|
| **OIDC Integration** | High | Q2 2026 |
| **Automated Cert Rotation** | Medium | Q3 2026 |
| **Multi-DC Replication** | Low | Q4 2026 |
| **Built-in PKI/CA** | Medium | Q3 2026 |

---

## 12. Summary & Sign-Off

### 12.1 Validation Checklist

| Gate | Status | Evidence |
|------|--------|----------|
| **Gate 7 (Security)** | ✅ PASS | All auth/multi-tenancy tests passed |
| **Gate 8 (Observability)** | ✅ PASS | OTEL traces/metrics functional |
| **Gate 9 (Moat-Mode)** | 🟡 PENDING | Awaiting acceptance test execution |

**Required Actions:**
1. Execute all acceptance tests (Section 7)
2. Complete HA readiness checklist (Section 8)
3. Document deployment environment specifics
4. Obtain sign-off from security/compliance team

---

### 12.2 Approval Signatures

| Role | Name | Date | Signature |
|------|------|------|-----------|
| **Architect** | ___________ | _________ | ____________ |
| **Security Lead** | ___________ | _________ | ____________ |
| **Operations Lead** | ___________ | _________ | ____________ |
| **Compliance Officer** | ___________ | _________ | ____________ |

---

**Document Control:**
- **Next Review Date:** 2026-03-22 (90 days)
- **Change Log:** See Git history
- **Contact:** DevOps Team <devops@company.internal>

---

## Appendix A: Reference Architecture Diagram

```mermaid
graph TB
    subgraph "Moat Perimeter (No External Internet)"
        Client[External Client]

        subgraph "Edge Layer (TLS Termination)"
            LB[Nginx Load Balancer<br/>TLS 1.3]
        end

        subgraph "Application Layer"
            GW1[Gateway 1<br/>:4000]
            GW2[Gateway 2<br/>:4000]
            GW3[Gateway 3<br/>:4000]
        end

        subgraph "Data Layer"
            PG[(PostgreSQL<br/>Multi-AZ)]
            Redis[(Redis Sentinel<br/>3 nodes)]
        end

        subgraph "Observability Layer"
            OTEL[OTEL Collector<br/>:4317]
            Jaeger[Jaeger<br/>Trace Storage]
            Prom[Prometheus<br/>Metrics]
        end

        subgraph "LLM Layer (On-Prem)"
            LLM1[Bedrock via<br/>PrivateLink]
            LLM2[Self-Hosted<br/>vLLM]
        end
    end

    Client -->|HTTPS| LB
    LB -->|HTTP| GW1
    LB -->|HTTP| GW2
    LB -->|HTTP| GW3

    GW1 --> PG
    GW2 --> PG
    GW3 --> PG

    GW1 --> Redis
    GW2 --> Redis
    GW3 --> Redis

    GW1 -->|OTLP| OTEL
    GW2 -->|OTLP| OTEL
    GW3 -->|OTLP| OTEL

    OTEL --> Jaeger
    OTEL --> Prom

    GW1 -.->|HTTPS| LLM1
    GW2 -.->|HTTPS| LLM2
```

---

## Appendix B: Environment Variable Reference

| Variable | Required | Default | Moat-Mode Notes |
|----------|----------|---------|-----------------|
| `LITELLM_MASTER_KEY` | ✅ Yes | N/A | Store in Kubernetes Secret, not env file |
| `DATABASE_URL` | ✅ Yes | N/A | Use `sslmode=verify-full` for mTLS |
| `REDIS_HOST` | ✅ Yes | `redis` | Point to Sentinel VIP |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | ⚠️ Observability | N/A | Must be internal collector, not cloud |
| `CONFIG_S3_BUCKET` | ⬜ Optional | N/A | Use MinIO, not AWS S3 |
| `AWS_ENDPOINT_URL` | ⬜ If S3 used | N/A | Override to MinIO endpoint |

---

**End of Document**
