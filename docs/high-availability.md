# High Availability Setup

This guide covers deploying the LiteLLM + LLMRouter gateway in a high-availability configuration.

## Architecture Overview

```
                    ┌─────────────┐
                    │   Nginx     │
                    │   (LB)      │
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              │                         │
       ┌──────▼──────┐          ┌──────▼──────┐
       │  Gateway 1  │          │  Gateway 2  │
       └──────┬──────┘          └──────┬──────┘
              │                         │
              └────────────┬────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
  ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐
  │  PostgreSQL │   │    Redis    │   │     S3      │
  │  (State)    │   │   (Cache)   │   │  (Config)   │
  └─────────────┘   └─────────────┘   └─────────────┘
```

## Quick Start

```bash
docker compose -f docker-compose.ha.yml up -d
```

## Components

### PostgreSQL
Persistent storage for:
- User/team configurations
- API keys and budgets
- Request logs

### Redis
Distributed caching for:
- Response caching
- Rate limiting state
- Session data

### S3 (Optional)
Configuration sync:
- Config files
- Routing models
- Custom routers

## Configuration

### Environment Variables

```bash
# PostgreSQL
POSTGRES_USER=litellm
POSTGRES_PASSWORD=your_secure_password
POSTGRES_DB=litellm

# Redis
REDIS_HOST=redis
REDIS_PORT=6379

# S3 Config Sync
CONFIG_S3_BUCKET=my-config-bucket
CONFIG_S3_KEY=configs/config.yaml
LLMROUTER_MODEL_S3_BUCKET=my-models-bucket
LLMROUTER_MODEL_S3_KEY=models/router.pt
```

### Nginx Configuration

Load balancing configuration in `config/nginx.conf`:

```nginx
upstream litellm_backend {
    least_conn;
    server litellm-gateway-1:4000;
    server litellm-gateway-2:4000 backup;
}
```

## Scaling

### Horizontal Scaling

Add more gateway instances in `docker-compose.ha.yml`:

```yaml
litellm-gateway-3:
  image: ghcr.io/baladithyab/litellm-llm-router:latest
  # ... same config as gateway-1
```

Update Nginx upstream:

```nginx
upstream litellm_backend {
    server litellm-gateway-1:4000;
    server litellm-gateway-2:4000;
    server litellm-gateway-3:4000;
}
```

### Kubernetes Deployment

For Kubernetes deployments, use a Deployment with HorizontalPodAutoscaler.

#### Sample Deployment Manifest

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: litellm-llmrouter
  labels:
    app: litellm-gateway
spec:
  replicas: 3
  selector:
    matchLabels:
      app: litellm-gateway
  template:
    metadata:
      labels:
        app: litellm-gateway
    spec:
      # Security context for the pod
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      
      containers:
      - name: gateway
        image: ghcr.io/baladithyab/litellm-llm-router:latest
        ports:
        - containerPort: 4000
          name: http
        
        # Environment variables from ConfigMap and Secrets
        envFrom:
        - configMapRef:
            name: litellm-config
        - secretRef:
            name: litellm-secrets
        
        # Additional environment variables
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: litellm-secrets
              key: database-url
        - name: STORE_MODEL_IN_DB
          value: "true"
        # Do NOT run migrations on every replica
        - name: LITELLM_RUN_DB_MIGRATIONS
          value: "false"
        
        # Health probes using internal unauthenticated endpoints
        livenessProbe:
          httpGet:
            path: /_health/live
            port: 4000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        
        readinessProbe:
          httpGet:
            path: /_health/ready
            port: 4000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 5
          failureThreshold: 3
        
        # Resource limits
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        
        # Security context for the container
        securityContext:
          readOnlyRootFilesystem: true
          allowPrivilegeEscalation: false
        
        # Volume mounts for read-only filesystem
        volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: data
          mountPath: /app/data
      
      volumes:
      - name: tmp
        emptyDir: {}
      - name: data
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: litellm-gateway
spec:
  selector:
    app: litellm-gateway
  ports:
  - port: 4000
    targetPort: 4000
    name: http
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: litellm-gateway-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: litellm-llmrouter
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

#### Database Migration Job

Run migrations separately, not on every replica:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: litellm-db-migrate
  annotations:
    # Run before deployment with Argo CD hooks or Helm hooks
    helm.sh/hook: pre-install,pre-upgrade
    helm.sh/hook-weight: "-1"
spec:
  template:
    spec:
      containers:
      - name: migrate
        image: ghcr.io/baladithyab/litellm-llm-router:latest
        command: ["/bin/bash", "-c"]
        args:
          - |
            SCHEMA_PATH=$(python -c "import litellm; import os; print(os.path.join(os.path.dirname(litellm.__file__), 'proxy', 'schema.prisma'))")
            if [ -f "$SCHEMA_PATH" ]; then
              echo "Running migrations from $SCHEMA_PATH"
              prisma migrate deploy --schema="$SCHEMA_PATH"
            else
              echo "Schema not found, skipping migrations"
            fi
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: litellm-secrets
              key: database-url
      restartPolicy: Never
  backoffLimit: 3
```

## Health Checks

All gateways expose health endpoints:

### Internal Endpoints (Recommended for K8s)

These endpoints are unauthenticated and optimized for Kubernetes probes:

| Endpoint | Purpose | Checks | Auth |
|----------|---------|--------|------|
| `GET /_health/live` | Liveness probe | Process is alive | None |
| `GET /_health/ready` | Readiness probe | DB, Redis (if configured) | None |

**Liveness (`/_health/live`):**
- Returns 200 immediately if process is alive
- Does NOT check external dependencies
- Failure triggers pod restart

**Readiness (`/_health/ready`):**
- Returns 200 if all configured dependencies are healthy
- Checks database connection with 2s timeout
- Checks Redis connection with 2s timeout
- Checks MCP gateway status (if enabled)
- Returns 503 if any check fails
- Failure removes pod from Service endpoints

### LiteLLM Native Endpoints

| Endpoint | Purpose | Auth |
|----------|---------|------|
| `GET /health` | Basic health check | Depends on config |
| `GET /health/liveliness` | Kubernetes liveness | Depends on config |
| `GET /health/readiness` | Kubernetes readiness | Depends on config |

**Note:** LiteLLM's native endpoints may be auth-protected depending on your configuration. Use the `/_health/*` endpoints for unauthenticated K8s probes.

### Example Probe Configuration

```yaml
livenessProbe:
  httpGet:
    path: /_health/live
    port: 4000
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /_health/ready
    port: 4000
  initialDelaySeconds: 10
  periodSeconds: 5
  timeoutSeconds: 5
  failureThreshold: 3

startupProbe:
  httpGet:
    path: /_health/live
    port: 4000
  initialDelaySeconds: 10
  periodSeconds: 5
  timeoutSeconds: 5
  failureThreshold: 30  # 30 * 5s = 150s max startup time
```

## Monitoring

Enable Prometheus metrics:

```yaml
litellm_settings:
  success_callback: ["prometheus"]
```

Access metrics at `GET /metrics`.

## External Dependencies for HA

For a fully HA Kubernetes deployment, you need:

| Component | Purpose | Managed Options |
|-----------|---------|-----------------|
| PostgreSQL | State persistence | RDS, Cloud SQL, Aurora |
| Redis | Caching, rate limiting | ElastiCache, MemoryStore |
| S3/GCS | Config sync | S3, GCS, MinIO |
| OTEL Collector | Observability | AWS Distro, GCP Ops Agent |

**Network Policies:** Ensure egress is allowed to:
- External LLM providers (OpenAI, Anthropic, Azure, etc.)
- PostgreSQL and Redis services
- S3/GCS endpoints
- MCP server URLs (if `MCP_GATEWAY_ENABLED=true`)
- A2A agent URLs (if `A2A_GATEWAY_ENABLED=true`)

See [Configuration Guide](configuration.md#network-policy-considerations) for a sample NetworkPolicy.
