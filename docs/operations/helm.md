# Helm Chart

RouteIQ provides Helm charts for Kubernetes deployment.

## Installation

```bash
helm install routeiq deploy/charts/routeiq \
  --namespace routeiq \
  --create-namespace \
  --set config.masterKey=sk-your-key \
  --set config.openaiApiKey=sk-...
```

## Configuration

Key values:

```yaml
# values.yaml
replicaCount: 3

image:
  repository: routeiq
  tag: latest

config:
  masterKey: ""  # Required
  configPath: /app/config/config.yaml

resources:
  requests:
    memory: 512Mi
    cpu: 250m
  limits:
    memory: 1Gi
    cpu: 1000m

redis:
  enabled: true

postgresql:
  enabled: false

ingress:
  enabled: true
  className: nginx
  hosts:
    - host: routeiq.example.com
      paths:
        - path: /
          pathType: Prefix
```

## Health Probes

```yaml
livenessProbe:
  httpGet:
    path: /_health/live
    port: 4000

readinessProbe:
  httpGet:
    path: /_health/ready
    port: 4000
```

## Scaling

```yaml
autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
```

## Deployment Topology

The following diagram shows a production Kubernetes deployment with HPA,
backed by PostgreSQL, Redis, an OTel Collector, and an OIDC provider.
Leader election ensures only one pod runs config sync and migrations.

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
