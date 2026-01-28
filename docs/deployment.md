# Deployment Guide

RouteIQ Gateway is designed to be cloud-native and deployment-agnostic. This guide covers deployment using Docker, Docker Compose, and Kubernetes.

## Docker

The easiest way to run RouteIQ Gateway is using the official Docker image.

```bash
docker run -p 4000:4000 \
  -e OPENAI_API_KEY="sk-..." \
  ghcr.io/baladithyab/litellm-llm-router:latest
```

## Docker Compose

We provide several Docker Compose configurations for different use cases.

### Standard (Development)
For local development and testing.

```bash
docker compose up -d
```

### High Availability (HA)
Includes Redis for caching/rate-limiting and PostgreSQL for data persistence.

```bash
docker compose -f docker-compose.ha.yml up -d
```

### Observability (OTEL)
Includes Jaeger and Prometheus for tracing and metrics.

```bash
docker compose -f docker-compose.otel.yml up -d
```

## Kubernetes

RouteIQ Gateway is Kubernetes-ready. Below is a blueprint for a production deployment.

### Deployment Blueprint

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: routeiq-gateway
spec:
  replicas: 3
  selector:
    matchLabels:
      app: routeiq-gateway
  template:
    metadata:
      labels:
        app: routeiq-gateway
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 2000
      containers:
        - name: gateway
          image: ghcr.io/baladithyab/litellm-llm-router:latest
          ports:
            - containerPort: 4000
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: routeiq-secrets
                  key: database-url
            - name: REDIS_HOST
              value: "redis-master"
          volumeMounts:
            - name: config-volume
              mountPath: /app/config
          readinessProbe:
            httpGet:
              path: /health/ready
              port: 4000
            initialDelaySeconds: 5
            periodSeconds: 10
      volumes:
        - name: config-volume
          configMap:
            name: routeiq-config
```

### Helm Chart Values
(Coming soon)

## Configuration Management

RouteIQ Gateway supports multiple configuration sources:

1.  **Local Files**: Mount `config.yaml` to `/app/config/config.yaml`.
2.  **Environment Variables**: Override settings using `LITELLM_*` env vars.
3.  **S3**: Load configuration and models from an S3 bucket.

```bash
# Enable S3 config loading
export CONFIG_SOURCE="s3"
export S3_BUCKET_NAME="my-config-bucket"
```

## Cloud Deployment

For specific cloud provider guides, see:

- [AWS Deployment Guide](architecture/aws-deployment.md)
