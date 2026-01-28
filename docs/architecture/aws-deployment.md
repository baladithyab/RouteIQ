# AWS Deployment Guide

This guide covers deploying LiteLLM + LLMRouter to AWS using various compute options.

## Component Overview

### Mandatory Components

| Component | Description | Local Testing | AWS Production |
|-----------|-------------|---------------|----------------|
| **LiteLLM Gateway** | Core API proxy and routing engine | Docker container | ECS/EKS/Fargate/Lambda |

### Optional Components

| Component | Purpose | Local Testing | AWS Production Alternative |
|-----------|---------|---------------|---------------------------|
| **PostgreSQL** | API key mgmt, spend tracking, team mgmt | Docker (postgres) | **Amazon RDS** or **Aurora PostgreSQL** |
| **Redis** | Caching, rate limiting, session store | Docker (redis) | **ElastiCache Redis/Valkey** or **MemoryDB** |
| **Object Storage** | Config files, ML models, artifacts | Docker (minio) | **Amazon S3** |
| **Tracing/OTEL** | Distributed tracing, observability | Docker (jaeger) | **AWS X-Ray**, **CloudWatch**, **Amazon Managed Grafana** |
| **MLflow** | Model experiment tracking | Docker (mlflow) | **SageMaker**, **S3 + DynamoDB**, or **self-hosted on ECS** |

> **Note**: The LiteLLM Gateway can run in a **minimal configuration** with just environment variables and no external dependencies. Add optional components as needed for your use case.

### Minimal Deployment (Gateway Only)

For simple use cases, deploy just the gateway with environment variables:

```bash
# No database, no Redis, no external storage
docker run -p 4000:4000 \
  -e LITELLM_MASTER_KEY=sk-your-key \
  -e AWS_REGION=us-east-1 \
  litellm-llmrouter:latest
```

### Full Production Deployment

For enterprise use with spend tracking, team management, and caching:

```bash
# With RDS, ElastiCache, S3, and X-Ray
docker run -p 4000:4000 \
  -e LITELLM_MASTER_KEY=sk-your-key \
  -e DATABASE_URL=postgresql://user:pass@rds-endpoint:5432/litellm \
  -e REDIS_HOST=elasticache-endpoint \
  -e CONFIG_S3_BUCKET=my-config-bucket \
  -e OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 \
  litellm-llmrouter:latest
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AWS Cloud                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        VPC (10.0.0.0/16)                              │   │
│  │  ┌────────────────────┐    ┌────────────────────┐                    │   │
│  │  │  Public Subnet(s)  │    │  Private Subnet(s)  │                    │   │
│  │  │                    │    │                     │                    │   │
│  │  │  ┌──────────────┐  │    │  ┌───────────────┐  │                    │   │
│  │  │  │     ALB      │  │    │  │  ECS/EKS/     │  │  ◄── MANDATORY    │   │
│  │  │  │ (Internet    │──┼────┼──│  Fargate      │  │                    │   │
│  │  │  │  Facing)     │  │    │  │  (Gateway)    │  │                    │   │
│  │  │  └──────────────┘  │    │  └───────┬───────┘  │                    │   │
│  │  │                    │    │          │          │                    │   │
│  │  └────────────────────┘    │  ┌───────▼───────┐  │                    │   │
│  │                            │  │  ElastiCache  │  │  ◄── OPTIONAL     │   │
│  │                            │  │ (Redis/Valkey)│  │      (Caching)    │   │
│  │                            │  └───────┬───────┘  │                    │   │
│  │                            │          │          │                    │   │
│  │                            │  ┌───────▼───────┐  │                    │   │
│  │                            │  │  RDS/Aurora   │  │  ◄── OPTIONAL     │   │
│  │                            │  │  (PostgreSQL) │  │      (Persistence)│   │
│  │                            │  └───────────────┘  │                    │   │
│  │                            └─────────────────────┘                    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                     │                                        │
│  ┌──────────────────────────────────┼──────────────────────────────────┐    │
│  │                    AWS Services  │                                   │    │
│  │  ┌─────────────┐  ┌─────────────▼──┐  ┌─────────────┐               │    │
│  │  │   Amazon    │  │    Amazon      │  │ CloudWatch  │               │    │
│  │  │   Bedrock   │  │      S3        │  │   X-Ray     │  ◄── OPTIONAL │    │
│  │  │  (LLMs)     │  │ (Config/Models)│  │ (Observ.)   │               │    │
│  │  └─────────────┘  └────────────────┘  └─────────────┘               │    │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Local Testing vs AWS Production Mapping

| Local (docker-compose) | AWS Production | Purpose |
|------------------------|----------------|---------|
| `litellm-gateway` container | ECS Fargate / EKS / App Runner | LLM Gateway (mandatory) |
| `postgres` container | Amazon RDS / Aurora PostgreSQL | API keys, spend tracking |
| `redis` container | ElastiCache Redis / Valkey / MemoryDB | Caching, rate limiting |
| `minio` container | Amazon S3 | Config files, ML models |
| `jaeger` container | AWS X-Ray + CloudWatch | Distributed tracing |
| `mlflow` container | SageMaker / Self-hosted on ECS | ML experiment tracking |

## Deployment Options

| Option | Best For | Scaling | Cost |
|--------|----------|---------|------|
| **ECS Fargate** | Simplicity, no cluster mgmt | Auto-scaling | Pay per request |
| **ECS on EC2** | Cost optimization at scale | Auto-scaling | Reserved capacity |
| **EKS** | Kubernetes expertise, multi-cloud | Fine-grained | Higher base cost |
| **App Runner** | Fastest deployment | Automatic | Pay per request |
| **Lambda** | Low traffic, cost optimization | Automatic | Pay per invocation |

---

## Option 1: ECS Fargate (Recommended)

### Prerequisites

- AWS CLI configured
- ECR repository for container images
- VPC with public and private subnets

### Step 1: Push Container to ECR

```bash
# Create ECR repository
aws ecr create-repository --repository-name litellm-llmrouter

# Build and push
docker build -t litellm-llmrouter -f docker/Dockerfile .
docker tag litellm-llmrouter:latest <account>.dkr.ecr.<region>.amazonaws.com/litellm-llmrouter:latest
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker push <account>.dkr.ecr.<region>.amazonaws.com/litellm-llmrouter:latest
```

### Step 2: Create ECS Task Definition

```json
{
  "family": "litellm-llmrouter",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::<account>:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::<account>:role/litellm-task-role",
  "containerDefinitions": [
    {
      "name": "litellm-gateway",
      "image": "<account>.dkr.ecr.<region>.amazonaws.com/litellm-llmrouter:latest",
      "portMappings": [{"containerPort": 4000, "protocol": "tcp"}],
      "environment": [
        {"name": "LITELLM_MASTER_KEY", "value": "sk-production-key"},
        {"name": "AWS_DEFAULT_REGION", "value": "us-east-1"},
        {"name": "OTEL_EXPORTER_OTLP_ENDPOINT", "value": "http://localhost:4317"},
        {"name": "CONFIG_S3_BUCKET", "value": "my-config-bucket"},
        {"name": "CONFIG_S3_KEY", "value": "config/config.yaml"}
      ],
      "secrets": [
        {"name": "DATABASE_URL", "valueFrom": "arn:aws:secretsmanager:<region>:<account>:secret:litellm/db-url"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/litellm-llmrouter",
          "awslogs-region": "<region>",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:4000/_health/live || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

> **Why use `/_health/live`?** It is an internal, unauthenticated liveness endpoint intended for orchestration health checks. LiteLLM's native `/health/*` endpoints can be auth-protected depending on configuration.

### Step 3: Create ECS Service with ALB

```bash
# Create ALB target group
aws elbv2 create-target-group \
  --name litellm-tg \
  --protocol HTTP \
  --port 4000 \


---

## Option 2: EKS (Kubernetes)

For teams with Kubernetes expertise:

### Helm Chart Values

```yaml
# values.yaml
replicaCount: 3

image:
  repository: <account>.dkr.ecr.<region>.amazonaws.com/litellm-llmrouter
  tag: latest
  pullPolicy: Always

service:
  type: ClusterIP
  port: 4000

ingress:
  enabled: true
  className: alb
  annotations:
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
  hosts:
    - host: llm-gateway.example.com
      paths:
        - path: /
          pathType: Prefix

resources:
  requests:
    cpu: 500m
    memory: 1Gi
  limits:
    cpu: 2000m
    memory: 4Gi

env:
  - name: LITELLM_MASTER_KEY
    valueFrom:
      secretKeyRef:
        name: litellm-secrets
        key: master-key
  - name: CONFIG_S3_BUCKET
    value: my-config-bucket

serviceAccount:
  create: true
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::<account>:role/litellm-eks-role

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
```

> **Health probes**: For Kubernetes, configure probes against `/_health/live` and `/_health/ready`.

---

## IAM Roles and Permissions

### Task/Pod Role Policy

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "BedrockAccess",
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": "arn:aws:bedrock:*:*:foundation-model/*"
    },
    {
      "Sid": "S3ConfigAccess",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::my-config-bucket",
        "arn:aws:s3:::my-config-bucket/*",
        "arn:aws:s3:::my-models-bucket",
        "arn:aws:s3:::my-models-bucket/*"
      ]
    },
    {
      "Sid": "CloudWatchLogs",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:log-group:/ecs/litellm-*"
    },
    {
      "Sid": "XRayTracing",
      "Effect": "Allow",
      "Action": [
        "xray:PutTraceSegments",
        "xray:PutTelemetryRecords"
      ],
      "Resource": "*"
    },
    {
      "Sid": "SecretsManager",
      "Effect": "Allow",
      "Action": ["secretsmanager:GetSecretValue"],
      "Resource": "arn:aws:secretsmanager:*:*:secret:litellm/*"
    }
  ]
}
```

---

## CloudWatch Integration

### X-Ray Tracing with ADOT Sidecar

Add AWS Distro for OpenTelemetry collector as a sidecar:

```json
{
  "name": "aws-otel-collector",
  "image": "amazon/aws-otel-collector:latest",
  "essential": true,
  "command": ["--config=/etc/otel-config.yaml"],
  "environment": [
    {"name": "AWS_REGION", "value": "<region>"}
  ],
  "logConfiguration": {
    "logDriver": "awslogs",
    "options": {
      "awslogs-group": "/ecs/otel-collector",
      "awslogs-region": "<region>",
      "awslogs-stream-prefix": "otel"
    }
  }
}
```

### OTEL Collector Config for X-Ray

```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:
    timeout: 1s
    send_batch_size: 50

exporters:
  awsxray:
    region: us-east-1

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [awsxray]
```

---

## Database Configuration (Optional)

> **Note**: Database is optional. Without it, API key management, spend tracking, and team features are disabled. The gateway still functions for basic routing.

### Option A: Amazon RDS PostgreSQL (Recommended for most use cases)

```bash
# Create RDS instance
aws rds create-db-instance \
  --db-instance-identifier litellm-db \
  --db-instance-class db.t3.medium \
  --engine postgres \
  --engine-version 15 \
  --master-username litellm \
  --master-user-password <secure-password> \
  --allocated-storage 20 \
  --storage-encrypted \
  --vpc-security-group-ids sg-xxx \
  --db-subnet-group-name my-subnet-group
```

### Option B: Aurora PostgreSQL Serverless v2 (Best for variable workloads)

```bash
# Create Aurora Serverless v2 cluster
aws rds create-db-cluster \
  --db-cluster-identifier litellm-aurora \
  --engine aurora-postgresql \
  --engine-version 15.4 \
  --master-username litellm \
  --master-user-password <secure-password> \
  --serverless-v2-scaling-configuration MinCapacity=0.5,MaxCapacity=8 \
  --storage-encrypted \
  --vpc-security-group-ids sg-xxx \
  --db-subnet-group-name my-subnet-group
```

### Database Connection String

```bash
# Store in Secrets Manager
aws secretsmanager create-secret \
  --name litellm/database-url \
  --secret-string "postgresql://litellm:<password>@<rds-endpoint>:5432/litellm"
```

---

## Caching Configuration (Optional)

> **Note**: Redis/Valkey caching is optional. Without it, response caching and distributed rate limiting are disabled.

### Option A: ElastiCache Redis

```bash
# Create ElastiCache Redis cluster
aws elasticache create-cache-cluster \
  --cache-cluster-id litellm-redis \
  --cache-node-type cache.t3.medium \
  --engine redis \
  --engine-version 7.1 \
  --num-cache-nodes 1 \
  --security-group-ids sg-xxx \
  --cache-subnet-group-name my-cache-subnet
```

### Option B: ElastiCache Valkey (Open-source Redis alternative)

```bash
# Create ElastiCache Valkey cluster
aws elasticache create-cache-cluster \
  --cache-cluster-id litellm-valkey \
  --cache-node-type cache.t3.medium \
  --engine valkey \
  --num-cache-nodes 1 \
  --security-group-ids sg-xxx \
  --cache-subnet-group-name my-cache-subnet
```

### Option C: Amazon MemoryDB (Redis-compatible with durability)

```bash
# Create MemoryDB cluster (for persistence needs)
aws memorydb create-cluster \
  --cluster-name litellm-memorydb \
  --node-type db.t4g.medium \
  --num-shards 1 \
  --security-group-ids sg-xxx \
  --subnet-group-name my-memorydb-subnet
```

### Gateway Environment Variables for Caching

```yaml
environment:
  - REDIS_HOST=<elasticache-endpoint>
  - REDIS_PORT=6379
  # For TLS-enabled clusters:
  - REDIS_SSL=true
```

---

## Object Storage Configuration (Optional)

> **Note**: S3 is optional. Use it for storing config files and for delivering routing artifacts.

### S3 Bucket for Config and Models

```bash
# Create S3 bucket
aws s3 mb s3://litellm-config-${AWS_ACCOUNT_ID}

# Upload config
aws s3 cp config/config.yaml s3://litellm-config-${AWS_ACCOUNT_ID}/config/

# Upload routing artifacts (if using ML-based routing)
aws s3 cp models/ s3://litellm-config-${AWS_ACCOUNT_ID}/models/ --recursive
```

### Gateway Environment Variables for S3

```yaml
environment:
  - CONFIG_S3_BUCKET=litellm-config-123456789
  - CONFIG_S3_KEY=config/config.yaml
  - LLMROUTER_MODEL_S3_BUCKET=litellm-config-123456789
  - LLMROUTER_MODEL_S3_KEY=models/
  - CONFIG_HOT_RELOAD=true  # Enable config sync + reload from S3
```

**Important behavior notes**:

- `CONFIG_S3_*` + `CONFIG_HOT_RELOAD=true` enables a background sync loop that checks the S3 object ETag and triggers a reload when the config file changes.
- `LLMROUTER_MODEL_S3_*` performs a **startup-time download** of routing artifacts into the container filesystem (e.g., `/app/models/`). The routing strategy then reloads based on **local file changes**.

---

## Observability Configuration (Optional)

> **Note**: Observability is optional but recommended for production. Multiple AWS-native options available.

### Option A: AWS X-Ray with ADOT Collector (Recommended)

Use AWS Distro for OpenTelemetry (ADOT) as a sidecar to send traces to X-Ray:

```yaml
# Gateway environment
environment:
  - OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
  - OTEL_SERVICE_NAME=litellm-gateway
  - OTEL_TRACES_EXPORTER=otlp
```

### Option B: CloudWatch Logs + Container Insights

For simpler observability without distributed tracing:

```json
{
  "logConfiguration": {
    "logDriver": "awslogs",
    "options": {
      "awslogs-group": "/ecs/litellm-gateway",
      "awslogs-region": "us-east-1",
      "awslogs-stream-prefix": "gateway"
    }
  }
}
```

Enable Container Insights for metrics:

```bash
aws ecs update-cluster-settings \
  --cluster litellm-cluster \
  --settings name=containerInsights,value=enabled
```

### Option C: Amazon Managed Grafana + Prometheus

For teams using Grafana:

1. Create Amazon Managed Grafana workspace
2. Deploy Prometheus as a sidecar or use Amazon Managed Prometheus
3. Configure LiteLLM to expose `/metrics` endpoint
4. Set up dashboards in Grafana

### Observability Comparison

| Option | Traces | Metrics | Logs | Cost | Complexity |
|--------|--------|---------|------|------|------------|
| X-Ray + ADOT | ✅ | ✅ | ❌ | Low | Medium |
| CloudWatch + Container Insights | ❌ | ✅ | ✅ | Low | Low |
| Managed Grafana + Prometheus | ✅ | ✅ | ✅ | Medium | High |
| Self-hosted Jaeger | ✅ | ❌ | ❌ | Compute only | High |

---

## Security Best Practices

1. **Secrets Management**: Store API keys in AWS Secrets Manager
2. **Network Isolation**: Deploy in private subnets with NAT Gateway
3. **Encryption**: Enable encryption at rest for RDS, ElastiCache, S3
4. **TLS**: Use ACM certificates with ALB for HTTPS
5. **IAM**: Use least-privilege task roles with IRSA (EKS)
6. **WAF**: Enable AWS WAF on ALB for rate limiting and protection

---

## Cost Optimization

| Component | Cost Factor | Optimization |
|-----------|-------------|--------------|
| Fargate | vCPU + Memory hours | Right-size tasks, use Spot |
| ALB | LCU hours | Combine with other services |
| RDS | Instance hours | Use Aurora Serverless v2 |
| Redis | Node hours | Use smaller nodes or Serverless |
| S3 | Storage + requests | Lifecycle policies |
| X-Ray | Traces sampled | Sample rate configuration |

---

## Monitoring & Alerting

### CloudWatch Alarms

```bash
# High latency alarm
aws cloudwatch put-metric-alarm \
  --alarm-name litellm-high-latency \
  --metric-name TargetResponseTime \
  --namespace AWS/ApplicationELB \
  --statistic Average \
  --period 60 \
  --threshold 2.0 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 3 \
  --alarm-actions arn:aws:sns:<region>:<account>:alerts

# Error rate alarm
aws cloudwatch put-metric-alarm \
  --alarm-name litellm-errors \
  --metric-name HTTPCode_Target_5XX_Count \
  --namespace AWS/ApplicationELB \
  --statistic Sum \
  --period 60 \
  --threshold 10 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2 \
  --alarm-actions arn:aws:sns:<region>:<account>:alerts
```

---

## Next Steps

- [High Availability Setup](../high-availability.md)
- [Observability Guide](../observability.md)
- [Hot Reloading Configuration](../hot-reloading.md)
