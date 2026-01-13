# AWS Deployment Guide

This guide covers deploying LiteLLM + LLMRouter to AWS using various compute options.

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
│  │  │  │     ALB      │  │    │  │  ECS/EKS/     │  │                    │   │
│  │  │  │ (Internet    │──┼────┼──│  Fargate      │  │                    │   │
│  │  │  │  Facing)     │  │    │  │  Cluster      │  │                    │   │
│  │  │  └──────────────┘  │    │  └───────┬───────┘  │                    │   │
│  │  │                    │    │          │          │                    │   │
│  │  └────────────────────┘    │  ┌───────▼───────┐  │                    │   │
│  │                            │  │  ElastiCache  │  │                    │   │
│  │                            │  │    (Redis)    │  │                    │   │
│  │                            │  └───────┬───────┘  │                    │   │
│  │                            │          │          │                    │   │
│  │                            │  ┌───────▼───────┐  │                    │   │
│  │                            │  │   RDS/Aurora  │  │                    │   │
│  │                            │  │  (PostgreSQL) │  │                    │   │
│  │                            │  └───────────────┘  │                    │   │
│  │                            └─────────────────────┘                    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                     │                                        │
│  ┌──────────────────────────────────┼──────────────────────────────────┐    │
│  │                    AWS Services  │                                   │    │
│  │  ┌─────────────┐  ┌─────────────▼──┐  ┌─────────────┐               │    │
│  │  │   Amazon    │  │    Amazon      │  │  CloudWatch │               │    │
│  │  │   Bedrock   │  │      S3        │  │   X-Ray     │               │    │
│  │  │  (LLMs)     │  │ (Models/Config)│  │  (Traces)   │               │    │
│  │  └─────────────┘  └────────────────┘  └─────────────┘               │    │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

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
        "command": ["CMD-SHELL", "curl -f http://localhost:4000/health/liveliness || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

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

## Database Configuration

### RDS PostgreSQL

```bash
# Create RDS instance
aws rds create-db-instance \
  --db-instance-identifier litellm-db \
  --db-instance-class db.t3.medium \
  --engine postgres \
  --master-username litellm \
  --master-user-password <secure-password> \
  --allocated-storage 20 \
  --vpc-security-group-ids sg-xxx \
  --db-subnet-group-name my-subnet-group
```

### ElastiCache Redis

```bash
# Create ElastiCache cluster
aws elasticache create-cache-cluster \
  --cache-cluster-id litellm-redis \
  --cache-node-type cache.t3.medium \
  --engine redis \
  --num-cache-nodes 1 \
  --security-group-ids sg-xxx \
  --cache-subnet-group-name my-cache-subnet
```

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
