# Observability Guide

LiteLLM + LLMRouter supports OpenTelemetry (OTEL) for distributed tracing. This guide covers three backends:

1. **Jaeger** - Simple local setup
2. **Grafana Tempo** - Production-grade with S3 storage
3. **AWS CloudWatch** - Native AWS integration

---

## Quick Start: Jaeger (Recommended for Development)

```bash
docker compose -f docker-compose.otel.yml up -d
```

- **Gateway**: http://localhost:4000
- **Jaeger UI**: http://localhost:16686

Make a request and view traces:
```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-dev-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-haiku", "messages": [{"role": "user", "content": "Hello"}]}'
```

Then open Jaeger UI → Select "litellm-gateway" service → Find Traces

---

## Option 1: Jaeger

Best for: Local development, debugging, simple deployments

### Environment Variables

```yaml
environment:
  - OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317
  - OTEL_EXPORTER_OTLP_PROTOCOL=grpc
  - OTEL_SERVICE_NAME=litellm-gateway
  - OTEL_TRACES_EXPORTER=otlp
```

### Standalone Jaeger

```bash
docker run -d --name jaeger \
  -p 16686:16686 \
  -p 4317:4317 \
  -p 4318:4318 \
  -e COLLECTOR_OTLP_ENABLED=true \
  jaegertracing/all-in-one:1.54
```

---

## Option 2: Grafana Tempo

Best for: Production, S3 storage, Grafana ecosystem integration

### Environment Variables

```yaml
environment:
  - OTEL_EXPORTER_OTLP_ENDPOINT=http://tempo:4317
  - OTEL_EXPORTER_OTLP_PROTOCOL=grpc
  - OTEL_SERVICE_NAME=litellm-gateway
  - OTEL_TRACES_EXPORTER=otlp
```

### Tempo with S3 Backend

Create `tempo-config.yaml`:
```yaml
server:
  http_listen_port: 3200

distributor:
  receivers:
    otlp:
      protocols:
        grpc:
          endpoint: 0.0.0.0:4317
        http:
          endpoint: 0.0.0.0:4318

storage:
  trace:
    backend: s3
    s3:
      bucket: your-tempo-bucket
      endpoint: s3.us-east-1.amazonaws.com
      region: us-east-1
      # Uses IAM role - no credentials needed on EC2
```

```bash
docker run -d --name tempo \
  -p 3200:3200 \
  -p 4317:4317 \
  -v ./tempo-config.yaml:/etc/tempo.yaml \
  grafana/tempo:latest \
  -config.file=/etc/tempo.yaml
```

### Grafana Dashboard

Add Tempo as a data source in Grafana:
- URL: `http://tempo:3200`
- Enable TraceQL for querying

---

## Option 3: AWS CloudWatch (X-Ray)

Best for: AWS-native, no additional infrastructure, IAM-based auth

### Using AWS Distro for OpenTelemetry (ADOT)

The ADOT Collector receives OTLP traces and exports to CloudWatch X-Ray.

#### Run ADOT Collector as Sidecar

```yaml
# docker-compose.cloudwatch.yml
services:
  litellm-gateway:
    # ... your gateway config ...
    environment:
      - OTEL_EXPORTER_OTLP_ENDPOINT=http://adot-collector:4317
      - OTEL_SERVICE_NAME=litellm-gateway
      - OTEL_TRACES_EXPORTER=otlp

  adot-collector:
    image: amazon/aws-otel-collector:latest
    command: ["--config=/etc/otel-config.yaml"]
    volumes:
      - ./config/otel-collector-config.yaml:/etc/otel-config.yaml:ro
    # Uses IAM Instance Profile on EC2 - no credentials needed
```

#### ADOT Collector Config (`config/otel-collector-config.yaml`)

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
    # No credentials needed on EC2 with IAM role

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [awsxray]
```

#### Required IAM Permissions

Attach this policy to your EC2 instance role:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "xray:PutTraceSegments",
        "xray:PutTelemetryRecords",
        "xray:GetSamplingRules",
        "xray:GetSamplingTargets"
      ],
      "Resource": "*"
    }
  ]
}
```

#### View Traces

1. Open AWS Console → CloudWatch → X-Ray traces
2. Filter by service name: `litellm-gateway`
3. Use Service Map for dependency visualization

---

## Trace Attributes

The gateway automatically adds these attributes to spans:

| Attribute | Description |
|-----------|-------------|
| `llm.model` | Target model name |
| `llm.provider` | Provider (bedrock, openai, etc.) |
| `llm.tokens.prompt` | Input token count |
| `llm.tokens.completion` | Output token count |
| `http.status_code` | Response status |

---

## Disabling Tracing

To disable OTEL tracing entirely:

```yaml
environment:
  - OTEL_TRACES_EXPORTER=none
  - OTEL_METRICS_EXPORTER=none
  - OTEL_LOGS_EXPORTER=none
```
