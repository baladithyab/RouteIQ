# Observability

RouteIQ supports OpenTelemetry (OTEL) for distributed tracing, metrics, and logging.

## Quick Start

```bash
docker compose -f docker-compose.otel.yml up -d
```

- **Gateway**: `http://localhost:4000`
- **Jaeger UI**: `http://localhost:16686`

Make a request and view traces:

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-dev-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-haiku", "messages": [{"role": "user", "content": "Hello"}]}'
```

Then open Jaeger UI → Select "routeiq-gateway" service → Find Traces.

## Backends

### Jaeger (Development)

Best for: Local development, debugging, simple deployments.

```yaml
environment:
  - OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317
  - OTEL_EXPORTER_OTLP_PROTOCOL=grpc
  - OTEL_SERVICE_NAME=routeiq-gateway
  - OTEL_TRACES_EXPORTER=otlp
```

Standalone:

```bash
docker run -d --name jaeger \
  -p 16686:16686 -p 4317:4317 -p 4318:4318 \
  -e COLLECTOR_OTLP_ENABLED=true \
  jaegertracing/all-in-one:1.54
```

### Grafana Tempo (Production)

Best for: Production, S3 storage, Grafana ecosystem integration.

```yaml
environment:
  - OTEL_EXPORTER_OTLP_ENDPOINT=http://tempo:4317
  - OTEL_EXPORTER_OTLP_PROTOCOL=grpc
  - OTEL_SERVICE_NAME=routeiq-gateway
  - OTEL_TRACES_EXPORTER=otlp
```

See the Grafana documentation for Tempo configuration with S3 backend.

### AWS CloudWatch (X-Ray)

Best for: AWS-native, no additional infrastructure, IAM-based auth.

Use the AWS Distro for OpenTelemetry (ADOT) Collector as a sidecar:

```yaml
environment:
  - OTEL_EXPORTER_OTLP_ENDPOINT=http://adot-collector:4317
  - OTEL_SERVICE_NAME=routeiq-gateway
  - OTEL_TRACES_EXPORTER=otlp
```

Required IAM permissions: `xray:PutTraceSegments`, `xray:PutTelemetryRecords`,
`xray:GetSamplingRules`, `xray:GetSamplingTargets`.

## Routing Decision Telemetry

RouteIQ emits `router.*` span attributes for every routing decision:

- `router.strategy` — Which strategy was used
- `router.selected_model` — Which model was selected
- `router.decision_latency_ms` — How long the decision took

Enabled by default via `LLMROUTER_ROUTER_CALLBACK_ENABLED=true`.

## Trace Attributes

| Attribute | Description |
|-----------|-------------|
| `llm.model` | Target model name |
| `llm.provider` | Provider (bedrock, openai, etc.) |
| `llm.tokens.prompt` | Input token count |
| `llm.tokens.completion` | Output token count |
| `http.status_code` | Response status |

## Exporter Configuration

### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4317` | OTLP collector endpoint |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `grpc` | Protocol: `grpc`, `http/protobuf`, `http/json` |
| `OTEL_EXPORTER_OTLP_HEADERS` | — | Headers for auth (format: `key1=value1,key2=value2`) |
| `OTEL_EXPORTER_OTLP_COMPRESSION` | — | Compression: `gzip` or `none` |

### Service Identity

| Variable | Default | Description |
|----------|---------|-------------|
| `OTEL_SERVICE_NAME` | `routeiq-gateway` | Service name |
| `OTEL_SERVICE_VERSION` | `1.0.0` | Service version |
| `OTEL_DEPLOYMENT_ENVIRONMENT` | `production` | Deployment environment |

### Exporter Selection

| Variable | Default | Description |
|----------|---------|-------------|
| `OTEL_TRACES_EXPORTER` | `otlp` | `otlp`, `jaeger`, `zipkin`, or `none` |
| `OTEL_METRICS_EXPORTER` | `otlp` | `otlp`, `prometheus`, or `none` |
| `OTEL_LOGS_EXPORTER` | `otlp` | `otlp` or `none` |

### Full Production Example

```yaml
environment:
  - OTEL_EXPORTER_OTLP_ENDPOINT=https://otel-collector.internal:4317
  - OTEL_EXPORTER_OTLP_PROTOCOL=grpc
  - OTEL_EXPORTER_OTLP_COMPRESSION=gzip
  - OTEL_EXPORTER_OTLP_HEADERS=x-honeycomb-team=your-api-key
  - OTEL_SERVICE_NAME=routeiq-gateway
  - OTEL_SERVICE_VERSION=2.1.0
  - OTEL_DEPLOYMENT_ENVIRONMENT=production
  - ROUTEIQ_OTEL_TRACES_SAMPLER=parentbased_traceidratio
  - ROUTEIQ_OTEL_TRACES_SAMPLER_ARG=0.1
```

## Trace Sampling

In production, sampling all traces can be expensive. RouteIQ defaults to **10%**
using `parentbased_traceidratio`.

### Sampling Options

| Variable | Default | Description |
|----------|---------|-------------|
| `ROUTEIQ_OTEL_TRACES_SAMPLER` | `parentbased_traceidratio` | Sampler type |
| `ROUTEIQ_OTEL_TRACES_SAMPLER_ARG` | `0.1` | Ratio (0.0-1.0) |

**Sampler types:**

| Sampler | Description |
|---------|-------------|
| `always_on` | Sample 100% |
| `always_off` | Sample 0% |
| `traceidratio` | Sample based on trace ID ratio |
| `parentbased_always_on` | Respect parent, default to always_on |
| `parentbased_traceidratio` | Respect parent, default to ratio (recommended) |

**Priority order**: `OTEL_TRACES_SAMPLER` > `ROUTEIQ_OTEL_TRACES_SAMPLER` >
`LLMROUTER_OTEL_SAMPLE_RATE` (deprecated) > default.

### Best Practices

- **Development**: `always_on` (100%)
- **Staging**: 50-100%
- **Production**: 1-10% depending on traffic
- Use ParentBased samplers to preserve distributed trace context

## Metrics

Prometheus metrics exposed at `/metrics`:

- Request latency histograms
- Token usage counters
- Model selection distribution
- Error rates by provider

## Multiprocess Metrics

When running with multiple workers, use an OTEL Collector for metric aggregation:

```
Worker 1 ──►
Worker 2 ──► OTEL Collector ──► Prometheus/Jaeger
Worker N ──►
```

Alternative: Set `PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_multiproc` with a
tmpfs mount for Prometheus pull-based metrics.

## Log Level Configuration

```yaml
environment:
  - LOG_LEVEL=INFO          # DEBUG, INFO, WARNING, ERROR, CRITICAL
  - LOG_QUERIES=false       # Enable query logging (disabled for privacy)
```

## Disabling Tracing

```yaml
environment:
  - OTEL_TRACES_EXPORTER=none
  - OTEL_METRICS_EXPORTER=none
  - OTEL_LOGS_EXPORTER=none
```

## Troubleshooting

- **No traces appearing**: Verify endpoint is reachable, exporter is `otlp` not `none`, sampling isn't `always_off`
- **Duplicate metrics**: Use OTEL Collector for aggregation, or set `PROMETHEUS_MULTIPROC_DIR`
- **High trace volume**: Enable sampling with `ROUTEIQ_OTEL_TRACES_SAMPLER_ARG=0.1`
- **TLS issues**: Verify `_CERTIFICATE`, `_CLIENT_KEY` env vars are correct
