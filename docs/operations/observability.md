# Observability

RouteIQ supports OpenTelemetry for distributed tracing, metrics, and logging.

## Quick Start

```bash
docker compose -f docker-compose.otel.yml up -d
```

- Gateway: `http://localhost:4000`
- Jaeger UI: `http://localhost:16686`

## Backends

### Jaeger (Development)

```yaml
environment:
  - OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317
  - OTEL_EXPORTER_OTLP_PROTOCOL=grpc
  - OTEL_SERVICE_NAME=routeiq-gateway
```

### Grafana Tempo (Production)

```yaml
environment:
  - OTEL_EXPORTER_OTLP_ENDPOINT=http://tempo:4317
  - OTEL_SERVICE_NAME=routeiq-gateway
```

### AWS CloudWatch

```yaml
environment:
  - OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
  - OTEL_SERVICE_NAME=routeiq-gateway
```

Use the ADOT Collector to bridge OTLP to CloudWatch.

## Routing Decision Telemetry

RouteIQ emits `router.*` span attributes for every routing decision:

- `router.strategy` - Which strategy was used
- `router.selected_model` - Which model was selected
- `router.decision_latency_ms` - How long the decision took

Enabled by default via `LLMROUTER_ROUTER_CALLBACK_ENABLED=true`.

## Metrics

Prometheus metrics are exposed at `/metrics`:

- Request latency histograms
- Token usage counters
- Model selection distribution
- Error rates by provider
