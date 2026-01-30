# Quickstart: Observability with OpenTelemetry

This guide sets up RouteIQ Gateway with a full OpenTelemetry (OTel) stack using Jaeger for tracing. This allows you to visualize request flows, routing decisions, and latency breakdowns.

## Architecture

- **RouteIQ Gateway**: Instrumented to emit OTel traces and metrics.
- **Jaeger**: All-in-one backend for collecting and visualizing traces.
- **Redis**: Optional, for caching (included in the compose file).

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/).

## 1. Start the Stack

First, ensure your environment is configured:

```bash
cp .env.example .env
# Edit .env to set LITELLM_MASTER_KEY and provider keys
```

Then start the observability stack using the `docker-compose.otel.yml` file:

```bash
docker-compose -f docker-compose.otel.yml up -d
```

## 2. Access Interfaces

- **Gateway**: `http://localhost:4001`
- **Jaeger UI**: `http://localhost:16686`

## 3. Generate Traces

Make some requests to the gateway to generate trace data:

```bash
# Simple chat completion
curl -X POST http://localhost:4001/chat/completions \
  -H "Authorization: Bearer sk-dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello OTel!"}]
  }'
```

## 4. Visualize in Jaeger

1.  Open [http://localhost:16686](http://localhost:16686).
2.  Select Service: `litellm-gateway`.
3.  Click **Find Traces**.
4.  Click on a trace to see the full breakdown, including:
    - **`llm.routing.decision`**: Which model was selected and why.
    - **`cache.lookup`**: Cache hits/misses.
    - **`litellm.proxy`**: Internal proxy operations.

## Configuration Knobs

You can fine-tune observability using environment variables:

### Sampling Control

Control how many traces are recorded to manage volume and cost.

| Variable | Description | Example |
|----------|-------------|---------|
| `OTEL_TRACES_SAMPLER` | Standard OTel sampler type. | `traceidratio`, `always_on`, `always_off` |
| `OTEL_TRACES_SAMPLER_ARG` | Argument for the sampler (e.g., ratio). | `0.1` (10%), `1.0` (100%) |
| `LLMROUTER_OTEL_SAMPLE_RATE` | **Convenience**. Simple ratio (0.0-1.0). | `0.05` (5%) |

**Example: Sample 10% of requests**
```bash
export LLMROUTER_OTEL_SAMPLE_RATE=0.1
```

### OTLP Exporter

Point the gateway to your OTel collector (e.g., Jaeger, Honeycomb, Datadog).

| Variable | Description | Default |
|----------|-------------|---------|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP gRPC/HTTP endpoint. | `http://localhost:4317` |
| `OTEL_SERVICE_NAME` | Service name in traces. | `litellm-gateway` |

## Next Steps

- **Production Setup:** [High Availability Guide](quickstart-ha-compose.md).
- **MLOps:** [MLOps Training Pipeline](mlops-training.md).
- **Deep Dive:** [Observability Guide](observability.md).
