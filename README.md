# RouteIQ Gateway

> **Attribution**:
> RouteIQ is built on top of upstream [LiteLLM](https://github.com/BerriAI/litellm) for proxy/API compatibility and [LLMRouter](https://github.com/ulab-uiuc/LLMRouter) for ML routing.

Cloud-native General AI Gateway with pluggable ML routing intelligence and end-to-end MLOps pipeline

<div align="center">

  **A production-grade AI gateway with ML-based routing, multi-protocol support, and enterprise hardening**

  [![Docker Build](https://github.com/baladithyab/RouteIQ/actions/workflows/docker-build.yml/badge.svg)](https://github.com/baladithyab/RouteIQ/actions/workflows/docker-build.yml)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
</div>

## Overview

**RouteIQ Gateway** is a production-grade, cloud-native **General AI Gateway** that extends [LiteLLM Proxy](https://github.com/BerriAI/litellm) with ML-based routing intelligence. It serves as a unified control plane for all AI interactions -- LLMs, Agents (A2A), Tools (MCP), and Skills -- while adding:

- **Intelligent Routing**: 18+ ML strategies (KNN, MLP, SVM, ELO, hybrid, etc.) that learn from your traffic
- **MLOps Pipeline**: End-to-end tooling to collect telemetry, train routing models, and deploy them with hot-reload
- **Enterprise Hardening**: RBAC, quotas, audit logging, policy engine, SSRF protection, circuit breakers
- **Cloud-Native**: Helm charts, HA with leader election, graceful shutdown, health probes, OpenTelemetry

## Gateway Surfaces

RouteIQ unifies multiple AI interaction patterns under a single endpoint:

| Surface | Status | Description |
|---------|--------|-------------|
| **LLM Proxy** | Stable | OpenAI-compatible chat/completions for 100+ providers |
| **Observability** | Stable | OpenTelemetry tracing, metrics, and structured logging |
| **Security** | Stable | SSRF protection, admin auth, RBAC, policy engine, audit logging |
| **Resilience** | Stable | Backpressure, circuit breakers, graceful drain, connection pooling |
| **A2A Gateway** | Beta | Agent-to-Agent protocol for multi-agent orchestration |
| **MCP Gateway** | Beta | Model Context Protocol (JSON-RPC, SSE, REST surfaces) |
| **Skills** | Beta | Anthropic Computer Use, Bash, and Text Editor skills |
| **Plugin System** | Beta | Extensible plugin architecture with lifecycle management |

## Architecture

The gateway operates as the central nervous system for AI infrastructure:

```
                    ┌─────────────────────────────────────────┐
   API Traffic ───> │          RouteIQ Gateway                │
                    │  ┌───────────┐  ┌────────────────────┐  │
                    │  │ LiteLLM   │  │ Routing Intelligence│  │
                    │  │ Proxy     │──│ Layer (ML models)   │  │──> LLM Providers
                    │  └───────────┘  └────────────────────┘  │
                    │  ┌──────┐ ┌─────┐ ┌──────┐ ┌────────┐  │
                    │  │ MCP  │ │ A2A │ │Skills│ │Plugins │  │
                    │  └──────┘ └─────┘ └──────┘ └────────┘  │
                    └───────────────┬──────────────────────────┘
                                    │ OpenTelemetry
                                    v
                    ┌───────────────────────────────────────┐
                    │  MLOps Pipeline                       │
                    │  Traces → Train → Deploy → Hot-Reload │
                    └───────────────────────────────────────┘
```

### Core Loop

1. **Route**: Incoming requests are analyzed by the ML routing layer and sent to the optimal model
2. **Observe**: Execution data (latency, cost, quality) is captured via OpenTelemetry
3. **Learn**: The MLOps pipeline uses telemetry data to train improved routing models
4. **Update**: New models are hot-reloaded into the gateway without restarts

## Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose
- [uv](https://docs.astral.sh/uv/) (for local development)

### 1. Docker Compose (Basic)

```bash
git clone https://github.com/baladithyab/RouteIQ.git
cd RouteIQ
cp .env.example .env  # Edit with your API keys
docker compose up -d
```

### 2. High Availability (Production)

Multi-replica with Redis, PostgreSQL, and Nginx load balancing:

```bash
docker compose -f docker-compose.ha.yml up -d
```

### 3. With Observability (OTel + Jaeger)

Full trace visualization with Jaeger:

```bash
docker compose -f docker-compose.otel.yml up -d
# Jaeger UI: http://localhost:16686
```

### 4. Local Development

```bash
uv sync
docker compose -f docker-compose.local-test.yml up -d
uv run python -m litellm_llmrouter.startup --config config/config.local-test.yaml --port 4000
```

## Deployment

RouteIQ is designed for cloud-native deployment:

- **Docker**: Multi-stage production images on GHCR (non-root, read-only filesystem)
- **Kubernetes**: Helm chart with HPA, PDB, NetworkPolicy, IRSA support (`deploy/charts/`)
- **Docker Compose**: Variants for dev, HA, observability, testing
- **Health Probes**: `/_health/live` (liveness) and `/_health/ready` (readiness with dependency checks)
- **Config Management**: YAML files, S3/GCS sync with ETag change detection, hot-reload

See the [Deployment Guide](docs/deployment.md) for production checklist.

## Routing Strategies

### LiteLLM Built-in

| Strategy | Description |
|----------|-------------|
| `simple-shuffle` | Random load balancing (default) |
| `least-busy` | Route to model with fewest active requests |
| `latency-based-routing` | Route based on historical latency |
| `cost-based-routing` | Route to minimize cost |
| `usage-based-routing` | Route based on token usage |

### RouteIQ ML Strategies (18+)

| Strategy | Algorithm |
|----------|-----------|
| `llmrouter-knn` | K-Nearest Neighbors (embedding similarity) |
| `llmrouter-mlp` | Multi-Layer Perceptron neural network |
| `llmrouter-svm` | Support Vector Machine |
| `llmrouter-mf` | Matrix Factorization |
| `llmrouter-elo` | Elo Rating |
| `llmrouter-hybrid` | Probabilistic hybrid |
| `llmrouter-routerdc` | Dual Contrastive (BERT-based) |
| `llmrouter-causallm` | Transformer-based (GPT-2) |
| `llmrouter-graph` | Graph neural network |
| `llmrouter-automix` | Automatic model mixing |
| `llmrouter-smallest` / `llmrouter-largest` | Baseline strategies |
| `llmrouter-custom` | User-defined routing logic |

Strategies support **A/B testing** with deterministic hash-based assignment and **hot-reload** for zero-downtime model updates.

See [Routing Strategies](docs/routing-strategies.md) for details.

## Configuration

```yaml
model_list:
  - model_name: claude-sonnet
    litellm_params:
      model: anthropic/claude-3-sonnet-20240229
      api_key: os.environ/ANTHROPIC_API_KEY

  - model_name: claude-haiku
    litellm_params:
      model: anthropic/claude-3-haiku-20240307
      api_key: os.environ/ANTHROPIC_API_KEY

router_settings:
  routing_strategy: llmrouter-knn
  routing_strategy_args:
    model_path: /app/models/knn_router
    llm_data_path: /app/config/llm_candidates.json
    hot_reload: true
    reload_interval: 300

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
```

See [Configuration Guide](docs/configuration.md) for all options.

## Security

Security is a first-class concern:

- **SSRF Protection**: Deny-by-default for private IPs with allowlist support
- **Artifact Safety**: Pickle loading disabled by default; optional manifest verification with Ed25519/HMAC signatures
- **Admin Auth**: Fail-closed design (no keys configured = deny all control-plane requests)
- **RBAC**: Role-based access control for API endpoints
- **Policy Engine**: OPA-style pre-request policy evaluation
- **Quotas**: Per-team/per-key rate limiting and budget enforcement
- **Audit Logging**: Structured audit trail with fail-closed mode
- **Secret Scrubbing**: Automatic redaction of secrets in error logs

See [Security Guide](docs/security.md) for details.

## Documentation

| Document | Description |
|----------|-------------|
| [Getting Started](docs/index.md) | Setup and usage guide |
| [Deployment Guide](docs/deployment.md) | Docker, K8s, and cloud deployment |
| [Configuration](docs/configuration.md) | Config options and hot-reloading |
| [API Reference](docs/api-reference.md) | API endpoint documentation |
| [Routing Strategies](docs/routing-strategies.md) | ML routing strategy guide |
| [MLOps Training](docs/mlops-training.md) | Training pipeline guide |
| [MCP Gateway](docs/mcp-gateway.md) | Model Context Protocol |
| [A2A Gateway](docs/a2a-gateway.md) | Agent-to-Agent protocol |
| [Plugins](docs/plugins.md) | Plugin system guide |
| [Security](docs/security.md) | Security considerations |
| [Observability](docs/observability.md) | OpenTelemetry setup |
| [High Availability](docs/high-availability.md) | HA configuration |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding guidelines, and PR process.

For AI agent instructions, see [AGENTS.md](AGENTS.md) and [CLAUDE.md](CLAUDE.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LiteLLM](https://github.com/BerriAI/litellm) for the proxy and provider translation layer
- [LLMRouter](https://github.com/ulab-uiuc/LLMRouter) for ML routing strategy research and implementations
