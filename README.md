# RouteIQ Gateway
Cloud-native General AI Gateway (powered by LiteLLM) with pluggable routing intelligence + scripted MLOps pipeline

<div align="center">

  **A cloud-grade AI gateway with pluggable routing and end-to-end MLOps tooling**

  [![Docker Build](https://github.com/baladithyab/litellm-llm-router/actions/workflows/docker-build.yml/badge.svg)](https://github.com/baladithyab/litellm-llm-router/actions/workflows/docker-build.yml)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>

## Overview

**RouteIQ Gateway** is a production-grade, cloud-native **General AI Gateway** that extends the capabilities of the upstream [LiteLLM Proxy](https://github.com/BerriAI/litellm). It serves as a unified control plane for all AI interactions—LLMs, Agents, Tools (MCP), and Skills—while adding a layer of **pluggable routing intelligence** and a **scripted MLOps pipeline**.

While LiteLLM provides the core proxy and protocol translation, RouteIQ Gateway adds:
- **Intelligent Routing**: Pluggable strategies (KNN, MLP, etc.) that learn from your traffic.
- **MLOps Pipeline**: A complete set of scripts and tools to collect telemetry, train routing models, and roll them out (CI/CD driven).
- **Enterprise Hardening**: Enhanced security, observability, and deployment patterns for cloud-native environments.

## Gateway Surfaces

RouteIQ Gateway unifies multiple AI interaction patterns under a single endpoint:

| Feature | Status | Description |
|---------|--------|-------------|
| **LLM Proxy** | 🟢 **Stable** | Standard OpenAI-compatible chat/completions. |
| **Observability** | 🟢 **Stable** | OpenTelemetry tracing, metrics, and logging. |
| **Security** | 🟢 **Stable** | SSRF protection, Admin Auth, Role-based access. |
| **A2A (Agent-to-Agent)** | 🟡 **Beta** | Protocol for multi-agent orchestration. |
| **MCP Gateway** | 🟡 **Beta** | Connect LLMs to external tools via Model Context Protocol. <br> *Note: Tool invocation is disabled by default.* |
| **Skills** | 🟡 **Beta** | Anthropic Computer Use, Bash, and Text Editor skills. |
| **Vector Stores** | 🔴 **Experimental** | Inherited OpenAI endpoints; deep external DB integration planned. |

See [Feature Parity & Roadmap](plans/parity-roadmap.md) for details.

## Architecture

The gateway operates as the central nervous system for your AI infrastructure, organized into two logical planes:

1.  **Data Plane (Gateway Runtime)**: The in-path serving component (LiteLLM + Routing Intelligence Layer) that receives API traffic and forwards it to LLM providers.
2.  **Control Plane (Management + Delivery)**: The out-of-path systems that configure the gateway and deliver routing models.

### Core Loop

1.  **Route**: Incoming requests are analyzed by the **Routing Intelligence Layer** (inside the Data Plane) and routed to the best model using the active strategy.
2.  **Observe**: Execution data (latency, cost, feedback) is captured via OpenTelemetry.
3.  **Learn**: The MLOps pipeline (external scripts/CI) uses this data to train improved routing models.
4.  **Update**: New models are hot-reloaded into the gateway without restarting.

## Quick Start

### Choose Your Deployment Mode

| Goal | Recommended Setup |
|------|-------------------|
| **"I just want to try it out."** | [**Basic Docker Compose**](docs/quickstart-docker-compose.md) <br> *Simple, single container, local config.* |
| **"I need production reliability."** | [**High Availability (HA)**](docs/tutorials/ha-quickstart.md) <br> *Multi-replica, Redis/Postgres backed, Nginx load balancing.* |
| **"I need to debug or optimize."** | [**Observability Stack**](docs/tutorials/observability-quickstart.md) <br> *Includes Jaeger for full trace visualization.* |

### 1. Docker Compose (Basic)

Ideal for local development.

```bash
# Clone and start
git clone https://github.com/baladithyab/litellm-llm-router.git
cd litellm-llm-router
cp .env.example .env
docker-compose up -d
```

[View Docker Compose Quickstart](docs/quickstart-docker-compose.md)

### 2. High Availability (Production)

Includes Redis and PostgreSQL for state and caching.

```bash
docker-compose -f docker-compose.ha.yml up -d
```

[View HA Quickstart](docs/tutorials/ha-quickstart.md)

### 3. Observability (OTel + Jaeger)

Includes Jaeger for full trace visualization.

```bash
docker-compose -f docker-compose.otel.yml up -d
```

[View Observability Quickstart](docs/tutorials/observability-quickstart.md)

## Deployment

RouteIQ Gateway is designed for cloud-native deployment:

- **Docker**: Official images available on GHCR.
- **Docker Compose**: Variants for local dev, HA (Redis/Postgres), and Observability (OTEL).
- **Kubernetes**: Helm charts and K8s manifests available for scalable deployments.
- **Health Probes**: Native support for cloud-native probes:
  - `/_health/live`: Liveness probe
  - `/_health/ready`: Readiness probe
- **Config Management**: Supports loading configuration from local files, S3, or environment variables.

## Production Checklist

See the [Deployment Guide](docs/deployment.md) for a comprehensive production checklist, including:

- Security & Key Management
- Persistence (PostgreSQL/Redis)
- TLS/SSL Termination
- Observability Setup
- Resource Limits & Scaling

## Security

Security is a first-class citizen in RouteIQ Gateway:

- **SSRF Protection**: Built-in guards against Server-Side Request Forgery.
- **Artifact Safety**: Pickle loading for ML models is **disabled by default** (opt-in only).
- **Key Management**: Secure handling of API keys.
- **Kubernetes Security**: Recommended security contexts included in deployment examples.

See the [Security Guide](docs/security.md) for details.

## Documentation

| Document | Description |
|----------|-------------|
| [Getting Started](docs/index.md) | Comprehensive guide to setting up and using the gateway. |
| [Deployment Guide](docs/deployment.md) | Docker, Kubernetes, and Cloud deployment patterns. |
| [Configuration Guide](docs/configuration.md) | Configuration options and Hot Reloading. |
| [API Reference](docs/api-reference.md) | Detailed API documentation. |
| [Routing Strategies](docs/routing-strategies.md) | Explanation of available routing strategies. |
| [MLOps Training](docs/mlops-training.md) | Guide to the MLOps training loop. |
| [MCP Gateway](docs/mcp-gateway.md) | Using the Model Context Protocol. |

## Supported Routing Strategies

### LiteLLM Built-in Strategies
- `simple-shuffle` - Random load balancing (default)
- `least-busy` - Route to the model with the fewest active requests
- `latency-based` - Route based on historical latency
- `usage-based` - Route based on token usage

### RouteIQ ML Strategies (18+ available)
- `llmrouter-knn` - K-Nearest Neighbors routing based on query embeddings
- `llmrouter-mlp` - Multi-Layer Perceptron routing
- `llmrouter-svm` - Support Vector Machine routing
- ... and more

## Configuration Example

```yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: os.environ/OPENAI_API_KEY

router_settings:
  routing_strategy: llmrouter-knn  # Use RouteIQ KNN strategy
  routing_strategy_args:
    model_path: /app/models/router_model.pkl
    hot_reload: true
    reload_interval: 300  # Check for updates every 5 minutes
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LiteLLM](https://github.com/BerriAI/litellm) for the amazing proxy foundation.
- [LLMRouter](https://github.com/ray-project/llm-router) for the routing intelligence concepts.
