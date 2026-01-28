# RouteIQ Gateway
Cloud-native General AI Gateway (powered by LiteLLM) with pluggable routing intelligence + closed-loop MLOps

<div align="center">

  **A cloud-grade AI gateway with pluggable routing and end-to-end MLOps**

  [![Docker Build](https://github.com/baladithyab/litellm-llm-router/actions/workflows/docker-build.yml/badge.svg)](https://github.com/baladithyab/litellm-llm-router/actions/workflows/docker-build.yml)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>

## Overview

**RouteIQ Gateway** is a production-grade, cloud-native AI Gateway that extends the capabilities of the upstream [LiteLLM Proxy](https://github.com/BerriAI/litellm). It adds a layer of **pluggable routing intelligence** and a **closed-loop MLOps workflow**, enabling organizations to optimize LLM traffic for cost, latency, and quality using their own data.

While LiteLLM provides the core proxy and protocol translation, RouteIQ Gateway adds:
- **Intelligent Routing**: Pluggable strategies (KNN, MLP, etc.) that learn from your traffic.
- **Closed-Loop MLOps**: A complete loop to collect telemetry, train routing models, register them, and roll them out without downtime.
- **Enterprise Hardening**: Enhanced security, observability, and deployment patterns for cloud-native environments.

## Gateway Surfaces

RouteIQ Gateway unifies multiple AI interaction patterns under a single endpoint:

- **Standard OpenAI API**: Full compatibility for chat, completions, and embeddings.
- **Extended API Families**: Inherits full support for `/v1/responses`, `/v1/assistants`, `/v1/files`, and more. See [API Parity Analysis](docs/api-parity-analysis.md).
- **MCP (Model Context Protocol)**: Bridge MCP servers to LLMs, allowing models to use tools and resources securely.
- **A2A (Agent-to-Agent)**: Standardized protocol for multi-agent communication and orchestration.
- **Skills**: Register and execute Python functions as "skills" that can be invoked by models.
- **Vector Stores**: *Inherited* OpenAI-compatible `/v1/vector_stores*` endpoints for file search.
  > **Note**: Deep integration with external Vector Databases (Pinecone, Weaviate, etc.) is currently *planned* and not yet fully implemented.

## Architecture

The gateway operates as the central nervous system for your AI infrastructure, organized into two logical planes:

1.  **Data Plane (Gateway Runtime)**: The in-path serving component (LiteLLM + Routing Intelligence Layer) that receives API traffic and forwards it to LLM providers.
2.  **Control Plane (Management + Delivery)**: The out-of-path systems that configure the gateway and deliver routing models.

### Core Loop

1.  **Route**: Incoming requests are analyzed by the **Routing Intelligence Layer** (inside the Data Plane) and routed to the best model using the active strategy.
2.  **Observe**: Execution data (latency, cost, feedback) is captured via OpenTelemetry.
3.  **Learn**: The MLOps pipeline uses this data to train improved routing models.
4.  **Update**: New models are hot-reloaded into the gateway without restarting.

## Quick Start

### Using Docker Compose

```bash
# Clone the repository
git clone https://github.com/baladithyab/litellm-llm-router.git
cd litellm-llm-router

# Start with basic setup
docker compose up -d

# Or start with full HA stack (Redis + PostgreSQL)
docker compose -f docker-compose.ha.yml up -d

# Or start with Observability stack (Jaeger + Prometheus)
docker compose -f docker-compose.otel.yml up -d
```

### Using Pre-built Image

```bash
docker run -p 4000:4000 ghcr.io/baladithyab/litellm-llm-router:latest
```

## Deployment

RouteIQ Gateway is designed for cloud-native deployment:

- **Docker**: Official images available on GHCR.
- **Docker Compose**: Variants for local dev, HA (Redis/Postgres), and Observability (OTEL).
- **Kubernetes**: Helm charts and K8s manifests available for scalable deployments.
- **Health Probes**: Native support for cloud-native probes:
  - `/_health/live`: Liveness probe
  - `/_health/ready`: Readiness probe
- **Config Management**: Supports loading configuration from local files, S3, or environment variables.

## Security

Security is a first-class citizen in RouteIQ Gateway:

- **SSRF Protection**: Built-in guards against Server-Side Request Forgery.
- **Artifact Safety**: Pickle loading for ML models is **disabled by default** (opt-in only) to prevent arbitrary code execution. Use `LLMROUTER_ALLOW_PICKLE_MODELS=true` to enable if necessary.
- **Key Management**: Secure handling of API keys via environment variables or secret managers.
- **Kubernetes Security**: Recommended security contexts (non-root user, read-only root filesystem) included in deployment examples.

## Documentation

| Document | Description |
|----------|-------------|
| [Getting Started](docs/index.md) | Comprehensive guide to setting up and using the gateway. |
| [API Reference](docs/api-reference.md) | Detailed API documentation. |
| [Routing Strategies](docs/routing-strategies.md) | Explanation of available routing strategies. |
| [MLOps Training](docs/mlops-training.md) | Guide to the MLOps training loop. |
| [MCP Gateway](docs/mcp-gateway.md) | Using the Model Context Protocol. |
| [Skills Gateway](docs/skills-gateway.md) | Registering and using skills. |

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
