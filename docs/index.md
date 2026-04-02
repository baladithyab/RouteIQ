# RouteIQ Gateway

**Cloud-Native AI Gateway with Intelligent Routing**

RouteIQ is an open-source AI gateway that provides intelligent ML-based routing,
enterprise governance, and cloud-native infrastructure for LLM APIs. Built on
[LiteLLM](https://github.com/BerriAI/litellm) for proxy/API compatibility and
[LLMRouter](https://github.com/ulab-uiuc/LLMRouter) for ML-based routing.

## Key Features

- **18+ ML Routing Strategies** — KNN, SVM, MLP, ELO, MF, centroid, personalized
- **Enterprise Governance** — Workspaces, budgets, rate limits, guardrails, OIDC/SSO
- **MCP + A2A + Skills** — Full protocol support for model context, agents, and skills
- **Context Optimization** — Token savings via lossless transforms
- **Cloud-Native** — K8s-native, multi-worker, Redis-backed state, Helm charts
- **Plugin System** — 13 built-in plugins, extensible architecture with lifecycle management
- **Observability** — OpenTelemetry traces, metrics, and structured logging

## Quick Start

=== "uv (recommended)"

    ```bash
    uv add routeiq
    uv run routeiq start --config config.yaml
    ```

=== "pip"

    ```bash
    pip install routeiq
    routeiq start --config config.yaml
    ```

=== "Docker"

    ```bash
    docker run -p 4000:4000 \
      -e LITELLM_MASTER_KEY=sk-your-key \
      routeiq:latest
    ```

=== "Docker Compose"

    ```bash
    git clone https://github.com/baladithyab/RouteIQ.git
    cd routeiq
    docker compose up -d
    ```

## Verify Installation

```bash
# Health check
curl http://localhost:4000/_health/ready

# Make a request
curl -X POST http://localhost:4000/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello from RouteIQ!"}]
  }'
```

## Documentation Overview

| Section | Description |
|---------|-------------|
| [Getting Started](getting-started/quickstart.md) | 5-minute quickstart, installation, and configuration |
| [Features](features/routing.md) | Routing strategies, MCP, A2A, Skills, and more |
| [Governance](governance/overview.md) | Workspaces, policies, guardrails, and identity |
| [Operations](operations/deployment.md) | Deployment, observability, Docker, Helm, security |
| [Architecture](architecture/overview.md) | System architecture and ADRs |
| [API Reference](api/gateway.md) | Gateway, governance, and routing APIs |
| [Contributing](contributing.md) | Development setup and contribution guidelines |
