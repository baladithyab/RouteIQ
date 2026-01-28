# RouteIQ Gateway Documentation

Welcome to the documentation for **RouteIQ Gateway**, the cloud-native General AI Gateway powered by LiteLLM with pluggable routing intelligence and closed-loop MLOps.

## Quick Links

- [GitHub Repository](https://github.com/baladithyab/litellm-llm-router)
- [API Reference](api-reference.md)
- [API Parity Analysis](api-parity-analysis.md)
- [Configuration Guide](configuration.md)

## Getting Started

### Quick Start

```bash
# Clone and start
git clone https://github.com/baladithyab/litellm-llm-router.git
cd litellm-llm-router
docker compose up -d

# Test the gateway
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Architecture

RouteIQ Gateway is organized into a **Data Plane** (serving traffic) and a **Control Plane** (managing config/models).

- [Overview](architecture/overview.md) - High-level architecture of RouteIQ Gateway.
- [Cloud-Native Enhancements](litellm-cloud-native-enhancements.md) - How we extend LiteLLM for production.
- [MLOps Loop](architecture/ml-routing-cloud-native.md) - The closed-loop feedback system.

## Gateway Surfaces

RouteIQ Gateway supports multiple interaction patterns:

- [MCP Gateway](mcp-gateway.md) - Model Context Protocol integration.
- [A2A Gateway](a2a-gateway.md) - Agent-to-Agent communication.
- [Skills Gateway](skills-gateway.md) - Executable skills and functions.
- [Vector Stores](vector-stores.md) - Vector store endpoints (OpenAI-compatible).

## Routing & Intelligence

- [Routing Strategies](routing-strategies.md) - Available routing strategies (KNN, MLP, etc.).
- [MLOps Training](mlops-training.md) - Training and deploying routing models.
- [Moat Mode](moat-mode.md) - Defensive routing and fallback strategies.

## Deployment & Operations

- [Deployment Guide](deployment.md) - Docker, Kubernetes, and Cloud deployment patterns.
- [Security Guide](security.md) - SSRF, Artifact Safety, and Key Management.
- [High Availability](high-availability.md) - Setting up for HA with Redis.
- [Observability](observability.md) - Tracing and metrics with OpenTelemetry.
- [Hot Reloading](hot-reloading.md) - Updating configuration and models without downtime.

## Support

For issues and feature requests, please file an issue on [GitHub](https://github.com/baladithyab/litellm-llm-router/issues).
