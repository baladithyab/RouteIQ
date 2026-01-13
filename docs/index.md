# LiteLLM + LLMRouter Documentation

Welcome to the LiteLLM + LLMRouter documentation. This gateway combines LiteLLM's unified LLM API with LLMRouter's intelligent ML-powered routing strategies.

## Quick Links

- **[Getting Started](#getting-started)** - Set up the gateway quickly
- **[Architecture](#architecture)** - Understand system design
- **[Deployment](#deployment)** - Production deployment guides
- **[Observability](#observability)** - Monitoring and tracing

---

## Getting Started

| Document | Description |
|----------|-------------|
| [Configuration](configuration.md) | Complete configuration reference |
| [Routing Strategies](routing-strategies.md) | Available routing strategies |
| [API Reference](api-reference.md) | REST API documentation |

### Quick Start

```bash
# Clone and start
git clone https://github.com/baladithyab/litellm-llm-router.git
cd litellm-llm-router
docker compose up -d

# Test the gateway
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-dev-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}'
```

---

## Architecture

| Document | Description |
|----------|-------------|
| [Overview](architecture/overview.md) | System architecture and data flow |
| [AWS Deployment](architecture/aws-deployment.md) | ECS, EKS, Fargate patterns |

### Key Components

```
┌────────────────────────────────────────────────────┐
│              LiteLLM + LLMRouter Gateway           │
│  ┌──────────────┐  ┌──────────────┐               │
│  │   LiteLLM    │  │   LLMRouter  │               │
│  │   Proxy      │◄─│   Strategies │               │
│  └──────┬───────┘  └──────────────┘               │
│         │                                          │
│  ┌──────▼───────────────────────────────────────┐ │
│  │          LLM Providers                        │ │
│  │  OpenAI | Anthropic | Bedrock | Azure | ...   │ │
│  └───────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────┘
```

---

## Deployment

| Document | Description |
|----------|-------------|
| [High Availability](high-availability.md) | HA with Redis, PostgreSQL, S3 |
| [Hot Reloading](hot-reloading.md) | Dynamic updates without restarts |
| [AWS Deployment](architecture/aws-deployment.md) | ECS/EKS/Fargate deployment |

### Deployment Options

| Platform | Best For | Scaling |
|----------|----------|---------|
| Docker Compose | Development, small deployments | Manual |
| ECS Fargate | AWS, serverless compute | Auto-scaling |
| EKS | Kubernetes expertise | Fine-grained |
| Local Kubernetes | On-premises, hybrid cloud | Configurable |

---

## Observability

| Document | Description |
|----------|-------------|
| [Observability Guide](observability.md) | Tracing with Jaeger, Tempo, X-Ray |
| [Observability Training](observability-training.md) | Using traces for ML training |
| [MLOps Training](mlops-training.md) | Training routing models |

### Supported Backends

- **Jaeger** - Local development, quick setup
- **Grafana Tempo** - Production with S3 storage
- **AWS CloudWatch X-Ray** - Native AWS integration

---

## Extensions

| Document | Description |
|----------|-------------|
| [A2A Gateway](a2a-gateway.md) | Agent-to-Agent protocol |
| [MCP Gateway](mcp-gateway.md) | Model Context Protocol |
| [Vector Stores](vector-stores.md) | Vector database integration |

---

## Docker Images

| Image | Purpose |
|-------|---------|
| `docker/Dockerfile` | Production multi-stage build |
| `docker/Dockerfile.local` | Development with local reference |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LITELLM_CONFIG_PATH` | `/app/config/config.yaml` | Config file path |
| `LITELLM_MASTER_KEY` | - | API authentication key |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | - | OpenTelemetry collector |
| `CONFIG_S3_BUCKET` | - | S3 bucket for config sync |
| `LLMROUTER_MODEL_S3_BUCKET` | - | S3 bucket for router models |

---

## Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/baladithyab/litellm-llm-router/issues)
- **LiteLLM Docs**: [docs.litellm.ai](https://docs.litellm.ai)
- **LLMRouter Paper**: [arxiv.org/abs/2406.12345](https://arxiv.org/abs/2406.12345)
