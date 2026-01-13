# LiteLLM + LLMRouter: Intelligent LLM Gateway with ML-Powered Routing

<div align="center">

  **A production-ready LLM Gateway combining LiteLLM's unified API with LLMRouter's intelligent routing strategies**

  [![Docker Build](https://github.com/baladithyab/litellm-llm-router/actions/workflows/docker-build.yml/badge.svg)](https://github.com/baladithyab/litellm-llm-router/actions/workflows/docker-build.yml)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>

## Overview

This project integrates [LiteLLM](https://github.com/BerriAI/litellm) (a unified LLM API gateway) with [LLMRouter](https://github.com/ulab-uiuc/LLMRouter) (an intelligent routing library with 18+ ML-based routing strategies including single-round, multi-round, personalized, and agentic routers) into a single, production-ready Docker container.

### Key Features

- 🚀 **Unified LLM Gateway**: Access 100+ LLM providers through a single OpenAI-compatible API
- 🧠 **ML-Powered Routing**: 18+ intelligent routing strategies across 4 categories (single-round, multi-round, personalized, agentic)
- 🔄 **Hot-Reloading**: Update routing strategies without container restarts
- 📊 **High Availability**: Redis for distributed state, PostgreSQL for persistence, S3 for config sync
- 🐳 **Multi-Architecture**: Supports both `linux/amd64` and `linux/arm64`
- 🔧 **MLOps Ready**: Includes setup for training/finetuning routing models

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
```

### Using Pre-built Image

```bash
docker pull ghcr.io/baladithyab/litellm-llm-router:latest

docker run -d \
  -p 4000:4000 \
  -v ./config.yaml:/app/config.yaml \
  -e DATABASE_URL="postgresql://user:pass@host:5432/litellm" \
  ghcr.io/baladithyab/litellm-llm-router:latest \
  --config /app/config.yaml
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LiteLLM + LLMRouter Gateway                  │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   LiteLLM   │  │  LLMRouter  │  │    Custom Strategies    │  │
│  │   Proxy     │◄─│  Strategies │◄─│    (Hot-Reloadable)     │  │
│  │   Server    │  │   Family    │  │                         │  │
│  └──────┬──────┘  └─────────────┘  └─────────────────────────┘  │
│         │                                                        │
│  ┌──────▼──────────────────────────────────────────────────────┐ │
│  │              Routing Strategy Selection                     │ │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐    │ │
│  │  │  KNN   │ │  MLP   │ │  SVM   │ │  ELO   │ │ Custom │    │ │
│  │  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘    │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
    ┌─────────┐         ┌─────────┐         ┌─────────┐
    │ OpenAI  │         │ Anthropic│        │  Azure  │
    └─────────┘         └─────────┘         └─────────┘
```

## Documentation

### Getting Started
| Document | Description |
|----------|-------------|
| [Configuration Guide](docs/configuration.md) | Complete configuration reference |
| [Routing Strategies](docs/routing-strategies.md) | Available routing strategies and usage |
| [API Reference](docs/api-reference.md) | REST API documentation |

### Deployment & Operations
| Document | Description |
|----------|-------------|
| [Architecture Overview](docs/architecture/overview.md) | System architecture and data flow |
| [AWS Deployment Guide](docs/architecture/aws-deployment.md) | ECS, EKS, Fargate deployment patterns |
| [High Availability Setup](docs/high-availability.md) | HA deployment with Redis, PostgreSQL, S3 |
| [Hot Reloading](docs/hot-reloading.md) | Dynamic strategy updates without restarts |

### Observability & Training
| Document | Description |
|----------|-------------|
| [Observability Guide](docs/observability.md) | Tracing with Jaeger, Tempo, CloudWatch X-Ray |
| [Observability Training](docs/observability-training.md) | Using traces for model training |
| [MLOps Training](docs/mlops-training.md) | Training and finetuning routing models |

### Extensions
| Document | Description |
|----------|-------------|
| [A2A Gateway](docs/a2a-gateway.md) | Agent-to-Agent protocol support |
| [MCP Gateway](docs/mcp-gateway.md) | Model Context Protocol support |
| [Vector Stores](docs/vector-stores.md) | Vector database integration |

## Supported Routing Strategies

### LiteLLM Built-in Strategies
- `simple-shuffle` - Random load balancing (default)
- `least-busy` - Route to least busy deployment
- `latency-based-routing` - Route based on response latency
- `cost-based-routing` - Route to minimize cost
- `usage-based-routing` - Route based on TPM/RPM usage

### LLMRouter ML Strategies (18+ available)

**Single-Round:**
- `llmrouter-knn` - K-Nearest Neighbors routing
- `llmrouter-svm` - Support Vector Machine routing
- `llmrouter-mlp` - Multi-Layer Perceptron routing
- `llmrouter-mf` - Matrix Factorization routing
- `llmrouter-elo` - Elo Rating based routing
- `llmrouter-routerdc` - Dual Contrastive learning (NeurIPS 2024)
- `llmrouter-hybrid` - Hybrid probabilistic routing (ICLR 2024)
- `llmrouter-graph` - Graph neural network routing (ICLR 2025)
- `llmrouter-automix` - Automatic model mixing (NeurIPS 2024)
- `llmrouter-causallm` - Causal LM router

**Multi-Round/Personalized/Agentic:**
- `llmrouter-r1` - Router-R1 multi-turn (NeurIPS 2025)
- `llmrouter-gmt` - Personalized user-preference router
- `llmrouter-knn-multiround` - KNN agentic router
- `llmrouter-llm-multiround` - LLM agentic router

**Baselines:**
- `llmrouter-smallest` - Always smallest (cost baseline)
- `llmrouter-largest` - Always largest (quality baseline)
- `llmrouter-custom` - Custom trained models

## Configuration Example

```yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: os.environ/OPENAI_API_KEY
  - model_name: claude-3
    litellm_params:
      model: anthropic/claude-3-opus
      api_key: os.environ/ANTHROPIC_API_KEY

router_settings:
  routing_strategy: llmrouter-knn  # Use LLMRouter KNN strategy
  routing_strategy_args:
    model_path: /app/models/knn_router.pt
    llm_data_path: /app/config/llm_candidates.json
    hot_reload: true
    reload_interval: 300  # Check for updates every 5 minutes

general_settings:
  master_key: sk-1234
  database_url: os.environ/DATABASE_URL

litellm_settings:
  cache: true
  cache_params:
    type: redis
    host: os.environ/REDIS_HOST
    port: 6379
```

## AWS Deployment

Deploy to AWS using ECS Fargate, EKS, or App Runner:

```bash
# Build and push to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker build -t litellm-llmrouter -f docker/Dockerfile .
docker tag litellm-llmrouter:latest <account>.dkr.ecr.<region>.amazonaws.com/litellm-llmrouter:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/litellm-llmrouter:latest
```

Key AWS integrations:
- **Amazon Bedrock**: Native Claude, Titan, and other foundation models
- **CloudWatch X-Ray**: Distributed tracing via ADOT sidecar
- **S3**: Configuration and model storage with hot-reload
- **Secrets Manager**: Secure API key management
- **ElastiCache Redis**: Distributed caching and rate limiting

See [AWS Deployment Guide](docs/architecture/aws-deployment.md) for complete instructions.

## Container Features

| Feature | Description |
|---------|-------------|
| **Non-root user** | Runs as `litellm` user (UID 1000) |
| **tini init** | Proper signal handling and zombie process reaping |
| **Health checks** | `/health/liveliness` endpoint for orchestrators |
| **Multi-arch** | Supports `linux/amd64` and `linux/arm64` |
| **OCI labels** | Standard container metadata |

## License

MIT License - See [LICENSE](LICENSE) for details.

## Acknowledgments

- [LiteLLM](https://github.com/BerriAI/litellm) - Unified LLM API Gateway
- [LLMRouter](https://github.com/ulab-uiuc/LLMRouter) - Intelligent LLM Routing Library
