# Architecture Overview

LiteLLM + LLMRouter is an intelligent LLM gateway that combines LiteLLM's unified API with ML-powered routing.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LiteLLM + LLMRouter Gateway                        │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        LiteLLM Proxy Server                          │    │
│  │                                                                       │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │    │
│  │  │   OpenAI     │  │   Unified    │  │    Request Processing    │   │    │
│  │  │ Compatible   │──│     API      │──│  ─────────────────────── │   │    │
│  │  │  Endpoints   │  │   Layer      │  │  │ Rate Limiting       │ │   │    │
│  │  │              │  │              │  │  │ Auth/Keys           │ │   │    │
│  │  │ /v1/chat     │  │              │  │  │ Caching             │ │   │    │
│  │  │ /v1/complete │  │              │  │  │ Logging             │ │   │    │
│  │  └──────────────┘  └──────────────┘  └──────────┬───────────────┘   │    │
│  │                                                  │                    │    │
│  │  ┌───────────────────────────────────────────────▼────────────────┐  │    │
│  │  │                     Router Integration                          │  │    │
│  │  │  ┌─────────────────────────────────────────────────────────┐   │  │    │
│  │  │  │              LLMRouter (Custom Strategy)                 │   │  │    │
│  │  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │   │  │    │
│  │  │  │  │  KNN Router │  │ MLP Router  │  │ Custom Routers  │  │   │  │    │
│  │  │  │  │ (Embedding  │  │ (Neural     │  │ (Plugin-based)  │  │   │  │    │
│  │  │  │  │  Based)     │  │  Network)   │  │                 │  │   │  │    │
│  │  │  │  └─────────────┘  └─────────────┘  └─────────────────┘  │   │  │    │
│  │  │  └─────────────────────────────────────────────────────────┘   │  │    │
│  │  │  ┌─────────────────────────────────────────────────────────┐   │  │    │
│  │  │  │              LiteLLM Built-in Strategies                 │   │  │    │
│  │  │  │  simple-shuffle │ least-busy │ latency-based │ cost-based│   │  │    │
│  │  │  └─────────────────────────────────────────────────────────┘   │  │    │
│  │  └────────────────────────────────────────────────────────────────┘  │    │
│  │                                    │                                  │    │
│  └────────────────────────────────────┼──────────────────────────────────┘    │
│                                       │                                       │
│  ┌────────────────────────────────────▼──────────────────────────────────┐   │
│  │                         LLM Providers                                  │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────┐  │   │
│  │  │ OpenAI  │  │ Anthropic│  │ Bedrock │  │  Azure  │  │ Local LLMs  │  │   │
│  │  │ GPT-4   │  │ Claude   │  │ Claude  │  │ OpenAI  │  │ Ollama/vLLM │  │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────────┘  │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. LiteLLM Proxy Server
- OpenAI-compatible REST API (`/v1/chat/completions`, `/v1/completions`, etc.)
- Request authentication, rate limiting, and budget management
- Response caching with Redis support
- Comprehensive logging and metrics

### 2. LLMRouter Integration
ML-powered routing strategies that select the optimal model based on:
- **Query Embeddings**: Semantic understanding of requests
- **Historical Performance**: Latency, cost, and quality metrics
- **Model Specialization**: Match queries to model strengths

### 3. Supported Routers

| Router | Description | Use Case |
|--------|-------------|----------|
| **KNN** | K-nearest neighbors on embeddings | When you have labeled training data |
| **MLP** | Multi-layer perceptron classifier | Complex decision boundaries |
| **Cost-Based** | Minimize cost within quality threshold | Budget optimization |
| **Latency-Based** | Minimize response time | Real-time applications |
| **Custom** | Plugin-based extensible routers | Domain-specific logic |

## Data Flow

```
Client Request
      │
      ▼
┌───────────────┐
│  API Gateway  │ ─── Authentication, Rate Limiting
└───────┬───────┘
        │
        ▼
┌───────────────┐
│   LiteLLM     │ ─── Request parsing, validation
│   Proxy       │
└───────┬───────┘
        │
        ▼
┌───────────────┐     ┌─────────────────┐
│   Router      │ ◄── │  Model Registry │
│   Strategy    │     │  (S3, Config)   │
└───────┬───────┘     └─────────────────┘
        │
        │ Select optimal deployment
        ▼
┌───────────────┐
│  LLM Provider │ ─── OpenAI, Anthropic, Bedrock, etc.
└───────┬───────┘
        │
        ▼
┌───────────────┐     ┌─────────────────┐
│   Response    │ ──► │  Observability  │
│   Processing  │     │  (OTEL, Jaeger) │
└───────┬───────┘     └─────────────────┘
        │
        ▼
   Client Response
```

## Directory Structure

```
litellm-llm-router/
├── config/                    # Configuration files
│   ├── config.yaml           # Main gateway config
│   ├── config.bedrock.yaml   # AWS Bedrock config
│   └── llm_candidates.json   # Model definitions
├── custom_routers/           # Custom routing strategies
├── docker/                   # Docker configurations
│   ├── Dockerfile           # Production image
│   ├── Dockerfile.local     # Development image
│   └── entrypoint.sh        # Container entrypoint
├── docs/                     # Documentation
│   ├── architecture/        # Architecture docs
│   ├── observability.md     # Tracing/metrics
│   └── routing-strategies.md # Router details
├── examples/                 # Example configurations
│   └── mlops/               # MLOps training stack
├── models/                   # Trained router models
├── scripts/                  # Utility scripts
│   └── train_from_traces.py # Training pipeline
├── src/                      # Source code
│   └── litellm_llmrouter/   # Custom integration
└── tests/                    # Test suite
```

## Key Features

- **🔌 100+ LLM Providers**: Unified API for OpenAI, Anthropic, Azure, Bedrock, and more
- **🎯 ML-Powered Routing**: Intelligent model selection based on query analysis
- **📊 Full Observability**: OpenTelemetry integration with Jaeger, X-Ray, Grafana
- **🔄 Hot Reload**: Dynamic model and configuration updates without downtime
- **🔐 Enterprise Security**: API key management, rate limiting, SSO integration
- **📈 Cost Management**: Token budgets, cost tracking, and optimization

## Related Documentation

- [AWS Deployment Guide](./aws-deployment.md)
- [Configuration Reference](../configuration.md)
- [Routing Strategies](../routing-strategies.md)
- [Observability Guide](../observability.md)
- [High Availability](../high-availability.md)
