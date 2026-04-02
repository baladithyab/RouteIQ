# Feature Parity & Roadmap

> **Attribution**:
> RouteIQ is built on top of upstream [LiteLLM](https://github.com/BerriAI/litellm) for proxy/API compatibility and [LLMRouter](https://github.com/ulab-uiuc/LLMRouter) for ML routing.

This document tracks the feature parity of RouteIQ Gateway compared to standard LiteLLM Proxy and outlines the roadmap for future enhancements.

## Feature Status

| Feature Area | Status | Notes |
|--------------|--------|-------|
| **Core Routing** | 🟢 **Stable** | ML-based routing (KNN, SVM), simple shuffle, least-busy. |
| **A2A (Agent-to-Agent)** | 🟡 **Beta** | Full support for A2A protocol, agent registration, and invocation. |
| **MCP (Model Context Protocol)** | 🟡 **Beta** | Server registration, tool discovery, and invocation (flag-gated). |
| **Skills Gateway** | 🟡 **Beta** | Support for Anthropic Computer Use, Bash, and Text Editor skills. |
| **Vector Stores** | 🔴 **Experimental** | Inherits OpenAI-compatible endpoints. Deep external DB integration (Pinecone, Qdrant) is planned. |
| **Observability** | 🟢 **Stable** | OpenTelemetry (OTel) tracing, metrics, and logging. Jaeger integration. |
| **Security** | 🟢 **Stable** | SSRF protection (deny-by-default), Admin Auth, Role-based access. |
| **High Availability** | 🟢 **Stable** | Redis-backed state sync, multi-replica support, load balancing. |

## Roadmap

Our roadmap is prioritized into P0 (Critical), P1 (High), and P2 (Medium) items.

### P0: Critical Stability & Core Features (Q1 2026)

- [x] **Unified Docker Container**: Single image for LiteLLM + LLMRouter.
- [x] **ML Routing Strategies**: KNN and SVM implementations.
- [x] **MLOps Tooling**: Scripts for training and deploying models (manual/CI trigger).
- [x] **Hot-Reload Config Sync**: Dynamic updates without restart.
- [x] **Routing Decision Visibility**: Full OTel instrumentation for *why* a model was chosen.

### P1: Enterprise Hardening (Q2 2026)

- [x] **Backpressure & Load Shedding**: Protect gateway from overload during spikes. ✅ Implemented in `resilience.py` (backpressure middleware, drain manager).
- [ ] **Multi-Replica Trace Correlation**: End-to-end tracing across load balancers.
- [x] **Durable Audit Export**: Compliance logging to PostgreSQL. ✅ Implemented in `audit.py`. S3/Kafka export is a future enhancement.
- [ ] **Secret Rotation**: Patterns for dynamic secret updates without downtime.

### P2: Advanced MLOps & Scale (Q3 2026)

- [ ] **Autoscaling Guidance**: KEDA/HPA metrics based on token throughput.
- [x] **Degraded Mode**: Circuit breakers for DB/Redis failures. ✅ Implemented in `resilience.py` (per-provider circuit breakers).
- [ ] **Multi-Region Sync**: Patterns for global deployments.
- [ ] **Automated MLOps Pipeline**: Trace -> Train -> Deploy loop.

## Parity Notes

### LiteLLM Proxy Compatibility

RouteIQ Gateway is built *on top of* LiteLLM Proxy. It inherits 100% of LiteLLM's features unless explicitly overridden.

- **Configuration**: Uses standard `config.yaml`.
- **API**: Fully compatible with OpenAI API format.
- **Providers**: Supports all 100+ LLM providers supported by LiteLLM.

### Key Differences

1.  **Routing Logic**: RouteIQ injects a custom router that uses ML models instead of simple round-robin.
2.  **Security Defaults**: RouteIQ enforces stricter security defaults (e.g., SSRF protection) suitable for enterprise "Moat Mode".
3.  **Protocol Support**: Native support for MCP and A2A protocols alongside standard chat completion.
