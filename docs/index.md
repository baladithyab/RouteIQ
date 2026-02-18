# RouteIQ Documentation

> Comprehensive documentation for the RouteIQ AI Gateway — a cloud-native General AI Gateway
> built on [LiteLLM](https://github.com/BerriAI/litellm) and [LLMRouter](https://github.com/ulab-uiuc/LLMRouter).

---

## Getting Started

| Document | Description |
|----------|-------------|
| [Docker Compose Quickstart](quickstart-docker-compose.md) | Get RouteIQ running in 5 minutes with Docker Compose |
| [Configuration Guide](configuration.md) | Configure models, routing strategies, and gateway features |
| [Tutorial: High Availability](tutorials/ha-quickstart.md) | Step-by-step HA setup with Redis, Postgres, and Nginx |
| [Tutorial: Observability](tutorials/observability-quickstart.md) | Set up OpenTelemetry tracing with Jaeger |

## Core Features

| Document | Description |
|----------|-------------|
| [Routing Strategies](routing-strategies.md) | ML-based routing: KNN, MLP, SVM, ELO, MF, hybrid, and custom strategies |
| [Plugin System](plugins.md) | Extensible plugin architecture with lifecycle management and development guide |
| [MCP Gateway](mcp-gateway.md) | Model Context Protocol support (JSON-RPC, SSE, REST surfaces) |
| [A2A Gateway](a2a-gateway.md) | Agent-to-Agent communication protocol |
| [Skills Gateway](skills-gateway.md) | Anthropic Computer Use, Bash, and Text Editor skill execution |
| [Vector Stores](vector-stores.md) | Vector store integration for RAG and semantic search |

## Deployment

| Document | Description |
|----------|-------------|
| [Deployment Overview](deployment.md) | Docker, Kubernetes (Helm), and HA deployment options |
| [AWS Production Guide](aws-production-guide.md) | Complete AWS production deployment with ECS, ALB, and CloudWatch |
| [AWS Compute Options](deployment/aws.md) | ECS, EKS, Lambda, and EC2 deployment patterns |
| [CloudFront + ALB](deployment/cloudfront-alb.md) | CloudFront distribution with Application Load Balancer |
| [Air-Gapped Deployment](deployment/air-gapped.md) | Deploy RouteIQ in disconnected or restricted environments |

## Operations

| Document | Description |
|----------|-------------|
| [Observability Guide](observability.md) | OpenTelemetry traces, metrics, and logs configuration |
| [Security Guide](security.md) | Authentication, RBAC, SSRF protection, and secret management |
| [MLOps Training Pipeline](mlops-training.md) | Closed-loop MLOps: telemetry extraction, model training, deployment |
| [Streaming Verification](streaming-verification.md) | SSE streaming correctness and performance verification report |

## Architecture

### Core Architecture

| Document | Description |
|----------|-------------|
| [Architecture Overview](architecture/overview.md) | High-level system architecture and component diagram |
| [Cloud-Native Architecture](architecture/cloud-native.md) | Cloud-native design principles and implementation |
| [MLOps Loop Architecture](architecture/mlops-loop.md) | ML training loop architecture and data flow |
| [Bedrock Discovery](architecture/bedrock-discovery.md) | AWS Bedrock model discovery and integration architecture |

### Architecture Reviews

| Document | Description |
|----------|-------------|
| [Deep Review v0.2.0](architecture/deep-review-v0.2.0.md) | Deep architecture review for v0.2.0 release |
| [Architecture Review (Feb 2026)](architecture/ROUTEIQ-ARCHITECTURE-REVIEW-2026-02.md) | Comprehensive February 2026 architecture review |
| [TG2 Architecture Evaluation](architecture/tg2-architecture-evaluation.md) | Task Group 2 architecture evaluation and findings |

### TG3 Design Documents

| Document | Description |
|----------|-------------|
| [TG3 Rearchitecture Proposal](architecture/tg3-rearchitecture-proposal.md) | Proposed rearchitecture for Task Group 3 |
| [TG3 Alternative Patterns](architecture/tg3-alternative-patterns.md) | Alternative architecture patterns explored for TG3 |
| [TG3 Admin UI Design](architecture/tg3-admin-ui-design.md) | Admin UI design specification |
| [TG3 Cloud-Native Design](architecture/tg3-cloud-native-design.md) | Cloud-native design document for TG3 |
| [TG3 NadirClaw Integration](architecture/tg3-nadirclaw-integration.md) | NadirClaw integration design and architecture |

## Development

| Document | Description |
|----------|-------------|
| [API Reference](api-reference.md) | Complete API endpoint reference for all gateway surfaces |
| [Project State](project-state.md) | Current project state, known gaps, and status tracking |
| [Road Runner Workflow](rr-workflow.md) | Remote push workflow for code deployment |
| [HA CI Gate](ha-ci-gate.md) | High-availability continuous integration gate documentation |
| [Load & Soak Test Gates](load-soak-gates.md) | Load testing and soak testing quality gates |

## Planning

| Document | Description |
|----------|-------------|
| [Implementation Decomposition](../plans/implementation-decomposition.md) | Master task decomposition and implementation plan |
| [v0.1.0 Design Document](plans/2026-02-13-routeiq-v0.1.0-design.md) | Original v0.1.0 design specification |
| [v0.1.0 Implementation Plan](plans/2026-02-13-routeiq-v0.1.0-implementation.md) | v0.1.0 implementation plan and milestones |

---

*See also: [README](../README.md) · [Contributing](../CONTRIBUTING.md) · [Agent Instructions](../AGENTS.md)*
