# Technical Roadmap

This document outlines the technical direction for the Production AI Gateway. It focuses on the integration of LiteLLM and LLMRouter, emphasizing stability, observability, and cloud-native capabilities.

For a detailed backlog of specific enhancements, see **[LiteLLM Cloud-Native Enhancements](litellm-cloud-native-enhancements.md)**.

## Q1 2026: Core Integration & Stability (Current Focus)

*   **Goal**: A stable, production-ready container combining LiteLLM Proxy and LLMRouter.
*   **Key Deliverables**:
    *   [x] Unified Docker container (LiteLLM + LLMRouter).
    *   [x] Basic ML Routing strategies (KNN, SVM) - see [ML Routing Architecture](architecture/ml-routing-cloud-native.md).
    *   [ ] **P0: Hot-Reload Config Sync**: Dynamic updates for models and config without restart.
    *   [ ] **P0: Routing Decision Visibility**: Full OTel instrumentation for routing logic.
    *   [ ] **P1: Streaming-Aware Shutdown**: Graceful termination for long-running LLM streams.

## Q2 2026: Enterprise Hardening

*   **Goal**: Advanced security, resilience, and compliance features for "Moat-Mode" deployments.
*   **Key Deliverables**:
    *   [ ] **P1: Backpressure & Load Shedding**: Protect gateway from overload.
    *   [ ] **P1: Multi-Replica Trace Correlation**: End-to-end tracing across load balancers and replicas.
    *   [ ] **P1: Durable Audit Export**: Compliance logging to S3/Kafka.
    *   [ ] **P1: Secret Rotation**: Patterns for dynamic secret updates.

## Q3 2026: Advanced MLOps & Scale

*   **Goal**: Closed-loop MLOps for routing models and autoscaling guidance.
*   **Key Deliverables**:
    *   [ ] **P2: Autoscaling Guidance**: KEDA/HPA metrics for token throughput.
    *   [ ] **P2: Degraded Mode**: Circuit breakers for DB/Redis failures.
    *   [ ] **P2: Multi-Region Sync**: Patterns for global deployments.
    *   [ ] Automated MLOps pipeline (Trace -> Train -> Deploy).

## Long Term

*   **Goal**: Fully autonomous, self-optimizing gateway.
*   **Themes**:
    *   Online Learning for Routers (Bandit algorithms).
    *   Marketplace for Routing Models.
    *   Deep integration with Kubernetes Operators.
