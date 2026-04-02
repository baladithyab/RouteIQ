# ADR-0025: Personalized Per-User Routing via Preference Embeddings

**Status**: Accepted
**Date**: 2026-04-02

## Context
Different users prefer different models. An engineer may prefer Claude for code,
a researcher may prefer GPT-4 for analysis. Static routing ignores these preferences.
GMTRouter (LLMRouter v0.2.0) uses graph neural networks for personalization, but
this requires PyG and complex data pipelines.

## Decision
Implement a lightweight preference learning system:
- 128-dim embedding vectors per user stored in Redis
- Static model embeddings derived from cost, context window, and capabilities
- Scoring: dot(user_preference, model_embedding) * 0.7 + quality_bias * 0.3
- Online learning via exponential moving average from feedback signals
- Cold-start fallback to centroid routing (zero-preference users)

## Implementation
- `PersonalizedRouter` integrated as Step 4 in routing pipeline
  (after ML routing, before final selection)
- `PreferenceStore` uses Redis with 90-day TTL, in-memory fallback
- `_build_model_embeddings()` derives features from MODEL_COSTS,
  MODEL_CONTEXT_WINDOWS, MODEL_CAPABILITIES (hash-based for diversity)
- Feedback via POST /routing/feedback updates preferences via EMA
- Temporal decay: preferences decay toward neutral at configurable rate

## Consequences
### Positive
- Zero cold-start latency (falls back to centroid)
- No PyTorch-geometric dependency
- ~0.5ms overhead per request (NumPy dot product)
- Works across workers via Redis

### Negative
- 128-dim is a simplification vs GMTRouter's heterogeneous GNN
- Requires explicit feedback (no implicit learning from response quality yet)
- Preference embeddings are hash-based approximations, not learned
