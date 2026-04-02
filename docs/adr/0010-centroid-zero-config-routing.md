# ADR-0010: Centroid-Based Zero-Config Routing (NadirClaw Integration)

**Status**: Accepted
**Date**: 2026-04-02
**Decision Makers**: RouteIQ Core Team

## Context

### The Cold-Start Problem

RouteIQ's ML routing strategies (KNN, SVM, MLP, MF, ELO, hybrid) require
trained models built from telemetry data. The training pipeline
(`examples/mlops/`) extracts traces, trains models, and deploys them. This
creates a chicken-and-egg problem:

1. **New deployments have no telemetry** -> no trained models -> no intelligent
   routing -> falls back to LiteLLM's simple strategies (round-robin, least-busy)
2. **Time to first intelligent routing**: Days to weeks of telemetry collection
   + training before ML routing provides value
3. **High barrier to entry**: Users must set up the MLOps pipeline before seeing
   any benefit from RouteIQ's routing capabilities

This meant RouteIQ's primary differentiator (intelligent routing) was invisible
to new users, making evaluation and adoption difficult.

### NadirClaw's Approach

NadirClaw, a related project, solved this problem with a centroid-based binary
complexity classifier:

- Pre-compute embedding centroids for "simple" and "complex" prompts using a
  curated dataset
- At request time, compute cosine similarity between the prompt embedding and
  each centroid
- Classify prompts as simple or complex in ~2ms
- Route simple prompts to cheap/fast models, complex prompts to capable models

The centroids are small (~100KB), require no training, and provide immediate
value from the first request.

## Decision

Integrate NadirClaw's centroid-based routing as RouteIQ's zero-config default
routing strategy, with progressive enhancement to ML routing when trained
models are available.

### Implementation: `centroid_routing.py`

The implementation (~1,113 lines) provides:

#### CentroidClassifier

Binary prompt complexity classification using pre-computed centroid vectors:

```python
class CentroidClassifier:
    """~2ms prompt classification using cosine similarity."""

    def __init__(self, centroids_dir: Path):
        # Load pre-computed 384-dim embedding vectors
        self.simple_centroid = np.load(centroids_dir / "simple_centroid.npy")
        self.complex_centroid = np.load(centroids_dir / "complex_centroid.npy")

    def classify(self, prompt_embedding: np.ndarray) -> str:
        simple_sim = cosine_similarity(prompt_embedding, self.simple_centroid)
        complex_sim = cosine_similarity(prompt_embedding, self.complex_centroid)
        return "complex" if complex_sim > simple_sim else "simple"
```

The centroids are shipped as `.npy` files in `models/centroids/` (~100KB total).

#### AgenticDetector

Cumulative scoring for agentic patterns:

- Tool use indicators (`function_call`, `tools`, `tool_choice`)
- Multi-step execution markers
- Agentic keywords ("execute", "run", "invoke", etc.)

Prompts with high agentic scores are routed to models with strong tool-use
capabilities regardless of complexity classification.

#### ReasoningDetector

Regex-based reasoning marker detection:

- Step-by-step reasoning indicators
- Chain-of-thought markers
- Mathematical proof patterns
- Code generation patterns

Prompts with reasoning markers are routed to reasoning-optimized models.

#### 5-Tier Classification

The classifier produces a 5-tier complexity assessment:

| Tier | Description | Example |
|------|-------------|--------|
| 1 - Trivial | Simple factual, greetings, yes/no | "What is 2+2?" |
| 2 - Simple | Straightforward tasks, short answers | "Summarize this paragraph" |
| 3 - Moderate | Multi-step reasoning, analysis | "Compare these two approaches" |
| 4 - Complex | Deep reasoning, code generation | "Implement a B-tree in Rust" |
| 5 - Expert | Research, multi-domain synthesis | "Design a distributed system for..." |

#### 5 Routing Profiles

Profiles constrain model selection based on deployment preferences:

| Profile | Behavior | Use Case |
|---------|----------|----------|
| `auto` | Full tier-based routing | Default, optimal quality/cost |
| `eco` | Prefer cheaper models, only escalate for complex | Cost-sensitive deployments |
| `premium` | Always use most capable models | Quality-critical applications |
| `free` | Only use free/open-source models | Budget-constrained, privacy |
| `reasoning` | Always use reasoning-optimized models | Math, logic, code tasks |

Configured via `ROUTEIQ_ROUTING_PROFILE` (default: `auto`).

#### SessionCache

In-memory routing affinity cache with TTL and LRU eviction:

- Caches routing decisions per conversation
- Prevents model-switching mid-conversation
- Configurable TTL (default: 30 minutes)
- LRU eviction when cache exceeds max size

### Progressive Enhancement

Centroid routing is the **fallback**, not the ceiling. When ML models are
available, the routing pipeline uses them preferentially:

```
Request -> RoutingPipeline
  |-- ML strategy available? -> Use KNN/SVM/MLP/etc.
  |-- Centroid routing available? -> Use CentroidRoutingStrategy
  |-- Neither? -> Fall back to LiteLLM's default routing
```

As users collect telemetry and train ML models, routing automatically
upgrades from centroid-based to ML-based without any configuration change.

### Pre-Warming

Optional startup warming via `ROUTEIQ_CENTROID_WARMUP=true`:

```python
def warmup_centroid_classifier():
    """Pre-load centroids and warm numpy at startup."""
    classifier = get_centroid_classifier()
    # Run a dummy classification to warm JIT/caches
    dummy = np.random.randn(384).astype(np.float32)
    classifier.classify(dummy)
```

This adds ~100ms to startup but eliminates the first-request latency spike.

## Consequences

### Positive

- **Zero time to first intelligent routing**: From the first request,
  RouteIQ routes prompts to appropriate models based on complexity. No
  training pipeline, no telemetry collection, no MLOps setup.

- **Tiny footprint**: ~100KB of centroid vectors vs ~500MB+ of ML model
  artifacts. No PyTorch, no sentence-transformers needed.

- **Fast classification**: ~2ms per classification vs 10-50ms for KNN
  embedding + similarity search.

- **Smooth upgrade path**: Users start with centroid routing, see immediate
  value, then optionally invest in ML routing for better accuracy.

- **Slim image compatibility**: Centroid routing works in the slim Docker
  image (see [ADR-0009](0009-multi-tier-docker-images.md)) without ML
  dependencies.

### Negative

- **Lower accuracy than ML**: Centroid-based binary classification is less
  nuanced than trained ML models. It may misclassify borderline prompts.
  Acceptable as a starting point that gets replaced by ML over time.

- **Static centroids**: The pre-computed centroids represent a fixed
  understanding of prompt complexity. They don't adapt to the specific
  workload or model capabilities of a given deployment. ML routing
  addresses this via deployment-specific training.

- **numpy dependency**: Even centroid routing requires numpy (~20MB).
  This is acceptable as numpy is a near-universal Python dependency.

- **English-centric**: The centroid vectors were computed from predominantly
  English text. Classification accuracy for non-English prompts may be lower.

## Alternatives Considered

### Alternative A: No Default Routing

Require users to set up ML routing before getting intelligent routing.

- **Pros**: Simpler codebase; no centroid maintenance.
- **Cons**: Zero value until MLOps pipeline is running; high barrier to
  entry; poor first-run experience.
- **Rejected**: The cold-start problem makes RouteIQ unevaluable without
  some form of zero-config routing.

### Alternative B: Rule-Based Routing

Use hand-crafted rules (prompt length, keyword presence) instead of
embedding-based classification.

- **Pros**: No numpy dependency; no centroid files; fully deterministic.
- **Cons**: Extremely fragile; prompt length is a poor proxy for complexity;
  keyword-based rules are easy to game or misclassify; no semantic
  understanding.
- **Rejected**: Embedding-based classification is fundamentally more accurate
  than rule-based approaches.

### Alternative C: Online Learning

Train the classifier incrementally from production traffic without requiring
the full MLOps pipeline.

- **Pros**: Adapts to workload; no separate training step.
- **Cons**: Requires careful handling of concept drift, feedback loops, and
  model stability. Online learning for routing decisions is a research
  problem, not a production-ready solution.
- **Deferred**: Interesting future direction but premature for current needs.
  Centroid + offline ML training covers the immediate requirements.

## References

- `src/litellm_llmrouter/centroid_routing.py` — Full implementation
- `models/centroids/` — Pre-computed centroid vectors
- `src/litellm_llmrouter/custom_routing_strategy.py` — Integration with routing pipeline
- [ADR-0002: Plugin Routing Strategy](0002-plugin-routing-strategy.md)
- [ADR-0009: Multi-Tier Docker Images](0009-multi-tier-docker-images.md)
- [TG3 NadirClaw Integration](../architecture/tg3-nadirclaw-integration.md)
- NadirClaw BinaryComplexityClassifier architecture documentation
