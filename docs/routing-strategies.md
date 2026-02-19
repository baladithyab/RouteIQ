# Routing Strategies

> **Attribution**:
> RouteIQ is built on top of upstream [LiteLLM](https://github.com/BerriAI/litellm) for proxy/API compatibility and [LLMRouter](https://github.com/ulab-uiuc/LLMRouter) for ML routing.

This document covers all available routing strategies in the RouteIQ Gateway.

## Table of Contents

- [LiteLLM Built-in Strategies](#litellm-built-in-strategies)
- [LLMRouter ML-Based Strategies](#llmrouter-ml-based-strategies)
- [Centroid Routing (Zero-Config)](#centroid-routing-zero-config)
- [A/B Testing & Runtime Hot-Swapping](#ab-testing--runtime-hot-swapping)
- [Model Artifact Security](#model-artifact-security)
- [Hot Reloading](#hot-reloading)
- [Training Custom Models](#training-custom-models)

## LiteLLM Built-in Strategies

These strategies are built into LiteLLM:

### simple-shuffle (default)
Random load balancing across deployments.

```yaml
router_settings:
  routing_strategy: simple-shuffle
```

### least-busy
Routes to the deployment with the fewest active requests.

```yaml
router_settings:
  routing_strategy: least-busy
```

### latency-based-routing
Routes based on historical response latency.

```yaml
router_settings:
  routing_strategy: latency-based-routing
```

### cost-based-routing
Routes to minimize token costs.

```yaml
router_settings:
  routing_strategy: cost-based-routing
```

## LLMRouter ML-Based Strategies

These strategies use machine learning models from [LLMRouter](https://github.com/ulab-uiuc/LLMRouter) to intelligently route queries. All 18+ strategies are available across four categories.

### Single-Round Routers

#### llmrouter-knn
K-Nearest Neighbors based routing. Fast and interpretable.

```yaml
router_settings:
  routing_strategy: llmrouter-knn
  routing_strategy_args:
    config_path: /app/config/knn_config.yaml
```

**Best for:** Quick deployment, interpretable decisions

#### llmrouter-svm
Support Vector Machine routing. Good generalization with margin-based selection.

```yaml
router_settings:
  routing_strategy: llmrouter-svm
  routing_strategy_args:
    config_path: /app/config/svm_config.yaml
```

**Best for:** Binary routing decisions, high generalization

#### llmrouter-mlp
Multi-Layer Perceptron routing. Neural network based.

```yaml
router_settings:
  routing_strategy: llmrouter-mlp
  routing_strategy_args:
    config_path: /app/config/mlp_config.yaml
```

**Best for:** Complex query patterns, high accuracy requirements

#### llmrouter-mf
Matrix Factorization routing. Collaborative filtering approach.

```yaml
router_settings:
  routing_strategy: llmrouter-mf
  routing_strategy_args:
    config_path: /app/config/mf_config.yaml
```

**Best for:** User preference learning, collaborative scenarios

#### llmrouter-elo
Elo Rating based routing. Uses competitive ranking to select models.

```yaml
router_settings:
  routing_strategy: llmrouter-elo
  routing_strategy_args:
    config_path: /app/config/elo_config.yaml
```

**Best for:** Model quality ranking, tournament-style selection

#### llmrouter-routerdc
Dual Contrastive learning based routing (RouterDC). Uses contrastive representations.

```yaml
router_settings:
  routing_strategy: llmrouter-routerdc
  routing_strategy_args:
    config_path: /app/config/routerdc_config.yaml
```

**Best for:** High accuracy, representation learning
**Reference:** [RouterDC (NeurIPS 2024)](https://arxiv.org/abs/2409.19886)

#### llmrouter-hybrid
HybridLLM probabilistic routing. Combines multiple signals.

```yaml
router_settings:
  routing_strategy: llmrouter-hybrid
  routing_strategy_args:
    config_path: /app/config/hybrid_config.yaml
```

**Best for:** Production deployments, balanced cost-quality decisions
**Reference:** [Hybrid LLM (ICLR 2024)](https://arxiv.org/abs/2404.14618)

#### llmrouter-causallm
Causal Language Model router. Transformer-based routing decisions.

```yaml
router_settings:
  routing_strategy: llmrouter-causallm
  routing_strategy_args:
    config_path: /app/config/causallm_config.yaml
```

**Best for:** Deep semantic understanding, complex reasoning tasks
**Note:** Requires PyTorch with transformers support

#### llmrouter-graph
Graph Neural Network routing. Models query-model relationships as graphs.

```yaml
router_settings:
  routing_strategy: llmrouter-graph
  routing_strategy_args:
    config_path: /app/config/graph_config.yaml
```

**Best for:** Relationship modeling, complex query dependencies
**Reference:** [GraphRouter (ICLR 2025)](https://arxiv.org/abs/2410.03834)
**Note:** Requires PyTorch Geometric

#### llmrouter-automix
Automatic model mixing. Dynamically blends responses from multiple models.

```yaml
router_settings:
  routing_strategy: llmrouter-automix
  routing_strategy_args:
    config_path: /app/config/automix_config.yaml
```

**Best for:** Ensemble approaches, quality optimization
**Reference:** [AutoMix (NeurIPS 2024)](https://arxiv.org/abs/2310.12963)

### Multi-Round Routers

#### llmrouter-r1
Pre-trained Router-R1 for multi-turn conversations. Uses reinforcement learning.

```yaml
router_settings:
  routing_strategy: llmrouter-r1
  routing_strategy_args:
    config_path: /app/config/r1_config.yaml
```

**Best for:** Multi-turn conversations, complex dialogues
**Reference:** [Router-R1 (NeurIPS 2025)](https://arxiv.org/abs/2506.09033)
**Note:** Requires vLLM (tested with vllm==0.6.3, torch==2.4.0)

### Personalized Routers

#### llmrouter-gmt
Graph-based personalized router with user preference learning.

```yaml
router_settings:
  routing_strategy: llmrouter-gmt
  routing_strategy_args:
    config_path: /app/config/gmt_config.yaml
```

**Best for:** Per-user model preferences, personalized experiences
**Reference:** [GMTRouter](https://arxiv.org/abs/2511.08590)

### Agentic Routers

#### llmrouter-knn-multiround
KNN-based agentic router for complex multi-step tasks.

```yaml
router_settings:
  routing_strategy: llmrouter-knn-multiround
  routing_strategy_args:
    config_path: /app/config/knn_multiround_config.yaml
```

**Best for:** Agentic workflows, multi-step reasoning

#### llmrouter-llm-multiround
LLM-based agentic router that uses an LLM to decide routing.

```yaml
router_settings:
  routing_strategy: llmrouter-llm-multiround
  routing_strategy_args:
    config_path: /app/config/llm_multiround_config.yaml
```

**Best for:** Complex agentic tasks, meta-reasoning
**Note:** Inference-only (no training required)

### Baseline Routers

#### llmrouter-smallest
Always routes to the smallest model. Useful for cost optimization baseline.

```yaml
router_settings:
  routing_strategy: llmrouter-smallest
  routing_strategy_args:
    config_path: /app/config/baseline_config.yaml
```

**Best for:** Cost minimization, testing, baseline comparison

#### llmrouter-largest
Always routes to the largest model. Useful for quality baseline.

```yaml
router_settings:
  routing_strategy: llmrouter-largest
  routing_strategy_args:
    config_path: /app/config/baseline_config.yaml
```

**Best for:** Maximum quality, testing, baseline comparison

### Custom Routers

#### llmrouter-custom
Load your own trained routing model from the custom routers directory.

```yaml
router_settings:
  routing_strategy: llmrouter-custom
  routing_strategy_args:
    config_path: /app/custom_routers/my_router_config.yaml
```

See [Creating Custom Routers](https://github.com/ulab-uiuc/LLMRouter#-creating-custom-routers) for details.

## Centroid Routing (Zero-Config)

Centroid routing is a NadirClaw-inspired intelligent routing strategy that works **immediately without ML model training**. It uses pre-computed centroid vectors to classify prompts into complexity tiers in approximately **~2ms**, making it ideal as a zero-config default or fallback strategy.

### Overview

Unlike the LLMRouter ML-based strategies above (which require trained models), centroid routing uses pre-computed embedding centroids to perform binary complexity classification via cosine similarity. This provides intelligent routing out of the box:

- **Zero-config**: Works immediately with pre-trained centroid vectors
- **Fast**: ~2ms classification latency (no GPU required)
- **Progressive enhancement**: Sits in the fallback chain behind ML strategies
- **Profile-based**: 5 routing profiles for different cost/quality tradeoffs
- **Context-aware**: Detects agentic and reasoning patterns in prompts
- **Session-persistent**: Maintains routing affinity across conversations

### How It Works

```
Prompt → Embedding → Cosine Similarity → Tier Classification → Model Selection
                     (vs centroids)       (simple/complex)     (per profile)
```

1. **Embedding**: The incoming prompt is converted to a dense vector using a sentence-transformer model
2. **Cosine Similarity**: The embedding is compared against pre-computed centroid vectors for each complexity tier
3. **Tier Classification**: Based on the cosine distance and confidence threshold, the prompt is classified as `simple` or `complex`
4. **Profile Application**: The active routing profile determines which models serve each tier
5. **Model Selection**: A deployment matching the tier is selected from the configured model list

### Progressive Enhancement Chain

Centroid routing integrates into RouteIQ's progressive fallback chain:

```
Pipeline ML Strategies (llmrouter-knn, etc.)
        ↓ (if not configured or fails)
Centroid Routing (~2ms classification)
        ↓ (if disabled or fails)
Random Fallback (simple-shuffle)
```

When `ROUTEIQ_CENTROID_ROUTING=true` (default), centroid routing acts as an intelligent fallback when no ML pipeline strategies are configured. This means RouteIQ provides intelligent routing even in fresh deployments with no trained models.

### Routing Profiles

Five built-in profiles control the cost/quality tradeoff:

| Profile | Description | Simple Tier | Complex Tier |
|---------|-------------|-------------|--------------|
| `auto` | Automatic selection based on complexity | Cheap/fast models | Premium models |
| `eco` | Cost-optimized, upgrades only when necessary | Cheapest available | Mid-tier models |
| `premium` | Always high quality | Premium models | Premium models |
| `free` | Free-tier models only | Free models | Free models |
| `reasoning` | Optimized for math/logic tasks | Standard models | Reasoning-optimized models |

Set the profile via environment variable:
```bash
ROUTEIQ_ROUTING_PROFILE=auto  # default
```

Or via config:
```yaml
router_settings:
  routing_strategy: nadirclaw-centroid
  routing_strategy_args:
    profile: eco
```

### Agentic Detection

The centroid router includes an **AgenticDetector** that identifies tool-use patterns in prompts. When agentic patterns are detected (function calls, tool invocations, multi-step instructions), the router automatically escalates to more capable models that handle tool use well.

Agentic detection uses cumulative scoring across multiple signal types:
- Tool/function call patterns in message content
- System prompt indicators for agent-like behavior
- Multi-step planning language

### Reasoning Detection

A **ReasoningDetector** identifies math, logic, and analytical reasoning tasks using regex-based marker detection. When reasoning markers are found, the router prefers models optimized for step-by-step reasoning (e.g., models with chain-of-thought capabilities).

Detected patterns include:
- Mathematical expressions and equations
- Logical operators and proof language
- Algorithm and data structure references
- Scientific notation and formulas

### Session Persistence

The **SessionCache** provides in-memory conversation affinity with LRU eviction:

- Routes within a conversation session stick to the same model tier
- Prevents mid-conversation model switches that could degrade coherence
- Configurable TTL (default: 1800 seconds / 30 minutes)
- LRU eviction prevents unbounded memory growth

```yaml
router_settings:
  routing_strategy: nadirclaw-centroid
  routing_strategy_args:
    session_ttl: 1800  # 30 minutes
```

### Configuration

#### Zero-Config (Recommended for Getting Started)

No configuration needed. Centroid routing is enabled by default as a fallback:

```bash
# These are the defaults — you don't need to set them
ROUTEIQ_CENTROID_ROUTING=true
ROUTEIQ_ROUTING_PROFILE=auto
```

#### Explicit Primary Strategy

To use centroid routing as the primary (not just fallback) strategy:

```yaml
router_settings:
  routing_strategy: nadirclaw-centroid
  routing_strategy_args:
    centroid_dir: models/centroids
    confidence_threshold: 0.06
    profile: auto
    session_ttl: 1800
    tier_mapping:
      simple:
        - gpt-4o-mini
        - claude-haiku
      complex:
        - gpt-4o
        - claude-sonnet
```

#### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ROUTEIQ_CENTROID_ROUTING` | `true` | Enable centroid routing fallback |
| `ROUTEIQ_ROUTING_PROFILE` | `auto` | Default routing profile |
| `ROUTEIQ_CENTROID_WARMUP` | `false` | Pre-warm classifier at startup |
| `ROUTEIQ_CENTROID_DIR` | `models/centroids` | Directory for centroid `.npy` files |
| `ROUTEIQ_CONFIDENCE_THRESHOLD` | `0.06` | Classification confidence threshold |

#### Tier Mapping

The `tier_mapping` configuration maps complexity tiers to model deployment names. Model names must match entries in `model_list`:

```yaml
tier_mapping:
  simple:           # Low-complexity prompts
    - gpt-4o-mini   # Fast, cheap
    - claude-haiku  # Fast, cheap
  complex:          # High-complexity prompts
    - gpt-4o        # High quality
    - claude-sonnet # High quality
```

If no tier mapping is configured, centroid routing uses heuristics based on model names in the configured `model_list` to auto-assign tiers.

### Performance

- **Classification latency**: ~2ms (CPU only, no GPU required)
- **Memory footprint**: Minimal — centroid vectors are small numpy arrays
- **Startup**: Optional pre-warming via `ROUTEIQ_CENTROID_WARMUP=true`
- **No training required**: Ships with pre-computed centroids

## Model Artifact Security

When using pickle-based models (required for sklearn KNN/SVM/MLP routers), RouteIQ provides manifest-based verification to prevent loading of unauthorized or tampered model files.

### Enabling Pickle Models with Verification

```yaml
# config.yaml
router_settings:
  routing_strategy: llmrouter-knn
  routing_strategy_args:
    model_path: /app/models/knn_router.pkl
```

```bash
# Environment variables
LLMROUTER_ALLOW_PICKLE_MODELS=true
LLMROUTER_MODEL_MANIFEST_PATH=/app/models/manifest.json
LLMROUTER_MODEL_PUBLIC_KEY_B64=<your-ed25519-public-key>
```

When `LLMROUTER_ALLOW_PICKLE_MODELS=true`, manifest verification is automatically enforced unless explicitly bypassed with `LLMROUTER_ENFORCE_SIGNED_MODELS=false`.

See the [Security Guide](security.md#artifact-safety) for complete setup instructions.

## Hot Reloading

All LLMRouter strategies support hot reloading:

```yaml
router_settings:
  routing_strategy_args:
    hot_reload: true
    reload_interval: 300  # Check every 5 minutes
```

The gateway will automatically detect model file changes and reload.

## Training Custom Models

- [MLOps Training Guide](mlops-training.md) - Full training pipeline with Docker
- [Training from Observability Data](observability-training.md) - Use Jaeger/Tempo/CloudWatch traces
- [LLMRouter Data Pipeline](https://github.com/ulab-uiuc/LLMRouter#-preparing-training-data) - Official data preparation guide

## A/B Testing & Runtime Hot-Swapping

RouteIQ supports runtime strategy hot-swapping and A/B testing through a strategy registry and routing pipeline architecture. This allows you to:

- **Switch strategies at runtime** without restarts
- **Run A/B tests** with deterministic weighted selection
- **Fall back gracefully** when primary strategies fail
- **Emit telemetry** for observing routing decisions via OpenTelemetry

### Configuration

Configure A/B testing via environment variables:

```bash
# Set a single active strategy
LLMROUTER_ACTIVE_ROUTING_STRATEGY=llmrouter-knn

# Or configure A/B weights (JSON format)
LLMROUTER_STRATEGY_WEIGHTS='{"baseline": 90, "candidate": 10}'
```

When `LLMROUTER_STRATEGY_WEIGHTS` is set, the gateway performs weighted A/B selection. The weights are relative (not percentages):

- `{"baseline": 90, "candidate": 10}` → 90% baseline, 10% candidate
- `{"a": 1, "b": 1, "c": 1}` → ~33% each

### Deterministic Assignment

A/B selection is **deterministic** based on a hash key, ensuring:

1. **Same user → same variant**: If a `user_id` is present in the request, that user always gets the same strategy variant
2. **Same request → same variant**: If only `request_id` is available, that request is consistently assigned
3. **Reproducible experiments**: The same hash key always produces the same selection

Priority for hash key selection:
1. `metadata.user_id` or `user` in request
2. `metadata.request_id` or `litellm_call_id`
3. Random (no stickiness guarantee)

### Programmatic Configuration

You can also configure A/B testing programmatically:

```python
from litellm_llmrouter import (
    get_routing_registry,
    RoutingStrategy,
    RoutingContext,
)

# Get the global registry
registry = get_routing_registry()

# Custom strategy implementation
class MyCustomStrategy(RoutingStrategy):
    def select_deployment(self, context: RoutingContext):
        # Your custom routing logic
        router = context.router
        # ... analyze context.messages, context.model, etc.
        return deployment_dict  # or None

# Register strategies
registry.register("baseline", ExistingStrategy())
registry.register("candidate", MyCustomStrategy())

# Option 1: Set single active strategy
registry.set_active("baseline")

# Option 2: Configure A/B weights
registry.set_weights({"baseline": 90, "candidate": 10})

# Check current status
status = registry.get_status()
# {
#     "registered_strategies": ["baseline", "candidate"],
#     "active_strategy": None,
#     "ab_weights": {"baseline": 90, "candidate": 10},
#     "ab_enabled": True
# }

# Clear A/B and revert to single strategy
registry.clear_weights()
```

### Telemetry & Observability

When A/B testing is enabled, routing decisions emit telemetry via the `routeiq.router_decision.v1` contract as OpenTelemetry span events:

```json
{
  "contract_name": "routeiq.router_decision.v1",
  "strategy_name": "candidate",
  "selected_deployment": "gpt-4-turbo",
  "selection_reason": "ab_test",
  "custom_attributes": {
    "ab_enabled": true,
    "ab_weights": {"baseline": 90, "candidate": 10},
    "ab_hash_key": "user:abc123..."
  },
  "timings": {
    "total_ms": 2.5
  },
  "outcome": {
    "status": "success"
  }
}
```

Use this telemetry to:
- Compare strategy performance in observability tools (Jaeger, Grafana, etc.)
- Build dashboards showing A/B test results
- Extract data for offline analysis and model retraining

### Fallback Behavior

The routing pipeline supports automatic fallback:

1. **Primary strategy selected** via registry (A/B or active)
2. **If primary fails**, fallback to default strategy
3. **Telemetry marks fallback** for analysis

```python
# Fallback is tracked in routing results
result = pipeline.route(context)
if result.is_fallback:
    print(f"Fell back due to: {result.fallback_reason}")
```

### Disabling Pipeline Routing

To disable pipeline routing and revert to direct LLMRouter calls:

```bash
LLMROUTER_USE_PIPELINE=false
```

This maintains backward compatibility while allowing the new A/B testing features to be opted into.

### Example: Canary Deployment

Run a new routing strategy on 5% of traffic:

```bash
# Step 1: Deploy with small canary weight
LLMROUTER_STRATEGY_WEIGHTS='{"production": 95, "canary-v2": 5}'

# Step 2: Monitor telemetry for errors/latency
# Look for strategy_name="canary-v2" in traces

# Step 3: Gradually increase weight
LLMROUTER_STRATEGY_WEIGHTS='{"production": 80, "canary-v2": 20}'

# Step 4: Full rollout
LLMROUTER_ACTIVE_ROUTING_STRATEGY=canary-v2
unset LLMROUTER_STRATEGY_WEIGHTS
```

### Thread Safety

The registry and pipeline are thread-safe:
- Multiple strategies can be registered concurrently
- Weight updates are atomic
- Selection operations don't block on writes

All configuration updates trigger registered callbacks for integration with admin endpoints or monitoring systems.
