# Intelligent Routing

RouteIQ provides 18+ routing strategies for intelligently directing LLM requests
to the optimal model based on query characteristics, cost, latency, and quality.

## Zero-Config Centroid Routing

RouteIQ ships with **centroid-based routing** that works out of the box with no model
training required. It classifies queries in ~2ms using pre-computed centroid vectors.

```bash
# Set routing profile via environment variable
export ROUTEIQ_ROUTING_PROFILE=auto  # auto | eco | premium | free | reasoning
```

| Profile | Behavior |
|---------|----------|
| `auto` | Balanced quality/cost (default) |
| `eco` | Minimize cost, use cheaper models |
| `premium` | Maximize quality, prefer frontier models |
| `free` | Use only free-tier models |
| `reasoning` | Prefer models with strong reasoning |

## LiteLLM Built-in Strategies

| Strategy | Description |
|----------|-------------|
| `simple-shuffle` | Random load balancing (default) |
| `least-busy` | Route to deployment with fewest active requests |
| `latency-based-routing` | Route based on historical response latency |
| `cost-based-routing` | Route to minimize token costs |
| `usage-based-routing` | Route based on token usage |

## LLMRouter ML Strategies

ML-based strategies from [LLMRouter](https://github.com/ulab-uiuc/LLMRouter):

### Single-Round Routers

| Strategy | Algorithm | Best For |
|----------|-----------|----------|
| `llmrouter-knn` | K-Nearest Neighbors | Quick deployment, interpretable |
| `llmrouter-svm` | Support Vector Machine | Binary decisions, generalization |
| `llmrouter-mlp` | Multi-Layer Perceptron | Complex patterns, high accuracy |
| `llmrouter-mf` | Matrix Factorization | Collaborative filtering |
| `llmrouter-elo` | Elo Rating | Simple preference-based |
| `llmrouter-hybrid` | Probabilistic Hybrid | Ensemble approach |
| `llmrouter-routerdc` | Dual Contrastive (BERT) | Semantic understanding |
| `llmrouter-causallm` | Transformer (GPT-2) | Sequence modeling |
| `llmrouter-graph` | Graph Neural Network | Relationship modeling |
| `llmrouter-automix` | Automatic Mixing | Self-tuning ensemble |

### Multi-Round Routers

| Strategy | Description |
|----------|-------------|
| `llmrouter-r1` | Pre-trained multi-turn router |
| `llmrouter-knn-multiround` | KNN agentic router |
| `llmrouter-llm-multiround` | LLM agentic router |

## Routing Decision Flow

The following diagram shows how a request is routed from arrival through
governance, guardrails, capability detection, profile selection, and
personalized re-ranking to the final LLM deployment.

```mermaid
flowchart TD
    REQ[Request arrives] --> GOV{Governance<br/>check}
    GOV -->|Denied| R403[403 Forbidden]
    GOV -->|Allowed| GUARD{Input<br/>guardrails}
    GUARD -->|Denied| R446[446 Guardrail Denied]
    GUARD -->|Passed| CAP[Detect required<br/>capabilities]
    CAP --> FILTER[Filter deployments<br/>by capabilities]
    FILTER --> SESSION{Session<br/>cache hit?}
    SESSION -->|Hit| CACHED[Return cached model]
    SESSION -->|Miss| PROFILE{Routing<br/>profile?}
    PROFILE -->|eco| COST[Sort by cost<br/>cheapest first]
    PROFILE -->|premium| QUALITY[Sort by quality<br/>best first]
    PROFILE -->|auto| CENTROID[Centroid<br/>classify tier]
    CENTROID --> VISION{Vision<br/>content?}
    VISION -->|Yes, non-vision model| SWAP[Swap to vision model]
    VISION -->|No or already vision| CTX{Context<br/>window OK?}
    SWAP --> CTX
    CTX -->|Exceeds| UPGRADE[Try larger model]
    CTX -->|Fits| PERS{Personalized<br/>routing enabled?}
    UPGRADE --> PERS
    PERS -->|Yes + user_id| RERANK[Re-rank by<br/>user preference]
    PERS -->|No| SELECT[Select deployment]
    RERANK --> SELECT
    COST --> SELECT
    QUALITY --> SELECT
    SELECT --> CACHE_STORE[Store in session cache]
    CACHE_STORE --> LLM[Forward to LLM]
```

## Configuration

```yaml
router_settings:
  routing_strategy: llmrouter-knn
  routing_strategy_args:
    model_path: /app/models/knn_router.pt
    llm_data_path: /app/config/llm_candidates.json
    hot_reload: true
    reload_interval: 300
```

## A/B Testing

RouteIQ supports runtime A/B testing between routing strategies:

```python
from litellm_llmrouter import get_routing_registry

registry = get_routing_registry()
registry.set_weights({"baseline": 90, "candidate": 10})
```

## Training Custom Models

The MLOps pipeline supports training custom routing models:

```bash
# Extract telemetry data
python examples/mlops/scripts/extract_traces.py

# Train a KNN model
python examples/mlops/scripts/train.py --config examples/mlops/configs/knn.yaml

# Deploy the model
python examples/mlops/scripts/deploy.py --model-path models/knn_router.pt
```

See [MLOps Training](../operations/observability.md) for the full training pipeline.
