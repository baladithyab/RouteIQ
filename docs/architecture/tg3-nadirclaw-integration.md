# TG3: NadirClaw Integration Strategy

## Executive Summary

RouteIQ's existing 18 ML routing strategies (KNN, SVM, MLP, MF, ELO, hybrid, etc.)
deliver excellent routing quality — once trained on user telemetry data. The critical
gap is the **cold-start problem**: new users receive zero intelligent routing until they
have collected production traces, run the MLOps training pipeline, and deployed a trained
model. This document designs how NadirClaw's centroid-based routing patterns integrate
into RouteIQ to deliver **intelligent routing from day one** while preserving the
path to ML-trained strategy superiority.

**Key design decisions:**

1. **Centroid-Based Zero-Config Routing** — Pre-trained centroids ship with the
   `routeiq-router` package, providing immediate prompt-to-model routing via embedding
   similarity with < 5ms latency overhead.
2. **Agentic/Reasoning Detection** — A `CustomLogger` pre-routing hook classifies
   prompts to detect agentic patterns, multi-step reasoning, and code generation before
   the routing strategy runs.
3. **Routing Profiles** — User-selectable presets (`auto`, `eco`, `premium`, `free`,
   `reasoning`) that combine strategy selection with cost/quality constraints.
4. **Session Persistence** — Extends the existing `ConversationAffinityTracker` to
   support conversation-level model affinity (not just provider affinity).
5. **Multi-Tier Complexity Classification** — Five tiers from trivial → expert replace
   binary simple/complex, each mapping to configurable model selections.
6. **Unified Routing Pipeline** — A six-stage pipeline that orchestrates all components
   in sequence, with progressive enhancement as users graduate from centroids to
   trained ML models.

---

## The Cold-Start Problem

### Current State

RouteIQ inherits UIUC LLMRouter's 18 ML strategies, all registered in
[`LLMROUTER_STRATEGIES`](src/litellm_llmrouter/strategies.py:460) and integrated via
[`LLMRouterStrategyFamily`](src/litellm_llmrouter/strategies.py:558). The KNN inference
path in [`InferenceKNNRouter`](src/litellm_llmrouter/strategies.py:189) demonstrates the
typical flow:

```
User prompt → sentence-transformers encode → sklearn KNN predict → model label
```

This requires a pre-trained `.pkl` model produced by the MLOps pipeline in
`examples/mlops/`. The training pipeline needs:

1. **Telemetry data** — Jaeger traces from production usage
2. **Training compute** — Running `train_router.py` with extracted traces
3. **Deployment** — Placing the `.pkl` file at `LLMROUTER_MODEL_PATH`

**For a new deployment, this means days to weeks before intelligent routing activates.**

### What NadirClaw Solves

NadirClaw's centroid-based approach provides routing intelligence from the first request:

| Aspect | Current (ML-trained) | NadirClaw (Centroids) |
|--------|---------------------|----------------------|
| Time to first route | Days–weeks | Immediate |
| Setup required | MLOps pipeline, training data | Zero |
| Accuracy | High (domain-specific) | Good (general-purpose) |
| Data dependency | Requires production traces | Pre-trained centroids |
| Improvement path | Retrain with more data | Refine centroids + graduate to ML |

The strategic insight is that centroids and ML strategies are **complementary, not
competitive**. Centroids solve cold-start; ML models optimize for production traffic
patterns.

---

## Centroid-Based Zero-Config Routing

### How Centroids Work

A centroid is a representative embedding vector for a category of prompts. Instead
of training a full KNN/SVM/MLP model, we pre-compute mean embedding vectors for
common prompt categories and store them alongside model affinity scores.

```
     User Prompt
          |
          v
   +--------------+
   | Embed prompt  |   sentence-transformers/all-MiniLM-L6-v2
   | (384-dim vec) |   (reuses existing KNN embedding model)
   +--------------+
          |
          v
   +--------------+
   | Cosine sim    |   Compare against 8-16 category centroids
   | to centroids  |   Single matrix multiply: O(n_centroids)
   +--------------+
          |
          v
   +--------------+
   | Lookup model  |   category → model affinity scores
   | affinity      |   Select best model for this category
   +--------------+
          |
          v
   Selected Model
```

### Pre-Trained Centroid Design

Ship 8 core category centroids with the package, each derived from public benchmark
datasets (MMLU, HumanEval, MT-Bench, etc.):

| Category | Description | Typical Best Model Tier |
|----------|-------------|------------------------|
| `code_generation` | Code writing, debugging, refactoring | premium |
| `creative_writing` | Stories, poetry, marketing copy | premium |
| `analysis` | Data analysis, comparison, evaluation | mid |
| `summarization` | Condensing documents, TL;DR | budget |
| `translation` | Language conversion, localization | mid |
| `conversation` | Casual chat, simple Q&A | budget |
| `reasoning` | Logic, math, multi-step analysis | premium |
| `tool_use` | Function calling, API interaction, agentic | premium |

Additional optional centroids (loaded from config, not shipped):

| Category | Description |
|----------|-------------|
| `medical` | Healthcare domain queries |
| `legal` | Legal analysis and contract review |
| `financial` | Financial modeling and analysis |
| `scientific` | Research and scientific reasoning |

### Centroid Data Format and Shipping

**Centroid file format**: JSON (no pickle security concerns, human-auditable):

```json
{
  "version": "1.0.0",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "embedding_dim": 384,
  "created_at": "2026-01-15T00:00:00Z",
  "categories": {
    "code_generation": {
      "centroid": [0.0231, -0.0456, ...],
      "model_affinity": {
        "premium": 0.92,
        "mid": 0.75,
        "budget": 0.45
      },
      "sample_count": 5000,
      "description": "Source code generation, debugging, refactoring"
    },
    "conversation": {
      "centroid": [-0.0123, 0.0567, ...],
      "model_affinity": {
        "premium": 0.60,
        "mid": 0.85,
        "budget": 0.90
      },
      "sample_count": 8000,
      "description": "Casual chat, simple Q&A, greetings"
    }
  }
}
```

**Shipping strategy:**

```
routeiq-router/
  routeiq/
    data/
      centroids/
        v1.0.0/
          centroids.json          # 8 core centroids (~50KB)
          centroids.npy           # Precomputed numpy array (8x384 float32 = 12KB)
          model_affinity.json     # Category → tier affinity map
          metadata.json           # Version, embedding model, checksums
```

The centroids are included as package data via `pyproject.toml`:

```toml
[tool.setuptools.package-data]
routeiq = ["data/centroids/**/*.json", "data/centroids/**/*.npy"]
```

**Total package size impact**: ~100KB (negligible compared to sentence-transformers
model download of ~90MB).

### Embedding Strategy

**Reuse `sentence-transformers/all-MiniLM-L6-v2`** already used by
[`InferenceKNNRouter`](src/litellm_llmrouter/strategies.py:132):

- 384-dimensional embeddings
- ~90MB model download (cached after first use)
- ~2ms encode time on CPU for a single prompt
- Already lazy-loaded via [`_get_sentence_transformer()`](src/litellm_llmrouter/strategies.py:93)
  with thread-safe singleton

**Performance budget for centroid routing (target < 5ms):**

| Step | Time |
|------|------|
| Prompt embedding (cached model) | ~2ms |
| Cosine similarity (8 centroids x 384-dim) | ~0.1ms |
| Model affinity lookup | ~0.01ms |
| **Total** | **~2.1ms** |

For deployments where sentence-transformers is too heavy, support a lighter
fallback:

```python
CENTROID_EMBEDDING_BACKENDS = {
    "sentence-transformers": {  # Default, highest quality
        "model": "all-MiniLM-L6-v2",
        "dim": 384,
        "encode_ms": 2,
    },
    "onnx-minilm": {  # Lighter alternative, no PyTorch needed
        "model": "all-MiniLM-L6-v2-onnx",
        "dim": 384,
        "encode_ms": 1,
    },
    "tfidf-hash": {  # Zero-dependency fallback
        "model": "builtin-tfidf-256",
        "dim": 256,
        "encode_ms": 0.5,
    },
}
```

### Progressive Refinement Path

Users can refine centroids with their own data without running the full MLOps pipeline:

```
Stage 0: Zero-config (shipped centroids)
    |
    v
Stage 1: User adds custom model affinity overrides
    |     (config.yaml: centroid_overrides.code_generation.premium = 0.95)
    |
    v
Stage 2: User provides custom centroids from their data
    |     (CENTROID_PATH=/app/data/custom_centroids.json)
    |
    v
Stage 3: Centroid auto-refinement from production traces
    |     (background task computes running mean of per-category embeddings)
    |
    v
Stage 4: Graduate to trained ML model (KNN/MLP/SVM)
          (centroid becomes fast pre-filter, ML model makes final decision)
```

### Expected Quality vs Trained Models

Based on routing benchmark literature and centroid classification accuracy:

| Strategy | Cold-Start Quality | Warm Quality | Latency |
|----------|-------------------|--------------|---------|
| Random (no routing) | 50% | 50% | 0ms |
| **Centroid (NadirClaw)** | **72–78%** | **72–78%** | **2ms** |
| KNN (trained, 1K samples) | N/A (needs data) | 82–86% | 5ms |
| MLP (trained, 10K samples) | N/A (needs data) | 87–91% | 3ms |
| Hybrid (centroid + KNN) | 72–78% | 85–89% | 7ms |

Quality is measured as: percentage of prompts routed to the model tier that a human
evaluator would choose. Centroids trade 10–15% accuracy for zero setup cost.

---

## Agentic and Reasoning Detection

### Detection Signals and Heuristics

A pre-classification step identifies prompt characteristics before routing:

**Structural signals (from message format):**

| Signal | Detection Method | Classification |
|--------|-----------------|----------------|
| `tools` parameter present | Check `request_kwargs` | `agentic` |
| `tool_choice` specified | Check `request_kwargs` | `agentic` |
| `function_call` in messages | Scan message content | `agentic` |
| System prompt mentions "agent" | Regex on system message | `agentic` |
| Multi-turn with tool results | Count `tool` role messages | `agentic` |
| `response_format: json_schema` | Check `request_kwargs` | `structured` |

**Content signals (from prompt text):**

| Signal | Detection Method | Classification |
|--------|-----------------|----------------|
| Code blocks (triple backtick) | Regex scan | `code` |
| "step by step" / "think through" | Keyword detection | `reasoning` |
| "write a story" / "compose" | Keyword + centroid | `creative` |
| Short factual question | Length < 50 chars + question mark | `factual` |
| Multi-paragraph instruction | Length > 500 chars | `complex` |

**Heuristic scoring:**

```python
@dataclass
class PromptClassification:
    primary_type: str          # e.g., "agentic", "reasoning", "code"
    confidence: float          # 0.0 - 1.0
    complexity_tier: int       # 0-4 (trivial to expert)
    detected_signals: list     # e.g., ["tools_present", "multi_turn"]
    requires_reasoning: bool   # Route to reasoning-capable model?
    requires_tool_use: bool    # Route to tool-use-capable model?
```

### Classification Pipeline

```
                    Request
                       |
                       v
              +----------------+
              | Structural     |  Check tools, function_call, system prompt
              | Signal Scan    |  Cost: ~0.1ms (no embedding needed)
              +----------------+
                       |
                       v
              +----------------+
              | Content        |  Keyword detection, length analysis
              | Heuristics     |  Cost: ~0.2ms
              +----------------+
                       |
                       v
              +----------------+
              | Centroid        |  Only if structural/heuristic signals
              | Confirmation   |  are ambiguous. Cost: ~2ms
              +----------------+
                       |
                       v
              PromptClassification
```

The pipeline is designed to short-circuit early. If structural signals clearly
indicate `agentic` (tools present), we skip content analysis and centroid matching.

### Integration via pre_routing_hook

LiteLLM provides `async_pre_routing_hook` on `CustomLogger` and
[`PreRoutingHookResponse`](reference/litellm/litellm/types/router.py:843) which can
modify `model` and `messages` before routing.

```python
class NadirClawPreClassifier(CustomLogger):
    """Pre-routing classifier that detects prompt characteristics."""

    async def async_pre_routing_hook(
        self,
        user_api_key_dict,
        cache,
        data: dict,
        call_type: str,
    ) -> PreRoutingHookResponse | None:
        messages = data.get("messages", [])
        request_kwargs = data.get("request_kwargs", {})

        # Classify the prompt
        classification = self.classify(messages, request_kwargs)

        # Inject classification into metadata for downstream routing
        metadata = data.setdefault("metadata", {})
        metadata["nadirclaw_classification"] = asdict(classification)

        # If agentic and model does not support tools, redirect
        if classification.requires_tool_use:
            return PreRoutingHookResponse(
                model=self._get_tool_capable_model(data["model"]),
                messages=messages,
            )

        return None
```

### Model Capability Matrix

| Model | Tool Use | Reasoning | Code Gen | Vision | Extended Thinking |
|-------|----------|-----------|----------|--------|-------------------|
| Claude 4.5 Opus | Yes | Yes | Yes | Yes | Yes |
| Claude 4.5 Sonnet | Yes | Yes | Yes | Yes | No |
| Claude 4.5 Haiku | Limited | No | Yes | No | No |
| Nova Pro | Yes | Yes | Yes | Yes | No |
| Nova Lite | Limited | No | Limited | Yes | No |
| Nova Micro | No | No | No | No | No |

This matrix is stored in [`llm_candidates.json`](config/llm_candidates.json) which
already has a `capabilities` field per model. The agentic detection system reads this
to determine which models can handle classified prompt types.

**Fallback behavior**: If classified as `agentic` but no tool-capable model is
available in the deployment's model list, the classifier:

1. Logs a warning with the classification details
2. Falls through to the standard routing pipeline
3. Sets `metadata.nadirclaw_fallback = true` for observability

---

## Routing Profiles

### Profile Definitions

Each profile combines a strategy preference with cost/quality/latency constraints:

#### `auto` (Default)

Best-quality model for the detected prompt type. The routing pipeline runs in full:
classification → tier detection → strategy routing → cost filtering.

```yaml
profile: auto
strategy: centroid   # or trained ML strategy if available
cost_cap: null       # no cost limit
quality_floor: 0.0   # accept any quality tier
latency_cap: null    # no latency limit
tier_mapping: default
```

#### `eco`

Cheapest model that meets a minimum quality threshold. Uses
[`CostAwareRoutingStrategy`](src/litellm_llmrouter/strategies.py:1094) with aggressive
cost weighting.

```yaml
profile: eco
strategy: cost-aware
cost_cap: 0.005      # max $0.005 per 1K tokens
quality_floor: 0.7   # minimum acceptable quality
latency_cap: null
cost_weight: 0.85    # strongly favor cheaper models
tier_mapping:
  0: budget          # trivial → cheapest
  1: budget          # simple → cheapest
  2: budget          # moderate → still budget
  3: mid             # complex → mid-tier only
  4: mid             # expert → mid-tier (best eco can do)
```

#### `premium`

Always use the best available model regardless of cost.

```yaml
profile: premium
strategy: centroid   # or ML strategy, but override to top tier
cost_cap: null       # no cost limit
quality_floor: 0.9   # only top-tier models
latency_cap: null
tier_mapping:
  0: premium         # even trivial → premium
  1: premium
  2: premium
  3: premium
  4: premium
```

#### `free`

Only use free or open-source models (Llama, Mixtral, Mistral, etc.).

```yaml
profile: free
strategy: centroid
cost_cap: 0.0        # $0 — only free models
quality_floor: 0.0
latency_cap: null
model_filter:
  - "ollama/*"
  - "groq/*"         # free-tier models
  - "together_ai/meta-llama/*"
  - "together_ai/mistralai/*"
tier_mapping: default
```

#### `reasoning`

Route to reasoning-capable models for extended thinking tasks.

```yaml
profile: reasoning
strategy: centroid
cost_cap: null
quality_floor: 0.85
latency_cap: null
model_filter:
  capabilities_required: ["reasoning", "extended-thinking"]
tier_mapping:
  0: mid             # even trivial gets reasoning model
  1: mid
  2: premium
  3: premium
  4: premium         # expert → best reasoning model
```

### Profile Selection Mechanism

Profiles can be selected through multiple channels (in priority order):

1. **Request header**: `X-RouteIQ-Profile: eco`
2. **Request metadata**: `{"metadata": {"routing_profile": "eco"}}`
3. **Per-team default**: Configured in team settings via admin API
4. **Per-key default**: Associated with API key configuration
5. **Global default**: `ROUTEIQ_DEFAULT_PROFILE=auto` environment variable

```python
def resolve_profile(request_kwargs: dict) -> RoutingProfile:
    """Resolve the routing profile for this request."""
    # 1. Explicit request header
    headers = request_kwargs.get("headers", {})
    if "x-routeiq-profile" in headers:
        return get_profile(headers["x-routeiq-profile"])

    # 2. Request metadata
    metadata = request_kwargs.get("metadata", {})
    if "routing_profile" in metadata:
        return get_profile(metadata["routing_profile"])

    # 3. Per-team default
    team_id = metadata.get("team_id")
    if team_id:
        team_profile = get_team_profile(team_id)
        if team_profile:
            return team_profile

    # 4. Per-key default
    api_key = metadata.get("api_key_alias")
    if api_key:
        key_profile = get_key_profile(api_key)
        if key_profile:
            return key_profile

    # 5. Global default
    return get_profile(os.getenv("ROUTEIQ_DEFAULT_PROFILE", "auto"))
```

### Custom Profile Definition

Users can define custom profiles in `config.yaml`:

```yaml
routing_profiles:
  my_custom_profile:
    base: eco                    # inherit from eco
    cost_cap: 0.002              # override cost cap
    quality_floor: 0.75          # override quality floor
    tier_mapping:
      3: premium                 # upgrade complex tier to premium
      4: premium
    model_filter:
      providers: ["bedrock"]     # only AWS Bedrock models
```

### Profile-Strategy Mapping

```
Profile       Strategy Used          Cost Filter    Quality Filter
-------       ----------------       -----------    --------------
auto          centroid/ML/hybrid     none           none
eco           cost-aware             aggressive     min 0.7
premium       centroid (top-tier)    none           min 0.9
free          centroid (free only)   $0             none
reasoning     centroid (reasoning)   none           min 0.85
custom        configurable           configurable   configurable
```

---

## Session Persistence

### Session Identification

RouteIQ already has [`ConversationAffinityTracker`](src/litellm_llmrouter/conversation_affinity.py:86)
which maps `response_id` → `provider_deployment` for the Responses API. NadirClaw
extends this with conversation-level model affinity:

**Session ID resolution (priority order):**

1. `X-RouteIQ-Session-Id` header (explicit)
2. `metadata.conversation_id` in request body
3. `metadata.thread_id` in request body
4. `previous_response_id` (existing Responses API support)
5. No session — fresh routing decision

```python
def resolve_session_id(request_kwargs: dict) -> str | None:
    """Extract session ID from request."""
    headers = request_kwargs.get("headers", {})
    if sid := headers.get("x-routeiq-session-id"):
        return f"session:{sid}"

    metadata = request_kwargs.get("metadata", {})
    if cid := metadata.get("conversation_id"):
        return f"conv:{cid}"
    if tid := metadata.get("thread_id"):
        return f"thread:{tid}"

    # Existing Responses API support
    if prev_id := request_kwargs.get("previous_response_id"):
        return f"resp:{prev_id}"

    return None
```

### Affinity Storage Design

Extend the existing [`AffinityRecord`](src/litellm_llmrouter/conversation_affinity.py:59)
to support model-level affinity (not just provider):

```python
@dataclass(frozen=True)
class ModelAffinityRecord(AffinityRecord):
    """Extended affinity record with model-level routing persistence."""
    selected_model: str           # e.g., "claude-4.5-sonnet"
    selected_tier: str            # e.g., "premium"
    routing_profile: str          # e.g., "auto"
    classification_type: str      # e.g., "code_generation"
    request_count: int = 1        # number of requests in this session
```

**Storage key format:**

| Backend | Key Pattern | TTL |
|---------|-------------|-----|
| Redis | `routeiq:session:{session_id}` | Configurable (default 30 min) |
| In-memory | Same key in `_store` dict | Same TTL |

### Cross-Replica Consistency

The existing `ConversationAffinityTracker` already supports Redis backend with
automatic fallback to in-memory. The same approach applies:

- **Redis available**: All replicas share session state. Session created on replica A
  is immediately visible on replica B.
- **Redis unavailable**: Each replica maintains its own in-memory store. Sessions are
  sticky to the replica that first handled the conversation (requires sticky load
  balancing or accept some inconsistency).

### TTL and Expiry Strategy

```
+- Session start (first message) ------+
|  TTL: 30 minutes (sliding window)    |
|                                       |
|  Message 2 → TTL resets to 30 min    |
|  Message 3 → TTL resets to 30 min    |
|  (idle 30 min)                        |
|                                       |
+- Session expires, next msg = fresh ---+
```

Configuration:

```yaml
session_persistence:
  enabled: true                    # ROUTEIQ_SESSION_PERSISTENCE_ENABLED
  ttl_seconds: 1800                # 30 minutes, ROUTEIQ_SESSION_TTL
  max_entries: 50000               # In-memory LRU cap
  sliding_window: true             # Reset TTL on each request
  allow_override: true             # Allow explicit model override within session
```

**Override behavior**: If a user sends `model: "claude-4.5-opus"` explicitly within
a session, the explicit request is honored and the session affinity is updated to the
new model.

---

## Multi-Tier Complexity Classification

### Tier Definitions

| Tier | Name | Prompt Characteristics | Model Selection |
|------|------|----------------------|-----------------|
| 0 | **Trivial** | Greetings, yes/no, single-word answers, status checks | Cheapest/fastest (Nova Micro, Haiku) |
| 1 | **Simple** | Basic Q&A, short summaries, translations < 200 words | Good-enough model (Nova Lite, Haiku) |
| 2 | **Moderate** | Multi-paragraph analysis, code explanation, comparison | Mid-tier (Nova Pro, Sonnet) |
| 3 | **Complex** | Multi-step reasoning, code generation, creative writing | Top-tier (Sonnet, limited Opus) |
| 4 | **Expert** | Agentic workflows, multi-tool orchestration, extended reasoning | Best available (Opus with extended thinking) |

### Classification Method

**Hybrid approach**: Fast heuristics first, centroid confirmation for ambiguous cases.

**Stage 1: Heuristic features (< 0.5ms)**

```python
def heuristic_tier(messages: list, request_kwargs: dict) -> int | None:
    """Fast heuristic tier classification. Returns None if ambiguous."""
    prompt = extract_last_user_message(messages)
    prompt_len = len(prompt)
    word_count = len(prompt.split())

    # Tier 4: Expert (structural signals dominate)
    if request_kwargs.get("tools") and len(request_kwargs["tools"]) > 3:
        return 4
    if any(r in prompt.lower() for r in ["orchestrate", "multi-step", "agent"]):
        return 4

    # Tier 0: Trivial
    if word_count <= 5 and any(g in prompt.lower() for g in
            ["hello", "hi", "hey", "thanks", "ok", "yes", "no"]):
        return 0
    if prompt_len < 20:
        return 0

    # Tier 1: Simple
    if word_count <= 20 and "?" in prompt:
        return 1
    if prompt_len < 100 and not any(kw in prompt.lower() for kw in
            ["explain", "analyze", "compare", "write", "create", "implement"]):
        return 1

    # Tier 3: Complex (keyword signals)
    if any(kw in prompt.lower() for kw in
            ["implement", "write a program", "design", "architect"]):
        return 3
    if prompt_len > 500:
        return 3

    # Ambiguous — fall through to centroid classification
    return None
```

**Stage 2: Centroid-based tier classification (< 2ms)**

For ambiguous prompts, compute embedding similarity to tier-specific centroids.
Each tier has its own set of representative centroids:

```python
def centroid_tier(embedding: np.ndarray, tier_centroids: dict) -> int:
    """Classify tier using centroid similarity."""
    best_tier = 2  # default: moderate
    best_sim = -1.0

    for tier, centroid_vec in tier_centroids.items():
        sim = cosine_similarity(embedding, centroid_vec)
        if sim > best_sim:
            best_sim = sim
            best_tier = tier

    return best_tier
```

### Tier-to-Model Mapping

The mapping is configurable per deployment via `config.yaml`:

```yaml
routing_tier_mapping:
  # Default mapping (used by 'auto' profile)
  default:
    0: ["nova-micro", "nova-lite"]           # Trivial
    1: ["nova-lite", "claude-4.5-haiku"]     # Simple
    2: ["nova-pro", "claude-4-sonnet"]       # Moderate
    3: ["claude-4.5-sonnet", "claude-4-sonnet"]  # Complex
    4: ["claude-4.5-opus", "claude-4.5-sonnet"]  # Expert

  # Eco profile overrides
  eco:
    0: ["nova-micro"]
    1: ["nova-micro", "nova-lite"]
    2: ["nova-lite", "claude-4.5-haiku"]
    3: ["nova-pro", "claude-4-sonnet"]
    4: ["claude-4-sonnet"]
```

The tier mapping resolves to the first available model in the list. If the first
choice is unavailable (circuit breaker open, rate limited), fall through to the next.

### Override Mechanism

Users can force a specific tier via:

1. **Request metadata**: `{"metadata": {"routing_tier": 3}}`
2. **Model suffix**: `model: "auto:tier3"` (parsed in pre-routing hook)
3. **Profile preset**: The profile's `tier_mapping` effectively overrides defaults

---

## Unified Routing Pipeline

### Pipeline Architecture

```
  Request arrives
       |
       v
  +---------------------------+
  | Stage 1: Pre-Classification|   NadirClawPreClassifier (CustomLogger)
  | - Structural signal scan   |   Detects: agentic, reasoning, code, etc.
  | - Content heuristics       |   Injects classification into metadata
  | - Capability gating        |   May redirect model if capabilities needed
  +---------------------------+
       |
       v
  +---------------------------+
  | Stage 2: Profile Resolution|   Resolve routing profile from headers/
  | - Header / metadata / team |   metadata / team config / global default
  | - Load profile constraints |
  +---------------------------+
       |
       v
  +---------------------------+
  | Stage 3: Session Check     |   Check ConversationAffinityTracker
  | - Lookup session affinity  |   If found and not overridden:
  | - If found, short-circuit  |   return cached model selection
  +---------------------------+       |
       |                               v (short-circuit to Stage 6)
       v
  +---------------------------+
  | Stage 4: Tier Classification|  Heuristic + centroid tier assignment
  | - Fast heuristics (< 0.5ms)|  Tier 0-4 classification
  | - Centroid fallback (< 2ms)|  Uses prompt embedding
  +---------------------------+
       |
       v
  +---------------------------+
  | Stage 5: Strategy Routing  |   Delegates to RoutingPipeline
  | - Centroid routing         |   Uses strategy_registry A/B selection
  | - OR trained ML (KNN/MLP)  |   Falls back to default on error
  | - OR cost-aware Pareto     |
  +---------------------------+
       |
       v
  +---------------------------+
  | Stage 6: Cost/Quality      |   Profile-based filtering
  | Filtering                  |   - Apply cost cap
  | - Apply profile constraints|   - Apply quality floor
  | - Tier-to-model mapping    |   - Select from tier candidates
  | - Circuit breaker check    |
  +---------------------------+
       |
       v
  Selected Deployment
  (record session affinity)
```

### Stage 1: Pre-Classification

Implemented as a `CustomLogger` with `async_pre_routing_hook`:

```python
class NadirClawPreClassifier:
    def classify(self, messages, request_kwargs) -> PromptClassification:
        # 1. Structural signals (tools, function_call)
        # 2. Content heuristics (keywords, length)
        # 3. Centroid confirmation (only if ambiguous)
        ...
```

**Integration point**: Registered in `config.yaml` under `litellm_settings.callbacks`
or loaded by the plugin system in
[`gateway/app.py`](src/litellm_llmrouter/gateway/app.py).

### Stage 2: Profile Resolution

Resolves the active routing profile for this request. Profile is injected into the
[`RoutingContext.metadata`](src/litellm_llmrouter/strategy_registry.py:259) so
downstream stages can access it.

### Stage 3: Session Check

Queries [`ConversationAffinityTracker`](src/litellm_llmrouter/conversation_affinity.py:86)
for existing session affinity. If a valid, non-expired record exists and the request
does not include an explicit model override, the pipeline short-circuits to Stage 6
with the cached model selection.

### Stage 4: Tier Classification

Combines heuristic and centroid-based tier assignment. The tier is stored in
`RoutingContext.metadata["complexity_tier"]` for use in Stage 6.

### Stage 5: Strategy Routing

Delegates to the existing [`RoutingPipeline`](src/litellm_llmrouter/strategy_registry.py:1121)
which handles strategy selection (single or A/B) and execution. The strategy pool
now includes:

| Strategy | Registration Name | When Available |
|----------|------------------|----------------|
| Centroid routing | `nadirclaw-centroid` | Always (ships with package) |
| KNN inference | `llmrouter-knn` | After model training |
| MLP routing | `llmrouter-mlp` | After model training |
| Cost-aware Pareto | `llmrouter-cost-aware` | Always |
| Hybrid (centroid + ML) | `nadirclaw-hybrid` | After model training |

### Stage 6: Cost/Quality Filtering

Applies profile-specific constraints and the tier-to-model mapping:

```python
def filter_and_select(
    candidates: list,
    profile: RoutingProfile,
    tier: int,
    classification: PromptClassification,
) -> dict | None:
    # 1. Filter by profile model_filter
    # 2. Filter by cost cap
    # 3. Filter by quality floor
    # 4. Filter by tier mapping
    # 5. Filter by capability requirements
    # 6. Select best remaining candidate
    ...
```

### Progressive Enhancement Path

```
Day 0           Week 1           Month 1          Month 3+
  |                |                 |                |
  v                v                 v                v
Centroid      Centroid +       Centroid +        Trained ML
routing       refined          auto-refined      (KNN/MLP)
only          affinity maps    centroids from    + centroid
              from config      production        pre-filter
                               traces
```

The A/B testing framework in [`RoutingStrategyRegistry`](src/litellm_llmrouter/strategy_registry.py:493)
enables safe comparison at each transition:

```yaml
# A/B test centroid vs newly trained KNN
routing_strategy_weights:
  nadirclaw-centroid: 90
  llmrouter-knn: 10

# Once KNN proves superior, ramp up
routing_strategy_weights:
  nadirclaw-centroid: 20
  llmrouter-knn: 80
```

---

## Implementation Plan

### Phase 1: Centroid Routing + Profiles

**Scope**: Core centroid routing strategy, profile system, basic tier classification

**Deliverables**:
- `CentroidRoutingStrategy(RoutingStrategy)` implementation in `strategies.py`
- Pre-trained centroid data files in `routeiq/data/centroids/`
- Centroid loading, caching, and similarity computation
- `RoutingProfile` data model and resolution logic
- 5 built-in profiles (auto, eco, premium, free, reasoning)
- Custom profile definition via config.yaml
- Heuristic tier classification (fast path)
- Unit tests for centroid routing, profile resolution, tier classification
- Integration with existing `RoutingPipeline` and `RoutingStrategyRegistry`

**Files to create/modify**:
- `src/litellm_llmrouter/centroid_router.py` (new)
- `src/litellm_llmrouter/routing_profiles.py` (new)
- `src/litellm_llmrouter/tier_classifier.py` (new)
- `src/litellm_llmrouter/strategies.py` (register new strategies)
- `routeiq/data/centroids/v1.0.0/` (new, package data)
- `tests/unit/test_centroid_router.py` (new)
- `tests/unit/test_routing_profiles.py` (new)
- `tests/unit/test_tier_classifier.py` (new)

### Phase 2: Session Persistence + Multi-Tier

**Scope**: Extend conversation affinity for model-level sessions, full
centroid-based tier classification

**Deliverables**:
- Extend `ConversationAffinityTracker` with `ModelAffinityRecord`
- Session ID resolution from multiple sources
- Sliding-window TTL with configurable timeout
- Centroid-based tier classification (Stage 2, for ambiguous prompts)
- Tier-to-model mapping with configurable overrides
- Tier override mechanism via request metadata
- Integration tests for session persistence across requests

**Files to create/modify**:
- `src/litellm_llmrouter/conversation_affinity.py` (extend)
- `src/litellm_llmrouter/tier_classifier.py` (add centroid tier)
- `config/config.yaml` (add tier_mapping, session config)
- `tests/unit/test_session_persistence.py` (new)
- `tests/integration/test_session_routing.py` (new)

### Phase 3: Agentic Detection + Pipeline Integration

**Scope**: Pre-routing hook for agentic/reasoning detection, unified six-stage
pipeline, A/B testing between centroid and trained strategies

**Deliverables**:
- `NadirClawPreClassifier` implementing `CustomLogger.async_pre_routing_hook`
- `PromptClassification` data model
- Model capability matrix integrated with `llm_candidates.json`
- Unified routing pipeline orchestrating all six stages
- A/B testing configuration for centroid vs ML strategies
- Progressive refinement path (auto-centroid updates from traces)
- Hybrid strategy (centroid pre-filter + ML final decision)
- End-to-end integration tests
- Documentation updates to `docs/routing-strategies.md`

**Files to create/modify**:
- `src/litellm_llmrouter/prompt_classifier.py` (new)
- `src/litellm_llmrouter/nadirclaw_pipeline.py` (new, unified pipeline)
- `src/litellm_llmrouter/centroid_router.py` (add hybrid mode)
- `config/llm_candidates.json` (add capability matrix fields)
- `docs/routing-strategies.md` (update with NadirClaw strategies)
- `tests/unit/test_prompt_classifier.py` (new)
- `tests/unit/test_nadirclaw_pipeline.py` (new)
- `tests/integration/test_nadirclaw_e2e.py` (new)

---

## Appendix: Centroid Training Methodology

### Data Sources for Pre-Trained Centroids

| Category | Dataset | Samples | License |
|----------|---------|---------|---------|
| code_generation | HumanEval, MBPP, CodeContests | 5,000 | MIT/Apache |
| creative_writing | MT-Bench creative subset | 3,000 | MIT |
| analysis | MMLU analysis questions | 5,000 | MIT |
| summarization | CNN/DailyMail, XSum | 5,000 | Apache |
| translation | WMT, FLORES | 3,000 | CC-BY |
| conversation | ShareGPT conversation subset | 8,000 | MIT |
| reasoning | GSM8K, MATH, ARC | 5,000 | MIT |
| tool_use | ToolBench, API-Bank | 3,000 | Apache |

### Centroid Computation

```python
def compute_centroids(dataset: dict, embedding_model: str) -> dict:
    """Compute category centroids from labeled dataset."""
    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer(embedding_model)
    centroids = {}

    for category, prompts in dataset.items():
        embeddings = model.encode(prompts, convert_to_numpy=True)
        centroid = np.mean(embeddings, axis=0)
        # L2-normalize for cosine similarity
        centroid = centroid / np.linalg.norm(centroid)
        centroids[category] = centroid

    return centroids
```

### Model Affinity Score Computation

Model affinity scores are derived from benchmark performance data:

```python
def compute_affinity(category: str, model: str, benchmarks: dict) -> float:
    """Compute how well a model handles a category (0.0-1.0)."""
    category_benchmarks = benchmarks.get(category, {})
    model_scores = category_benchmarks.get(model, {})

    if not model_scores:
        return 0.5  # Unknown: neutral affinity

    # Weighted average of relevant benchmark scores
    weights = {"accuracy": 0.4, "quality": 0.3, "latency": 0.15, "cost_efficiency": 0.15}
    score = sum(model_scores.get(k, 0.5) * w for k, w in weights.items())
    return min(1.0, max(0.0, score))
```

---

## Appendix: Model Capability Matrix

This extends the existing [`llm_candidates.json`](config/llm_candidates.json)
format with additional fields for NadirClaw routing:

```json
{
  "claude-4.5-sonnet": {
    "provider": "bedrock",
    "model_id": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "capabilities": ["reasoning", "coding", "creative", "analysis", "vision"],
    "cost_per_1k_tokens": {"input": 0.003, "output": 0.015},
    "context_window": 200000,
    "quality_score": 0.94,
    "tier": "premium",

    "nadirclaw_extensions": {
      "supports_tool_use": true,
      "supports_extended_thinking": false,
      "supports_vision": true,
      "max_tool_calls_per_turn": 20,
      "reasoning_quality": 0.91,
      "code_quality": 0.93,
      "creative_quality": 0.88,
      "latency_p50_ms": 1200,
      "latency_p99_ms": 5000
    }
  }
}
```

These extension fields allow the routing pipeline to make fine-grained decisions
when multiple models in the same tier are available. For example, within the
`premium` tier, route code generation to the model with the highest `code_quality`
score.
