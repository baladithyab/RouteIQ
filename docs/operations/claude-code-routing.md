# Claude Code -> RouteIQ -> mixed-Bedrock cost routing

Point [Claude Code](https://claude.com/claude-code) at RouteIQ and let RouteIQ
route every turn across a **mixed pool of serverless Bedrock models** (Claude +
Nova + gpt-oss + anything else your account is entitled to), learning which arm
is cheapest-for-quality with the Kumaraswamy-Thompson online bandit. The result:
you stop paying Opus-only prices for every request while keeping the strong
models in reserve for the turns that actually need them.

```
Claude Code  --ANTHROPIC_BASE_URL-->  RouteIQ
   1. model-alias layer rewrites the pinned id  ->  "claude-auto" group
   2. Kumaraswamy-Thompson bandit picks the best arm in the group
   3. eval-judge -> bandit MLOps loop feeds quality back so it keeps learning
```

## 1. Point Claude Code at RouteIQ

Claude Code speaks the Anthropic Messages API. Set its base URL to your RouteIQ
gateway and pin the model to the routing group name:

```bash
export ANTHROPIC_BASE_URL="http://localhost:4000"   # your RouteIQ endpoint
export ANTHROPIC_API_KEY="$LITELLM_MASTER_KEY"       # or a RouteIQ-issued key
export ANTHROPIC_MODEL="claude-auto"                  # the routing GROUP
```

`claude-auto` is a routing **group**, not a single model: RouteIQ fans the
request across every arm in the group and the bandit chooses. If you cannot set
`ANTHROPIC_MODEL` (Claude Code pins a concrete id like
`claude-sonnet-4-20250514`), use the **model-alias layer** in step 4 to rewrite
the pinned id to `claude-auto` transparently.

RouteIQ serves the Anthropic Messages API at the bare `/v1/messages` path, so an
unmodified Anthropic client works unchanged.

## 2. Build the `claude-auto` group

### Option A -- auto-discovery (recommended on AWS)

Let RouteIQ assemble the group from whatever serverless Bedrock models the
account is entitled to. No hand-maintained `model_list`:

```bash
export ROUTEIQ_BEDROCK_DISCOVERY__ENABLED=true
export ROUTEIQ_BEDROCK_DISCOVERY__SOURCE_REGIONS="us-east-1,us-west-2"
export ROUTEIQ_BEDROCK_DISCOVERY__AUTO_GROUP=true
export ROUTEIQ_BEDROCK_DISCOVERY__AUTO_GROUP_NAME="claude-auto"   # default
```

Discovery is a read-only control-plane scan (`ListFoundationModels` /
`ListInferenceProfiles`) and is **off by default** -- nothing happens until you
opt in. With `AUTO_GROUP=true` every discovered serverless arm (Claude, Nova,
gpt-oss, any future provider) collapses to the single `claude-auto`
`model_name`; each arm keeps an `arm_id` in `model_info` so telemetry still shows
which physical model served the request.

#### Per-logical-model groups (RouteIQ-1c9d)

`AUTO_GROUP` collapses *everything* into one group. When you instead want **one
logical model_name per model, each fanned out across its region / account /
mantle arms**, use `DiscoveryResult.synthesize_model_groups()`. It binds arms by
their *logical model identity* -- the region-varying parts (the `global.` /
`us.` / `eu.` tier-geo prefix and the `-v1:0` version suffix) are stripped -- so:

- `global.anthropic.claude-sonnet-4-v1:0` discovered in `us-east-1`,
- the `eu.anthropic.claude-sonnet-4-v1:0` geo profile in `eu-west-1`, and
- a Bedrock Marketplace / **mantle** custom-deployment endpoint of the same model

all land under the single `anthropic.claude-sonnet-4` `model_name` as three arms,
while a *different* logical model (e.g. Nova) forms its own group. Each arm keeps
a distinct `model_info.arm_id` (`region/invocation_id`) for telemetry. This is
the building block for routing one logical alias across regions/accounts when you
don't want a single all-models group.

### Option B -- static recipe

Use the worked recipe at [`config/config.claude-code.yaml`](https://github.com/baladithyab/RouteIQ/blob/main/config/config.claude-code.yaml).
It declares the `claude-auto` group with explicit mixed-tier Bedrock arms
(Haiku / Nova-lite / Nova-pro / gpt-oss / Sonnet-4 / Opus-4). Edit the model ids
for your region's entitlements. Works without discovery.

```bash
uv run python -m litellm_llmrouter.startup \
  --config config/config.claude-code.yaml --port 4000
```

## 3. Turn on the Kumaraswamy-Thompson bandit

The bandit is a registered routing strategy that rides the RouteIQ routing
pipeline. Enable it and make it the active strategy:

```bash
export ROUTEIQ_KUMARASWAMY_THOMPSON__ENABLED=true
export ROUTEIQ_ROUTING__ACTIVE_STRATEGY="routeiq-kumaraswamy-thompson"
# Optional: survive restarts (resume convergence) instead of relearning cold.
export ROUTEIQ_KUMARASWAMY_THOMPSON__BACKEND=file
export ROUTEIQ_KUMARASWAMY_THOMPSON__STATE_PATH=/var/lib/routeiq/kts.json
```

The bandit's reward blends quality, cost, and latency
(`w_quality` / `w_cost` / `w_latency`), so a cheap arm that is "good enough"
wins until quality feedback says otherwise.

## 4. (Optional) Transparent model-alias rewrite

If you cannot point `ANTHROPIC_MODEL` at `claude-auto` directly -- e.g. Claude
Code pins a concrete id -- enable the **model-alias rewrite layer**. It rewrites
the request `model` field PRE-routing, at the RouteIQ app entry layer (a
raw-ASGI middleware, *not* the unreliable `on_llm_pre_call` mutation seam), so an
unmodified client is transparently routed through the group:

```bash
export ROUTEIQ_MODEL_ALIAS__ENABLED=true
# Exact id -> group:
export ROUTEIQ_MODEL_ALIAS__EXACT='{"claude-sonnet-4-20250514":"claude-auto","claude-opus-4-20250514":"claude-auto"}'
# Or a regex catch-all (first fullmatch wins):
export ROUTEIQ_MODEL_ALIAS__REGEX='{"^claude-.*$":"claude-auto"}'
```

The layer is **off by default (identity)**. With no rules it adds zero overhead.
A bad regex is logged and skipped (fail-open) -- it never blocks traffic.

## 5. The learning loop (on by default)

For this cost-routing use case the bandit is only worth it if it **learns
automatically** which arm is best. So the eval-judge -> bandit MLOps feedback
loop is **on by default** (`ROUTEIQ_ADAPTER_FRAMEWORK__MLOPS_FEEDBACK_LOOP=true`).

The loop is a no-op unless both (a) a continuous-learning strategy is registered
(the bandit, off by default) **and** (b) the eval pipeline is enabled -- so the
default flag costs nothing on its own. When both are on, the eval pipeline's
aggregate `{model: quality}` is fanned out to the bandit on every batch so its
posteriors track real response quality.

!!! warning "Judge cost"
    When the eval pipeline's LLM-as-judge evaluator is enabled it makes one
    extra judge call per **sampled** decision. Size the judge model and the
    sample rate accordingly -- a small/cheap judge and a modest sample rate keep
    the learning signal flowing without doubling your spend.

Opt out entirely:

```bash
export ROUTEIQ_ADAPTER_FRAMEWORK__MLOPS_FEEDBACK_LOOP=false
```

## Putting it together

```bash
# 1. Build the group (auto-discovery)
export ROUTEIQ_BEDROCK_DISCOVERY__ENABLED=true
export ROUTEIQ_BEDROCK_DISCOVERY__SOURCE_REGIONS="us-east-1,us-west-2"
export ROUTEIQ_BEDROCK_DISCOVERY__AUTO_GROUP=true

# 2. Bandit + learning loop
export ROUTEIQ_KUMARASWAMY_THOMPSON__ENABLED=true
export ROUTEIQ_ROUTING__ACTIVE_STRATEGY="routeiq-kumaraswamy-thompson"
# (MLOPS_FEEDBACK_LOOP defaults to true)

# 3. Transparent alias (only if you cannot set ANTHROPIC_MODEL=claude-auto)
export ROUTEIQ_MODEL_ALIAS__ENABLED=true
export ROUTEIQ_MODEL_ALIAS__REGEX='{"^claude-.*$":"claude-auto"}'

# 4. Point Claude Code at RouteIQ
export ANTHROPIC_BASE_URL="http://localhost:4000"
export ANTHROPIC_API_KEY="$LITELLM_MASTER_KEY"
export ANTHROPIC_MODEL="claude-auto"
```

Now every Claude Code turn flows through RouteIQ, gets routed across your mixed
Bedrock pool by the bandit, and the loop keeps tuning the mix to your real
quality bar.
