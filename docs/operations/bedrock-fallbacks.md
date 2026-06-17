# Bedrock Cross-Region Fallbacks

Bedrock on-demand throughput is throttled **per region**. Under burst, a single
region returns `429 ThrottlingException` / `ServiceUnavailableException` even
while a sibling region has capacity. The standard mitigation is **cross-region
failover**: route the same logical request to a different AWS region (or a
cross-region inference profile) when the primary region is throttled.

RouteIQ ships a ready-to-use recipe for this at
[`config/config.bedrock-fallbacks.yaml`](https://github.com/baladithyab/RouteIQ/blob/main/config/config.bedrock-fallbacks.yaml).

## How it composes with RouteIQ's ML routing

RouteIQ's custom routing and LiteLLM's model-group fallbacks live on **two
different layers** and compose rather than conflict:

1. **Deployment selection (RouteIQ).** RouteIQ installs its own
   `CustomRoutingStrategyBase` (`set_custom_routing_strategy()`), which only
   replaces the Router's per-attempt deployment selection inside a model group
   (`get_available_deployment` / `async_get_available_deployment`).
2. **Model-group fallbacks (LiteLLM).** `fallbacks`,
   `context_window_fallbacks`, and `content_policy_fallbacks` are read by
   LiteLLM's outer completion wrapper (`Router.async_function_with_fallbacks`).
   They are set from `router_settings` in `Router.__init__` and are **never**
   touched by `set_custom_routing_strategy()`.

So the flow is:

```
ML routing picks a deployment in the requested group
        │
        ▼  (group/attempt fails: 429, ContextWindowExceeded, content block)
LiteLLM fallback wrapper retries the next model GROUP declared in config
```

## The recipe

The config declares **one model group per region** so a fallback can target a
specific region, plus a cross-region inference-profile arm as the last line of
defence:

```yaml
model_list:
  - model_name: claude-sonnet-us-east-1
    litellm_params:
      model: bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0
      aws_region_name: us-east-1

  - model_name: claude-sonnet-us-west-2
    litellm_params:
      model: bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0
      aws_region_name: us-west-2

  # Cross-region (US) inference-profile arm — spreads across the US region group.
  - model_name: claude-sonnet-us-xregion
    litellm_params:
      model: bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0
      aws_region_name: us-east-1
```

### Region failover (`fallbacks`)

Each fallback dict has **exactly one key** (LiteLLM's `validate_fallbacks`
requirement). On a Bedrock 429 in `us-east-1`, LiteLLM retries the same request
in `us-west-2`, then the cross-region US inference profile:

```yaml
router_settings:
  fallbacks:
    - claude-sonnet-us-east-1:
        - claude-sonnet-us-west-2
        - claude-sonnet-us-xregion
    - claude-sonnet-us-west-2:
        - claude-sonnet-us-east-1
        - claude-sonnet-us-xregion
```

### Context-window overflow (`context_window_fallbacks`)

When a request exceeds the chosen model's context window, retry against a
larger-context group instead of failing:

```yaml
  context_window_fallbacks:
    - claude-haiku-us-east-1:
        - claude-sonnet-longctx
```

### Content-policy blocks (`content_policy_fallbacks`)

When a provider-side content policy blocks a response, retry against an
alternate group:

```yaml
  content_policy_fallbacks:
    - claude-sonnet-us-east-1:
        - claude-sonnet-us-west-2
```

## Inference-profile prefixes

| Form | Example | Notes |
|------|---------|-------|
| Per-region on-demand | `bedrock/anthropic.claude-3-5-sonnet-...` | Pinned to a region via `aws_region_name`. |
| Cross-region (system-defined) profile | `bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0` | `us.` / `eu.` regional prefix; already spreads load across the region group. |

The recipe adds an explicit other-region group **in addition to** the
cross-region profile as a second line of defence for hard regional outages.

## Run it

No `api_key` is needed — Bedrock auth uses the IAM instance/pod identity.

```bash
uv run python -m litellm_llmrouter.startup \
  --config config/config.bedrock-fallbacks.yaml --port 4000
```

The file ships with `routing_strategy: simple-shuffle` so it runs with zero ML
model present. Swap to `llmrouter-knn` / `llmrouter-hybrid` to enable ML routing
— the fallbacks above are unaffected by that choice.

!!! tip "Retries happen before group fallback"
    `num_retries` / `retry_after` control intra-deployment retries that run
    **before** a model-group fallback is attempted. Tune both together.
