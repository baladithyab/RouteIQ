# ADR-0002: Use CustomRoutingStrategyBase Instead of Monkey-Patching

**Status**: Accepted
**Date**: 2026-04-02
**Decision Makers**: RouteIQ Core Team

## Context

### The Legacy Monkey-Patch

RouteIQ's original routing integration worked by monkey-patching three methods on
LiteLLM's `Router` class at the module level:

1. `Router.routing_strategy_init()` — to recognize `llmrouter-*` strategy names
2. `Router.get_available_deployment()` — synchronous routing decision
3. `Router.async_get_available_deployment()` — async routing decision

This was implemented in `routing_strategy_patch.py` (~592 lines) and applied via
`patch_litellm_router()` before any Router instance was created. The patch used
`functools.wraps` to preserve the original method signatures and stored originals
for potential restoration.

### Problems with Monkey-Patching

1. **Class-level mutation**: Patching `Router.get_available_deployment` at the class
   level means ALL Router instances in the process share the patched behavior.
   Under `os.fork()` (multi-worker uvicorn), the class-level mutation is carried
   into child processes, but only if applied before the fork. This created a
   fragile ordering dependency and forced `workers=1` in startup.py.

2. **Upstream fragility**: If LiteLLM changes the signature of
   `routing_strategy_init()`, `get_available_deployment()`, or
   `async_get_available_deployment()`, the patch silently fails or produces
   incorrect behavior. No compile-time or import-time check catches this.

3. **Testing difficulty**: Tests must carefully manage patch state. The global
   `_patch_applied` flag and stored `_original_*` methods are module-level
   singletons that leak between tests unless explicitly reset.

4. **Single point of failure**: The entire ML routing system depends on
   monkey-patching succeeding silently at startup. There's no fallback if the
   patch fails to apply correctly.

### The Discovery

During the TG3 architecture review, we discovered that LiteLLM ships an official
extension point for custom routing:

- `CustomRoutingStrategyBase` in `litellm.types.router` — an abstract base class
  with `get_available_deployment()` and `async_get_available_deployment()` methods.

- `Router.set_custom_routing_strategy(strategy)` — a method that replaces the
  Router instance's deployment selection methods via `setattr()` at the **instance**
  level (not class level), binding the strategy's methods to the specific Router
  instance.

This is exactly the API surface that RouteIQ was monkey-patching, but done
correctly via LiteLLM's own supported mechanism.

## Decision

Replace the monkey-patch approach with a proper `CustomRoutingStrategyBase`
implementation.

### Implementation: `custom_routing_strategy.py`

A new module (`custom_routing_strategy.py`, ~757 lines) implements
`RouteIQRoutingStrategy(CustomRoutingStrategyBase)` that:

1. **Wraps the existing routing pipeline**: The strategy delegates to
   `LLMRouterStrategyFamily` and `RoutingPipeline` for ML-based routing decisions,
   preserving all 18 strategies, A/B testing, and telemetry.

2. **Integrates centroid routing**: Falls back to `CentroidRoutingStrategy` for
   zero-config routing when no ML models are trained (see
   [ADR-0010](0010-centroid-zero-config-routing.md)).

3. **Supports routing profiles**: Honors `ROUTEIQ_ROUTING_PROFILE` (auto, eco,
   premium, free, reasoning) to constrain model selection.

4. **Emits telemetry**: Sets `router.*` span attributes on the active OTel span
   for every routing decision.

### Installation

The strategy is installed via a helper function:

```python
from litellm_llmrouter.custom_routing_strategy import install_routeiq_strategy

# After LiteLLM creates its Router instance:
strategy = install_routeiq_strategy(router, strategy_name="llmrouter-knn")
```

This calls `router.set_custom_routing_strategy(strategy)`, which uses `setattr()`
to replace the instance methods. No class-level mutation occurs.

### Gateway Integration

The `gateway/app.py` composition root checks `ROUTEIQ_USE_PLUGIN_STRATEGY`
(defaults to `true`) and calls either:

- `install_routeiq_strategy()` (new, default) — per-instance binding
- `patch_litellm_router()` (legacy, deprecated) — class-level monkey-patch

```python
def _use_plugin_strategy() -> bool:
    return os.environ.get("ROUTEIQ_USE_PLUGIN_STRATEGY", "true").lower() in (
        "true", "1", "yes",
    )
```

### Deprecation of `routing_strategy_patch.py`

The legacy monkey-patch module is deprecated (since v0.3.0) with:

- A `DeprecationWarning` emitted when `patch_litellm_router()` is called
- Module-level docstring marked as deprecated
- Retained solely for backward compatibility with deployments that explicitly
  set `ROUTEIQ_USE_PLUGIN_STRATEGY=false`
- Planned for removal in the next major release

## Consequences

### Positive

- **Multi-worker support**: Since `set_custom_routing_strategy()` operates on the
  Router instance (not the class), `os.fork()` preserves the strategy correctly.
  `ROUTEIQ_WORKERS` can now be set to any value. Workers > 1 is only available
  when using the plugin strategy.

- **Upstream compatibility**: Using LiteLLM's official API means routing integration
  survives LiteLLM upgrades as long as the `CustomRoutingStrategyBase` contract is
  maintained. This is a public, documented API.

- **Simpler testing**: No global patch state to manage. Tests can create a Router
  mock, install the strategy, and test routing behavior in isolation.

- **Clean error handling**: If strategy installation fails, it's a clear error at
  startup rather than a silent patch failure that manifests later.

- **No method signature coupling**: The strategy only needs to implement the
  `CustomRoutingStrategyBase` interface, not match internal LiteLLM method
  signatures exactly.

### Negative

- **Dual code paths during migration**: Both `custom_routing_strategy.py` and
  `routing_strategy_patch.py` must be maintained until the legacy path is removed.
  This adds ~592 lines of deprecated code to maintain.

- **Router instance dependency**: The strategy must be installed after the Router
  is created, not before. This creates a temporal dependency in the boot sequence
  that the composition root must manage.

- **Feature parity validation**: Both paths must produce identical routing decisions.
  Integration tests must cover both `ROUTEIQ_USE_PLUGIN_STRATEGY=true` and `false`
  to ensure no behavioral divergence during the migration period.

## Alternatives Considered

### Alternative A: Keep Monkey-Patching, Fix Multi-Worker

Attempt to make the monkey-patch work with multi-worker by ensuring the patch is
applied after `os.fork()` in each worker process.

- **Pros**: No new code; minimal change to existing architecture.
- **Cons**: uvicorn's worker lifecycle doesn't provide a reliable post-fork hook
  for application code. Using `os.register_at_fork()` is fragile and
  platform-dependent. The fundamental fragility of monkey-patching upstream
  methods remains.
- **Rejected**: Fixes the symptom (multi-worker) without addressing the root
  cause (tight coupling to LiteLLM internals).

### Alternative B: LiteLLM AutoRouter Pre-Routing Hook

Use `AutoRouter.async_pre_routing_hook()` for pre-classification instead of
replacing the deployment selection entirely.

- **Pros**: Even lighter integration; only pre-classifies prompts before
  LiteLLM's built-in routing.
- **Cons**: Pre-routing hooks can influence model group selection but cannot
  fully replace deployment selection. RouteIQ's strategies need control over
  which specific deployment is selected, not just which model group.
- **Rejected**: Insufficient control for ML-based routing strategies that
  need to select specific deployments based on embedding similarity.

### Alternative C: LiteLLM Callback-Based Routing

Use LiteLLM's `CustomLogger` callback system to intercept and redirect requests.

- **Pros**: No patching; uses existing callback infrastructure.
- **Cons**: Callbacks fire after routing decisions, not during. There's no
  callback hook that allows modifying the deployment selection. Callbacks are
  for observability and side effects, not control flow.
- **Rejected**: Architecturally inappropriate — callbacks are post-hoc, not
  pre-decision.

## References

- `src/litellm_llmrouter/custom_routing_strategy.py` — Plugin strategy implementation
- `src/litellm_llmrouter/routing_strategy_patch.py` — Deprecated monkey-patch
- `src/litellm_llmrouter/gateway/app.py` — Composition root (strategy selection logic)
- [TG3 Alternative Patterns Analysis](../architecture/tg3-alternative-patterns.md)
- LiteLLM `CustomRoutingStrategyBase`: `litellm.types.router`
- LiteLLM `Router.set_custom_routing_strategy()`: `litellm.router`
