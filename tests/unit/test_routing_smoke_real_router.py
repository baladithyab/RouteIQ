"""Gate-3 routing smoke test — binds a REAL ``litellm.Router``.

This is the bump gate that the existing routing tests CANNOT provide:

- ``test_custom_routing_strategy.py`` builds a ``MagicMock`` router whose
  ``set_custom_routing_strategy`` is a hand-written side_effect, so it never
  exercises LiteLLM's real ``Router.set_custom_routing_strategy`` and would
  stay green even if that method (or ``CustomRoutingStrategyBase``) changed
  signature/location upstream.
- ``test_multi_worker.py`` only parses env vars; it binds no Router.

A bumped litellm with a moved/renamed ``CustomRoutingStrategyBase`` would let
``custom_routing_strategy.py`` fall back to its local *stub* base class (the
import is guarded by ``except ImportError`` only) and ML routing would degrade
to LiteLLM-default selection **silently**. This test fails loudly in that case.

Assertions:
  1. No silent fallback — the REAL Router rebound BOTH hooks to OUR strategy's
     bound methods (verified by ``__func__`` / ``__self__`` identity, which
     survives ``set_custom_routing_strategy`` capturing the bound method at
     install time; a later ``setattr`` spy would never be observed).
  2. A real decision flows THROUGH ``RouteIQRoutingStrategy`` — the ML-chosen
     deployment is returned, NOT the first-deployment fallback.
  3. Gate-4 deployment-dict shape (``model_name`` + ``litellm_params.model``)
     on the real Router is unchanged.
"""

from __future__ import annotations

import pytest

# Skip cleanly if litellm is somehow unavailable in the env (it is a hard dep,
# but the import guard keeps the suite honest rather than erroring at collect).
litellm = pytest.importorskip("litellm")

from litellm import Router  # noqa: E402

from litellm_llmrouter.custom_routing_strategy import (  # noqa: E402
    RouteIQRoutingStrategy,
    install_routeiq_strategy,
)


# Two deployments under the same model group "gpt-4" with DISTINCT
# litellm_params.model + model_info.id so the ML choice is distinguishable
# from the first-deployment fallback. Placeholder keys only — no secrets.
_MODEL_LIST = [
    {
        "model_name": "gpt-4",
        "litellm_params": {"model": "openai/gpt-4", "api_key": "sk-test"},
        "model_info": {"id": "d1"},
    },
    {
        "model_name": "gpt-4",
        "litellm_params": {
            "model": "anthropic/claude-3-opus",
            "api_key": "sk-test",
        },
        "model_info": {"id": "d2"},
    },
]


@pytest.fixture
def real_router() -> Router:
    """A REAL litellm.Router (not a MagicMock)."""
    return Router(model_list=[dict(d) for d in _MODEL_LIST])


class _FakeMLStrategy:
    """Stand-in for LLMRouterStrategyFamily that picks a deterministic model.

    Injected as ``strategy._strategy_instance`` so ``_route_via_llmrouter``
    skips the heavy ML lazy-load and we can assert the routed deployment is the
    ML-chosen one rather than the first-deployment fallback.
    """

    def __init__(self, chosen_model: str) -> None:
        self.chosen_model = chosen_model
        self.calls: list[tuple] = []

    def route_with_observability(self, query, model_list):  # noqa: ANN001
        self.calls.append((query, tuple(model_list)))
        return self.chosen_model


def test_real_router_rebinds_both_hooks_to_routeiq_strategy(
    real_router: Router,
) -> None:
    """Assert 1: no silent fallback — both hooks point at OUR strategy."""
    strategy = install_routeiq_strategy(real_router, strategy_name="llmrouter-knn")

    assert isinstance(strategy, RouteIQRoutingStrategy)

    # The REAL Router must support the plugin API at all (catches a removed method).
    assert hasattr(real_router, "set_custom_routing_strategy")

    # Both hooks must be OUR strategy's bound methods. Compare __func__/__self__
    # identity rather than a call-spy: set_custom_routing_strategy captures the
    # bound method at install time, so a later setattr spy would never be seen.
    assert (
        real_router.get_available_deployment.__func__
        is RouteIQRoutingStrategy.get_available_deployment
    )
    assert (
        real_router.async_get_available_deployment.__func__
        is RouteIQRoutingStrategy.async_get_available_deployment
    )
    assert real_router.get_available_deployment.__self__ is strategy
    assert real_router.async_get_available_deployment.__self__ is strategy


def test_real_decision_flows_through_routeiq_strategy(
    real_router: Router, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Assert 2: a real decision is the ML choice, NOT the first-deploy fallback."""
    strategy = install_routeiq_strategy(real_router, strategy_name="llmrouter-knn")

    # Force the direct-LLMRouter path (skip pipeline) and inject a deterministic
    # ML strategy that picks the SECOND deployment's model.
    monkeypatch.setattr(
        "litellm_llmrouter.custom_routing_strategy.USE_PIPELINE_ROUTING",
        False,
    )
    fake = _FakeMLStrategy(chosen_model="anthropic/claude-3-opus")
    strategy._strategy_instance = fake

    deployment = strategy.get_available_deployment(
        model="gpt-4",
        messages=[{"role": "user", "content": "hi"}],
    )

    # The strategy's ML path was actually invoked...
    assert fake.calls, "route_with_observability was never called"
    # ...and the returned deployment is the ML-chosen one, not the gpt-4 fallback.
    assert deployment is not None
    assert deployment["litellm_params"]["model"] == "anthropic/claude-3-opus"
    assert deployment["model_info"]["id"] == "d2"


def test_deployment_dict_shape_unchanged_on_real_router(real_router: Router) -> None:
    """Assert 3 (Gate 4): deployment-dict shape on the REAL Router is unchanged."""
    d = real_router.model_list[0]
    assert "model_name" in d
    assert "litellm_params" in d
    assert "model" in d["litellm_params"]
