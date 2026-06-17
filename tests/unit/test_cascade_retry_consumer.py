"""Unit tests for the cost-cascade mode-(a) RETRY consumer (RouteIQ-3ff5).

``CostCascadeRoutingStrategy`` attaches the escalation ladder under
``metadata['routeiq_cascade']`` but nothing climbed it. These tests prove the
consumer closes mode (a):

* a LOW-confidence cheapest-rung response is RE-ISSUED at the next (stronger)
  rung -- the ladder is climbed;
* a CONFIDENT response stops climbing (no re-issue);
* climbing is bounded by ``max_rungs`` / ladder length;
* default OFF -> ``get_cascade_retry_consumer`` returns None (byte-stable);
* the LIVE callsite ``RouteIQRoutingStrategy.acompletion_with_cascade`` climbs
  the ladder when enabled, and is a single-shot completion when disabled.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from litellm_llmrouter.cascade_retry import (
    CascadeRetryConsumer,
    extract_response_confidence,
    get_cascade_retry_consumer,
    reset_cascade_retry_consumer,
)
from litellm_llmrouter.settings import reset_settings


@pytest.fixture(autouse=True)
def _reset():
    reset_settings()
    reset_cascade_retry_consumer()
    yield
    reset_settings()
    reset_cascade_retry_consumer()


_LADDER = [
    {"model": "cheap", "model_name": "grp", "cost_per_1k": 0.1},
    {"model": "mid", "model_name": "grp", "cost_per_1k": 1.0},
    {"model": "strong", "model_name": "grp", "cost_per_1k": 5.0},
]


def _rung_deployment(rung_index):
    return {
        "model_name": "grp",
        "litellm_params": {"model": _LADDER[rung_index]["model"]},
        "metadata": {"routeiq_cascade": {"ladder": _LADDER}},
    }


# ---------------------------------------------------------------------------
# confidence extraction
# ---------------------------------------------------------------------------


def test_extract_explicit_confidence():
    assert extract_response_confidence({"routeiq_confidence": 0.42}) == pytest.approx(
        0.42
    )


def test_extract_no_signal_returns_none():
    assert extract_response_confidence({"choices": [{"text": "hi"}]}) is None


def test_extract_from_logprobs():
    import math

    resp = {
        "choices": [
            {
                "logprobs": {
                    "content": [{"logprob": math.log(0.5)}, {"logprob": math.log(0.5)}]
                }
            }
        ]
    }
    assert extract_response_confidence(resp) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# the consumer climbs the ladder
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_climbs_ladder_on_low_confidence():
    """Low-confidence rung-0 -> escalate up the ladder until confident."""
    consumer = CascadeRetryConsumer(confidence_threshold=0.7, max_rungs=4)

    # rung 0 + rung 1 return low confidence; rung 2 returns high.
    confidences = {"cheap": 0.2, "mid": 0.3, "strong": 0.95}
    calls = []

    def _select(rung):
        return _rung_deployment(rung)

    def _complete(deployment, rung):
        model = deployment["litellm_params"]["model"]
        calls.append(model)
        return {"routeiq_confidence": confidences[model], "model": model}

    result = await consumer.complete_with_cascade(
        select_rung=_select, complete=_complete
    )

    # The ladder was climbed cheap -> mid -> strong.
    assert calls == ["cheap", "mid", "strong"]
    assert result["model"] == "strong"


@pytest.mark.asyncio
async def test_confident_response_does_not_climb():
    """A confident cheapest-rung response is returned without escalation."""
    consumer = CascadeRetryConsumer(confidence_threshold=0.7, max_rungs=4)
    calls = []

    def _complete(deployment, rung):
        calls.append(deployment["litellm_params"]["model"])
        return {"routeiq_confidence": 0.99}

    result = await consumer.complete_with_cascade(
        select_rung=lambda r: _rung_deployment(r), complete=_complete
    )
    assert calls == ["cheap"]
    assert result["routeiq_confidence"] == 0.99


@pytest.mark.asyncio
async def test_climb_bounded_by_max_rungs():
    """max_rungs caps how far the ladder is climbed."""
    consumer = CascadeRetryConsumer(confidence_threshold=0.7, max_rungs=2)
    calls = []

    def _complete(deployment, rung):
        calls.append(deployment["litellm_params"]["model"])
        return {"routeiq_confidence": 0.0}  # always low

    await consumer.complete_with_cascade(
        select_rung=lambda r: _rung_deployment(r), complete=_complete
    )
    # max_rungs=2 -> only rung 0 and rung 1 are tried.
    assert calls == ["cheap", "mid"]


@pytest.mark.asyncio
async def test_no_signal_does_not_climb():
    """A response with no confidence signal does not trigger escalation."""
    consumer = CascadeRetryConsumer(confidence_threshold=0.7, max_rungs=4)
    calls = []

    def _complete(deployment, rung):
        calls.append(deployment["litellm_params"]["model"])
        return {"choices": [{"text": "no logprobs"}]}

    await consumer.complete_with_cascade(
        select_rung=lambda r: _rung_deployment(r), complete=_complete
    )
    assert calls == ["cheap"]


# ---------------------------------------------------------------------------
# gating
# ---------------------------------------------------------------------------


def test_consumer_disabled_by_default(monkeypatch):
    monkeypatch.delenv("ROUTEIQ_COST_CASCADE__RETRY_CONSUMER_ENABLED", raising=False)
    reset_settings()
    assert get_cascade_retry_consumer() is None


def test_consumer_enabled_via_settings(monkeypatch):
    monkeypatch.setenv("ROUTEIQ_COST_CASCADE__ENABLED", "true")
    monkeypatch.setenv("ROUTEIQ_COST_CASCADE__RETRY_CONSUMER_ENABLED", "true")
    reset_settings()
    consumer = get_cascade_retry_consumer()
    assert isinstance(consumer, CascadeRetryConsumer)


# ---------------------------------------------------------------------------
# LIVE callsite: RouteIQRoutingStrategy.acompletion_with_cascade
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_callsite_climbs_when_enabled(monkeypatch):
    """acompletion_with_cascade climbs the ladder when the consumer is enabled."""
    monkeypatch.setenv("ROUTEIQ_COST_CASCADE__ENABLED", "true")
    monkeypatch.setenv("ROUTEIQ_COST_CASCADE__RETRY_CONSUMER_ENABLED", "true")
    reset_settings()

    from litellm_llmrouter import custom_routing_strategy as crs
    from litellm_llmrouter.strategy_registry import (
        get_routing_registry,
        reset_routing_singletons,
    )

    reset_routing_singletons()

    # A fake cascade strategy that returns the rung-pinned deployment.
    class _FakeCascade:
        def select_deployment(self, ctx):
            prior = (ctx.request_kwargs or {}).get("cascade_rung")
            rung = 0 if prior is None else prior + 1
            return _rung_deployment(rung)

    get_routing_registry().register("llmrouter-cost-cascade", _FakeCascade())

    confidences = {"cheap": 0.1, "mid": 0.9, "strong": 0.99}
    completed = []

    router = AsyncMock()

    async def _acompletion(model, messages, **kwargs):
        completed.append(model)
        return {"routeiq_confidence": confidences[model], "model": model}

    router.acompletion = AsyncMock(side_effect=_acompletion)

    strategy = crs.RouteIQRoutingStrategy(router_instance=router)
    result = await strategy.acompletion_with_cascade(
        model="grp", messages=[{"role": "user", "content": "hi"}]
    )

    # cheap (0.1 < 0.7) -> escalate to mid (0.9 >= 0.7) -> stop.
    assert completed == ["cheap", "mid"]
    assert result["model"] == "mid"

    reset_routing_singletons()


@pytest.mark.asyncio
async def test_live_callsite_single_shot_when_disabled(monkeypatch):
    """Disabled -> a single router.acompletion, no cascade orchestration."""
    monkeypatch.delenv("ROUTEIQ_COST_CASCADE__RETRY_CONSUMER_ENABLED", raising=False)
    reset_settings()
    reset_cascade_retry_consumer()

    from litellm_llmrouter import custom_routing_strategy as crs

    router = AsyncMock()
    router.acompletion = AsyncMock(return_value={"model": "default"})

    strategy = crs.RouteIQRoutingStrategy(router_instance=router)
    result = await strategy.acompletion_with_cascade(
        model="grp", messages=[{"role": "user", "content": "hi"}]
    )

    router.acompletion.assert_awaited_once()
    assert result == {"model": "default"}


# ---------------------------------------------------------------------------
# LIVE hot-path callsite: the registered post-call success hook climbs the ladder
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_call_hook_climbs_ladder_when_enabled(monkeypatch):
    """RouterDecisionCallback.async_post_call_success_hook re-issues up the ladder.

    This is the registered (litellm.callbacks) hot-path hook -- the genuine LIVE
    caller of the cascade consumer (RouteIQ-3ff5).
    """
    import sys
    import types

    monkeypatch.setenv("ROUTEIQ_COST_CASCADE__ENABLED", "true")
    monkeypatch.setenv("ROUTEIQ_COST_CASCADE__RETRY_CONSUMER_ENABLED", "true")
    reset_settings()

    from litellm_llmrouter.router_decision_callback import RouterDecisionCallback
    from litellm_llmrouter.strategy_registry import (
        get_routing_registry,
        reset_routing_singletons,
    )

    reset_routing_singletons()

    class _FakeCascade:
        def select_deployment(self, ctx):
            prior = (ctx.request_kwargs or {}).get("cascade_rung")
            rung = 0 if prior is None else prior + 1
            return _rung_deployment(rung)

    get_routing_registry().register("llmrouter-cost-cascade", _FakeCascade())

    # Fake the live router the hook reaches via litellm.proxy.proxy_server.
    reissued = []

    async def _acompletion(model, messages, **kwargs):
        reissued.append(model)
        # the stronger rung returns a confident escalated answer
        return {"choices": [{"message": {"content": "strong answer"}}], "model": model}

    fake_router = types.SimpleNamespace(acompletion=_acompletion)
    mod = types.ModuleType("litellm.proxy.proxy_server")
    mod.llm_router = fake_router
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", mod)

    # Rung-0 (cheap) response that was already served: LOW confidence -> escalate.
    response = {
        "choices": [{"message": {"content": "weak answer"}}],
        "routeiq_confidence": 0.1,
    }
    data = {"model": "grp", "messages": [{"role": "user", "content": "hi"}]}

    cb = RouterDecisionCallback(enabled=True)
    await cb.async_post_call_success_hook(
        data, user_api_key_dict=None, response=response
    )

    # The hook climbed to the next rung (re-issued once at "mid").
    assert reissued == ["mid"]
    # ...and the original response now carries the escalated content.
    assert response["choices"][0]["message"]["content"] == "strong answer"
    assert response.get("routeiq_cascade_escalated") is True

    reset_routing_singletons()


@pytest.mark.asyncio
async def test_post_call_hook_noop_when_disabled(monkeypatch):
    """Default OFF -> the post-call hook does not re-issue (byte-stable)."""
    monkeypatch.delenv("ROUTEIQ_COST_CASCADE__RETRY_CONSUMER_ENABLED", raising=False)
    reset_settings()
    reset_cascade_retry_consumer()

    from litellm_llmrouter.router_decision_callback import RouterDecisionCallback

    response = {
        "choices": [{"message": {"content": "weak answer"}}],
        "routeiq_confidence": 0.1,
    }
    data = {"model": "grp", "messages": [{"role": "user", "content": "hi"}]}

    cb = RouterDecisionCallback(enabled=True)
    await cb.async_post_call_success_hook(
        data, user_api_key_dict=None, response=response
    )

    # Unchanged: no escalation marker, original content intact.
    assert "routeiq_cascade_escalated" not in response
    assert response["choices"][0]["message"]["content"] == "weak answer"
