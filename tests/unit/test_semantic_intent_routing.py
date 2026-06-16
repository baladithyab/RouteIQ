"""
Tests for the SemanticIntentRoutingStrategy (RouteIQ-7936).

Semantic / embedding intent router: classify request -> intent -> model group.
Verifies:
1. intent -> correct model-group dispatch (mock the embedder).
2. unmapped intent / below-threshold -> graceful fallback.
3. classifier NOT loaded -> graceful fallthrough (NO cold load on hot path).
4. no model in the mapped group matches -> fallback.
5. registration: in LLMROUTER_STRATEGIES + selectable by name.
6. validate() bounds-checks config.

The hot-path discipline is the load-bearing assertion: when the shared encoder
singleton (``strategies._sentence_transformer_model``) is ``None`` the strategy
must NOT call ``_get_sentence_transformer`` (which would cold-load the model).
"""

from typing import Dict, List
from unittest.mock import MagicMock

import numpy as np
import pytest

from litellm_llmrouter import strategies as strategies_mod
from litellm_llmrouter.strategies import (
    SemanticIntentRoutingStrategy,
    LLMROUTER_STRATEGIES,
    DEFAULT_ROUTER_HPARAMS,
    register_semantic_intent_strategy,
)
from litellm_llmrouter.strategy_registry import (
    RoutingContext,
    get_routing_registry,
    reset_routing_singletons,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_deployment(model: str, model_name: str = "test-model") -> Dict:
    return {"model_name": model_name, "litellm_params": {"model": model}}


def _make_router(deployments: List[Dict]) -> MagicMock:
    router = MagicMock()
    router.model_list = deployments
    router.healthy_deployments = deployments
    return router


def _make_context(
    deployments: List[Dict],
    text: str = "write a python function",
    model_name: str = "test-model",
) -> RoutingContext:
    return RoutingContext(
        router=_make_router(deployments),
        model=model_name,
        messages=[{"role": "user", "content": text}],
    )


# A deterministic fake encoder: maps a small set of seed/text strings to fixed
# unit vectors so cosine similarity is fully controllable in tests. The
# "code" centroid points at e0; "chat" centroid points at e1.
_VECTORS = {
    # centroid seeds (intent label + patterns joined by space)
    "code gpt-4o": np.array([1.0, 0.0, 0.0]),
    "chat gpt-4o-mini": np.array([0.0, 1.0, 0.0]),
    # request texts
    "write a python function": np.array([1.0, 0.0, 0.0]),  # -> code
    "hi how are you": np.array([0.0, 1.0, 0.0]),  # -> chat
    "totally unrelated noise": np.array([0.0, 0.0, 1.0]),  # orthogonal to both
}


class _FakeEncoder:
    def encode(self, texts, show_progress_bar=False):
        out = []
        for t in texts:
            vec = _VECTORS.get(t)
            if vec is None:
                vec = np.array([0.0, 0.0, 0.0])
            out.append(vec.astype(float))
        return np.array(out)


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    reset_routing_singletons()
    # Ensure each test controls the encoder-loaded state explicitly.
    monkeypatch.setattr(strategies_mod, "_sentence_transformer_model", None)
    yield
    reset_routing_singletons()
    monkeypatch.setattr(strategies_mod, "_sentence_transformer_model", None)


def _mark_encoder_loaded(monkeypatch):
    """Mark the shared encoder singleton as loaded and return the fake."""
    fake = _FakeEncoder()
    monkeypatch.setattr(strategies_mod, "_sentence_transformer_model", fake)
    # _get_sentence_transformer would return the singleton; patch it to the fake
    # so build/classify use the deterministic vectors.
    monkeypatch.setattr(
        strategies_mod, "_get_sentence_transformer", lambda *a, **k: fake
    )
    return fake


INTENT_GROUPS = {
    "code": ["gpt-4o"],
    "chat": ["gpt-4o-mini"],
}


# ---------------------------------------------------------------------------
# intent -> correct model-group dispatch
# ---------------------------------------------------------------------------


def test_code_intent_routes_to_code_group(monkeypatch):
    _mark_encoder_loaded(monkeypatch)
    deployments = [_make_deployment("gpt-4o-mini"), _make_deployment("gpt-4o")]
    ctx = _make_context(deployments, text="write a python function")
    strat = SemanticIntentRoutingStrategy(intent_model_groups=INTENT_GROUPS)
    result = strat.select_deployment(ctx)
    assert result is not None
    assert result["litellm_params"]["model"] == "gpt-4o"


def test_chat_intent_routes_to_chat_group(monkeypatch):
    _mark_encoder_loaded(monkeypatch)
    deployments = [_make_deployment("gpt-4o"), _make_deployment("gpt-4o-mini")]
    ctx = _make_context(deployments, text="hi how are you")
    strat = SemanticIntentRoutingStrategy(intent_model_groups=INTENT_GROUPS)
    result = strat.select_deployment(ctx)
    assert result["litellm_params"]["model"] == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# unmapped / below-threshold -> fallback
# ---------------------------------------------------------------------------


def test_below_threshold_falls_back(monkeypatch):
    _mark_encoder_loaded(monkeypatch)
    deployments = [_make_deployment("gpt-4o-mini"), _make_deployment("gpt-4o")]
    # orthogonal text -> top-1 similarity 0.0; threshold 0.5 rejects -> fallback
    ctx = _make_context(deployments, text="totally unrelated noise")
    strat = SemanticIntentRoutingStrategy(
        intent_model_groups=INTENT_GROUPS, similarity_threshold=0.5
    )
    result = strat.select_deployment(ctx)
    # fallback = first available candidate
    assert result["litellm_params"]["model"] == "gpt-4o-mini"


def test_unmapped_intent_falls_back(monkeypatch):
    _mark_encoder_loaded(monkeypatch)
    deployments = [_make_deployment("gpt-4o-mini"), _make_deployment("gpt-4o")]
    # "code" intent classifies, but its group lists a model not deployed here
    ctx = _make_context(deployments, text="write a python function")
    strat = SemanticIntentRoutingStrategy(
        intent_model_groups={"code": ["model-that-is-not-deployed"]}
    )
    result = strat.select_deployment(ctx)
    # group has no matching candidate -> graceful fallback to first available
    assert result["litellm_params"]["model"] == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# classifier NOT loaded -> graceful fallthrough, NO cold load
# ---------------------------------------------------------------------------


def test_not_loaded_does_not_cold_load(monkeypatch):
    # encoder singleton is None (set by autouse fixture). Spy on the loader to
    # PROVE it is never called on the hot path.
    loader_spy = MagicMock(side_effect=AssertionError("cold load on hot path!"))
    monkeypatch.setattr(strategies_mod, "_get_sentence_transformer", loader_spy)

    deployments = [_make_deployment("gpt-4o-mini"), _make_deployment("gpt-4o")]
    ctx = _make_context(deployments, text="write a python function")
    strat = SemanticIntentRoutingStrategy(intent_model_groups=INTENT_GROUPS)
    result = strat.select_deployment(ctx)

    # graceful fallthrough to first candidate, and the loader was never called
    assert result["litellm_params"]["model"] == "gpt-4o-mini"
    loader_spy.assert_not_called()


def test_classify_intent_returns_none_when_not_loaded(monkeypatch):
    # direct check of the classify guard
    loader_spy = MagicMock(side_effect=AssertionError("cold load!"))
    monkeypatch.setattr(strategies_mod, "_get_sentence_transformer", loader_spy)
    strat = SemanticIntentRoutingStrategy(intent_model_groups=INTENT_GROUPS)
    intent, sim = strat._classify_intent("write a python function")
    assert intent is None
    assert sim == 0.0
    loader_spy.assert_not_called()


def test_no_candidates_returns_none(monkeypatch):
    _mark_encoder_loaded(monkeypatch)
    ctx = _make_context([], model_name="nonexistent")
    strat = SemanticIntentRoutingStrategy(intent_model_groups=INTENT_GROUPS)
    assert strat.select_deployment(ctx) is None


# ---------------------------------------------------------------------------
# Registration / selectability
# ---------------------------------------------------------------------------


def test_registered_in_strategy_list():
    assert "llmrouter-semantic-intent" in LLMROUTER_STRATEGIES
    assert "semantic-intent" in DEFAULT_ROUTER_HPARAMS


def test_register_disabled_by_default():
    assert register_semantic_intent_strategy() is False


def test_register_when_enabled(monkeypatch):
    from litellm_llmrouter import settings as settings_mod

    settings_mod.reset_settings()
    monkeypatch.setenv("ROUTEIQ_SEMANTIC_INTENT__ENABLED", "true")
    monkeypatch.setenv(
        "ROUTEIQ_SEMANTIC_INTENT__INTENT_MODEL_GROUPS",
        '{"code": ["gpt-4o"], "chat": ["gpt-4o-mini"]}',
    )
    settings_mod.reset_settings()
    try:
        assert register_semantic_intent_strategy() is True
        registry = get_routing_registry()
        strat = registry.get("llmrouter-semantic-intent")
        assert isinstance(strat, SemanticIntentRoutingStrategy)
        assert strat.name == "llmrouter-semantic-intent"
        assert strat._intent_model_groups == {
            "code": ["gpt-4o"],
            "chat": ["gpt-4o-mini"],
        }
    finally:
        settings_mod.reset_settings()


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------


def test_validate_ok():
    ok, err = SemanticIntentRoutingStrategy().validate()
    assert ok and err is None


def test_validate_rejects_bad_threshold():
    strat = SemanticIntentRoutingStrategy()
    strat._similarity_threshold = 1.5  # bypass clamp
    ok, err = strat.validate()
    assert not ok
    assert "similarity_threshold" in err
