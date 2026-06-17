"""
Tests for the TagRegexUserAgentRoutingStrategy (RouteIQ-6865).

Tag / regex / User-Agent -> model-group routing (upstream LiteLLM
``tag_based_routing`` is bypassed by the RouteIQ custom strategy). Verifies:
1. A request tag dispatches to its mapped model group.
2. A regex over request text dispatches to its mapped group.
3. A User-Agent header substring dispatches to its mapped group.
4. Priority order: tag > regex > User-Agent.
5. No-match falls back to the first available candidate.
6. A matched group whose models are absent falls back gracefully.
7. Registration: in LLMROUTER_STRATEGIES + selectable by name + opt-in.
8. validate() requires at least one configured map.
9. An invalid regex in config is skipped, not raised.
"""

from typing import Dict, List
from unittest.mock import MagicMock

import pytest

from litellm_llmrouter.strategies import (
    DEFAULT_ROUTER_HPARAMS,
    LLMROUTER_STRATEGIES,
    TagRegexUserAgentRoutingStrategy,
    register_tag_routing_strategy,
)
from litellm_llmrouter.strategy_registry import (
    RoutingContext,
    get_routing_registry,
    reset_routing_singletons,
)


def _make_deployment(model: str, model_name: str = "test-model") -> Dict:
    return {"model_name": model_name, "litellm_params": {"model": model}}


def _make_router(deployments: List[Dict]) -> MagicMock:
    router = MagicMock()
    router.model_list = deployments
    router.healthy_deployments = deployments
    return router


def _make_context(
    deployments: List[Dict],
    model_name: str = "test-model",
    request_kwargs: Dict = None,
    metadata: Dict = None,
    messages: List[Dict] = None,
) -> RoutingContext:
    return RoutingContext(
        router=_make_router(deployments),
        model=model_name,
        request_kwargs=request_kwargs,
        metadata=metadata or {},
        messages=messages,
    )


@pytest.fixture(autouse=True)
def _reset_registry():
    reset_routing_singletons()
    yield
    reset_routing_singletons()


# ---------------------------------------------------------------------------
# Tag match
# ---------------------------------------------------------------------------


def test_tag_dispatches_to_mapped_group():
    deployments = [_make_deployment("gpt-4o-mini"), _make_deployment("gpt-4o")]
    strat = TagRegexUserAgentRoutingStrategy(
        tag_model_groups={"premium": ["gpt-4o"], "cheap": ["gpt-4o-mini"]}
    )
    ctx = _make_context(deployments, request_kwargs={"tags": ["premium"]})
    result = strat.select_deployment(ctx)
    assert result["litellm_params"]["model"] == "gpt-4o"


def test_tag_from_metadata_scalar():
    deployments = [_make_deployment("gpt-4o-mini"), _make_deployment("gpt-4o")]
    strat = TagRegexUserAgentRoutingStrategy(
        tag_model_groups={"cheap": ["gpt-4o-mini"]}
    )
    # scalar tag (not a list) in metadata
    ctx = _make_context(deployments, metadata={"tags": "cheap"})
    result = strat.select_deployment(ctx)
    assert result["litellm_params"]["model"] == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Regex match
# ---------------------------------------------------------------------------


def test_regex_dispatches_to_mapped_group():
    deployments = [_make_deployment("gpt-4o-mini"), _make_deployment("gpt-4o")]
    strat = TagRegexUserAgentRoutingStrategy(
        regex_model_groups={r"(?i)\bcode\b": ["gpt-4o"]}
    )
    ctx = _make_context(
        deployments,
        messages=[{"role": "user", "content": "please write CODE for me"}],
    )
    result = strat.select_deployment(ctx)
    assert result["litellm_params"]["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# User-Agent match
# ---------------------------------------------------------------------------


def test_user_agent_dispatches_to_mapped_group():
    deployments = [_make_deployment("gpt-4o-mini"), _make_deployment("gpt-4o")]
    strat = TagRegexUserAgentRoutingStrategy(
        user_agent_model_groups={"mobile-app": ["gpt-4o-mini"]}
    )
    ctx = _make_context(
        deployments,
        request_kwargs={"headers": {"User-Agent": "MyMobile-App/2.1 iOS"}},
    )
    result = strat.select_deployment(ctx)
    assert result["litellm_params"]["model"] == "gpt-4o-mini"


def test_user_agent_header_case_insensitive_key():
    deployments = [_make_deployment("gpt-4o-mini"), _make_deployment("gpt-4o")]
    strat = TagRegexUserAgentRoutingStrategy(
        user_agent_model_groups={"curl": ["gpt-4o"]}
    )
    # lowercase header key
    ctx = _make_context(deployments, metadata={"headers": {"user-agent": "curl/8.1.2"}})
    result = strat.select_deployment(ctx)
    assert result["litellm_params"]["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# Priority: tag > regex > User-Agent
# ---------------------------------------------------------------------------


def test_tag_wins_over_regex_and_user_agent():
    deployments = [_make_deployment("gpt-4o-mini"), _make_deployment("gpt-4o")]
    strat = TagRegexUserAgentRoutingStrategy(
        tag_model_groups={"premium": ["gpt-4o"]},
        regex_model_groups={r"hello": ["gpt-4o-mini"]},
        user_agent_model_groups={"curl": ["gpt-4o-mini"]},
    )
    ctx = _make_context(
        deployments,
        request_kwargs={
            "tags": ["premium"],
            "headers": {"User-Agent": "curl/8"},
        },
        messages=[{"role": "user", "content": "hello there"}],
    )
    result = strat.select_deployment(ctx)
    # tag should win -> gpt-4o
    assert result["litellm_params"]["model"] == "gpt-4o"


def test_regex_wins_over_user_agent_when_no_tag():
    deployments = [_make_deployment("gpt-4o-mini"), _make_deployment("gpt-4o")]
    strat = TagRegexUserAgentRoutingStrategy(
        regex_model_groups={r"hello": ["gpt-4o"]},
        user_agent_model_groups={"curl": ["gpt-4o-mini"]},
    )
    ctx = _make_context(
        deployments,
        request_kwargs={"headers": {"User-Agent": "curl/8"}},
        messages=[{"role": "user", "content": "hello there"}],
    )
    result = strat.select_deployment(ctx)
    assert result["litellm_params"]["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# No-match fallback
# ---------------------------------------------------------------------------


def test_no_match_falls_back_to_first_candidate():
    deployments = [_make_deployment("gpt-4o-mini"), _make_deployment("gpt-4o")]
    strat = TagRegexUserAgentRoutingStrategy(tag_model_groups={"premium": ["gpt-4o"]})
    # request has no premium tag -> fallback to first candidate
    ctx = _make_context(deployments, request_kwargs={"tags": ["unknown-tag"]})
    result = strat.select_deployment(ctx)
    assert result["litellm_params"]["model"] == "gpt-4o-mini"


def test_matched_group_absent_models_falls_back():
    deployments = [_make_deployment("gpt-4o-mini")]
    strat = TagRegexUserAgentRoutingStrategy(
        # tag maps to a model that is NOT among the candidates
        tag_model_groups={"premium": ["claude-opus"]}
    )
    ctx = _make_context(deployments, request_kwargs={"tags": ["premium"]})
    result = strat.select_deployment(ctx)
    # graceful: still routes to the available candidate
    assert result["litellm_params"]["model"] == "gpt-4o-mini"


def test_no_candidates_returns_none():
    strat = TagRegexUserAgentRoutingStrategy(tag_model_groups={"x": ["y"]})
    ctx = _make_context([], model_name="nonexistent", request_kwargs={"tags": ["x"]})
    assert strat.select_deployment(ctx) is None


# ---------------------------------------------------------------------------
# Invalid regex is skipped, not raised
# ---------------------------------------------------------------------------


def test_invalid_regex_skipped():
    deployments = [_make_deployment("gpt-4o-mini"), _make_deployment("gpt-4o")]
    # "(" is an invalid regex -> skipped at construction, valid one still works
    strat = TagRegexUserAgentRoutingStrategy(
        regex_model_groups={"(": ["gpt-4o-mini"], r"deploy": ["gpt-4o"]}
    )
    ctx = _make_context(
        deployments, messages=[{"role": "user", "content": "deploy the app"}]
    )
    result = strat.select_deployment(ctx)
    assert result["litellm_params"]["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# Registration / selectability / validate
# ---------------------------------------------------------------------------


def test_registered_in_strategy_list():
    assert "llmrouter-tag-regex-ua" in LLMROUTER_STRATEGIES
    assert "tag-regex-ua" in DEFAULT_ROUTER_HPARAMS


def test_register_disabled_by_default():
    assert register_tag_routing_strategy() is False


def test_register_when_enabled(monkeypatch):
    from litellm_llmrouter import settings as settings_mod

    settings_mod.reset_settings()
    monkeypatch.setenv("ROUTEIQ_TAG_ROUTING__ENABLED", "true")
    monkeypatch.setenv(
        "ROUTEIQ_TAG_ROUTING__TAG_MODEL_GROUPS", '{"premium": ["gpt-4o"]}'
    )
    settings_mod.reset_settings()
    try:
        assert register_tag_routing_strategy() is True
        registry = get_routing_registry()
        strat = registry.get("llmrouter-tag-regex-ua")
        assert isinstance(strat, TagRegexUserAgentRoutingStrategy)
        assert strat.name == "llmrouter-tag-regex-ua"
    finally:
        settings_mod.reset_settings()


def test_validate_requires_a_map():
    ok, err = TagRegexUserAgentRoutingStrategy().validate()
    assert not ok
    assert "model groups" in err


def test_validate_ok_with_a_map():
    ok, err = TagRegexUserAgentRoutingStrategy(tag_model_groups={"x": ["y"]}).validate()
    assert ok and err is None
