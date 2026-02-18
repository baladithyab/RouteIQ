"""
Unit tests for RouteIQRoutingStrategy — the official CustomRoutingStrategyBase
implementation that replaces the monkey-patch approach.

Tests cover:
- Constructor / state initialization
- Async and sync routing methods
- Amplification guard
- Pipeline routing precedence
- Direct LLMRouter routing fallback
- Final fallback to first deployment
- Query extraction helpers
- Deployment matching (exact and partial)
- install_routeiq_strategy wiring
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from litellm_llmrouter.custom_routing_strategy import (
    MAX_ROUTING_ATTEMPTS,
    RouteIQRoutingStrategy,
    create_routeiq_strategy,
    install_routeiq_strategy,
)


# ======================================================================
# Fixtures and helpers
# ======================================================================

# Sample deployment dicts matching LiteLLM Router's model_list format
_DEPLOYMENT_GPT4 = {
    "model_name": "gpt-4",
    "litellm_params": {"model": "openai/gpt-4", "api_key": "test-key"},
    "model_info": {"id": "d1"},
}

_DEPLOYMENT_CLAUDE = {
    "model_name": "gpt-4",
    "litellm_params": {"model": "anthropic/claude-3-opus", "api_key": "test-key"},
    "model_info": {"id": "d2"},
}

_DEPLOYMENT_HAIKU = {
    "model_name": "gpt-4",
    "litellm_params": {"model": "anthropic/claude-3-haiku", "api_key": "test-key"},
    "model_info": {"id": "d3"},
}


def _make_mock_router(
    model_list: Optional[List[Dict]] = None,
    healthy_deployments: Optional[List[Dict]] = None,
) -> MagicMock:
    """Create a mock LiteLLM Router with model_list and optional healthy_deployments."""
    router = MagicMock()
    router.model_list = model_list or [_DEPLOYMENT_GPT4, _DEPLOYMENT_CLAUDE]

    if healthy_deployments is not None:
        router.healthy_deployments = healthy_deployments
    else:
        # Default: healthy_deployments matches model_list
        router.healthy_deployments = router.model_list

    # set_custom_routing_strategy mock
    def _set_custom(strategy: Any) -> None:
        router.get_available_deployment = strategy.get_available_deployment
        router.async_get_available_deployment = strategy.async_get_available_deployment

    router.set_custom_routing_strategy = MagicMock(side_effect=_set_custom)
    return router


# ======================================================================
# TestRouteIQRoutingStrategy
# ======================================================================


class TestRouteIQRoutingStrategy:
    """Tests for the core RouteIQRoutingStrategy class."""

    def test_init_stores_router_reference(self) -> None:
        """Constructor stores the router reference and strategy name."""
        router = _make_mock_router()
        strategy = RouteIQRoutingStrategy(
            router_instance=router,
            strategy_name="llmrouter-knn",
        )

        assert strategy._router is router
        assert strategy._strategy_name == "llmrouter-knn"
        assert strategy._routing_attempts == {}
        assert strategy._strategy_instance is None

    def test_init_without_strategy_name(self) -> None:
        """Constructor works with no strategy name (pipeline-only mode)."""
        router = _make_mock_router()
        strategy = RouteIQRoutingStrategy(router_instance=router)

        assert strategy._strategy_name is None

    # ------------------------------------------------------------------
    # Async routing
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_async_get_available_deployment_returns_deployment(self) -> None:
        """async_get_available_deployment returns a valid deployment dict."""
        router = _make_mock_router()
        strategy = RouteIQRoutingStrategy(
            router_instance=router,
            strategy_name="llmrouter-knn",
        )

        # Mock the ML strategy to return a known model
        mock_llmrouter = MagicMock()
        mock_llmrouter.route_with_observability.return_value = "openai/gpt-4"
        strategy._strategy_instance = mock_llmrouter

        # Disable pipeline to test direct routing
        with patch(
            "litellm_llmrouter.custom_routing_strategy.USE_PIPELINE_ROUTING",
            False,
        ):
            result = await strategy.async_get_available_deployment(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
            )

        assert result is not None
        assert result["litellm_params"]["model"] == "openai/gpt-4"

    @pytest.mark.asyncio
    async def test_async_fallback_when_ml_returns_none(self) -> None:
        """Falls back to first deployment when ML routing returns None."""
        router = _make_mock_router()
        strategy = RouteIQRoutingStrategy(
            router_instance=router,
            strategy_name="llmrouter-knn",
        )

        mock_llmrouter = MagicMock()
        mock_llmrouter.route_with_observability.return_value = None
        strategy._strategy_instance = mock_llmrouter

        with patch(
            "litellm_llmrouter.custom_routing_strategy.USE_PIPELINE_ROUTING",
            False,
        ):
            result = await strategy.async_get_available_deployment(model="gpt-4")

        # Should fall back to first deployment matching "gpt-4"
        assert result is not None
        assert result["model_name"] == "gpt-4"

    # ------------------------------------------------------------------
    # Sync routing
    # ------------------------------------------------------------------

    def test_sync_get_available_deployment_returns_deployment(self) -> None:
        """get_available_deployment returns a valid deployment dict."""
        router = _make_mock_router()
        strategy = RouteIQRoutingStrategy(
            router_instance=router,
            strategy_name="llmrouter-knn",
        )

        mock_llmrouter = MagicMock()
        mock_llmrouter.route_with_observability.return_value = "anthropic/claude-3-opus"
        strategy._strategy_instance = mock_llmrouter

        with patch(
            "litellm_llmrouter.custom_routing_strategy.USE_PIPELINE_ROUTING",
            False,
        ):
            result = strategy.get_available_deployment(
                model="gpt-4",
                messages=[{"role": "user", "content": "Tell me a story"}],
            )

        assert result is not None
        assert result["litellm_params"]["model"] == "anthropic/claude-3-opus"

    def test_sync_fallback_to_first_deployment(self) -> None:
        """Falls back to first deployment when everything fails."""
        router = _make_mock_router()
        strategy = RouteIQRoutingStrategy(
            router_instance=router,
            strategy_name=None,  # No ML strategy
        )

        with patch(
            "litellm_llmrouter.custom_routing_strategy.USE_PIPELINE_ROUTING",
            False,
        ):
            result = strategy.get_available_deployment(model="gpt-4")

        # No ML strategy -> falls back to first deployment
        assert result is not None
        assert result["model_name"] == "gpt-4"

    # ------------------------------------------------------------------
    # Amplification guard
    # ------------------------------------------------------------------

    def test_amplification_guard_blocks_excess_attempts(self) -> None:
        """Raises RuntimeError after MAX_ROUTING_ATTEMPTS for same request."""
        router = _make_mock_router()
        strategy = RouteIQRoutingStrategy(router_instance=router)

        kwargs = {"litellm_call_id": "req-123"}

        with patch(
            "litellm_llmrouter.custom_routing_strategy.USE_PIPELINE_ROUTING",
            False,
        ):
            # First MAX_ROUTING_ATTEMPTS calls should succeed
            for _ in range(MAX_ROUTING_ATTEMPTS):
                strategy.get_available_deployment(
                    model="gpt-4",
                    request_kwargs=kwargs,
                )

            # Next call should raise
            with pytest.raises(RuntimeError, match="amplification guard"):
                strategy.get_available_deployment(
                    model="gpt-4",
                    request_kwargs=kwargs,
                )

    def test_amplification_guard_uses_request_id_fallback(self) -> None:
        """Guard also checks request_id when litellm_call_id is absent."""
        router = _make_mock_router()
        strategy = RouteIQRoutingStrategy(router_instance=router)

        kwargs = {"request_id": "req-456"}

        with patch(
            "litellm_llmrouter.custom_routing_strategy.USE_PIPELINE_ROUTING",
            False,
        ):
            for _ in range(MAX_ROUTING_ATTEMPTS):
                strategy.get_available_deployment(
                    model="gpt-4",
                    request_kwargs=kwargs,
                )

            with pytest.raises(RuntimeError, match="amplification guard"):
                strategy.get_available_deployment(
                    model="gpt-4",
                    request_kwargs=kwargs,
                )

    def test_amplification_guard_no_kwargs(self) -> None:
        """No request_kwargs should not trigger the guard."""
        router = _make_mock_router()
        strategy = RouteIQRoutingStrategy(router_instance=router)

        with patch(
            "litellm_llmrouter.custom_routing_strategy.USE_PIPELINE_ROUTING",
            False,
        ):
            # Should not raise even after many calls
            for _ in range(10):
                strategy.get_available_deployment(model="gpt-4")

    def test_amplification_guard_bounded_cleanup(self) -> None:
        """Guard cleans up tracking dict when it exceeds 10k entries."""
        router = _make_mock_router()
        strategy = RouteIQRoutingStrategy(router_instance=router)

        # Fill it up
        strategy._routing_attempts = {f"req-{i}": 1 for i in range(10_001)}

        with patch(
            "litellm_llmrouter.custom_routing_strategy.USE_PIPELINE_ROUTING",
            False,
        ):
            # Triggering the guard should clear the dict
            strategy.get_available_deployment(
                model="gpt-4",
                request_kwargs={"litellm_call_id": "new-req"},
            )
            # After cleanup, dict is cleared entirely (including new entry)
            assert len(strategy._routing_attempts) == 0

    # ------------------------------------------------------------------
    # Pipeline routing
    # ------------------------------------------------------------------

    def test_pipeline_routing_takes_precedence(self) -> None:
        """When pipeline returns a valid deployment, ML routing is not called."""
        router = _make_mock_router()
        strategy = RouteIQRoutingStrategy(
            router_instance=router,
            strategy_name="llmrouter-knn",
        )

        # Mock pipeline to return a specific deployment
        mock_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.deployment = _DEPLOYMENT_CLAUDE
        mock_pipeline.route.return_value = mock_result
        strategy._pipeline = mock_pipeline
        strategy._pipeline_initialized = True

        # ML strategy should NOT be called
        mock_llmrouter = MagicMock()
        strategy._strategy_instance = mock_llmrouter

        result = strategy.get_available_deployment(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hi"}],
            request_kwargs={"user": "test-user"},
        )

        assert result is _DEPLOYMENT_CLAUDE
        mock_llmrouter.route_with_observability.assert_not_called()

    def test_fallback_to_direct_llmrouter(self) -> None:
        """When pipeline returns None, falls back to direct ML routing."""
        router = _make_mock_router()
        strategy = RouteIQRoutingStrategy(
            router_instance=router,
            strategy_name="llmrouter-knn",
        )

        # Pipeline returns None
        mock_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.deployment = None
        mock_pipeline.route.return_value = mock_result
        strategy._pipeline = mock_pipeline
        strategy._pipeline_initialized = True

        # ML strategy returns a model
        mock_llmrouter = MagicMock()
        mock_llmrouter.route_with_observability.return_value = "openai/gpt-4"
        strategy._strategy_instance = mock_llmrouter

        result = strategy.get_available_deployment(
            model="gpt-4",
            messages=[{"role": "user", "content": "Test"}],
        )

        assert result is not None
        assert result["litellm_params"]["model"] == "openai/gpt-4"
        mock_llmrouter.route_with_observability.assert_called_once()

    def test_fallback_to_first_deployment(self) -> None:
        """When all routing fails, returns first matching deployment."""
        router = _make_mock_router()
        strategy = RouteIQRoutingStrategy(
            router_instance=router,
            strategy_name="llmrouter-knn",
        )

        # Pipeline raises
        mock_pipeline = MagicMock()
        mock_pipeline.route.side_effect = Exception("pipeline broken")
        strategy._pipeline = mock_pipeline
        strategy._pipeline_initialized = True

        # ML strategy also fails
        mock_llmrouter = MagicMock()
        mock_llmrouter.route_with_observability.side_effect = Exception("ML broken")
        strategy._strategy_instance = mock_llmrouter

        result = strategy.get_available_deployment(model="gpt-4")

        assert result is not None
        assert result["model_name"] == "gpt-4"

    # ------------------------------------------------------------------
    # Query extraction
    # ------------------------------------------------------------------

    def test_extract_query_from_messages(self) -> None:
        """Extracts concatenated text from chat messages."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is Python?"},
        ]
        result = RouteIQRoutingStrategy._extract_query(messages=messages)
        assert "You are helpful." in result
        assert "What is Python?" in result

    def test_extract_query_from_multimodal_messages(self) -> None:
        """Extracts text from multi-modal message format."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": {"url": "http://example.com"}},
                ],
            },
        ]
        result = RouteIQRoutingStrategy._extract_query(messages=messages)
        assert result == "Describe this image"

    def test_extract_query_from_input_string(self) -> None:
        """Extracts query from a plain string input."""
        result = RouteIQRoutingStrategy._extract_query(input="embed this text")
        assert result == "embed this text"

    def test_extract_query_from_input_list(self) -> None:
        """Extracts query from a list input."""
        result = RouteIQRoutingStrategy._extract_query(input=["foo", "bar"])
        assert result == "foo bar"

    def test_extract_query_empty(self) -> None:
        """Returns empty string when no messages or input provided."""
        result = RouteIQRoutingStrategy._extract_query()
        assert result == ""

    # ------------------------------------------------------------------
    # Deployment matching
    # ------------------------------------------------------------------

    def test_match_deployment_exact(self) -> None:
        """Exact match on litellm_params.model works."""
        deployments = [_DEPLOYMENT_GPT4, _DEPLOYMENT_CLAUDE]
        result = RouteIQRoutingStrategy._match_deployment(
            "anthropic/claude-3-opus",
            "gpt-4",
            deployments,
        )
        assert result is _DEPLOYMENT_CLAUDE

    def test_match_deployment_partial(self) -> None:
        """Partial/substring match works when exact match fails."""
        deployments = [_DEPLOYMENT_GPT4, _DEPLOYMENT_CLAUDE, _DEPLOYMENT_HAIKU]
        # "claude-3-opus" is a substring of "anthropic/claude-3-opus"
        result = RouteIQRoutingStrategy._match_deployment(
            "claude-3-opus",
            "gpt-4",
            deployments,
        )
        assert result is _DEPLOYMENT_CLAUDE

    def test_match_deployment_fallback_to_model_group(self) -> None:
        """Falls back to first deployment in the model group when no match."""
        deployments = [_DEPLOYMENT_GPT4, _DEPLOYMENT_CLAUDE]
        result = RouteIQRoutingStrategy._match_deployment(
            "unknown-model",
            "gpt-4",
            deployments,
        )
        assert result is _DEPLOYMENT_GPT4

    def test_match_deployment_returns_none_no_match(self) -> None:
        """Returns None when no deployments match at all."""
        deployments = [_DEPLOYMENT_GPT4]
        result = RouteIQRoutingStrategy._match_deployment(
            "unknown-model",
            "other-group",
            deployments,
        )
        assert result is None

    # ------------------------------------------------------------------
    # Model list extraction
    # ------------------------------------------------------------------

    def test_get_model_list_from_healthy_deployments(self) -> None:
        """_get_model_list prefers healthy_deployments over model_list."""
        router = _make_mock_router(
            model_list=[_DEPLOYMENT_GPT4],
            healthy_deployments=[_DEPLOYMENT_GPT4, _DEPLOYMENT_CLAUDE],
        )
        strategy = RouteIQRoutingStrategy(router_instance=router)

        model_list, healthy = strategy._get_model_list("gpt-4")
        assert "openai/gpt-4" in model_list
        assert "anthropic/claude-3-opus" in model_list
        assert len(healthy) == 2

    def test_get_model_list_empty(self) -> None:
        """Returns empty list when no deployments match the model."""
        router = _make_mock_router()
        strategy = RouteIQRoutingStrategy(router_instance=router)

        model_list, _ = strategy._get_model_list("nonexistent-model")
        assert model_list == []


# ======================================================================
# TestInstallation
# ======================================================================


class TestInstallation:
    """Tests for create_routeiq_strategy and install_routeiq_strategy."""

    def test_create_routeiq_strategy(self) -> None:
        """Factory creates a properly configured strategy."""
        router = _make_mock_router()
        strategy = create_routeiq_strategy(router, "llmrouter-svm")

        assert isinstance(strategy, RouteIQRoutingStrategy)
        assert strategy._router is router
        assert strategy._strategy_name == "llmrouter-svm"

    def test_install_routeiq_strategy(self) -> None:
        """install_routeiq_strategy calls set_custom_routing_strategy on the router."""
        router = _make_mock_router()
        strategy = install_routeiq_strategy(router, "llmrouter-knn")

        # Verify set_custom_routing_strategy was called
        router.set_custom_routing_strategy.assert_called_once_with(strategy)

    def test_install_replaces_router_methods(self) -> None:
        """After installation, Router's methods point to our strategy."""
        router = _make_mock_router()
        strategy = install_routeiq_strategy(router, "llmrouter-knn")

        # The mock's side_effect replaces methods
        assert router.get_available_deployment == strategy.get_available_deployment
        assert (
            router.async_get_available_deployment
            == strategy.async_get_available_deployment
        )

    def test_strategy_receives_correct_kwargs(self) -> None:
        """Verify request_kwargs flow through to the strategy."""
        router = _make_mock_router()
        strategy = install_routeiq_strategy(router, "llmrouter-knn")

        # Mock internal routing to avoid actual ML calls
        mock_llmrouter = MagicMock()
        mock_llmrouter.route_with_observability.return_value = "openai/gpt-4"
        strategy._strategy_instance = mock_llmrouter

        request_kwargs = {
            "litellm_call_id": "test-call-1",
            "user": "test-user",
            "metadata": {"request_id": "meta-req-1"},
        }

        with patch(
            "litellm_llmrouter.custom_routing_strategy.USE_PIPELINE_ROUTING",
            False,
        ):
            result = router.get_available_deployment(
                model="gpt-4",
                messages=[{"role": "user", "content": "hello"}],
                request_kwargs=request_kwargs,
            )

        assert result is not None
        # Verify the ML strategy was called with extracted query
        mock_llmrouter.route_with_observability.assert_called_once()
        call_args = mock_llmrouter.route_with_observability.call_args
        assert "hello" in call_args[0][0]  # query text

    def test_install_without_set_custom_method(self) -> None:
        """Gracefully handles router without set_custom_routing_strategy."""
        router = MagicMock(spec=[])  # No methods
        router.model_list = [_DEPLOYMENT_GPT4]

        # Should not raise, just warn
        strategy = install_routeiq_strategy(router, "llmrouter-knn")
        assert isinstance(strategy, RouteIQRoutingStrategy)

    def test_create_without_strategy_name(self) -> None:
        """Factory works without a strategy name (fallback-only mode)."""
        router = _make_mock_router()
        strategy = create_routeiq_strategy(router)

        assert strategy._strategy_name is None


# ======================================================================
# TestEdgeCases
# ======================================================================


class TestEdgeCases:
    """Edge case and integration-style tests."""

    def test_no_deployments_returns_none(self) -> None:
        """Returns None when model_list is empty."""
        router = _make_mock_router(model_list=[], healthy_deployments=[])
        strategy = RouteIQRoutingStrategy(router_instance=router)

        with patch(
            "litellm_llmrouter.custom_routing_strategy.USE_PIPELINE_ROUTING",
            False,
        ):
            result = strategy.get_available_deployment(model="gpt-4")

        assert result is None

    @pytest.mark.asyncio
    async def test_async_routing_with_ml_exception(self) -> None:
        """Async routing falls back gracefully when ML strategy raises."""
        router = _make_mock_router()
        strategy = RouteIQRoutingStrategy(
            router_instance=router,
            strategy_name="llmrouter-knn",
        )

        mock_llmrouter = MagicMock()
        mock_llmrouter.route_with_observability.side_effect = ValueError(
            "model not loaded"
        )
        strategy._strategy_instance = mock_llmrouter

        with patch(
            "litellm_llmrouter.custom_routing_strategy.USE_PIPELINE_ROUTING",
            False,
        ):
            result = await strategy.async_get_available_deployment(model="gpt-4")

        # Should fall back to first deployment
        assert result is not None
        assert result["model_name"] == "gpt-4"

    def test_pipeline_import_error_falls_through(self) -> None:
        """When pipeline import fails, falls through to ML routing."""
        router = _make_mock_router()
        strategy = RouteIQRoutingStrategy(
            router_instance=router,
            strategy_name="llmrouter-knn",
        )

        mock_llmrouter = MagicMock()
        mock_llmrouter.route_with_observability.return_value = "openai/gpt-4"
        strategy._strategy_instance = mock_llmrouter

        # Force pipeline to raise ImportError
        with patch(
            "litellm_llmrouter.custom_routing_strategy.RouteIQRoutingStrategy._route_via_pipeline",
            return_value=None,
        ):
            result = strategy.get_available_deployment(
                model="gpt-4",
                messages=[{"role": "user", "content": "test"}],
            )

        assert result is not None
        assert result["litellm_params"]["model"] == "openai/gpt-4"

    def test_multiple_requests_different_ids(self) -> None:
        """Different request IDs should each get their own amplification count."""
        router = _make_mock_router()
        strategy = RouteIQRoutingStrategy(router_instance=router)

        with patch(
            "litellm_llmrouter.custom_routing_strategy.USE_PIPELINE_ROUTING",
            False,
        ):
            for i in range(MAX_ROUTING_ATTEMPTS):
                strategy.get_available_deployment(
                    model="gpt-4",
                    request_kwargs={"litellm_call_id": f"req-{i}"},
                )

        # Each request ID only used once, so no amplification error
        assert len(strategy._routing_attempts) == MAX_ROUTING_ATTEMPTS
