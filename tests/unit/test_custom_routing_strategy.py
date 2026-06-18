"""
Unit tests for RouteIQRoutingStrategy — the official CustomRoutingStrategyBase
implementation that replaces the monkey-patch approach.

Tests cover:
- Constructor / state initialization
- Async and sync routing methods
- Amplification guard
- Pipeline routing precedence
- Direct LLMRouter routing fallback
- Centroid routing fallback (zero-config intelligent routing)
- Routing profile support (auto, eco, premium, etc.)
- Final fallback to first deployment
- Query extraction helpers
- Deployment matching (exact and partial)
- install_routeiq_strategy wiring
- Centroid strategy registration
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from litellm_llmrouter.custom_routing_strategy import (
    MAX_ROUTING_ATTEMPTS,
    RouteIQRoutingStrategy,
    _resolve_routing_profile,
    create_routeiq_strategy,
    install_routeiq_strategy,
    register_centroid_strategy,
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


# ======================================================================
# TestCentroidIntegration
# ======================================================================


class TestCentroidIntegration:
    """Tests for centroid routing integration in RouteIQRoutingStrategy."""

    def test_centroid_fallback_when_pipeline_and_ml_return_none(self) -> None:
        """Centroid routing is used when pipeline returns None and no ML strategy."""
        router = _make_mock_router(
            model_list=[_DEPLOYMENT_GPT4, _DEPLOYMENT_CLAUDE, _DEPLOYMENT_HAIKU],
            healthy_deployments=[
                _DEPLOYMENT_GPT4,
                _DEPLOYMENT_CLAUDE,
                _DEPLOYMENT_HAIKU,
            ],
        )
        strategy = RouteIQRoutingStrategy(
            router_instance=router,
            strategy_name=None,  # No ML strategy
        )

        # Mock centroid strategy to return a specific deployment
        mock_centroid = MagicMock()
        mock_centroid.select_deployment.return_value = _DEPLOYMENT_CLAUDE
        strategy._centroid_strategy = mock_centroid
        strategy._centroid_initialized = True

        with (
            patch(
                "litellm_llmrouter.custom_routing_strategy.USE_PIPELINE_ROUTING",
                False,
            ),
            patch(
                "litellm_llmrouter.custom_routing_strategy.CENTROID_ROUTING_ENABLED",
                True,
            ),
            patch(
                "litellm_llmrouter.custom_routing_strategy.CENTROID_ROUTING_AVAILABLE",
                True,
            ),
        ):
            result = strategy.get_available_deployment(
                model="gpt-4",
                messages=[{"role": "user", "content": "Explain quantum physics"}],
            )

        assert result is _DEPLOYMENT_CLAUDE
        mock_centroid.select_deployment.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_centroid_fallback(self) -> None:
        """Async routing also falls back to centroid when ML returns None."""
        router = _make_mock_router()
        strategy = RouteIQRoutingStrategy(
            router_instance=router,
            strategy_name=None,
        )

        mock_centroid = MagicMock()
        mock_centroid.select_deployment.return_value = _DEPLOYMENT_GPT4
        strategy._centroid_strategy = mock_centroid
        strategy._centroid_initialized = True

        with (
            patch(
                "litellm_llmrouter.custom_routing_strategy.USE_PIPELINE_ROUTING",
                False,
            ),
            patch(
                "litellm_llmrouter.custom_routing_strategy.CENTROID_ROUTING_ENABLED",
                True,
            ),
            patch(
                "litellm_llmrouter.custom_routing_strategy.CENTROID_ROUTING_AVAILABLE",
                True,
            ),
        ):
            result = await strategy.async_get_available_deployment(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
            )

        assert result is _DEPLOYMENT_GPT4

    def test_centroid_not_called_when_ml_succeeds(self) -> None:
        """Centroid routing is NOT called when ML strategy succeeds."""
        router = _make_mock_router()
        strategy = RouteIQRoutingStrategy(
            router_instance=router,
            strategy_name="llmrouter-knn",
        )

        # ML strategy succeeds
        mock_llmrouter = MagicMock()
        mock_llmrouter.route_with_observability.return_value = "openai/gpt-4"
        strategy._strategy_instance = mock_llmrouter

        # Centroid should not be called
        mock_centroid = MagicMock()
        strategy._centroid_strategy = mock_centroid
        strategy._centroid_initialized = True

        with patch(
            "litellm_llmrouter.custom_routing_strategy.USE_PIPELINE_ROUTING",
            False,
        ):
            result = strategy.get_available_deployment(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hi"}],
            )

        assert result is not None
        mock_centroid.select_deployment.assert_not_called()

    def test_centroid_disabled_falls_through_to_fallback(self) -> None:
        """When centroid is disabled, falls through to first healthy deployment."""
        router = _make_mock_router()
        strategy = RouteIQRoutingStrategy(
            router_instance=router,
            strategy_name=None,
        )

        with (
            patch(
                "litellm_llmrouter.custom_routing_strategy.USE_PIPELINE_ROUTING",
                False,
            ),
            patch(
                "litellm_llmrouter.custom_routing_strategy.CENTROID_ROUTING_ENABLED",
                False,  # Disabled
            ),
        ):
            result = strategy.get_available_deployment(model="gpt-4")

        # Falls through to first healthy deployment
        assert result is not None
        assert result["model_name"] == "gpt-4"

    def test_centroid_unavailable_falls_through(self) -> None:
        """When centroid is unavailable (import error), falls through gracefully."""
        router = _make_mock_router()
        strategy = RouteIQRoutingStrategy(
            router_instance=router,
            strategy_name=None,
        )

        with (
            patch(
                "litellm_llmrouter.custom_routing_strategy.USE_PIPELINE_ROUTING",
                False,
            ),
            patch(
                "litellm_llmrouter.custom_routing_strategy.CENTROID_ROUTING_AVAILABLE",
                False,  # Unavailable
            ),
        ):
            result = strategy.get_available_deployment(model="gpt-4")

        assert result is not None
        assert result["model_name"] == "gpt-4"

    def test_centroid_exception_falls_through(self) -> None:
        """When centroid routing raises an exception, falls through gracefully."""
        router = _make_mock_router()
        strategy = RouteIQRoutingStrategy(
            router_instance=router,
            strategy_name=None,
        )

        mock_centroid = MagicMock()
        mock_centroid.select_deployment.side_effect = RuntimeError("centroid broken")
        strategy._centroid_strategy = mock_centroid
        strategy._centroid_initialized = True

        with (
            patch(
                "litellm_llmrouter.custom_routing_strategy.USE_PIPELINE_ROUTING",
                False,
            ),
            patch(
                "litellm_llmrouter.custom_routing_strategy.CENTROID_ROUTING_ENABLED",
                True,
            ),
            patch(
                "litellm_llmrouter.custom_routing_strategy.CENTROID_ROUTING_AVAILABLE",
                True,
            ),
        ):
            result = strategy.get_available_deployment(model="gpt-4")

        # Should fall back to first healthy deployment
        assert result is not None
        assert result["model_name"] == "gpt-4"


# ======================================================================
# TestRoutingProfile
# ======================================================================


class TestRoutingProfile:
    """Tests for routing profile resolution."""

    def test_profile_from_request_metadata(self) -> None:
        """Profile is extracted from request metadata."""
        result = _resolve_routing_profile(
            request_kwargs={"metadata": {"routing_profile": "eco"}}
        )
        assert result == "eco"

    def test_profile_from_request_metadata_case_insensitive(self) -> None:
        """Profile from metadata is lowercased."""
        result = _resolve_routing_profile(
            request_kwargs={"metadata": {"routing_profile": "PREMIUM"}}
        )
        assert result == "premium"

    def test_profile_default_env_var(self) -> None:
        """Falls back to ROUTEIQ_ROUTING_PROFILE env var."""
        with patch(
            "litellm_llmrouter.custom_routing_strategy.DEFAULT_ROUTING_PROFILE",
            "eco",
        ):
            result = _resolve_routing_profile(request_kwargs={})
        assert result == "eco"

    def test_profile_default_auto(self) -> None:
        """Default profile is 'auto' when no metadata or env var."""
        result = _resolve_routing_profile(request_kwargs=None)
        assert result == "auto"

    def test_profile_metadata_takes_precedence(self) -> None:
        """Request metadata profile overrides env var default."""
        with patch(
            "litellm_llmrouter.custom_routing_strategy.DEFAULT_ROUTING_PROFILE",
            "eco",
        ):
            result = _resolve_routing_profile(
                request_kwargs={"metadata": {"routing_profile": "premium"}}
            )
        assert result == "premium"

    def test_profile_empty_metadata_uses_default(self) -> None:
        """Empty metadata falls back to default."""
        result = _resolve_routing_profile(request_kwargs={"metadata": {}})
        assert result == "auto"


# ======================================================================
# TestCentroidRegistration
# ======================================================================


class TestCentroidRegistration:
    """Tests for register_centroid_strategy function."""

    def test_register_centroid_disabled(self) -> None:
        """Registration returns False when centroid is disabled."""
        with patch(
            "litellm_llmrouter.custom_routing_strategy.CENTROID_ROUTING_ENABLED",
            False,
        ):
            assert register_centroid_strategy() is False

    def test_register_centroid_unavailable(self) -> None:
        """Registration returns False when centroid is unavailable."""
        with patch(
            "litellm_llmrouter.custom_routing_strategy.CENTROID_ROUTING_AVAILABLE",
            False,
        ):
            assert register_centroid_strategy() is False

    def test_register_centroid_success(self) -> None:
        """Registration succeeds when enabled and available."""
        mock_registry = MagicMock()

        with (
            patch(
                "litellm_llmrouter.custom_routing_strategy.CENTROID_ROUTING_ENABLED",
                True,
            ),
            patch(
                "litellm_llmrouter.custom_routing_strategy.CENTROID_ROUTING_AVAILABLE",
                True,
            ),
            patch(
                "litellm_llmrouter.strategy_registry.get_routing_registry",
                return_value=mock_registry,
            ),
        ):
            result = register_centroid_strategy()

        assert result is True
        # Verify registry.register was called with the correct strategy name
        mock_registry.register.assert_called_once()
        call_args = mock_registry.register.call_args
        assert call_args[0][0] == "llmrouter-nadirclaw-centroid"

    def test_register_centroid_handles_exception(self) -> None:
        """Registration returns False when registry raises."""
        with (
            patch(
                "litellm_llmrouter.custom_routing_strategy.CENTROID_ROUTING_ENABLED",
                True,
            ),
            patch(
                "litellm_llmrouter.custom_routing_strategy.CENTROID_ROUTING_AVAILABLE",
                True,
            ),
            patch(
                "litellm_llmrouter.custom_routing_strategy.get_centroid_strategy",
                side_effect=RuntimeError("registry broken"),
            ),
        ):
            assert register_centroid_strategy() is False


# ======================================================================
# TestPreScoringCandidateFilter — RouteIQ-99e8 (cooldown) + RouteIQ-badb (gov-ban)
# ======================================================================
#
# The seed bug lived in custom_routing_strategy.py: the custom-strategy path
# (set_custom_routing_strategy hot-swaps async_get_available_deployment) BYPASSES
# LiteLLM's built-in healthy-deployment pipeline (_get_healthy_deployments ->
# _filter_cooldown_deployments) AND its async_filter_deployments hook. So every
# RouteIQ candidate source read the STATIC `healthy_deployments` alias, which is
# neither cooldown-aware nor gov-ban-aware. RouteIQRoutingStrategy then scored
# over that full static set and could select a cooled-down / gov-banned arm,
# failing only AFTER selection (reactive retry, not proactive exclusion).
#
# These tests assert the fix THROUGH the custom_routing_strategy.py seam itself:
# _get_model_list (the direct ML-routing candidate path) and _fallback_deployment
# (the final fallback) now run filter_routable_candidates() BEFORE returning the
# candidate set, so a cooled-down / gov-banned arm never enters the scored set.


class TestPreScoringCandidateFilter:
    """RouteIQ-99e8 + RouteIQ-badb at the custom_routing_strategy.py seam."""

    _FABLE5 = "bedrock/global.anthropic.claude-fable-5"

    @staticmethod
    def _dep(model_name: str, arm: str, dep_id: str) -> Dict[str, Any]:
        return {
            "model_name": model_name,
            "litellm_params": {"model": arm},
            "model_info": {"id": dep_id},
        }

    # ------------------------------------------------------------------
    # RouteIQ-99e8 — cooldown excluded from _get_model_list candidate set
    # ------------------------------------------------------------------

    def test_get_model_list_excludes_cooled_down_arm(self, monkeypatch) -> None:
        """99e8 acceptance: a cooled-down deployment is EXCLUDED from the
        candidate set _get_model_list returns to the strategy (not retried
        after selection)."""
        live = self._dep("gpt-4", "openai/gpt-4", "d1")
        cooled = self._dep("gpt-4", "anthropic/claude-3-opus", "d2")
        router = _make_mock_router(
            model_list=[live, cooled],
            healthy_deployments=[live, cooled],
        )
        # d2 is in LiteLLM's live cooldown set.
        monkeypatch.setattr(
            "litellm_llmrouter.candidate_filter.cooled_down_ids",
            lambda router: {"d2"},
        )
        strategy = RouteIQRoutingStrategy(router_instance=router)
        model_list, _ = strategy._get_model_list("gpt-4")
        # The cooled-down arm's litellm model is gone from the scored set;
        # the healthy arm remains.
        assert "openai/gpt-4" in model_list
        assert "anthropic/claude-3-opus" not in model_list

    def test_get_model_list_cooldown_fail_open_when_all_cooled(
        self, monkeypatch
    ) -> None:
        """If EVERY arm is cooled down, _get_model_list still returns candidates
        (fail-open: availability over a zero-candidate dead-end)."""
        a = self._dep("gpt-4", "openai/gpt-4", "d1")
        b = self._dep("gpt-4", "anthropic/claude-3-opus", "d2")
        router = _make_mock_router(model_list=[a, b], healthy_deployments=[a, b])
        monkeypatch.setattr(
            "litellm_llmrouter.candidate_filter.cooled_down_ids",
            lambda router: {"d1", "d2"},
        )
        strategy = RouteIQRoutingStrategy(router_instance=router)
        model_list, _ = strategy._get_model_list("gpt-4")
        assert set(model_list) == {"openai/gpt-4", "anthropic/claude-3-opus"}

    def test_fallback_deployment_skips_cooled_down_arm(self, monkeypatch) -> None:
        """_fallback_deployment (the final fallback) also never returns a
        cooled-down arm — the fallback path also bypasses LiteLLM's pipeline."""
        cooled = self._dep("gpt-4", "openai/gpt-4", "d1")
        live = self._dep("gpt-4", "anthropic/claude-3-opus", "d2")
        router = _make_mock_router(
            model_list=[cooled, live], healthy_deployments=[cooled, live]
        )
        monkeypatch.setattr(
            "litellm_llmrouter.candidate_filter.cooled_down_ids",
            lambda router: {"d1"},  # the FIRST listed arm is cooled
        )
        strategy = RouteIQRoutingStrategy(router_instance=router)
        result = strategy._fallback_deployment("gpt-4")
        # Without the fix the first (cooled) arm would be returned; the fix
        # skips it and returns the live arm.
        assert result is not None
        assert result["litellm_params"]["model"] == "anthropic/claude-3-opus"

    # ------------------------------------------------------------------
    # RouteIQ-badb — gov-banned arm never enters the candidate set
    # ------------------------------------------------------------------

    def test_get_model_list_excludes_gov_banned_arm(self) -> None:
        """badb acceptance: a gov-banned arm (Fable 5) never appears in the
        candidate set _get_model_list returns to the strategy/bandit."""
        from litellm_llmrouter.settings import get_settings, reset_settings

        reset_settings()
        get_settings(governance={"banned_models": [self._FABLE5]})
        legal = self._dep("g", "bedrock/anthropic.claude-3-sonnet", "d1")
        banned = self._dep("g", self._FABLE5, "d2")
        router = _make_mock_router(
            model_list=[legal, banned], healthy_deployments=[legal, banned]
        )
        strategy = RouteIQRoutingStrategy(router_instance=router)
        model_list, _ = strategy._get_model_list("g")
        assert "bedrock/anthropic.claude-3-sonnet" in model_list
        assert self._FABLE5 not in model_list

    def test_fallback_deployment_never_returns_gov_banned_sole_arm(self) -> None:
        """Compliance fail-closed-to-removal: when the only candidate is gov-
        banned, _fallback_deployment returns None — never a banned arm."""
        from litellm_llmrouter.settings import get_settings, reset_settings

        reset_settings()
        get_settings(governance={"banned_models": [self._FABLE5]})
        banned = self._dep("g", self._FABLE5, "d1")
        router = _make_mock_router(model_list=[banned], healthy_deployments=[banned])
        strategy = RouteIQRoutingStrategy(router_instance=router)
        assert strategy._fallback_deployment("g") is None

    def test_get_model_list_default_no_ban_byte_stable(self) -> None:
        """Default-empty ban + no cooldown -> the candidate set is unchanged
        (byte-stable no-op); both arms remain scorable."""
        from litellm_llmrouter.settings import get_settings, reset_settings

        reset_settings()
        get_settings()  # default GovernanceSettings -> banned_models == []
        a = self._dep("g", "bedrock/a", "d1")
        b = self._dep("g", "bedrock/b", "d2")
        router = _make_mock_router(model_list=[a, b], healthy_deployments=[a, b])
        strategy = RouteIQRoutingStrategy(router_instance=router)
        model_list, _ = strategy._get_model_list("g")
        assert set(model_list) == {"bedrock/a", "bedrock/b"}


# ======================================================================
# RouteIQ-5007 — FAIL-OPEN data-residency leak on the ML path
# ======================================================================
#
# The region pre-filter (RouteIQ-60cc) correctly EMPTIES the candidate set for a
# HARD-residency request that has no in-region arm. But _route_via_llmrouter then
# did `if not model_list: model_list = [model]` — DISCARDING the fail-closed empty
# verdict — and _match_deployment was handed the FULL UNFILTERED
# healthy_deployments, whose group-fallback branch returned the FIRST out-of-region
# arm. Net: a hard-residency request leaked an out-of-region deployment on the
# primary ML path. _guard_selected only re-checks gov-ban, NOT region, so it could
# not catch the leak.
#
# The existing region tests only cover DefaultStrategy._get_deployments and
# KumaraswamyThompson.select_deployment — the RouteIQRoutingStrategy ML path was
# UNTESTED, which is why this slipped. These tests drive the REAL
# _route_via_llmrouter AND async_get_available_deployment end-to-end (a controlled
# _strategy_instance stub stands in for the ML scorer, exactly as the other ML-path
# tests do) and assert NO out-of-region arm is ever returned.


class _StubScorer:
    """Deterministic stand-in for LLMRouterStrategyFamily.route_with_observability.

    Mirrors what a real ML strategy does when handed a candidate list: returns a
    chosen model name from it (here, configurable). Used to drive the REAL
    _route_via_llmrouter without ML deps."""

    def __init__(self, returns: Any) -> None:
        self._returns = returns
        self.seen_model_lists: List[List[str]] = []

    def route_with_observability(self, query: str, model_list: List[str]) -> Any:
        self.seen_model_lists.append(list(model_list))
        # Echo the configured return, or (callable) compute it from the list.
        if callable(self._returns):
            return self._returns(model_list)
        return self._returns


class TestRegionResidencyMLPath:
    """RouteIQ-5007: the ML (LLMRouterStrategyFamily) path must fail CLOSED for a
    HARD-residency request with no in-region arm — never leak out-of-region."""

    @staticmethod
    def _dep(arm: str, region: str, dep_id: str, model_name: str = "claude") -> Dict:
        return {
            "model_name": model_name,
            "litellm_params": {"model": arm, "aws_region_name": region},
            "model_info": {"id": dep_id},
        }

    @staticmethod
    def _configure_hard_eu() -> None:
        from litellm_llmrouter.settings import get_settings, reset_settings

        reset_settings()
        get_settings(
            region_routing={
                "enabled": True,
                "region_header": "X-RouteIQ-Region",
                "region_map": {"eu": ["eu-west-1", "eu-central-1"]},
            }
        )

    @staticmethod
    def _hard_eu_kwargs() -> Dict[str, Any]:
        # request_kwargs is the FIRST source the region filter scans; the header
        # carries the region token and a truthy residency flag makes it HARD.
        return {"headers": {"X-RouteIQ-Region": "eu"}, "residency": True}

    def _strategy(self, deployments: List[Dict], scorer: _StubScorer):
        router = _make_mock_router(
            model_list=deployments, healthy_deployments=deployments
        )
        strat = RouteIQRoutingStrategy(
            router_instance=router, strategy_name="llmrouter-knn"
        )
        strat._strategy_instance = scorer  # bypass lazy ML load
        return strat

    def test_route_via_llmrouter_fails_closed_no_in_region(self, monkeypatch) -> None:
        """ACCEPTANCE (the P8 repro): _route_via_llmrouter returns None for a
        HARD-residency eu request when only us-* arms exist — the fail-closed empty
        verdict is NOT repopulated to [model], and no out-of-region arm leaks."""
        monkeypatch.setattr(
            "litellm_llmrouter.candidate_filter.cooled_down_ids",
            lambda router: set(),
        )
        self._configure_hard_eu()
        us1 = self._dep("us-east/claude", "us-east-1", "d1")
        us2 = self._dep("us-west/claude", "us-west-2", "d2")
        # If the strategy WERE consulted with [model] (the bug), it would echo
        # the group name and _match_deployment would group-fall-back to us1.
        scorer = _StubScorer(returns="claude")
        strat = self._strategy([us1, us2], scorer)

        result = strat._route_via_llmrouter(
            model="claude",
            messages=[{"role": "user", "content": "Hello"}],
            request_kwargs=self._hard_eu_kwargs(),
        )

        assert result is None
        # The fail-closed branch returns BEFORE scoring -> the stub was never
        # handed the unfiltered [model] candidate set.
        assert scorer.seen_model_lists == []

    async def test_async_end_to_end_no_out_of_region_leak(self, monkeypatch) -> None:
        """ACCEPTANCE: async_get_available_deployment (the public entry LiteLLM
        calls) returns None / no out-of-region arm for a HARD-residency request
        with only us-* arms — through the full pipeline -> ML -> personalized ->
        centroid -> fallback chain AND the _guard_selected chokepoint."""
        monkeypatch.setattr(
            "litellm_llmrouter.candidate_filter.cooled_down_ids",
            lambda router: set(),
        )
        self._configure_hard_eu()
        us1 = self._dep("us-east/claude", "us-east-1", "d1")
        us2 = self._dep("us-west/claude", "us-west-2", "d2")
        scorer = _StubScorer(returns="claude")
        strat = self._strategy([us1, us2], scorer)

        with patch(
            "litellm_llmrouter.custom_routing_strategy.USE_PIPELINE_ROUTING",
            False,
        ):
            result = await strat.async_get_available_deployment(
                model="claude",
                messages=[{"role": "user", "content": "Hello"}],
                request_kwargs=self._hard_eu_kwargs(),
            )

        # No deployment may be returned (every arm is out-of-region for a HARD
        # residency request). Critically: NOT an out-of-region arm.
        assert result is None

    def test_sync_end_to_end_no_out_of_region_leak(self, monkeypatch) -> None:
        """ACCEPTANCE (sync entry): get_available_deployment likewise never leaks
        an out-of-region arm for a HARD-residency request."""
        monkeypatch.setattr(
            "litellm_llmrouter.candidate_filter.cooled_down_ids",
            lambda router: set(),
        )
        self._configure_hard_eu()
        us1 = self._dep("us-east/claude", "us-east-1", "d1")
        us2 = self._dep("us-west/claude", "us-west-2", "d2")
        scorer = _StubScorer(returns="claude")
        strat = self._strategy([us1, us2], scorer)

        with patch(
            "litellm_llmrouter.custom_routing_strategy.USE_PIPELINE_ROUTING",
            False,
        ):
            result = strat.get_available_deployment(
                model="claude",
                messages=[{"role": "user", "content": "Hello"}],
                request_kwargs=self._hard_eu_kwargs(),
            )

        assert result is None

    def test_match_pool_is_filtered_not_unfiltered(self, monkeypatch) -> None:
        """Even if the scorer (perversely) returns an out-of-region arm NAME, the
        match pool is the FILTERED routable set, so _match_deployment cannot reach
        the out-of-region arm — it returns the in-region arm or None."""
        monkeypatch.setattr(
            "litellm_llmrouter.candidate_filter.cooled_down_ids",
            lambda router: set(),
        )
        self._configure_hard_eu()
        us = self._dep("us-east/claude", "us-east-1", "d1")
        eu = self._dep("eu-west/claude", "eu-west-1", "d2")
        # The scorer maliciously names the out-of-region arm; the filtered pool
        # (eu only) means _match_deployment can never return us.
        scorer = _StubScorer(returns="us-east/claude")
        strat = self._strategy([us, eu], scorer)

        result = strat._route_via_llmrouter(
            model="claude",
            messages=[{"role": "user", "content": "Hello"}],
            request_kwargs=self._hard_eu_kwargs(),
        )

        # The scorer's out-of-region name has no match in the eu-only filtered
        # pool -> no exact/partial match, group-fallback yields the eu arm.
        assert result is not None
        assert result["litellm_params"]["aws_region_name"] == "eu-west-1"
        assert result["litellm_params"]["model"] != "us-east/claude"
        # The candidate set scored was the FILTERED (eu-only) set, not [model].
        assert scorer.seen_model_lists == [["eu-west/claude"]]

    def test_in_region_arm_is_selectable(self, monkeypatch) -> None:
        """CONTROL: a HARD-residency request DOES select the in-region arm when
        one exists (the filter is active, not a blanket empty)."""
        monkeypatch.setattr(
            "litellm_llmrouter.candidate_filter.cooled_down_ids",
            lambda router: set(),
        )
        self._configure_hard_eu()
        us = self._dep("us-east/claude", "us-east-1", "d1")
        eu = self._dep("eu-west/claude", "eu-west-1", "d2")
        scorer = _StubScorer(returns="eu-west/claude")
        strat = self._strategy([us, eu], scorer)

        result = strat._route_via_llmrouter(
            model="claude",
            messages=[{"role": "user", "content": "Hello"}],
            request_kwargs=self._hard_eu_kwargs(),
        )

        assert result is not None
        assert result["litellm_params"]["model"] == "eu-west/claude"
        assert result["litellm_params"]["aws_region_name"] == "eu-west-1"


class TestNoContextByteStable:
    """RouteIQ-5007 byte-stability: with NO per-request context the legacy [model]
    fallback + full-healthy_deployments match pool are preserved unchanged."""

    @staticmethod
    def _dep(arm: str, dep_id: str, model_name: str = "gpt-4") -> Dict:
        return {
            "model_name": model_name,
            "litellm_params": {"model": arm},
            "model_info": {"id": dep_id},
        }

    def test_no_context_empty_group_falls_back_to_model(self, monkeypatch) -> None:
        """No request_kwargs AND no messages (context is None) + a requested group
        with NO members (genuinely empty group): the legacy `model_list = [model]`
        fallback is preserved (NOT failed-closed), the stub is handed [model], and
        the group-fallback resolves against the full healthy_deployments. This is
        the byte-stable legacy behaviour that MUST remain."""
        monkeypatch.setattr(
            "litellm_llmrouter.candidate_filter.cooled_down_ids",
            lambda router: set(),
        )
        from litellm_llmrouter.settings import get_settings, reset_settings

        reset_settings()
        get_settings()  # region_routing disabled by default

        # healthy_deployments hold a DIFFERENT group; the requested "gpt-4" group
        # is empty -> model_list == [] with context None -> legacy [model] path.
        other = self._dep("openai/gpt-4o", "d1", model_name="other-group")
        router = _make_mock_router(model_list=[other], healthy_deployments=[other])
        # The scorer echoes the requested group name; the legacy match pool is the
        # FULL healthy_deployments, so a substring/partial match can resolve it.
        scorer = _StubScorer(returns="gpt-4")
        strat = RouteIQRoutingStrategy(
            router_instance=router, strategy_name="llmrouter-knn"
        )
        strat._strategy_instance = scorer

        # No messages, no request_kwargs -> context is None -> legacy path.
        result = strat._route_via_llmrouter(model="gpt-4")

        # Legacy: model_list was [] (empty group) but context is None, so it is
        # repopulated to ["gpt-4"] (NOT failed-closed) and the stub sees ["gpt-4"].
        assert scorer.seen_model_lists == [["gpt-4"]]
        # Partial match ("gpt-4" in "openai/gpt-4o") against the FULL set resolves;
        # this is the legacy behaviour preserved byte-stable.
        assert result is not None
        assert result["litellm_params"]["model"] == "openai/gpt-4o"

    def test_no_context_nonempty_group_scores_filtered_set(self, monkeypatch) -> None:
        """No context + a non-empty group: the strategy scores the gov-ban/cooldown
        filtered group (legacy), NOT [model], and the match pool is the full
        healthy_deployments (byte-stable, unchanged from before the fix)."""
        monkeypatch.setattr(
            "litellm_llmrouter.candidate_filter.cooled_down_ids",
            lambda router: set(),
        )
        from litellm_llmrouter.settings import get_settings, reset_settings

        reset_settings()
        get_settings()
        a = self._dep("openai/gpt-4", "d1", model_name="gpt-4")
        b = self._dep("anthropic/claude-3-opus", "d2", model_name="gpt-4")
        router = _make_mock_router(model_list=[a, b], healthy_deployments=[a, b])
        scorer = _StubScorer(returns="anthropic/claude-3-opus")
        strat = RouteIQRoutingStrategy(
            router_instance=router, strategy_name="llmrouter-knn"
        )
        strat._strategy_instance = scorer

        result = strat._route_via_llmrouter(model="gpt-4")

        # The scored set is the real group arms (NOT [model]).
        assert scorer.seen_model_lists == [["openai/gpt-4", "anthropic/claude-3-opus"]]
        assert result is not None
        assert result["litellm_params"]["model"] == "anthropic/claude-3-opus"
