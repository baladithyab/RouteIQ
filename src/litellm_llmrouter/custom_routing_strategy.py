"""
RouteIQ Custom Routing Strategy — Uses LiteLLM's official CustomRoutingStrategyBase API.

This module replaces the monkey-patch approach in routing_strategy_patch.py with LiteLLM's
official plugin API for custom routing strategies. It provides the same routing capabilities
(ML-based strategy selection, A/B testing, telemetry) without modifying LiteLLM's Router class.

Usage::

    from litellm_llmrouter.custom_routing_strategy import install_routeiq_strategy

    # After creating a LiteLLM Router instance:
    strategy = install_routeiq_strategy(router, strategy_name="llmrouter-knn")

    # Or create and install separately:
    from litellm_llmrouter.custom_routing_strategy import create_routeiq_strategy
    strategy = create_routeiq_strategy(router, strategy_name="llmrouter-knn")
    router.set_custom_routing_strategy(strategy)

The ``install_routeiq_strategy`` helper calls ``router.set_custom_routing_strategy()``
which replaces the Router's ``get_available_deployment`` and
``async_get_available_deployment`` methods with the strategy's implementations.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Union

# Import CustomRoutingStrategyBase with graceful fallback for test environments
try:
    from litellm.types.router import CustomRoutingStrategyBase
except ImportError:  # pragma: no cover – litellm may not be installed in test env

    class CustomRoutingStrategyBase:  # type: ignore[no-redef]
        """Stub base class when litellm is not available."""

        async def async_get_available_deployment(
            self,
            model: str,
            messages: Optional[List[Dict[str, str]]] = None,
            input: Optional[Union[str, List]] = None,
            specific_deployment: Optional[bool] = False,
            request_kwargs: Optional[Dict] = None,
        ):
            pass

        def get_available_deployment(
            self,
            model: str,
            messages: Optional[List[Dict[str, str]]] = None,
            input: Optional[Union[str, List]] = None,
            specific_deployment: Optional[bool] = False,
            request_kwargs: Optional[Dict] = None,
        ):
            pass


logger = logging.getLogger(__name__)

# Feature flag: Use pipeline routing (enables A/B testing)
# Set LLMROUTER_USE_PIPELINE=false to disable
USE_PIPELINE_ROUTING = os.getenv("LLMROUTER_USE_PIPELINE", "true").lower() == "true"

# Maximum routing attempts per request to prevent amplification loops
MAX_ROUTING_ATTEMPTS = 3


class RouteIQRoutingStrategy(CustomRoutingStrategyBase):
    """
    Custom routing strategy using LiteLLM's official ``CustomRoutingStrategyBase`` API.

    This class replaces the monkey-patch system in ``routing_strategy_patch.py`` by
    implementing LiteLLM's plugin API for custom routing. When installed via
    ``router.set_custom_routing_strategy()``, the Router delegates all deployment
    selection to this class.

    Features:
    - **ML-based routing** via ``LLMRouterStrategyFamily`` (KNN, SVM, MLP, etc.)
    - **A/B testing** via ``RoutingPipeline`` with deterministic hashing
    - **Amplification guard** prevents infinite routing loops (max 3 per request)
    - **Graceful fallback** to first available deployment if ML routing fails

    Args:
        router_instance: The LiteLLM Router instance (for accessing model_list, etc.)
        strategy_name: Optional ML strategy name (e.g., "llmrouter-knn")
    """

    def __init__(
        self,
        router_instance: Any,
        strategy_name: Optional[str] = None,
    ) -> None:
        self._router = router_instance
        self._strategy_name = strategy_name
        self._strategy_args: Dict[str, Any] = {}

        # Amplification guard: {request_id: attempt_count}
        self._routing_attempts: Dict[str, int] = {}

        # Lazy-initialized components
        self._strategy_instance: Any = None
        self._pipeline: Any = None
        self._pipeline_initialized = False

    # ------------------------------------------------------------------
    # Public API: LiteLLM CustomRoutingStrategyBase
    # ------------------------------------------------------------------

    async def async_get_available_deployment(
        self,
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        specific_deployment: Optional[bool] = False,
        request_kwargs: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """
        Async routing method called by LiteLLM's Router.

        Order of operations:
        1. Check amplification guard
        2. Try pipeline routing (A/B testing)
        3. Fall back to direct LLMRouter ML routing
        4. Fall back to first available deployment
        """
        # 1. Amplification guard
        self._check_amplification_guard(request_kwargs)

        # 2. Try pipeline routing
        if USE_PIPELINE_ROUTING:
            result = self._route_via_pipeline(
                model=model,
                messages=messages,
                input=input,
                request_kwargs=request_kwargs,
            )
            if result is not None:
                return result

        # 3. Try direct LLMRouter routing
        try:
            result = self._route_via_llmrouter(
                model=model,
                messages=messages,
                input=input,
            )
            if result is not None:
                return result
        except Exception as e:
            logger.warning(f"ML routing failed, falling back: {e}")

        # 4. Fallback to first available deployment
        return self._fallback_deployment(model)

    def get_available_deployment(
        self,
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        specific_deployment: Optional[bool] = False,
        request_kwargs: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """
        Sync routing method called by LiteLLM's Router.

        Same logic as the async version but uses synchronous calls.
        """
        # 1. Amplification guard
        self._check_amplification_guard(request_kwargs)

        # 2. Try pipeline routing
        if USE_PIPELINE_ROUTING:
            result = self._route_via_pipeline(
                model=model,
                messages=messages,
                input=input,
                request_kwargs=request_kwargs,
            )
            if result is not None:
                return result

        # 3. Try direct LLMRouter routing
        try:
            result = self._route_via_llmrouter(
                model=model,
                messages=messages,
                input=input,
            )
            if result is not None:
                return result
        except Exception as e:
            logger.warning(f"ML routing failed, falling back: {e}")

        # 4. Fallback to first available deployment
        return self._fallback_deployment(model)

    # ------------------------------------------------------------------
    # Internal: Amplification guard
    # ------------------------------------------------------------------

    def _check_amplification_guard(
        self,
        request_kwargs: Optional[Dict],
    ) -> None:
        """
        Prevent routing amplification (LiteLLM bug #17329).

        If the same ``litellm_call_id`` / ``request_id`` is routed more than
        ``MAX_ROUTING_ATTEMPTS`` times, raise ``RuntimeError`` to short-circuit.
        """
        req_id = None
        if request_kwargs:
            req_id = request_kwargs.get("litellm_call_id") or request_kwargs.get(
                "request_id"
            )
        if not req_id:
            return

        attempts = self._routing_attempts.get(req_id, 0) + 1
        self._routing_attempts[req_id] = attempts

        if attempts > MAX_ROUTING_ATTEMPTS:
            logger.error(
                f"Routing amplification detected for request {req_id}: "
                f"{attempts} attempts (max {MAX_ROUTING_ATTEMPTS}). "
                f"Short-circuiting to prevent 38x amplification bug (#17329)."
            )
            raise RuntimeError(
                f"Routing amplification guard: {attempts} routing attempts "
                f"for request {req_id} exceeds limit of {MAX_ROUTING_ATTEMPTS}"
            )

        # Bounded cleanup to prevent memory leak
        if len(self._routing_attempts) > 10_000:
            self._routing_attempts.clear()

    # ------------------------------------------------------------------
    # Internal: Pipeline routing (A/B testing)
    # ------------------------------------------------------------------

    def _route_via_pipeline(
        self,
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        request_kwargs: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """
        Route through the ``RoutingPipeline`` for A/B testing support.

        Builds a ``RoutingContext`` and delegates to the pipeline, which
        handles strategy selection (single or A/B) and telemetry emission.
        """
        try:
            from litellm_llmrouter.strategy_registry import (
                RoutingContext,
            )

            # Extract identifiers from request_kwargs
            request_id = None
            user_id = None
            if request_kwargs:
                request_id = request_kwargs.get("request_id") or request_kwargs.get(
                    "litellm_call_id"
                )
                user_id = request_kwargs.get("user") or request_kwargs.get("user_id")
                metadata = request_kwargs.get("metadata", {})
                if isinstance(metadata, dict):
                    request_id = request_id or metadata.get("request_id")
                    user_id = user_id or metadata.get("user_id")

            context = RoutingContext(
                router=self._router,
                model=model,
                messages=messages,
                input=input,
                request_kwargs=request_kwargs,
                request_id=request_id,
                user_id=user_id,
            )

            pipeline = self._get_or_create_pipeline()
            result = pipeline.route(context)
            return result.deployment

        except ImportError:
            logger.debug("Pipeline routing not available, using direct LLMRouter")
            return None
        except Exception as e:
            logger.warning(f"Pipeline routing failed, falling back: {e}")
            return None

    # ------------------------------------------------------------------
    # Internal: Direct LLMRouter routing
    # ------------------------------------------------------------------

    def _route_via_llmrouter(
        self,
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
    ) -> Optional[Dict]:
        """
        Route directly using ``LLMRouterStrategyFamily``.

        Steps:
        1. Lazy-load the strategy family instance
        2. Extract query text from messages/input
        3. Get available model list from router
        4. Call ``strategy.route_with_observability()``
        5. Match the selected model name back to a deployment dict
        """
        if not self._strategy_name:
            return None

        strategy = self._get_or_create_strategy()
        if strategy is None:
            return None

        # Extract query
        query = self._extract_query(messages, input)

        # Get model list and deployment map
        model_list, healthy_deployments = self._get_model_list(model)

        if not model_list:
            model_list = [model]

        # Route using ML strategy
        selected_model = strategy.route_with_observability(query, model_list)

        if not selected_model:
            logger.warning(
                f"LLMRouter strategy {self._strategy_name} returned no model, "
                f"falling back to first available deployment"
            )
            return None

        # Match selected model to a deployment dict
        return self._match_deployment(selected_model, model, healthy_deployments)

    # ------------------------------------------------------------------
    # Internal: Helper methods
    # ------------------------------------------------------------------

    def _get_model_list(self, model: str) -> tuple:
        """
        Get available deployments from the Router.

        Prefers ``healthy_deployments`` over ``model_list`` for freshness.

        Returns:
            Tuple of (model_name_list, healthy_deployments_list)
        """
        healthy_deployments = getattr(
            self._router, "healthy_deployments", self._router.model_list
        )

        model_list: List[str] = []
        for deployment in healthy_deployments:
            if deployment.get("model_name") == model:
                litellm_model = deployment.get("litellm_params", {}).get("model", "")
                if litellm_model:
                    model_list.append(litellm_model)

        return model_list, healthy_deployments

    @staticmethod
    def _extract_query(
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
    ) -> str:
        """
        Extract query text from messages or input for ML routing.

        Handles both simple string messages and multi-modal message formats.
        """
        if messages:
            parts: List[str] = []
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    # Handle multi-modal messages
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            parts.append(item.get("text", ""))
            return " ".join(parts).strip()

        if input is not None:
            if isinstance(input, str):
                return input
            return " ".join(str(i) for i in input)

        return ""

    @staticmethod
    def _match_deployment(
        selected_model: str,
        model: str,
        healthy_deployments: List[Dict],
    ) -> Optional[Dict]:
        """
        Map a model name from LLMRouter back to a Router deployment dict.

        First tries exact match on ``litellm_params.model``, then falls
        back to partial/substring matching, and finally to the first
        deployment matching the requested model group.
        """
        # Exact match
        for deployment in healthy_deployments:
            litellm_model = deployment.get("litellm_params", {}).get("model", "")
            if litellm_model == selected_model:
                return deployment

        # Partial match: selected_model is a substring or vice versa
        for deployment in healthy_deployments:
            litellm_model = deployment.get("litellm_params", {}).get("model", "")
            if selected_model in litellm_model or litellm_model in selected_model:
                return deployment

        # Fallback: first deployment for the requested model group
        for deployment in healthy_deployments:
            if deployment.get("model_name") == model:
                return deployment

        return None

    def _get_or_create_strategy(self) -> Any:
        """Lazy-load the ``LLMRouterStrategyFamily`` instance."""
        if self._strategy_instance is None and self._strategy_name:
            try:
                from litellm_llmrouter.strategies import LLMRouterStrategyFamily

                self._strategy_instance = LLMRouterStrategyFamily(
                    strategy_name=self._strategy_name,
                    **self._strategy_args,
                )
            except Exception as e:
                logger.error(f"Failed to create LLMRouterStrategyFamily: {e}")
                return None
        return self._strategy_instance

    def _get_or_create_pipeline(self) -> Any:
        """Lazy-load the ``RoutingPipeline`` singleton."""
        if not self._pipeline_initialized:
            try:
                from litellm_llmrouter.strategy_registry import get_routing_pipeline

                self._pipeline = get_routing_pipeline()
                self._pipeline_initialized = True
            except ImportError:
                logger.debug("RoutingPipeline not available")
                self._pipeline_initialized = True  # Don't retry
        return self._pipeline

    def _fallback_deployment(self, model: str) -> Optional[Dict]:
        """Return first healthy deployment for the given model group."""
        healthy_deployments = getattr(
            self._router, "healthy_deployments", self._router.model_list
        )
        for deployment in healthy_deployments:
            if deployment.get("model_name") == model:
                return deployment
        return None


# ======================================================================
# Factory and installation functions
# ======================================================================


def create_routeiq_strategy(
    router: Any,
    strategy_name: Optional[str] = None,
) -> RouteIQRoutingStrategy:
    """
    Create a configured ``RouteIQRoutingStrategy`` instance.

    Args:
        router: LiteLLM Router instance
        strategy_name: Optional ML strategy name (e.g., "llmrouter-knn")

    Returns:
        Configured strategy instance
    """
    return RouteIQRoutingStrategy(
        router_instance=router,
        strategy_name=strategy_name,
    )


def install_routeiq_strategy(
    router: Any,
    strategy_name: Optional[str] = None,
) -> RouteIQRoutingStrategy:
    """
    Create and install a ``RouteIQRoutingStrategy`` on the given Router.

    Calls ``router.set_custom_routing_strategy()`` to wire the strategy
    into the Router's deployment selection path.

    Args:
        router: LiteLLM Router instance
        strategy_name: Optional ML strategy name (e.g., "llmrouter-knn")

    Returns:
        The installed strategy instance (useful for testing/inspection)
    """
    strategy = create_routeiq_strategy(router, strategy_name)

    if hasattr(router, "set_custom_routing_strategy"):
        router.set_custom_routing_strategy(strategy)
        logger.info(
            f"Installed RouteIQ custom routing strategy "
            f"(strategy={strategy_name or 'default'}, "
            f"pipeline={'enabled' if USE_PIPELINE_ROUTING else 'disabled'})"
        )
    else:
        logger.warning(
            "Router does not support set_custom_routing_strategy(). "
            "Strategy created but NOT installed — is your LiteLLM version compatible?"
        )

    return strategy
