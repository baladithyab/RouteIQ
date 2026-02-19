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


# Import centroid routing with graceful fallback
try:
    from litellm_llmrouter.centroid_routing import (
        CentroidRoutingStrategy,
        get_centroid_strategy,
        warmup_centroid_classifier,
        RoutingProfile,
    )

    CENTROID_ROUTING_AVAILABLE = True
except ImportError:
    CENTROID_ROUTING_AVAILABLE = False


logger = logging.getLogger(__name__)

# Feature flag: Use pipeline routing (enables A/B testing)
# Set LLMROUTER_USE_PIPELINE=false to disable
USE_PIPELINE_ROUTING = os.getenv("LLMROUTER_USE_PIPELINE", "true").lower() == "true"

# Feature flag: Enable centroid routing as fallback (zero-config intelligent routing)
# Set ROUTEIQ_CENTROID_ROUTING=false to disable
CENTROID_ROUTING_ENABLED = (
    os.getenv("ROUTEIQ_CENTROID_ROUTING", "true").lower() == "true"
)

# Default routing profile (auto, eco, premium, free, reasoning)
DEFAULT_ROUTING_PROFILE = os.getenv("ROUTEIQ_ROUTING_PROFILE", "auto")

# Maximum routing attempts per request to prevent amplification loops
MAX_ROUTING_ATTEMPTS = 3


def _resolve_routing_profile(
    request_kwargs: Optional[Dict] = None,
) -> Optional[str]:
    """Resolve the routing profile from request metadata or env var.

    Args:
        request_kwargs: Request keyword arguments (may contain metadata).

    Returns:
        Routing profile string, or None if not configured.
    """
    # 1. Check request metadata
    if request_kwargs:
        metadata = request_kwargs.get("metadata", {})
        if isinstance(metadata, dict):
            profile = metadata.get("routing_profile")
            if profile:
                return str(profile).lower()

    # 2. Fall back to env var default
    return DEFAULT_ROUTING_PROFILE


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
    - **Centroid routing fallback** for zero-config intelligent routing (~2ms)
    - **Routing profiles** (auto, eco, premium, free, reasoning)
    - **Amplification guard** prevents infinite routing loops (max 3 per request)
    - **Graceful fallback** to first available deployment if all routing fails

    Progressive enhancement chain:
    Pipeline strategies (KNN, SVM, etc.) → Centroid routing (~2ms) → First healthy deployment

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

        # Centroid routing state (lazy-initialized)
        self._centroid_strategy: Any = None
        self._centroid_initialized = False

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
        4. Fall back to centroid routing (zero-config intelligent routing)
        5. Fall back to first available deployment
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

        # 4. Try centroid routing (zero-config intelligent routing)
        result = self._route_via_centroid(
            model=model,
            messages=messages,
            request_kwargs=request_kwargs,
        )
        if result is not None:
            return result

        # 5. Fallback to first available deployment
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

        # 4. Try centroid routing (zero-config intelligent routing)
        result = self._route_via_centroid(
            model=model,
            messages=messages,
            request_kwargs=request_kwargs,
        )
        if result is not None:
            return result

        # 5. Fallback to first available deployment
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
    # Internal: Centroid routing (zero-config fallback)
    # ------------------------------------------------------------------

    def _route_via_centroid(
        self,
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
        request_kwargs: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """
        Route using centroid-based classification (zero-config intelligent routing).

        This is the fallback when no ML model is configured. It provides
        intelligent routing without requiring ML model training, using
        ~2ms centroid-based prompt classification.

        Args:
            model: The requested model group name.
            messages: Chat messages for classification.
            request_kwargs: Request keyword arguments (may contain routing_profile).

        Returns:
            Selected deployment dict, or None if centroid routing is unavailable.
        """
        if not CENTROID_ROUTING_ENABLED or not CENTROID_ROUTING_AVAILABLE:
            return None

        try:
            centroid = self._get_or_create_centroid_strategy(request_kwargs)
            if centroid is None:
                return None

            # Build a lightweight routing context for centroid strategy
            from litellm_llmrouter.strategy_registry import RoutingContext

            context = RoutingContext(
                router=self._router,
                model=model,
                messages=messages,
                input=None,
                request_kwargs=request_kwargs,
            )

            result = centroid.select_deployment(context)
            if result is not None:
                logger.debug(
                    "Centroid routing selected: %s (model=%s)",
                    result.get("litellm_params", {}).get("model", "unknown"),
                    model,
                )
            return result

        except ImportError:
            logger.debug("Centroid routing not available (missing dependencies)")
            return None
        except Exception as e:
            logger.warning(f"Centroid routing failed, falling back: {e}")
            return None

    def _get_or_create_centroid_strategy(
        self,
        request_kwargs: Optional[Dict] = None,
    ) -> Any:
        """Lazy-load or return the centroid routing strategy.

        Resolves the routing profile from request metadata or env var
        and creates the centroid strategy with the appropriate profile.

        Args:
            request_kwargs: Request keyword arguments for profile resolution.

        Returns:
            CentroidRoutingStrategy instance, or None if unavailable.
        """
        if not CENTROID_ROUTING_AVAILABLE:
            return None

        # Resolve the routing profile for this request
        profile_str = _resolve_routing_profile(request_kwargs)

        # Map profile string to RoutingProfile enum
        try:
            profile = (
                RoutingProfile(profile_str) if profile_str else RoutingProfile.AUTO
            )
        except ValueError:
            logger.warning(
                "Unknown routing profile '%s', defaulting to 'auto'",
                profile_str,
            )
            profile = RoutingProfile.AUTO

        if not self._centroid_initialized:
            try:
                self._centroid_strategy = CentroidRoutingStrategy(profile=profile)
                self._centroid_initialized = True
            except Exception as e:
                logger.warning(f"Failed to create centroid strategy: {e}")
                self._centroid_initialized = True  # Don't retry
                return None
        elif self._centroid_strategy is not None:
            # Update profile per-request (it may change between requests)
            self._centroid_strategy._profile = profile

        return self._centroid_strategy

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

    Optionally warms up the centroid classifier for fast first-request latency.
    Controlled by ``ROUTEIQ_CENTROID_WARMUP=true`` env var.

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
            f"pipeline={'enabled' if USE_PIPELINE_ROUTING else 'disabled'}, "
            f"centroid={'enabled' if CENTROID_ROUTING_ENABLED and CENTROID_ROUTING_AVAILABLE else 'disabled'})"
        )
    else:
        logger.warning(
            "Router does not support set_custom_routing_strategy(). "
            "Strategy created but NOT installed — is your LiteLLM version compatible?"
        )

    # Optionally warmup centroid classifier
    if (
        os.getenv("ROUTEIQ_CENTROID_WARMUP", "false").lower() == "true"
        and CENTROID_ROUTING_AVAILABLE
        and CENTROID_ROUTING_ENABLED
    ):
        try:
            warmup_centroid_classifier()
            logger.info("Centroid classifier warmed up during strategy installation")
        except Exception as e:
            logger.warning(f"Centroid classifier warmup failed: {e}")

    return strategy


def register_centroid_strategy() -> bool:
    """Register the centroid strategy in the routing registry.

    Makes the centroid strategy available as ``"llmrouter-nadirclaw-centroid"`` in the
    routing registry for direct use or A/B testing.

    Auto-registers if ``ROUTEIQ_CENTROID_ROUTING=true`` (default).

    Returns:
        True if registration succeeded, False otherwise.
    """
    if not CENTROID_ROUTING_AVAILABLE:
        logger.debug("Centroid routing not available (missing imports)")
        return False

    if not CENTROID_ROUTING_ENABLED:
        logger.debug("Centroid routing disabled via ROUTEIQ_CENTROID_ROUTING=false")
        return False

    try:
        from litellm_llmrouter.strategy_registry import get_routing_registry

        registry = get_routing_registry()
        strategy = get_centroid_strategy()
        registry.register("llmrouter-nadirclaw-centroid", strategy)
        logger.info(
            "Registered centroid strategy as 'llmrouter-nadirclaw-centroid' in routing registry"
        )
        return True
    except Exception as e:
        logger.warning(f"Failed to register centroid strategy: {e}")
        return False
