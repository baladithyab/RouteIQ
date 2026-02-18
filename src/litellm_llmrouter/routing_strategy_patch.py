"""
LiteLLM Router Strategy Patch
==============================

This module patches LiteLLM's Router class to accept `llmrouter-*` routing strategies.
It must be imported BEFORE any Router initialization occurs.

The patch works by:
1. Monkey-patching the routing_strategy_init() method to accept llmrouter-* prefixed strategies
2. Registering custom routing strategy handlers that delegate to LLMRouterStrategyFamily
3. Integrating with RoutingPipeline for A/B testing support and telemetry

This approach is necessary because LiteLLM validates routing_strategy against a fixed enum
at runtime (see router.py lines 719-736), and we cannot extend Python enums at runtime.

Version Compatibility:
- Tested with litellm >= 1.50.0
- The patch checks for method signature compatibility

Usage:
    # Import this module before creating any Router instances:
    import litellm_llmrouter.routing_strategy_patch

    # Or explicitly call:
    from litellm_llmrouter.routing_strategy_patch import patch_litellm_router
    patch_litellm_router()
"""

import functools
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Track whether patch has been applied
_patch_applied = False

# Store original method for potential restoration
_original_routing_strategy_init = None
_original_get_available_deployment = None
_original_async_get_available_deployment = None

# Feature flag: Use pipeline routing (enables A/B testing)
# Set LLMROUTER_USE_PIPELINE=false to disable
USE_PIPELINE_ROUTING = os.getenv("LLMROUTER_USE_PIPELINE", "true").lower() == "true"


def is_llmrouter_strategy(strategy: Any) -> bool:
    """Check if a routing strategy is an llmrouter-* strategy."""
    if isinstance(strategy, str):
        return strategy.startswith("llmrouter-")
    return False


def get_llmrouter_strategy_name(strategy: str) -> str:
    """Extract the strategy name from llmrouter-* format."""
    if strategy.startswith("llmrouter-"):
        return strategy[len("llmrouter-") :]
    return strategy


def create_patched_routing_strategy_init(original_method: Callable) -> Callable:
    """
    Create a patched version of routing_strategy_init that accepts llmrouter-* strategies.

    The patched method:
    1. Checks if the strategy is an llmrouter-* strategy
    2. If yes, stores the strategy and skips validation (handled by LLMRouterStrategyFamily)
    3. If no, delegates to the original method
    """

    @functools.wraps(original_method)
    def patched_routing_strategy_init(
        self, routing_strategy: Union[Any, str], routing_strategy_args: dict
    ):
        # Check if this is an llmrouter-* strategy
        if is_llmrouter_strategy(routing_strategy):
            logger.info(f"LLMRouter strategy detected: {routing_strategy}")

            # Store the strategy for later use by get_available_deployment
            # We don't initialize any logging handlers here - LLMRouterStrategyFamily
            # handles its own routing logic
            self._llmrouter_strategy = routing_strategy
            self._llmrouter_strategy_args = routing_strategy_args

            # Initialize LLMRouterStrategyFamily instance lazily
            # This will be used by the patched get_available_deployment
            self._llmrouter_strategy_instance = None

            # Initialize pipeline routing if enabled
            if USE_PIPELINE_ROUTING:
                _initialize_pipeline_strategy(
                    self, routing_strategy, routing_strategy_args
                )

            return

        # Delegate to original method for standard strategies
        return original_method(self, routing_strategy, routing_strategy_args)

    return patched_routing_strategy_init


def _initialize_pipeline_strategy(
    router: Any,
    strategy_name: str,
    strategy_args: dict,
) -> None:
    """
    Initialize pipeline routing for a router instance.

    This registers the LLMRouterStrategyFamily as the default strategy
    in the routing registry, enabling A/B testing support.
    """
    try:
        from litellm_llmrouter.strategy_registry import (
            get_routing_registry,
            DefaultStrategy,
        )

        # Register the default strategy if not already registered
        registry = get_routing_registry()

        # Use router-specific strategy name to support multiple routers
        router_strategy_name = f"llmrouter-default-{id(router)}"

        # Check if we need to register
        if router_strategy_name not in registry.list_strategies():
            default_strategy = DefaultStrategy()
            registry.register(router_strategy_name, default_strategy)
            logger.debug(
                f"Registered default pipeline strategy: {router_strategy_name}"
            )

    except ImportError as e:
        logger.debug(f"Pipeline routing not available: {e}")
    except Exception as e:
        logger.warning(f"Failed to initialize pipeline routing: {e}")


def create_patched_get_available_deployment(original_method: Callable) -> Callable:
    """
    Create a patched version of get_available_deployment that uses LLMRouterStrategyFamily
    for llmrouter-* strategies, with pipeline routing support for A/B testing.

    Includes a defensive guard: if the same request_id is routed more than
    MAX_ROUTING_ATTEMPTS times, short-circuit to prevent LiteLLM's 38x request
    amplification bug (#17329).
    """
    # Per-request routing attempt counter {request_id: count}
    _routing_attempts: Dict[str, int] = {}
    MAX_ROUTING_ATTEMPTS = 3

    @functools.wraps(original_method)
    def patched_get_available_deployment(
        self,
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        specific_deployment: Optional[bool] = False,
        request_kwargs: Optional[Dict] = None,
    ):
        # Defensive guard: detect routing amplification
        req_id = None
        if request_kwargs:
            req_id = request_kwargs.get("request_id") or request_kwargs.get(
                "litellm_call_id"
            )
        if req_id:
            attempts = _routing_attempts.get(req_id, 0) + 1
            _routing_attempts[req_id] = attempts
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
            # Cleanup old entries (simple bounded dict)
            if len(_routing_attempts) > 10000:
                _routing_attempts.clear()

        # Check if we're using an llmrouter strategy
        if hasattr(self, "_llmrouter_strategy") and self._llmrouter_strategy:
            # Try pipeline routing first if enabled
            if USE_PIPELINE_ROUTING:
                result = _get_deployment_via_pipeline(
                    router=self,
                    model=model,
                    messages=messages,
                    input=input,
                    specific_deployment=specific_deployment,
                    request_kwargs=request_kwargs,
                )
                if result is not None:
                    return result

            # Fallback to direct LLMRouter call
            return _get_deployment_via_llmrouter(
                router=self,
                model=model,
                messages=messages,
                input=input,
                specific_deployment=specific_deployment,
                request_kwargs=request_kwargs,
            )

        # Delegate to original method
        return original_method(
            self,
            model=model,
            messages=messages,
            input=input,
            specific_deployment=specific_deployment,
            request_kwargs=request_kwargs,
        )

    return patched_get_available_deployment


def _get_deployment_via_pipeline(
    router: Any,
    model: str,
    messages: Optional[List[Dict[str, str]]] = None,
    input: Optional[Union[str, List]] = None,
    specific_deployment: Optional[bool] = False,
    request_kwargs: Optional[Dict] = None,
) -> Optional[Dict]:
    """
    Get deployment using the routing pipeline.

    This enables A/B testing and telemetry via the strategy registry.
    Falls back to None if pipeline is not available.
    """
    try:
        from litellm_llmrouter.strategy_registry import (
            get_routing_pipeline,
            RoutingContext,
        )

        # Extract request_id and user_id from request_kwargs if available
        request_id = None
        user_id = None
        if request_kwargs:
            request_id = request_kwargs.get("request_id") or request_kwargs.get(
                "litellm_call_id"
            )
            user_id = request_kwargs.get("user") or request_kwargs.get("user_id")
            # Also check metadata
            metadata = request_kwargs.get("metadata", {})
            if isinstance(metadata, dict):
                request_id = request_id or metadata.get("request_id")
                user_id = user_id or metadata.get("user_id")

        # Build routing context
        context = RoutingContext(
            router=router,
            model=model,
            messages=messages,
            request_kwargs=request_kwargs,
            request_id=request_id,
            user_id=user_id,
        )

        # Execute pipeline
        pipeline = get_routing_pipeline()
        result = pipeline.route(context)

        # Return deployment or None
        return result.deployment

    except ImportError:
        logger.debug("Pipeline routing not available, using direct LLMRouter")
        return None
    except Exception as e:
        logger.warning(f"Pipeline routing failed, falling back: {e}")
        return None


def _get_deployment_via_llmrouter(
    router: Any,
    model: str,
    messages: Optional[List[Dict[str, str]]] = None,
    input: Optional[Union[str, List]] = None,
    specific_deployment: Optional[bool] = False,
    request_kwargs: Optional[Dict] = None,
) -> Optional[Dict]:
    """
    Get deployment using LLMRouterStrategyFamily.

    This function:
    1. Lazily initializes the LLMRouterStrategyFamily if needed
    2. Uses it to select a model
    3. Maps the selected model back to a LiteLLM deployment
    """
    from litellm_llmrouter.strategies import LLMRouterStrategyFamily

    # Lazily initialize the strategy instance
    if router._llmrouter_strategy_instance is None:
        router._llmrouter_strategy_instance = LLMRouterStrategyFamily(
            strategy_name=router._llmrouter_strategy, **router._llmrouter_strategy_args
        )

    strategy: LLMRouterStrategyFamily = router._llmrouter_strategy_instance

    # Extract query text from messages or input
    query = ""
    if messages:
        # Concatenate message contents for routing decision
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                query += content + " "
            elif isinstance(content, list):
                # Handle multi-modal messages
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        query += item.get("text", "") + " "
    elif input:
        query = input if isinstance(input, str) else " ".join(str(i) for i in input)

    # Get available model names from router's model list
    model_list = []
    healthy_deployments = getattr(router, "healthy_deployments", router.model_list)

    for deployment in healthy_deployments:
        if deployment.get("model_name") == model:
            litellm_model = deployment.get("litellm_params", {}).get("model", "")
            if litellm_model:
                model_list.append(litellm_model)

    if not model_list:
        # Fall back to using model_name directly
        model_list = [model]

    # Route using LLMRouter strategy
    selected_model = strategy.route_with_observability(query.strip(), model_list)

    if not selected_model:
        # If no model selected, use simple-shuffle fallback
        logger.warning(
            f"LLMRouter strategy {router._llmrouter_strategy} returned no model, "
            f"falling back to first available deployment"
        )
        # Return first healthy deployment for the model
        for deployment in healthy_deployments:
            if deployment.get("model_name") == model:
                return deployment
        return None

    # Find the deployment matching the selected model
    for deployment in healthy_deployments:
        litellm_model = deployment.get("litellm_params", {}).get("model", "")
        if litellm_model == selected_model:
            return deployment

    # Fallback: return first deployment if no exact match
    for deployment in healthy_deployments:
        if deployment.get("model_name") == model:
            return deployment

    return None


async def _async_get_deployment_via_llmrouter(
    router: Any,
    model: str,
    messages: Optional[List[Dict[str, str]]] = None,
    input: Optional[Union[str, List]] = None,
    specific_deployment: Optional[bool] = False,
    request_kwargs: Optional[Dict] = None,
) -> Optional[Dict]:
    """Async version of _get_deployment_via_llmrouter."""
    # Try pipeline routing first
    if USE_PIPELINE_ROUTING:
        result = _get_deployment_via_pipeline(
            router=router,
            model=model,
            messages=messages,
            input=input,
            specific_deployment=specific_deployment,
            request_kwargs=request_kwargs,
        )
        if result is not None:
            return result

    # Fallback to sync version as LLMRouter doesn't have async API
    return _get_deployment_via_llmrouter(
        router=router,
        model=model,
        messages=messages,
        input=input,
        specific_deployment=specific_deployment,
        request_kwargs=request_kwargs,
    )


def create_patched_async_get_available_deployment(
    original_method: Callable,
) -> Callable:
    """
    Create a patched version of async_get_available_deployment that uses
    LLMRouterStrategyFamily for llmrouter-* strategies.
    """

    @functools.wraps(original_method)
    async def patched_async_get_available_deployment(
        self,
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        specific_deployment: Optional[bool] = False,
        request_kwargs: Optional[Dict] = None,
    ):
        # Check if we're using an llmrouter strategy
        if hasattr(self, "_llmrouter_strategy") and self._llmrouter_strategy:
            return await _async_get_deployment_via_llmrouter(
                router=self,
                model=model,
                messages=messages,
                input=input,
                specific_deployment=specific_deployment,
                request_kwargs=request_kwargs,
            )

        # Delegate to original method
        return await original_method(
            self,
            model=model,
            messages=messages,
            input=input,
            specific_deployment=specific_deployment,
            request_kwargs=request_kwargs,
        )

    return patched_async_get_available_deployment


def patch_litellm_router() -> bool:
    """
    Apply the patch to LiteLLM's Router class.

    Returns:
        True if patch was applied successfully, False otherwise.
    """
    global _patch_applied, _original_routing_strategy_init
    global _original_get_available_deployment, _original_async_get_available_deployment

    if _patch_applied:
        logger.debug("LiteLLM Router patch already applied")
        return True

    try:
        from litellm.router import Router

        # Verify the method exists and has expected signature
        if not hasattr(Router, "routing_strategy_init"):
            logger.error(
                "Router.routing_strategy_init not found - LiteLLM version incompatible"
            )
            return False

        # Store original methods
        _original_routing_strategy_init = Router.routing_strategy_init
        _original_get_available_deployment = Router.get_available_deployment

        # Check if async method exists
        has_async_method = hasattr(Router, "async_get_available_deployment")
        if has_async_method:
            _original_async_get_available_deployment = (
                Router.async_get_available_deployment
            )

        # Apply patches
        Router.routing_strategy_init = create_patched_routing_strategy_init(
            _original_routing_strategy_init
        )
        Router.get_available_deployment = create_patched_get_available_deployment(
            _original_get_available_deployment
        )

        if has_async_method:
            Router.async_get_available_deployment = (
                create_patched_async_get_available_deployment(
                    _original_async_get_available_deployment
                )
            )

        _patch_applied = True
        logger.info(
            f"LiteLLM Router patched to accept llmrouter-* strategies "
            f"(pipeline_routing={'enabled' if USE_PIPELINE_ROUTING else 'disabled'})"
        )
        return True

    except ImportError as e:
        logger.error(f"Failed to import litellm.router: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to patch LiteLLM Router: {e}")
        return False


def unpatch_litellm_router() -> bool:
    """
    Remove the patch from LiteLLM's Router class.

    Returns:
        True if unpatch was successful, False otherwise.
    """
    global _patch_applied, _original_routing_strategy_init
    global _original_get_available_deployment, _original_async_get_available_deployment

    if not _patch_applied:
        logger.debug("LiteLLM Router patch not applied, nothing to unpatch")
        return True

    try:
        from litellm.router import Router

        if _original_routing_strategy_init is not None:
            Router.routing_strategy_init = _original_routing_strategy_init
        if _original_get_available_deployment is not None:
            Router.get_available_deployment = _original_get_available_deployment
        if _original_async_get_available_deployment is not None:
            Router.async_get_available_deployment = (
                _original_async_get_available_deployment
            )

        _patch_applied = False
        _original_routing_strategy_init = None
        _original_get_available_deployment = None
        _original_async_get_available_deployment = None
        logger.info("LiteLLM Router patch removed")
        return True

    except Exception as e:
        logger.error(f"Failed to unpatch LiteLLM Router: {e}")
        return False


def is_patch_applied() -> bool:
    """Check if the patch has been applied."""
    return _patch_applied


def is_pipeline_routing_enabled() -> bool:
    """Check if pipeline routing is enabled."""
    return USE_PIPELINE_ROUTING


# NOTE: Patch is NOT applied automatically on import.
# Call patch_litellm_router() explicitly from your application startup.
# This ensures importing the module has no side effects.
