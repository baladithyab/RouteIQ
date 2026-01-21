"""
LiteLLM Router Strategy Patch
==============================

This module patches LiteLLM's Router class to accept `llmrouter-*` routing strategies.
It must be imported BEFORE any Router initialization occurs.

The patch works by:
1. Monkey-patching the routing_strategy_init() method to accept llmrouter-* prefixed strategies
2. Registering custom routing strategy handlers that delegate to LLMRouterStrategyFamily

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
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Track whether patch has been applied
_patch_applied = False

# Store original method for potential restoration
_original_routing_strategy_init = None


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

            return

        # Delegate to original method for standard strategies
        return original_method(self, routing_strategy, routing_strategy_args)

    return patched_routing_strategy_init


def create_patched_get_available_deployment(original_method: Callable) -> Callable:
    """
    Create a patched version of get_available_deployment that uses LLMRouterStrategyFamily
    for llmrouter-* strategies.
    """

    @functools.wraps(original_method)
    def patched_get_available_deployment(
        self,
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        specific_deployment: Optional[bool] = False,
        request_kwargs: Optional[Dict] = None,
    ):
        # Check if we're using an llmrouter strategy
        if hasattr(self, "_llmrouter_strategy") and self._llmrouter_strategy:
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
    # For now, delegate to sync version as LLMRouter doesn't have async API
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
        original_get_available_deployment = Router.get_available_deployment

        # Check if async method exists
        has_async_method = hasattr(Router, "async_get_available_deployment")
        if has_async_method:
            original_async_get_available_deployment = (
                Router.async_get_available_deployment
            )

        # Apply patches
        Router.routing_strategy_init = create_patched_routing_strategy_init(
            _original_routing_strategy_init
        )
        Router.get_available_deployment = create_patched_get_available_deployment(
            original_get_available_deployment
        )

        if has_async_method:
            Router.async_get_available_deployment = (
                create_patched_async_get_available_deployment(
                    original_async_get_available_deployment
                )
            )

        _patch_applied = True
        logger.info("LiteLLM Router patched to accept llmrouter-* strategies")
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

    if not _patch_applied:
        logger.debug("LiteLLM Router patch not applied, nothing to unpatch")
        return True

    try:
        from litellm.router import Router

        if _original_routing_strategy_init is not None:
            Router.routing_strategy_init = _original_routing_strategy_init

        _patch_applied = False
        _original_routing_strategy_init = None
        logger.info("LiteLLM Router patch removed")
        return True

    except Exception as e:
        logger.error(f"Failed to unpatch LiteLLM Router: {e}")
        return False


def is_patch_applied() -> bool:
    """Check if the patch has been applied."""
    return _patch_applied


# Auto-apply patch on module import
# This ensures the patch is in place before any Router is created
patch_litellm_router()
