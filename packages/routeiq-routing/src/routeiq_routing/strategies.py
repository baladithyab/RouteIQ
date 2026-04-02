"""Re-export ML routing strategies from the main RouteIQ package.

When installed standalone, this module provides the LLMRouterStrategyFamily
and related strategy classes for ML-based routing with LiteLLM.
"""

try:
    from litellm_llmrouter.strategies import (
        LLMRouterStrategyFamily,
        register_llmrouter_strategies,
        LLMROUTER_STRATEGIES,
    )
except ImportError:
    raise ImportError(
        "routeiq-routing requires either the full routeiq gateway or "
        "a standalone copy of strategies.py. "
        "Install the full gateway: pip install routeiq"
    )
