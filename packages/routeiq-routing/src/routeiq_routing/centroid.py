"""Re-export centroid routing from the main RouteIQ package.

When installed standalone, this module provides the CentroidRoutingStrategy
that works with any LiteLLM deployment.
"""

try:
    from litellm_llmrouter.centroid_routing import (
        CentroidRoutingStrategy,
        CentroidClassifier,
        MODEL_COSTS,
        MODEL_CONTEXT_WINDOWS,
        MODEL_CAPABILITIES,
        VISION_CAPABLE_MODELS,
        RoutingProfile,
    )
except ImportError:
    # Standalone mode — routing code is bundled in this package
    raise ImportError(
        "routeiq-routing requires either the full routeiq gateway or "
        "a standalone copy of centroid_routing.py. "
        "Install the full gateway: pip install routeiq"
    )
