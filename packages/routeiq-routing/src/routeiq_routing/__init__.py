"""RouteIQ Routing — Intelligent ML-based routing for LLM APIs.

Two modes:
1. Standalone: uv add routeiq-routing (or pip install routeiq-routing)
   Uses StandaloneCentroidRouter with cost-tier routing

2. Full gateway: uv add routeiq (or pip install routeiq)
   Uses CentroidRoutingStrategy with ML routing, vision, personalization

Standalone usage with LiteLLM:
    from routeiq_routing import CentroidRoutingStrategy

    router = litellm.Router(model_list=[...])
    strategy = CentroidRoutingStrategy()
    router.set_custom_routing_strategy(strategy)

Or configure via litellm config.yaml:
    router_settings:
      routing_strategy: "routeiq-centroid"
"""

# Try full RouteIQ first, fall back to standalone
try:
    from litellm_llmrouter.centroid_routing import (
        CentroidRoutingStrategy,
    )
    from litellm_llmrouter.personalized_routing import PersonalizedRouter

    _FULL_GATEWAY = True
except ImportError:
    from routeiq_routing._centroid_standalone import (
        StandaloneCentroidRouter as CentroidRoutingStrategy,  # type: ignore[assignment]
    )

    PersonalizedRouter = None  # type: ignore[assignment,misc]
    _FULL_GATEWAY = False

# Always expose the standalone router for explicit use
from routeiq_routing._centroid_standalone import StandaloneCentroidRouter

__version__ = "1.0.0rc1"
__all__ = ["CentroidRoutingStrategy", "PersonalizedRouter", "StandaloneCentroidRouter"]
