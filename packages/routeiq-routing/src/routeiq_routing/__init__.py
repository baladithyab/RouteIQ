"""RouteIQ Routing — Intelligent ML-based routing for LLM APIs.

Standalone usage with LiteLLM:
    from routeiq_routing import CentroidRoutingStrategy

    router = litellm.Router(model_list=[...])
    strategy = CentroidRoutingStrategy()
    router.set_custom_routing_strategy(strategy)

Or configure via litellm config.yaml:
    router_settings:
      routing_strategy: "routeiq-centroid"
"""

from routeiq_routing.centroid import CentroidRoutingStrategy
from routeiq_routing.personalized import PersonalizedRouter

__version__ = "0.1.0"
__all__ = ["CentroidRoutingStrategy", "PersonalizedRouter"]
