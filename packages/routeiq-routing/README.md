# routeiq-routing

Intelligent ML-based routing for LLM APIs — works standalone with LiteLLM or as part of RouteIQ Gateway.

## Installation

```bash
# Core (centroid routing, routing profiles)
pip install routeiq-routing

# With KNN strategy support (requires sentence-transformers)
pip install routeiq-routing[knn]

# With all ML strategies
pip install routeiq-routing[all]
```

## Quick Start

### Centroid Routing (Zero-Config)

The centroid routing strategy provides intelligent model selection with ~2ms
latency and zero configuration. It classifies prompts by complexity and routes
to the most appropriate model.

```python
import litellm
from routeiq_routing import CentroidRoutingStrategy

router = litellm.Router(
    model_list=[
        {
            "model_name": "gpt",
            "litellm_params": {"model": "gpt-4o-mini", "api_key": "sk-..."},
        },
        {
            "model_name": "gpt",
            "litellm_params": {"model": "gpt-4o", "api_key": "sk-..."},
        },
    ]
)

# Install the routing strategy
strategy = CentroidRoutingStrategy()
router.set_custom_routing_strategy(strategy)

# Requests are now routed intelligently
response = router.completion(
    model="gpt",
    messages=[{"role": "user", "content": "What is 2+2?"}],
)
# -> Routes to gpt-4o-mini (simple query)

response = router.completion(
    model="gpt",
    messages=[{"role": "user", "content": "Analyze the economic implications of..."}],
)
# -> Routes to gpt-4o (complex query)
```

### Routing Profiles

Control the cost/quality trade-off:

```python
from routeiq_routing.centroid import RoutingProfile

# Prefer cheaper models when quality is sufficient
strategy = CentroidRoutingStrategy(default_profile=RoutingProfile.ECO)

# Always use the best model
strategy = CentroidRoutingStrategy(default_profile=RoutingProfile.PREMIUM)

# Only use free-tier models
strategy = CentroidRoutingStrategy(default_profile=RoutingProfile.FREE)
```

### LiteLLM Config (Entry Point)

You can also configure via LiteLLM's config YAML using the entry point:

```yaml
router_settings:
  routing_strategy: "routeiq-centroid"
```

This works because `routeiq-routing` registers a `litellm.custom_routing_strategy`
entry point that LiteLLM discovers automatically.

### Personalized Routing

Learn per-user model preferences from feedback:

```python
from routeiq_routing import PersonalizedRouter

router = PersonalizedRouter()
# Routes adapt to user preferences over time based on feedback signals
```

## Relationship to RouteIQ Gateway

This package extracts the routing engine from [RouteIQ Gateway](https://github.com/your-org/RouteIQ)
for standalone use. If you need the full gateway with security, observability,
plugins, and the composition root, install the full package:

```bash
pip install routeiq[prod]
```

## License

Apache-2.0
