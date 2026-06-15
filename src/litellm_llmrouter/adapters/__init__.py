"""
RouteIQ Routing-Strategy Adapter Framework
==========================================

Strategy-agnostic machinery for plugging routing strategies/algorithms into the
gateway without forking: a versioned adapter contract, an entry-point loader, and
the MLOps feedback/artifact loop. Attaches to ANY registered strategy — the
Kumaraswamy-Thompson bandit is one consumer, not the only target.
"""

from litellm_llmrouter.adapters.contract import (
    ADAPTER_API_VERSION,
    TRAIN_MODE_CONTINUOUS,
    TRAIN_MODE_ONE_TIME,
    AdapterManifest,
    ArtifactRef,
    RoutingAdapter,
    RoutingFeedback,
    _abi_compatible,
    attach_route_alias,
)
from litellm_llmrouter.adapters.loader import AdapterLoaderPlugin
from litellm_llmrouter.adapters.mlops import (
    MLOpsCoordinator,
    get_mlops_coordinator,
    reset_mlops_coordinator,
    wire_mlops_feedback_loop,
)

__all__ = [
    "ADAPTER_API_VERSION",
    "TRAIN_MODE_CONTINUOUS",
    "TRAIN_MODE_ONE_TIME",
    "AdapterManifest",
    "ArtifactRef",
    "RoutingAdapter",
    "RoutingFeedback",
    "_abi_compatible",
    "attach_route_alias",
    "AdapterLoaderPlugin",
    "MLOpsCoordinator",
    "get_mlops_coordinator",
    "reset_mlops_coordinator",
    "wire_mlops_feedback_loop",
]
