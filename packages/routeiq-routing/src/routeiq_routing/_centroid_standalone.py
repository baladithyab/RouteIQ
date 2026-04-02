"""Standalone centroid routing for use without the full RouteIQ gateway.

This is a minimal, self-contained implementation of CentroidRoutingStrategy
that works with vanilla LiteLLM. For the full-featured version (vision routing,
context optimization, governance, etc.), install the full routeiq package.

Usage:
    from routeiq_routing import StandaloneCentroidRouter

    router = litellm.Router(model_list=[...])
    strategy = StandaloneCentroidRouter()
    router.set_custom_routing_strategy(strategy)
"""

import logging
import random
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

try:
    from litellm.router_strategy.base_routing_strategy import (
        CustomRoutingStrategyBase,
    )
except ImportError:
    # Provide a stub so the module can be imported without litellm
    # (e.g. for inspecting MODEL_COSTS). The strategy won't be usable
    # at runtime without litellm installed.
    class CustomRoutingStrategyBase:  # type: ignore[no-redef]
        """Stub base class when litellm is not installed."""

        async def async_get_available_deployment(
            self, *args: Any, **kwargs: Any
        ) -> Any:
            raise NotImplementedError("litellm is required for routing")

        def get_available_deployment(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError("litellm is required for routing")


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Cost per million tokens (USD) — top 20 models
MODEL_COSTS: Dict[str, Dict[str, float]] = {
    # GPT-4o family
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    # GPT-4.1 family
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    # o-series
    "o1": {"input": 15.00, "output": 60.00},
    "o1-mini": {"input": 1.10, "output": 4.40},
    "o3-mini": {"input": 1.10, "output": 4.40},
    "o4-mini": {"input": 1.10, "output": 4.40},
    # Claude
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-opus-4-6-20250918": {"input": 5.00, "output": 25.00},
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    # Gemini
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.5-pro-preview-05-06": {"input": 1.25, "output": 10.00},
    # DeepSeek
    "deepseek-chat": {"input": 0.27, "output": 1.10},
    "deepseek-reasoner": {"input": 0.55, "output": 2.19},
}

# Model context windows (tokens)
MODEL_CONTEXT_WINDOWS: Dict[str, int] = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4.1": 1_050_000,
    "gpt-4.1-mini": 1_050_000,
    "gpt-4.1-nano": 1_050_000,
    "o1": 200_000,
    "o1-mini": 128_000,
    "o3-mini": 200_000,
    "o4-mini": 200_000,
    "claude-3-5-sonnet-20241022": 200_000,
    "claude-3-5-haiku-20241022": 200_000,
    "claude-3-opus-20240229": 200_000,
    "claude-sonnet-4-20250514": 200_000,
    "claude-opus-4-6-20250918": 200_000,
    "claude-haiku-4-5-20251001": 200_000,
    "gemini-2.0-flash": 1_048_576,
    "gemini-2.5-pro-preview-05-06": 1_048_576,
    "deepseek-chat": 128_000,
    "deepseek-reasoner": 128_000,
}


# ---------------------------------------------------------------------------
# RoutingProfile enum
# ---------------------------------------------------------------------------


class RoutingProfile(str, Enum):
    """Routing profiles that control tier selection behavior."""

    AUTO = "auto"
    """Use cost-tier heuristic (default)."""

    ECO = "eco"
    """Always route to cheapest available model."""

    PREMIUM = "premium"
    """Always route to most capable (complex tier) model."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def estimate_token_count(messages: List[Dict[str, Any]]) -> int:
    """Estimate token count from messages (~4 chars per token heuristic)."""
    total_chars = sum(len(str(m.get("content", ""))) for m in messages)
    return total_chars // 4


def check_context_window(
    model_name: str, messages: List[Dict[str, Any]], threshold: float = 0.9
) -> bool:
    """Check if messages fit within the model's context window.

    Returns True if the estimated tokens are below threshold * max_tokens.
    """
    estimated = estimate_token_count(messages)
    max_tokens = MODEL_CONTEXT_WINDOWS.get(model_name)
    if max_tokens is None:
        for key, val in MODEL_CONTEXT_WINDOWS.items():
            if key in model_name or model_name in key:
                max_tokens = val
                break
    if max_tokens is None:
        max_tokens = 128_000  # safe default
    return estimated < int(max_tokens * threshold)


def _get_model_cost(model_name: str) -> Dict[str, float]:
    """Look up cost for a model, with fuzzy matching."""
    costs = MODEL_COSTS.get(model_name)
    if costs is not None:
        return costs
    for key, val in MODEL_COSTS.items():
        if key in model_name or model_name in key:
            return val
    return {"input": 1.0, "output": 3.0}  # safe default


def _estimate_request_cost(
    model_name: str, input_tokens: int, estimated_output_tokens: int = 500
) -> float:
    """Estimate cost in USD for a request."""
    costs = _get_model_cost(model_name)
    return (
        input_tokens * costs["input"] + estimated_output_tokens * costs["output"]
    ) / 1_000_000


# ---------------------------------------------------------------------------
# StandaloneCentroidRouter
# ---------------------------------------------------------------------------

# Heuristic model-name indicators for tier classification
_SIMPLE_INDICATORS = ["mini", "nano", "haiku", "flash", "small", "light", "instant"]
_COMPLEX_INDICATORS = [
    "opus",
    "sonnet",
    "pro",
    "large",
    "turbo",
    "4o",
    "gpt-4",
    "claude-3-opus",
]


class StandaloneCentroidRouter(CustomRoutingStrategyBase):
    """Minimal cost-tier routing strategy for standalone use with LiteLLM.

    Routes requests to cheaper or more capable models based on a simple
    cost-tier heuristic. Does NOT require sentence-transformers, centroid
    vectors, Redis, or any other RouteIQ gateway dependencies.

    For the full-featured centroid routing with embedding-based classification,
    vision routing, personalized routing, etc., install the full routeiq package.

    Routing logic:
    1. Classify available deployments into "simple" (cheap) and "complex" (capable)
       tiers based on model name heuristics.
    2. Apply the routing profile:
       - ``eco``: always pick the cheapest model
       - ``premium``: always pick the most capable model
       - ``auto``: use a simple prompt-length heuristic (long prompts → complex)
    3. Within the selected tier, prefer the cheapest model.
    4. Check context window fit; fall back to a larger model if needed.
    """

    def __init__(
        self,
        profile: str = "auto",
        tier_mapping: Optional[Dict[str, List[str]]] = None,
    ):
        """Initialize the standalone router.

        Args:
            profile: Routing profile — ``"auto"``, ``"eco"``, or ``"premium"``.
            tier_mapping: Optional explicit mapping of tier names to model patterns.
                Example: ``{"simple": ["gpt-4o-mini"], "complex": ["gpt-4o"]}``
        """
        try:
            self._profile = RoutingProfile(profile)
        except ValueError:
            self._profile = RoutingProfile.AUTO
        self._tier_mapping = tier_mapping or {}

    # ------------------------------------------------------------------
    # LiteLLM CustomRoutingStrategyBase interface
    # ------------------------------------------------------------------

    async def async_get_available_deployment(
        self,
        model: str,
        messages: Optional[List[Dict[str, Any]]] = None,
        input: Optional[Any] = None,  # noqa: A002
        specific_deployment: Optional[bool] = False,
        request_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Async deployment selection — delegates to sync implementation."""
        return self.get_available_deployment(
            model=model,
            messages=messages,
            input=input,
            specific_deployment=specific_deployment,
            request_kwargs=request_kwargs,
        )

    def get_available_deployment(
        self,
        model: str,
        messages: Optional[List[Dict[str, Any]]] = None,
        input: Optional[Any] = None,  # noqa: A002
        specific_deployment: Optional[bool] = False,
        request_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Select a deployment from the router's model list.

        This is the main LiteLLM custom routing strategy entry point.
        """
        request_kwargs = request_kwargs or {}
        # LiteLLM passes the router instance in request_kwargs
        router = request_kwargs.get("litellm_router_instance")
        if router is None:
            return None

        model_list = getattr(router, "model_list", None) or []
        deployments = [dep for dep in model_list if dep.get("model_name") == model]
        if not deployments:
            return None

        return self.select_deployment(deployments, messages=messages or [])

    # ------------------------------------------------------------------
    # Core routing logic
    # ------------------------------------------------------------------

    def select_deployment(
        self,
        deployments: List[Dict[str, Any]],
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Select the best deployment from a list based on profile and cost tier.

        This is the user-facing method for direct usage (outside the LiteLLM
        custom strategy interface).

        Args:
            deployments: List of LiteLLM deployment dicts.
            messages: Optional list of message dicts for prompt analysis.

        Returns:
            Selected deployment dict, or ``None`` if no deployments available.
        """
        if not deployments:
            return None

        messages = messages or []

        # Determine target tier
        tier = self._resolve_tier(messages)

        # Match tier to deployment
        deployment = self._match_tier(tier, deployments)

        # Context window check
        if deployment is not None and messages:
            dep_model = deployment.get("litellm_params", {}).get("model", "")
            if not check_context_window(dep_model, messages):
                logger.warning(
                    "Context window exceeded for %s (est=%d tokens), trying alt tier",
                    dep_model,
                    estimate_token_count(messages),
                )
                alt_tier = "complex" if tier == "simple" else "simple"
                alt = self._match_tier(alt_tier, deployments)
                if alt is not None:
                    alt_model = alt.get("litellm_params", {}).get("model", "")
                    if check_context_window(alt_model, messages):
                        deployment = alt

        return deployment

    def _resolve_tier(self, messages: List[Dict[str, Any]]) -> str:
        """Determine the target routing tier based on profile and prompt."""
        if self._profile == RoutingProfile.ECO:
            return "simple"
        if self._profile == RoutingProfile.PREMIUM:
            return "complex"

        # AUTO: simple heuristic based on prompt length
        # Short messages → simple tier, longer → complex
        token_estimate = estimate_token_count(messages)
        if token_estimate > 2000:
            return "complex"
        return "simple"

    def _match_tier(
        self, tier: str, deployments: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Match a tier to the best deployment."""
        if not deployments:
            return None

        # Use explicit tier_mapping if configured
        if self._tier_mapping and tier in self._tier_mapping:
            patterns = self._tier_mapping[tier]
            for dep in deployments:
                model = dep.get("litellm_params", {}).get("model", "")
                for pattern in patterns:
                    if pattern.lower() in model.lower():
                        return dep

        # Classify deployments by name heuristics
        simple_deps: List[Tuple[float, Dict[str, Any]]] = []
        complex_deps: List[Tuple[float, Dict[str, Any]]] = []
        other_deps: List[Tuple[float, Dict[str, Any]]] = []

        for dep in deployments:
            model = dep.get("litellm_params", {}).get("model", "").lower()
            cost = _get_model_cost(model).get("input", 1.0)

            is_simple = any(ind in model for ind in _SIMPLE_INDICATORS)
            is_complex = any(ind in model for ind in _COMPLEX_INDICATORS)

            if is_simple:
                simple_deps.append((cost, dep))
            elif is_complex:
                complex_deps.append((cost, dep))
            else:
                other_deps.append((cost, dep))

        # Sort each tier by cost (cheapest first)
        simple_deps.sort(key=lambda x: x[0])
        complex_deps.sort(key=lambda x: x[0])
        other_deps.sort(key=lambda x: x[0])

        if tier == "simple":
            candidates = simple_deps or other_deps or complex_deps
        else:
            candidates = complex_deps or other_deps or simple_deps

        if candidates:
            return candidates[0][1]

        # Final fallback
        return random.choice(deployments)
