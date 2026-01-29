"""
Routing Strategy Registry and Pipeline
========================================

This module provides runtime strategy hot-swapping and A/B testing support
for LLMRouter routing strategies.

Features:
- Thread-safe strategy registry for registering multiple implementations
- Weighted A/B strategy selection with deterministic hashing
- Routing pipeline with fallback support and telemetry emission
- Admin-safe update methods for runtime configuration changes

Configuration:
- LLMROUTER_ACTIVE_ROUTING_STRATEGY: Default active strategy (default: existing)
- LLMROUTER_STRATEGY_WEIGHTS: JSON dict of strategy weights for A/B testing
  Example: '{"baseline": 90, "candidate": 10}'

Usage:
    from litellm_llmrouter.strategy_registry import (
        get_routing_registry,
        get_routing_pipeline,
    )
    
    # Register strategies
    registry = get_routing_registry()
    registry.register("baseline", BaselineStrategy())
    registry.register("candidate", CandidateStrategy())
    
    # Set active strategy or A/B weights
    registry.set_active("baseline")
    registry.set_weights({"baseline": 90, "candidate": 10})
    
    # Use pipeline for routing decisions
    pipeline = get_routing_pipeline()
    deployment = pipeline.route(router, model, messages, request_kwargs)
"""

import hashlib
import json
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from litellm_llmrouter.telemetry_contracts import (
    RouterDecisionEventBuilder,
    RoutingOutcome,
    ROUTER_DECISION_EVENT_NAME,
    ROUTER_DECISION_PAYLOAD_KEY,
)

logger = logging.getLogger(__name__)

# Configuration environment variables
ENV_ACTIVE_STRATEGY = "LLMROUTER_ACTIVE_ROUTING_STRATEGY"
ENV_STRATEGY_WEIGHTS = "LLMROUTER_STRATEGY_WEIGHTS"

# Default strategy name - matches existing LLMRouterStrategyFamily behavior
DEFAULT_STRATEGY_NAME = "llmrouter-default"


@dataclass
class RoutingContext:
    """
    Context passed through the routing pipeline.
    
    Contains all information needed to make a routing decision,
    including request identifiers for deterministic A/B assignment.
    """
    
    router: Any
    """The LiteLLM Router instance."""
    
    model: str
    """Requested model name."""
    
    messages: Optional[List[Dict[str, str]]] = None
    """Chat messages (if applicable)."""
    
    input: Optional[Union[str, List]] = None
    """Input text/embeddings (if applicable)."""
    
    specific_deployment: bool = False
    """Whether a specific deployment was requested."""
    
    request_kwargs: Optional[Dict] = None
    """Additional request parameters."""
    
    # Identifiers for deterministic A/B hashing
    request_id: Optional[str] = None
    """Unique request identifier for A/B assignment."""
    
    user_id: Optional[str] = None
    """User identifier for sticky A/B assignment."""
    
    # Metadata for telemetry
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata for telemetry."""
    
    def get_ab_hash_key(self) -> str:
        """
        Get the key used for deterministic A/B hash assignment.
        
        Priority: user_id > request_id > random UUID
        Using user_id provides sticky assignment (same user always gets same variant).
        """
        if self.user_id:
            return f"user:{self.user_id}"
        if self.request_id:
            return f"request:{self.request_id}"
        # Fallback: generate random key (no stickiness)
        import uuid
        return f"random:{uuid.uuid4()}"


@dataclass
class RoutingResult:
    """Result of a routing decision."""
    
    deployment: Optional[Dict] = None
    """Selected deployment configuration."""
    
    strategy_name: str = ""
    """Name of the strategy that made the decision."""
    
    is_fallback: bool = False
    """Whether this result came from fallback."""
    
    fallback_reason: Optional[str] = None
    """Reason for fallback (if applicable)."""
    
    latency_ms: float = 0.0
    """Time taken for routing decision in milliseconds."""
    
    error: Optional[str] = None
    """Error message if routing failed."""


class RoutingStrategy(ABC):
    """
    Abstract base class for routing strategies.
    
    Implement this interface to create custom routing strategies
    that can be registered and hot-swapped at runtime.
    """
    
    @abstractmethod
    def select_deployment(
        self,
        context: RoutingContext,
    ) -> Optional[Dict]:
        """
        Select a deployment for the given routing context.
        
        Args:
            context: Routing context with request details
            
        Returns:
            Selected deployment dict, or None if no selection
        """
        pass
    
    @property
    def name(self) -> str:
        """Strategy name for telemetry and logging."""
        return self.__class__.__name__
    
    @property
    def version(self) -> Optional[str]:
        """Strategy version for telemetry."""
        return None


class DefaultStrategy(RoutingStrategy):
    """
    Default routing strategy that delegates to LLMRouterStrategyFamily.
    
    This wraps the existing UIUC LLMRouter integration, providing
    backwards compatibility while enabling the new pipeline architecture.
    """
    
    def __init__(
        self,
        strategy_factory: Optional[Callable[..., Any]] = None,
    ):
        """
        Initialize default strategy.
        
        Args:
            strategy_factory: Optional factory to create LLMRouterStrategyFamily.
                              If None, uses lazy import from strategies module.
        """
        self._strategy_factory = strategy_factory
        self._strategies: Dict[str, Any] = {}
        self._lock = threading.RLock()
    
    def _get_strategy_instance(self, router: Any) -> Optional[Any]:
        """Get or create LLMRouterStrategyFamily instance for router."""
        router_id = id(router)
        
        with self._lock:
            if router_id not in self._strategies:
                # Lazily create strategy instance
                if not hasattr(router, "_llmrouter_strategy"):
                    return None
                
                strategy_name = router._llmrouter_strategy
                strategy_args = getattr(router, "_llmrouter_strategy_args", {})
                
                if self._strategy_factory:
                    instance = self._strategy_factory(
                        strategy_name=strategy_name,
                        **strategy_args,
                    )
                else:
                    # Lazy import to avoid circular dependencies
                    from litellm_llmrouter.strategies import LLMRouterStrategyFamily
                    instance = LLMRouterStrategyFamily(
                        strategy_name=strategy_name,
                        **strategy_args,
                    )
                
                self._strategies[router_id] = instance
            
            return self._strategies.get(router_id)
    
    def select_deployment(
        self,
        context: RoutingContext,
    ) -> Optional[Dict]:
        """Select deployment using LLMRouterStrategyFamily."""
        strategy = self._get_strategy_instance(context.router)
        if not strategy:
            logger.warning("No LLMRouter strategy instance available")
            return None
        
        # Extract query from messages/input
        query = self._extract_query(context)
        
        # Get available deployments
        model_list, deployment_map = self._get_deployments(context)
        
        if not model_list:
            logger.warning("No models available for routing")
            return None
        
        # Route using the strategy
        selected_model = strategy.route_with_observability(query, model_list)
        
        if selected_model and selected_model in deployment_map:
            return deployment_map[selected_model]
        
        # Fallback: return first deployment
        if model_list:
            first_model = model_list[0]
            if first_model in deployment_map:
                return deployment_map[first_model]
        
        return None
    
    def _extract_query(self, context: RoutingContext) -> str:
        """Extract query text from context."""
        if context.messages:
            parts = []
            for msg in context.messages:
                content = msg.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            parts.append(item.get("text", ""))
            return " ".join(parts).strip()
        
        if context.input:
            if isinstance(context.input, str):
                return context.input
            return " ".join(str(i) for i in context.input)
        
        return ""
    
    def _get_deployments(
        self, context: RoutingContext
    ) -> Tuple[List[str], Dict[str, Dict]]:
        """Get available deployments for routing."""
        model_list = []
        deployment_map = {}
        
        router = context.router
        healthy_deployments = getattr(router, "healthy_deployments", router.model_list)
        
        for deployment in healthy_deployments:
            if deployment.get("model_name") == context.model:
                litellm_model = deployment.get("litellm_params", {}).get("model", "")
                if litellm_model:
                    model_list.append(litellm_model)
                    deployment_map[litellm_model] = deployment
        
        return model_list, deployment_map
    
    @property
    def name(self) -> str:
        return DEFAULT_STRATEGY_NAME


class RoutingStrategyRegistry:
    """
    Thread-safe registry for routing strategies.
    
    Supports:
    - Registering multiple strategy implementations by name
    - Setting an active strategy
    - Weighted A/B strategy selection with deterministic hashing
    - Thread-safe updates for runtime configuration
    """
    
    def __init__(self):
        self._strategies: Dict[str, RoutingStrategy] = {}
        self._active_strategy: Optional[str] = None
        self._weights: Dict[str, int] = {}
        self._lock = threading.RLock()
        self._update_callbacks: List[Callable[[], None]] = []
        
        # Load configuration from environment
        self._load_config()
    
    def _load_config(self):
        """Load configuration from environment variables."""
        # Active strategy
        active = os.getenv(ENV_ACTIVE_STRATEGY)
        if active:
            self._active_strategy = active
            logger.info(f"Loaded active strategy from env: {active}")
        
        # Strategy weights for A/B testing
        weights_json = os.getenv(ENV_STRATEGY_WEIGHTS)
        if weights_json:
            try:
                self._weights = json.loads(weights_json)
                logger.info(f"Loaded strategy weights from env: {self._weights}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse {ENV_STRATEGY_WEIGHTS}: {e}")
    
    def register(
        self,
        name: str,
        strategy: RoutingStrategy,
    ) -> None:
        """
        Register a routing strategy by name.
        
        Args:
            name: Unique name for the strategy
            strategy: Strategy implementation
        """
        with self._lock:
            self._strategies[name] = strategy
            logger.info(f"Registered routing strategy: {name}")
            
            # If no active strategy, set this as default
            if not self._active_strategy and not self._weights:
                self._active_strategy = name
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a routing strategy.
        
        Args:
            name: Name of strategy to remove
            
        Returns:
            True if strategy was removed, False if not found
        """
        with self._lock:
            if name in self._strategies:
                del self._strategies[name]
                logger.info(f"Unregistered routing strategy: {name}")
                
                # Clear active if this was it
                if self._active_strategy == name:
                    self._active_strategy = None
                
                return True
            return False
    
    def get(self, name: str) -> Optional[RoutingStrategy]:
        """Get a registered strategy by name."""
        with self._lock:
            return self._strategies.get(name)
    
    def list_strategies(self) -> List[str]:
        """List all registered strategy names."""
        with self._lock:
            return list(self._strategies.keys())
    
    def set_active(self, name: str) -> bool:
        """
        Set the active routing strategy.
        
        Clears any A/B weights - use set_weights() for A/B testing.
        
        Args:
            name: Name of strategy to activate
            
        Returns:
            True if successful, False if strategy not found
        """
        with self._lock:
            if name not in self._strategies:
                logger.error(f"Cannot set active: strategy '{name}' not registered")
                return False
            
            self._active_strategy = name
            self._weights = {}  # Clear A/B weights
            logger.info(f"Set active routing strategy: {name}")
            self._notify_update()
            return True
    
    def get_active(self) -> Optional[str]:
        """Get the name of the active strategy (if not using A/B)."""
        with self._lock:
            return self._active_strategy if not self._weights else None
    
    def set_weights(self, weights: Dict[str, int]) -> bool:
        """
        Set strategy weights for A/B testing.
        
        Weights are relative (not percentages). Example:
        - {"baseline": 90, "candidate": 10} gives 90% baseline, 10% candidate
        - {"a": 1, "b": 1, "c": 1} gives 33% each
        
        Args:
            weights: Dict mapping strategy names to relative weights
            
        Returns:
            True if all strategies exist, False otherwise
        """
        with self._lock:
            # Validate all strategies exist
            for name in weights:
                if name not in self._strategies:
                    logger.error(f"Cannot set weights: strategy '{name}' not registered")
                    return False
            
            self._weights = weights.copy()
            self._active_strategy = None  # Clear single active strategy
            logger.info(f"Set A/B strategy weights: {weights}")
            self._notify_update()
            return True
    
    def get_weights(self) -> Dict[str, int]:
        """Get current A/B weights."""
        with self._lock:
            return self._weights.copy()
    
    def clear_weights(self) -> None:
        """Clear A/B weights and revert to single active strategy."""
        with self._lock:
            if self._weights:
                # Set first weighted strategy as active
                if self._weights and not self._active_strategy:
                    self._active_strategy = next(iter(self._weights.keys()))
                self._weights = {}
                logger.info("Cleared A/B weights")
                self._notify_update()
    
    def select_strategy(
        self,
        hash_key: str,
    ) -> Optional[RoutingStrategy]:
        """
        Select a strategy for routing.
        
        If A/B weights are set, uses deterministic hashing for selection.
        Otherwise, returns the active strategy.
        
        Args:
            hash_key: Key for deterministic hash-based selection
            
        Returns:
            Selected strategy, or None if none configured
        """
        with self._lock:
            # If no weights, use active strategy
            if not self._weights:
                if self._active_strategy:
                    return self._strategies.get(self._active_strategy)
                # Return first registered strategy
                if self._strategies:
                    return next(iter(self._strategies.values()))
                return None
            
            # Weighted selection using deterministic hash
            return self._select_weighted(hash_key)
    
    def _select_weighted(self, hash_key: str) -> Optional[RoutingStrategy]:
        """Select strategy using weighted deterministic hashing."""
        if not self._weights:
            return None
        
        # Calculate total weight
        total_weight = sum(self._weights.values())
        if total_weight <= 0:
            return None
        
        # Generate deterministic hash value in [0, total_weight)
        hash_bytes = hashlib.sha256(hash_key.encode()).digest()
        hash_int = int.from_bytes(hash_bytes[:8], byteorder="big")
        hash_value = hash_int % total_weight
        
        # Select strategy based on hash position
        cumulative = 0
        for name, weight in self._weights.items():
            cumulative += weight
            if hash_value < cumulative:
                return self._strategies.get(name)
        
        # Should never reach here, but fallback to first
        first_name = next(iter(self._weights.keys()))
        return self._strategies.get(first_name)
    
    def add_update_callback(self, callback: Callable[[], None]) -> None:
        """Add callback to be notified on configuration updates."""
        with self._lock:
            self._update_callbacks.append(callback)
    
    def _notify_update(self) -> None:
        """Notify all callbacks of configuration update."""
        for callback in self._update_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Update callback error: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current registry status for admin/debugging."""
        with self._lock:
            return {
                "registered_strategies": list(self._strategies.keys()),
                "active_strategy": self._active_strategy,
                "ab_weights": self._weights,
                "ab_enabled": bool(self._weights),
            }


class RoutingPipeline:
    """
    Routing pipeline that orchestrates strategy selection and execution.
    
    Features:
    - Strategy selection via registry (single or A/B)
    - Automatic fallback to default strategy on errors
    - Telemetry emission via routeiq.router_decision.v1 contract
    """
    
    def __init__(
        self,
        registry: RoutingStrategyRegistry,
        default_strategy: Optional[RoutingStrategy] = None,
        emit_telemetry: bool = True,
    ):
        """
        Initialize routing pipeline.
        
        Args:
            registry: Strategy registry for selection
            default_strategy: Fallback strategy (auto-created if None)
            emit_telemetry: Whether to emit OTEL telemetry events
        """
        self._registry = registry
        self._default_strategy = default_strategy or DefaultStrategy()
        self._emit_telemetry = emit_telemetry
    
    def route(
        self,
        context: RoutingContext,
    ) -> RoutingResult:
        """
        Execute routing pipeline.
        
        1. Select strategy from registry (A/B or active)
        2. Execute strategy to get deployment
        3. Fallback to default on errors
        4. Emit telemetry event
        
        Args:
            context: Routing context with request details
            
        Returns:
            RoutingResult with selected deployment
        """
        start_time = time.time()
        result = RoutingResult()
        
        # Get hash key for A/B selection
        hash_key = context.get_ab_hash_key()
        
        # Select strategy
        strategy = self._registry.select_strategy(hash_key)
        strategy_name = strategy.name if strategy else DEFAULT_STRATEGY_NAME
        
        # Track which strategy was used
        result.strategy_name = strategy_name
        ab_weights = self._registry.get_weights()
        
        try:
            if strategy:
                deployment = strategy.select_deployment(context)
            else:
                # No strategy registered, use default
                strategy = self._default_strategy
                result.strategy_name = strategy.name
                deployment = strategy.select_deployment(context)
            
            result.deployment = deployment
            
        except Exception as e:
            logger.error(f"Strategy {strategy_name} failed: {e}")
            result.error = str(e)
            
            # Fallback to default strategy
            if strategy != self._default_strategy:
                try:
                    result.deployment = self._default_strategy.select_deployment(context)
                    result.is_fallback = True
                    result.fallback_reason = f"primary_failed: {e}"
                    result.strategy_name = self._default_strategy.name
                except Exception as fallback_error:
                    logger.error(f"Fallback strategy also failed: {fallback_error}")
                    result.error = f"Primary: {e}, Fallback: {fallback_error}"
        
        # Calculate latency
        result.latency_ms = (time.time() - start_time) * 1000
        
        # Emit telemetry
        if self._emit_telemetry:
            self._emit_routing_telemetry(context, result, ab_weights, hash_key)
        
        return result
    
    def _emit_routing_telemetry(
        self,
        context: RoutingContext,
        result: RoutingResult,
        ab_weights: Dict[str, int],
        hash_key: str,
    ) -> None:
        """Emit routing decision telemetry via OTEL."""
        try:
            from opentelemetry import trace
            
            # Get current span if any
            span = trace.get_current_span()
            if not span or not span.is_recording():
                return
            
            # Extract query length (no PII)
            query_length = 0
            if context.messages:
                for msg in context.messages:
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        query_length += len(content)
            elif context.input:
                query_length = len(str(context.input))
            
            # Get candidate models
            candidates = []
            router = context.router
            healthy = getattr(router, "healthy_deployments", router.model_list)
            for dep in healthy:
                if dep.get("model_name") == context.model:
                    candidates.append({
                        "model_name": dep.get("litellm_params", {}).get("model", ""),
                        "provider": dep.get("litellm_params", {}).get("custom_llm_provider"),
                        "available": True,
                    })
            
            # Build telemetry event
            builder = (
                RouterDecisionEventBuilder()
                .with_strategy(
                    name=result.strategy_name,
                    version=None,
                )
                .with_input(
                    query_length=query_length,
                    requested_model=context.model,
                    user_id=context.user_id,
                )
                .with_candidates(candidates)
                .with_selection(
                    selected=result.deployment.get("litellm_params", {}).get("model")
                    if result.deployment else None,
                    reason="ab_test" if ab_weights else "active_strategy",
                )
                .with_timing(total_ms=result.latency_ms)
            )
            
            # Add fallback info
            if result.is_fallback:
                builder.with_fallback(
                    triggered=True,
                    reason=result.fallback_reason,
                )
            
            # Add A/B testing info as custom attributes
            if ab_weights:
                builder.with_custom_attributes({
                    "ab_enabled": True,
                    "ab_weights": ab_weights,
                    "ab_hash_key": hash_key[:32],  # Truncate for privacy
                })
            
            # Set outcome
            if result.error and not result.deployment:
                builder.with_outcome(
                    status=RoutingOutcome.ERROR,
                    error_message=result.error,
                )
            elif result.deployment:
                builder.with_outcome(
                    status=RoutingOutcome.FALLBACK if result.is_fallback else RoutingOutcome.SUCCESS,
                )
            else:
                builder.with_outcome(status=RoutingOutcome.NO_CANDIDATES)
            
            # Add trace context
            span_context = span.get_span_context()
            if span_context.is_valid:
                builder.with_trace_context(
                    trace_id=format(span_context.trace_id, "032x"),
                    span_id=format(span_context.span_id, "016x"),
                )
            
            # Emit event
            event = builder.build()
            span.add_event(
                name=ROUTER_DECISION_EVENT_NAME,
                attributes={
                    ROUTER_DECISION_PAYLOAD_KEY: event.to_json(),
                },
            )
            
        except ImportError:
            # OTEL not available
            pass
        except Exception as e:
            logger.debug(f"Telemetry emission error: {e}")


# Singleton instances
_registry_instance: Optional[RoutingStrategyRegistry] = None
_pipeline_instance: Optional[RoutingPipeline] = None
_instance_lock = threading.Lock()


def get_routing_registry() -> RoutingStrategyRegistry:
    """Get the global routing strategy registry singleton."""
    global _registry_instance
    
    with _instance_lock:
        if _registry_instance is None:
            _registry_instance = RoutingStrategyRegistry()
        return _registry_instance


def get_routing_pipeline() -> RoutingPipeline:
    """Get the global routing pipeline singleton."""
    global _pipeline_instance
    
    with _instance_lock:
        if _pipeline_instance is None:
            registry = get_routing_registry()
            _pipeline_instance = RoutingPipeline(registry)
        return _pipeline_instance


def reset_routing_singletons() -> None:
    """Reset singletons (for testing)."""
    global _registry_instance, _pipeline_instance
    
    with _instance_lock:
        _registry_instance = None
        _pipeline_instance = None
