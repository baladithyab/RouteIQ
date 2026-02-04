"""
Routing Strategy Registry and Pipeline
========================================

This module provides runtime strategy hot-swapping and A/B testing support
for LLMRouter routing strategies.

Features:
- Thread-safe strategy registry for registering multiple implementations
- Versioned strategy support: multiple versions per family with hot-swap
- Weighted A/B strategy selection with deterministic hashing
- Staged loading: load new config in staging, validate, then promote
- Routing pipeline with fallback support and telemetry emission
- Admin-safe update methods for runtime configuration changes

Configuration:
- LLMROUTER_ACTIVE_ROUTING_STRATEGY: Default active strategy (default: existing)
- LLMROUTER_STRATEGY_WEIGHTS: JSON dict of strategy weights for A/B testing
  Example: '{"baseline": 90, "candidate": 10}'
- LLMROUTER_EXPERIMENT_ID: Optional experiment identifier for telemetry
- LLMROUTER_EXPERIMENT_CONFIG: JSON config for advanced experiment setup

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
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from litellm_llmrouter.telemetry_contracts import (
    ROUTER_DECISION_EVENT_NAME,
    RoutingOutcome,
    ROUTER_DECISION_PAYLOAD_KEY,
    RouterDecisionEventBuilder,
)

# Import TG4.1 router decision span attributes helper
from litellm_llmrouter.observability import set_router_decision_attributes

logger = logging.getLogger(__name__)

# Configuration environment variables
ENV_ACTIVE_STRATEGY = "LLMROUTER_ACTIVE_ROUTING_STRATEGY"
ENV_STRATEGY_WEIGHTS = "LLMROUTER_STRATEGY_WEIGHTS"
ENV_EXPERIMENT_ID = "LLMROUTER_EXPERIMENT_ID"
ENV_EXPERIMENT_CONFIG = "LLMROUTER_EXPERIMENT_CONFIG"

# Default strategy name - matches existing LLMRouterStrategyFamily behavior
DEFAULT_STRATEGY_NAME = "llmrouter-default"


class StrategyState(str, Enum):
    """State of a strategy in the registry."""

    ACTIVE = "active"
    """Strategy is active and can serve requests."""

    STAGED = "staged"
    """Strategy is staged and awaiting promotion."""

    DISABLED = "disabled"
    """Strategy is disabled and will not serve requests."""


@dataclass
class VersionedStrategyEntry:
    """
    Registry entry for a strategy with version tracking.

    Supports multiple versions of the same strategy family,
    enabling canary deployments and A/B testing between versions.
    """

    strategy: "RoutingStrategy"
    """The strategy implementation."""

    name: str
    """Unique identifier (family:version or just name)."""

    family: str = ""
    """Strategy family (e.g., 'llmrouter-knn')."""

    version: str = ""
    """Strategy version (e.g., 'v1.0', sha256 prefix)."""

    state: StrategyState = StrategyState.ACTIVE
    """Current state of the strategy."""

    registered_at: float = field(default_factory=time.time)
    """Unix timestamp when registered."""

    last_activated_at: Optional[float] = None
    """Unix timestamp when last promoted to active."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata (model hash, etc.)."""

    def get_version_string(self) -> str:
        """Get version string for telemetry."""
        if self.version:
            return self.version
        if hasattr(self.strategy, "version") and self.strategy.version:
            return self.strategy.version
        return "unknown"


@dataclass
class ExperimentConfig:
    """
    Configuration for an A/B testing experiment.

    Enables tracking of experiment assignments with consistent
    naming for downstream analysis.
    """

    experiment_id: str
    """Unique experiment identifier (e.g., 'routing-v2-to-rollout-2024Q1')."""

    variants: Dict[str, str]
    """Mapping of variant name to strategy name (e.g., {'control': 'baseline', 'treatment': 'candidate'})."""

    weights: Dict[str, int]
    """Weight for each variant (e.g., {'control': 90, 'treatment': 10})"""

    enabled: bool = True
    """Whether the experiment is active."""

    start_time: Optional[float] = None
    """Unix timestamp when experiment started."""

    end_time: Optional[float] = None
    """Unix timestamp when experiment should end (optional)."""

    description: str = ""
    """Human-readable description of the experiment."""


@dataclass
class StagedStrategy:
    """
    A strategy that is staged for promotion.

    Staged strategies are validated before being promoted to active.
    On validation failure, they remain staged without affecting
    the current active configuration.
    """

    entry: VersionedStrategyEntry
    """The staged strategy entry."""

    staged_at: float = field(default_factory=time.time)
    """Unix timestamp when staged."""

    validation_passed: bool = False
    """Whether validation has passed."""

    validation_error: Optional[str] = None
    """Error message if validation failed."""

    auto_promote: bool = False
    """Whether to auto-promote after successful validation."""


@dataclass
class ABSelectionResult:
    """Result of A/B strategy selection with full context for telemetry."""

    strategy: Optional["RoutingStrategy"]
    """Selected strategy instance."""

    strategy_name: str
    """Name of the selected strategy."""

    variant: Optional[str] = None
    """Variant name if from experiment (e.g., 'control', 'treatment')."""

    experiment_id: Optional[str] = None
    """Experiment ID if from experiment."""

    weight: Optional[int] = None
    """Weight of the selected variant."""

    total_weight: Optional[int] = None
    """Total weight across all variants."""

    hash_bucket: Optional[int] = None
    """Deterministic hash bucket (0 to total_weight-1)."""

    hash_key_type: str = "unknown"
    """Type of key used for hashing (user, request, random)."""

    version: Optional[str] = None
    """Strategy version string."""


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

    tenant_id: Optional[str] = None
    """Tenant identifier for multi-tenant sticky assignment."""

    # Metadata for telemetry
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata for telemetry."""

    def get_ab_hash_key(self) -> Tuple[str, str]:
        """
        Get the key used for deterministic A/B hash assignment.

        Priority: tenant_id+user_id > user_id > request_id > random UUID
        Using user_id provides sticky assignment (same user always gets same variant).

        Returns:
            Tuple of (hash_key, key_type) where key_type indicates the source.
        """
        if self.tenant_id and self.user_id:
            return (
                f"tenant:{self.tenant_id}:user:{self.user_id}:model:{self.model}",
                "tenant_user",
            )
        if self.user_id:
            return f"user:{self.user_id}:model:{self.model}", "user"
        if self.request_id:
            return f"request:{self.request_id}:model:{self.model}", "request"
        # Fallback: generate random key (no stickiness)
        import uuid

        return f"random:{uuid.uuid4()}", "random"


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

    # A/B testing context for telemetry
    ab_selection: Optional[ABSelectionResult] = None
    """Full A/B selection context."""


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

    def validate(self) -> Tuple[bool, Optional[str]]:
        """
        Validate the strategy is ready to serve requests.

        Override in subclasses for custom validation logic
        (e.g., checking model is loaded, dependencies available).

        Returns:
            Tuple of (is_valid, error_message)
        """
        return True, None


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
        self._strategies: Dict[int, Any] = {}
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
    Thread-safe registry for routing strategies with versioning and staged loading.

    Supports:
    - Registering multiple strategy implementations by name
    - Versioned strategies: multiple versions per family
    - Setting an active strategy
    - Weighted A/B strategy selection with deterministic hashing
    - Staged loading: load new config, validate, then promote
    - Thread-safe updates for runtime configuration
    """

    def __init__(self):
        self._strategies: Dict[str, VersionedStrategyEntry] = {}
        self._staged: Dict[str, StagedStrategy] = {}
        self._active_strategy: Optional[str] = None
        self._weights: Dict[str, int] = {}
        self._experiment: Optional[ExperimentConfig] = None
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

        # Experiment configuration
        experiment_id = os.getenv(ENV_EXPERIMENT_ID)
        experiment_json = os.getenv(ENV_EXPERIMENT_CONFIG)

        if experiment_json:
            try:
                config = json.loads(experiment_json)
                self._experiment = ExperimentConfig(
                    experiment_id=config.get(
                        "experiment_id", experiment_id or "default"
                    ),
                    variants=config.get("variants", {}),
                    weights=config.get("weights", {}),
                    enabled=config.get("enabled", True),
                    description=config.get("description", ""),
                )
                logger.info(
                    f"Loaded experiment config: {self._experiment.experiment_id}"
                )
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse {ENV_EXPERIMENT_CONFIG}: {e}")
        elif experiment_id:
            # Simple experiment ID without full config
            self._experiment = ExperimentConfig(
                experiment_id=experiment_id,
                variants={},
                weights=self._weights,
            )

    def register(
        self,
        name: str,
        strategy: RoutingStrategy,
        version: Optional[str] = None,
        family: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register a routing strategy by name.

        Args:
            name: Unique name for the strategy
            strategy: Strategy implementation
            version: Optional version string
            family: Optional family name (defaults to name without version)
            metadata: Optional metadata dict
        """
        with self._lock:
            entry = VersionedStrategyEntry(
                strategy=strategy,
                name=name,
                family=family or name.split(":")[0],
                version=version or getattr(strategy, "version", None) or "",
                state=StrategyState.ACTIVE,
                metadata=metadata or {},
            )
            self._strategies[name] = entry
            logger.info(
                f"Registered routing strategy: {name} (version={entry.version})"
            )

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
            entry = self._strategies.get(name)
            return entry.strategy if entry else None

    def get_entry(self, name: str) -> Optional[VersionedStrategyEntry]:
        """Get a strategy entry by name."""
        with self._lock:
            return self._strategies.get(name)

    def list_strategies(self) -> List[str]:
        """List all registered strategy names."""
        with self._lock:
            return list(self._strategies.keys())

    def list_versions(self, family: str) -> List[VersionedStrategyEntry]:
        """List all versions of a strategy family."""
        with self._lock:
            return [
                entry for entry in self._strategies.values() if entry.family == family
            ]

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
            self._experiment = None  # Clear experiment
            self._strategies[name].state = StrategyState.ACTIVE
            self._strategies[name].last_activated_at = time.time()
            logger.info(f"Set active routing strategy: {name}")
            self._notify_update()
            return True

    def get_active(self) -> Optional[str]:
        """Get the name of the active strategy (if not using A/B)."""
        with self._lock:
            return self._active_strategy if not self._weights else None

    def set_weights(
        self,
        weights: Dict[str, int],
        experiment_id: Optional[str] = None,
    ) -> bool:
        """
        Set strategy weights for A/B testing.

        Weights are relative (not percentages). Example:
        - {"baseline": 90, "candidate": 10} gives 90% baseline, 10% candidate
        - {"a": 1, "b": 1, "c": 1} gives 33% each

        Args:
            weights: Dict mapping strategy names to relative weights
            experiment_id: Optional experiment identifier for telemetry

        Returns:
            True if all strategies exist, False otherwise
        """
        with self._lock:
            # Validate all strategies exist
            for name in weights:
                if name not in self._strategies:
                    logger.error(
                        f"Cannot set weights: strategy '{name}' not registered"
                    )
                    return False

            self._weights = weights.copy()
            self._active_strategy = None  # Clear single active strategy

            # Create or update experiment config
            if experiment_id or self._experiment:
                self._experiment = ExperimentConfig(
                    experiment_id=experiment_id
                    or (
                        self._experiment.experiment_id
                        if self._experiment
                        else "default"
                    ),
                    variants={name: name for name in weights.keys()},
                    weights=weights,
                )

            logger.info(f"Set A/B strategy weights: {weights}")
            self._notify_update()
            return True

    def set_experiment(self, config: ExperimentConfig) -> bool:
        """
        Set full experiment configuration.

        Args:
            config: Experiment configuration with variants and weights

        Returns:
            True if all variant strategies exist, False otherwise
        """
        with self._lock:
            # Validate all variant strategies exist
            for variant_name, strategy_name in config.variants.items():
                if strategy_name not in self._strategies:
                    logger.error(
                        f"Cannot set experiment: strategy '{strategy_name}' for variant '{variant_name}' not registered"
                    )
                    return False

            self._experiment = config
            self._weights = config.weights.copy()
            self._active_strategy = None

            logger.info(
                f"Set experiment: {config.experiment_id} with variants {config.variants}"
            )
            self._notify_update()
            return True

    def get_experiment(self) -> Optional[ExperimentConfig]:
        """Get the current experiment configuration."""
        with self._lock:
            return self._experiment

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
                self._experiment = None
                logger.info("Cleared A/B weights")
                self._notify_update()

    # === Staged Loading API ===

    def stage_strategy(
        self,
        name: str,
        strategy: RoutingStrategy,
        version: Optional[str] = None,
        auto_promote: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Stage a strategy for validation before promotion.

        Staged strategies are validated but not yet active.
        Call promote_staged() to activate after validation.

        Args:
            name: Unique name for the strategy
            strategy: Strategy implementation
            version: Optional version string
            auto_promote: Whether to auto-promote after successful validation
            metadata: Optional metadata dict

        Returns:
            Tuple of (success, error_message)
        """
        with self._lock:
            entry = VersionedStrategyEntry(
                strategy=strategy,
                name=name,
                family=name.split(":")[0],
                version=version or "",
                state=StrategyState.STAGED,
                metadata=metadata or {},
            )

            validation_passed, error = strategy.validate()

            staged = StagedStrategy(
                entry=entry,
                validation_passed=validation_passed,
                validation_error=error,
                auto_promote=auto_promote,
            )

            self._staged[name] = staged
            logger.info(
                f"Staged strategy {name} (version={entry.version}, auto_promote={auto_promote})"
            )

            if validation_passed:
                return self.promote_staged(name)

            return True, error

    def promote_staged(self, name: str) -> Tuple[bool, Optional[str]]:
        """
        Promote a staged strategy to active.

        The strategy must have passed validation.
        On promotion, the staged entry is moved to the active registry.

        Args:
            name: Name of staged strategy to promote

        Returns:
            Tuple of (success, error_message)
        """
        with self._lock:
            if name not in self._staged:
                return False, f"No staged strategy named '{name}'"

            staged = self._staged[name]

            if not staged.validation_passed:
                return (
                    False,
                    f"Strategy '{name}' has not passed validation: {staged.validation_error}",
                )

            # Move from staged to active
            entry = staged.entry
            entry.state = StrategyState.ACTIVE
            entry.last_activated_at = time.time()

            self._strategies[name] = entry
            del self._staged[name]

            logger.info(f"Promoted staged strategy {name} to active")
            self._notify_update()

            return True, None

    def rollback_staged(self, name: str) -> bool:
        """
        Rollback (discard) a staged strategy.

        Args:
            name: Name of staged strategy to discard

        Returns:
            True if discarded, False if not found
        """
        with self._lock:
            if name in self._staged:
                del self._staged[name]
                logger.info(f"Rolled back staged strategy: {name}")
                return True
            return False

    def get_staged(self, name: str) -> Optional[StagedStrategy]:
        """Get a staged strategy by name."""
        with self._lock:
            return self._staged.get(name)

    def list_staged(self) -> List[str]:
        """List all staged strategy names."""
        with self._lock:
            return list(self._staged.keys())

    # === Strategy Selection ===

    def select_strategy(
        self,
        hash_key: str,
        hash_key_type: str = "unknown",
    ) -> ABSelectionResult:
        """
        Select a strategy for routing.

        If A/B weights are set, uses deterministic hashing for selection.
        Otherwise, returns the active strategy.

        Args:
            hash_key: Key for deterministic hash-based selection
            hash_key_type: Type of key for telemetry

        Returns:
            ABSelectionResult with full context for telemetry
        """
        with self._lock:
            # If no weights, use active strategy
            if not self._weights:
                if self._active_strategy:
                    entry = self._strategies.get(self._active_strategy)
                    if entry:
                        return ABSelectionResult(
                            strategy=entry.strategy,
                            strategy_name=self._active_strategy,
                            version=entry.get_version_string(),
                            hash_key_type=hash_key_type,
                        )
                # Return first registered strategy
                if self._strategies:
                    first_name = next(iter(self._strategies.keys()))
                    entry = self._strategies[first_name]
                    return ABSelectionResult(
                        strategy=entry.strategy,
                        strategy_name=first_name,
                        version=entry.get_version_string(),
                        hash_key_type=hash_key_type,
                    )
                return ABSelectionResult(
                    strategy=None,
                    strategy_name="",
                    hash_key_type=hash_key_type,
                )

            # Weighted selection using deterministic hash
            return self._select_weighted(hash_key, hash_key_type)

    def _select_weighted(
        self,
        hash_key: str,
        hash_key_type: str,
    ) -> ABSelectionResult:
        """Select strategy using weighted deterministic hashing."""
        if not self._weights:
            return ABSelectionResult(
                strategy=None,
                strategy_name="",
                hash_key_type=hash_key_type,
            )

        # Calculate total weight
        total_weight = sum(self._weights.values())
        if total_weight <= 0:
            return ABSelectionResult(
                strategy=None,
                strategy_name="",
                hash_key_type=hash_key_type,
            )

        # Generate deterministic hash value in [0, total_weight)
        hash_bytes = hashlib.sha256(hash_key.encode()).digest()
        hash_int = int.from_bytes(hash_bytes[:8], byteorder="big")
        hash_bucket = hash_int % total_weight

        # Select strategy based on hash position
        cumulative = 0
        selected_name = ""
        selected_weight = 0

        for name, weight in self._weights.items():
            cumulative += weight
            if hash_bucket < cumulative:
                selected_name = name
                selected_weight = weight
                break

        # Should never reach here, but fallback to first
        if not selected_name:
            selected_name = next(iter(self._weights.keys()))
            selected_weight = self._weights[selected_name]

        # Get strategy entry
        entry = self._strategies.get(selected_name)
        strategy = entry.strategy if entry else None
        version = entry.get_version_string() if entry else None

        # Determine variant name from experiment config
        variant = None
        experiment_id = None
        if self._experiment:
            experiment_id = self._experiment.experiment_id
            # Reverse lookup: strategy_name -> variant
            for var_name, strat_name in self._experiment.variants.items():
                if strat_name == selected_name:
                    variant = var_name
                    break
            if not variant:
                variant = selected_name  # Use strategy name as variant

        return ABSelectionResult(
            strategy=strategy,
            strategy_name=selected_name,
            variant=variant,
            experiment_id=experiment_id,
            weight=selected_weight,
            total_weight=total_weight,
            hash_bucket=hash_bucket,
            hash_key_type=hash_key_type,
            version=version,
        )

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
                "experiment": (
                    {
                        "id": (
                            self._experiment.experiment_id if self._experiment else None
                        ),
                        "variants": (
                            self._experiment.variants if self._experiment else {}
                        ),
                        "enabled": (
                            self._experiment.enabled if self._experiment else False
                        ),
                    }
                    if self._experiment
                    else None
                ),
                "staged_strategies": list(self._staged.keys()),
                "strategy_versions": {
                    name: entry.get_version_string()
                    for name, entry in self._strategies.items()
                },
            }

    def reload_from_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Reload registry configuration from a config dict.

        Used by hot reload to update weights/experiment without restart.

        Args:
            config: Dict with optional keys: weights, experiment, active_strategy

        Returns:
            Tuple of (success, list of errors)
        """
        errors = []

        with self._lock:
            # Update weights
            if "weights" in config:
                weights = config["weights"]
                for name in weights:
                    if name not in self._strategies:
                        errors.append(f"Unknown strategy in weights: {name}")
                if not errors:
                    self._weights = weights
                    self._active_strategy = None
                    logger.info(f"Reloaded weights from config: {weights}")

            # Update experiment
            if "experiment" in config:
                exp_config = config["experiment"]
                try:
                    self._experiment = ExperimentConfig(
                        experiment_id=exp_config.get("experiment_id", "default"),
                        variants=exp_config.get("variants", {}),
                        weights=exp_config.get("weights", self._weights),
                        enabled=exp_config.get("enabled", True),
                        description=exp_config.get("description", ""),
                    )
                    logger.info(
                        f"Reloaded experiment config: {self._experiment.experiment_id}"
                    )
                except Exception as e:
                    errors.append(f"Invalid experiment config: {e}")

            # Update active strategy (if not using weights)
            if "active_strategy" in config and not self._weights:
                active = config["active_strategy"]
                if active in self._strategies:
                    self._active_strategy = active
                    logger.info(f"Reloaded active strategy: {active}")
                else:
                    errors.append(f"Unknown active strategy: {active}")

            if not errors:
                self._notify_update()

        return len(errors) == 0, errors


class RoutingPipeline:
    """
    Routing pipeline that orchestrates strategy selection and execution.

    Features:
    - Strategy selection via registry (single or A/B)
    - Automatic fallback to default strategy on errors
    - Telemetry emission via routeiq.router_decision.v1.1 contract
    - Full experiment assignment tracking
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
        4. Emit telemetry event with experiment assignment

        Args:
            context: Routing context with request details

        Returns:
            RoutingResult with selected deployment
        """
        start_time = time.time()
        result = RoutingResult()

        # Get hash key for A/B selection
        hash_key, hash_key_type = context.get_ab_hash_key()

        # Select strategy with full A/B context
        ab_selection = self._registry.select_strategy(hash_key, hash_key_type)
        strategy = ab_selection.strategy
        strategy_name = ab_selection.strategy_name or DEFAULT_STRATEGY_NAME

        # Track which strategy was used
        result.strategy_name = strategy_name
        result.ab_selection = ab_selection

        try:
            if strategy:
                deployment = strategy.select_deployment(context)
            else:
                # No strategy registered, use default
                strategy = self._default_strategy
                result.strategy_name = strategy.name
                ab_selection.strategy_name = strategy.name
                deployment = strategy.select_deployment(context)

            result.deployment = deployment

        except Exception as e:
            logger.error(f"Strategy {strategy_name} failed: {e}")
            result.error = str(e)

            # Fallback to default strategy
            if strategy != self._default_strategy:
                try:
                    result.deployment = self._default_strategy.select_deployment(
                        context
                    )
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
            self._emit_routing_telemetry(context, result, hash_key)

        return result

    def _emit_routing_telemetry(
        self,
        context: RoutingContext,
        result: RoutingResult,
        hash_key: str,
    ) -> None:
        """Emit routing decision telemetry via OTEL with experiment assignment."""
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
                    candidates.append(
                        {
                            "model_name": dep.get("litellm_params", {}).get(
                                "model", ""
                            ),
                            "provider": dep.get("litellm_params", {}).get(
                                "custom_llm_provider"
                            ),
                            "available": True,
                        }
                    )

            # Get A/B selection info
            ab = result.ab_selection
            ab_weights = self._registry.get_weights()

            # Build telemetry event
            builder = (
                RouterDecisionEventBuilder()
                .with_strategy(
                    name=result.strategy_name,
                    version=ab.version if ab else None,
                )
                .with_input(
                    query_length=query_length,
                    requested_model=context.model,
                    user_id=context.user_id,
                )
                .with_candidates(candidates)
                .with_selection(
                    selected=(
                        result.deployment.get("litellm_params", {}).get("model")
                        if result.deployment
                        else None
                    ),
                    reason="ab_test" if ab_weights else "active_strategy",
                )
                .with_timing(total_ms=result.latency_ms)
            )

            # Add experiment assignment (v1.1)
            if ab and (ab.experiment_id or ab_weights):
                builder.with_experiment(
                    experiment_id=ab.experiment_id,
                    variant=ab.variant,
                    strategy_name=ab.strategy_name,
                    strategy_version=ab.version,
                    weight=ab.weight,
                    total_weight=ab.total_weight,
                    hash_bucket=ab.hash_bucket,
                    hash_key_type=ab.hash_key_type,
                )

            # Add fallback info
            if result.is_fallback:
                builder.with_fallback(
                    triggered=True,
                    reason=result.fallback_reason,
                )

            # Add trace context
            span_context = span.get_span_context()
            if span_context.is_valid:
                builder.with_trace_context(
                    trace_id=format(span_context.trace_id, "032x"),
                    span_id=format(span_context.span_id, "016x"),
                )

            # Add span attributes for experiment tracking
            span.set_attribute("routing.strategy_name", result.strategy_name)
            if ab:
                if ab.version:
                    span.set_attribute("routing.strategy_version", ab.version)
                if ab.experiment_id:
                    span.set_attribute("routing.experiment", ab.experiment_id)
                if ab.variant:
                    span.set_attribute("routing.variant", ab.variant)

            # TG4.1: Set router decision span attributes for observability
            selected_model = (
                result.deployment.get("litellm_params", {}).get("model")
                if result.deployment
                else None
            )

            # Determine outcome for TG4.1 attributes
            if result.error and not result.deployment:
                outcome = RoutingOutcome.ERROR.value
                reason = f"strategy_error: {result.error}"
            elif result.deployment and result.is_fallback:
                outcome = RoutingOutcome.FALLBACK.value
                reason = result.fallback_reason or "primary_failed"
            elif result.deployment:
                outcome = RoutingOutcome.SUCCESS.value
                reason = "ab_test" if ab_weights else "active_strategy"
            else:
                outcome = RoutingOutcome.NO_CANDIDATES.value
                reason = "no_candidates_available"

            set_router_decision_attributes(
                span,
                strategy=result.strategy_name,
                model_selected=selected_model,
                candidates_evaluated=len(candidates),
                outcome=outcome,
                reason=reason,
                latency_ms=result.latency_ms,
                error_type=type(Exception(result.error)).__name__ if result.error else None,
                error_message=result.error if result.error else None,
                strategy_version=ab.version if ab else None,
                fallback_triggered=result.is_fallback,
            )

            # Emit event
            event = builder.build()
            span.add_event(
                ROUTER_DECISION_EVENT_NAME,
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
