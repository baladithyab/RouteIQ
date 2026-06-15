"""
Routing-Strategy Adapter Contract
=================================

The STRATEGY-AGNOSTIC adapter ABI. Defines a ``RoutingAdapter`` Protocol that is
a superset of the existing ``RoutingStrategy`` ABC, an ``AdapterManifest`` that
carries the capability declaration (including the STEERING-required
``train_mode``), the feedback / artifact dataclasses the MLOps loop passes
around, and the SemVer ABI negotiation helper.

This is the contract every routing strategy/algorithm plugs into — the imported
LLMRouter classifiers, centroid, personalized, AND the Kumaraswamy-Thompson
bandit. The bandit is one *consumer*, the example — not the only target.

Design reference:
``docs/architecture/aws-rearchitecture/40-pluggable-routing-and-mlops.md`` §2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Optional,
    Protocol,
    Set,
    Tuple,
    runtime_checkable,
)

from litellm_llmrouter.gateway.plugin_manager import PluginCapability

if TYPE_CHECKING:
    from litellm_llmrouter.strategy_registry import RoutingContext

# SemVer; bump MINOR for additive, MAJOR for breaking.
ADAPTER_API_VERSION = "1.0"

# Allowed train modes (STEERING: per-strategy one-time-train vs continuous).
TRAIN_MODE_ONE_TIME = "one_time"
TRAIN_MODE_CONTINUOUS = "continuous"
_VALID_TRAIN_MODES = frozenset({TRAIN_MODE_ONE_TIME, TRAIN_MODE_CONTINUOUS})


@dataclass
class AdapterManifest:
    """Self-describing capability declaration for a routing adapter.

    The routing-specific analogue of ``PluginMetadata``. Lets the gateway answer,
    before a request, "can this adapter handle this request, does it learn, and
    what state/artifacts does it keep?"
    """

    name: str
    version: str = "v1"
    adapter_api_version: str = ADAPTER_API_VERSION
    family: str = ""

    # --- capabilities (what it can route) ---
    capabilities: Set[PluginCapability] = field(
        default_factory=lambda: {PluginCapability.ROUTING_STRATEGY}
    )
    api_formats: Set[str] = field(
        default_factory=lambda: {"chat", "responses", "messages"}
    )
    request_kinds: Set[str] = field(default_factory=lambda: {"completion"})

    # --- required signals (capability negotiation) ---
    required_signals: Set[str] = field(default_factory=set)
    optional_signals: Set[str] = field(default_factory=set)

    # --- learning / state ---
    learns: bool = False
    uses_artifact: bool = False
    train_mode: str = TRAIN_MODE_ONE_TIME  # one_time | continuous
    state_backend: str = "none"  # none | memory | redis | aurora
    artifact_kinds: Set[str] = field(default_factory=set)

    description: str = ""

    def __post_init__(self) -> None:
        if self.train_mode not in _VALID_TRAIN_MODES:
            self.train_mode = TRAIN_MODE_ONE_TIME
        if not self.family:
            self.family = self.name


@dataclass
class RoutingFeedback:
    """One reward observation routed to a learning adapter's ``update``."""

    model: str
    score: float
    request_id: Optional[str] = None
    bucket: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArtifactRef:
    """Reference to a trained artifact for ``load`` / ``reload``."""

    path: str = ""
    sha256: str = ""
    manifest: Optional[Dict[str, Any]] = None
    payload: Optional[Dict[str, Any]] = None


@runtime_checkable
class RoutingAdapter(Protocol):
    """The full adapter surface — a superset of ``RoutingStrategy``.

    ``route()`` is ``select_deployment()`` renamed; an alias on the base ABC
    (see :func:`attach_route_alias`) lets every existing strategy satisfy this
    Protocol with zero in-tree changes. ``update`` / ``load`` / ``reload`` are
    optional and present only when the manifest declares ``learns`` /
    ``uses_artifact``.
    """

    def declare_capabilities(self) -> AdapterManifest: ...

    def route(self, ctx: "RoutingContext") -> Optional[Dict[str, Any]]: ...


def _parse_semver(version: str) -> Tuple[int, int]:
    """Parse ``MAJOR.MINOR`` (ignoring patch/pre-release). Defaults to ``(0, 0)``."""
    try:
        parts = version.strip().split(".")
        major = int(parts[0]) if parts and parts[0] else 0
        minor = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        return (major, minor)
    except (ValueError, IndexError):
        return (0, 0)


def _abi_compatible(adapter_ver: str, gateway_ver: str = ADAPTER_API_VERSION) -> bool:
    """ABI negotiation: refuse MAJOR-newer adapters; accept MINOR-newer.

    A MINOR-newer adapter is loaded but only its known methods are used
    (duck-typed optionals make this safe). A MAJOR mismatch (either direction
    where the adapter's MAJOR exceeds the gateway's) is refused.

    Args:
        adapter_ver: The adapter's declared ``adapter_api_version``.
        gateway_ver: The gateway's ``ADAPTER_API_VERSION``.

    Returns:
        True if the adapter is loadable under the gateway's ABI.
    """
    a_major, _ = _parse_semver(adapter_ver)
    g_major, _ = _parse_semver(gateway_ver)
    # Refuse adapters whose MAJOR exceeds the gateway's; same MAJOR is fine
    # regardless of MINOR direction (additive-only contract).
    return a_major <= g_major


def attach_route_alias(strategy: Any) -> Any:
    """Give an existing ``RoutingStrategy`` a ``route()`` alias if it lacks one.

    The pipeline calls ``select_deployment``; the adapter contract calls
    ``route``. This binds ``route = select_deployment`` on instances that only
    have the ABC method, so they satisfy :class:`RoutingAdapter` without an
    in-tree edit.
    """
    if not hasattr(strategy, "route") and hasattr(strategy, "select_deployment"):
        try:
            strategy.route = strategy.select_deployment  # type: ignore[attr-defined]
        except Exception:
            pass
    return strategy
