"""
RouteIQ Custom Routing Strategy — Uses LiteLLM's official CustomRoutingStrategyBase API.

This module replaces the monkey-patch approach in routing_strategy_patch.py with LiteLLM's
official plugin API for custom routing strategies. It provides the same routing capabilities
(ML-based strategy selection, A/B testing, telemetry) without modifying LiteLLM's Router class.

Usage::

    from litellm_llmrouter.custom_routing_strategy import install_routeiq_strategy

    # After creating a LiteLLM Router instance:
    strategy = install_routeiq_strategy(router, strategy_name="llmrouter-knn")

    # Or create and install separately:
    from litellm_llmrouter.custom_routing_strategy import create_routeiq_strategy
    strategy = create_routeiq_strategy(router, strategy_name="llmrouter-knn")
    router.set_custom_routing_strategy(strategy)

The ``install_routeiq_strategy`` helper calls ``router.set_custom_routing_strategy()``
which replaces the Router's ``get_available_deployment`` and
``async_get_available_deployment`` methods with the strategy's implementations.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from fastapi import HTTPException

# Import CustomRoutingStrategyBase with graceful fallback for test environments
try:
    from litellm.types.router import CustomRoutingStrategyBase
except ImportError:  # pragma: no cover – litellm may not be installed in test env

    class CustomRoutingStrategyBase:  # type: ignore[no-redef]
        """Stub base class when litellm is not available."""

        async def async_get_available_deployment(
            self,
            model: str,
            messages: Optional[List[Dict[str, str]]] = None,
            input: Optional[Union[str, List]] = None,
            specific_deployment: Optional[bool] = False,
            request_kwargs: Optional[Dict] = None,
        ):
            pass

        def get_available_deployment(
            self,
            model: str,
            messages: Optional[List[Dict[str, str]]] = None,
            input: Optional[Union[str, List]] = None,
            specific_deployment: Optional[bool] = False,
            request_kwargs: Optional[Dict] = None,
        ):
            pass


# Import centroid routing with graceful fallback
try:
    from litellm_llmrouter.centroid_routing import (
        CentroidRoutingStrategy,
        get_centroid_strategy,
        warmup_centroid_classifier,
        RoutingProfile,
    )

    CENTROID_ROUTING_AVAILABLE = True
except ImportError:
    CENTROID_ROUTING_AVAILABLE = False

# Import personalized routing with graceful fallback
try:
    from litellm_llmrouter.personalized_routing import (
        _record_selection_metric as _record_personalized_selection,
        get_personalized_router,
    )

    PERSONALIZED_ROUTING_AVAILABLE = True
except ImportError:
    PERSONALIZED_ROUTING_AVAILABLE = False


logger = logging.getLogger(__name__)


def _load_routing_flags() -> tuple:
    """Load routing feature flags from typed settings, falling back to env vars.

    Returns:
        Tuple of ``(use_pipeline, centroid_enabled, default_profile)``.
    """
    try:
        from litellm_llmrouter.settings import get_settings

        settings = get_settings()
        use_pipeline = settings.routing.pipeline_enabled
        centroid_enabled = settings.routing.centroid_enabled
        default_profile = settings.routing.default_profile.value
        return use_pipeline, centroid_enabled, default_profile
    except Exception:
        use_pipeline = os.getenv("LLMROUTER_USE_PIPELINE", "true").lower() == "true"
        centroid_enabled = (
            os.getenv("ROUTEIQ_CENTROID_ROUTING", "true").lower() == "true"
        )
        default_profile = os.getenv("ROUTEIQ_ROUTING_PROFILE", "auto")
        return use_pipeline, centroid_enabled, default_profile


# Feature flags (loaded once at import time, with typed-settings-first)
USE_PIPELINE_ROUTING, CENTROID_ROUTING_ENABLED, DEFAULT_ROUTING_PROFILE = (
    _load_routing_flags()
)

# Maximum routing attempts per request to prevent amplification loops
MAX_ROUTING_ATTEMPTS = 3


@dataclass
class _CandidateSet:
    """The per-request candidate picture every selection path needs (RouteIQ-5007).

    Carries not just the scored ``model_list`` but the FILTERED ``routable`` set,
    the unfiltered ``healthy_deployments`` (legacy match pool), the pre-filter
    ``group_matched`` members, and the ``context`` so callers can DISTINGUISH a
    fail-closed empty (a HARD residency / STRICT capability constraint excluded
    every arm) from a genuinely-empty group or the legacy no-context path.

    Without this distinction the ML path repopulated an empty (fail-closed)
    ``model_list`` to ``[model]`` and matched it against the UNFILTERED
    ``healthy_deployments`` -- leaking an out-of-region arm for a hard-residency
    request (the RouteIQ-5007 leak).
    """

    model_list: List[str] = field(default_factory=list)
    healthy_deployments: List[Dict] = field(default_factory=list)
    routable: List[Dict] = field(default_factory=list)
    group_matched: List[Dict] = field(default_factory=list)
    context: Optional[Any] = None

    @property
    def filtered_empty(self) -> bool:
        """True when a per-request filter excluded EVERY arm of a non-empty group.

        This is the fail-closed verdict: candidates existed
        (``group_matched`` non-empty) but a context-driven HARD residency /
        STRICT capability filter dropped them all (``routable`` empty). The
        selection path MUST return None here -- NEVER repopulate from the
        unfiltered set. When ``context`` is ``None`` (legacy / no per-request
        signal) OR the group genuinely had no members, this is False and the
        legacy ``[model]`` fallback is preserved (byte-stable).
        """
        return (
            self.context is not None and bool(self.group_matched) and not self.routable
        )

    @property
    def context_constrained(self) -> bool:
        """True when a per-request signal context was applied to a non-empty group.

        On this path the FILTERED ``routable`` set (not the unfiltered
        ``healthy_deployments``) is the safe match pool, so a group-fallback can
        never reach an out-of-region / sub-tier arm. False keeps the legacy
        ``healthy_deployments`` match pool (byte-stable for no-context callers).
        """
        return self.context is not None and bool(self.group_matched)


def _resolve_routing_profile(
    request_kwargs: Optional[Dict] = None,
) -> Optional[str]:
    """Resolve the routing profile from request metadata or env var.

    Args:
        request_kwargs: Request keyword arguments (may contain metadata).

    Returns:
        Routing profile string, or None if not configured.
    """
    # 1. Check request metadata
    if request_kwargs:
        metadata = request_kwargs.get("metadata", {})
        if isinstance(metadata, dict):
            profile = metadata.get("routing_profile")
            if profile:
                return str(profile).lower()

    # 2. Fall back to env var default
    return DEFAULT_ROUTING_PROFILE


class RouteIQRoutingStrategy(CustomRoutingStrategyBase):
    """
    Custom routing strategy using LiteLLM's official ``CustomRoutingStrategyBase`` API.

    This class replaces the monkey-patch system in ``routing_strategy_patch.py`` by
    implementing LiteLLM's plugin API for custom routing. When installed via
    ``router.set_custom_routing_strategy()``, the Router delegates all deployment
    selection to this class.

    Features:
    - **ML-based routing** via ``LLMRouterStrategyFamily`` (KNN, SVM, MLP, etc.)
    - **A/B testing** via ``RoutingPipeline`` with deterministic hashing
    - **Centroid routing fallback** for zero-config intelligent routing (~2ms)
    - **Routing profiles** (auto, eco, premium, free, reasoning)
    - **Amplification guard** prevents infinite routing loops (max 3 per request)
    - **Graceful fallback** to first available deployment if all routing fails

    Progressive enhancement chain:
    Pipeline strategies (KNN, SVM, etc.) → Centroid routing (~2ms) → First healthy deployment

    Args:
        router_instance: The LiteLLM Router instance (for accessing model_list, etc.)
        strategy_name: Optional ML strategy name (e.g., "llmrouter-knn")
    """

    def __init__(
        self,
        router_instance: Any,
        strategy_name: Optional[str] = None,
    ) -> None:
        self._router = router_instance
        self._strategy_name = strategy_name
        self._strategy_args: Dict[str, Any] = {}

        # Amplification guard: {request_id: attempt_count}
        self._routing_attempts: Dict[str, int] = {}

        # Lazy-initialized components
        self._strategy_instance: Any = None
        self._pipeline: Any = None
        self._pipeline_initialized = False

        # Centroid routing state (lazy-initialized)
        self._centroid_strategy: Any = None
        self._centroid_initialized = False

    # ------------------------------------------------------------------
    # Public API: LiteLLM CustomRoutingStrategyBase
    # ------------------------------------------------------------------

    async def async_get_available_deployment(
        self,
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        specific_deployment: Optional[bool] = False,
        request_kwargs: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """Async entry point called by LiteLLM's Router.

        Thin wrapper: delegates selection to :meth:`_async_select` and routes
        the result through :meth:`_guard_selected` -- the single gov-ban
        chokepoint (RouteIQ-a073) every path returns through.
        """
        selected = await self._async_select(
            model=model,
            messages=messages,
            input=input,
            specific_deployment=specific_deployment,
            request_kwargs=request_kwargs,
        )
        return self._guard_selected(
            selected, model, request_kwargs=request_kwargs, messages=messages
        )

    async def _async_select(
        self,
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        specific_deployment: Optional[bool] = False,
        request_kwargs: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """
        Async deployment selection (guarded by the caller).

        Order of operations:
        0. Governance enforcement (budget, rate limit, model access)
        1. Check amplification guard
        2. Try pipeline routing (A/B testing)
        3. Fall back to direct LLMRouter ML routing
        4. Try personalized re-ranking (if enabled and user_id available)
        5. Fall back to centroid routing (zero-config intelligent routing)
        6. Fall back to first available deployment
        """
        # 0. Governance enforcement (before routing)
        await self._enforce_governance(model, request_kwargs)

        # 1. Amplification guard
        self._check_amplification_guard(request_kwargs)

        # 1b. KV-cache-/queue-aware engine routing (RouteIQ-08d6/6a89). Gated
        #     default OFF -> no-op (strategy absent from the registry). When ON,
        #     consumes the live scraped engine gauges to land load on the
        #     least-busy self-hosted arm BEFORE the ML/pipeline decision.
        result = await self._route_via_kv_cache_aware(
            model=model,
            messages=messages,
            request_kwargs=request_kwargs,
        )
        if result is not None:
            return result

        # 2. Try pipeline routing
        if USE_PIPELINE_ROUTING:
            result = self._route_via_pipeline(
                model=model,
                messages=messages,
                input=input,
                request_kwargs=request_kwargs,
            )
            if result is not None:
                return result

        # 3. Try direct LLMRouter routing
        try:
            result = self._route_via_llmrouter(
                model=model,
                messages=messages,
                input=input,
                request_kwargs=request_kwargs,
            )
            if result is not None:
                return result
        except Exception as e:
            logger.warning(f"ML routing failed, falling back: {e}")

        # 4. Try personalized routing (re-ranks centroid candidates)
        result = await self._route_via_personalized(
            model=model,
            request_kwargs=request_kwargs,
            messages=messages,
        )
        if result is not None:
            return result

        # 5. Try centroid routing (zero-config intelligent routing)
        result = self._route_via_centroid(
            model=model,
            messages=messages,
            request_kwargs=request_kwargs,
        )
        if result is not None:
            return result

        # 6. Fallback to first available deployment (RouteIQ-8e37: thread messages
        #    so the capability-tier floor applies on the terminal fallback too).
        return self._fallback_deployment(model, request_kwargs, messages)

    def get_available_deployment(
        self,
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        specific_deployment: Optional[bool] = False,
        request_kwargs: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """Sync entry point called by LiteLLM's Router.

        Thin wrapper: delegates to :meth:`_sync_select` and routes the result
        through :meth:`_guard_selected` -- the single gov-ban chokepoint
        (RouteIQ-a073) every path returns through.
        """
        selected = self._sync_select(
            model=model,
            messages=messages,
            input=input,
            specific_deployment=specific_deployment,
            request_kwargs=request_kwargs,
        )
        return self._guard_selected(
            selected, model, request_kwargs=request_kwargs, messages=messages
        )

    def _sync_select(
        self,
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        specific_deployment: Optional[bool] = False,
        request_kwargs: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """
        Sync deployment selection (guarded by the caller).

        Same logic as the async version but uses synchronous calls.
        Note: Personalized routing is skipped in sync path because the
        preference store requires async Redis operations.
        Note: Governance enforcement is skipped in sync path because the
        governance engine requires async operations.
        """
        # 1. Amplification guard
        self._check_amplification_guard(request_kwargs)

        # 2. Try pipeline routing
        if USE_PIPELINE_ROUTING:
            result = self._route_via_pipeline(
                model=model,
                messages=messages,
                input=input,
                request_kwargs=request_kwargs,
            )
            if result is not None:
                return result

        # 3. Try direct LLMRouter routing
        try:
            result = self._route_via_llmrouter(
                model=model,
                messages=messages,
                input=input,
                request_kwargs=request_kwargs,
            )
            if result is not None:
                return result
        except Exception as e:
            logger.warning(f"ML routing failed, falling back: {e}")

        # 4. Try centroid routing (zero-config intelligent routing)
        result = self._route_via_centroid(
            model=model,
            messages=messages,
            request_kwargs=request_kwargs,
        )
        if result is not None:
            return result

        # 5. Fallback to first available deployment (RouteIQ-8e37: thread messages
        #    so the capability-tier floor applies on the terminal fallback too).
        return self._fallback_deployment(model, request_kwargs, messages)

    # ------------------------------------------------------------------
    # Internal: Gov-ban chokepoint (RouteIQ-a073)
    # ------------------------------------------------------------------

    def _guard_selected(
        self,
        selected: Optional[Dict],
        model: str,
        request_kwargs: Optional[Dict] = None,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> Optional[Dict]:
        """Post-selection backstop: never return a gov-banned deployment.

        This is the SINGLE chokepoint mandated by RouteIQ-a073. Every routing
        path -- pipeline, ML, personalized, centroid, fallback, and any FUTURE
        strategy added later -- returns through here, so a gov-banned arm
        (Fable 5 family always-on, plus operator ``banned_models``) can never
        be selected even if a strategy's own candidate source forgot to filter.

        Defense-in-depth: each candidate source ALSO pre-filters (so banned arms
        are never scored), but this guard guarantees the invariant holds
        regardless. Fail-CLOSED: a banned selection is refused; we then attempt
        the (already-filtered) fallback, and if that is also banned/empty return
        ``None`` (a banned-only group must yield no deployment).

        RouteIQ-1216: ``request_kwargs`` + ``messages`` are threaded into the
        backstop ``_fallback_deployment`` probe so the per-request capability-tier
        FLOOR (RouteIQ-8e37) AND region/data-residency pre-filter (RouteIQ-60cc)
        apply on THIS path too. Without them the backstop probe ran context-free,
        so a banned-arm re-guard on a HARD reasoning request could fall back to a
        sub-tier (or out-of-region) arm the floor would have excluded everywhere
        else. Byte-stable when capability_routing / region_routing are off (both
        filters no-op and, with no context, fall back to the legacy candidate
        set).
        """
        if selected is None:
            return None

        from litellm_llmrouter.candidate_filter import is_gov_banned

        if not is_gov_banned(selected):
            return selected

        arm = (selected.get("litellm_params") or {}).get("model", "unknown")
        logger.error(
            "Gov-ban chokepoint REFUSED a selected deployment (arm=%s, model=%s); "
            "a routing strategy bypassed its pre-filter. Falling back to the "
            "filtered candidate set.",
            arm,
            model,
        )
        # _fallback_deployment is already gov-ban filtered AND (with the threaded
        # context) capability-floor / region filtered; re-guard it so a
        # double-banned group resolves to None rather than a banned arm.
        fallback = self._fallback_deployment(
            model, request_kwargs=request_kwargs, messages=messages
        )
        if fallback is not None and not is_gov_banned(fallback):
            return fallback
        return None

    # ------------------------------------------------------------------
    # Internal: Governance enforcement
    # ------------------------------------------------------------------

    async def _enforce_governance(
        self,
        model: str,
        request_kwargs: Optional[Dict] = None,
    ) -> None:
        """Enforce governance rules (budget, rate limit, model access) before routing.

        Extracts the API key from ``request_kwargs`` and calls the governance
        engine's ``enforce()`` method.  On violation, ``HTTPException`` is raised
        and propagated to the caller, short-circuiting routing entirely.

        When governance is not configured for the key (no key governance rules
        registered), the request passes through (fail-open).

        If governance enforcement itself fails (import error, unexpected
        exception), the request passes through and the failure is logged at
        debug level.

        Args:
            model: The requested model name.
            request_kwargs: Request keyword arguments containing api_key and metadata.
        """
        if not request_kwargs:
            return

        # Extract the API key from the various places LiteLLM may put it
        api_key = request_kwargs.get("api_key") or request_kwargs.get(
            "litellm_params", {}
        ).get("api_key")
        if not api_key:
            metadata = request_kwargs.get("metadata", {})
            if isinstance(metadata, dict):
                api_key = metadata.get("api_key")
        if not api_key:
            return

        try:
            from litellm_llmrouter.governance import get_governance_engine

            engine = get_governance_engine()

            # Fast path: if no governance rules are registered at all, skip
            if not engine._key_governance and not engine._workspaces:
                return

            ctx = await engine.enforce(api_key, model)

            # Store governance context in metadata for downstream use
            metadata = request_kwargs.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
                request_kwargs["metadata"] = metadata

            # Stamp the RESOLVED governance scope so the post-response spend
            # WRITER (router_decision_callback._derive_spend_scope) reuses the
            # EXACT scope the READ/enforce path uses -- workspace_id (ed7a) and
            # the RAW key_id (08dd).  Without key_id here the writer fell back to
            # LiteLLM's hashed user_api_key / user-id, which never matched the
            # read scope, leaving workspace + key budgets silently fail-open.
            metadata["_governance_ctx"] = {
                "workspace_id": ctx.workspace_id,
                "key_id": ctx.key_id,
                "effective_profile": ctx.effective_routing_profile,
            }

            # If governance specifies a routing profile, propagate it
            if ctx.effective_routing_profile:
                metadata["_routing_profile"] = ctx.effective_routing_profile

        except HTTPException:
            raise  # Budget/rate limit/model access denied — propagate
        except Exception as exc:
            logger.debug("Governance check skipped: %s", exc)

    # ------------------------------------------------------------------
    # Internal: Amplification guard
    # ------------------------------------------------------------------

    def _check_amplification_guard(
        self,
        request_kwargs: Optional[Dict],
    ) -> None:
        """
        Prevent routing amplification (LiteLLM bug #17329).

        If the same ``litellm_call_id`` / ``request_id`` is routed more than
        ``MAX_ROUTING_ATTEMPTS`` times, raise ``RuntimeError`` to short-circuit.
        """
        req_id = None
        if request_kwargs:
            req_id = request_kwargs.get("litellm_call_id") or request_kwargs.get(
                "request_id"
            )
        if not req_id:
            return

        attempts = self._routing_attempts.get(req_id, 0) + 1
        self._routing_attempts[req_id] = attempts

        if attempts > MAX_ROUTING_ATTEMPTS:
            logger.error(
                f"Routing amplification detected for request {req_id}: "
                f"{attempts} attempts (max {MAX_ROUTING_ATTEMPTS}). "
                f"Short-circuiting to prevent 38x amplification bug (#17329)."
            )
            raise RuntimeError(
                f"Routing amplification guard: {attempts} routing attempts "
                f"for request {req_id} exceeds limit of {MAX_ROUTING_ATTEMPTS}"
            )

        # Bounded cleanup to prevent memory leak
        if len(self._routing_attempts) > 10_000:
            self._routing_attempts.clear()

    # ------------------------------------------------------------------
    # Internal: Pipeline routing (A/B testing)
    # ------------------------------------------------------------------

    def _route_via_pipeline(
        self,
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        request_kwargs: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """
        Route through the ``RoutingPipeline`` for A/B testing support.

        Builds a ``RoutingContext`` and delegates to the pipeline, which
        handles strategy selection (single or A/B) and telemetry emission.
        """
        try:
            from litellm_llmrouter.strategy_registry import (
                RoutingContext,
            )

            # Extract identifiers from request_kwargs
            request_id = None
            user_id = None
            if request_kwargs:
                request_id = request_kwargs.get("request_id") or request_kwargs.get(
                    "litellm_call_id"
                )
                user_id = request_kwargs.get("user") or request_kwargs.get("user_id")
                metadata = request_kwargs.get("metadata", {})
                if isinstance(metadata, dict):
                    request_id = request_id or metadata.get("request_id")
                    user_id = user_id or metadata.get("user_id")

            context = RoutingContext(
                router=self._router,
                model=model,
                messages=messages,
                input=input,
                request_kwargs=request_kwargs,
                request_id=request_id,
                user_id=user_id,
            )

            pipeline = self._get_or_create_pipeline()
            result = pipeline.route(context)
            return result.deployment

        except ImportError:
            logger.debug("Pipeline routing not available, using direct LLMRouter")
            return None
        except Exception as e:
            logger.warning(f"Pipeline routing failed, falling back: {e}")
            return None

    # ------------------------------------------------------------------
    # Internal: Direct LLMRouter routing
    # ------------------------------------------------------------------

    def _route_via_llmrouter(
        self,
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        request_kwargs: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """
        Route directly using ``LLMRouterStrategyFamily``.

        Steps:
        1. Lazy-load the strategy family instance
        2. Extract query text from messages/input
        3. Get available model list from router
        4. Call ``strategy.route_with_observability()``
        5. Match the selected model name back to a deployment dict
        """
        if not self._strategy_name:
            return None

        strategy = self._get_or_create_strategy()
        if strategy is None:
            return None

        # Extract query
        query = self._extract_query(messages, input)

        # Get the full candidate picture (RouteIQ-60cc: pass request_kwargs so
        # the per-request region / data-residency pre-filter activates here;
        # RouteIQ-8e37: pass messages so the capability-tier floor resolves the
        # request difficulty on the ML path).
        candidates = self._get_candidates(model, request_kwargs, messages)
        model_list = candidates.model_list

        if not model_list:
            # RouteIQ-5007: a context-driven HARD residency / STRICT capability
            # filter that excluded EVERY arm of a NON-empty group is a fail-CLOSED
            # verdict -- do NOT repopulate to [model] and match against the
            # unfiltered healthy_deployments (that leaked an out-of-region arm).
            # Return None so the request fails closed / falls back safely.
            if candidates.filtered_empty:
                logger.warning(
                    "LLMRouter: per-request residency/capability filter excluded "
                    "all arms for model=%s -> failing closed (no out-of-region / "
                    "sub-tier leak).",
                    model,
                )
                return None
            # No per-request signal (legacy) OR a genuinely empty group: preserve
            # the byte-stable legacy [model] fallback.
            model_list = [model]

        # Route using ML strategy
        selected_model = strategy.route_with_observability(query, model_list)

        if not selected_model:
            logger.warning(
                f"LLMRouter strategy {self._strategy_name} returned no model, "
                f"falling back to first available deployment"
            )
            return None

        # Match the selected model to a deployment dict. When a per-request filter
        # context was applied (RouteIQ-5007), match against the FILTERED routable
        # set -- never the unfiltered healthy_deployments -- so the group-fallback
        # branch of _match_deployment cannot reach an out-of-region / sub-tier arm.
        # No context (legacy) -> the full healthy_deployments (byte-stable).
        match_pool = (
            candidates.routable
            if candidates.context_constrained
            else candidates.healthy_deployments
        )
        return self._match_deployment(selected_model, model, match_pool)

    # ------------------------------------------------------------------
    # Internal: Centroid routing (zero-config fallback)
    # ------------------------------------------------------------------

    def _route_via_centroid(
        self,
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
        request_kwargs: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """
        Route using centroid-based classification (zero-config intelligent routing).

        This is the fallback when no ML model is configured. It provides
        intelligent routing without requiring ML model training, using
        ~2ms centroid-based prompt classification.

        Args:
            model: The requested model group name.
            messages: Chat messages for classification.
            request_kwargs: Request keyword arguments (may contain routing_profile).

        Returns:
            Selected deployment dict, or None if centroid routing is unavailable.
        """
        if not CENTROID_ROUTING_ENABLED or not CENTROID_ROUTING_AVAILABLE:
            return None

        try:
            centroid = self._get_or_create_centroid_strategy(request_kwargs)
            if centroid is None:
                return None

            # Build a lightweight routing context for centroid strategy
            from litellm_llmrouter.strategy_registry import RoutingContext

            context = RoutingContext(
                router=self._router,
                model=model,
                messages=messages,
                input=None,
                request_kwargs=request_kwargs,
            )

            result = centroid.select_deployment(context)
            if result is not None:
                logger.debug(
                    "Centroid routing selected: %s (model=%s)",
                    result.get("litellm_params", {}).get("model", "unknown"),
                    model,
                )
            return result

        except ImportError:
            logger.debug("Centroid routing not available (missing dependencies)")
            return None
        except Exception as e:
            logger.warning(f"Centroid routing failed, falling back: {e}")
            return None

    # ------------------------------------------------------------------
    # Public: cost-cascade RETRY completion (RouteIQ-3ff5)
    # ------------------------------------------------------------------

    async def acompletion_with_cascade(
        self,
        model: str,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Any:
        """Completion entrypoint that climbs the cost-cascade ladder (RouteIQ-3ff5).

        Gated, default OFF -> byte-stable: when the cascade retry consumer is
        disabled (or the cost-cascade strategy is not registered) this issues a
        single ``router.acompletion`` exactly as before. When ON, it routes the
        cheapest rung via :class:`CostCascadeRoutingStrategy` (mode (a)), scores
        the response confidence, and RE-ISSUES pinned to the next rung up the
        ``routeiq_cascade`` ladder while confidence is below threshold.

        This is the LIVE consumer of the previously-dark mode-(a) ladder.
        """
        consumer = None
        cascade_strategy = None
        try:
            from litellm_llmrouter.cascade_retry import get_cascade_retry_consumer
            from litellm_llmrouter.strategy_registry import get_routing_registry

            consumer = get_cascade_retry_consumer()
            cascade_strategy = get_routing_registry().get("llmrouter-cost-cascade")
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("Cascade retry consumer unavailable: %s", e)

        # Disabled / not registered -> byte-stable single-shot completion.
        if consumer is None or cascade_strategy is None:
            return await self._router.acompletion(
                model=model, messages=messages, **kwargs
            )

        from litellm_llmrouter.strategy_registry import RoutingContext

        def _select_rung(rung_index: int):
            ctx = RoutingContext(
                router=self._router,
                model=model,
                messages=messages,
                input=None,
                request_kwargs={"cascade_rung": rung_index - 1}
                if rung_index > 0
                else {},
            )
            return cascade_strategy.select_deployment(ctx)

        async def _complete(deployment: dict, rung_index: int):
            rung_model = deployment.get("litellm_params", {}).get("model", model)
            return await self._router.acompletion(
                model=rung_model, messages=messages, **kwargs
            )

        return await consumer.complete_with_cascade(
            select_rung=_select_rung,
            complete=_complete,
        )

    # ------------------------------------------------------------------
    # Internal: KV-cache-/queue-aware engine routing (RouteIQ-08d6 / 6a89)
    # ------------------------------------------------------------------

    async def _route_via_kv_cache_aware(
        self,
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
        request_kwargs: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """Route to the least-loaded self-hosted engine arm (RouteIQ-08d6/6a89).

        Gated, default OFF -> byte-stable no-op: the registry only holds the
        ``llmrouter-kv-cache-aware`` strategy when
        ``engine_metrics.kv_aware_routing_enabled`` is True. When present, this
        scrapes each candidate arm's engine ``/metrics`` (the RouteIQ-6a89 live
        gauges) and picks the lowest queue-depth / KV-cache-pressure arm. Any
        error falls back (returns None) so routing continues down the normal
        path. This runs in the async select path so the live scrape is awaited.
        """
        try:
            from litellm_llmrouter.strategy_registry import (
                RoutingContext,
                get_routing_registry,
            )

            strategy = get_routing_registry().get("llmrouter-kv-cache-aware")
            if strategy is None:
                return None
            selector = getattr(strategy, "select_least_loaded", None)
            if not callable(selector):
                return None

            context = RoutingContext(
                router=self._router,
                model=model,
                messages=messages,
                input=None,
                request_kwargs=request_kwargs,
            )
            result = await selector(context)
            if result is not None:
                logger.debug(
                    "KV-cache-aware routing selected: %s (model=%s)",
                    result.get("litellm_params", {}).get("model", "unknown"),
                    model,
                )
            return result
        except Exception as e:
            logger.warning(f"KV-cache-aware routing failed, falling back: {e}")
            return None

    # ------------------------------------------------------------------
    # Internal: Personalized routing (preference-based re-ranking)
    # ------------------------------------------------------------------

    async def _route_via_personalized(
        self,
        model: str,
        request_kwargs: Optional[Dict] = None,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> Optional[Dict]:
        """Re-rank candidate deployments using learned user preferences.

        When personalized routing is enabled and a ``user_id`` is available
        in the request metadata, the personalized router re-ranks the
        candidate models by preference score. Cold-start users (no stored
        preferences) get quality-bias-only ranking.

        Args:
            model: The requested model group name.
            request_kwargs: Request keyword arguments (may contain user_id).
            messages: Chat messages, threaded so the capability-tier floor
                (RouteIQ-8e37) can resolve the request difficulty on this path.

        Returns:
            Selected deployment dict, or None if personalized routing is
            unavailable or the user has no preference data.
        """
        if not PERSONALIZED_ROUTING_AVAILABLE:
            return None

        try:
            p_router = get_personalized_router()
            if p_router is None:
                return None

            # Extract user_id from request metadata
            user_id = self._extract_user_id(request_kwargs)
            if not user_id:
                return None

            # Get candidate models from healthy deployments (RouteIQ-60cc:
            # pass request_kwargs so region / residency is honoured here too;
            # RouteIQ-8e37: pass messages so the capability-tier floor applies).
            candidates = self._get_candidates(model, request_kwargs, messages)
            model_list = candidates.model_list
            # RouteIQ-5007: the `< 2` guard already fails closed when a hard
            # residency / strict capability filter empties the routable set
            # (model_list == [] => return None, never repopulated). No leak on
            # this path. The match pool below is still narrowed to the FILTERED
            # routable set when a context applied, for defense-in-depth against a
            # cross-group partial match reaching an out-of-region / sub-tier arm.
            if not model_list or len(model_list) < 2:
                # No point re-ranking with 0 or 1 candidates
                return None

            # Get personalized ranking
            ranked = await p_router.rank_models(user_id, model_list)
            if not ranked:
                return None

            # Select the top-ranked model and map to deployment (filtered pool
            # when context-constrained -- RouteIQ-5007 -- else legacy full set).
            top_model = ranked[0][0]
            match_pool = (
                candidates.routable
                if candidates.context_constrained
                else candidates.healthy_deployments
            )
            result = self._match_deployment(top_model, model, match_pool)

            # Emit the routing.selection metric from the LIVE personalized path.
            # (get_top_model() is not on the live dispatch path, so the metric
            # must be recorded here where the selection actually happens.)
            _record_personalized_selection(top_model)

            if result is not None:
                logger.debug(
                    "Personalized routing selected: %s for user=%s (model=%s, score=%.3f)",
                    top_model,
                    user_id,
                    model,
                    ranked[0][1],
                )

            return result

        except Exception as e:
            logger.warning(f"Personalized routing failed, falling back: {e}")
            return None

    @staticmethod
    def _extract_user_id(request_kwargs: Optional[Dict]) -> Optional[str]:
        """Extract user_id from request kwargs metadata.

        Checks multiple fields where LiteLLM may store the user identifier:
        ``user``, ``user_id``, and ``metadata.user_id``.

        Args:
            request_kwargs: Request keyword arguments.

        Returns:
            User ID string, or None if not found.
        """
        if not request_kwargs:
            return None

        # Direct fields
        user_id = request_kwargs.get("user") or request_kwargs.get("user_id")
        if user_id:
            return str(user_id)

        # Nested in metadata
        metadata = request_kwargs.get("metadata", {})
        if isinstance(metadata, dict):
            user_id = metadata.get("user_id") or metadata.get("user")
            if user_id:
                return str(user_id)

        return None

    def _get_or_create_centroid_strategy(
        self,
        request_kwargs: Optional[Dict] = None,
    ) -> Any:
        """Lazy-load or return the centroid routing strategy.

        Resolves the routing profile from request metadata or env var
        and creates the centroid strategy with the appropriate profile.

        Args:
            request_kwargs: Request keyword arguments for profile resolution.

        Returns:
            CentroidRoutingStrategy instance, or None if unavailable.
        """
        if not CENTROID_ROUTING_AVAILABLE:
            return None

        # Resolve the routing profile for this request
        profile_str = _resolve_routing_profile(request_kwargs)

        # Map profile string to RoutingProfile enum
        try:
            profile = (
                RoutingProfile(profile_str) if profile_str else RoutingProfile.AUTO
            )
        except ValueError:
            logger.warning(
                "Unknown routing profile '%s', defaulting to 'auto'",
                profile_str,
            )
            profile = RoutingProfile.AUTO

        if not self._centroid_initialized:
            try:
                self._centroid_strategy = CentroidRoutingStrategy(profile=profile)
                self._centroid_initialized = True
            except Exception as e:
                logger.warning(f"Failed to create centroid strategy: {e}")
                self._centroid_initialized = True  # Don't retry
                return None
        elif self._centroid_strategy is not None:
            # Update profile per-request (it may change between requests)
            self._centroid_strategy._profile = profile

        return self._centroid_strategy

    # ------------------------------------------------------------------
    # Internal: Helper methods
    # ------------------------------------------------------------------

    def _get_candidates(
        self,
        model: str,
        request_kwargs: Optional[Dict] = None,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> "_CandidateSet":
        """Resolve the per-request candidate set WITH the fail-closed verdict.

        Unlike :meth:`_get_model_list` (the legacy 2-tuple shape), this returns
        the full picture every selection path needs to fail-closed correctly:

        - ``model_list``: the litellm model ids the strategy scores (from the
          FILTERED ``routable`` set).
        - ``healthy_deployments``: the FULL unfiltered alias (legacy match pool).
        - ``routable``: the gov-ban / cooldown / capability / region FILTERED
          subset of the model group -- the ONLY arms that may be selected.
        - ``group_matched``: the pre-filter members of the model group.
        - ``context``: the region/capability signal context, or ``None`` when
          there is no per-request signal (legacy byte-stable path).

        RouteIQ-5007: an EMPTY ``routable`` derived from a NON-empty
        ``group_matched`` while a ``context`` is present is a HARD fail-closed
        verdict (a hard data-residency / strict capability constraint excluded
        every arm). Callers MUST treat that as "no selection" and NOT repopulate
        the candidate set with the unfiltered ``[model]`` / ``healthy_deployments``
        -- doing so leaks an out-of-region (or sub-tier) arm. See
        :attr:`_CandidateSet.filtered_empty`.
        """
        healthy_deployments = getattr(
            self._router, "healthy_deployments", self._router.model_list
        )

        # RouteIQ-99e8 (cooldown) + RouteIQ-badb (gov-ban): the custom-strategy
        # path bypasses LiteLLM's healthy-deployment pipeline, so the static
        # ``healthy_deployments`` alias is NOT cooldown-aware and never applies a
        # gov-ban. Filter cooled-down / gov-banned arms out of the group-matched
        # subset BEFORE returning the candidate set the strategy scores.
        from litellm_llmrouter.candidate_filter import filter_routable_candidates

        # RouteIQ-60cc: build a minimal region-signal context from request_kwargs
        # so a HARD data-residency request never leaks out-of-region on the ML
        # path. RouteIQ-8e37: ``messages`` are threaded too so the capability-tier
        # floor can resolve the request difficulty here. ``context=None`` (no
        # request_kwargs AND no messages) keeps the legacy byte-stable behaviour.
        context = self._region_context(model, request_kwargs, messages)

        group_matched = [d for d in healthy_deployments if d.get("model_name") == model]
        routable = filter_routable_candidates(
            self._router, group_matched, context=context
        )

        model_list: List[str] = []
        for deployment in routable:
            litellm_model = deployment.get("litellm_params", {}).get("model", "")
            if litellm_model:
                model_list.append(litellm_model)

        return _CandidateSet(
            model_list=model_list,
            healthy_deployments=healthy_deployments,
            routable=routable,
            group_matched=group_matched,
            context=context,
        )

    def _get_model_list(
        self,
        model: str,
        request_kwargs: Optional[Dict] = None,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> tuple:
        """
        Get available deployments from the Router.

        Prefers ``healthy_deployments`` over ``model_list`` for freshness.

        ``request_kwargs`` / ``messages`` are OPTIONAL: when supplied, a minimal
        :class:`RoutingContext` is built so the per-request region /
        data-residency pre-filter (RouteIQ-60cc) AND the capability-tier FLOOR
        (RouteIQ-8e37) activate on the ML (``LLMRouterStrategyFamily``) path.
        ``messages`` feed the capability floor's difficulty resolution. When both
        are absent both filters no-op (byte-stable for callers with no context).

        Legacy 2-tuple shape (kept for backwards compatibility); selection paths
        that must honour the fail-closed verdict use :meth:`_get_candidates`.

        Returns:
            Tuple of (model_name_list, healthy_deployments_list)
        """
        candidates = self._get_candidates(model, request_kwargs, messages)
        return candidates.model_list, candidates.healthy_deployments

    def _region_context(
        self,
        model: str,
        request_kwargs: Optional[Dict],
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> Optional[Any]:
        """Build a minimal context for the per-request candidate pre-filters.

        RouteIQ-60cc: the ML (``LLMRouterStrategyFamily``) path operates on a
        bare model name, not a ``RoutingContext``, so the region / data-residency
        signal in ``request_kwargs`` (header + metadata) was previously dropped.
        This wraps ``request_kwargs`` in a lightweight ``RoutingContext`` whose
        ``request_kwargs`` / ``metadata`` are exactly the sources
        ``candidate_filter._ctx_sources`` scans.

        RouteIQ-8e37: ``messages`` are also threaded so the capability-tier FLOOR
        filter can resolve the request DIFFICULTY (reasoning markers + warm
        centroid tier) on the same ML / fallback paths — without the prompt the
        floor could not tell a simple request from a hard reasoning one.

        Returns ``None`` when there is NO signal at all (no ``request_kwargs`` AND
        no ``messages``) so both filters no-op and the legacy candidate set is
        byte-stable.
        """
        if not request_kwargs and not messages:
            return None
        try:
            from litellm_llmrouter.strategy_registry import RoutingContext

            metadata = (request_kwargs or {}).get("metadata")
            return RoutingContext(
                router=self._router,
                model=model,
                messages=messages,
                request_kwargs=request_kwargs,
                metadata=metadata if isinstance(metadata, dict) else {},
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("filter context build failed (filters no-op): %s", exc)
            return None

    @staticmethod
    def _extract_query(
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
    ) -> str:
        """
        Extract query text from messages or input for ML routing.

        Handles both simple string messages and multi-modal message formats.
        """
        if messages:
            parts: List[str] = []
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    # Handle multi-modal messages
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            parts.append(item.get("text", ""))
            return " ".join(parts).strip()

        if input is not None:
            if isinstance(input, str):
                return input
            return " ".join(str(i) for i in input)

        return ""

    @staticmethod
    def _match_deployment(
        selected_model: str,
        model: str,
        healthy_deployments: List[Dict],
    ) -> Optional[Dict]:
        """
        Map a model name from LLMRouter back to a Router deployment dict.

        First tries exact match on ``litellm_params.model``, then falls
        back to partial/substring matching, and finally to the first
        deployment matching the requested model group.
        """
        # Exact match
        for deployment in healthy_deployments:
            litellm_model = deployment.get("litellm_params", {}).get("model", "")
            if litellm_model == selected_model:
                return deployment

        # Partial match: selected_model is a substring or vice versa
        for deployment in healthy_deployments:
            litellm_model = deployment.get("litellm_params", {}).get("model", "")
            if selected_model in litellm_model or litellm_model in selected_model:
                return deployment

        # Fallback: first deployment for the requested model group
        for deployment in healthy_deployments:
            if deployment.get("model_name") == model:
                return deployment

        return None

    def _get_or_create_strategy(self) -> Any:
        """Lazy-load the ``LLMRouterStrategyFamily`` instance."""
        if self._strategy_instance is None and self._strategy_name:
            try:
                from litellm_llmrouter.strategies import LLMRouterStrategyFamily

                self._strategy_instance = LLMRouterStrategyFamily(
                    strategy_name=self._strategy_name,
                    **self._strategy_args,
                )
            except Exception as e:
                logger.error(f"Failed to create LLMRouterStrategyFamily: {e}")
                return None
        return self._strategy_instance

    def _get_or_create_pipeline(self) -> Any:
        """Lazy-load the ``RoutingPipeline`` singleton."""
        if not self._pipeline_initialized:
            try:
                from litellm_llmrouter.strategy_registry import get_routing_pipeline

                self._pipeline = get_routing_pipeline()
                self._pipeline_initialized = True
            except ImportError:
                logger.debug("RoutingPipeline not available")
                self._pipeline_initialized = True  # Don't retry
        return self._pipeline

    def _fallback_deployment(
        self,
        model: str,
        request_kwargs: Optional[Dict] = None,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> Optional[Dict]:
        """Return first routable deployment for the given model group.

        RouteIQ-99e8 / RouteIQ-badb: never fall back to a cooled-down or
        gov-banned arm (the fallback path also bypasses LiteLLM's pipeline).

        RouteIQ-60cc: this terminal fallback is the LAST deployment-selection
        seam, so it MUST honour per-request region/data-residency too — a
        hard-residency request that reaches the fallback must not leak
        out-of-region. RouteIQ-8e37: the capability-tier FLOOR is threaded here
        too — a HARD reasoning request must never fall back to a sub-tier model
        (in STRICT mode the set empties; in SOFT mode it degrades to the full
        set). ``request_kwargs`` + ``messages`` carry the same signal the other
        paths use; both absent => filters no-op (byte-stable).
        """
        from litellm_llmrouter.candidate_filter import filter_routable_candidates

        healthy_deployments = getattr(
            self._router, "healthy_deployments", self._router.model_list
        )
        group_matched = [d for d in healthy_deployments if d.get("model_name") == model]
        context = self._region_context(model, request_kwargs, messages)
        for deployment in filter_routable_candidates(
            self._router, group_matched, context=context
        ):
            return deployment
        return None


# ======================================================================
# Factory and installation functions
# ======================================================================


def create_routeiq_strategy(
    router: Any,
    strategy_name: Optional[str] = None,
) -> RouteIQRoutingStrategy:
    """
    Create a configured ``RouteIQRoutingStrategy`` instance.

    Args:
        router: LiteLLM Router instance
        strategy_name: Optional ML strategy name (e.g., "llmrouter-knn")

    Returns:
        Configured strategy instance
    """
    return RouteIQRoutingStrategy(
        router_instance=router,
        strategy_name=strategy_name,
    )


def install_routeiq_strategy(
    router: Any,
    strategy_name: Optional[str] = None,
) -> RouteIQRoutingStrategy:
    """
    Create and install a ``RouteIQRoutingStrategy`` on the given Router.

    Calls ``router.set_custom_routing_strategy()`` to wire the strategy
    into the Router's deployment selection path.

    Optionally warms up the centroid classifier for fast first-request latency.
    Controlled by ``ROUTEIQ_CENTROID_WARMUP=true`` env var.

    Args:
        router: LiteLLM Router instance
        strategy_name: Optional ML strategy name (e.g., "llmrouter-knn")

    Returns:
        The installed strategy instance (useful for testing/inspection)
    """
    strategy = create_routeiq_strategy(router, strategy_name)

    if hasattr(router, "set_custom_routing_strategy"):
        router.set_custom_routing_strategy(strategy)
        logger.info(
            f"Installed RouteIQ custom routing strategy "
            f"(strategy={strategy_name or 'default'}, "
            f"pipeline={'enabled' if USE_PIPELINE_ROUTING else 'disabled'}, "
            f"centroid={'enabled' if CENTROID_ROUTING_ENABLED and CENTROID_ROUTING_AVAILABLE else 'disabled'})"
        )
    else:
        logger.warning(
            "Router does not support set_custom_routing_strategy(). "
            "Strategy created but NOT installed — is your LiteLLM version compatible?"
        )

    # Optionally warmup centroid classifier
    try:
        from litellm_llmrouter.settings import get_settings as _gs_warmup

        _centroid_warmup = _gs_warmup().routing.centroid_warmup
    except Exception:
        _centroid_warmup = (
            os.getenv("ROUTEIQ_CENTROID_WARMUP", "false").lower() == "true"
        )
    if _centroid_warmup and CENTROID_ROUTING_AVAILABLE and CENTROID_ROUTING_ENABLED:
        try:
            warmup_centroid_classifier()
            logger.info("Centroid classifier warmed up during strategy installation")
        except Exception as e:
            logger.warning(f"Centroid classifier warmup failed: {e}")

    return strategy


def register_centroid_strategy() -> bool:
    """Register the centroid strategy in the routing registry.

    Makes the centroid strategy available as ``"llmrouter-nadirclaw-centroid"`` in the
    routing registry for direct use or A/B testing.

    Auto-registers if ``ROUTEIQ_CENTROID_ROUTING=true`` (default).

    Returns:
        True if registration succeeded, False otherwise.
    """
    if not CENTROID_ROUTING_AVAILABLE:
        logger.debug("Centroid routing not available (missing imports)")
        return False

    if not CENTROID_ROUTING_ENABLED:
        logger.debug("Centroid routing disabled via ROUTEIQ_CENTROID_ROUTING=false")
        return False

    try:
        from litellm_llmrouter.strategy_registry import get_routing_registry

        registry = get_routing_registry()
        strategy = get_centroid_strategy()
        registry.register("llmrouter-nadirclaw-centroid", strategy)
        logger.info(
            "Registered centroid strategy as 'llmrouter-nadirclaw-centroid' in routing registry"
        )
        return True
    except Exception as e:
        logger.warning(f"Failed to register centroid strategy: {e}")
        return False
