"""
Kumaraswamy-Thompson Routing Strategy
======================================

A net-new RouteIQ online-bandit routing strategy (the differentiation, not a
VSR port). It models each ``(task_bucket, arm)`` pair as a Bernoulli reward with
a conjugate Beta posterior, and samples via a **Kumaraswamy** closed-form
quantile instead of a Beta rejection sampler — one ``random()`` + a couple of
``pow`` calls, no numpy, branch-free on the hot path.

Design references (read fully before editing):
- ``docs/architecture/aws-rearchitecture/20-kumaraswamy-thompson-router.md``
  (the bandit math + reward shaping + cold start)
- ``docs/architecture/aws-rearchitecture/40-pluggable-routing-and-mlops.md``
  (the strategy-agnostic adapter/MLOps framework this strategy *consumes*)

Key facts:
- **Arm key = ``litellm_params.model``** (e.g. ``bedrock/...``). The arm set is
  dynamic: providers and circuit breakers add/remove arms; the bandit tolerates
  appearing/vanishing arms because each ``(bucket, model)`` posterior is keyed
  independently and the select loop only iterates *live* candidates.
- **Quantile is log-space stabilized.** The naive
  ``(1-(1-u)^(1/b))^(1/a)`` form returns exactly ``0.0`` on the small-``u`` /
  small-``b`` exploration tail (e.g. ``u=1e-18, b=0.05``), silently killing an
  arm's exploration. The default ``kumaraswamy_quantile`` uses
  ``-math.expm1(math.log1p(-u)/b)`` which stays positive. ``_q_naive`` is kept
  ONLY as a reference for the divergence unit test.
- **In-memory backend is the default** so the core bandit + every unit test run
  with NO external deps. Redis (hot) / Aurora (durable) backends are wired
  behind flags for the live substrate (P1/P2), default off.
- Determinism: the RNG is threaded as a ``random.Random`` *object*, never the
  global ``random`` module — seeded for tests/replay, no global-state pollution.

Fable 5 is GOV-BANNED as a routable arm and is never added to any default or
test arm set in this module.
"""

from __future__ import annotations

import logging
import math
import random
import threading
import time
from dataclasses import asdict, dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

from litellm_llmrouter.strategy_registry import RoutingContext, RoutingStrategy

if TYPE_CHECKING:
    from litellm_llmrouter.adapters.contract import (
        AdapterManifest,
        ArtifactRef,
        RoutingFeedback,
    )

logger = logging.getLogger(__name__)

# Static per-model quality biases used for cold-start warm-starting. Imported
# lazily from personalized_routing at call time so the bandit has no hard import
# dependency (and degrades to the neutral 0.5 prior if unavailable).
_DEFAULT_QUALITY_BIAS_FALLBACK = 0.5

STRATEGY_NAME = "routeiq-kumaraswamy-thompson"
STRATEGY_VERSION = "v1"


# ---------------------------------------------------------------------------
# Module-level pure functions (no numpy, stdlib ``math`` only)
# ---------------------------------------------------------------------------


def kumaraswamy_quantile(u: float, a: float, b: float) -> float:
    """Closed-form inverse-CDF of ``Kumaraswamy(a, b)``, log-space stabilized.

    Inverts ``F(x) = 1 - (1 - x^a)^b``. The naive algebraic form
    ``(1 - (1 - u)^(1/b))^(1/a)`` underflows to exactly ``0.0`` on the
    small-``u`` / small-``b`` exploration tail; the log-space form
    ``inner = -expm1(log1p(-u)/b)`` stays strictly positive there.

    Args:
        u: Uniform draw in ``(0, 1)`` (clamped into a safe interior band).
        a: First shape parameter, ``a > 0``.
        b: Second shape parameter, ``b > 0``.

    Returns:
        A sample in ``(0, 1)``.
    """
    # Clamp u into the open interval; degenerate posteriors must never yield
    # exactly 0 or 1 (which would zero out / saturate the argmax comparison).
    if not (0.0 < u < 1.0):
        u = min(max(u, 1e-12), 1.0 - 1e-12)
    a = a if a > 0.0 else 1e-9
    b = b if b > 0.0 else 1e-9
    # inner = 1 - (1 - u)^(1/b), computed in log-space to avoid underflow.
    inner = -math.expm1(math.log1p(-u) / b)
    # Guard against fp drift pushing inner just outside [0, 1].
    inner = min(max(inner, 0.0), 1.0)
    return float(inner ** (1.0 / a))


def kumaraswamy_cdf(x: float, a: float, b: float) -> float:
    """Stable CDF ``F(x) = 1 - (1 - x^a)^b`` (used by tests/sanity, not hot path).

    Computed as ``-expm1(b * log1p(-(x^a)))`` to match the log-space quantile so
    round-trip tests measure the quantile's accuracy, not the CDF's own
    cancellation.

    Args:
        x: Point in ``[0, 1]``.
        a: First shape parameter, ``a > 0``.
        b: Second shape parameter, ``b > 0``.

    Returns:
        ``F(x; a, b)`` in ``[0, 1]``.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    a = a if a > 0.0 else 1e-9
    b = b if b > 0.0 else 1e-9
    return -math.expm1(b * math.log1p(-(x**a)))


def _q_naive(u: float, a: float, b: float) -> float:
    """Naive (unstable) reference quantile — for the divergence test ONLY.

    Returns exactly ``0.0`` on the small-``u`` / small-``b`` tail, which is the
    bug the log-space :func:`kumaraswamy_quantile` fixes. Do not use on the hot
    path.
    """
    return float((1.0 - (1.0 - u) ** (1.0 / b)) ** (1.0 / a))


def sample_kumaraswamy(a: float, b: float, rng: random.Random) -> float:
    """Inverse-transform sample from ``Kumaraswamy(a, b)`` using ``rng``.

    Args:
        a: First shape parameter.
        b: Second shape parameter.
        rng: A ``random.Random`` instance (threaded for determinism).

    Returns:
        A sample in ``(0, 1)``.
    """
    return kumaraswamy_quantile(rng.random(), a, b)


# ---------------------------------------------------------------------------
# Moment-fit Beta(alpha,beta) -> Kumaraswamy(a,b)  (doc-20 §3.1 option-2)
# ---------------------------------------------------------------------------
#
# The §3.1 option-1 ``a=alpha, b=beta`` shortcut (``Posterior.shape()``) does
# NOT preserve the posterior mean: ``Kumaraswamy(a,b)`` is intrinsically
# right-skewed for ``a > 1``, so a symmetric peaked Beta gets its mass pushed
# toward 1 (Beta(51,51) mean 0.5 -> Kumaraswamy mean 0.9155). That distortion
# can INVERT the Thompson exploit decision (an over-concentrated symmetric arm
# beats a genuinely-better but less-concentrated arm).
#
# Option-2 maps Beta(alpha,beta) -> Kumaraswamy(a,b) by matching the mean
# ``alpha/(alpha+beta)`` and variance ``alpha*beta/((alpha+beta)^2*(alpha+beta+1))``
# to the Kumaraswamy moments ``E[X^n] = b*B(1+n/a, b)``. There is no closed
# form, so this is a 5-iteration damped log-space Newton solve. The log-space
# coordinates + log-variance residual + damping + a NON-shortcut warm start are
# each required: a naive raw-coordinate Newton seeded from the shortcut
# DIVERGES on the symmetric peaked posteriors a bandit produces most (it
# collapses ``a -> ~1e-3`` into a degenerate corner). Pure stdlib ``math`` only
# (no numpy), consistent with the module's no-numpy constraint.


def _lbeta(x: float, y: float) -> float:
    """``ln B(x, y)`` via stdlib ``lgamma`` (no numpy)."""
    return math.lgamma(x) + math.lgamma(y) - math.lgamma(x + y)


def _kuma_raw_moment(n: int, a: float, b: float) -> float:
    """``E[X^n] = b * B(1 + n/a, b)`` for ``Kumaraswamy(a, b)`` (log-space)."""
    return math.exp(math.log(b) + _lbeta(1.0 + n / a, b))


def kumaraswamy_mean_var(a: float, b: float) -> Tuple[float, float]:
    """Exact mean and variance of ``Kumaraswamy(a, b)``.

    ``mean = E[X] = b*B(1+1/a, b)``;
    ``var  = E[X^2] - E[X]^2`` (floored at ``1e-15`` for log-stability).
    """
    m1 = _kuma_raw_moment(1, a, b)
    m2 = _kuma_raw_moment(2, a, b)
    return m1, max(m2 - m1 * m1, 1e-15)


def _solve_b_for_mean(a: float, target_mean: float, iters: int = 30) -> float:
    """1-D log-bisection for ``b`` matching ``mean_K(a, b) = target_mean``.

    ``mean_K`` is strictly monotone-decreasing in ``b`` at fixed ``a``, so a
    log-space bisection over ``[1e-6, 1e12]`` converges robustly. Used to build
    a warm start that lands inside Newton's basin of attraction.
    """
    lo, hi = 1e-6, 1e12
    for _ in range(iters):
        mid = math.sqrt(lo * hi)
        m, _ = kumaraswamy_mean_var(a, mid)
        lo, hi = (mid, hi) if m > target_mean else (lo, mid)
    return math.sqrt(lo * hi)


def fit_kumaraswamy_moments(alpha: float, beta: float) -> Tuple[float, float]:
    """Map ``Beta(alpha, beta) -> Kumaraswamy(a, b)`` (doc-20 §3.1 option-2).

    Matches the mean ``alpha/(alpha+beta)`` and variance to the Kumaraswamy
    moments via a 5-iteration damped log-space Newton solve with a 1-D
    log-bisection warm start. Pure deterministic function of ``(alpha, beta)``
    (no RNG, no global state). Returns valid params (``a > 0``, ``b > 0``) for
    all inputs.

    Numerical structure (each element is required, validated against a grid):
    - Exact special-case short-circuits (``Kuma(1,b)=Beta(1,b)``,
      ``Kuma(a,1)=Beta(a,1)``) -> byte-stable corners.
    - Log-space coordinates ``(ln a, ln b)`` -> params positive by construction.
    - Log-variance residual -> conditions the 2x2 Jacobian (peaked posteriors
      have tiny variances that otherwise make the system ill-conditioned).
    - Damped step (``|delta| <= 1.5``) + box-guard -> prevents the single
      overshoot into the degenerate ``a -> 0`` corner.
    - Singular-Jacobian guard (``|det| < 1e-14`` -> break, keep best iterate).
    - Best-iterate tracking on the MEAN (mean-error first, variance-error tie-
      break). Kumaraswamy has a *minimum achievable variance* for a given mean;
      on high-evidence near-symmetric posteriors (e.g. Beta(185,190)) the target
      variance is below that floor, so Newton — chasing an infeasible variance —
      walks the mean off the warm start (which already nails the mean). Keeping
      the smallest-mean-error iterate is the doc's stated degradation contract:
      the right mean with a slightly-too-large variance (more exploration),
      NEVER the shortcut's wrong mean. Without it the fit returns a mean error of
      ~0.14 in that regime — the very distortion this fixes.
    """
    # Exact special cases (byte-stable corners). alpha==beta==1 falls out here.
    if abs(alpha - 1.0) < 1e-12:
        return (1.0, beta)
    if abs(beta - 1.0) < 1e-12:
        return (alpha, 1.0)

    s = alpha + beta
    t_mean = alpha / s
    t_var = (alpha * beta) / (s * s * (s + 1.0))

    # Warm start: a0 = sqrt(alpha) (NOT the shortcut), b0 from the mean eqn.
    a = max(0.1, math.sqrt(alpha))
    b = _solve_b_for_mean(a, t_mean)
    la, lb = math.log(a), math.log(b)

    h = 1e-5
    log_t_var = math.log(t_var)
    # Track the best iterate by (mean error, variance error) — mean is the
    # load-bearing quantity, so it dominates the comparison.
    best_la, best_lb = la, lb
    best_m, best_v = kumaraswamy_mean_var(a, b)
    best_key = (abs(best_m - t_mean), abs(math.log(best_v) - log_t_var))
    for _ in range(5):
        a, b = math.exp(la), math.exp(lb)
        m, v = kumaraswamy_mean_var(a, b)
        key = (abs(m - t_mean), abs(math.log(v) - log_t_var))
        if key < best_key:
            best_key = key
            best_la, best_lb = la, lb
        r0 = m - t_mean
        r1 = math.log(v) - log_t_var
        m_a, v_a = kumaraswamy_mean_var(math.exp(la + h), b)
        m_b, v_b = kumaraswamy_mean_var(a, math.exp(lb + h))
        j00 = (m_a - m) / h
        j01 = (m_b - m) / h
        j10 = (math.log(v_a) - math.log(v)) / h
        j11 = (math.log(v_b) - math.log(v)) / h
        det = j00 * j11 - j01 * j10
        if abs(det) < 1e-14:
            break  # singular -> keep best iterate seen
        d_la = (j11 * r0 - j01 * r1) / det
        d_lb = (-j10 * r0 + j00 * r1) / det
        d_la = max(-1.5, min(1.5, d_la))  # damping
        d_lb = max(-1.5, min(1.5, d_lb))
        la -= d_la
        lb -= d_lb
        la = max(min(la, 20.0), -8.0)  # box-guard
        lb = max(min(lb, 25.0), -8.0)
    # Final post-loop iterate may also be the best — check it.
    a, b = math.exp(la), math.exp(lb)
    m, v = kumaraswamy_mean_var(a, b)
    if (abs(m - t_mean), abs(math.log(v) - log_t_var)) < best_key:
        best_la, best_lb = la, lb
    return (math.exp(best_la), math.exp(best_lb))


def strength_bucket(s: float, base: float = 2.0) -> int:
    """Log-spaced evidence bucket ``floor(log_base(max(s, 1e-9)))``.

    Boundaries land at ``s in {2, 4, 8, 16, ...}`` for ``base=2``: the moment
    fit is refreshed once per doubling of evidence -> ``O(log(total evidence))``
    fits over an arm's lifetime, amortized to ~0 on the hot path.
    """
    return int(math.floor(math.log(max(s, 1e-9), base)))


# ---------------------------------------------------------------------------
# Posterior state + backend Protocol (in-memory DEFAULT; tests need no DB)
# ---------------------------------------------------------------------------


@dataclass
class Posterior:
    """Conjugate Beta posterior counts for one ``(bucket, arm)`` pair.

    ``alpha``/``beta`` are the pseudo-counts; ``mean()`` is the posterior mean
    reward estimate. :meth:`shape` returns the Kumaraswamy shape ``(a, b)``:
    the §3.1 option-1 ``(a, b) = (alpha, beta)`` shortcut by default, or the
    §3.1 option-2 moment-fit when ``moment_fit=True`` (cached on the three
    ``_fit_*`` fields below, refreshed only when the evidence ``alpha+beta``
    crosses a log-spaced bucket boundary).
    """

    alpha: float = 1.0
    beta: float = 1.0
    last_update: float = 0.0
    # Cached moment-fit (RouteIQ-f9e9). 0.0 a => unset; refreshed when the
    # strength bucket changes. Per-instance -> reset with the backend, no
    # module-level singleton.
    _fit_a: float = 0.0
    _fit_b: float = 0.0
    _fit_bucket: int = -(2**31)

    def mean(self) -> float:
        """Posterior mean reward ``alpha / (alpha + beta)``."""
        total = self.alpha + self.beta
        return self.alpha / total if total > 0 else 0.5

    def shape(self, *, moment_fit: bool = False) -> Tuple[float, float]:
        """Kumaraswamy shape ``(a, b)``.

        ``moment_fit=False`` (default): the §3.1 option-1 ``a=alpha, b=beta``
        shortcut -- unchanged, byte-stable. ``moment_fit=True``: the §3.1
        option-2 Newton moment-fit (cached; recomputed only when the strength
        bucket ``floor(log2(alpha+beta))`` differs from the cached fit).
        """
        if not moment_fit:
            return (self.alpha, self.beta)  # option-1 shortcut (default)
        bucket = strength_bucket(self.alpha + self.beta)
        if self._fit_a <= 0.0 or bucket != self._fit_bucket:
            self._fit_a, self._fit_b = fit_kumaraswamy_moments(self.alpha, self.beta)
            self._fit_bucket = bucket
        return (self._fit_a, self._fit_b)

    def strength(self) -> float:
        """Total pseudo-count ``alpha + beta`` (concentration / evidence)."""
        return self.alpha + self.beta


@runtime_checkable
class PosteriorBackend(Protocol):
    """Storage contract for ``(bucket, model) -> Posterior`` state."""

    def get(self, bucket: str, model: str) -> Posterior: ...

    def update(self, bucket: str, model: str, reward: float) -> None: ...

    def decay(
        self,
        bucket: str,
        model: str,
        gamma: float,
        days: float,
        prior: Tuple[float, float],
    ) -> None: ...

    def snapshot(self) -> Dict[str, Any]: ...

    def hydrate(self, data: Dict[str, Any]) -> None: ...


class InMemoryPosteriorBackend:
    """The DEFAULT backend: a plain in-process dict, synchronous, no deps.

    Every unit test and the offline backtest use this. Cold-start priors are
    supplied by the strategy via :meth:`ensure` so the backend stays storage-only
    and prior-policy lives in one place (the strategy).
    """

    def __init__(self) -> None:
        self._store: Dict[Tuple[str, str], Posterior] = {}
        self._lock = threading.RLock()

    def ensure(self, bucket: str, model: str, prior: Tuple[float, float]) -> Posterior:
        """Get-or-create the posterior for ``(bucket, model)`` with ``prior``."""
        key = (bucket, model)
        with self._lock:
            post = self._store.get(key)
            if post is None:
                post = Posterior(alpha=prior[0], beta=prior[1])
                self._store[key] = post
            return post

    def get(self, bucket: str, model: str) -> Posterior:
        """Get the posterior for ``(bucket, model)``, defaulting to ``Beta(1, 1)``."""
        return self.ensure(bucket, model, (1.0, 1.0))

    def update(self, bucket: str, model: str, reward: float) -> None:
        """Apply the conjugate update ``alpha += r; beta += (1 - r)``."""
        r = min(max(reward, 0.0), 1.0)
        key = (bucket, model)
        with self._lock:
            post = self._store.get(key)
            if post is None:
                post = Posterior()
                self._store[key] = post
            post.alpha += r
            post.beta += 1.0 - r
            post.last_update = time.time()

    def decay(
        self,
        bucket: str,
        model: str,
        gamma: float,
        days: float,
        prior: Tuple[float, float],
    ) -> None:
        """Decay the posterior toward its prior pseudo-counts by ``gamma^days``.

        Decaying toward the prior (NOT toward zero) keeps the posterior *mean*
        stable while re-inflating variance — which correctly re-opens
        exploration for stale arms.
        """
        if days <= 0 or gamma >= 1.0:
            return
        factor = gamma**days
        a0, b0 = prior
        key = (bucket, model)
        with self._lock:
            post = self._store.get(key)
            if post is None:
                return
            post.alpha = a0 + (post.alpha - a0) * factor
            post.beta = b0 + (post.beta - b0) * factor

    def snapshot(self) -> Dict[str, Any]:
        """Serialize all posteriors to a JSON-friendly dict (warm-start artifact)."""
        with self._lock:
            return {
                "posteriors": [
                    {
                        "bucket": bucket,
                        "model": model,
                        "alpha": post.alpha,
                        "beta": post.beta,
                        "last_update": post.last_update,
                    }
                    for (bucket, model), post in self._store.items()
                ]
            }

    def hydrate(self, data: Dict[str, Any]) -> None:
        """Load posteriors from a :meth:`snapshot` dict (replaces current state)."""
        new_store: Dict[Tuple[str, str], Posterior] = {}
        for row in data.get("posteriors", []):
            try:
                key = (str(row["bucket"]), str(row["model"]))
                new_store[key] = Posterior(
                    alpha=float(row["alpha"]),
                    beta=float(row["beta"]),
                    last_update=float(row.get("last_update", 0.0)),
                )
            except (KeyError, TypeError, ValueError):
                continue
        with self._lock:
            self._store = new_store


# ---------------------------------------------------------------------------
# The strategy
# ---------------------------------------------------------------------------


class KumaraswamyThompsonStrategy(RoutingStrategy):
    """Online Thompson-sampling bandit with Kumaraswamy closed-form sampling.

    Rides the existing routing pipeline as a registered ``RoutingStrategy`` —
    ZERO edits to the LiteLLM mount (``custom_routing_strategy.py``). It exposes
    the MLOps hooks (``update``, ``export_artifact``, ``load_artifact``,
    ``declare_capabilities``) that the strategy-agnostic loop *calls*; it does
    not contain any loop logic itself.
    """

    def __init__(
        self,
        backend: Optional[PosteriorBackend] = None,
        *,
        seed: Optional[int] = None,
        w_quality: float = 0.5,
        w_cost: float = 0.4,
        w_latency: float = 0.1,
        decay_gamma: float = 0.99,
        cold_start_kappa: float = 5.0,
        moment_fit: bool = False,
        settings: Any = None,
        bucket_log_capacity: int = 4096,
        **kwargs: Any,
    ) -> None:
        self._backend: PosteriorBackend = backend or InMemoryPosteriorBackend()
        # RNG threaded as an object — never the global ``random`` module.
        self._rng = random.Random(seed)
        self._seed = seed
        # RouteIQ-f9e9: use the moment-fit shape on the scoring path when on.
        self._moment_fit = bool(moment_fit)

        # Reward weights, sum-normalized.
        weights = (max(0.0, w_quality), max(0.0, w_cost), max(0.0, w_latency))
        total = sum(weights) or 1.0
        self._w_quality, self._w_cost, self._w_latency = (w / total for w in weights)

        self._decay_gamma = decay_gamma
        self._cold_start_kappa = max(0.0, cold_start_kappa)

        # request_id -> bucket recovery for the offline feedback path. Bounded;
        # FIFO eviction prevents unbounded growth on the hot path.
        self._bucket_log: Dict[str, str] = {}
        self._bucket_log_order: list[str] = []
        self._bucket_log_capacity = max(1, bucket_log_capacity)

        self._lock = threading.RLock()

    @property
    def name(self) -> str:
        return STRATEGY_NAME

    @property
    def version(self) -> Optional[str]:
        return STRATEGY_VERSION

    # ------------------------------------------------------------------
    # Cold start prior
    # ------------------------------------------------------------------

    def _cold_start_prior(self, model: str) -> Tuple[float, float]:
        """Warm-start ``(alpha0, beta0)`` from the static quality table.

        ``alpha0 = 1 + kappa*q0``, ``beta0 = 1 + kappa*(1 - q0)`` where ``q0`` is
        the model's ``_DEFAULT_QUALITY_BIASES`` value. ``kappa = 0`` reduces to
        the uniform ``Beta(1, 1)`` prior.

        A model with NO entry in the quality table has no *known* prior quality,
        so it gets the neutral uniform ``Beta(1, 1)`` prior regardless of
        ``kappa`` — warm-starting an unknown arm to its neutral q0=0.5 would
        wrongly bias it AND break the Laplace-smoothed convergence guarantee for
        unknown arms (the posterior mean would not converge to the empirical mean
        with the standard ``+1`` Laplace counts).
        """
        try:
            from litellm_llmrouter.personalized_routing import (
                _DEFAULT_QUALITY_BIASES,
            )

            if model not in _DEFAULT_QUALITY_BIASES:
                return (1.0, 1.0)
            q0 = _DEFAULT_QUALITY_BIASES[model]
        except Exception:
            return (1.0, 1.0)
        kappa = self._cold_start_kappa
        return (1.0 + kappa * q0, 1.0 + kappa * (1.0 - q0))

    def _get_posterior(self, bucket: str, model: str) -> Posterior:
        """Get-or-create the posterior, applying the cold-start prior on create."""
        backend = self._backend
        ensure = getattr(backend, "ensure", None)
        if callable(ensure):
            post: Posterior = ensure(bucket, model, self._cold_start_prior(model))
            return post
        return backend.get(bucket, model)

    # ------------------------------------------------------------------
    # Candidate selection (dynamic arms)
    # ------------------------------------------------------------------

    def _candidates(self, context: RoutingContext) -> list[Dict]:
        """Candidate deployments matching the requested model group.

        Mirrors ``CostAwareRoutingStrategy._get_candidates``.
        """
        router = context.router
        healthy = getattr(router, "healthy_deployments", None)
        if healthy is None:
            healthy = getattr(router, "model_list", []) or []
        return [d for d in healthy if d.get("model_name") == context.model]

    def _drop_open_breakers(self, candidates: list[Dict]) -> list[Dict]:
        """Exclude candidates whose provider circuit breaker is open.

        This is where *vanishing* arms are handled: the select loop only ever
        iterates live candidates, while the posterior for an excluded arm
        persists in the backend and re-enters the moment the breaker closes.
        Falls back to the full list if every provider is down (no worse than
        not filtering).
        """
        try:
            from litellm_llmrouter.resilience import get_circuit_breaker_manager

            cb_manager = get_circuit_breaker_manager()
            available: list[Dict] = []
            for cand in candidates:
                provider = cand.get("litellm_params", {}).get("custom_llm_provider", "")
                if provider:
                    breaker = cb_manager.get_breaker(provider)
                    if not breaker.is_open:
                        available.append(cand)
                else:
                    available.append(cand)
            return available if available else candidates
        except Exception:
            return candidates

    @staticmethod
    def _arm_key(deployment: Dict) -> str:
        """The arm key: ``litellm_params.model``."""
        return str(deployment.get("litellm_params", {}).get("model", ""))

    # ------------------------------------------------------------------
    # Bucketing
    # ------------------------------------------------------------------

    def _bucket(self, context: RoutingContext) -> str:
        """Coarse task bucket. Must NEVER raise and must NEVER block.

        v1: use the centroid complexity tier **only when the centroid model is
        already warm** (its embedding model loaded). Cold-loading the
        sentence-transformer on the routing hot path would blow the ~2ms budget
        (and block in tests), so a not-yet-loaded classifier falls through to the
        deterministic length/model fallback. Wrapped in try/except → ``"default"``.
        """
        try:
            from litellm_llmrouter.custom_routing_strategy import (
                CENTROID_ROUTING_AVAILABLE,
            )

            if CENTROID_ROUTING_AVAILABLE:
                from litellm_llmrouter.centroid_routing import get_centroid_strategy

                strategy = get_centroid_strategy()
                classifier = getattr(strategy, "_classifier", None)
                # Only use the centroid tier if the model is ALREADY loaded —
                # never trigger a cold model load from the hot path.
                if classifier is not None and getattr(classifier, "_loaded", False):
                    text = self._context_text(context)
                    if text:
                        result = classifier.classify(text)
                        tier = getattr(result, "complexity", None) or getattr(
                            result, "tier", None
                        )
                        if tier:
                            return f"tier:{tier}"
        except Exception:
            pass
        # Deterministic, dependency-free fallback: a coarse prompt-length bucket
        # (stable, no model load). Falls back to the model group if no text.
        text = self._context_text(context)
        if text:
            approx_tokens = len(text) // 4
            if approx_tokens < 64:
                return "len:xs"
            if approx_tokens < 512:
                return "len:s"
            if approx_tokens < 2048:
                return "len:m"
            return "len:l"
        return context.model or "default"

    @staticmethod
    def _context_text(context: RoutingContext) -> str:
        """Best-effort prompt text extraction for bucketing."""
        if isinstance(context.input, str) and context.input:
            return context.input
        if context.messages:
            for msg in reversed(context.messages):
                content = msg.get("content")
                if isinstance(content, str) and content:
                    return content
        return ""

    # ------------------------------------------------------------------
    # The hot path: select_deployment
    # ------------------------------------------------------------------

    def select_deployment(self, context: RoutingContext) -> Optional[Dict]:
        """Thompson-sample over live candidate arms and pick the argmax draw."""
        from litellm_llmrouter.candidate_filter import filter_routable_candidates

        cands = self._candidates(context)
        # RouteIQ-99e8 (cooldown) + RouteIQ-badb (gov-ban): remove cooled-down /
        # gov-banned arms BEFORE scoring, so a cooled-down/unhealthy or banned
        # arm is never scored or selected (proactive, not retried-after-failure).
        cands = filter_routable_candidates(context.router, cands)
        cands = self._drop_open_breakers(cands)
        if not cands:
            return None
        if len(cands) == 1:
            self._log_bucket(context.request_id, self._bucket(context))
            return cands[0]

        bucket = self._bucket(context)
        best: Optional[Dict] = None
        best_draw = -1.0
        for dep in cands:
            model = self._arm_key(dep)
            a, b = self._get_posterior(bucket, model).shape(moment_fit=self._moment_fit)
            x = sample_kumaraswamy(a, b, self._rng)
            if x > best_draw:
                best_draw, best = x, dep

        self._log_bucket(context.request_id, bucket)
        return best

    def validate(self) -> Tuple[bool, Optional[str]]:
        """The strategy is ready iff a posterior backend is present."""
        return (self._backend is not None), None

    # ------------------------------------------------------------------
    # bucket logging for feedback recovery
    # ------------------------------------------------------------------

    def _log_bucket(self, request_id: Optional[str], bucket: str) -> None:
        if not request_id:
            return
        with self._lock:
            if request_id not in self._bucket_log:
                self._bucket_log_order.append(request_id)
                if len(self._bucket_log_order) > self._bucket_log_capacity:
                    oldest = self._bucket_log_order.pop(0)
                    self._bucket_log.pop(oldest, None)
            self._bucket_log[request_id] = bucket

    def _recover_bucket(self, request_id: Optional[str]) -> Optional[str]:
        if not request_id:
            return None
        with self._lock:
            return self._bucket_log.get(request_id)

    # ------------------------------------------------------------------
    # Online update + reward shaping (called by the generalized loop)
    # ------------------------------------------------------------------

    def update(
        self,
        model: str,
        score: float,
        *,
        request_id: Optional[str] = None,
        bucket: Optional[str] = None,
        norm_cost: Optional[float] = None,
        norm_lat: Optional[float] = None,
    ) -> None:
        """Apply one reward observation to the ``(bucket, model)`` posterior.

        Reward shaping (cost/latency terms with ``None`` drop out and the
        remaining weights renormalize):

            r_quality = (score + 1) / 2  if score in [-1, 1] else clamp(score)
            r = w_q*r_quality + w_c*(1 - norm_cost) + w_l*(1 - norm_lat)

        then ``alpha += r; beta += (1 - r)``.

        Args:
            model: Arm key (``litellm_params.model``).
            score: Quality score, ``[-1, 1]`` (feedback contract) or ``[0, 1]``.
            request_id: Used to recover the bucket logged at decision time.
            bucket: Explicit bucket (overrides ``request_id`` recovery).
            norm_cost: Normalized cost in ``[0, 1]`` (cheaper => higher reward).
            norm_lat: Normalized latency in ``[0, 1]`` (faster => higher reward).
        """
        resolved_bucket = bucket or self._recover_bucket(request_id) or "default"

        if -1.0 <= score <= 1.0:
            r_quality = (score + 1.0) / 2.0
        else:
            r_quality = min(max(score, 0.0), 1.0)

        terms: list[Tuple[float, float]] = [(self._w_quality, r_quality)]
        if norm_cost is not None:
            terms.append((self._w_cost, 1.0 - min(max(norm_cost, 0.0), 1.0)))
        if norm_lat is not None:
            terms.append((self._w_latency, 1.0 - min(max(norm_lat, 0.0), 1.0)))

        weight_sum = sum(w for w, _ in terms) or 1.0
        r = sum(w * v for w, v in terms) / weight_sum
        r = min(max(r, 0.0), 1.0)

        # Ensure the posterior exists with its cold-start prior before updating.
        self._get_posterior(resolved_bucket, model)
        self._backend.update(resolved_bucket, model, r)

    def update_from_feedback(self, feedback: "RoutingFeedback") -> None:
        """Protocol-conformant adapter entry — delegates to :meth:`update`."""
        meta = getattr(feedback, "metadata", None) or {}
        self.update(
            feedback.model,
            feedback.score,
            request_id=getattr(feedback, "request_id", None),
            bucket=getattr(feedback, "bucket", None),
            norm_cost=meta.get("norm_cost"),
            norm_lat=meta.get("norm_lat"),
        )

    def maybe_decay(self, bucket: str, model: str, days_elapsed: float) -> None:
        """Decay a stale posterior toward its prior (re-opens exploration)."""
        self._backend.decay(
            bucket,
            model,
            self._decay_gamma,
            days_elapsed,
            self._cold_start_prior(model),
        )

    # ------------------------------------------------------------------
    # MLOps artifact hooks (strategy EXPOSES; the loop CALLS)
    # ------------------------------------------------------------------

    def export_artifact(self) -> Dict[str, Any]:
        """Snapshot the posteriors + a manifest stamp (warm-start artifact)."""
        snapshot_fn = getattr(self._backend, "snapshot", None)
        body = snapshot_fn() if callable(snapshot_fn) else {"posteriors": []}
        return {
            "kind": "posterior_json",
            "strategy": STRATEGY_NAME,
            "version": STRATEGY_VERSION,
            "state": body,
        }

    def load_artifact(self, ref: "ArtifactRef") -> bool:
        """Hydrate posteriors from a verified artifact (safe-swap on success)."""
        try:
            payload = getattr(ref, "payload", None)
            if payload is None:
                manifest = getattr(ref, "manifest", None) or {}
                payload = manifest.get("state")
            if not isinstance(payload, dict):
                return False
            state = payload.get("state", payload)
            hydrate_fn = getattr(self._backend, "hydrate", None)
            if not callable(hydrate_fn):
                return False
            hydrate_fn(state)
            return True
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("KTS load_artifact failed: %s", e)
            return False

    def reload(self, ref: "ArtifactRef") -> bool:
        """Hot-swap alias for :meth:`load_artifact`."""
        return self.load_artifact(ref)

    def declare_capabilities(self) -> "AdapterManifest":
        """Return the adapter manifest describing this strategy."""
        from litellm_llmrouter.adapters.contract import (
            ADAPTER_API_VERSION,
            AdapterManifest,
        )
        from litellm_llmrouter.gateway.plugin_manager import PluginCapability

        return AdapterManifest(
            name=STRATEGY_NAME,
            version=STRATEGY_VERSION,
            adapter_api_version=ADAPTER_API_VERSION,
            family=STRATEGY_NAME,
            capabilities={PluginCapability.ROUTING_STRATEGY},
            learns=True,
            train_mode="continuous",
            uses_artifact=True,
            state_backend="memory",
            artifact_kinds={"posterior_json"},
            required_signals=set(),
            description="Kumaraswamy-Thompson online bandit router.",
        )


# ---------------------------------------------------------------------------
# Registration helper (NO edit to custom_routing_strategy.py)
# ---------------------------------------------------------------------------


def register_kumaraswamy_thompson_strategy() -> bool:
    """Register the bandit in the routing registry (gated by settings).

    Mirrors ``register_centroid_strategy``: self-contained, called from
    ``startup.register_strategies()`` with one added line. No edit to
    ``CustomRoutingStrategyBase`` / ``async_get_available_deployment`` /
    ``_route_via_pipeline`` — the strategy rides the existing pipeline.

    Returns:
        True if registered, False if disabled or registration failed.
    """
    try:
        from litellm_llmrouter.settings import get_settings

        settings = get_settings()
        kts = getattr(settings, "kumaraswamy_thompson", None)
        if kts is None or not getattr(kts, "enabled", False):
            return False

        from litellm_llmrouter.strategy_registry import get_routing_registry

        strategy = KumaraswamyThompsonStrategy(
            seed=getattr(kts, "seed", None),
            w_quality=getattr(kts, "w_quality", 0.5),
            w_cost=getattr(kts, "w_cost", 0.4),
            w_latency=getattr(kts, "w_latency", 0.1),
            decay_gamma=getattr(kts, "decay_gamma", 0.99),
            cold_start_kappa=getattr(kts, "cold_start_kappa", 5.0),
            moment_fit=getattr(kts, "moment_fit", False),
        )
        registry = get_routing_registry()
        manifest_dict: Dict[str, Any] = {}
        try:
            manifest_dict = {"manifest": asdict(strategy.declare_capabilities())}
        except Exception:
            pass
        registry.register(
            STRATEGY_NAME,
            strategy,
            version=STRATEGY_VERSION,
            family=STRATEGY_NAME,
            metadata=manifest_dict,
        )
        logger.info("Registered Kumaraswamy-Thompson strategy as %r", STRATEGY_NAME)
        return True
    except Exception as e:
        logger.warning("KTS register failed: %s", e)
        return False
