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
- **In-memory backend is the DEFAULT backend.** ``InMemoryPosteriorBackend`` is
  the byte-stable default; the core bandit + every unit test run with NO external
  deps. ``FilePosteriorBackend`` (RouteIQ-95a8) is the cred-free DURABLE backend:
  it mirrors the in-memory semantics but persists posteriors to a JSON file via
  an atomic ``json.dump`` + ``os.replace`` (the governance-store pattern) so a
  worker restart RESUMES convergence instead of starting cold. It LOADS prior
  posteriors at construction and persists on ``update`` (debounced — see the
  ``flush_interval``/``dirty_threshold`` knobs — never fsync per update on the
  hot path). Backend selection is wired through the ``backend`` settings field
  (``memory`` | ``file``); the default stays ``memory`` for byte-stable behavior.
  Durable Redis (hot) / Aurora (durable) backends are still NOT built — the
  ``durable`` settings field remains a reserved placeholder (P2). The bandit is
  numpy-free; ``FilePosteriorBackend`` is likewise pure stdlib.
- Determinism: the RNG is threaded as a ``random.Random`` *object*, never the
  global ``random`` module — seeded for tests/replay, no global-state pollution.

Fable 5 is GOV-BANNED as a routable arm and is never added to any default or
test arm set in this module.
"""

from __future__ import annotations

import json
import logging
import math
import os
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


def _record_selection_metric(strategy: str, model: str) -> None:
    """Emit the routing.selection metric (best-effort).

    Telemetry must never raise: a missing meter (OTel disabled) is a no-op and
    any recording error is swallowed.
    """
    try:
        from litellm_llmrouter.metrics import get_gateway_metrics

        m = get_gateway_metrics()
        if m is not None:
            m.record_routing_selection(strategy, model)
    except Exception:  # pragma: no cover - telemetry must not break flow
        pass


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
# form.
#
# The fit is a 1-D root-find, NOT a 2-D Newton solve (RouteIQ-f9e9 defect 2).
# The key observation: at a FIXED ``a``, the mean is strictly monotone in ``b``,
# so ``_solve_b_for_mean`` pins the mean EXACTLY by bisection. That collapses
# the problem to one free variable ``a``: choose ``a`` so the variance matches.
# At a fixed mean, ``var_K(a, b_solve(a, mean))`` is monotone-DECREASING in ``a``
# across the holdable low-``a`` branch ``[tiny, a_cap]`` (``a_cap`` is the largest
# ``a`` whose mean ``_solve_b_for_mean`` can still hold before ``b`` exceeds the
# bisection ceiling -- see ``fit_kumaraswamy_moments``). So the fit is a SINGLE
# bisection over that branch (RouteIQ-37d6: this replaced the earlier golden-
# section variance-floor search; there is no golden-section step any more):
#   1. if the target variance is FEASIBLE (>= the variance at ``a_cap``), bisect
#      ``[tiny, a_cap]`` for the exact variance match -> exact mean + variance;
#   2. if INFEASIBLE (target below the tightest reachable variance -- high-
#      evidence near-symmetric posteriors past the ceiling), return ``a_cap``:
#      the tightest Kumaraswamy at the correct mean. Its variance is slightly
#      ABOVE target (more exploration), which is the doc's degradation contract
#      -- never a wrong mean, never UNDER-exploration.
# This holds the mean to ~1e-7 and the variance to ~1.0x across the full
# evidence grid (the old 2-D Newton inflated variance ~3-7x on peaked
# posteriors -> over-exploration). Pure stdlib ``math`` only (no numpy).


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


def _solve_b_for_mean(a: float, target_mean: float, iters: int = 50) -> float:
    """1-D log-bisection for ``b`` matching ``mean_K(a, b) = target_mean``.

    ``mean_K`` is strictly monotone-decreasing in ``b`` at fixed ``a``, so a
    log-space bisection over ``[1e-12, 1e18]`` converges robustly. This pins the
    mean EXACTLY for any ``a`` in the search range, which is what makes the
    moment fit a 1-D problem (root-find ``a`` for the variance; the mean is held
    here). The 1e18 ceiling (vs a tighter 1e12) extends the ``a``-range over
    which the mean stays holdable to ``a ~ 16`` -- the feasible-variance branch
    for asymmetric mid-evidence posteriors (e.g. Beta(100,200), mean 1/3) needs
    ``b ~ 4e7`` there. 50 iterations halve the ~69-nat ``ln b`` range to ~6e-14,
    which holds the mean to ~1e-7 -- ample, and far cheaper than the 120 the
    nested floor search would otherwise pay per evaluation.
    """
    lo, hi = 1e-12, 1e18
    for _ in range(iters):
        mid = math.sqrt(lo * hi)
        m, _ = kumaraswamy_mean_var(a, mid)
        lo, hi = (mid, hi) if m > target_mean else (lo, mid)
    return math.sqrt(lo * hi)


def _mean_holdable(a: float, target_mean: float) -> bool:
    """True if ``_solve_b_for_mean`` can hold the mean at this ``a``.

    For very large ``a`` the required ``b`` exceeds the bisection ceiling and the
    mean can no longer be matched; the golden-section search must not wander
    into that region. Cheap re-check used to cap the search upper bound.
    """
    b = _solve_b_for_mean(a, target_mean)
    m, _ = kumaraswamy_mean_var(a, b)
    return abs(m - target_mean) < 1e-6


def fit_kumaraswamy_moments(alpha: float, beta: float) -> Tuple[float, float]:
    """Map ``Beta(alpha, beta) -> Kumaraswamy(a, b)`` (doc-20 §3.1 option-2).

    Matches the mean ``alpha/(alpha+beta)`` exactly and the variance as closely
    as the Kumaraswamy family allows, via a 1-D root-find on ``a`` (the mean is
    held by :func:`_solve_b_for_mean` at every step). Pure deterministic
    function of ``(alpha, beta)`` (no RNG, no global state). Returns valid params
    (``a > 0``, ``b > 0``) for all inputs.

    Structure (RouteIQ-f9e9 defect-2 fix -- replaces the old 2-D Newton solve
    that inflated variance ~3-7x on peaked posteriors):
    - Exact special-case short-circuits (``Kuma(1,b)=Beta(1,b)``,
      ``Kuma(a,1)=Beta(a,1)``) -> byte-stable corners.
    - Cap the ``a`` search at the largest value where the mean is still holdable
      (large ``a`` needs ``b`` past the bisection ceiling) so the search never
      wanders into the region where the mean breaks. ``var_at`` is monotone-
      decreasing across ``[-12, la_hi]`` (the low-``a`` branch up to the cap), so
      a single bisection both decides feasibility and finds the solution:
    - FEASIBLE (``var_at(la_hi) <= t_var <= var_at(-12)``): bisect for the exact
      variance match (the mean is held exactly at every step).
    - INFEASIBLE (``t_var < var_at(la_hi)``, the tightest reachable variance is
      still above target -- high-evidence asymmetric posteriors past the
      ceiling): return the cap point. Its variance is slightly ABOVE target
      (more exploration), the doc's degradation contract: never a wrong mean,
      never under-explore.
    The single-bisection form avoids the prior golden-section floor search:
    within the bandit's 0.3..200 evidence band the target is always on the low-a
    branch, so the common path is one ~46-step bisection.
    """
    # Exact special cases (byte-stable corners). alpha==beta==1 falls out here.
    if abs(alpha - 1.0) < 1e-12:
        return (1.0, beta)
    if abs(beta - 1.0) < 1e-12:
        return (alpha, 1.0)

    s = alpha + beta
    t_mean = alpha / s
    t_var = (alpha * beta) / (s * s * (s + 1.0))

    def var_at(la: float) -> float:
        a = math.exp(la)
        _, v = kumaraswamy_mean_var(a, _solve_b_for_mean(a, t_mean))
        return v

    # Upper bound on ln a: the largest ln a where the mean is still holdable
    # (past it the required b exceeds the bisection ceiling and the mean breaks).
    # Coarse unit-step walk to bracket the boundary, THEN bisect for it -- a
    # plain unit-step cap lands up to a full e-fold below the true boundary and
    # can skip the feasible-variance solution (e.g. Beta(100,200): the feasible
    # a~15 is holdable but a~20 is not, so a unit cap stops at a~7).
    if not _mean_holdable(math.e, t_mean):  # even a=e unholdable -> tiny range
        la_hi = 0.0
    else:
        lo_h, hi_h = 0.0, 1.0
        while hi_h < 14.0 and _mean_holdable(math.exp(hi_h), t_mean):
            lo_h, hi_h = hi_h, hi_h + 1.0
        # boundary in [lo_h, hi_h): bisect (holdable at lo_h, not at hi_h).
        for _ in range(24):
            mid_h = (lo_h + hi_h) / 2.0
            if _mean_holdable(math.exp(mid_h), t_mean):
                lo_h = mid_h
            else:
                hi_h = mid_h
        la_hi = lo_h

    # Infeasible: even the tightest holdable a (the cap) has variance above the
    # target -> return the cap (var slightly high = more exploration, mean exact).
    if var_at(la_hi) >= t_var:
        return (math.exp(la_hi), _solve_b_for_mean(math.exp(la_hi), t_mean))

    # Feasible: var_at is monotone-decreasing on [-12, la_hi]; bisect for the
    # exact variance match. ~46 steps -> ln a to ~(la_hi+12)/2^46 ~ 1e-13.
    lo2, hi2 = -12.0, la_hi
    for _ in range(46):
        mid = (lo2 + hi2) / 2.0
        if var_at(mid) > t_var:
            lo2 = mid
        else:
            hi2 = mid
    a = math.exp((lo2 + hi2) / 2.0)
    return (a, _solve_b_for_mean(a, t_mean))


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
    # Cached moment-fit (RouteIQ-f9e9). 0.0 a => unset. The cache key is the
    # EXACT (alpha, beta) the fit was computed for, NOT a log-spaced bucket: a
    # bucket key served a stale (a,b) for evidence changes WITHIN a bucket
    # (e.g. Beta(8,8) -> Beta(23,8), both in bucket 4 -> a ~0.24-wrong sampled
    # mean). The fit is a pure deterministic function of (alpha, beta), so
    # keying on (alpha, beta) is exact: a hit only when the counts are unchanged
    # since the last fit, a miss (recompute) on ANY evidence change. The fit is
    # ~30 lgamma calls; arms update at most once per request, so this is still
    # off the hot path. Per-instance -> reset with the backend, no singleton.
    _fit_a: float = 0.0
    _fit_b: float = 0.0
    _fit_alpha: float = -1.0
    _fit_beta: float = -1.0

    def mean(self) -> float:
        """Posterior mean reward ``alpha / (alpha + beta)``."""
        total = self.alpha + self.beta
        return self.alpha / total if total > 0 else 0.5

    def shape(self, *, moment_fit: bool = False) -> Tuple[float, float]:
        """Kumaraswamy shape ``(a, b)``.

        ``moment_fit=False`` (default): the §3.1 option-1 ``a=alpha, b=beta``
        shortcut -- unchanged, byte-stable. ``moment_fit=True``: the §3.1
        option-2 moment-fit (:func:`fit_kumaraswamy_moments`), cached on the
        EXACT ``(alpha, beta)`` so any evidence change recomputes (RouteIQ-f9e9
        defect 1 -- the old bucket key served a stale fit within a bucket).
        """
        if not moment_fit:
            return (self.alpha, self.beta)  # option-1 shortcut (default)
        if (
            self._fit_a <= 0.0
            or self._fit_alpha != self.alpha
            or self._fit_beta != self.beta
        ):
            self._fit_a, self._fit_b = fit_kumaraswamy_moments(self.alpha, self.beta)
            self._fit_alpha, self._fit_beta = self.alpha, self.beta
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


class FilePosteriorBackend(InMemoryPosteriorBackend):
    """DURABLE file-backed backend (RouteIQ-95a8) — cred-free, pure stdlib.

    The in-memory backend is per-worker and volatile: posteriors are lost on
    restart, never shared, and in an HA fleet the bandit never truly converges
    because each replica relearns from its uniform prior. This backend fixes the
    *durability* gap for the cred-free default deploy (Redis hot / Aurora durable
    remain a later seed): it keeps the exact in-memory semantics (it subclasses
    :class:`InMemoryPosteriorBackend`) but mirrors the live store to a JSON file
    so a fresh process resumes from the persisted posteriors.

    Atomicity: persistence mirrors ``governance_store`` — ``json.dump`` to a
    sibling ``*.tmp`` in the SAME directory then ``os.replace`` (an atomic rename
    on POSIX/NTFS). A crash mid-write leaves the prior complete file intact; a
    reader never sees a half-written file. An ``os.fsync`` is issued before the
    replace so the bytes are on disk before the rename is durable.

    Hot-path cost: persistence is DEBOUNCED — :meth:`update` / :meth:`decay`
    increment a dirty counter and only flush when either ``dirty_threshold``
    mutations have accumulated OR ``flush_interval`` seconds have elapsed since
    the last flush. The default knobs (32 updates / 5 s) keep the steady-state
    cost at ~1 serialize per 32 requests instead of per request. Set
    ``dirty_threshold=1`` for write-through (used by the durability unit test).
    The graceful-shutdown drain path calls :meth:`flush` (via
    :func:`flush_posteriors_on_shutdown`) to force a final persist of the
    debounced tail so a clean stop does not lose up to ``dirty_threshold - 1``
    updates (RouteIQ-95a8 DEFECT-2).

    Thread-safety: inherits the parent ``RLock``; the flush serializes under the
    same lock so the snapshot is internally consistent.
    """

    def __init__(
        self,
        path: str,
        *,
        flush_interval: float = 5.0,
        dirty_threshold: int = 32,
    ) -> None:
        super().__init__()
        self._path = path
        self._flush_interval = max(0.0, float(flush_interval))
        self._dirty_threshold = max(1, int(dirty_threshold))
        self._dirty = 0
        self._last_flush = 0.0
        # LOAD prior posteriors at construction so a restart resumes convergence.
        self._load_from_disk()

    # -- persistence internals ----------------------------------------------

    def _load_from_disk(self) -> None:
        """Hydrate from the on-disk snapshot if present (no-op when absent)."""
        if not self._path or not os.path.exists(self._path):
            return
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError) as exc:
            # A corrupt/partial file must not crash startup — log and start cold.
            logger.warning(
                "FilePosteriorBackend: failed to load %s, starting cold: %s",
                self._path,
                exc,
            )
            return
        # hydrate() acquires the lock and replaces the store atomically.
        self.hydrate(data if isinstance(data, dict) else {})

    def _atomic_write(self, data: Dict[str, Any]) -> None:
        """Serialize ``data`` to ``self._path`` atomically (tmp + os.replace).

        Mirrors the governance-store durable pattern. Writes to a uniquely-named
        temp file in the SAME directory (so ``os.replace`` is an atomic rename on
        the same filesystem), fsyncs, then replaces. Best-effort — a write error
        is logged, never raised (durability is an optimization, not a hard dep on
        the hot path).
        """
        if not self._path:
            return
        directory = os.path.dirname(os.path.abspath(self._path)) or "."
        tmp_path = f"{self._path}.{os.getpid()}.tmp"
        try:
            os.makedirs(directory, exist_ok=True)
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self._path)
        except OSError as exc:
            logger.warning(
                "FilePosteriorBackend: failed to persist %s: %s", self._path, exc
            )
            # Best-effort cleanup of a stranded temp file.
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass

    def flush(self) -> None:
        """Force-persist the current posteriors regardless of the debounce state.

        Wired into the graceful-shutdown drain path via
        :func:`flush_posteriors_on_shutdown` (called from ``_routeiq_lifespan``)
        so the debounced tail (up to ``dirty_threshold - 1`` updates) is persisted
        on a clean stop instead of being lost (RouteIQ-95a8 DEFECT-2).
        """
        with self._lock:
            data = self.snapshot()
            self._dirty = 0
            self._last_flush = time.time()
        # Write OUTSIDE the lock-built snapshot copy is already detached; serialize
        # under the lock-released phase to avoid holding the lock during disk I/O.
        self._atomic_write(data)

    def _maybe_flush(self) -> None:
        """Persist iff the dirty count or the time-since-flush crossed a bound."""
        now = time.time()
        with self._lock:
            self._dirty += 1
            due = self._dirty >= self._dirty_threshold or (
                self._flush_interval > 0.0
                and (now - self._last_flush) >= self._flush_interval
            )
            if not due:
                return
            data = self.snapshot()
            self._dirty = 0
            self._last_flush = now
        self._atomic_write(data)

    # -- mutating ops: persist (debounced) after the in-memory update --------

    def update(self, bucket: str, model: str, reward: float) -> None:
        """Apply the conjugate update in memory, then debounced-persist."""
        super().update(bucket, model, reward)
        self._maybe_flush()

    def decay(
        self,
        bucket: str,
        model: str,
        gamma: float,
        days: float,
        prior: Tuple[float, float],
    ) -> None:
        """Decay in memory, then debounced-persist (a no-op decay does not dirty)."""
        if days <= 0 or gamma >= 1.0:
            return
        super().decay(bucket, model, gamma, days, prior)
        self._maybe_flush()

    def hydrate(self, data: Dict[str, Any]) -> None:
        """Replace state from a snapshot dict and persist the new state."""
        super().hydrate(data)
        # Persist the hydrated state so a load_artifact swap is durable too. Reset
        # the debounce clock; this is an explicit full-state replacement.
        with self._lock:
            self._dirty = 0
            self._last_flush = time.time()
            snap = self.snapshot()
        self._atomic_write(snap)


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
            _record_selection_metric(STRATEGY_NAME, self._arm_key(cands[0]))
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
        if best is not None:
            _record_selection_metric(STRATEGY_NAME, self._arm_key(best))
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


def build_posterior_backend(kts: Any) -> PosteriorBackend:
    """Construct the posterior backend selected by the settings ``backend`` field.

    ``memory`` (default, or any unknown value) -> :class:`InMemoryPosteriorBackend`
    (byte-stable, per-worker, volatile). ``file`` -> :class:`FilePosteriorBackend`
    (cred-free DURABLE; persists to ``state_path`` and reloads at startup so a
    restart resumes convergence — RouteIQ-95a8). A ``file`` selection with an
    empty ``state_path`` degrades gracefully to a non-persisting file backend
    (logged), so a misconfig never crashes startup.
    """
    backend_kind = str(getattr(kts, "backend", "memory") or "memory").lower()
    if backend_kind != "file":
        return InMemoryPosteriorBackend()

    state_path = str(getattr(kts, "state_path", "") or "")
    if not state_path:
        logger.warning(
            "KTS backend='file' but state_path is empty; persistence disabled "
            "(posteriors will not survive restart). Set "
            "ROUTEIQ_KUMARASWAMY_THOMPSON__STATE_PATH to enable durability."
        )
    return FilePosteriorBackend(
        state_path,
        flush_interval=float(getattr(kts, "flush_interval_seconds", 5.0)),
        dirty_threshold=int(getattr(kts, "flush_dirty_threshold", 32)),
    )


def flush_posteriors_on_shutdown() -> bool:
    """Force-persist the active KTS strategy's posteriors at graceful shutdown.

    DEFECT-2 fix (RouteIQ-95a8): :meth:`FilePosteriorBackend.flush` is documented
    as "call explicitly at drain", but without a live caller a clean shutdown
    loses up to ``dirty_threshold - 1`` (default 31) tail updates — undercutting
    the seed's convergence-across-restarts goal. This wires the drain hook.

    It resolves the active strategy from the routing registry the SAME way
    :func:`register_kumaraswamy_thompson_strategy` registered it, and if that
    strategy's backend is a :class:`FilePosteriorBackend`, calls its
    :meth:`~FilePosteriorBackend.flush`. Best-effort and fail-open: any error
    (no registry, strategy absent, non-file backend, flush failure) is logged
    and swallowed so shutdown never blocks on durability.

    Returns:
        True if a FilePosteriorBackend was found and flushed, else False.
    """
    try:
        from litellm_llmrouter.strategy_registry import get_routing_registry

        strategy = get_routing_registry().get(STRATEGY_NAME)
        if strategy is None:
            return False
        backend = getattr(strategy, "_backend", None)
        flush = getattr(backend, "flush", None)
        if isinstance(backend, FilePosteriorBackend) and callable(flush):
            flush()
            logger.info("Flushed Kumaraswamy-Thompson posteriors to disk at shutdown")
            return True
        return False
    except Exception as exc:  # pragma: no cover - defensive, must never block drain
        logger.warning("KTS posterior flush-on-shutdown failed: %s", exc)
        return False


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

        # Select the posterior backend (RouteIQ-95a8). 'memory' (default) stays
        # byte-stable; 'file' wires the cred-free DURABLE backend so a restart
        # resumes convergence. Unknown values fall back to memory.
        backend = build_posterior_backend(kts)

        strategy = KumaraswamyThompsonStrategy(
            backend=backend,
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


# ===========================================================================
# LinUCB feature-vector contextual bandit (RouteIQ-6c67)
# ===========================================================================
#
# The Kumaraswamy-Thompson bandit above is BUCKET-contextual: one Beta posterior
# per coarse ``(task_bucket, arm)`` pair. LinUCB is FEATURE-VECTOR contextual --
# it learns a per-arm linear reward model ``r ~= theta·x`` over a REAL-VALUED
# context vector ``x`` and scores each arm by its Upper Confidence Bound:
#
#     UCB(arm) = theta·x + alpha * sqrt(x^T A^-1 x)
#
# where ``A = I + sum_t x_t x_t^T`` (ridge design matrix) and ``theta = A^-1 b``,
# ``b = sum_t r_t x_t``. The exploration term ``alpha*sqrt(...)`` is LARGE when an
# arm has seen little evidence in the direction of ``x`` (cold-start explores) and
# SHRINKS as evidence accumulates in that direction (converges to greedy exploit).
#
# DISCIPLINE: pure stdlib, NO numpy (matches the KT module). The only nontrivial
# linear-algebra op LinUCB needs is ``A^-1`` applied to a vector; we maintain the
# inverse ``A^-1`` DIRECTLY and update it with the rank-1 Sherman-Morrison
# formula on each observation, so there is never a full O(d^3) inversion and no
# matrix library is required. The feature dimension ``d`` is small (a prompt-
# length feature, a few requested-profile one-hots, and a hashed tenant bucket),
# so the per-decision cost is O(num_arms * d^2) with tiny d.
#
# This is ADDITIVE alongside the KT bandit: a separate strategy class, a separate
# registry name (``llmrouter-linucb``), and a separate settings flag.

LINUCB_STRATEGY_NAME = "llmrouter-linucb"
LINUCB_STRATEGY_VERSION = "v1"

# Requested-profile labels that get a dedicated one-hot slot in the feature
# vector. An unrecognized profile contributes no profile slot (graceful).
_LINUCB_PROFILES = ("fast", "balanced", "powerful")


def identity_matrix(d: int) -> list[list[float]]:
    """A ``d x d`` identity matrix as a list-of-lists (stdlib, no numpy)."""
    return [[1.0 if i == j else 0.0 for j in range(d)] for i in range(d)]


def mat_vec(m: list[list[float]], v: list[float]) -> list[float]:
    """Matrix-vector product ``m @ v`` (stdlib)."""
    return [sum(m[i][k] * v[k] for k in range(len(v))) for i in range(len(m))]


def dot(a: list[float], b: list[float]) -> float:
    """Vector dot product (stdlib)."""
    return sum(x * y for x, y in zip(a, b))


def sherman_morrison_update(
    a_inv: list[list[float]], x: list[float]
) -> list[list[float]]:
    """Return ``(A + x x^T)^-1`` from ``A^-1`` via the Sherman-Morrison formula.

    ``(A + x x^T)^-1 = A^-1 - (A^-1 x x^T A^-1) / (1 + x^T A^-1 x)``.

    Pure stdlib (no numpy). Because every LinUCB update adds the rank-1 term
    ``x x^T`` to a positive-definite ``A`` (seeded as the identity ridge), the
    denominator ``1 + x^T A^-1 x`` is always ``>= 1`` so the update is stable and
    never divides by zero. Returns a NEW matrix; the input is not mutated.
    """
    d = len(a_inv)
    ainv_x = mat_vec(a_inv, x)  # A^-1 x  (column vector)
    denom = 1.0 + dot(x, ainv_x)  # 1 + x^T A^-1 x  (scalar, >= 1)
    # x^T A^-1  (row vector) == (A^-1 x)^T because A^-1 is symmetric.
    new = [[0.0] * d for _ in range(d)]
    for i in range(d):
        for j in range(d):
            new[i][j] = a_inv[i][j] - (ainv_x[i] * ainv_x[j]) / denom
    return new


@dataclass
class _LinUCBArm:
    """Per-arm LinUCB state: the inverse design matrix ``A^-1`` and ``b``.

    ``A`` starts as the identity (ridge regularization ``lambda=1``); ``A_inv``
    is maintained directly via Sherman-Morrison. ``b = sum_t r_t x_t`` and
    ``theta = A_inv @ b`` (recomputed lazily on score).
    """

    a_inv: list[list[float]]
    b: list[float]

    @classmethod
    def fresh(cls, d: int) -> "_LinUCBArm":
        return cls(a_inv=identity_matrix(d), b=[0.0] * d)

    def theta(self) -> list[float]:
        """``theta = A^-1 b`` (the ridge-regression weight estimate)."""
        return mat_vec(self.a_inv, self.b)

    def ucb(self, x: list[float], alpha: float) -> float:
        """UCB score ``theta·x + alpha*sqrt(x^T A^-1 x)`` for context ``x``."""
        mean = dot(self.theta(), x)
        ainv_x = mat_vec(self.a_inv, x)
        var = max(0.0, dot(x, ainv_x))  # x^T A^-1 x (>= 0; floored for fp drift)
        return mean + alpha * math.sqrt(var)

    def update(self, x: list[float], reward: float) -> None:
        """Observe ``(x, reward)``: rank-1 ``A`` update + ``b += reward*x``."""
        self.a_inv = sherman_morrison_update(self.a_inv, x)
        for i in range(len(self.b)):
            self.b[i] += reward * x[i]


class LinUCBRoutingStrategy(RoutingStrategy):
    """Feature-vector contextual bandit via LinUCB (RouteIQ-6c67).

    Learns a per-arm linear reward model over a real-valued context vector and
    selects the arm with the highest Upper Confidence Bound. ADDITIVE alongside
    the bucket-contextual :class:`KumaraswamyThompsonStrategy` (separate class,
    registry name ``llmrouter-linucb``, and settings flag).

    Feature vector (fixed dimension ``d``, all stdlib floats):
        - bias term (constant 1.0),
        - prompt-length feature (``log1p(approx_tokens) / 10``, ~[0, 1.x]),
        - requested-profile one-hots for ``fast`` / ``balanced`` / ``powerful``
          (read from ``request_kwargs`` / ``metadata`` key ``profile``),
        - a one-hot over ``tenant_buckets`` hashed tenant slots
          (``hash(tenant_id) % tenant_buckets``).

    Cold start: a fresh arm's ``A`` is the identity, so its UCB exploration term
    ``alpha*sqrt(x^T x)`` is large and the bandit EXPLORES untried arms. As an arm
    accumulates ``(x, reward)`` observations the confidence width shrinks in the
    seen directions and the bandit converges to exploiting the higher-reward arm.

    Reward is fed via :meth:`update` (model, score, request_id) -- the score is
    mapped from the ``[-1, 1]`` feedback contract into ``[0, 1]`` (matching the KT
    bandit's reward shaping). The context vector used at decision time is logged
    per ``request_id`` (bounded FIFO) so the offline feedback path can recover it.

    Pure stdlib, no numpy: the per-arm inverse design matrix is maintained via
    rank-1 :func:`sherman_morrison_update`. The RNG (tie-breaking only) is a
    threaded ``random.Random`` object.

    Stress-harness validation: ``tools/stress_harness`` resolves the active
    strategy by name; ``linucb`` is not a registered verdict family so it hits the
    generic distribution verdict (``stress_harness.verdicts.generic_verdict``).
    Feed rewards via :meth:`update` and confirm the higher-reward arm comes to
    dominate the per-bucket selection distribution.
    """

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        tenant_buckets: int = 8,
        seed: Optional[int] = None,
        bucket_log_capacity: int = 4096,
        **kwargs: Any,
    ) -> None:
        self._alpha = max(0.0, float(alpha))
        self._tenant_buckets = max(1, int(tenant_buckets))
        # d = bias(1) + prompt_len(1) + profiles(3) + tenant one-hot(tenant_buckets)
        self._dim = 2 + len(_LINUCB_PROFILES) + self._tenant_buckets
        self._arms: Dict[str, _LinUCBArm] = {}
        self._rng = random.Random(seed)
        self._lock = threading.RLock()
        # request_id -> feature vector recovery for the offline feedback path.
        self._ctx_log: Dict[str, list[float]] = {}
        self._ctx_log_order: list[str] = []
        self._ctx_log_capacity = max(1, bucket_log_capacity)

    @property
    def name(self) -> str:
        return LINUCB_STRATEGY_NAME

    @property
    def version(self) -> Optional[str]:
        return LINUCB_STRATEGY_VERSION

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    @staticmethod
    def _context_text(context: RoutingContext) -> str:
        if isinstance(context.input, str) and context.input:
            return context.input
        for msg in reversed(context.messages or []):
            content = msg.get("content")
            if isinstance(content, str) and content:
                return content
        return ""

    @staticmethod
    def _requested_profile(context: RoutingContext) -> str:
        for src in (context.request_kwargs, context.metadata):
            if isinstance(src, dict):
                prof = src.get("profile")
                if isinstance(prof, str) and prof:
                    return prof.lower()
        return ""

    def _featurize(self, context: RoutingContext) -> list[float]:
        """Build the fixed-dimension feature vector for the request (stdlib)."""
        x = [0.0] * self._dim
        x[0] = 1.0  # bias
        text = self._context_text(context)
        approx_tokens = len(text) // 4 if text else 0
        x[1] = math.log1p(approx_tokens) / 10.0  # prompt-length feature
        prof = self._requested_profile(context)
        for i, label in enumerate(_LINUCB_PROFILES):
            if prof == label:
                x[2 + i] = 1.0
        tenant = context.tenant_id or context.user_id or ""
        if tenant:
            slot = hash(tenant) % self._tenant_buckets
            x[2 + len(_LINUCB_PROFILES) + slot] = 1.0
        return x

    # ------------------------------------------------------------------
    # Candidate selection (mirrors the KT bandit)
    # ------------------------------------------------------------------

    def _candidates(self, context: RoutingContext) -> list[Dict]:
        router = context.router
        healthy = getattr(router, "healthy_deployments", None)
        if healthy is None:
            healthy = getattr(router, "model_list", []) or []
        return [d for d in healthy if d.get("model_name") == context.model]

    @staticmethod
    def _arm_key(deployment: Dict) -> str:
        return str(deployment.get("litellm_params", {}).get("model", ""))

    def _get_arm(self, model: str) -> _LinUCBArm:
        arm = self._arms.get(model)
        if arm is None:
            arm = _LinUCBArm.fresh(self._dim)
            self._arms[model] = arm
        return arm

    # ------------------------------------------------------------------
    # The hot path
    # ------------------------------------------------------------------

    def select_deployment(self, context: RoutingContext) -> Optional[Dict]:
        """Pick the arm with the highest LinUCB score for the request context."""
        from litellm_llmrouter.candidate_filter import filter_routable_candidates

        cands = self._candidates(context)
        cands = filter_routable_candidates(context.router, cands)
        if not cands:
            return None

        x = self._featurize(context)
        if len(cands) == 1:
            self._log_ctx(context.request_id, x)
            _record_selection_metric(LINUCB_STRATEGY_NAME, self._arm_key(cands[0]))
            return cands[0]

        best: Optional[Dict] = None
        best_score = -float("inf")
        with self._lock:
            for dep in cands:
                model = self._arm_key(dep)
                score = self._get_arm(model).ucb(x, self._alpha)
                # Tie-break with a tiny deterministic-RNG jitter so equal cold
                # arms don't always pick the first (spreads cold-start explore).
                score += self._rng.random() * 1e-9
                if score > best_score:
                    best_score, best = score, dep

        self._log_ctx(context.request_id, x)
        if best is not None:
            _record_selection_metric(LINUCB_STRATEGY_NAME, self._arm_key(best))
        return best

    def validate(self) -> Tuple[bool, Optional[str]]:
        if self._alpha < 0.0:
            return False, "alpha must be >= 0"
        if self._tenant_buckets < 1:
            return False, "tenant_buckets must be >= 1"
        return True, None

    # ------------------------------------------------------------------
    # Feedback recovery + online update
    # ------------------------------------------------------------------

    def _log_ctx(self, request_id: Optional[str], x: list[float]) -> None:
        if not request_id:
            return
        with self._lock:
            if request_id not in self._ctx_log:
                self._ctx_log_order.append(request_id)
                if len(self._ctx_log_order) > self._ctx_log_capacity:
                    oldest = self._ctx_log_order.pop(0)
                    self._ctx_log.pop(oldest, None)
            self._ctx_log[request_id] = x

    def _recover_ctx(self, request_id: Optional[str]) -> Optional[list[float]]:
        if not request_id:
            return None
        with self._lock:
            return self._ctx_log.get(request_id)

    def update(
        self,
        model: str,
        score: float,
        *,
        request_id: Optional[str] = None,
        features: Optional[list[float]] = None,
    ) -> None:
        """Apply one reward observation to ``model``'s LinUCB arm.

        The reward is mapped from the ``[-1, 1]`` feedback contract into
        ``[0, 1]`` (matching the KT bandit). The context vector is taken from
        ``features`` if supplied, else recovered from ``request_id`` (the vector
        logged at decision time). With neither, the update is a no-op (there is
        no context to attribute the reward to).

        Args:
            model: Arm key (``litellm_params.model``).
            score: Quality score, ``[-1, 1]`` (feedback contract) or ``[0, 1]``.
            request_id: Recovers the feature vector logged at decision time.
            features: Explicit feature vector (overrides ``request_id`` recovery).
        """
        x = features or self._recover_ctx(request_id)
        if x is None:
            return
        if -1.0 <= score <= 1.0:
            reward = (score + 1.0) / 2.0
        else:
            reward = min(max(score, 0.0), 1.0)
        with self._lock:
            self._get_arm(model).update(list(x), reward)

    def update_from_feedback(self, feedback: "RoutingFeedback") -> None:
        """Protocol-conformant adapter entry — delegates to :meth:`update`."""
        self.update(
            feedback.model,
            feedback.score,
            request_id=getattr(feedback, "request_id", None),
        )


def register_linucb_strategy() -> bool:
    """Register the LinUCB contextual bandit in the routing registry.

    Mirrors :func:`register_kumaraswamy_thompson_strategy`: self-contained,
    gated by ``settings.linucb.enabled`` (default off, byte-stable startup).
    ADDITIVE alongside the KT bandit — separate name / flag.

    Returns:
        True if registered, False if disabled or registration failed.
    """
    try:
        from litellm_llmrouter.settings import get_settings

        cfg = getattr(get_settings(), "linucb", None)
        if cfg is None or not getattr(cfg, "enabled", False):
            return False

        from litellm_llmrouter.strategy_registry import get_routing_registry

        strategy = LinUCBRoutingStrategy(
            alpha=getattr(cfg, "alpha", 1.0),
            tenant_buckets=getattr(cfg, "tenant_buckets", 8),
            seed=getattr(cfg, "seed", None),
        )
        get_routing_registry().register(
            LINUCB_STRATEGY_NAME,
            strategy,
            version=LINUCB_STRATEGY_VERSION,
            family=LINUCB_STRATEGY_NAME,
        )
        logger.info("Registered LinUCB strategy as %r", LINUCB_STRATEGY_NAME)
        return True
    except Exception as e:
        logger.warning("LinUCB register failed: %s", e)
        return False
