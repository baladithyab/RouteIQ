"""Offline corpus replay/backtest + distribution goodness-of-fit for the bandit.

OFFLINE DE-RISK FIRST. This module validates the Kumaraswamy-Thompson bandit
math against the *preserved* MLOps corpus (the VSR `sft_corpus.csv` export:
241 rows / 31 distinct ``selected_model`` arms, real ``actual_cost`` signal),
with NO live DB/cache. Everything runs on the in-memory backend.

What it proves, beyond the unit math in ``test_kumaraswamy_thompson.py``:

1. **A small, self-contained corpus loader** that reads the CSV and yields
   ``(arm, cost)`` observations — gracefully ``skip``s when the export is absent
   (it lives outside the worktree), so the suite stays hermetic.
2. **A replay/backtest**: replay corpus-grounded rewards through the bandit and
   assert the online update drives *selection* toward the high-reward (cheapest)
   arm — i.e. cumulative-regret shrinks vs. a no-learning baseline, with
   known-seed determinism. This is the "the online update moves the posterior
   toward high-reward arms" requirement exercised end-to-end on real arm names.
3. **Distribution goodness-of-fit**: the sampled mean of ``Kumaraswamy(a, b)``
   approximates the analytic Kumaraswamy mean within tolerance at a fixed seed,
   for several shapes — including the closed-form special cases.
4. **Quantile boundary correctness**: ``Q(0) -> 0``, ``Q(1) -> 1`` (within the
   interior clamp), monotone, and matches a reference inverse-CDF at known
   points.

Fable 5 is GOV-BANNED as a routable arm. The corpus does not contain it; this
module asserts that invariant explicitly (a guard against a future corpus that
might).
"""

from __future__ import annotations

import csv
import math
import os
import random
from collections import Counter
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import pytest

from litellm_llmrouter.kumaraswamy_thompson import (
    InMemoryPosteriorBackend,
    KumaraswamyThompsonStrategy,
    kumaraswamy_quantile,
    sample_kumaraswamy,
)

# GOV-banned arm — must never appear in any routable arm set.
_FABLE5_TOKENS = ("fable-5", "fable5", "fable_5", "fable 5")


# ---------------------------------------------------------------------------
# 1. The corpus loader (self-contained; graceful skip when absent)
# ---------------------------------------------------------------------------

# The preserved export lives OUTSIDE this worktree (it is the VSR de-risk
# corpus, not RouteIQ source). Resolve via the ROUTEIQ_BANDIT_CORPUS_CSV env
# override (preferred) or a portable sibling-repo relative path; skip if neither
# resolves. NO machine-specific absolute path (it never resolves on CI / another
# dev box and was flagged by the P3 review).
_CORPUS_CANDIDATES = [
    # ../../../vllm-sr-on-aws/mlops-corpus-export/ relative to this repo's parent
    Path(__file__).resolve().parents[3]
    / "vllm-sr-on-aws"
    / "mlops-corpus-export"
    / "sft_corpus.csv",
]


def _resolve_corpus() -> Optional[Path]:
    env = os.environ.get("ROUTEIQ_BANDIT_CORPUS_CSV")
    if env and Path(env).is_file():
        return Path(env)
    for cand in _CORPUS_CANDIDATES:
        if cand.is_file():
            return cand
    return None


class CorpusRow:
    """One replay observation derived from a corpus row."""

    __slots__ = ("arm", "cost", "prompt", "prompt_tokens")

    def __init__(
        self, arm: str, cost: Optional[float], prompt: str, prompt_tokens: int
    ) -> None:
        self.arm = arm
        self.cost = cost
        self.prompt = prompt
        self.prompt_tokens = prompt_tokens


def load_corpus(path: Path) -> List[CorpusRow]:
    """Load the SFT corpus CSV into replay rows.

    Pure stdlib ``csv`` — no pandas, no DB. The arm key is the corpus
    ``selected_model`` (the model a real router historically routed to). Cost is
    the real ``actual_cost`` where present, else ``None``.
    """
    rows: List[CorpusRow] = []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for rec in reader:
            arm = (rec.get("selected_model") or "").strip()
            if not arm:
                continue
            cost_raw = (rec.get("actual_cost") or "").strip()
            cost: Optional[float] = None
            if cost_raw:
                try:
                    cost = float(cost_raw)
                except ValueError:
                    cost = None
            ptoks_raw = (rec.get("prompt_tokens") or "").strip()
            try:
                ptoks = int(float(ptoks_raw)) if ptoks_raw else 0
            except ValueError:
                ptoks = 0
            rows.append(
                CorpusRow(
                    arm=arm,
                    cost=cost,
                    prompt=(rec.get("prompt") or "").strip(),
                    prompt_tokens=ptoks,
                )
            )
    return rows


@pytest.fixture(scope="module")
def corpus() -> List[CorpusRow]:
    path = _resolve_corpus()
    if path is None:
        pytest.skip("preserved MLOps corpus (sft_corpus.csv) not available")
    rows = load_corpus(path)
    if not rows:
        pytest.skip("corpus loaded empty")
    return rows


# ---------------------------------------------------------------------------
# 2. Loader sanity (these run only when the corpus is present)
# ---------------------------------------------------------------------------


def test_corpus_loads_expected_shape(corpus: List[CorpusRow]) -> None:
    # The preserved export is 241 rows / 31 distinct arms; assert the shape is
    # in the right neighborhood without over-fitting to the exact count.
    assert len(corpus) >= 200
    arms = {r.arm for r in corpus}
    assert len(arms) >= 25


def test_corpus_has_no_fable5_arm(corpus: List[CorpusRow]) -> None:
    # GOV invariant: Fable 5 is never a routable arm. Guards a future corpus.
    for row in corpus:
        low = row.arm.lower()
        assert not any(tok in low for tok in _FABLE5_TOKENS), (
            f"Fable 5 must never be a routable arm; found {row.arm!r}"
        )


def test_corpus_cost_signal_present(corpus: List[CorpusRow]) -> None:
    with_cost = [r for r in corpus if r.cost is not None]
    # The vast majority of rows carry a real cost signal.
    assert len(with_cost) >= 0.8 * len(corpus)
    assert all(r.cost is not None and r.cost >= 0.0 for r in with_cost)


# ---------------------------------------------------------------------------
# 3. Replay / backtest — the online update moves selection toward best arms
# ---------------------------------------------------------------------------


def _cost_table(corpus: List[CorpusRow]) -> dict[str, float]:
    """Mean ``actual_cost`` per arm (the corpus-grounded cost signal)."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for r in corpus:
        if r.cost is None:
            continue
        sums[r.arm] = sums.get(r.arm, 0.0) + r.cost
        counts[r.arm] = counts.get(r.arm, 0) + 1
    return {arm: sums[arm] / counts[arm] for arm in sums}


def _top_arms_by_count(corpus: List[CorpusRow], k: int) -> List[str]:
    """The k most-frequently-selected arms — a realistic candidate set."""
    return [arm for arm, _ in Counter(r.arm for r in corpus).most_common(k)]


def _norm_cost_map(arms: List[str], cost_table: dict[str, float]) -> dict[str, float]:
    """Min-max normalize the per-arm cost over the candidate set into [0, 1]."""
    costs = [cost_table.get(a, 0.0) for a in arms]
    lo, hi = min(costs), max(costs)
    span = (hi - lo) or 1.0
    return {a: (cost_table.get(a, 0.0) - lo) / span for a in arms}


def _replay_selection_share(
    arms: List[str],
    norm_cost: dict[str, float],
    *,
    seed: int,
    n_rounds: int,
    learn: bool,
) -> Tuple[str, float, float]:
    """Replay the bandit over a fixed arm set; return (best_arm, best_share, regret).

    The corpus-grounded reward ``r = 1 - norm_cost`` (cheaper arm => higher
    reward) is the *probability* a round is a "win". Each round's win/loss is fed
    through the REAL feedback contract as a ``[-1, 1]`` quality ``score``
    (``+1`` win / ``-1`` loss) — exactly the shape
    ``POST /routing/feedback`` produces — so the bandit's own reward shaping
    (``r_quality = (score + 1) / 2``) maps it back to ``{0, 1}`` Bernoulli
    counts. With ``learn=False`` the posterior is never updated (the no-learning
    baseline). Regret is summed against an oracle that always picks the cheapest
    arm. Deterministic at the fixed seed (threaded RNG, no global state).
    """
    strat = KumaraswamyThompsonStrategy(
        backend=InMemoryPosteriorBackend(),
        seed=seed,
        # Quality-channel reward: feed the corpus cost signal as the [-1,1]
        # feedback score, the supported online-update path.
        w_quality=1.0,
        w_cost=0.0,
        w_latency=0.0,
        cold_start_kappa=0.0,  # uniform Beta(1,1) prior — no quality-table bias.
    )
    rng = random.Random(seed)
    bucket = "replay"
    best_arm = min(arms, key=lambda a: norm_cost[a])
    best_reward = 1.0 - norm_cost[best_arm]

    picks: Counter[str] = Counter()
    regret = 0.0
    for _ in range(n_rounds):
        # Thompson-sample directly over the candidate posteriors (the same draw
        # the hot path uses), with the threaded RNG for determinism.
        chosen = None
        best_draw = -1.0
        for arm in arms:
            a, b = strat._get_posterior(bucket, arm).shape()
            x = sample_kumaraswamy(a, b, strat._rng)
            if x > best_draw:
                best_draw, chosen = x, arm
        assert chosen is not None
        picks[chosen] += 1
        reward = 1.0 - norm_cost[chosen]
        regret += best_reward - reward
        if learn:
            win = rng.random() < reward
            strat.update(chosen, 1.0 if win else -1.0, bucket=bucket)

    share = picks[best_arm] / n_rounds
    return best_arm, share, regret


def test_replay_learning_beats_no_learning(corpus: List[CorpusRow]) -> None:
    """The headline backtest: learning concentrates on the cheapest arm.

    With learning ON the bandit must pick the empirically-cheapest arm a far
    larger fraction of the time than the no-learning baseline, AND accrue lower
    cumulative regret against the cost oracle. Deterministic at the fixed seed.
    """
    cost_table = _cost_table(corpus)
    arms = _top_arms_by_count(corpus, 6)
    # Keep only arms we have a cost for; need a non-degenerate spread.
    arms = [a for a in arms if a in cost_table]
    norm = _norm_cost_map(arms, cost_table)
    if max(norm.values()) - min(norm.values()) < 0.2:
        pytest.skip("corpus cost spread too small for a meaningful backtest")

    _, share_learn, regret_learn = _replay_selection_share(
        arms, norm, seed=1234, n_rounds=600, learn=True
    )
    _, share_base, regret_base = _replay_selection_share(
        arms, norm, seed=1234, n_rounds=600, learn=False
    )

    # Learning concentrates probability mass on the best (cheapest) arm.
    assert share_learn > share_base
    # And it does so decisively (Thompson should exploit hard once converged).
    # Empirically 0.84-0.92 across seeds on the preserved corpus; assert > 0.6.
    assert share_learn > 0.6
    # Lower cumulative regret vs. the cost oracle than the no-learning control.
    assert regret_learn < regret_base


def test_replay_is_seed_deterministic(corpus: List[CorpusRow]) -> None:
    cost_table = _cost_table(corpus)
    arms = [a for a in _top_arms_by_count(corpus, 5) if a in cost_table]
    norm = _norm_cost_map(arms, cost_table)
    r1 = _replay_selection_share(arms, norm, seed=7, n_rounds=300, learn=True)
    r2 = _replay_selection_share(arms, norm, seed=7, n_rounds=300, learn=True)
    assert r1 == r2


def test_replay_posterior_orders_by_reward(corpus: List[CorpusRow]) -> None:
    """After replay, the cheapest arm's posterior mean ranks above the dearest."""
    cost_table = _cost_table(corpus)
    arms = [a for a in _top_arms_by_count(corpus, 5) if a in cost_table]
    norm = _norm_cost_map(arms, cost_table)
    if max(norm.values()) - min(norm.values()) < 0.2:
        pytest.skip("corpus cost spread too small")

    strat = KumaraswamyThompsonStrategy(
        seed=99, w_quality=1.0, w_cost=0.0, w_latency=0.0, cold_start_kappa=0.0
    )
    rng = random.Random(99)
    for _ in range(800):
        chosen = None
        best_draw = -1.0
        for arm in arms:
            a, b = strat._get_posterior("b", arm).shape()
            x = sample_kumaraswamy(a, b, strat._rng)
            if x > best_draw:
                best_draw, chosen = x, arm
        reward = 1.0 - norm[chosen]
        win = rng.random() < reward
        strat.update(chosen, 1.0 if win else -1.0, bucket="b")

    cheapest = min(arms, key=lambda a: norm[a])
    dearest = max(arms, key=lambda a: norm[a])
    mean_cheap = strat._backend.get("b", cheapest).mean()
    mean_dear = strat._backend.get("b", dearest).mean()
    assert mean_cheap > mean_dear


# ---------------------------------------------------------------------------
# 4. Distribution goodness-of-fit (sampled mean ~= analytic Kumaraswamy mean)
# ---------------------------------------------------------------------------


def _analytic_kumaraswamy_mean(a: float, b: float) -> float:
    """Closed-form ``E[X] = b * B(1 + 1/a, b) = b*Γ(1+1/a)*Γ(b)/Γ(1+1/a+b)``."""
    return b * math.gamma(1.0 + 1.0 / a) * math.gamma(b) / math.gamma(1.0 + 1.0 / a + b)


@pytest.mark.parametrize(
    "a,b",
    [
        (1.0, 1.0),  # Uniform — analytic mean 0.5
        (2.0, 3.0),
        (3.0, 2.0),
        (5.0, 5.0),
        (1.0, 4.0),  # == Beta(1, 4) special case
        (4.0, 1.0),  # == Beta(4, 1) special case
        (0.7, 0.7),
    ],
)
def test_sampled_mean_matches_analytic(a: float, b: float) -> None:
    rng = random.Random(20240614)
    n = 40000
    total = 0.0
    for _ in range(n):
        total += sample_kumaraswamy(a, b, rng)
    sampled_mean = total / n
    analytic = _analytic_kumaraswamy_mean(a, b)
    # Monte-Carlo std-error ~ 0.3/sqrt(40k) ~ 1.5e-3; allow a generous 2e-2 band
    # so the test is robust at the fixed seed without flaking.
    assert abs(sampled_mean - analytic) < 2e-2, (
        f"K({a},{b}): sampled {sampled_mean:.4f} vs analytic {analytic:.4f}"
    )


def test_sampled_mean_is_seed_reproducible() -> None:
    def mc_mean(seed: int) -> float:
        rng = random.Random(seed)
        return sum(sample_kumaraswamy(2.0, 5.0, rng) for _ in range(5000)) / 5000

    assert mc_mean(11) == mc_mean(11)
    assert mc_mean(11) != mc_mean(12)


def test_uniform_special_case_mean_half() -> None:
    # K(1,1) == Uniform(0,1): sampled mean ~= 0.5.
    rng = random.Random(5)
    n = 50000
    m = sum(sample_kumaraswamy(1.0, 1.0, rng) for _ in range(n)) / n
    assert m == pytest.approx(0.5, abs=1.5e-2)


# ---------------------------------------------------------------------------
# 5. Quantile boundary correctness + reference-CDF match at known points
# ---------------------------------------------------------------------------


def _reference_inverse_cdf(u: float, a: float, b: float) -> float:
    """Independent reference inverse-CDF of Kumaraswamy via the algebraic form.

    ``Q(u) = (1 - (1 - u)^(1/b))^(1/a)``. Used only at well-conditioned points
    where it does not underflow, as an independent cross-check of the
    log-space-stabilized production quantile.
    """
    return (1.0 - (1.0 - u) ** (1.0 / b)) ** (1.0 / a)


@pytest.mark.parametrize("a,b", [(2.0, 3.0), (1.0, 1.0), (5.0, 2.0), (0.5, 4.0)])
def test_quantile_q0_approaches_zero(a: float, b: float) -> None:
    # Q(0) -> 0. The production quantile clamps u into the open interior
    # (u_min = 1e-12) so a degenerate posterior never lands on the exact 0/1
    # boundary; the residual is then raised to 1/a, so for large a the clamped
    # value is pulled up to a few x 1e-3. The invariant is: strictly positive
    # and far below the mode — assert < 1e-2.
    x = kumaraswamy_quantile(0.0, a, b)
    assert 0.0 < x < 1e-2


@pytest.mark.parametrize("a,b", [(2.0, 3.0), (1.0, 1.0), (5.0, 2.0), (0.5, 4.0)])
def test_quantile_q1_approaches_one(a: float, b: float) -> None:
    # Q(1) -> 1 (within the interior clamp). For a=1, b<=1 the clamped residual
    # rounds up to exactly 1.0 in float64, which is the correct limit, so allow
    # the closed endpoint here.
    x = kumaraswamy_quantile(1.0, a, b)
    assert 1.0 - 1e-2 < x <= 1.0


def test_quantile_monotone_strict_over_grid() -> None:
    a, b = 2.5, 4.0
    grid = [i / 200.0 for i in range(1, 200)]
    xs = [kumaraswamy_quantile(u, a, b) for u in grid]
    assert all(xs[i] < xs[i + 1] for i in range(len(xs) - 1))


@pytest.mark.parametrize("a,b", [(2.0, 3.0), (3.0, 2.0), (1.5, 1.5), (4.0, 6.0)])
@pytest.mark.parametrize("u", [0.1, 0.25, 0.5, 0.75, 0.9])
def test_quantile_matches_reference_at_known_points(
    a: float, b: float, u: float
) -> None:
    # The production (log-space) quantile must agree with the independent
    # algebraic reference inverse-CDF in the well-conditioned interior band.
    prod = kumaraswamy_quantile(u, a, b)
    ref = _reference_inverse_cdf(u, a, b)
    assert prod == pytest.approx(ref, abs=1e-9)


def test_quantile_uniform_identity_at_known_points() -> None:
    # K(1,1) is Uniform: Q(u) == u exactly (up to fp) for interior u.
    for u in (0.1, 0.25, 0.5, 0.75, 0.9):
        assert kumaraswamy_quantile(u, 1.0, 1.0) == pytest.approx(u, abs=1e-12)


# ---------------------------------------------------------------------------
# 6. No external deps: the loader is pure-stdlib, the bandit is in-memory
# ---------------------------------------------------------------------------


def test_backtest_uses_no_numpy_or_db() -> None:
    # The whole replay path runs with the in-memory backend and stdlib only.
    strat = KumaraswamyThompsonStrategy(
        seed=1,
        backend=InMemoryPosteriorBackend(),
        w_quality=1.0,
        w_cost=0.0,
        w_latency=0.0,
        cold_start_kappa=0.0,
    )
    assert isinstance(strat._backend, InMemoryPosteriorBackend)
    # A tiny self-contained replay with synthetic arms (no corpus needed).
    arms = ["bedrock/cheap", "bedrock/dear"]
    norm = {"bedrock/cheap": 0.0, "bedrock/dear": 1.0}

    def _gen() -> Iterator[Tuple[str, float]]:
        rng = random.Random(3)
        for _ in range(400):
            chosen, best = None, -1.0
            for arm in arms:
                a, b = strat._get_posterior("b", arm).shape()
                x = sample_kumaraswamy(a, b, strat._rng)
                if x > best:
                    best, chosen = x, arm
            reward = 1.0 - norm[chosen]
            # win/loss as the [-1, 1] feedback contract score.
            yield chosen, (1.0 if rng.random() < reward else -1.0)

    for arm, score in _gen():
        strat.update(arm, score, bucket="b")

    mean_cheap = strat._backend.get("b", "bedrock/cheap").mean()
    mean_dear = strat._backend.get("b", "bedrock/dear").mean()
    assert mean_cheap > mean_dear
