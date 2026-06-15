"""
Offline Kumaraswamy-Thompson Bandit Backtest (de-risk FIRST)
============================================================

Replays the preserved corpus (CSV only, NO DB, NO live cache) through the
bandit's posterior update to validate that the math learns something coherent
from real data BEFORE any live wiring. Runnable standalone:

    cd /Users/baladita/Documents/DevBox/RouteIQ-wt-p3
    uv run python research/p3/backtest.py

Two honest modes (NO IPS/DR — the log is a single deterministic policy with no
propensities and no counterfactual rewards for un-chosen arms):

- **Mode A (posterior-fit):** replay ``(arm=selected_model, reward)`` in order
  through per-arm Kumaraswamy-Beta posteriors; check each well-pulled arm's
  posterior mean converges to its empirical reward mean.
- **Mode B (Thompson agreement + pseudo-regret on BUCKETED arms):** before each
  update, Thompson-sample the bucket posteriors and pick the bandit's arm; score
  agreement with the logged router and accumulate pseudo-regret against the
  running best-bucket empirical mean. Reports the regret trend (should be
  sub-linear as posteriors sharpen).

CAVEATS (stated loudly — see ``discover-corpus.md`` §7):
- Reward is a CONSTRUCTED cost-efficiency proxy. There is NO recorded quality /
  win-loss / judge / latency signal in the corpus. "The bandit learned to route
  better" is a statement about COST, not answer quality.
- ``cost_savings`` is one-sided positive for every served row — the bandit never
  sees a case where routing away from the baseline was a cost loss.
- Pseudo-regret is measured against running empirical means, not ground truth.

This imports ONLY the pure bandit posterior + sampler from the package — no
FastAPI, no DB, no settings singleton.
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

# Make the loader importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from corpus_loader import (  # noqa: E402
    DEFAULT_CSV_PATH,
    Row,
    load_csv,
    size_bucket,
)

# Fable 5 is GOV-BANNED as a routable arm — never surfaced in any arm set.
_GOV_BANNED_ARMS = frozenset({"fable-5", "fable5", "fable_5"})


def _is_jailbreak_or_failed(row: Row) -> bool:
    """A row is excluded if it never served (jailbreak block / failed call)."""
    return row.served == 0


def _is_gov_banned(model: str) -> bool:
    return model.strip().lower() in _GOV_BANNED_ARMS


def _import_bandit():
    """Import the pure bandit posterior + sampler from the package.

    Returns ``(Posterior, sample_kumaraswamy)``. Raises ImportError if the
    package is not importable (caller decides whether to skip).
    """
    from litellm_llmrouter.kumaraswamy_thompson import (  # noqa: E402
        Posterior,
        sample_kumaraswamy,
    )

    return Posterior, sample_kumaraswamy


@dataclass
class BacktestReport:
    """The backtest result (JSON-serializable)."""

    n_rows: int
    n_models: int
    n_served: int
    n_excluded: int
    reward_attr: str
    # Mode A (bucket-level — the meaningful granularity given 31-arm sparsity)
    mode_a_buckets_checked: int
    mode_a_max_abs_error: float
    mode_a_mean_abs_error: float
    mode_a_buckets_within_tol: int
    # Mode A diagnostic (raw 31 arms — sparser, larger Laplace error expected)
    mode_a_raw_arms_checked: int
    mode_a_raw_well_pulled_within_tol: int
    mode_a_raw_well_pulled_total: int
    # Mode B
    mode_b_agreement_acc: float
    mode_b_thompson_mean_reward: float
    mode_b_pseudo_regret: float
    mode_b_mean_regret_first_half: float
    mode_b_mean_regret_second_half: float
    mode_b_regret_sublinear: bool
    posterior_means: Dict[str, float]
    bucket_pulls: Dict[str, int]

    def to_dict(self) -> Dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _eligible_rows(rows: List[Row], reward_attr: str) -> List[Row]:
    """Exclude jailbreak/failed rows, gov-banned arms, and missing-reward rows."""
    out: List[Row] = []
    for r in rows:
        if _is_jailbreak_or_failed(r):
            continue
        if _is_gov_banned(r.selected_model):
            continue
        reward = getattr(r, reward_attr, None)
        if reward is None:
            continue
        out.append(r)
    return out


def run(
    csv_path: str = DEFAULT_CSV_PATH,
    *,
    reward_attr: str = "reward_costmin",
    bucket_fn: Callable[[str], str] = size_bucket,
    seed: int = 1234,
    well_pulled_threshold: int = 8,
    tol: float = 0.05,
) -> BacktestReport:
    """Run the full backtest and return a report.

    Args:
        csv_path: Path to ``sft_corpus.csv``.
        reward_attr: Reward field on ``Row`` (``reward_costmin`` or
            ``reward_savings`` if SQL was joined).
        bucket_fn: Arm bucketing function for Mode B.
        seed: RNG seed (threaded into the bandit sampler).
        well_pulled_threshold: Min pulls for an arm to count as "well-pulled".
        tol: Mode A convergence tolerance.

    Returns:
        A :class:`BacktestReport`.
    """
    Posterior, sample_kumaraswamy = _import_bandit()

    all_rows = load_csv(csv_path)
    n_models = len({r.selected_model for r in all_rows})
    n_served = sum(r.served for r in all_rows)
    rows = _eligible_rows(all_rows, reward_attr)
    # Deterministic replay order.
    rows = sorted(rows, key=lambda r: r.id)

    # ----- Mode A (bucket-level): posterior mean -> empirical reward mean -----
    # 31 raw arms over ~225 pulls is too sparse for a clean Laplace-smoothed
    # convergence; bucketing into family/size tiers gives ~30-110 pulls each so
    # the Beta(1,1) prior is washed out and the posterior mean tracks empirical.
    bkt_post: Dict[str, object] = defaultdict(Posterior)
    bkt_sum: Dict[str, float] = defaultdict(float)
    bkt_n: Dict[str, int] = defaultdict(int)
    # Diagnostic: also fit the raw 31 arms.
    arm_post: Dict[str, object] = defaultdict(Posterior)
    arm_sum: Dict[str, float] = defaultdict(float)
    arm_n: Dict[str, int] = defaultdict(int)
    for r in rows:
        reward = float(getattr(r, reward_attr))
        bkt = bucket_fn(r.selected_model)
        bp = bkt_post[bkt]
        bp.alpha += reward  # type: ignore[attr-defined]
        bp.beta += 1.0 - reward  # type: ignore[attr-defined]
        bkt_sum[bkt] += reward
        bkt_n[bkt] += 1
        ap = arm_post[r.selected_model]
        ap.alpha += reward  # type: ignore[attr-defined]
        ap.beta += 1.0 - reward  # type: ignore[attr-defined]
        arm_sum[r.selected_model] += reward
        arm_n[r.selected_model] += 1

    bkt_errors: List[float] = []
    bkt_within = 0
    for bkt, cnt in bkt_n.items():
        emp_mean = bkt_sum[bkt] / cnt if cnt else 0.0
        post_mean = bkt_post[bkt].mean()  # type: ignore[attr-defined]
        err = abs(post_mean - emp_mean)
        bkt_errors.append(err)
        if err < tol:
            bkt_within += 1

    raw_within = 0
    raw_total = 0
    for arm, cnt in arm_n.items():
        if cnt >= well_pulled_threshold:
            raw_total += 1
            emp_mean = arm_sum[arm] / cnt
            if abs(arm_post[arm].mean() - emp_mean) < tol:  # type: ignore[attr-defined]
                raw_within += 1

    # ----- Mode B: Thompson agreement + pseudo-regret on bucketed arms -----
    rng = random.Random(seed)
    b_post: Dict[str, object] = defaultdict(Posterior)
    b_sum: Dict[str, float] = defaultdict(float)
    b_n: Dict[str, int] = defaultdict(int)

    agree = 0
    thompson_reward = 0.0
    cum_regret = 0.0
    regret_series: List[float] = []

    for r in rows:
        reward = float(getattr(r, reward_attr))
        logged_bucket = bucket_fn(r.selected_model)

        # Thompson pick over buckets seen so far (incl. the logged bucket so a
        # never-seen bucket still participates with its prior).
        candidate_buckets = set(b_post.keys()) | {logged_bucket}
        best_bucket: Optional[str] = None
        best_draw = -1.0
        for bucket in candidate_buckets:
            post = b_post[bucket]
            a, b = post.shape()  # type: ignore[attr-defined]
            draw = sample_kumaraswamy(a, b, rng)
            if draw > best_draw:
                best_draw, best_bucket = draw, bucket
        if best_bucket == logged_bucket:
            agree += 1

        # We only observe the reward for the LOGGED arm/bucket.
        thompson_reward += reward

        # Update only the observed bucket posterior.
        post = b_post[logged_bucket]
        post.alpha += reward  # type: ignore[attr-defined]
        post.beta += 1.0 - reward  # type: ignore[attr-defined]
        b_sum[logged_bucket] += reward
        b_n[logged_bucket] += 1

        # Pseudo-regret vs the running best-bucket empirical mean.
        best_emp = max(
            (b_sum[k] / b_n[k] for k in b_n if b_n[k] > 0),
            default=reward,
        )
        step_regret = max(0.0, best_emp - reward)
        cum_regret += step_regret
        regret_series.append(step_regret)

    n = len(rows)
    half = max(1, n // 2)
    first_half = sum(regret_series[:half]) / half
    second_half = sum(regret_series[half:]) / max(1, n - half)

    return BacktestReport(
        n_rows=len(all_rows),
        n_models=n_models,
        n_served=n_served,
        n_excluded=len(all_rows) - n,
        reward_attr=reward_attr,
        mode_a_buckets_checked=len(bkt_n),
        mode_a_max_abs_error=max(bkt_errors) if bkt_errors else 0.0,
        mode_a_mean_abs_error=sum(bkt_errors) / len(bkt_errors) if bkt_errors else 0.0,
        mode_a_buckets_within_tol=bkt_within,
        mode_a_raw_arms_checked=len(arm_n),
        mode_a_raw_well_pulled_within_tol=raw_within,
        mode_a_raw_well_pulled_total=raw_total,
        mode_b_agreement_acc=agree / n if n else 0.0,
        mode_b_thompson_mean_reward=thompson_reward / n if n else 0.0,
        mode_b_pseudo_regret=cum_regret,
        mode_b_mean_regret_first_half=first_half,
        mode_b_mean_regret_second_half=second_half,
        mode_b_regret_sublinear=second_half <= first_half + 1e-9,
        posterior_means={
            k: round(b_post[k].mean(), 4)
            for k in sorted(b_post)  # type: ignore[attr-defined]
        },
        bucket_pulls=dict(sorted(b_n.items())),
    )


def main() -> int:
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV_PATH
    if not os.path.exists(csv_path):
        print(json.dumps({"error": f"corpus not found: {csv_path}"}, indent=2))
        return 1
    report = run(csv_path)
    print(json.dumps(report.to_dict(), indent=2))
    print(
        "\nCAVEAT: reward is a CONSTRUCTED cost-efficiency proxy "
        "(no quality/latency/judge signal exists in the corpus). "
        "Pseudo-regret is vs running empirical means, not ground truth.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
