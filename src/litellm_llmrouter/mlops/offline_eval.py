"""
Offline Eval Harness over a Golden Dataset (RouteIQ-8d24)
=========================================================

Replays a VERSIONED golden / held-out dataset (request -> expected-quality)
through a scorer and reports per-strategy quality OFFLINE -- no live traffic.
This is the regression-safety net for routing changes: run the harness before
shipping a new strategy/version and compare per-strategy quality against the
golden expectations, with zero production calls and zero cost (the default
scorer needs no LLM).

Dataset schema (``GoldenDataset``)::

    {
      "version": "1",
      "name": "routeiq-routing-golden-v1",
      "cases": [
        {
          "case_id": "math-001",
          "messages": [{"role": "user", "content": "What is 17 * 23?"}],
          "strategy": "llmrouter-knn",          # which strategy served it
          "model": "gpt-4o",                     # the model picked (optional)
          "expected_quality": 0.9,               # golden target in [0,1]
          "tier": "complex"                      # optional bucket
        },
        ...
      ]
    }

A *scorer* maps one ``GoldenCase`` to an OBSERVED quality in ``[0, 1]``. Two are
built in:

  * :func:`expected_quality_scorer` -- returns the case's ``expected_quality``;
    a deterministic, cred-free oracle used to validate the harness mechanics and
    as the offline replay baseline.
  * :func:`judge_scorer` -- wraps the eval pipeline's ``EvalJudge`` (LLM-as-judge)
    for a LIVE offline replay; this is the one that costs judge calls and is
    used when an operator wants a real re-score of held-out responses.

The harness aggregates the observed quality PER STRATEGY and emits a
:class:`OfflineEvalReport` with the per-strategy mean observed vs mean expected,
a delta, and a pass flag (within ``tolerance``). Pure stdlib (no numpy); the
default scorer makes the whole thing runnable in a unit test with no network.

Settings-gated under ``settings.mlops.offline_eval`` (default off), but the
harness is ALSO directly invokable (tests / a CLI / a batch job) without the
flag -- the flag only governs any startup auto-run, not direct use.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

__all__ = [
    "GoldenCase",
    "GoldenDataset",
    "StrategyComparison",
    "OfflineEvalReport",
    "OfflineEvalHarness",
    "expected_quality_scorer",
    "judge_scorer",
    "load_golden_dataset",
    "default_golden_dataset_path",
]

# A scorer maps a GoldenCase to an observed quality in [0, 1]. May be sync or
# async (the judge scorer is async); the harness awaits coroutine results.
Scorer = Callable[["GoldenCase"], Union[float, Awaitable[float]]]


@dataclass
class GoldenCase:
    """One golden / held-out request with its expected quality."""

    case_id: str
    messages: List[Dict[str, Any]]
    strategy: str
    expected_quality: float
    model: str = ""
    tier: str = ""
    response_content: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GoldenCase":
        return cls(
            case_id=str(d.get("case_id", "")),
            messages=list(d.get("messages", []) or []),
            strategy=str(d.get("strategy", "")),
            expected_quality=float(d.get("expected_quality", 0.0)),
            model=str(d.get("model", "")),
            tier=str(d.get("tier", "")),
            response_content=str(d.get("response_content", "")),
        )


@dataclass
class GoldenDataset:
    """A versioned collection of golden cases."""

    version: str
    name: str
    cases: List[GoldenCase] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GoldenDataset":
        return cls(
            version=str(d.get("version", "0")),
            name=str(d.get("name", "unnamed")),
            cases=[GoldenCase.from_dict(c) for c in d.get("cases", []) or []],
        )


@dataclass
class StrategyComparison:
    """Per-strategy offline comparison (observed vs golden expected)."""

    strategy: str
    n: int
    mean_observed: float
    mean_expected: float
    delta: float
    """``mean_observed - mean_expected`` (negative => below golden)."""

    passed: bool
    """True iff ``abs(delta) <= tolerance``."""


@dataclass
class OfflineEvalReport:
    """Outcome of one offline replay over a golden dataset."""

    dataset_name: str
    dataset_version: str
    total_cases: int
    scored_cases: int
    comparisons: List[StrategyComparison] = field(default_factory=list)
    all_passed: bool = True

    def by_strategy(self) -> Dict[str, StrategyComparison]:
        return {c.strategy: c for c in self.comparisons}


def expected_quality_scorer(case: GoldenCase) -> float:
    """Deterministic, cred-free oracle: return the case's expected quality.

    Used to validate the harness mechanics offline (no network, no cost) and as
    the replay baseline. By construction the per-strategy delta is 0.0, so a
    test can assert the aggregation/pass logic exactly.
    """
    return float(case.expected_quality)


def judge_scorer(judge: Any) -> Scorer:
    """Build an async scorer that re-scores a case via the eval ``EvalJudge``.

    Returns an async callable suitable as a harness scorer. Reconstructs a
    minimal ``EvalSample`` from the case and averages the judge's metric scores
    into one quality scalar. Used for a LIVE offline replay; it costs one judge
    call per case (mock the judge in unit tests).
    """

    async def _score(case: GoldenCase) -> float:
        from litellm_llmrouter.eval_pipeline import EvalSample

        sample = EvalSample(
            sample_id=case.case_id,
            timestamp=0.0,
            model=case.model or "unknown",
            strategy=case.strategy,
            tier=case.tier,
            messages=case.messages,
            response_content=case.response_content,
        )
        scores = await judge.evaluate(sample)
        if not scores:
            return 0.0
        return float(sum(scores.values()) / len(scores))

    return _score


class OfflineEvalHarness:
    """Replays a golden dataset through a scorer, aggregates per-strategy.

    Args:
        tolerance: Max ``abs(observed - expected)`` per strategy to count as a
            pass (default 0.1).
    """

    def __init__(self, *, tolerance: float = 0.1) -> None:
        self.tolerance = tolerance

    async def run(
        self,
        dataset: GoldenDataset,
        scorer: Optional[Scorer] = None,
    ) -> OfflineEvalReport:
        """Score every case and aggregate observed quality per strategy.

        ``scorer`` defaults to :func:`expected_quality_scorer` (offline, cred-free).
        Each case is scored; a scorer error contributes 0.0 for that case (so a
        flaky judge degrades the strategy's mean rather than aborting the run).
        """
        score_fn: Scorer = scorer or expected_quality_scorer

        observed: Dict[str, List[float]] = {}
        expected: Dict[str, List[float]] = {}
        scored = 0
        for case in dataset.cases:
            try:
                value: Any = score_fn(case)
                if _is_awaitable(value):
                    value = await value
                obs = float(value)
            except Exception as e:  # pragma: no cover - scorer failures degrade
                logger.debug("Offline scorer failed for %s: %s", case.case_id, e)
                obs = 0.0
            observed.setdefault(case.strategy, []).append(obs)
            expected.setdefault(case.strategy, []).append(float(case.expected_quality))
            scored += 1

        comparisons: List[StrategyComparison] = []
        all_passed = True
        for strategy in sorted(observed):
            obs_list = observed[strategy]
            exp_list = expected[strategy]
            mean_obs = sum(obs_list) / len(obs_list)
            mean_exp = sum(exp_list) / len(exp_list)
            delta = mean_obs - mean_exp
            passed = abs(delta) <= self.tolerance
            all_passed = all_passed and passed
            comparisons.append(
                StrategyComparison(
                    strategy=strategy,
                    n=len(obs_list),
                    mean_observed=mean_obs,
                    mean_expected=mean_exp,
                    delta=delta,
                    passed=passed,
                )
            )

        return OfflineEvalReport(
            dataset_name=dataset.name,
            dataset_version=dataset.version,
            total_cases=len(dataset.cases),
            scored_cases=scored,
            comparisons=comparisons,
            all_passed=all_passed if comparisons else True,
        )


def _is_awaitable(value: Any) -> bool:
    return hasattr(value, "__await__")


def default_golden_dataset_path() -> str:
    """Path to the bundled default golden dataset fixture (config/golden_eval/)."""
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    return os.path.join(here, "config", "golden_eval", "routing_golden_v1.json")


def load_golden_dataset(path: Optional[str] = None) -> GoldenDataset:
    """Load a versioned golden dataset from JSON.

    ``path`` defaults to ``settings.mlops.offline_eval.golden_dataset_path`` when
    set, else the bundled fixture under ``config/golden_eval/``.
    """
    resolved = path or _settings_golden_path() or default_golden_dataset_path()
    with open(resolved, "r", encoding="utf-8") as f:
        return GoldenDataset.from_dict(json.load(f))


def _settings_golden_path() -> Optional[str]:
    """Read ``settings.mlops.offline_eval.golden_dataset_path`` (None on failure)."""
    try:
        from litellm_llmrouter.settings import get_settings

        mlops = getattr(get_settings(), "mlops", None)
        oe = getattr(mlops, "offline_eval", None) if mlops is not None else None
        p = getattr(oe, "golden_dataset_path", "") if oe is not None else ""
        return p or None
    except Exception:  # pragma: no cover - defensive
        return None
