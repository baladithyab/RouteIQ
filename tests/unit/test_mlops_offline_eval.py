"""Unit tests for the offline eval harness over a golden dataset
(RouteIQ-8d24).

Verifies the harness replays a versioned golden dataset (request ->
expected-quality) through a scorer and emits a per-strategy comparison report --
all offline, no live traffic, no cost (the default scorer is a deterministic
oracle). Also exercises the judge scorer with a MOCK ``EvalJudge`` (cred-free)
and loads the bundled versioned fixture.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from litellm_llmrouter.mlops.offline_eval import (
    GoldenCase,
    GoldenDataset,
    OfflineEvalHarness,
    default_golden_dataset_path,
    expected_quality_scorer,
    judge_scorer,
    load_golden_dataset,
)


def _dataset() -> GoldenDataset:
    return GoldenDataset(
        version="1",
        name="test-golden",
        cases=[
            GoldenCase(
                case_id="a1",
                messages=[{"role": "user", "content": "q1"}],
                strategy="knn",
                expected_quality=0.9,
                model="gpt-4o",
            ),
            GoldenCase(
                case_id="a2",
                messages=[{"role": "user", "content": "q2"}],
                strategy="knn",
                expected_quality=0.8,
                model="gpt-4o",
            ),
            GoldenCase(
                case_id="b1",
                messages=[{"role": "user", "content": "q3"}],
                strategy="centroid",
                expected_quality=0.6,
                model="gpt-4o-mini",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Default oracle scorer (offline, cred-free)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_harness_scores_golden_set_per_strategy():
    harness = OfflineEvalHarness(tolerance=0.1)
    report = await harness.run(_dataset())  # default = expected_quality_scorer

    assert report.dataset_name == "test-golden"
    assert report.dataset_version == "1"
    assert report.total_cases == 3
    assert report.scored_cases == 3

    by_strat = report.by_strategy()
    assert set(by_strat) == {"knn", "centroid"}

    knn = by_strat["knn"]
    assert knn.n == 2
    # Oracle scorer => observed == expected, so mean_observed == mean_expected.
    assert knn.mean_observed == pytest.approx(0.85)
    assert knn.mean_expected == pytest.approx(0.85)
    assert knn.delta == pytest.approx(0.0)
    assert knn.passed is True

    centroid = by_strat["centroid"]
    assert centroid.n == 1
    assert centroid.mean_observed == pytest.approx(0.6)

    assert report.all_passed is True


@pytest.mark.asyncio
async def test_harness_flags_regression_beyond_tolerance():
    # Scorer that under-scores every case by 0.3 -> exceeds tolerance.
    def under_scorer(case: GoldenCase) -> float:
        return max(0.0, case.expected_quality - 0.3)

    harness = OfflineEvalHarness(tolerance=0.1)
    report = await harness.run(_dataset(), scorer=under_scorer)

    knn = report.by_strategy()["knn"]
    assert knn.delta == pytest.approx(-0.3, abs=1e-6)
    assert knn.passed is False
    assert report.all_passed is False


def test_expected_quality_scorer_returns_expected():
    case = GoldenCase(case_id="x", messages=[], strategy="s", expected_quality=0.77)
    assert expected_quality_scorer(case) == pytest.approx(0.77)


# ---------------------------------------------------------------------------
# Judge scorer (LLM-as-judge offline replay, cred-free via mock judge)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_judge_scorer_uses_mock_judge():
    # Mock EvalJudge.evaluate -> normalized metric dict; scorer averages it.
    judge = AsyncMock()
    judge.evaluate = AsyncMock(
        return_value={"relevance": 0.8, "helpfulness": 1.0, "coherence": 0.6}
    )
    scorer = judge_scorer(judge)

    harness = OfflineEvalHarness(tolerance=0.5)
    report = await harness.run(_dataset(), scorer=scorer)

    # Every case scored to the mean of the mock metrics = 0.8.
    knn = report.by_strategy()["knn"]
    assert knn.mean_observed == pytest.approx(0.8)
    # The judge was invoked once per case (3 cases).
    assert judge.evaluate.await_count == 3


@pytest.mark.asyncio
async def test_judge_scorer_empty_scores_yield_zero():
    judge = AsyncMock()
    judge.evaluate = AsyncMock(return_value={})  # judge failure -> {}
    scorer = judge_scorer(judge)
    harness = OfflineEvalHarness()
    report = await harness.run(_dataset(), scorer=scorer)
    assert report.by_strategy()["knn"].mean_observed == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Bundled versioned fixture
# ---------------------------------------------------------------------------


def test_default_fixture_path_exists():
    import os

    path = default_golden_dataset_path()
    assert path.endswith("config/golden_eval/routing_golden_v1.json")
    assert os.path.exists(path)


@pytest.mark.asyncio
async def test_load_and_replay_bundled_fixture():
    dataset = load_golden_dataset()
    assert dataset.version == "1"
    assert dataset.name == "routeiq-routing-golden-v1"
    assert len(dataset.cases) >= 3

    harness = OfflineEvalHarness(tolerance=0.1)
    report = await harness.run(dataset)
    # Oracle replay over the golden set passes by construction (regression net).
    assert report.scored_cases == len(dataset.cases)
    assert report.all_passed is True
    # Per-strategy comparison emitted for each distinct strategy in the fixture.
    strategies = {c.strategy for c in dataset.cases}
    assert set(report.by_strategy()) == strategies
