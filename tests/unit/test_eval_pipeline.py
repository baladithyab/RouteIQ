"""Unit tests for the evaluation feedback loop pipeline."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm_llmrouter.eval_pipeline import (
    EvalJudge,
    EvalMetric,
    EvalPipeline,
    EvalSample,
    ModelQualityTracker,
    get_eval_pipeline,
    reset_eval_pipeline,
)


@pytest.fixture(autouse=True)
def _reset_pipeline():
    """Reset the eval pipeline singleton between tests."""
    reset_eval_pipeline()
    yield
    reset_eval_pipeline()


def _make_sample(
    model: str = "gpt-4o",
    strategy: str = "llmrouter-knn",
    tier: str = "premium",
    response: str = "Hello world",
    **kwargs,
) -> EvalSample:
    return EvalSample(
        sample_id=f"test-{time.time()}",
        timestamp=time.time(),
        model=model,
        strategy=strategy,
        tier=tier,
        messages=[{"role": "user", "content": "Test prompt"}],
        response_content=response,
        **kwargs,
    )


class TestEvalMetric:
    def test_metric_values(self):
        assert EvalMetric.RELEVANCE == "relevance"
        assert EvalMetric.CORRECTNESS == "correctness"
        assert EvalMetric.HELPFULNESS == "helpfulness"
        assert EvalMetric.COHERENCE == "coherence"
        assert EvalMetric.SAFETY == "safety"
        assert EvalMetric.CUSTOM == "custom"


class TestEvalSample:
    def test_default_fields(self):
        sample = _make_sample()
        assert sample.model == "gpt-4o"
        assert sample.strategy == "llmrouter-knn"
        assert sample.evaluated is False
        assert sample.scores == {}
        assert sample.evaluated_at is None

    def test_with_context(self):
        sample = _make_sample(user_id="user-1", workspace_id="ws-1")
        assert sample.user_id == "user-1"
        assert sample.workspace_id == "ws-1"


class TestModelQualityTracker:
    def test_record_and_get(self):
        tracker = ModelQualityTracker()
        tracker.record("gpt-4o", 0.9)
        tracker.record("gpt-4o", 0.8)
        assert tracker.get_quality("gpt-4o") == pytest.approx(0.85)

    def test_no_data(self):
        tracker = ModelQualityTracker()
        assert tracker.get_quality("nonexistent") is None

    def test_window_size(self):
        tracker = ModelQualityTracker(window_size=3)
        tracker.record("m", 0.1)
        tracker.record("m", 0.2)
        tracker.record("m", 0.3)
        tracker.record("m", 0.9)  # Pushes out 0.1
        assert tracker.get_quality("m") == pytest.approx((0.2 + 0.3 + 0.9) / 3)

    def test_get_all_qualities(self):
        tracker = ModelQualityTracker()
        tracker.record("a", 0.8)
        tracker.record("b", 0.6)
        qualities = tracker.get_all_qualities()
        assert qualities == {"a": 0.8, "b": 0.6}

    def test_get_ranking(self):
        tracker = ModelQualityTracker()
        tracker.record("low", 0.3)
        tracker.record("high", 0.9)
        tracker.record("mid", 0.6)
        ranking = tracker.get_ranking()
        assert ranking[0] == ("high", 0.9)
        assert ranking[-1] == ("low", 0.3)

    def test_get_sample_counts(self):
        tracker = ModelQualityTracker()
        tracker.record("a", 0.8)
        tracker.record("a", 0.9)
        tracker.record("b", 0.6)
        counts = tracker.get_sample_counts()
        assert counts == {"a": 2, "b": 1}


class TestEvalJudge:
    @pytest.mark.asyncio
    async def test_evaluate_success(self):
        judge = EvalJudge("test-model")
        sample = _make_sample()

        mock_resp = MagicMock()
        mock_resp.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"relevance": 4, "helpfulness": 3, "coherence": 5}'
                )
            )
        ]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_completion:
            mock_completion.return_value = mock_resp
            scores = await judge.evaluate(sample)

        assert scores["relevance"] == pytest.approx(0.8)
        assert scores["helpfulness"] == pytest.approx(0.6)
        assert scores["coherence"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_evaluate_failure_returns_empty(self):
        judge = EvalJudge("test-model")
        sample = _make_sample()

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_completion:
            mock_completion.side_effect = Exception("API error")
            scores = await judge.evaluate(sample)

        assert scores == {}

    @pytest.mark.asyncio
    async def test_evaluate_normalizes_scores(self):
        """Scores are clamped to 0.0-1.0."""
        judge = EvalJudge("test-model")
        sample = _make_sample()

        mock_resp = MagicMock()
        mock_resp.choices = [
            MagicMock(message=MagicMock(content='{"relevance": 10, "helpfulness": -1}'))
        ]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_completion:
            mock_completion.return_value = mock_resp
            scores = await judge.evaluate(sample)

        assert scores["relevance"] == 1.0
        assert scores["helpfulness"] == 0.0

    @pytest.mark.asyncio
    async def test_evaluate_batch(self):
        judge = EvalJudge("test-model")
        samples = [_make_sample(model=f"m{i}") for i in range(3)]

        mock_resp = MagicMock()
        mock_resp.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"relevance": 4, "helpfulness": 4, "coherence": 4}'
                )
            )
        ]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_completion:
            mock_completion.return_value = mock_resp
            results = await judge.evaluate_batch(samples)

        assert len(results) == 3
        for r in results:
            assert "relevance" in r

    @pytest.mark.asyncio
    async def test_evaluate_batch_handles_exceptions(self):
        judge = EvalJudge("test-model")
        samples = [_make_sample()]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_completion:
            mock_completion.side_effect = Exception("fail")
            results = await judge.evaluate_batch(samples)

        assert len(results) == 1
        assert results[0] == {}


class TestEvalPipeline:
    def test_should_sample_rate(self):
        pipeline = EvalPipeline(sample_rate=1.0)
        # With rate 1.0, should always sample
        assert all(pipeline.should_sample() for _ in range(10))

        pipeline2 = EvalPipeline(sample_rate=0.0)
        assert not any(pipeline2.should_sample() for _ in range(10))

    def test_collect(self):
        pipeline = EvalPipeline()
        sample = _make_sample()
        pipeline.collect(sample)
        assert len(pipeline._pending_samples) == 1
        assert pipeline._total_collected == 1

    def test_collect_bounded(self):
        pipeline = EvalPipeline()
        pipeline._max_pending = 3
        for i in range(5):
            pipeline.collect(_make_sample(model=f"m{i}"))
        assert len(pipeline._pending_samples) == 3
        # Oldest dropped
        assert pipeline._pending_samples[0].model == "m2"

    @pytest.mark.asyncio
    async def test_run_evaluation_batch_empty(self):
        pipeline = EvalPipeline()
        count = await pipeline.run_evaluation_batch()
        assert count == 0

    @pytest.mark.asyncio
    async def test_run_evaluation_batch(self):
        pipeline = EvalPipeline(batch_size=2)

        for i in range(3):
            pipeline.collect(_make_sample(model=f"m{i}"))

        mock_scores = {"relevance": 0.8, "helpfulness": 0.7, "coherence": 0.9}
        with patch.object(
            pipeline._judge,
            "evaluate_batch",
            new_callable=AsyncMock,
        ) as mock_eval:
            mock_eval.return_value = [mock_scores, mock_scores]
            count = await pipeline.run_evaluation_batch()

        assert count == 2
        assert len(pipeline._pending_samples) == 1  # 1 remaining
        assert len(pipeline._evaluated_samples) == 2
        assert pipeline._total_evaluated == 2

    @pytest.mark.asyncio
    async def test_push_feedback_no_data(self):
        pipeline = EvalPipeline()
        result = await pipeline.push_feedback()
        assert result["updated"] is False

    @pytest.mark.asyncio
    async def test_push_feedback_updates_router(self):
        pipeline = EvalPipeline()
        pipeline._tracker.record("gpt-4o", 0.9)

        mock_router = MagicMock()
        mock_router.update_quality_bias = MagicMock()

        with patch(
            "litellm_llmrouter.personalized_routing.get_personalized_router",
            return_value=mock_router,
        ):
            result = await pipeline.push_feedback()

        assert result["updated"] is True
        assert result["model_count"] == 1
        mock_router.update_quality_bias.assert_called_once_with("gpt-4o", 0.9)

    @pytest.mark.asyncio
    async def test_push_feedback_custom_callback(self):
        callback = MagicMock()
        pipeline = EvalPipeline(feedback_callbacks=[callback])
        pipeline._tracker.record("gpt-4o", 0.9)

        with patch(
            "litellm_llmrouter.personalized_routing.get_personalized_router",
            return_value=None,
        ):
            result = await pipeline.push_feedback()

        callback.assert_called_once()
        assert result["updated"] is True

    @pytest.mark.asyncio
    async def test_push_feedback_async_callback(self):
        callback = AsyncMock()
        pipeline = EvalPipeline(feedback_callbacks=[callback])
        pipeline._tracker.record("gpt-4o", 0.9)

        with patch(
            "litellm_llmrouter.personalized_routing.get_personalized_router",
            return_value=None,
        ):
            await pipeline.push_feedback()

        callback.assert_awaited_once()

    def test_get_stats(self):
        pipeline = EvalPipeline(sample_rate=0.2, batch_size=5, feedback_interval=60)
        stats = pipeline.get_stats()
        assert stats["pending_samples"] == 0
        assert stats["evaluated_samples"] == 0
        assert stats["running"] is False
        assert stats["sample_rate"] == 0.2
        assert stats["batch_size"] == 5

    def test_get_recent_samples(self):
        pipeline = EvalPipeline()

        for i in range(5):
            s = _make_sample(model=f"m{i}")
            if i < 3:
                s.evaluated = True
            pipeline._evaluated_samples.append(
                s
            ) if i < 3 else pipeline._pending_samples.append(s)

        # All samples
        samples = pipeline.get_recent_samples(limit=10)
        assert len(samples) == 5

        # Evaluated only
        samples = pipeline.get_recent_samples(evaluated_only=True)
        assert len(samples) == 3

        # Filter by model
        samples = pipeline.get_recent_samples(model="m0")
        assert len(samples) == 1

    @pytest.mark.asyncio
    async def test_start_stop_background_loop(self):
        pipeline = EvalPipeline()
        await pipeline.start_background_loop()
        assert pipeline._running is True
        assert pipeline._task is not None

        await pipeline.stop()
        assert pipeline._running is False
        assert pipeline._task is None


class TestSingleton:
    def test_get_eval_pipeline_disabled_by_default(self):
        with patch.dict("os.environ", {}, clear=False):
            pipeline = get_eval_pipeline()
            assert pipeline is None

    def test_get_eval_pipeline_enabled(self):
        with patch.dict(
            "os.environ",
            {"ROUTEIQ_EVAL_PIPELINE": "true"},
            clear=False,
        ):
            pipeline = get_eval_pipeline()
            assert pipeline is not None
            assert isinstance(pipeline, EvalPipeline)

    def test_reset_eval_pipeline(self):
        with patch.dict(
            "os.environ",
            {"ROUTEIQ_EVAL_PIPELINE": "true"},
            clear=False,
        ):
            pipeline = get_eval_pipeline()
            assert pipeline is not None
            reset_eval_pipeline()
            # After reset and with env still set, should create new
            pipeline2 = get_eval_pipeline()
            assert pipeline2 is not pipeline
