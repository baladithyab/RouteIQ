"""Evaluation Pipeline for RouteIQ Gateway.

Closes the feedback loop: routing decisions → responses → evaluation → routing improvement.

Pipeline stages:
1. COLLECT: Capture routing decisions + responses as evaluation samples
2. EVALUATE: Score samples using LLM-as-judge or custom metrics
3. AGGREGATE: Compute per-model quality scores from evaluations
4. FEEDBACK: Feed scores back to routing strategies (personalized router, quality bias)

This integrates with:
- router_decision_callback.py (captures routing telemetry)
- personalized_routing.py (accepts quality feedback)
- centroid_routing.py (accepts model quality bias updates)
- gateway/plugins/evaluator.py (LLM-as-judge evaluations)

Configuration:
  ROUTEIQ_EVAL_PIPELINE=false (disabled by default)
  ROUTEIQ_EVAL_SAMPLE_RATE=0.1 (evaluate 10% of requests)
  ROUTEIQ_EVAL_JUDGE_MODEL=gpt-4o-mini (model for LLM-as-judge)
  ROUTEIQ_EVAL_BATCH_SIZE=10 (evaluate in batches)
  ROUTEIQ_EVAL_FEEDBACK_INTERVAL=300 (update routing every 5 min)
"""

import asyncio
import inspect
import json
import logging
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


__all__ = [
    "EvalJudge",
    "EvalMetric",
    "EvalPipeline",
    "EvalSample",
    "ModelQualityTracker",
    "get_eval_pipeline",
    "reset_eval_pipeline",
]

logger = logging.getLogger("litellm_llmrouter.eval_pipeline")


class EvalMetric(str, Enum):
    """Built-in evaluation metrics."""

    RELEVANCE = "relevance"  # Is the response relevant to the query?
    CORRECTNESS = "correctness"  # Is the response factually correct?
    HELPFULNESS = "helpfulness"  # Is the response helpful?
    COHERENCE = "coherence"  # Is the response well-structured?
    SAFETY = "safety"  # Is the response safe/appropriate?
    CUSTOM = "custom"  # User-defined metric


@dataclass
class EvalSample:
    """A captured request-response pair for evaluation."""

    sample_id: str
    timestamp: float
    model: str
    strategy: str
    tier: str

    # Request
    messages: List[Dict[str, Any]]
    request_tokens: int = 0

    # Response
    response_content: str = ""
    response_tokens: int = 0
    latency_ms: float = 0.0

    # Context
    user_id: Optional[str] = None
    workspace_id: Optional[str] = None
    prompt_name: Optional[str] = None

    # Evaluation results (filled after evaluation)
    scores: Dict[str, float] = field(default_factory=dict)
    evaluated: bool = False
    evaluated_at: Optional[float] = None


class EvalJudge:
    """LLM-as-judge evaluator for routing quality assessment."""

    JUDGE_PROMPT = (
        "You are an AI response quality evaluator. Rate the following "
        "response on these dimensions (1-5 scale):\n\n"
        "Query: {query}\n"
        "Response: {response}\n"
        "Model used: {model}\n\n"
        "Rate each dimension (1=terrible, 5=excellent):\n"
        "1. relevance: How relevant is the response to the query?\n"
        "2. helpfulness: How helpful and actionable is the response?\n"
        "3. coherence: How well-structured and clear is the response?\n\n"
        'Respond in JSON: {{"relevance": N, "helpfulness": N, "coherence": N}}'
    )

    def __init__(self, judge_model: str = "gpt-4o-mini"):
        self._model = judge_model

    async def evaluate(self, sample: EvalSample) -> Dict[str, float]:
        """Evaluate a single sample using LLM-as-judge.

        Sends the user query and model response to the judge model,
        which returns scores for relevance, helpfulness, and coherence.

        Args:
            sample: The evaluation sample containing messages and response.

        Returns:
            Dictionary of metric names to normalized scores (0.0-1.0).
            Empty dict if evaluation fails.
        """
        import litellm

        query = ""
        for msg in sample.messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    query = content[:2000]
                break

        prompt = self.JUDGE_PROMPT.format(
            query=query,
            response=sample.response_content[:3000],
            model=sample.model,
        )

        try:
            resp = await litellm.acompletion(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=100,
            )
            content = resp.choices[0].message.content
            scores_raw = json.loads(content)
            if not isinstance(scores_raw, dict):
                return {}
            # Normalize to 0-1
            return {
                k: min(max(v / 5.0, 0.0), 1.0)
                for k, v in scores_raw.items()
                if isinstance(v, (int, float))
            }
        except Exception as e:
            logger.warning("Judge evaluation failed: %s", e)
            return {}

    async def evaluate_batch(self, samples: List[EvalSample]) -> List[Dict[str, float]]:
        """Evaluate a batch of samples concurrently.

        Args:
            samples: List of evaluation samples to score.

        Returns:
            List of score dictionaries, one per sample. Failed evaluations
            return empty dicts.
        """
        tasks = [self.evaluate(s) for s in samples]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # Convert exceptions to empty dicts
        return [r if isinstance(r, dict) else {} for r in results]


class ModelQualityTracker:
    """Tracks per-model quality scores from evaluations.

    Maintains a sliding window of scores per model for computing
    rolling average quality metrics.
    """

    def __init__(self, window_size: int = 100):
        self._scores: Dict[str, List[float]] = defaultdict(list)
        self._window_size = window_size

    def record(self, model: str, score: float) -> None:
        """Record an evaluation score for a model.

        Args:
            model: Model identifier (e.g. "gpt-4o", "claude-3-opus").
            score: Normalized quality score (0.0-1.0).
        """
        scores = self._scores[model]
        scores.append(score)
        if len(scores) > self._window_size:
            scores.pop(0)

    def get_quality(self, model: str) -> Optional[float]:
        """Get average quality score for a model.

        Args:
            model: Model identifier.

        Returns:
            Average quality score, or None if no data exists.
        """
        scores = self._scores.get(model)
        if not scores:
            return None
        return sum(scores) / len(scores)

    def get_all_qualities(self) -> Dict[str, float]:
        """Get quality scores for all tracked models.

        Returns:
            Dictionary mapping model names to average quality scores.
        """
        return {m: sum(s) / len(s) for m, s in self._scores.items() if s}

    def get_ranking(self) -> List[tuple]:
        """Get models ranked by quality (best first).

        Returns:
            List of (model, score) tuples sorted by score descending.
        """
        qualities = self.get_all_qualities()
        return sorted(qualities.items(), key=lambda x: x[1], reverse=True)

    def get_sample_counts(self) -> Dict[str, int]:
        """Get the number of evaluation samples per model.

        Returns:
            Dictionary mapping model names to sample counts.
        """
        return {m: len(s) for m, s in self._scores.items() if s}


class EvalPipeline:
    """Orchestrates the evaluation feedback loop.

    Manages the full lifecycle: sample collection → batch evaluation →
    quality tracking → routing feedback. Runs a background asyncio task
    that periodically evaluates pending samples and pushes updated
    quality scores to the routing subsystem.

    Args:
        sample_rate: Fraction of requests to evaluate (0.0-1.0).
        judge_model: LLM model to use for LLM-as-judge evaluations.
        batch_size: Number of samples per evaluation batch.
        feedback_interval: Seconds between routing feedback updates.
        feedback_callbacks: Optional list of additional callbacks invoked
            with ``(model_qualities: Dict[str, float])`` during feedback push.
    """

    def __init__(
        self,
        sample_rate: float = 0.1,
        judge_model: str = "gpt-4o-mini",
        batch_size: int = 10,
        feedback_interval: int = 300,
        feedback_callbacks: Optional[List[Callable]] = None,
        max_pending: int = 1000,
    ):
        self._sample_rate = sample_rate
        self._batch_size = batch_size
        self._feedback_interval = feedback_interval
        self._judge = EvalJudge(judge_model)
        self._tracker = ModelQualityTracker()
        self.__max_pending = max_pending
        self._pending_samples: deque = deque(maxlen=max_pending)
        self._evaluated_samples: List[EvalSample] = []
        self._max_evaluated = 10000
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._feedback_callbacks: List[Callable] = feedback_callbacks or []
        self._total_collected = 0
        self._total_evaluated = 0
        self._last_feedback_time: Optional[float] = None

    @property
    def _max_pending(self) -> int:
        """Maximum pending samples (synced with deque maxlen)."""
        return self.__max_pending

    @_max_pending.setter
    def _max_pending(self, value: int) -> None:
        self.__max_pending = value
        # Recreate deque with new maxlen, preserving existing items
        self._pending_samples = deque(self._pending_samples, maxlen=value)

    @property
    def tracker(self) -> ModelQualityTracker:
        """Access the model quality tracker."""
        return self._tracker

    def should_sample(self) -> bool:
        """Probabilistically decide whether to sample this request.

        Returns:
            True if this request should be captured for evaluation.
        """
        return random.random() < self._sample_rate

    def collect(self, sample: EvalSample) -> None:
        """Add a sample to the pending evaluation queue.

        Samples are bounded by ``_max_pending``; oldest samples are
        dropped when the buffer is full.

        Args:
            sample: The evaluation sample to enqueue.
        """
        self._pending_samples.append(sample)
        self._total_collected += 1

    async def run_evaluation_batch(self) -> int:
        """Evaluate a batch of pending samples.

        Takes up to ``batch_size`` samples from the pending queue,
        evaluates them via the LLM judge, records scores in the
        quality tracker, and moves them to the evaluated list.

        Returns:
            Number of samples successfully evaluated.
        """
        if not self._pending_samples:
            return 0

        batch = [
            self._pending_samples.popleft()
            for _ in range(min(self._batch_size, len(self._pending_samples)))
        ]

        results = await self._judge.evaluate_batch(batch)

        evaluated = 0
        for sample, scores in zip(batch, results):
            if scores:
                sample.scores = scores
                sample.evaluated = True
                sample.evaluated_at = time.time()

                # Record per-model quality
                avg_score = sum(scores.values()) / len(scores) if scores else 0.5
                self._tracker.record(sample.model, avg_score)
                evaluated += 1

        self._total_evaluated += evaluated

        # Keep evaluated samples (bounded)
        self._evaluated_samples.extend(batch)
        if len(self._evaluated_samples) > self._max_evaluated:
            self._evaluated_samples = self._evaluated_samples[-self._max_evaluated :]

        return evaluated

    async def push_feedback(self) -> Dict[str, Any]:
        """Push quality scores to routing strategies.

        Updates the personalized router's quality bias with the latest
        per-model scores from evaluations. Also invokes any registered
        feedback callbacks.

        Returns:
            Summary of the feedback push including model count and scores.
        """
        qualities = self._tracker.get_all_qualities()
        if not qualities:
            return {"updated": False, "reason": "no_data"}

        feedback_result: Dict[str, Any] = {
            "updated": True,
            "model_count": len(qualities),
            "qualities": qualities,
            "targets": [],
        }

        # Feed to personalized router quality bias
        try:
            from litellm_llmrouter.personalized_routing import (
                get_personalized_router,
            )

            router = get_personalized_router()
            if router is not None:
                for model, score in qualities.items():
                    router.update_quality_bias(model, score)
                feedback_result["targets"].append("personalized_router")
                logger.info(
                    "Updated personalized router quality bias for %d models",
                    len(qualities),
                )
        except Exception as e:
            logger.debug("Could not update personalized router: %s", e)

        # Invoke registered feedback callbacks
        for cb in self._feedback_callbacks:
            try:
                if inspect.iscoroutinefunction(cb):
                    await cb(qualities)
                else:
                    cb(qualities)
                feedback_result["targets"].append(getattr(cb, "__name__", "callback"))
            except Exception as e:
                logger.warning("Feedback callback failed: %s", e)

        self._last_feedback_time = time.time()
        return feedback_result

    async def start_background_loop(self) -> None:
        """Start the background evaluation loop.

        Creates an asyncio task that periodically evaluates pending
        samples and pushes feedback to routing strategies.
        """
        if self._running:
            logger.warning("Eval pipeline background loop already running")
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "Eval pipeline background loop started "
            "(sample_rate=%.2f, batch_size=%d, feedback_interval=%ds)",
            self._sample_rate,
            self._batch_size,
            self._feedback_interval,
        )

    async def stop(self) -> None:
        """Stop the background loop gracefully."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Eval pipeline background loop stopped")

    async def _loop(self) -> None:
        """Background loop: evaluate + push feedback periodically."""
        last_feedback = time.time()
        while self._running:
            try:
                # Evaluate pending samples
                if self._pending_samples:
                    count = await self.run_evaluation_batch()
                    if count > 0:
                        logger.debug("Evaluated %d samples", count)

                # Push feedback periodically
                if time.time() - last_feedback > self._feedback_interval:
                    await self.push_feedback()
                    last_feedback = time.time()

                await asyncio.sleep(10)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Eval pipeline error: %s", e)
                await asyncio.sleep(30)

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics.

        Returns:
            Dictionary with pending/evaluated counts, model qualities,
            rankings, and pipeline state.
        """
        return {
            "pending_samples": len(self._pending_samples),
            "evaluated_samples": len(self._evaluated_samples),
            "total_collected": self._total_collected,
            "total_evaluated": self._total_evaluated,
            "model_qualities": self._tracker.get_all_qualities(),
            "model_ranking": self._tracker.get_ranking()[:10],
            "model_sample_counts": self._tracker.get_sample_counts(),
            "running": self._running,
            "sample_rate": self._sample_rate,
            "batch_size": self._batch_size,
            "feedback_interval": self._feedback_interval,
            "last_feedback_time": self._last_feedback_time,
        }

    def get_recent_samples(
        self,
        limit: int = 50,
        model: Optional[str] = None,
        evaluated_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """Get recent evaluation samples.

        Args:
            limit: Maximum number of samples to return.
            model: Filter by model name (optional).
            evaluated_only: Only return evaluated samples.

        Returns:
            List of sample dictionaries (most recent first).
        """
        source = (
            self._evaluated_samples
            if evaluated_only
            else self._evaluated_samples + list(self._pending_samples)
        )

        if model:
            source = [s for s in source if s.model == model]

        # Most recent first, bounded by limit
        recent = source[-limit:][::-1]

        return [
            {
                "sample_id": s.sample_id,
                "timestamp": s.timestamp,
                "model": s.model,
                "strategy": s.strategy,
                "tier": s.tier,
                "request_tokens": s.request_tokens,
                "response_tokens": s.response_tokens,
                "latency_ms": s.latency_ms,
                "user_id": s.user_id,
                "scores": s.scores,
                "evaluated": s.evaluated,
                "evaluated_at": s.evaluated_at,
            }
            for s in recent
        ]


# ============================================================================
# Singleton
# ============================================================================

_pipeline: Optional[EvalPipeline] = None


def get_eval_pipeline() -> Optional[EvalPipeline]:
    """Get the eval pipeline singleton.

    Returns None if the pipeline is not enabled
    (``ROUTEIQ_EVAL_PIPELINE`` is not ``true``).

    Returns:
        The eval pipeline instance, or None if disabled.
    """
    global _pipeline
    if _pipeline is None:
        if not _is_eval_pipeline_enabled():
            return None
        _pipeline = _create_pipeline_from_settings()
    return _pipeline


def _is_eval_pipeline_enabled() -> bool:
    """Check if the eval pipeline is enabled via settings or env var."""
    try:
        from litellm_llmrouter.settings import get_settings

        settings = get_settings()
        eval_settings = getattr(settings, "eval_pipeline", None)
        if eval_settings is not None:
            return eval_settings.enabled
    except Exception:
        pass
    import os

    return os.environ.get("ROUTEIQ_EVAL_PIPELINE", "false").lower() in (
        "true",
        "1",
        "yes",
    )


def _create_pipeline_from_settings() -> EvalPipeline:
    """Create an EvalPipeline from settings or env vars."""
    import os

    try:
        from litellm_llmrouter.settings import get_settings

        settings = get_settings()
        eval_settings = getattr(settings, "eval_pipeline", None)
        if eval_settings is not None:
            return EvalPipeline(
                sample_rate=eval_settings.sample_rate,
                judge_model=eval_settings.judge_model,
                batch_size=eval_settings.batch_size,
                feedback_interval=eval_settings.feedback_interval,
            )
    except Exception:
        pass

    return EvalPipeline(
        sample_rate=float(os.environ.get("ROUTEIQ_EVAL_SAMPLE_RATE", "0.1")),
        judge_model=os.environ.get("ROUTEIQ_EVAL_JUDGE_MODEL", "gpt-4o-mini"),
        batch_size=int(os.environ.get("ROUTEIQ_EVAL_BATCH_SIZE", "10")),
        feedback_interval=int(os.environ.get("ROUTEIQ_EVAL_FEEDBACK_INTERVAL", "300")),
    )


def reset_eval_pipeline() -> None:
    """Reset the eval pipeline singleton.

    **Must** be called in test fixtures (``autouse=True``) to prevent
    cross-test contamination.
    """
    global _pipeline
    if _pipeline is not None:
        if _pipeline._task is not None and not _pipeline._task.done():
            _pipeline._task.cancel()
        _pipeline._running = False
    _pipeline = None
