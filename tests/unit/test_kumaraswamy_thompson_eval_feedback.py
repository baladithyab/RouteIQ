"""End-to-end proof that eval-derived quality LEARNING reaches the posterior the
bandit READS (RouteIQ-e019).

The bug: the Kumaraswamy-Thompson bandit keys posteriors by ``(bucket, arm)`` and
the hot path (``select_deployment``) READS ``(_bucket(context), arm)``
(e.g. ``"len:s"``). The eval->bandit FEEDBACK path aggregated quality PER-MODEL
and built ``RoutingFeedback(model, score)`` with NO bucket, so ``update`` resolved
``bucket or _recover_bucket(None) or "default"`` = ``"default"`` -- every
eval-derived update landed on the ``("default", arm)`` posterior the router NEVER
samples. The loop was fully wired (COLLECT->EVALUATE->AGGREGATE->FEEDBACK all
fired) yet the bandit's per-bucket learning was INERT end-to-end.

These tests assert on the READER's key: feedback for a request whose hot-path
bucket is ``"len:s"`` must move the ``("len:s", arm)`` posterior the router
samples, and NOTHING may land on ``("default", arm)``.

Credential-free: the LLM judge is stubbed with deterministic per-model scores; no
live AWS / LiteLLM. Every singleton is reset in the autouse fixture.
"""

from __future__ import annotations

import pytest

from litellm_llmrouter.adapters.mlops import (
    MLOpsCoordinator,
    reset_mlops_coordinator,
    wire_mlops_feedback_loop,
)
from litellm_llmrouter.eval_pipeline import (
    EvalPipeline,
    EvalSample,
    reset_eval_pipeline,
)
from litellm_llmrouter.kumaraswamy_thompson import (
    STRATEGY_NAME,
    InMemoryPosteriorBackend,
    KumaraswamyThompsonStrategy,
)
from litellm_llmrouter.strategy_registry import (
    RoutingContext,
    get_routing_registry,
    reset_routing_singletons,
)

# Arms (concrete litellm_params.model, the bandit's arm key).
ARM_X = "bedrock/anthropic.claude-3-sonnet"
ARM_Y = "bedrock/meta.llama3-8b"

# A prompt long enough to bucket as "len:s" (64 <= tokens < 512 => 256 <= chars
# < 2048). ~300 chars / 4 = ~75 tokens -> "len:s" via _bucket's length fallback.
_PROMPT_LEN_S = "summarize the following passage in detail. " * 8


@pytest.fixture(autouse=True)
def _reset():
    reset_eval_pipeline()
    reset_routing_singletons()
    reset_mlops_coordinator()
    yield
    reset_eval_pipeline()
    reset_routing_singletons()
    reset_mlops_coordinator()


class _FakeRouter:
    def __init__(self, deployments):
        self.healthy_deployments = deployments
        self.model_list = deployments


def _dep(model_name: str, arm: str) -> dict:
    return {"model_name": model_name, "litellm_params": {"model": arm}}


def _ctx_len_s(router, request_id: str) -> RoutingContext:
    return RoutingContext(
        router=router,
        model="auto",
        messages=[{"role": "user", "content": _PROMPT_LEN_S}],
        request_id=request_id,
    )


class _PerModelJudge:
    """Deterministic judge: high quality for ``high_model``, low for the rest.

    Replaces ``EvalPipeline._judge`` so ``run_evaluation_batch`` runs its REAL
    code path (EVALUATE -> AGGREGATE) without a live LLM. Scores are on the
    judge's 0-1 metric scale (the same scale ``litellm.acompletion`` would yield
    after normalization).
    """

    def __init__(self, high_model: str, high: float = 0.95, low: float = 0.05):
        self._high_model = high_model
        self._high = high
        self._low = low

    async def evaluate_batch(self, samples):
        out = []
        for s in samples:
            q = self._high if s.model == self._high_model else self._low
            out.append({"relevance": q, "helpfulness": q, "coherence": q})
        return out


def _mean(strategy: KumaraswamyThompsonStrategy, bucket: str, arm: str) -> float:
    return strategy._get_posterior(bucket, arm).mean()


def _strength(strategy: KumaraswamyThompsonStrategy, bucket: str, arm: str) -> float:
    return strategy._get_posterior(bucket, arm).strength()


def _seed_decision_buckets(strategy, router):
    """Drive the bandit hot path so it LOGS the "len:s" bucket per request_id.

    Returns the (call_id, decision_bucket) pairs. The decision bucket recovered
    here is the EXACT string the bandit will key its posterior on, so the test
    asserts against what the hot path actually produced (not a hard-coded guess).
    """
    pairs = []
    for i in range(2):
        call_id = f"call-{i}"
        ctx = _ctx_len_s(router, request_id=call_id)
        strategy.select_deployment(ctx)
        bucket = strategy._recover_bucket(call_id)
        pairs.append((call_id, bucket))
    return pairs


@pytest.mark.asyncio
async def test_eval_feedback_moves_the_posterior_the_router_reads(monkeypatch):
    """DECISIVE: eval quality moves the (decision-bucket, arm) posterior, and
    NOTHING leaks to ("default", arm).

    Drives COLLECT -> EVALUATE -> AGGREGATE -> FEEDBACK end-to-end through the
    real seams and asserts on the READER's key.
    """
    # 1) Register the bandit and make it the active routing strategy. Explicit
    #    quality-only-effective weights make the reward == the quality reward so
    #    the mean moves cleanly in the judged direction (cost/lat terms absent).
    backend = InMemoryPosteriorBackend()
    strategy = KumaraswamyThompsonStrategy(
        backend=backend, seed=7, w_quality=1.0, w_cost=0.0, w_latency=0.0
    )
    get_routing_registry().register(STRATEGY_NAME, strategy, version="v1")

    router = _FakeRouter([_dep("auto", ARM_X), _dep("auto", ARM_Y)])

    # 2) Decision time: the bandit logs its bucket per request_id (== call_id).
    pairs = _seed_decision_buckets(strategy, router)
    decision_bucket = pairs[0][1]
    assert decision_bucket == "len:s", (
        "this test requires the length-fallback bucket; the centroid classifier "
        f"must not be warm in unit tests (got {decision_bucket!r})"
    )

    # Record the pre-feedback posterior means at the READER's key.
    pre_x = _mean(strategy, decision_bucket, ARM_X)
    pre_y = _mean(strategy, decision_bucket, ARM_Y)

    # 3) COLLECT: build samples carrying the EXACT recovered decision bucket,
    #    exactly as the live COLLECT arm does (_recover_bandit_bucket threads it).
    pipeline = EvalPipeline(sample_rate=1.0)
    for call_id, bucket in pairs:
        for arm in (ARM_X, ARM_Y):
            pipeline.collect(
                EvalSample(
                    sample_id=f"{call_id}-{arm}",
                    timestamp=0.0,
                    model=arm,
                    strategy=STRATEGY_NAME,
                    tier="complex",  # the WRONG key (model-name tier) on purpose
                    bandit_bucket=bucket,  # the RIGHT key (decision-time bucket)
                    messages=[{"role": "user", "content": _PROMPT_LEN_S}],
                )
            )

    # 4) EVALUATE + AGGREGATE: real run_evaluation_batch with a stubbed judge.
    pipeline._batch_size = 100
    pipeline._judge = _PerModelJudge(high_model=ARM_X)
    evaluated = await pipeline.run_evaluation_batch()
    assert evaluated == 4

    # AGGREGATE actually keyed per-(bucket, model).
    bq = pipeline.tracker.get_all_bucket_qualities()
    assert (decision_bucket, ARM_X) in bq
    assert (decision_bucket, ARM_Y) in bq
    assert bq[(decision_bucket, ARM_X)] > bq[(decision_bucket, ARM_Y)]

    # 5) FEEDBACK: wire the coordinator to THIS pipeline and push.
    coord = MLOpsCoordinator()
    assert (
        wire_mlops_feedback_loop(coordinator=coord, eval_pipeline=pipeline, force=True)
        is True
    )
    result = await pipeline.push_feedback()
    assert result["updated"] is True

    # 6) ASSERT ON THE READER'S KEY: the bucket the router samples moved.
    post_x = _mean(strategy, decision_bucket, ARM_X)
    post_y = _mean(strategy, decision_bucket, ARM_Y)
    assert post_x > pre_x, "high-quality arm posterior at the READ key did not rise"
    assert post_y < pre_y, "low-quality arm posterior at the READ key did not fall"
    assert post_x > post_y, "router-read posterior does not prefer the better arm"

    # 7) NO DEFAULT-BUCKET LEAK: the inert ("default", arm) posterior is untouched
    #    (still at the Beta(1,1) prior strength -> never updated).
    assert _strength(strategy, "default", ARM_X) == pytest.approx(2.0)
    assert _strength(strategy, "default", ARM_Y) == pytest.approx(2.0)
    assert _mean(strategy, "default", ARM_X) == pytest.approx(0.5)
    assert _mean(strategy, "default", ARM_Y) == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_regression_bucket_mismatch_leaves_read_posterior_inert(monkeypatch):
    """REGRESSION: the OLD behaviour (no bucket carried) leaves the READ posterior
    unmoved and lands everything on ("default", arm).

    This reproduces the pre-fix path: build samples WITHOUT a bandit_bucket so the
    per-bucket aggregate is empty and the fan-out falls back to per-model
    ``RoutingFeedback`` with NO bucket. The bandit then resolves ``"default"`` and
    the ("len:s", arm) posterior the hot path reads stays at its prior -- proving
    the test genuinely catches the inert loop (and that the fix is what moves it).
    """
    backend = InMemoryPosteriorBackend()
    strategy = KumaraswamyThompsonStrategy(
        backend=backend, seed=7, w_quality=1.0, w_cost=0.0, w_latency=0.0
    )
    get_routing_registry().register(STRATEGY_NAME, strategy, version="v1")

    router = _FakeRouter([_dep("auto", ARM_X), _dep("auto", ARM_Y)])
    pairs = _seed_decision_buckets(strategy, router)
    decision_bucket = pairs[0][1]
    assert decision_bucket == "len:s"

    pre_x = _mean(strategy, decision_bucket, ARM_X)
    pre_y = _mean(strategy, decision_bucket, ARM_Y)

    pipeline = EvalPipeline(sample_rate=1.0)
    for call_id, _bucket in pairs:
        for arm in (ARM_X, ARM_Y):
            pipeline.collect(
                EvalSample(
                    sample_id=f"{call_id}-{arm}",
                    timestamp=0.0,
                    model=arm,
                    strategy=STRATEGY_NAME,
                    tier="complex",
                    bandit_bucket="",  # OLD behaviour: no decision bucket carried
                    messages=[{"role": "user", "content": _PROMPT_LEN_S}],
                )
            )

    pipeline._batch_size = 100
    pipeline._judge = _PerModelJudge(high_model=ARM_X)
    await pipeline.run_evaluation_batch()

    # No per-bucket aggregate => fan-out can only go per-model (the old, inert path).
    assert pipeline.tracker.get_all_bucket_qualities() == {}

    coord = MLOpsCoordinator()
    wire_mlops_feedback_loop(coordinator=coord, eval_pipeline=pipeline, force=True)
    await pipeline.push_feedback()

    # The READ posterior the router samples is UNMOVED (the bug).
    assert _mean(strategy, decision_bucket, ARM_X) == pytest.approx(pre_x)
    assert _mean(strategy, decision_bucket, ARM_Y) == pytest.approx(pre_y)

    # Everything landed on the inert ("default", arm) posterior the router never reads.
    assert _strength(strategy, "default", ARM_X) > 2.0
    assert _strength(strategy, "default", ARM_Y) > 2.0
    assert _mean(strategy, "default", ARM_X) > _mean(strategy, "default", ARM_Y)


@pytest.mark.asyncio
async def test_per_model_fanout_byte_stable_when_no_bandit_bucket(monkeypatch):
    """A non-bandit aggregate (no per-bucket data) still fans out per-model.

    Guards the byte-stable fallback: ``on_aggregate_feedback`` with an empty
    per-bucket aggregate dispatches exactly one per-model ``RoutingFeedback`` per
    model (bucket=None), preserving the prior contract for non-bandit adapters.
    """
    seen = []

    class _RecordingAdapter:
        def declare_capabilities(self):
            from litellm_llmrouter.adapters.contract import (
                TRAIN_MODE_CONTINUOUS,
                AdapterManifest,
            )

            return AdapterManifest(
                name="rec", learns=True, train_mode=TRAIN_MODE_CONTINUOUS
            )

        def update_from_feedback(self, feedback):
            seen.append((feedback.model, feedback.bucket, feedback.score))

    coord = MLOpsCoordinator()
    coord.register_learning_adapter("rec", _RecordingAdapter())
    # No eval pipeline wired -> _bucket_qualities_for_dispatch() returns {} ->
    # per-model fallback.
    coord.on_aggregate_feedback({"gpt-4o": 1.0, "claude": 0.0})

    assert sorted(seen) == [
        ("claude", None, -1.0),
        ("gpt-4o", None, 1.0),
    ]
