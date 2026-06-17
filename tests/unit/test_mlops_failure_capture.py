"""Unit tests for the eval FAILURE-path capture (RouteIQ-d365).

The eval COLLECT arm only samples SUCCESS responses, so AGGREGATE/FEEDBACK is
blind to which models/strategies are error-prone. This adds a parallel,
clearly-labeled, low-rate capture from the FAILURE/timeout path
(``async_log_failure_event``). Verifies:

- a failure event is captured (labeled ``outcome="failure"`` + ``error_type``)
  when the gate is enabled,
- it is a safe no-op when the gate is OFF (default) -- byte-stable,
- the low sample-rate is honored,
- a failure sample is scored to 0.0 (downweight) WITHOUT a judge call when the
  eval batch runs.
"""

from __future__ import annotations

import time

import pytest

from litellm_llmrouter.eval_pipeline import (
    EvalSample,
    get_eval_pipeline,
    reset_eval_pipeline,
)
from litellm_llmrouter.router_decision_callback import (
    RouterDecisionCallback,
    _collect_failure_eval_sample,
)
from litellm_llmrouter.settings import reset_settings


@pytest.fixture(autouse=True)
def _reset():
    reset_settings()
    reset_eval_pipeline()
    yield
    reset_settings()
    reset_eval_pipeline()


def _enable_eval_and_capture(monkeypatch, *, capture_rate: str = "1.0") -> None:
    """Enable the eval pipeline AND the failure-capture gate (nested env)."""
    monkeypatch.setenv("ROUTEIQ_EVAL_PIPELINE__ENABLED", "true")
    monkeypatch.setenv("ROUTEIQ_EVAL_PIPELINE__SAMPLE_RATE", "1.0")
    monkeypatch.setenv("ROUTEIQ_MLOPS__FAILURE_CAPTURE__ENABLED", "true")
    monkeypatch.setenv("ROUTEIQ_MLOPS__FAILURE_CAPTURE__SAMPLE_RATE", capture_rate)
    reset_settings()
    reset_eval_pipeline()


def _failure_kwargs(**overrides):
    kwargs = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "boom"}],
        "litellm_call_id": "fail-1",
        "metadata": {"routing_strategy": "llmrouter-knn"},
        "exception": TimeoutError("upstream timed out"),
    }
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# Capture enabled
# ---------------------------------------------------------------------------


def test_captures_labeled_failure_sample_when_enabled(monkeypatch):
    _enable_eval_and_capture(monkeypatch, capture_rate="1.0")
    pipeline = get_eval_pipeline()
    assert pipeline is not None

    _collect_failure_eval_sample(_failure_kwargs(), start_time=0.0, end_time=0.5)

    assert pipeline._total_collected == 1
    sample = pipeline._pending_samples[0]
    assert isinstance(sample, EvalSample)
    # Clearly labeled so AGGREGATE never confuses it with a success.
    assert sample.outcome == "failure"
    assert sample.error_type == "TimeoutError"
    assert sample.model == "gpt-4o"
    assert sample.strategy == "llmrouter-knn"
    assert sample.response_content == ""
    assert sample.latency_ms == pytest.approx(500.0, abs=1.0)


async def test_async_failure_event_drives_capture(monkeypatch):
    """The live failure callback path drives the FAILURE capture end-to-end."""
    monkeypatch.setenv("LLMROUTER_GOVERNANCE_SPEND_TRACKING", "false")
    _enable_eval_and_capture(monkeypatch, capture_rate="1.0")

    pipeline = get_eval_pipeline()
    callback = RouterDecisionCallback(enabled=False)  # telemetry off, capture on

    await callback.async_log_failure_event(
        kwargs=_failure_kwargs(),
        response_obj=None,
        start_time=time.time(),
        end_time=time.time() + 0.1,
    )

    assert pipeline._total_collected == 1
    assert pipeline._pending_samples[0].outcome == "failure"


@pytest.mark.asyncio
async def test_failure_sample_scored_zero_without_judge(monkeypatch):
    """A captured failure sample downweights its model/strategy WITHOUT a
    judge call when the eval batch runs."""
    _enable_eval_and_capture(monkeypatch, capture_rate="1.0")
    pipeline = get_eval_pipeline()

    # Replace the judge so any accidental call would blow up the test.
    async def _boom_batch(samples):  # pragma: no cover - must not be called
        raise AssertionError("failure samples must not be judged")

    pipeline._judge.evaluate_batch = _boom_batch  # type: ignore[assignment]

    _collect_failure_eval_sample(_failure_kwargs(), start_time=0.0, end_time=0.1)
    evaluated = await pipeline.run_evaluation_batch()

    assert evaluated == 1
    # The failure quality (0.0) is recorded per model AND per strategy so
    # AGGREGATE/FEEDBACK downweights the error-prone target.
    assert pipeline.tracker.get_quality("gpt-4o") == pytest.approx(0.0)
    assert pipeline.tracker.get_strategy_quality("llmrouter-knn") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Disabled / sampling (byte-stable default off)
# ---------------------------------------------------------------------------


def test_noop_when_capture_gate_off(monkeypatch):
    # Eval pipeline ON but failure-capture gate OFF (default).
    monkeypatch.setenv("ROUTEIQ_EVAL_PIPELINE__ENABLED", "true")
    monkeypatch.setenv("ROUTEIQ_EVAL_PIPELINE__SAMPLE_RATE", "1.0")
    reset_settings()
    reset_eval_pipeline()

    pipeline = get_eval_pipeline()
    assert pipeline is not None

    _collect_failure_eval_sample(_failure_kwargs(), start_time=0.0, end_time=0.1)

    # Capture gate off -> nothing collected even though the pipeline is enabled.
    assert pipeline._total_collected == 0


def test_noop_when_pipeline_disabled(monkeypatch):
    # Capture gate ON but eval pipeline OFF -> nowhere to send the sample.
    monkeypatch.setenv("ROUTEIQ_MLOPS__FAILURE_CAPTURE__ENABLED", "true")
    monkeypatch.setenv("ROUTEIQ_MLOPS__FAILURE_CAPTURE__SAMPLE_RATE", "1.0")
    monkeypatch.setenv("ROUTEIQ_EVAL_PIPELINE__ENABLED", "false")
    reset_settings()
    reset_eval_pipeline()
    assert get_eval_pipeline() is None

    # Must not raise.
    _collect_failure_eval_sample(_failure_kwargs(), start_time=0.0, end_time=0.1)
    assert get_eval_pipeline() is None


def test_low_sample_rate_honored(monkeypatch):
    _enable_eval_and_capture(monkeypatch, capture_rate="0.0")
    pipeline = get_eval_pipeline()
    assert pipeline is not None

    _collect_failure_eval_sample(_failure_kwargs(), start_time=0.0, end_time=0.1)

    # sample_rate=0.0 -> never captured.
    assert pipeline._total_collected == 0


def test_capture_never_raises_on_bad_kwargs(monkeypatch):
    _enable_eval_and_capture(monkeypatch, capture_rate="1.0")
    pipeline = get_eval_pipeline()

    # Garbage kwargs / no exception must not break the failure path.
    _collect_failure_eval_sample(
        {"model": None, "messages": "nope", "metadata": "bad"},
        start_time="x",
        end_time="y",
    )
    sample = pipeline._pending_samples[0]
    assert sample.outcome == "failure"
    assert sample.error_type == "unknown"
    assert sample.messages == []
