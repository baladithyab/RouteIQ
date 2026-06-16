"""
Unit Tests for the eval-pipeline COLLECT arm (RouteIQ-295a)
===========================================================

Proves that ``router_decision_callback`` feeds the closed MLOps loop:

- A routing decision (LiteLLM success callback) results in a collected
  ``EvalSample`` (``pipeline._total_collected`` increments / sample captured)
  when the eval pipeline is present.
- It is a safe no-op when ``get_eval_pipeline()`` returns ``None`` (disabled).
- Sampling is honored: ``should_sample() == False`` collects nothing.
- The collected sample carries the request/response fields the EvalSample
  schema models, and collection never breaks the hot response path.
"""

import time
from unittest.mock import MagicMock

import pytest

from litellm_llmrouter.eval_pipeline import (
    EvalPipeline,
    EvalSample,
    get_eval_pipeline,
    reset_eval_pipeline,
)
from litellm_llmrouter.router_decision_callback import (
    RouterDecisionCallback,
    _collect_eval_sample,
    _extract_response_content,
    _resolve_tier_from_model,
)
from litellm_llmrouter.settings import reset_settings


def _enable_pipeline_env(monkeypatch, *, sample_rate: str = "1.0") -> None:
    """Turn the eval pipeline ON via the env vars settings actually binds.

    The ``eval_pipeline`` field is a NESTED ``BaseModel`` on ``GatewaySettings``,
    so its ``enabled`` / ``sample_rate`` subfields bind to the nested
    ``ROUTEIQ_EVAL_PIPELINE__ENABLED`` / ``ROUTEIQ_EVAL_PIPELINE__SAMPLE_RATE``
    form (``env_nested_delimiter="__"``), NOT the flat ``ROUTEIQ_EVAL_PIPELINE``
    -- the flat name collides with the ``eval_pipeline`` field itself and makes
    ``GatewaySettings()`` raise.  ``get_eval_pipeline()`` reads settings first
    (``_is_eval_pipeline_enabled``), so we MUST set the nested form AND
    ``reset_settings()`` so the next ``get_settings()`` rebuilds with these
    values (the Settings singleton is otherwise built once at import time).
    """
    monkeypatch.setenv("ROUTEIQ_EVAL_PIPELINE__ENABLED", "true")
    monkeypatch.setenv("ROUTEIQ_EVAL_PIPELINE__SAMPLE_RATE", sample_rate)
    reset_settings()


@pytest.fixture(autouse=True)
def _reset_eval_pipeline_singleton():
    """Reset the eval pipeline + settings singletons before and after each test.

    Mandatory per CLAUDE.md: every subsystem uses singletons; tests MUST reset
    them in an autouse fixture to avoid cross-test contamination.  The settings
    singleton is reset too because ``get_eval_pipeline()`` reads its enablement
    from ``get_settings()`` (a process-wide cache).
    """
    reset_settings()
    reset_eval_pipeline()
    yield
    reset_eval_pipeline()
    reset_settings()


def _make_response(prompt_tokens=12, completion_tokens=34, content="Hi there"):
    """Build a mock OpenAI-compatible response with usage + content."""
    resp = MagicMock()
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    choice = MagicMock()
    choice.message.content = content
    resp.choices = [choice]
    return resp


def _make_kwargs(**overrides):
    kwargs = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "litellm_call_id": "call-123",
        "metadata": {"routing_strategy": "llmrouter-knn"},
    }
    kwargs.update(overrides)
    return kwargs


# =============================================================================
# COLLECT arm: pipeline present
# =============================================================================


class TestCollectEvalSampleEnabled:
    def test_collects_sample_when_pipeline_present(self, monkeypatch):
        # Force-enable the pipeline with sample_rate=1.0 (always sample).
        _enable_pipeline_env(monkeypatch, sample_rate="1.0")

        pipeline = get_eval_pipeline()
        assert pipeline is not None
        assert pipeline._total_collected == 0

        _collect_eval_sample(
            kwargs=_make_kwargs(),
            response_obj=_make_response(),
            start_time=100.0,
            end_time=100.5,
        )

        assert pipeline._total_collected == 1
        assert len(pipeline._pending_samples) == 1

    def test_collected_sample_carries_request_response_fields(self, monkeypatch):
        _enable_pipeline_env(monkeypatch, sample_rate="1.0")

        pipeline = get_eval_pipeline()

        _collect_eval_sample(
            kwargs=_make_kwargs(),
            response_obj=_make_response(
                prompt_tokens=12, completion_tokens=34, content="four"
            ),
            start_time=100.0,
            end_time=100.25,
        )

        sample = pipeline._pending_samples[0]
        assert isinstance(sample, EvalSample)
        assert sample.model == "gpt-4o"
        assert sample.strategy == "llmrouter-knn"
        assert sample.messages == [{"role": "user", "content": "What is 2+2?"}]
        assert sample.request_tokens == 12
        assert sample.response_tokens == 34
        assert sample.response_content == "four"
        # 0.25s window -> 250ms
        assert sample.latency_ms == pytest.approx(250.0, abs=1.0)
        assert sample.sample_id == "call-123"

    def test_governance_scope_populates_user_and_workspace(self, monkeypatch):
        _enable_pipeline_env(monkeypatch, sample_rate="1.0")

        pipeline = get_eval_pipeline()

        kwargs = _make_kwargs(
            metadata={
                "routing_strategy": "llmrouter-knn",
                "_governance_ctx": {
                    "key_id": "fake-api-key",
                    "workspace_id": "ws-1",
                },
                "user_api_key_user_id": "user-9",
            }
        )
        _collect_eval_sample(
            kwargs=kwargs,
            response_obj=_make_response(),
            start_time=0.0,
            end_time=0.1,
        )

        sample = pipeline._pending_samples[0]
        assert sample.workspace_id == "ws-1"
        assert sample.user_id == "user-9"

    async def test_async_success_event_collects(self, monkeypatch):
        """The live success callback path drives COLLECT end-to-end."""
        # Keep the governance spend writer out of the way for this test.
        monkeypatch.setenv("LLMROUTER_GOVERNANCE_SPEND_TRACKING", "false")
        _enable_pipeline_env(monkeypatch, sample_rate="1.0")

        pipeline = get_eval_pipeline()
        callback = RouterDecisionCallback(enabled=False)  # telemetry off, COLLECT on

        await callback.async_log_success_event(
            kwargs=_make_kwargs(),
            response_obj=_make_response(),
            start_time=time.time(),
            end_time=time.time() + 0.1,
        )

        assert pipeline._total_collected == 1


# =============================================================================
# COLLECT arm: safe no-op when disabled / not sampled
# =============================================================================


class TestCollectEvalSampleDisabled:
    def test_noop_when_pipeline_none(self, monkeypatch):
        # Pipeline disabled (default) -> get_eval_pipeline() returns None.
        # The autouse fixture reset settings; eval_pipeline.enabled defaults
        # to False, so no env var needs to be set to keep it off.
        monkeypatch.setenv("ROUTEIQ_EVAL_PIPELINE__ENABLED", "false")
        reset_settings()
        assert get_eval_pipeline() is None

        # Must not raise.
        _collect_eval_sample(
            kwargs=_make_kwargs(),
            response_obj=_make_response(),
            start_time=0.0,
            end_time=0.1,
        )

        # Still None (collection did not lazily create a pipeline).
        assert get_eval_pipeline() is None

    def test_noop_when_not_sampled(self, monkeypatch):
        # Pipeline enabled but sample_rate=0.0 -> should_sample() always False.
        _enable_pipeline_env(monkeypatch, sample_rate="0.0")

        pipeline = get_eval_pipeline()
        assert pipeline is not None

        _collect_eval_sample(
            kwargs=_make_kwargs(),
            response_obj=_make_response(),
            start_time=0.0,
            end_time=0.1,
        )

        assert pipeline._total_collected == 0
        assert len(pipeline._pending_samples) == 0

    def test_collection_never_raises_on_bad_response(self, monkeypatch):
        _enable_pipeline_env(monkeypatch, sample_rate="1.0")
        pipeline = get_eval_pipeline()

        # response_obj=None and weird kwargs must not break the hot path.
        _collect_eval_sample(
            kwargs={"model": None, "messages": "not-a-list", "metadata": "nope"},
            response_obj=None,
            start_time="bad",
            end_time="bad",
        )
        # A sample is still collected (fail-open builds a best-effort sample).
        assert pipeline._total_collected == 1
        sample = pipeline._pending_samples[0]
        assert sample.response_content == ""
        assert sample.messages == []


# =============================================================================
# Helper extractors
# =============================================================================


class TestExtractHelpers:
    def test_extract_response_content_happy(self):
        assert _extract_response_content(_make_response(content="hello")) == "hello"

    def test_extract_response_content_no_choices(self):
        resp = MagicMock()
        resp.choices = []
        assert _extract_response_content(resp) == ""

    def test_extract_response_content_none(self):
        assert _extract_response_content(None) == ""

    def test_extract_response_content_truncates(self):
        from litellm_llmrouter.router_decision_callback import (
            _EVAL_RESPONSE_CONTENT_CAP,
        )

        big = "x" * (_EVAL_RESPONSE_CONTENT_CAP + 100)
        out = _extract_response_content(_make_response(content=big))
        assert len(out) == _EVAL_RESPONSE_CONTENT_CAP

    def test_resolve_tier_reasoning(self):
        assert _resolve_tier_from_model("o3-mini") == "reasoning"

    def test_resolve_tier_simple(self):
        assert _resolve_tier_from_model("claude-3-haiku") == "simple"

    def test_resolve_tier_complex(self):
        assert _resolve_tier_from_model("gpt-4o") == "complex"

    def test_resolve_tier_empty(self):
        assert _resolve_tier_from_model("") == ""


# =============================================================================
# Pipeline wiring sanity
# =============================================================================


def test_collect_increments_total_collected_directly():
    """Sanity check the EvalPipeline.collect contract the COLLECT arm relies on."""
    pipeline = EvalPipeline(sample_rate=1.0)
    assert pipeline._total_collected == 0
    pipeline.collect(
        EvalSample(
            sample_id="s1",
            timestamp=time.time(),
            model="gpt-4o",
            strategy="knn",
            tier="complex",
            messages=[{"role": "user", "content": "hi"}],
        )
    )
    assert pipeline._total_collected == 1
