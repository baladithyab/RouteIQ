"""
Tests for the Agentic Pipeline Plugin.

Tests cover:
1. ComplexityDetector scoring (simple queries, complex queries, edge cases)
2. QueryDecomposer (mock litellm.acompletion)
3. ResponseAggregator (mock litellm.acompletion)
4. SubQueryExecutor (parallel, sequential, error handling)
5. Full AgenticPipelinePlugin lifecycle (startup, shutdown, health_check)
6. on_llm_pre_call hook (threshold gating, streaming bypass, recursion guard)
7. Helper functions (_get_or_create_metadata, _extract_total_tokens)
8. PipelineResult and SubQuery data structures
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm_llmrouter.gateway.plugins.agentic_pipeline import (
    AgenticPipelinePlugin,
    ComplexityDetector,
    PipelineResult,
    QueryDecomposer,
    ResponseAggregator,
    SubQuery,
    SubQueryExecutor,
    _extract_total_tokens,
    _get_or_create_metadata,
)


# ============================================================================
# Helpers
# ============================================================================


def _user(content: str) -> dict:
    return {"role": "user", "content": content}


def _system(content: str) -> dict:
    return {"role": "system", "content": content}


def _assistant(content: str) -> dict:
    return {"role": "assistant", "content": content}


@dataclass
class FakeUsage:
    prompt_tokens: int = 10
    completion_tokens: int = 20
    total_tokens: int = 30


@dataclass
class FakeChoice:
    message: Any = None


@dataclass
class FakeMessage:
    content: str = ""


@dataclass
class FakeResponse:
    choices: list = None
    usage: FakeUsage = None
    model: str = "gpt-4o-mini"

    def __post_init__(self):
        if self.choices is None:
            self.choices = [FakeChoice(message=FakeMessage(content="test response"))]
        if self.usage is None:
            self.usage = FakeUsage()


# ============================================================================
# SubQuery and PipelineResult data structures
# ============================================================================


class TestSubQuery:
    """Test the SubQuery dataclass."""

    def test_defaults(self):
        sq = SubQuery(index=0, query="test")
        assert sq.index == 0
        assert sq.query == "test"
        assert sq.suggested_capability == "balanced"
        assert sq.assigned_model is None
        assert sq.response is None
        assert sq.input_tokens == 0
        assert sq.output_tokens == 0
        assert sq.latency_ms == 0.0
        assert sq.error is None

    def test_custom_values(self):
        sq = SubQuery(
            index=1,
            query="complex query",
            suggested_capability="powerful",
            assigned_model="gpt-4",
            response="answer",
            input_tokens=100,
            output_tokens=200,
            latency_ms=150.5,
        )
        assert sq.suggested_capability == "powerful"
        assert sq.assigned_model == "gpt-4"


class TestPipelineResult:
    """Test the PipelineResult dataclass."""

    def test_defaults(self):
        r = PipelineResult(original_query="test")
        assert r.original_query == "test"
        assert r.was_decomposed is False
        assert r.complexity_score == 0.0
        assert r.sub_queries == []
        assert r.final_response is None
        assert r.total_tokens == 0
        assert r.orchestrator_tokens == 0
        assert r.error_count == 0


# ============================================================================
# ComplexityDetector
# ============================================================================


class TestComplexityDetector:
    """Test the heuristic complexity scorer."""

    def setup_method(self):
        self.detector = ComplexityDetector()

    def test_empty_messages_score_zero(self):
        assert self.detector.score([]) == 0.0

    def test_no_user_message_score_zero(self):
        assert self.detector.score([_system("You are helpful")]) == 0.0

    def test_simple_short_query_low_score(self):
        """A short, single-question query should score low."""
        messages = [_user("What is Python?")]
        score = self.detector.score(messages)
        assert score < 0.3

    def test_long_query_higher_score(self):
        """A long query should have a higher length factor."""
        long_text = "word " * 300  # 300 words
        messages = [_user(long_text)]
        score = self.detector.score(messages)
        # Length factor alone: 300/500 = 0.6, capped at 0.3
        assert score >= 0.3

    def test_multiple_questions_increase_score(self):
        messages = [_user("What is Python? How does it work? Why use it?")]
        score = self.detector.score(messages)
        # 3 question marks -> 0.2 (capped)
        assert score >= 0.1

    def test_multi_step_markers_increase_score(self):
        messages = [
            _user(
                "First, analyze the data. Then, create a report. "
                "After that, email it. Finally, archive it."
            )
        ]
        score = self.detector.score(messages)
        # Multiple multi-step markers present
        assert score >= 0.15

    def test_complexity_markers_increase_score(self):
        messages = [_user("Analyze and compare the performance, then optimize it.")]
        score = self.detector.score(messages)
        # "analyze", "compare", "optimize" are complexity markers
        assert score >= 0.1

    def test_code_presence_increases_score(self):
        messages = [_user("Fix this:\n```python\ndef foo():\n    pass\n```")]
        score = self.detector.score(messages)
        # Code block present
        assert score >= 0.1

    def test_function_keyword_detected(self):
        messages = [_user("The function def process_data(items) is broken.")]
        score = self.detector.score(messages)
        assert score > 0.0

    def test_very_complex_query_high_score(self):
        """A query with all complexity signals should score near 1.0."""
        complex_query = (
            "First, analyze the current architecture. Then, compare it with "
            "the proposed design. After that, evaluate the performance implications. "
            "Finally, implement the changes and debug any issues. "
            "Also, optimize the database queries? Is there a way to refactor "
            "the authentication module? "
            "```python\ndef heavy_computation():\n    pass\n```\n"
            + "Additional context. "
            * 50
        )
        messages = [_user(complex_query)]
        score = self.detector.score(messages)
        assert score >= 0.6

    def test_score_capped_at_one(self):
        """Score should never exceed 1.0."""
        extreme_query = (
            "? " * 100
            + "first then after that finally next step 1 step 2 "
            + "analyze compare evaluate synthesize design implement "
            + "```code```\n"
            + "word " * 600
        )
        messages = [_user(extreme_query)]
        score = self.detector.score(messages)
        assert score <= 1.0

    def test_uses_last_user_message(self):
        """Should extract the LAST user message."""
        messages = [
            _user("Simple question"),
            _assistant("Answer"),
            _user("First, analyze X. Then compare Y? Also evaluate Z?"),
        ]
        score = self.detector.score(messages)
        # The complex last message should be scored, not the simple first one
        assert score > 0.0

    def test_multipart_content_handled(self):
        """Multi-part content (list format) should be handled."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is Python?"},
                    {"type": "image_url", "image_url": {"url": "..."}},
                ],
            }
        ]
        score = self.detector.score(messages)
        assert isinstance(score, float)


# ============================================================================
# QueryDecomposer
# ============================================================================


class TestQueryDecomposer:
    """Test the query decomposition stage."""

    async def test_decompose_returns_sub_queries(self):
        decomposer = QueryDecomposer("gpt-4o-mini", max_subqueries=4)

        mock_response = FakeResponse(
            choices=[
                FakeChoice(
                    message=FakeMessage(
                        content='{"sub_queries": ['
                        '{"query": "What is Python?", "capability": "fast"},'
                        '{"query": "Compare Python vs Rust", "capability": "powerful"}'
                        '], "aggregation_hint": "Combine both answers"}'
                    )
                )
            ],
            usage=FakeUsage(total_tokens=50),
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
            mock_acomp.return_value = mock_response
            sub_queries, hint, tokens = await decomposer.decompose(
                "Tell me about Python and compare it with Rust"
            )

        assert len(sub_queries) == 2
        assert sub_queries[0].query == "What is Python?"
        assert sub_queries[0].suggested_capability == "fast"
        assert sub_queries[1].suggested_capability == "powerful"
        assert hint == "Combine both answers"
        assert tokens == 50

    async def test_decompose_respects_max_subqueries(self):
        decomposer = QueryDecomposer("gpt-4o-mini", max_subqueries=2)

        # Return 5 sub-queries, but max is 2
        mock_response = FakeResponse(
            choices=[
                FakeChoice(
                    message=FakeMessage(
                        content='{"sub_queries": ['
                        '{"query": "q1"}, {"query": "q2"}, {"query": "q3"}, '
                        '{"query": "q4"}, {"query": "q5"}'
                        "]}"
                    )
                )
            ],
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
            mock_acomp.return_value = mock_response
            sub_queries, _, _ = await decomposer.decompose("complex query")

        assert len(sub_queries) == 2

    async def test_decompose_fallback_on_empty_response(self):
        """If decomposer returns no sub-queries, fallback to original query."""
        decomposer = QueryDecomposer("gpt-4o-mini", max_subqueries=4)

        mock_response = FakeResponse(
            choices=[FakeChoice(message=FakeMessage(content='{"sub_queries": []}'))],
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
            mock_acomp.return_value = mock_response
            sub_queries, hint, _ = await decomposer.decompose("original query")

        assert len(sub_queries) == 1
        assert sub_queries[0].query == "original query"
        assert sub_queries[0].suggested_capability == "balanced"

    async def test_decompose_default_hint(self):
        decomposer = QueryDecomposer("gpt-4o-mini", max_subqueries=4)

        mock_response = FakeResponse(
            choices=[
                FakeChoice(
                    message=FakeMessage(content='{"sub_queries": [{"query": "q1"}]}')
                )
            ],
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
            mock_acomp.return_value = mock_response
            _, hint, _ = await decomposer.decompose("test")

        assert "Combine" in hint


# ============================================================================
# SubQueryExecutor
# ============================================================================


class TestSubQueryExecutor:
    """Test the sub-query execution stage."""

    async def test_parallel_execution(self):
        executor = SubQueryExecutor(parallel=True, timeout=10.0)
        sub_queries = [
            SubQuery(index=0, query="q0"),
            SubQuery(index=1, query="q1"),
        ]

        mock_response = FakeResponse()

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
            mock_acomp.return_value = mock_response
            results = await executor.execute_all(sub_queries, model="gpt-4")

        assert len(results) == 2
        assert results[0].response == "test response"
        assert results[1].response == "test response"
        # Both should have been called
        assert mock_acomp.call_count == 2

    async def test_sequential_execution(self):
        executor = SubQueryExecutor(parallel=False, timeout=10.0)
        sub_queries = [SubQuery(index=0, query="q0")]

        mock_response = FakeResponse()

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
            mock_acomp.return_value = mock_response
            results = await executor.execute_all(sub_queries, model="gpt-4")

        assert len(results) == 1
        assert results[0].response == "test response"

    async def test_timeout_handling(self):
        executor = SubQueryExecutor(parallel=False, timeout=0.001)
        sub_queries = [SubQuery(index=0, query="q0")]

        async def slow_response(*args, **kwargs):
            await asyncio.sleep(10)
            return FakeResponse()

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
            mock_acomp.side_effect = slow_response
            results = await executor.execute_all(sub_queries, model="gpt-4")

        assert results[0].error is not None
        assert "timed out" in results[0].error

    async def test_error_handling(self):
        executor = SubQueryExecutor(parallel=False, timeout=10.0)
        sub_queries = [SubQuery(index=0, query="q0")]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
            mock_acomp.side_effect = RuntimeError("LLM API error")
            results = await executor.execute_all(sub_queries, model="gpt-4")

        assert results[0].error is not None
        assert "failed" in results[0].error
        assert results[0].response is None

    async def test_system_messages_prepended(self):
        """System messages should be prepended to each sub-query."""
        system_msgs = [_system("You are helpful")]
        executor = SubQueryExecutor(
            parallel=False, timeout=10.0, system_messages=system_msgs
        )
        sub_queries = [SubQuery(index=0, query="question")]

        mock_response = FakeResponse()

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
            mock_acomp.return_value = mock_response
            await executor.execute_all(sub_queries, model="gpt-4")

        call_kwargs = mock_acomp.call_args
        messages = call_kwargs.kwargs.get("messages", call_kwargs[1].get("messages"))
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    async def test_latency_recorded(self):
        executor = SubQueryExecutor(parallel=False, timeout=10.0)
        sub_queries = [SubQuery(index=0, query="q0")]

        mock_response = FakeResponse()

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
            mock_acomp.return_value = mock_response
            results = await executor.execute_all(sub_queries, model="gpt-4")

        assert results[0].latency_ms >= 0.0

    async def test_usage_tokens_extracted(self):
        executor = SubQueryExecutor(parallel=False, timeout=10.0)
        sub_queries = [SubQuery(index=0, query="q0")]

        mock_response = FakeResponse(
            usage=FakeUsage(prompt_tokens=15, completion_tokens=25)
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
            mock_acomp.return_value = mock_response
            results = await executor.execute_all(sub_queries, model="gpt-4")

        assert results[0].input_tokens == 15
        assert results[0].output_tokens == 25


# ============================================================================
# ResponseAggregator
# ============================================================================


class TestResponseAggregator:
    """Test the response aggregation stage."""

    async def test_aggregate_returns_text_and_tokens(self):
        aggregator = ResponseAggregator("gpt-4o-mini")

        mock_response = FakeResponse(
            choices=[
                FakeChoice(
                    message=FakeMessage(content="Combined answer from all parts")
                )
            ],
            usage=FakeUsage(total_tokens=75),
        )

        sub_queries = [
            SubQuery(index=0, query="q0", response="answer 0"),
            SubQuery(index=1, query="q1", response="answer 1"),
        ]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
            mock_acomp.return_value = mock_response
            text, tokens = await aggregator.aggregate(
                "original question", sub_queries, "combine them"
            )

        assert text == "Combined answer from all parts"
        assert tokens == 75

    async def test_aggregate_handles_failed_sub_queries(self):
        """Sub-queries with errors should still be included in aggregation."""
        aggregator = ResponseAggregator("gpt-4o-mini")

        sub_queries = [
            SubQuery(index=0, query="q0", response="good answer"),
            SubQuery(index=1, query="q1", error="timed out"),
        ]

        mock_response = FakeResponse(
            choices=[FakeChoice(message=FakeMessage(content="Partial aggregation"))],
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
            mock_acomp.return_value = mock_response
            text, _ = await aggregator.aggregate("question", sub_queries, "combine")

        assert text == "Partial aggregation"
        # Check that the prompt included error info
        call_args = mock_acomp.call_args
        prompt = call_args.kwargs.get("messages", call_args[1].get("messages"))[0][
            "content"
        ]
        assert "timed out" in prompt


# ============================================================================
# AgenticPipelinePlugin lifecycle
# ============================================================================


class TestAgenticPipelinePlugin:
    """Test the plugin wrapper."""

    def test_metadata(self):
        plugin = AgenticPipelinePlugin()
        meta = plugin.metadata
        assert meta.name == "agentic_pipeline"
        assert meta.version == "1.0.0"

    async def test_startup_disabled_by_default(self):
        plugin = AgenticPipelinePlugin()
        app = MagicMock()

        import os

        os.environ.pop("ROUTEIQ_AGENTIC_PIPELINE", None)
        await plugin.startup(app)

        assert plugin._enabled is False

    async def test_startup_enabled(self, monkeypatch):
        plugin = AgenticPipelinePlugin()
        app = MagicMock()
        monkeypatch.setenv("ROUTEIQ_AGENTIC_PIPELINE", "true")
        monkeypatch.setenv("ROUTEIQ_AGENTIC_ORCHESTRATOR_MODEL", "gpt-4")
        monkeypatch.setenv("ROUTEIQ_AGENTIC_COMPLEXITY_THRESHOLD", "0.5")
        monkeypatch.setenv("ROUTEIQ_AGENTIC_MAX_SUBQUERIES", "3")
        monkeypatch.setenv("ROUTEIQ_AGENTIC_PARALLEL", "false")
        monkeypatch.setenv("ROUTEIQ_AGENTIC_TIMEOUT", "15")

        await plugin.startup(app)

        assert plugin._enabled is True
        assert plugin._orchestrator_model == "gpt-4"
        assert plugin._threshold == 0.5
        assert plugin._max_subqueries == 3
        assert plugin._parallel is False
        assert plugin._timeout == 15.0

    async def test_shutdown(self, monkeypatch):
        plugin = AgenticPipelinePlugin()
        app = MagicMock()
        monkeypatch.setenv("ROUTEIQ_AGENTIC_PIPELINE", "true")
        await plugin.startup(app)
        assert plugin._enabled is True

        await plugin.shutdown(app)
        assert plugin._enabled is False

    async def test_health_check_disabled(self):
        plugin = AgenticPipelinePlugin()
        health = await plugin.health_check()
        assert health["status"] == "ok"
        assert "disabled" in health.get("detail", "")

    async def test_health_check_enabled(self, monkeypatch):
        plugin = AgenticPipelinePlugin()
        app = MagicMock()
        monkeypatch.setenv("ROUTEIQ_AGENTIC_PIPELINE", "true")
        await plugin.startup(app)

        health = await plugin.health_check()
        assert health["status"] == "ok"
        assert "orchestrator_model" in health

    async def test_pre_call_disabled_returns_none(self):
        """When disabled, returns None (pass-through)."""
        plugin = AgenticPipelinePlugin()
        result = await plugin.on_llm_pre_call("gpt-4", [_user("Hi")], {})
        assert result is None

    async def test_pre_call_empty_messages_returns_none(self, monkeypatch):
        plugin = AgenticPipelinePlugin()
        app = MagicMock()
        monkeypatch.setenv("ROUTEIQ_AGENTIC_PIPELINE", "true")
        await plugin.startup(app)

        result = await plugin.on_llm_pre_call("gpt-4", [], {})
        assert result is None

    async def test_pre_call_streaming_bypassed(self, monkeypatch):
        """Streaming requests should bypass the pipeline."""
        plugin = AgenticPipelinePlugin()
        app = MagicMock()
        monkeypatch.setenv("ROUTEIQ_AGENTIC_PIPELINE", "true")
        await plugin.startup(app)

        result = await plugin.on_llm_pre_call(
            "gpt-4", [_user("Complex query? Analyze and compare.")], {"stream": True}
        )
        assert result is None

    async def test_pre_call_recursion_guard(self, monkeypatch):
        """Sub-queries (marked with _agentic_sub_query) should be skipped."""
        plugin = AgenticPipelinePlugin()
        app = MagicMock()
        monkeypatch.setenv("ROUTEIQ_AGENTIC_PIPELINE", "true")
        await plugin.startup(app)

        kwargs = {"litellm_params": {"metadata": {"_agentic_sub_query": True}}}
        result = await plugin.on_llm_pre_call("gpt-4", [_user("test")], kwargs)
        assert result is None

    async def test_pre_call_below_threshold_passthrough(self, monkeypatch):
        """Simple queries below threshold should pass through."""
        plugin = AgenticPipelinePlugin()
        app = MagicMock()
        monkeypatch.setenv("ROUTEIQ_AGENTIC_PIPELINE", "true")
        monkeypatch.setenv("ROUTEIQ_AGENTIC_COMPLEXITY_THRESHOLD", "0.9")
        await plugin.startup(app)

        result = await plugin.on_llm_pre_call("gpt-4", [_user("Hello")], {})
        assert result is None


# ============================================================================
# Helper functions
# ============================================================================


class TestGetOrCreateMetadata:
    """Test the _get_or_create_metadata helper."""

    def test_creates_nested_structure(self):
        kwargs: dict = {}
        meta = _get_or_create_metadata(kwargs)
        assert isinstance(meta, dict)
        assert kwargs["litellm_params"]["metadata"] is meta

    def test_reuses_existing_metadata(self):
        existing = {"key": "value"}
        kwargs = {"litellm_params": {"metadata": existing}}
        meta = _get_or_create_metadata(kwargs)
        assert meta is existing

    def test_replaces_non_dict_metadata(self):
        kwargs = {"litellm_params": {"metadata": "not a dict"}}
        meta = _get_or_create_metadata(kwargs)
        assert isinstance(meta, dict)


class TestExtractTotalTokens:
    """Test the _extract_total_tokens helper."""

    def test_with_usage(self):
        resp = FakeResponse(usage=FakeUsage(total_tokens=42))
        assert _extract_total_tokens(resp) == 42

    def test_no_usage(self):
        resp = MagicMock(spec=[])  # No usage attribute
        assert _extract_total_tokens(resp) == 0

    def test_none_usage(self):
        resp = MagicMock(usage=None)
        assert _extract_total_tokens(resp) == 0


# ============================================================================
# Build sub-query kwargs
# ============================================================================


class TestBuildSubQueryKwargs:
    """Test _build_sub_query_kwargs static method."""

    def test_strips_messages_and_stream(self):
        original = {
            "model": "gpt-4",
            "messages": [_user("test")],
            "stream": True,
            "response_format": {"type": "json"},
            "temperature": 0.5,
        }
        result = AgenticPipelinePlugin._build_sub_query_kwargs(original)
        assert "messages" not in result
        assert "stream" not in result
        assert "response_format" not in result
        assert result["temperature"] == 0.5
        assert result["model"] == "gpt-4"

    def test_injects_recursion_guard(self):
        original = {"model": "gpt-4"}
        result = AgenticPipelinePlugin._build_sub_query_kwargs(original)
        meta = result["litellm_params"]["metadata"]
        assert meta["_agentic_sub_query"] is True

    def test_shallow_copy_of_top_level_keys(self):
        """Top-level keys like 'messages' are stripped in the copy, not the original."""
        original = {"model": "gpt-4", "messages": [_user("test")], "stream": True}
        result = AgenticPipelinePlugin._build_sub_query_kwargs(original)
        # Top-level keys stripped from result but original untouched
        assert "messages" in original
        assert "stream" in original
        assert "messages" not in result
