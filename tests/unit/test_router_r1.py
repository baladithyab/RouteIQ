"""Unit tests for Router-R1 iterative reasoning router."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm_llmrouter.router_r1 import (
    R1Result,
    RouterR1,
    RoutingStep,
    get_router_r1,
    reset_router_r1,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_r1():
    """Reset Router-R1 singleton between tests."""
    reset_router_r1()
    yield
    reset_router_r1()


@pytest.fixture
def router():
    """Create a RouterR1 instance with defaults."""
    return RouterR1(
        router_model="gpt-4o-mini",
        max_iterations=3,
        timeout_per_iteration=10.0,
    )


@pytest.fixture
def sample_deployments():
    """Sample deployments list for testing."""
    return [
        {"model_name": "gpt-4o-mini"},
        {"model_name": "gpt-4o"},
        {"model_name": "claude-3-5-sonnet-20241022"},
        {"model_name": "gpt-4o-mini"},  # duplicate
    ]


def _make_response(content: str, total_tokens: int = 100):
    """Build a mock LiteLLM completion response."""
    usage = MagicMock()
    usage.total_tokens = total_tokens
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


# ---------------------------------------------------------------------------
# _build_model_info tests
# ---------------------------------------------------------------------------


class TestBuildModelInfo:
    """Tests for the _build_model_info method."""

    def test_deduplicates_models(self, router, sample_deployments):
        """Should only include each model once."""
        info = router._build_model_info(sample_deployments)
        lines = [line for line in info.split("\n") if line.strip()]
        model_names = [line.split(":")[0].strip("- ") for line in lines]
        assert len(model_names) == len(set(model_names))

    def test_includes_known_models(self, router, sample_deployments):
        """Should include models from the deployments list."""
        info = router._build_model_info(sample_deployments)
        assert "gpt-4o-mini" in info
        assert "gpt-4o" in info

    def test_limits_to_20_models(self, router):
        """Should cap at 20 models."""
        deployments = [{"model_name": f"model-{i}"} for i in range(30)]
        info = router._build_model_info(deployments)
        lines = [line for line in info.split("\n") if line.strip()]
        assert len(lines) <= 20

    def test_empty_deployments(self, router):
        """Should return empty string for no deployments."""
        info = router._build_model_info([])
        assert info == ""

    def test_unknown_model_uses_defaults(self, router):
        """Unknown models should still appear with '?' costs."""
        info = router._build_model_info([{"model_name": "unknown-model-xyz"}])
        assert "unknown-model-xyz" in info


# ---------------------------------------------------------------------------
# Regex pattern tests
# ---------------------------------------------------------------------------


class TestRegexPatterns:
    """Tests for the XML tag extraction patterns."""

    def test_route_pattern(self, router):
        text = '<route model="gpt-4o">What is 2+2?</route>'
        match = router._route_pattern.search(text)
        assert match is not None
        assert match.group(1) == "gpt-4o"
        assert match.group(2) == "What is 2+2?"

    def test_route_pattern_multiline(self, router):
        text = (
            '<route model="claude-3-5-sonnet-20241022">Explain\nquantum physics</route>'
        )
        match = router._route_pattern.search(text)
        assert match is not None
        assert match.group(1) == "claude-3-5-sonnet-20241022"
        assert "quantum physics" in match.group(2)

    def test_route_pattern_no_match(self, router):
        text = "Just some regular text without route tags"
        match = router._route_pattern.search(text)
        assert match is None

    def test_answer_pattern(self, router):
        text = "<answer>The answer is 42</answer>"
        match = router._answer_pattern.search(text)
        assert match is not None
        assert match.group(1) == "The answer is 42"

    def test_answer_pattern_multiline(self, router):
        text = "<answer>Line 1\nLine 2\nLine 3</answer>"
        match = router._answer_pattern.search(text)
        assert match is not None
        assert "Line 2" in match.group(1)

    def test_think_pattern(self, router):
        text = "<think>I need to consider the options</think>"
        match = router._think_pattern.search(text)
        assert match is not None
        assert match.group(1) == "I need to consider the options"

    def test_think_pattern_no_match(self, router):
        text = "No thinking here"
        match = router._think_pattern.search(text)
        assert match is None

    def test_all_patterns_in_one_response(self, router):
        text = (
            "<think>Let me think about this</think>\n"
            '<route model="gpt-4o-mini">What is the capital of France?</route>\n'
            "<answer>Paris is the capital of France</answer>"
        )
        assert router._think_pattern.search(text) is not None
        assert router._route_pattern.search(text) is not None
        assert router._answer_pattern.search(text) is not None


# ---------------------------------------------------------------------------
# route() tests with mocked litellm.acompletion
# ---------------------------------------------------------------------------


class TestRoute:
    """Tests for the main route() method."""

    @pytest.mark.asyncio
    async def test_direct_answer(self, router, sample_deployments):
        """Router immediately provides an answer without routing."""
        mock_response = _make_response(
            "<think>Simple query</think>\n<answer>The answer is 42</answer>"
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_ac:
            mock_ac.return_value = mock_response
            result = await router.route("What is 42?", sample_deployments)

        assert result.answer == "The answer is 42"
        assert result.total_iterations == 1
        assert result.router_model == "gpt-4o-mini"
        assert len(result.models_used) == 0
        assert result.total_tokens == 100

    @pytest.mark.asyncio
    async def test_route_then_answer(self, router, sample_deployments):
        """Router routes to a model, gets result, then answers."""
        router_response_1 = _make_response(
            "<think>Need expert help</think>\n"
            '<route model="gpt-4o">Explain quantum entanglement</route>',
            total_tokens=80,
        )
        sub_response = _make_response(
            "Quantum entanglement is a phenomenon...",
            total_tokens=200,
        )
        router_response_2 = _make_response(
            "<answer>Based on the expert response: "
            "Quantum entanglement is a phenomenon...</answer>",
            total_tokens=120,
        )

        call_count = 0

        async def mock_acompletion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return router_response_1
            elif call_count == 2:
                return sub_response
            else:
                return router_response_2

        with patch("litellm.acompletion", side_effect=mock_acompletion):
            result = await router.route(
                "Explain quantum entanglement", sample_deployments
            )

        assert result.answer.startswith("Based on the expert response")
        assert result.total_iterations == 2
        assert "gpt-4o" in result.models_used
        assert result.total_tokens == 80 + 200 + 120

    @pytest.mark.asyncio
    async def test_sub_query_failure_handled(self, router, sample_deployments):
        """Sub-query failure is captured and returned to the router."""
        router_response_1 = _make_response(
            '<route model="nonexistent-model">test query</route>',
            total_tokens=50,
        )
        router_response_2 = _make_response(
            "<answer>I could not get a response from the model</answer>",
            total_tokens=60,
        )

        call_count = 0

        async def mock_acompletion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return router_response_1
            elif call_count == 2:
                raise ConnectionError("Model not available")
            else:
                return router_response_2

        with patch("litellm.acompletion", side_effect=mock_acompletion):
            result = await router.route("test query", sample_deployments)

        assert result.answer == "I could not get a response from the model"
        assert result.steps[0].result is not None
        assert "Error from nonexistent-model" in result.steps[0].result

    @pytest.mark.asyncio
    async def test_iteration_limit(self, router, sample_deployments):
        """Router stops after max_iterations."""
        # Always return partial reasoning without answer or route
        partial_response = _make_response(
            "<think>Still thinking...</think>\nLet me continue...",
            total_tokens=30,
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_ac:
            mock_ac.return_value = partial_response
            result = await router.route("Complex question", sample_deployments)

        assert result.total_iterations == router._max_iterations
        # Falls back to last think content
        assert result.answer == "Still thinking..."

    @pytest.mark.asyncio
    async def test_timeout_handling(self, sample_deployments):
        """Router handles timeout per iteration."""
        # Use very short timeout
        router = RouterR1(
            router_model="gpt-4o-mini",
            max_iterations=3,
            timeout_per_iteration=0.01,
        )

        async def slow_completion(**kwargs):
            await asyncio.sleep(10)  # Will be cancelled by timeout
            return _make_response("<answer>too late</answer>")

        with patch("litellm.acompletion", side_effect=slow_completion):
            result = await router.route("test", sample_deployments)

        # Should have 1 step (the timed-out one) and no answer
        assert result.total_iterations == 1
        assert result.answer == ""  # No answer produced

    @pytest.mark.asyncio
    async def test_system_message_included(self, router, sample_deployments):
        """System message is passed to the router model."""
        mock_response = _make_response("<answer>done</answer>")

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_ac:
            mock_ac.return_value = mock_response
            await router.route(
                "query",
                sample_deployments,
                system_message="Be helpful",
            )

        call_args = mock_ac.call_args
        messages = call_args.kwargs["messages"]
        # system + system_message_user + query = 3 messages
        assert len(messages) == 3
        assert "[Original system context]" in messages[1]["content"]

    @pytest.mark.asyncio
    async def test_exception_in_router_call(self, router, sample_deployments):
        """Exception from the router LLM is handled gracefully."""
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_ac:
            mock_ac.side_effect = RuntimeError("API error")
            result = await router.route("test", sample_deployments)

        assert result.total_iterations == 1
        assert result.answer == ""  # No answer produced

    @pytest.mark.asyncio
    async def test_no_route_no_answer_prompts_continuation(
        self, router, sample_deployments
    ):
        """When neither route nor answer is found, the router is prompted to continue."""
        partial = _make_response(
            "I'm not sure yet, let me consider...",
            total_tokens=40,
        )
        final = _make_response(
            "<answer>After consideration, the answer is X</answer>",
            total_tokens=60,
        )

        call_count = 0

        async def mock_acompletion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return partial
            return final

        with patch("litellm.acompletion", side_effect=mock_acompletion):
            result = await router.route("test", sample_deployments)

        assert result.answer == "After consideration, the answer is X"
        assert result.total_iterations == 2

    @pytest.mark.asyncio
    async def test_fallback_to_result_when_no_answer(self, router, sample_deployments):
        """If max iterations reached after routing, fallback to last result."""
        router = RouterR1(
            router_model="gpt-4o-mini",
            max_iterations=1,
            timeout_per_iteration=10.0,
        )
        route_response = _make_response(
            '<route model="gpt-4o">What is X?</route>',
            total_tokens=40,
        )
        sub_response = _make_response(
            "X is a variable",
            total_tokens=30,
        )

        call_count = 0

        async def mock_acompletion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return route_response
            return sub_response

        with patch("litellm.acompletion", side_effect=mock_acompletion):
            result = await router.route("What is X?", sample_deployments)

        # Should use the sub-query result as fallback answer
        assert result.answer == "X is a variable"


# ---------------------------------------------------------------------------
# Singleton tests
# ---------------------------------------------------------------------------


class TestSingleton:
    """Tests for get_router_r1 / reset_router_r1 singleton."""

    def test_disabled_by_default(self):
        """Router-R1 is disabled by default."""
        assert get_router_r1() is None

    def test_enabled_via_env(self, monkeypatch):
        """Router-R1 can be enabled via environment variable."""
        monkeypatch.setenv("ROUTEIQ_ROUTER_R1_ENABLED", "true")
        r = get_router_r1()
        assert r is not None
        assert isinstance(r, RouterR1)
        assert r._router_model == "gpt-4o-mini"

    def test_custom_model_via_env(self, monkeypatch):
        """Router model can be configured via env."""
        monkeypatch.setenv("ROUTEIQ_ROUTER_R1_ENABLED", "true")
        monkeypatch.setenv("ROUTEIQ_ROUTER_R1_MODEL", "claude-3-5-sonnet-20241022")
        r = get_router_r1()
        assert r is not None
        assert r._router_model == "claude-3-5-sonnet-20241022"

    def test_custom_iterations_via_env(self, monkeypatch):
        """Max iterations can be configured via env."""
        monkeypatch.setenv("ROUTEIQ_ROUTER_R1_ENABLED", "true")
        monkeypatch.setenv("ROUTEIQ_ROUTER_R1_MAX_ITERATIONS", "5")
        r = get_router_r1()
        assert r is not None
        assert r._max_iterations == 5

    def test_custom_timeout_via_env(self, monkeypatch):
        """Timeout can be configured via env."""
        monkeypatch.setenv("ROUTEIQ_ROUTER_R1_ENABLED", "true")
        monkeypatch.setenv("ROUTEIQ_ROUTER_R1_TIMEOUT", "60")
        r = get_router_r1()
        assert r is not None
        assert r._timeout == 60.0

    def test_singleton_returns_same_instance(self, monkeypatch):
        """get_router_r1 returns the same instance on repeated calls."""
        monkeypatch.setenv("ROUTEIQ_ROUTER_R1_ENABLED", "true")
        r1 = get_router_r1()
        r2 = get_router_r1()
        assert r1 is r2

    def test_reset_clears_singleton(self, monkeypatch):
        """reset_router_r1 clears the cached instance."""
        monkeypatch.setenv("ROUTEIQ_ROUTER_R1_ENABLED", "true")
        r1 = get_router_r1()
        assert r1 is not None
        reset_router_r1()
        # After reset, disabled env returns None
        monkeypatch.setenv("ROUTEIQ_ROUTER_R1_ENABLED", "false")
        assert get_router_r1() is None


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


class TestDataclasses:
    """Tests for RoutingStep and R1Result dataclasses."""

    def test_routing_step_defaults(self):
        step = RoutingStep(iteration=0)
        assert step.iteration == 0
        assert step.think == ""
        assert step.routed_model is None
        assert step.routed_query is None
        assert step.result is None
        assert step.latency_ms == 0.0
        assert step.tokens_used == 0

    def test_r1_result_defaults(self):
        result = R1Result(answer="test")
        assert result.answer == "test"
        assert result.steps == []
        assert result.total_iterations == 0
        assert result.total_tokens == 0
        assert result.total_latency_ms == 0.0
        assert result.router_model == ""
        assert result.models_used == []

    def test_r1_result_independent_lists(self):
        """Each R1Result instance should have independent lists."""
        r1 = R1Result(answer="a")
        r2 = R1Result(answer="b")
        r1.models_used.append("gpt-4o")
        assert len(r2.models_used) == 0

    def test_r1_result_stop_reason_default_empty(self):
        assert R1Result(answer="x").stop_reason == ""


# ---------------------------------------------------------------------------
# RouteIQ-81bc — cost/latency gating + eval-loop feedback
# ---------------------------------------------------------------------------


class TestCostLatencyGating:
    """The iterative router trades cost/latency for quality; an operator caps
    that tradeoff with token / latency budgets (RouteIQ-81bc)."""

    def test_no_gate_by_default(self):
        r = RouterR1()
        assert r._max_total_tokens == 0
        assert r._max_total_latency_ms == 0.0
        res = R1Result(answer="x", total_tokens=10_000)
        # 0 disables both gates -> never trips.
        assert r._budget_tripped(res, start=0.0) is None

    def test_token_budget_trips(self):
        r = RouterR1(max_total_tokens=100)
        res = R1Result(answer="x", total_tokens=150)
        assert r._budget_tripped(res, start=0.0) == "token_budget"

    def test_token_budget_not_tripped_below_cap(self):
        r = RouterR1(max_total_tokens=100)
        res = R1Result(answer="x", total_tokens=50)
        assert r._budget_tripped(res, start=0.0) is None

    def test_latency_budget_trips(self):
        import time

        r = RouterR1(max_total_latency_ms=1.0)
        # start far in the past -> elapsed exceeds the 1ms cap.
        assert (
            r._budget_tripped(R1Result(answer="x"), start=time.monotonic() - 5.0)
            == "latency_budget"
        )

    @pytest.mark.asyncio
    async def test_token_gate_stops_loop_and_sets_stop_reason(self):
        """A token gate set below one round's cost stops the loop BEFORE a second
        round and records the stop_reason."""
        r = RouterR1(
            router_model="gpt-4o-mini",
            max_iterations=5,
            timeout_per_iteration=10.0,
            max_total_tokens=50,  # one round (~80 tokens) trips it next iteration.
        )
        # never produces an <answer> so only the gate can stop the loop.
        partial = _make_response(
            "<think>still working</think>\nLet me continue...", total_tokens=80
        )
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_ac:
            mock_ac.return_value = partial
            result = await r.route("hard query", [{"model_name": "gpt-4o-mini"}])
        # iteration 0 runs (80 tokens), iteration 1's top-of-loop gate trips.
        assert result.stop_reason == "token_budget"
        assert result.total_iterations == 1

    @pytest.mark.asyncio
    async def test_answer_sets_stop_reason_answer(self, router, sample_deployments):
        mock_response = _make_response("<answer>done</answer>")
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_ac:
            mock_ac.return_value = mock_response
            result = await router.route("q", sample_deployments)
        assert result.stop_reason == "answer"

    @pytest.mark.asyncio
    async def test_max_iterations_sets_stop_reason(self, router, sample_deployments):
        partial = _make_response("<think>...</think>", total_tokens=10)
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_ac:
            mock_ac.return_value = partial
            result = await router.route("q", sample_deployments)
        assert result.stop_reason == "max_iterations"

    def test_gating_wired_from_env(self, monkeypatch):
        monkeypatch.setenv("ROUTEIQ_ROUTER_R1_ENABLED", "true")
        monkeypatch.setenv("ROUTEIQ_ROUTER_R1_MAX_TOTAL_TOKENS", "5000")
        monkeypatch.setenv("ROUTEIQ_ROUTER_R1_MAX_TOTAL_LATENCY_MS", "12000")
        r = get_router_r1()
        assert r is not None
        assert r._max_total_tokens == 5000
        assert r._max_total_latency_ms == 12000.0


class TestEvalLoopFeedback:
    """A completed R1 run is handed to the eval pipeline (COLLECT arm) so its
    answer is graded and its cost/latency feed the FEEDBACK loop (RouteIQ-81bc)."""

    @pytest.mark.asyncio
    async def test_run_emits_eval_sample_when_pipeline_enabled(
        self, router, sample_deployments
    ):
        from litellm_llmrouter import eval_pipeline as ep_mod

        collected = []

        class _FakePipeline:
            def should_sample(self):
                return True

            def collect(self, sample):
                collected.append(sample)

        with patch.object(ep_mod, "get_eval_pipeline", lambda: _FakePipeline()):
            with patch("litellm.acompletion", new_callable=AsyncMock) as mock_ac:
                mock_ac.return_value = _make_response(
                    "<answer>the answer is 42</answer>", total_tokens=123
                )
                result = await router.route("what is the answer?", sample_deployments)

        assert len(collected) == 1
        sample = collected[0]
        assert sample.strategy == "router-r1"
        assert sample.response_content == "the answer is 42"
        # observed cost/latency from the run flow into the sample.
        assert sample.response_tokens == 123
        assert sample.latency_ms == result.total_latency_ms

    @pytest.mark.asyncio
    async def test_no_eval_sample_when_pipeline_disabled(
        self, router, sample_deployments
    ):
        from litellm_llmrouter import eval_pipeline as ep_mod

        # get_eval_pipeline returns None when eval is disabled -> no-op.
        with patch.object(ep_mod, "get_eval_pipeline", lambda: None):
            with patch("litellm.acompletion", new_callable=AsyncMock) as mock_ac:
                mock_ac.return_value = _make_response("<answer>ok</answer>")
                # must not raise.
                result = await router.route("q", sample_deployments)
        assert result.answer == "ok"

    @pytest.mark.asyncio
    async def test_eval_emit_failure_does_not_break_routing(
        self, router, sample_deployments
    ):
        from litellm_llmrouter import eval_pipeline as ep_mod

        def _boom():
            raise RuntimeError("eval pipeline exploded")

        with patch.object(ep_mod, "get_eval_pipeline", _boom):
            with patch("litellm.acompletion", new_callable=AsyncMock) as mock_ac:
                mock_ac.return_value = _make_response("<answer>still ok</answer>")
                result = await router.route("q", sample_deployments)
        # routing returns its answer despite the feedback hiccup.
        assert result.answer == "still ok"

    @pytest.mark.asyncio
    async def test_not_collected_when_should_sample_false(
        self, router, sample_deployments
    ):
        from litellm_llmrouter import eval_pipeline as ep_mod

        collected = []

        class _FakePipeline:
            def should_sample(self):
                return False

            def collect(self, sample):  # pragma: no cover - must not be called
                collected.append(sample)

        with patch.object(ep_mod, "get_eval_pipeline", lambda: _FakePipeline()):
            with patch("litellm.acompletion", new_callable=AsyncMock) as mock_ac:
                mock_ac.return_value = _make_response("<answer>ok</answer>")
                await router.route("q", sample_deployments)
        assert collected == []
