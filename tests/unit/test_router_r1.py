"""Unit tests for Router-R1 iterative reasoning router."""

import asyncio
from dataclasses import dataclass
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
