"""
Unit Tests for TG4.1 Router Decision Telemetry
==============================================

Tests for TG4.1 router decision telemetry as first-class signals:
- Span attribute constants are correctly defined
- set_router_decision_attributes helper sets all expected attributes
- Success path emits correct attributes
- Error path emits correct attributes including error_type and error_message
- Fallback path emits correct attributes with fallback_triggered=True

Per TG4.1 acceptance criteria in plans/tg4-observability-epic.md:23-32:
- router.strategy: strategy name (knn, mlp, random)
- router.model_selected: selected model/deployment
- router.score: ML-based strategy score
- router.candidates_evaluated: count of evaluated candidates
"""

from unittest.mock import MagicMock

# Import the module under test directly (not through __init__.py)
import importlib.util

spec = importlib.util.spec_from_file_location(
    "observability", "src/litellm_llmrouter/observability.py"
)
observability = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observability)

# Import TG4.1 constants
ROUTER_STRATEGY_ATTR = observability.ROUTER_STRATEGY_ATTR
ROUTER_MODEL_SELECTED_ATTR = observability.ROUTER_MODEL_SELECTED_ATTR
ROUTER_SCORE_ATTR = observability.ROUTER_SCORE_ATTR
ROUTER_CANDIDATES_EVALUATED_ATTR = observability.ROUTER_CANDIDATES_EVALUATED_ATTR
ROUTER_DECISION_OUTCOME_ATTR = observability.ROUTER_DECISION_OUTCOME_ATTR
ROUTER_DECISION_REASON_ATTR = observability.ROUTER_DECISION_REASON_ATTR
ROUTER_LATENCY_MS_ATTR = observability.ROUTER_LATENCY_MS_ATTR
ROUTER_ERROR_TYPE_ATTR = observability.ROUTER_ERROR_TYPE_ATTR
ROUTER_ERROR_MESSAGE_ATTR = observability.ROUTER_ERROR_MESSAGE_ATTR
ROUTER_VERSION_ATTR = observability.ROUTER_VERSION_ATTR
ROUTER_FALLBACK_TRIGGERED_ATTR = observability.ROUTER_FALLBACK_TRIGGERED_ATTR

# Import the helper function
set_router_decision_attributes = observability.set_router_decision_attributes


class TestTG4_1_SpanAttributeConstants:
    """Test suite for TG4.1 span attribute constant definitions."""

    def test_router_strategy_attr_defined(self):
        """Test that ROUTER_STRATEGY_ATTR is correctly defined."""
        assert ROUTER_STRATEGY_ATTR == "router.strategy"

    def test_router_model_selected_attr_defined(self):
        """Test that ROUTER_MODEL_SELECTED_ATTR is correctly defined."""
        assert ROUTER_MODEL_SELECTED_ATTR == "router.model_selected"

    def test_router_score_attr_defined(self):
        """Test that ROUTER_SCORE_ATTR is correctly defined."""
        assert ROUTER_SCORE_ATTR == "router.score"

    def test_router_candidates_evaluated_attr_defined(self):
        """Test that ROUTER_CANDIDATES_EVALUATED_ATTR is correctly defined."""
        assert ROUTER_CANDIDATES_EVALUATED_ATTR == "router.candidates_evaluated"

    def test_router_decision_outcome_attr_defined(self):
        """Test that ROUTER_DECISION_OUTCOME_ATTR is correctly defined."""
        assert ROUTER_DECISION_OUTCOME_ATTR == "router.decision_outcome"

    def test_router_decision_reason_attr_defined(self):
        """Test that ROUTER_DECISION_REASON_ATTR is correctly defined."""
        assert ROUTER_DECISION_REASON_ATTR == "router.decision_reason"

    def test_router_latency_ms_attr_defined(self):
        """Test that ROUTER_LATENCY_MS_ATTR is correctly defined."""
        assert ROUTER_LATENCY_MS_ATTR == "router.latency_ms"

    def test_router_error_type_attr_defined(self):
        """Test that ROUTER_ERROR_TYPE_ATTR is correctly defined."""
        assert ROUTER_ERROR_TYPE_ATTR == "router.error_type"

    def test_router_error_message_attr_defined(self):
        """Test that ROUTER_ERROR_MESSAGE_ATTR is correctly defined."""
        assert ROUTER_ERROR_MESSAGE_ATTR == "router.error_message"

    def test_router_version_attr_defined(self):
        """Test that ROUTER_VERSION_ATTR is correctly defined."""
        assert ROUTER_VERSION_ATTR == "router.strategy_version"

    def test_router_fallback_triggered_attr_defined(self):
        """Test that ROUTER_FALLBACK_TRIGGERED_ATTR is correctly defined."""
        assert ROUTER_FALLBACK_TRIGGERED_ATTR == "router.fallback_triggered"


class TestSetRouterDecisionAttributes:
    """Test suite for set_router_decision_attributes helper function."""

    def _create_mock_span(self):
        """Create a mock span with is_recording returning True."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        return mock_span

    def test_sets_strategy_attribute(self):
        """Test that strategy attribute is set when provided."""
        mock_span = self._create_mock_span()

        set_router_decision_attributes(mock_span, strategy="knn")

        mock_span.set_attribute.assert_called_with(ROUTER_STRATEGY_ATTR, "knn")

    def test_sets_model_selected_attribute(self):
        """Test that model_selected attribute is set when provided."""
        mock_span = self._create_mock_span()

        set_router_decision_attributes(mock_span, model_selected="gpt-4")

        mock_span.set_attribute.assert_called_with(ROUTER_MODEL_SELECTED_ATTR, "gpt-4")

    def test_sets_score_attribute(self):
        """Test that score attribute is set when provided."""
        mock_span = self._create_mock_span()

        set_router_decision_attributes(mock_span, score=0.95)

        mock_span.set_attribute.assert_called_with(ROUTER_SCORE_ATTR, 0.95)

    def test_sets_candidates_evaluated_attribute(self):
        """Test that candidates_evaluated attribute is set when provided."""
        mock_span = self._create_mock_span()

        set_router_decision_attributes(mock_span, candidates_evaluated=5)

        mock_span.set_attribute.assert_called_with(ROUTER_CANDIDATES_EVALUATED_ATTR, 5)

    def test_sets_outcome_attribute(self):
        """Test that outcome attribute is set when provided."""
        mock_span = self._create_mock_span()

        set_router_decision_attributes(mock_span, outcome="success")

        mock_span.set_attribute.assert_called_with(
            ROUTER_DECISION_OUTCOME_ATTR, "success"
        )

    def test_sets_reason_attribute(self):
        """Test that reason attribute is set when provided."""
        mock_span = self._create_mock_span()

        set_router_decision_attributes(mock_span, reason="strategy_prediction")

        mock_span.set_attribute.assert_called_with(
            ROUTER_DECISION_REASON_ATTR, "strategy_prediction"
        )

    def test_sets_latency_ms_attribute(self):
        """Test that latency_ms attribute is set when provided."""
        mock_span = self._create_mock_span()

        set_router_decision_attributes(mock_span, latency_ms=123.45)

        mock_span.set_attribute.assert_called_with(ROUTER_LATENCY_MS_ATTR, 123.45)

    def test_sets_error_type_attribute(self):
        """Test that error_type attribute is set when provided."""
        mock_span = self._create_mock_span()

        set_router_decision_attributes(mock_span, error_type="ValueError")

        mock_span.set_attribute.assert_called_with(ROUTER_ERROR_TYPE_ATTR, "ValueError")

    def test_sets_error_message_attribute(self):
        """Test that error_message attribute is set when provided."""
        mock_span = self._create_mock_span()

        set_router_decision_attributes(mock_span, error_message="Model not found")

        mock_span.set_attribute.assert_called_with(
            ROUTER_ERROR_MESSAGE_ATTR, "Model not found"
        )

    def test_sets_strategy_version_attribute(self):
        """Test that strategy_version attribute is set when provided."""
        mock_span = self._create_mock_span()

        set_router_decision_attributes(mock_span, strategy_version="v1.2.3")

        mock_span.set_attribute.assert_called_with(ROUTER_VERSION_ATTR, "v1.2.3")

    def test_sets_fallback_triggered_attribute(self):
        """Test that fallback_triggered attribute is set when provided."""
        mock_span = self._create_mock_span()

        set_router_decision_attributes(mock_span, fallback_triggered=True)

        mock_span.set_attribute.assert_called_with(ROUTER_FALLBACK_TRIGGERED_ATTR, True)

    def test_does_not_set_none_values(self):
        """Test that None values do not result in set_attribute calls."""
        mock_span = self._create_mock_span()

        set_router_decision_attributes(mock_span)

        mock_span.set_attribute.assert_not_called()

    def test_skips_non_recording_span(self):
        """Test that non-recording spans are skipped."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = False

        set_router_decision_attributes(mock_span, strategy="knn")

        mock_span.set_attribute.assert_not_called()

    def test_skips_none_span(self):
        """Test that None span is handled gracefully."""
        # Should not raise
        set_router_decision_attributes(None, strategy="knn")


class TestSetRouterDecisionAttributesSuccessPath:
    """Test suite for set_router_decision_attributes success path scenarios."""

    def _create_mock_span(self):
        """Create a mock span with is_recording returning True."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        return mock_span

    def test_success_path_sets_all_required_attributes(self):
        """Test that success path sets all TG4.1 required attributes."""
        mock_span = self._create_mock_span()
        attributes_set = {}

        def capture_set_attribute(key, value):
            attributes_set[key] = value

        mock_span.set_attribute.side_effect = capture_set_attribute

        set_router_decision_attributes(
            mock_span,
            strategy="knn",
            model_selected="gpt-4",
            candidates_evaluated=5,
            outcome="success",
            reason="strategy_prediction",
            latency_ms=42.5,
            fallback_triggered=False,
        )

        # Verify all TG4.1 required attributes are present
        assert attributes_set[ROUTER_STRATEGY_ATTR] == "knn"
        assert attributes_set[ROUTER_MODEL_SELECTED_ATTR] == "gpt-4"
        assert attributes_set[ROUTER_CANDIDATES_EVALUATED_ATTR] == 5
        assert attributes_set[ROUTER_DECISION_OUTCOME_ATTR] == "success"
        assert attributes_set[ROUTER_DECISION_REASON_ATTR] == "strategy_prediction"
        assert attributes_set[ROUTER_LATENCY_MS_ATTR] == 42.5
        assert attributes_set[ROUTER_FALLBACK_TRIGGERED_ATTR] is False

    def test_success_path_with_score(self):
        """Test success path with optional ML score attribute."""
        mock_span = self._create_mock_span()
        attributes_set = {}

        def capture_set_attribute(key, value):
            attributes_set[key] = value

        mock_span.set_attribute.side_effect = capture_set_attribute

        set_router_decision_attributes(
            mock_span,
            strategy="mlp",
            model_selected="claude-3-opus",
            score=0.92,
            candidates_evaluated=3,
            outcome="success",
        )

        assert attributes_set[ROUTER_STRATEGY_ATTR] == "mlp"
        assert attributes_set[ROUTER_MODEL_SELECTED_ATTR] == "claude-3-opus"
        assert attributes_set[ROUTER_SCORE_ATTR] == 0.92
        assert attributes_set[ROUTER_CANDIDATES_EVALUATED_ATTR] == 3
        assert attributes_set[ROUTER_DECISION_OUTCOME_ATTR] == "success"


class TestSetRouterDecisionAttributesErrorPath:
    """Test suite for set_router_decision_attributes error path scenarios."""

    def _create_mock_span(self):
        """Create a mock span with is_recording returning True."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        return mock_span

    def test_error_path_sets_all_required_attributes(self):
        """Test that error path sets all TG4.1 required attributes including error info."""
        mock_span = self._create_mock_span()
        attributes_set = {}

        def capture_set_attribute(key, value):
            attributes_set[key] = value

        mock_span.set_attribute.side_effect = capture_set_attribute

        set_router_decision_attributes(
            mock_span,
            strategy="knn",
            model_selected=None,  # No model selected due to error
            candidates_evaluated=5,
            outcome="error",
            reason="strategy_error: ValueError",
            latency_ms=15.3,
            error_type="ValueError",
            error_message="Model file not found",
            fallback_triggered=False,
        )

        # Verify all TG4.1 required attributes for error path
        assert attributes_set[ROUTER_STRATEGY_ATTR] == "knn"
        assert attributes_set[ROUTER_CANDIDATES_EVALUATED_ATTR] == 5
        assert attributes_set[ROUTER_DECISION_OUTCOME_ATTR] == "error"
        assert (
            attributes_set[ROUTER_DECISION_REASON_ATTR] == "strategy_error: ValueError"
        )
        assert attributes_set[ROUTER_LATENCY_MS_ATTR] == 15.3
        assert attributes_set[ROUTER_ERROR_TYPE_ATTR] == "ValueError"
        assert attributes_set[ROUTER_ERROR_MESSAGE_ATTR] == "Model file not found"
        assert attributes_set[ROUTER_FALLBACK_TRIGGERED_ATTR] is False

        # model_selected should not be set (None value)
        assert ROUTER_MODEL_SELECTED_ATTR not in attributes_set

    def test_error_path_with_fallback_triggered(self):
        """Test error path with fallback triggered."""
        mock_span = self._create_mock_span()
        attributes_set = {}

        def capture_set_attribute(key, value):
            attributes_set[key] = value

        mock_span.set_attribute.side_effect = capture_set_attribute

        set_router_decision_attributes(
            mock_span,
            strategy="mlp",
            model_selected="gpt-3.5-turbo",  # Fallback model selected
            candidates_evaluated=3,
            outcome="fallback",
            reason="primary_failed: ConnectionError",
            latency_ms=250.0,
            error_type="ConnectionError",
            error_message="Failed to connect to ML router service",
            fallback_triggered=True,
        )

        assert attributes_set[ROUTER_STRATEGY_ATTR] == "mlp"
        assert attributes_set[ROUTER_MODEL_SELECTED_ATTR] == "gpt-3.5-turbo"
        assert attributes_set[ROUTER_CANDIDATES_EVALUATED_ATTR] == 3
        assert attributes_set[ROUTER_DECISION_OUTCOME_ATTR] == "fallback"
        assert attributes_set[ROUTER_FALLBACK_TRIGGERED_ATTR] is True


class TestSetRouterDecisionAttributesNoCandidatesPath:
    """Test suite for set_router_decision_attributes no_candidates path scenarios."""

    def _create_mock_span(self):
        """Create a mock span with is_recording returning True."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        return mock_span

    def test_no_candidates_path_sets_correct_attributes(self):
        """Test that no_candidates path sets appropriate attributes."""
        mock_span = self._create_mock_span()
        attributes_set = {}

        def capture_set_attribute(key, value):
            attributes_set[key] = value

        mock_span.set_attribute.side_effect = capture_set_attribute

        set_router_decision_attributes(
            mock_span,
            strategy="random",
            model_selected=None,
            candidates_evaluated=0,
            outcome="no_candidates",
            reason="no_candidates_available",
            latency_ms=1.5,
            fallback_triggered=False,
        )

        assert attributes_set[ROUTER_STRATEGY_ATTR] == "random"
        assert attributes_set[ROUTER_CANDIDATES_EVALUATED_ATTR] == 0
        assert attributes_set[ROUTER_DECISION_OUTCOME_ATTR] == "no_candidates"
        assert attributes_set[ROUTER_DECISION_REASON_ATTR] == "no_candidates_available"
        assert ROUTER_MODEL_SELECTED_ATTR not in attributes_set


class TestRouterDecisionTelemetryIntegration:
    """Integration-style tests for router decision telemetry emission."""

    def test_telemetry_attributes_count(self):
        """Test that the expected number of attributes are set for complete success path."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        call_count = 0

        def count_calls(key, value):
            nonlocal call_count
            call_count += 1

        mock_span.set_attribute.side_effect = count_calls

        set_router_decision_attributes(
            mock_span,
            strategy="knn",
            model_selected="gpt-4",
            score=0.95,
            candidates_evaluated=5,
            outcome="success",
            reason="strategy_prediction",
            latency_ms=42.5,
            strategy_version="abc12345",
            fallback_triggered=False,
        )

        # All 9 attributes should be set
        assert call_count == 9

    def test_telemetry_attributes_for_error_path(self):
        """Test that error path sets appropriate number of attributes."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        call_count = 0

        def count_calls(key, value):
            nonlocal call_count
            call_count += 1

        mock_span.set_attribute.side_effect = count_calls

        set_router_decision_attributes(
            mock_span,
            strategy="mlp",
            # model_selected=None (not set)
            candidates_evaluated=3,
            outcome="error",
            reason="strategy_error",
            latency_ms=15.0,
            error_type="ValueError",
            error_message="Something went wrong",
            fallback_triggered=True,
        )

        # 8 attributes should be set (no model_selected)
        assert call_count == 8


# Import the GenAI attributes helper
set_genai_attributes = observability.set_genai_attributes


class TestSetGenAIAttributes:
    """Test suite for set_genai_attributes helper function."""

    def _create_mock_span(self):
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        return mock_span

    def test_sets_system_attribute(self):
        mock_span = self._create_mock_span()
        set_genai_attributes(mock_span, system="openai")
        mock_span.set_attribute.assert_called_with("gen_ai.system", "openai")

    def test_sets_request_model_attribute(self):
        mock_span = self._create_mock_span()
        set_genai_attributes(mock_span, request_model="gpt-4")
        mock_span.set_attribute.assert_called_with("gen_ai.request.model", "gpt-4")

    def test_sets_response_model_attribute(self):
        mock_span = self._create_mock_span()
        set_genai_attributes(mock_span, response_model="gpt-4-0613")
        mock_span.set_attribute.assert_called_with(
            "gen_ai.response.model", "gpt-4-0613"
        )

    def test_sets_operation_name_attribute(self):
        mock_span = self._create_mock_span()
        set_genai_attributes(mock_span, operation_name="chat_completion")
        mock_span.set_attribute.assert_called_with(
            "gen_ai.operation.name", "chat_completion"
        )

    def test_sets_input_tokens_attribute(self):
        mock_span = self._create_mock_span()
        set_genai_attributes(mock_span, input_tokens=100)
        mock_span.set_attribute.assert_called_with("gen_ai.usage.input_tokens", 100)

    def test_sets_output_tokens_attribute(self):
        mock_span = self._create_mock_span()
        set_genai_attributes(mock_span, output_tokens=250)
        mock_span.set_attribute.assert_called_with("gen_ai.usage.output_tokens", 250)

    def test_sets_finish_reasons_attribute(self):
        mock_span = self._create_mock_span()
        set_genai_attributes(mock_span, finish_reasons=["stop"])
        mock_span.set_attribute.assert_called_with(
            "gen_ai.response.finish_reasons", ["stop"]
        )

    def test_sets_all_attributes(self):
        mock_span = self._create_mock_span()
        attrs_set = {}

        def capture(key, value):
            attrs_set[key] = value

        mock_span.set_attribute.side_effect = capture

        set_genai_attributes(
            mock_span,
            system="openai",
            request_model="gpt-4",
            response_model="gpt-4-0613",
            operation_name="chat_completion",
            input_tokens=100,
            output_tokens=250,
            finish_reasons=["stop"],
        )

        assert attrs_set["gen_ai.system"] == "openai"
        assert attrs_set["gen_ai.request.model"] == "gpt-4"
        assert attrs_set["gen_ai.response.model"] == "gpt-4-0613"
        assert attrs_set["gen_ai.operation.name"] == "chat_completion"
        assert attrs_set["gen_ai.usage.input_tokens"] == 100
        assert attrs_set["gen_ai.usage.output_tokens"] == 250
        assert attrs_set["gen_ai.response.finish_reasons"] == ["stop"]

    def test_does_not_set_none_values(self):
        mock_span = self._create_mock_span()
        set_genai_attributes(mock_span)
        mock_span.set_attribute.assert_not_called()

    def test_skips_non_recording_span(self):
        mock_span = MagicMock()
        mock_span.is_recording.return_value = False
        set_genai_attributes(mock_span, system="openai")
        mock_span.set_attribute.assert_not_called()

    def test_skips_none_span(self):
        # Should not raise
        set_genai_attributes(None, system="openai")


class TestGenAIRequestAttributesContract:
    """Test GenAIRequestAttributes dataclass from telemetry_contracts."""

    def test_default_values(self):
        from litellm_llmrouter.telemetry_contracts import GenAIRequestAttributes

        attrs = GenAIRequestAttributes()
        assert attrs.gen_ai_system == ""
        assert attrs.gen_ai_request_model == ""
        assert attrs.gen_ai_response_model == ""
        assert attrs.gen_ai_operation_name == ""
        assert attrs.gen_ai_usage_input_tokens == 0
        assert attrs.gen_ai_usage_output_tokens == 0
        assert attrs.gen_ai_response_finish_reasons == []

    def test_to_span_attributes_populated(self):
        from litellm_llmrouter.telemetry_contracts import GenAIRequestAttributes

        attrs = GenAIRequestAttributes(
            gen_ai_system="openai",
            gen_ai_request_model="gpt-4",
            gen_ai_response_model="gpt-4-0613",
            gen_ai_operation_name="chat_completion",
            gen_ai_usage_input_tokens=100,
            gen_ai_usage_output_tokens=250,
            gen_ai_response_finish_reasons=["stop"],
        )
        span_attrs = attrs.to_span_attributes()
        assert span_attrs["gen_ai.system"] == "openai"
        assert span_attrs["gen_ai.request.model"] == "gpt-4"
        assert span_attrs["gen_ai.response.model"] == "gpt-4-0613"
        assert span_attrs["gen_ai.operation.name"] == "chat_completion"
        assert span_attrs["gen_ai.usage.input_tokens"] == 100
        assert span_attrs["gen_ai.usage.output_tokens"] == 250
        assert span_attrs["gen_ai.response.finish_reasons"] == ["stop"]

    def test_to_span_attributes_omits_defaults(self):
        from litellm_llmrouter.telemetry_contracts import GenAIRequestAttributes

        attrs = GenAIRequestAttributes(gen_ai_system="anthropic")
        span_attrs = attrs.to_span_attributes()
        assert span_attrs == {"gen_ai.system": "anthropic"}
        assert "gen_ai.request.model" not in span_attrs
        assert "gen_ai.usage.input_tokens" not in span_attrs
