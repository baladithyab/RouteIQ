"""
Tests for versioned telemetry contracts (routeiq.router_decision.v1).

These tests verify:
- Contract schema construction
- Builder pattern API
- JSON serialization/deserialization
- Extraction from span event attributes
"""

import json

from litellm_llmrouter.telemetry_contracts import (
    CONTRACT_VERSION,
    CONTRACT_NAME,
    CONTRACT_FULL_NAME,
    ROUTER_DECISION_EVENT_NAME,
    ROUTER_DECISION_PAYLOAD_KEY,
    RoutingOutcome,
    RouterDecisionInput,
    CandidateDeployment,
    RoutingTimings,
    RouterDecisionEvent,
    RouterDecisionEventBuilder,
    extract_router_decision_from_span_event,
)


class TestContractConstants:
    """Test contract constants are properly defined."""

    def test_contract_version(self):
        """Verify contract version is v1."""
        assert CONTRACT_VERSION == "v1"

    def test_contract_name(self):
        """Verify contract name."""
        assert CONTRACT_NAME == "routeiq.router_decision"

    def test_contract_full_name(self):
        """Verify full contract name includes version."""
        assert CONTRACT_FULL_NAME == "routeiq.router_decision.v1"

    def test_event_name(self):
        """Verify span event name matches contract."""
        assert ROUTER_DECISION_EVENT_NAME == CONTRACT_FULL_NAME

    def test_payload_key(self):
        """Verify payload key format."""
        assert ROUTER_DECISION_PAYLOAD_KEY == "routeiq.router_decision.payload"


class TestRoutingOutcome:
    """Test RoutingOutcome enum."""

    def test_outcome_values(self):
        """Verify all outcome values are strings."""
        assert RoutingOutcome.SUCCESS.value == "success"
        assert RoutingOutcome.FAILURE.value == "failure"
        assert RoutingOutcome.FALLBACK.value == "fallback"
        assert RoutingOutcome.NO_CANDIDATES.value == "no_candidates"
        assert RoutingOutcome.TIMEOUT.value == "timeout"
        assert RoutingOutcome.ERROR.value == "error"


class TestRouterDecisionInput:
    """Test RouterDecisionInput dataclass."""

    def test_default_values(self):
        """Verify default values."""
        inp = RouterDecisionInput()
        assert inp.requested_model is None
        assert inp.query_length == 0
        assert inp.user_id is None
        assert inp.team_id is None
        assert inp.request_metadata == {}

    def test_custom_values(self):
        """Verify custom values are set."""
        inp = RouterDecisionInput(
            requested_model="gpt-4",
            query_length=150,
            user_id="user-123",
            team_id="team-456",
            request_metadata={"custom": "value"},
        )
        assert inp.requested_model == "gpt-4"
        assert inp.query_length == 150
        assert inp.user_id == "user-123"
        assert inp.team_id == "team-456"
        assert inp.request_metadata == {"custom": "value"}


class TestCandidateDeployment:
    """Test CandidateDeployment dataclass."""

    def test_required_model_name(self):
        """Verify model_name is required."""
        candidate = CandidateDeployment(model_name="gpt-4")
        assert candidate.model_name == "gpt-4"
        assert candidate.provider is None
        assert candidate.score is None
        assert candidate.available is True

    def test_full_candidate(self):
        """Verify all fields are set."""
        candidate = CandidateDeployment(
            model_name="claude-3-opus",
            provider="anthropic",
            score=0.95,
            available=True,
        )
        assert candidate.model_name == "claude-3-opus"
        assert candidate.provider == "anthropic"
        assert candidate.score == 0.95
        assert candidate.available is True


class TestRoutingTimings:
    """Test RoutingTimings dataclass."""

    def test_default_values(self):
        """Verify default timing values."""
        timings = RoutingTimings()
        assert timings.total_ms == 0.0
        assert timings.strategy_ms is None
        assert timings.embedding_ms is None
        assert timings.candidate_filter_ms is None

    def test_custom_timings(self):
        """Verify custom timing values."""
        timings = RoutingTimings(
            total_ms=15.5,
            strategy_ms=10.2,
            embedding_ms=3.1,
            candidate_filter_ms=2.2,
        )
        assert timings.total_ms == 15.5
        assert timings.strategy_ms == 10.2
        assert timings.embedding_ms == 3.1
        assert timings.candidate_filter_ms == 2.2


class TestRouterDecisionEvent:
    """Test RouterDecisionEvent dataclass."""

    def test_default_event(self):
        """Verify default event has correct contract info."""
        event = RouterDecisionEvent()
        assert event.contract_version == CONTRACT_VERSION
        assert event.contract_name == CONTRACT_FULL_NAME
        assert event.event_id is not None
        assert len(event.event_id) > 0
        assert event.timestamp_utc is not None
        assert event.timestamp_unix_ms > 0

    def test_to_dict(self):
        """Verify to_dict serialization."""
        event = RouterDecisionEvent(
            strategy_name="llmrouter-knn",
            selected_deployment="gpt-4",
        )
        data = event.to_dict()
        assert isinstance(data, dict)
        assert data["contract_version"] == CONTRACT_VERSION
        assert data["contract_name"] == CONTRACT_FULL_NAME
        assert data["strategy_name"] == "llmrouter-knn"
        assert data["selected_deployment"] == "gpt-4"

    def test_to_json(self):
        """Verify to_json serialization."""
        event = RouterDecisionEvent(
            strategy_name="llmrouter-knn",
            selected_deployment="gpt-4",
        )
        json_str = event.to_json()
        assert isinstance(json_str, str)
        # Parse back and verify
        data = json.loads(json_str)
        assert data["strategy_name"] == "llmrouter-knn"
        assert data["selected_deployment"] == "gpt-4"


class TestRouterDecisionEventBuilder:
    """Test RouterDecisionEventBuilder fluent API."""

    def test_builder_strategy(self):
        """Test with_strategy method."""
        event = (
            RouterDecisionEventBuilder()
            .with_strategy("llmrouter-knn", version="1.0.0")
            .build()
        )
        assert event.strategy_name == "llmrouter-knn"
        assert event.strategy_version == "1.0.0"

    def test_builder_trace_context(self):
        """Test with_trace_context method."""
        event = (
            RouterDecisionEventBuilder()
            .with_trace_context(
                trace_id="abc123",
                span_id="def456",
                parent_span_id="ghi789",
            )
            .build()
        )
        assert event.trace_id == "abc123"
        assert event.span_id == "def456"
        assert event.parent_span_id == "ghi789"

    def test_builder_input(self):
        """Test with_input method."""
        event = (
            RouterDecisionEventBuilder()
            .with_input(
                query_length=150,
                requested_model="gpt-4",
                user_id="user-123",
                team_id="team-456",
                metadata={"source": "test"},
            )
            .build()
        )
        assert event.input.query_length == 150
        assert event.input.requested_model == "gpt-4"
        assert event.input.user_id == "user-123"
        assert event.input.team_id == "team-456"
        assert event.input.request_metadata == {"source": "test"}

    def test_builder_candidates(self):
        """Test with_candidates method."""
        candidates = [
            {"model_name": "gpt-4", "provider": "openai", "score": 0.95},
            {"model_name": "claude-3", "provider": "anthropic", "score": 0.82},
        ]
        event = RouterDecisionEventBuilder().with_candidates(candidates).build()
        assert len(event.candidate_deployments) == 2
        assert event.candidate_deployments[0].model_name == "gpt-4"
        assert event.candidate_deployments[0].score == 0.95
        assert event.candidate_deployments[1].model_name == "claude-3"
        assert event.candidate_deployments[1].score == 0.82

    def test_builder_selection(self):
        """Test with_selection method."""
        event = (
            RouterDecisionEventBuilder()
            .with_selection(selected="gpt-4", reason="highest_score")
            .build()
        )
        assert event.selected_deployment == "gpt-4"
        assert event.selection_reason == "highest_score"

    def test_builder_timing(self):
        """Test with_timing method."""
        event = (
            RouterDecisionEventBuilder()
            .with_timing(
                total_ms=15.5,
                strategy_ms=10.2,
                embedding_ms=3.1,
                candidate_filter_ms=2.2,
            )
            .build()
        )
        assert event.timings.total_ms == 15.5
        assert event.timings.strategy_ms == 10.2
        assert event.timings.embedding_ms == 3.1
        assert event.timings.candidate_filter_ms == 2.2

    def test_builder_outcome_success(self):
        """Test with_outcome method for success."""
        event = (
            RouterDecisionEventBuilder()
            .with_outcome(
                status=RoutingOutcome.SUCCESS,
                input_tokens=100,
                output_tokens=200,
                total_tokens=300,
            )
            .build()
        )
        assert event.outcome.status == RoutingOutcome.SUCCESS
        assert event.outcome.input_tokens == 100
        assert event.outcome.output_tokens == 200
        assert event.outcome.total_tokens == 300

    def test_builder_outcome_error(self):
        """Test with_outcome method for error."""
        event = (
            RouterDecisionEventBuilder()
            .with_outcome(
                status=RoutingOutcome.ERROR,
                error_type="TimeoutError",
                error_message="Request timed out",
            )
            .build()
        )
        assert event.outcome.status == RoutingOutcome.ERROR
        assert event.outcome.error_type == "TimeoutError"
        assert event.outcome.error_message == "Request timed out"

    def test_builder_fallback(self):
        """Test with_fallback method."""
        event = (
            RouterDecisionEventBuilder()
            .with_fallback(
                triggered=True,
                original_model="gpt-4",
                reason="rate_limit",
                attempt=1,
            )
            .build()
        )
        assert event.fallback.fallback_triggered is True
        assert event.fallback.original_model == "gpt-4"
        assert event.fallback.fallback_reason == "rate_limit"
        assert event.fallback.fallback_attempt == 1

    def test_builder_custom_attributes(self):
        """Test with_custom_attributes method."""
        event = (
            RouterDecisionEventBuilder()
            .with_custom_attributes({"custom_key": "custom_value"})
            .build()
        )
        assert event.custom_attributes == {"custom_key": "custom_value"}

    def test_builder_full_chain(self):
        """Test full builder chain."""
        event = (
            RouterDecisionEventBuilder()
            .with_strategy("llmrouter-knn", version="1.0.0")
            .with_trace_context(trace_id="abc123", span_id="def456")
            .with_input(query_length=150, user_id="user-123")
            .with_candidates(
                [
                    {"model_name": "gpt-4", "score": 0.95},
                    {"model_name": "claude-3", "score": 0.82},
                ]
            )
            .with_selection(selected="gpt-4", reason="highest_score")
            .with_timing(total_ms=15.5, strategy_ms=10.2)
            .with_outcome(status=RoutingOutcome.SUCCESS)
            .build()
        )
        # Verify all fields are set correctly
        assert event.strategy_name == "llmrouter-knn"
        assert event.trace_id == "abc123"
        assert event.input.query_length == 150
        assert len(event.candidate_deployments) == 2
        assert event.selected_deployment == "gpt-4"
        assert event.timings.total_ms == 15.5
        assert event.outcome.status == RoutingOutcome.SUCCESS


class TestExtractRouterDecision:
    """Test extraction of RouterDecisionEvent from span event attributes."""

    def test_extract_valid_event(self):
        """Extract from valid span event attributes."""
        # Build an event
        original = (
            RouterDecisionEventBuilder()
            .with_strategy("llmrouter-knn", version="1.0.0")
            .with_input(query_length=150)
            .with_candidates(
                [
                    {"model_name": "gpt-4", "score": 0.95},
                ]
            )
            .with_selection(selected="gpt-4")
            .with_timing(total_ms=15.5)
            .with_outcome(status=RoutingOutcome.SUCCESS)
            .build()
        )

        # Simulate span event attributes
        attributes = {
            ROUTER_DECISION_PAYLOAD_KEY: original.to_json(),
        }

        # Extract and verify
        extracted = extract_router_decision_from_span_event(attributes)
        assert extracted is not None
        assert extracted.contract_name == CONTRACT_FULL_NAME
        assert extracted.strategy_name == "llmrouter-knn"
        assert extracted.selected_deployment == "gpt-4"
        assert extracted.input.query_length == 150

    def test_extract_missing_payload(self):
        """Return None when payload key is missing."""
        attributes = {"other_key": "value"}
        extracted = extract_router_decision_from_span_event(attributes)
        assert extracted is None

    def test_extract_invalid_json(self):
        """Return None for invalid JSON."""
        attributes = {
            ROUTER_DECISION_PAYLOAD_KEY: "not valid json{",
        }
        extracted = extract_router_decision_from_span_event(attributes)
        assert extracted is None

    def test_extract_wrong_contract(self):
        """Return None for wrong contract name."""
        attributes = {
            ROUTER_DECISION_PAYLOAD_KEY: json.dumps(
                {
                    "contract_name": "other.contract.v1",
                    "strategy_name": "test",
                }
            ),
        }
        extracted = extract_router_decision_from_span_event(attributes)
        assert extracted is None

    def test_extract_full_event(self):
        """Test extracting a fully populated event."""
        original = (
            RouterDecisionEventBuilder()
            .with_strategy("llmrouter-knn")
            .with_candidates(
                [
                    {"model_name": "gpt-4", "provider": "openai", "score": 0.95},
                    {"model_name": "claude-3", "provider": "anthropic", "score": 0.82},
                ]
            )
            .with_timing(total_ms=15.5, strategy_ms=10.2)
            .with_outcome(
                status=RoutingOutcome.ERROR,
                error_type="TestError",
                error_message="Something went wrong",
            )
            .with_fallback(triggered=True, original_model="gpt-4", reason="error")
            .build()
        )

        attributes = {
            ROUTER_DECISION_PAYLOAD_KEY: original.to_json(),
        }

        extracted = extract_router_decision_from_span_event(attributes)
        assert extracted is not None

        # Check candidates
        assert len(extracted.candidate_deployments) == 2
        assert extracted.candidate_deployments[0].model_name == "gpt-4"
        assert extracted.candidate_deployments[0].provider == "openai"

        # Check timings
        assert extracted.timings.total_ms == 15.5
        assert extracted.timings.strategy_ms == 10.2

        # Check outcome
        assert extracted.outcome.status == RoutingOutcome.ERROR
        assert extracted.outcome.error_type == "TestError"

        # Check fallback
        assert extracted.fallback.fallback_triggered is True
        assert extracted.fallback.original_model == "gpt-4"


class TestPIISafety:
    """Test that the contract enforces PII safety."""

    def test_no_query_content_in_input(self):
        """Verify input doesn't store query content."""
        inp = RouterDecisionInput(query_length=150)
        # RouterDecisionInput has no field for query content
        assert not hasattr(inp, "query")
        assert not hasattr(inp, "query_content")
        assert not hasattr(inp, "prompt")

    def test_builder_only_accepts_query_length(self):
        """Verify builder only accepts query_length, not content."""
        event = RouterDecisionEventBuilder().with_input(query_length=150).build()
        assert event.input.query_length == 150
        # The input dataclass doesn't have a field for query content
        data = event.to_dict()
        assert "query" not in data["input"]
        assert "prompt" not in data["input"]
