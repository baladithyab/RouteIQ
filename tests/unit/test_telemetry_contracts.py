"""
Unit tests for telemetry contracts.

Tests:
1. RouterDecisionEvent emission fields are stable
2. Extraction correctly parses emitted events from representative traces
3. Round-trip serialization works correctly
"""

import json
import pytest

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
    RoutingOutcomeData,
    FallbackInfo,
    RouterDecisionEvent,
    RouterDecisionEventBuilder,
    extract_router_decision_from_span_event,
)


class TestContractConstants:
    """Test that contract constants remain stable for backward compatibility."""

    def test_contract_version(self):
        """Contract version should be v1."""
        assert CONTRACT_VERSION == "v1"

    def test_contract_name(self):
        """Contract name should follow expected pattern."""
        assert CONTRACT_NAME == "routeiq.router_decision"
        assert CONTRACT_FULL_NAME == "routeiq.router_decision.v1"

    def test_event_name(self):
        """Event name for span events should match contract."""
        assert ROUTER_DECISION_EVENT_NAME == "routeiq.router_decision.v1"

    def test_payload_key(self):
        """Payload key for span event attributes should match contract."""
        assert ROUTER_DECISION_PAYLOAD_KEY == "routeiq.router_decision.payload"


class TestRoutingOutcome:
    """Test RoutingOutcome enum values."""

    def test_all_outcomes_defined(self):
        """All expected routing outcomes should be defined."""
        expected = {
            "success",
            "failure",
            "fallback",
            "error",
            "no_candidates",
            "timeout",
        }
        actual = {o.value for o in RoutingOutcome}
        assert actual == expected

    def test_outcome_string_values(self):
        """Outcome enum values should be lowercase strings."""
        assert RoutingOutcome.SUCCESS.value == "success"
        assert RoutingOutcome.FAILURE.value == "failure"
        assert RoutingOutcome.FALLBACK.value == "fallback"
        assert RoutingOutcome.ERROR.value == "error"
        assert RoutingOutcome.NO_CANDIDATES.value == "no_candidates"
        assert RoutingOutcome.TIMEOUT.value == "timeout"


class TestRouterDecisionEvent:
    """Test RouterDecisionEvent dataclass."""

    def test_default_event_has_required_fields(self):
        """Default event should have all required fields populated."""
        event = RouterDecisionEvent()

        # Identity & versioning
        assert event.contract_version == CONTRACT_VERSION
        assert event.contract_name == CONTRACT_FULL_NAME
        assert event.event_id  # Should be auto-generated UUID

        # Timestamps
        assert event.timestamp_utc
        assert event.timestamp_unix_ms > 0

        # Nested objects should be initialized
        assert isinstance(event.input, RouterDecisionInput)
        assert isinstance(event.timings, RoutingTimings)
        assert isinstance(event.outcome, RoutingOutcomeData)
        assert isinstance(event.fallback, FallbackInfo)

    def test_to_dict_serialization(self):
        """Event should serialize to dict correctly."""
        event = RouterDecisionEvent(
            strategy_name="llmrouter-knn",
            selected_deployment="gpt-4",
        )

        data = event.to_dict()

        assert data["contract_version"] == "v1"
        assert data["contract_name"] == "routeiq.router_decision.v1"
        assert data["strategy_name"] == "llmrouter-knn"
        assert data["selected_deployment"] == "gpt-4"
        assert "input" in data
        assert "timings" in data
        assert "outcome" in data

    def test_to_json_serialization(self):
        """Event should serialize to valid JSON."""
        event = RouterDecisionEvent(
            strategy_name="llmrouter-knn",
            selected_deployment="gpt-4",
        )

        json_str = event.to_json()
        parsed = json.loads(json_str)

        assert parsed["contract_name"] == "routeiq.router_decision.v1"
        assert parsed["strategy_name"] == "llmrouter-knn"

    def test_all_fields_serializable(self):
        """All fields including nested objects should be JSON serializable."""
        event = RouterDecisionEvent(
            strategy_name="llmrouter-knn",
            strategy_version="1.0.0",
            selected_deployment="gpt-4",
            selection_reason="highest_score",
            trace_id="abc123",
            span_id="def456",
            custom_attributes={"env": "production", "region": "us-east-1"},
        )
        event.input = RouterDecisionInput(
            requested_model="gpt-4",
            query_length=150,
            user_id="user-hash",
            team_id="team-123",
        )
        event.candidate_deployments = [
            CandidateDeployment(
                model_name="gpt-4", provider="openai", score=0.95, available=True
            ),
            CandidateDeployment(
                model_name="claude-3", provider="anthropic", score=0.82, available=True
            ),
        ]
        event.timings = RoutingTimings(
            total_ms=15.5, strategy_ms=10.2, embedding_ms=3.1, candidate_filter_ms=2.2
        )
        event.outcome = RoutingOutcomeData(
            status=RoutingOutcome.SUCCESS,
            input_tokens=100,
            output_tokens=200,
            total_tokens=300,
        )

        # Should not raise
        json_str = event.to_json()
        parsed = json.loads(json_str)

        # Verify nested structures
        assert parsed["input"]["query_length"] == 150
        assert len(parsed["candidate_deployments"]) == 2
        assert parsed["timings"]["total_ms"] == 15.5
        assert parsed["outcome"]["status"] == "success"
        assert parsed["custom_attributes"]["env"] == "production"


class TestRouterDecisionEventBuilder:
    """Test the fluent builder API."""

    def test_builder_creates_valid_event(self):
        """Builder should create a valid event with all methods."""
        event = (
            RouterDecisionEventBuilder()
            .with_strategy("llmrouter-knn", version="1.0.0")
            .with_trace_context(trace_id="abc123", span_id="def456")
            .with_input(query_length=150, requested_model="gpt-4", user_id="user-1")
            .with_candidates(
                [
                    {"model_name": "gpt-4", "provider": "openai", "score": 0.95},
                    {"model_name": "claude-3", "provider": "anthropic", "score": 0.82},
                ]
            )
            .with_selection(selected="gpt-4", reason="highest_score")
            .with_timing(total_ms=15.5, strategy_ms=10.2)
            .with_outcome(
                status=RoutingOutcome.SUCCESS,
                input_tokens=100,
                output_tokens=200,
                total_tokens=300,
            )
            .with_fallback(triggered=False)
            .with_custom_attributes({"env": "production"})
            .build()
        )

        assert event.strategy_name == "llmrouter-knn"
        assert event.strategy_version == "1.0.0"
        assert event.trace_id == "abc123"
        assert event.input.query_length == 150
        assert len(event.candidate_deployments) == 2
        assert event.selected_deployment == "gpt-4"
        assert event.timings.total_ms == 15.5
        assert event.outcome.status == RoutingOutcome.SUCCESS
        assert event.custom_attributes["env"] == "production"

    def test_builder_with_name_key_for_candidates(self):
        """Builder should accept 'name' as alternative to 'model_name'."""
        event = (
            RouterDecisionEventBuilder()
            .with_candidates([{"name": "gpt-4", "provider": "openai"}])
            .build()
        )

        assert event.candidate_deployments[0].model_name == "gpt-4"


class TestExtractRouterDecisionFromSpanEvent:
    """Test extraction of events from span event attributes."""

    def test_extract_valid_payload(self):
        """Should extract event from valid JSON payload."""
        event = RouterDecisionEvent(
            strategy_name="llmrouter-knn",
            selected_deployment="gpt-4",
        )

        # Simulate span event attributes
        attributes = {ROUTER_DECISION_PAYLOAD_KEY: event.to_json()}

        extracted = extract_router_decision_from_span_event(attributes)

        assert extracted is not None
        assert extracted.contract_name == CONTRACT_FULL_NAME
        assert extracted.strategy_name == "llmrouter-knn"
        assert extracted.selected_deployment == "gpt-4"

    def test_extract_with_dict_payload(self):
        """Should handle dict payload (not just string)."""
        event = RouterDecisionEvent(
            strategy_name="llmrouter-knn",
            selected_deployment="gpt-4",
        )

        attributes = {ROUTER_DECISION_PAYLOAD_KEY: event.to_dict()}

        extracted = extract_router_decision_from_span_event(attributes)

        assert extracted is not None
        assert extracted.strategy_name == "llmrouter-knn"

    def test_extract_missing_payload_returns_none(self):
        """Should return None if payload key is missing."""
        attributes = {"some.other.key": "value"}

        extracted = extract_router_decision_from_span_event(attributes)

        assert extracted is None

    def test_extract_wrong_contract_returns_none(self):
        """Should return None if contract name doesn't match."""
        attributes = {
            ROUTER_DECISION_PAYLOAD_KEY: json.dumps(
                {"contract_name": "wrong.contract.v1", "strategy_name": "test"}
            )
        }

        extracted = extract_router_decision_from_span_event(attributes)

        assert extracted is None

    def test_extract_invalid_json_returns_none(self):
        """Should return None for invalid JSON."""
        attributes = {ROUTER_DECISION_PAYLOAD_KEY: "not valid json{"}

        extracted = extract_router_decision_from_span_event(attributes)

        assert extracted is None

    def test_round_trip_serialization(self):
        """Event should round-trip through serialization correctly."""
        original = (
            RouterDecisionEventBuilder()
            .with_strategy("llmrouter-knn", version="1.0.0")
            .with_input(query_length=150, requested_model="gpt-4")
            .with_candidates(
                [
                    {"model_name": "gpt-4", "score": 0.95, "available": True},
                    {"model_name": "claude-3", "score": 0.82, "available": True},
                ]
            )
            .with_selection(selected="gpt-4", reason="highest_score")
            .with_timing(total_ms=15.5, strategy_ms=10.2, embedding_ms=3.1)
            .with_outcome(
                status=RoutingOutcome.SUCCESS,
                input_tokens=100,
                output_tokens=200,
                total_tokens=300,
            )
            .with_fallback(triggered=False)
            .build()
        )

        # Simulate emission and extraction
        attributes = {ROUTER_DECISION_PAYLOAD_KEY: original.to_json()}
        extracted = extract_router_decision_from_span_event(attributes)

        assert extracted is not None
        assert extracted.strategy_name == original.strategy_name
        assert extracted.strategy_version == original.strategy_version
        assert extracted.selected_deployment == original.selected_deployment
        assert extracted.input.query_length == original.input.query_length
        assert len(extracted.candidate_deployments) == len(
            original.candidate_deployments
        )
        assert extracted.timings.total_ms == original.timings.total_ms
        assert extracted.outcome.status == original.outcome.status
        assert extracted.outcome.total_tokens == original.outcome.total_tokens


class TestJaegerTraceEventParsing:
    """Test parsing of events from representative Jaeger trace JSON."""

    @pytest.fixture
    def sample_jaeger_span_with_routing_event(self):
        """Create a sample Jaeger span with a routing decision event."""
        event = RouterDecisionEvent(
            strategy_name="llmrouter-knn",
            strategy_version="1.0.0",
            selected_deployment="gpt-4",
            selection_reason="highest_score",
            trace_id="abc123def456789",
            span_id="span123456",
        )
        event.input = RouterDecisionInput(query_length=150)
        event.candidate_deployments = [
            CandidateDeployment(model_name="gpt-4", provider="openai", score=0.95),
            CandidateDeployment(
                model_name="claude-3", provider="anthropic", score=0.82
            ),
        ]
        event.timings = RoutingTimings(total_ms=15.5)
        event.outcome = RoutingOutcomeData(status=RoutingOutcome.SUCCESS)

        return {
            "traceID": "abc123def456789",
            "spanID": "span123456",
            "operationName": "llm.routing.decision",
            "references": [],
            "startTime": 1705315800000000,
            "duration": 15500,
            "tags": [
                {
                    "key": "llm.routing.strategy",
                    "type": "string",
                    "value": "llmrouter-knn",
                },
                {
                    "key": "llm.routing.selected_model",
                    "type": "string",
                    "value": "gpt-4",
                },
            ],
            "logs": [
                {
                    "timestamp": 1705315800000000,
                    "fields": [
                        {
                            "key": ROUTER_DECISION_PAYLOAD_KEY,
                            "type": "string",
                            "value": event.to_json(),
                        }
                    ],
                }
            ],
            "processID": "p1",
            "warnings": None,
        }

    def test_parse_routing_event_from_jaeger_logs(
        self, sample_jaeger_span_with_routing_event
    ):
        """Should parse routing decision from Jaeger span logs."""
        span = sample_jaeger_span_with_routing_event

        # Extract from logs (as the extraction script does)
        for log in span.get("logs", []):
            fields = {f.get("key"): f.get("value") for f in log.get("fields", [])}
            payload = fields.get(ROUTER_DECISION_PAYLOAD_KEY)
            if payload:
                data = json.loads(payload)
                assert data["contract_name"] == CONTRACT_FULL_NAME
                assert data["strategy_name"] == "llmrouter-knn"
                assert data["selected_deployment"] == "gpt-4"
                assert data["input"]["query_length"] == 150
                assert len(data["candidate_deployments"]) == 2
                return

        pytest.fail("No routing decision event found in span logs")

    def test_extract_function_works_with_jaeger_format(
        self, sample_jaeger_span_with_routing_event
    ):
        """extract_router_decision_from_span_event should work with Jaeger format."""
        span = sample_jaeger_span_with_routing_event

        for log in span.get("logs", []):
            fields = {f.get("key"): f.get("value") for f in log.get("fields", [])}
            extracted = extract_router_decision_from_span_event(fields)

            if extracted:
                assert extracted.strategy_name == "llmrouter-knn"
                assert extracted.selected_deployment == "gpt-4"
                assert extracted.input.query_length == 150
                return

        pytest.fail("No routing decision event extracted")


class TestPIISafety:
    """Test that contract is PII-safe by design."""

    def test_event_does_not_store_query_content(self):
        """Event should not store actual query content."""
        event = RouterDecisionEvent()
        event.input = RouterDecisionInput(query_length=150)

        data = event.to_dict()

        # Should have query_length but not query text
        assert data["input"]["query_length"] == 150
        assert "query" not in data["input"]
        assert "query_text" not in data["input"]
        assert "prompt" not in data["input"]

    def test_event_does_not_store_response_content(self):
        """Event should not store response content."""
        event = RouterDecisionEvent()
        event.outcome = RoutingOutcomeData(
            status=RoutingOutcome.SUCCESS,
            input_tokens=100,
            output_tokens=200,
        )

        data = event.to_dict()

        # Should have token counts but not content
        assert data["outcome"]["input_tokens"] == 100
        assert data["outcome"]["output_tokens"] == 200
        assert "response" not in data["outcome"]
        assert "completion" not in data["outcome"]
        assert "content" not in data["outcome"]


class TestBackwardCompatibility:
    """Test backward compatibility requirements."""

    def test_v1_schema_fields_present(self):
        """All v1 schema fields should be present in serialized output."""
        event = RouterDecisionEvent()
        data = event.to_dict()

        # Required v1 fields
        required_fields = [
            "contract_version",
            "contract_name",
            "event_id",
            "timestamp_utc",
            "timestamp_unix_ms",
            "input",
            "strategy_name",
            "candidate_deployments",
            "selected_deployment",
            "timings",
            "outcome",
            "fallback",
            "custom_attributes",
        ]

        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

    def test_input_nested_fields(self):
        """Input object should have all documented fields."""
        event = RouterDecisionEvent()
        data = event.to_dict()

        input_fields = [
            "requested_model",
            "query_length",
            "user_id",
            "team_id",
            "request_metadata",
        ]
        for field in input_fields:
            assert field in data["input"], f"Missing input field: {field}"

    def test_timings_nested_fields(self):
        """Timings object should have all documented fields."""
        event = RouterDecisionEvent()
        data = event.to_dict()

        timing_fields = [
            "total_ms",
            "strategy_ms",
            "embedding_ms",
            "candidate_filter_ms",
        ]
        for field in timing_fields:
            assert field in data["timings"], f"Missing timing field: {field}"

    def test_outcome_nested_fields(self):
        """Outcome object should have all documented fields."""
        event = RouterDecisionEvent()
        data = event.to_dict()

        outcome_fields = [
            "status",
            "error_message",
            "error_type",
            "input_tokens",
            "output_tokens",
            "total_tokens",
        ]
        for field in outcome_fields:
            assert field in data["outcome"], f"Missing outcome field: {field}"

    def test_fallback_nested_fields(self):
        """Fallback object should have all documented fields."""
        event = RouterDecisionEvent()
        data = event.to_dict()

        fallback_fields = [
            "fallback_triggered",
            "original_model",
            "fallback_reason",
            "fallback_attempt",
        ]
        for field in fallback_fields:
            assert field in data["fallback"], f"Missing fallback field: {field}"
