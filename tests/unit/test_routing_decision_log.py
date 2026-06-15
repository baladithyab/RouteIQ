"""Unit tests for the P2 structured ``routing_decision`` JSON log line (ADR-0027).

The gateway emits ONE flat, machine-parseable JSON line per routing decision on a
dedicated logger. The AWS observability stack consumes it two ways:

  - the CloudWatch Logs metric filters (aggregate + the PER-MODEL dimensioned
    latency filter keyed on the OTel ``gen_ai.response.model`` field); and
  - the Firehose subscription that promotes it to the Glue/Athena data lake
    (whose identity-SerDe columns are named ``selected_model`` + ``model``).

These tests assert the load-bearing contract:

  * the line is valid JSON with the event marker ``$.event == "routing_decision"``;
  * the model value appears under all THREE top-level keys (``selected_model``,
    ``model``, and ``gen_ai.response.model``) with the SAME value (the dual-key
    reconciliation the CW dimension + the lake column both depend on);
  * exactly the 14 flat lake columns are present (matches
    ``data_lake_construct._COLUMNS``);
  * it is PII-safe (no prompt/completion TEXT; ``query_length`` is an int only);
  * the feature is settings-gated (ADR-0013, not ``os.environ``) and fail-safe.
"""

from __future__ import annotations

import json
import logging

import pytest

from litellm_llmrouter import observability
from litellm_llmrouter.observability import (
    ROUTING_DECISION_COLUMNS,
    ROUTING_DECISION_EVENT,
    build_routing_decision_record,
    emit_routing_decision_log,
)
from litellm_llmrouter.settings import get_settings, reset_settings

_GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"


@pytest.fixture(autouse=True)
def _reset_settings_singleton():
    """Reset the cached settings singleton before and after each test.

    The structured emitter reads ``settings.otel.routing_decision_log_enabled``
    through ``get_settings`` (ADR-0013), which is cached; a stale cache would leak
    a per-test env override across tests.
    """
    reset_settings()
    yield
    reset_settings()


# --------------------------------------------------------------- record shape


def test_record_has_exactly_the_14_lake_columns() -> None:
    """The record carries exactly the 14 flat lake columns (identity SerDe).

    The Glue table extracts by identity name, so the column set IS the contract.
    Assert every ``_COLUMNS`` key is present (the OTel dimension key + optional
    sibling keys are extra and harmlessly dropped by the lake).
    """
    record = build_routing_decision_record(selected_model="bedrock/claude-haiku")
    for column in ROUTING_DECISION_COLUMNS:
        assert column in record, f"routing_decision line missing lake column {column!r}"
    assert len(ROUTING_DECISION_COLUMNS) == 14


def test_dual_key_model_value_is_consistent() -> None:
    """The model value is identical under selected_model / model / gen_ai.response.model.

    The single most load-bearing P2 assertion: the per-model CloudWatch
    MetricFilter dimensions on ``gen_ai.response.model`` while the lake columns are
    ``selected_model`` + ``model`` - all three MUST hold the same value off one
    line, or the CW dimension and the lake disagree on the model.
    """
    model = "bedrock/anthropic.claude-3-haiku-20240307-v1:0"
    record = build_routing_decision_record(selected_model=model)
    assert record["selected_model"] == model
    assert record["model"] == model
    assert record[_GEN_AI_RESPONSE_MODEL] == model


def test_event_marker_is_routing_decision() -> None:
    """The top-level event marker matches the CW + Firehose selector value."""
    record = build_routing_decision_record(selected_model="m")
    assert record["event"] == ROUTING_DECISION_EVENT == "routing_decision"


def test_numeric_and_boolean_columns_are_typed() -> None:
    """Latency/token columns are ints; reasoning/cache columns are bools.

    The Glue table types latency_ms/prompt_tokens/completion_tokens as ``int`` and
    reasoning_enabled/cache_hit as ``boolean``; the emitter must coerce so
    OpenXJsonSerDe->ParquetSerDe does not see a float/str where an int is declared.
    """
    record = build_routing_decision_record(
        selected_model="m",
        latency_ms=123.9,
        prompt_tokens=10,
        completion_tokens=20,
        reasoning_enabled=True,
        cache_hit=True,
    )
    assert isinstance(record["latency_ms"], int) and record["latency_ms"] == 123
    assert isinstance(record["prompt_tokens"], int)
    assert isinstance(record["completion_tokens"], int)
    assert record["reasoning_enabled"] is True
    assert record["cache_hit"] is True


def test_record_is_pii_safe() -> None:
    """No prompt/completion TEXT in the line - query_length (int) is the only
    query-derived field (telemetry_contracts PII posture)."""
    record = build_routing_decision_record(
        selected_model="m",
        query_length=4096,
    )
    assert record["query_length"] == 4096
    # No free-text query/prompt/completion keys leak into the line.
    for forbidden in ("query", "prompt", "completion", "messages", "content"):
        assert forbidden not in record, f"PII leak: {forbidden!r} present in line"


def test_strategy_and_query_length_omitted_when_not_supplied() -> None:
    """Optional sibling keys are absent unless supplied (byte-stable minimal line)."""
    record = build_routing_decision_record(selected_model="m")
    assert "strategy" not in record
    assert "query_length" not in record


# ------------------------------------------------------------- emission + JSON


def test_emit_writes_one_valid_json_line(caplog) -> None:
    """emit_routing_decision_log writes ONE compact JSON object on the dedicated
    logger, parseable back to the dual-key record."""
    caplog.set_level(logging.INFO, logger="routeiq.routing_decision")
    returned = emit_routing_decision_log(
        selected_model="bedrock/claude-haiku",
        strategy="knn",
        latency_ms=42,
        reason_code="model_selected",
    )
    assert returned is not None
    records = [r for r in caplog.records if r.name == "routeiq.routing_decision"]
    assert len(records) == 1, "expected exactly one routing_decision line"
    parsed = json.loads(records[0].getMessage())
    assert parsed["event"] == "routing_decision"
    assert parsed["selected_model"] == "bedrock/claude-haiku"
    assert parsed[_GEN_AI_RESPONSE_MODEL] == "bedrock/claude-haiku"
    assert parsed["strategy"] == "knn"


def test_emit_disabled_via_settings_emits_nothing(monkeypatch, caplog) -> None:
    """Setting ROUTEIQ_OTEL__ROUTING_DECISION_LOG_ENABLED=false suppresses the line.

    Routed through settings (ADR-0013), not os.environ in the emitter.
    """
    monkeypatch.setenv("ROUTEIQ_OTEL__ROUTING_DECISION_LOG_ENABLED", "false")
    reset_settings()
    # Sanity: the settings layer reflects the override.
    assert get_settings().otel.routing_decision_log_enabled is False

    caplog.set_level(logging.INFO, logger="routeiq.routing_decision")
    returned = emit_routing_decision_log(selected_model="m")
    assert returned is None
    records = [r for r in caplog.records if r.name == "routeiq.routing_decision"]
    assert not records, "disabled feature must emit no routing_decision line"


def test_emit_never_raises_on_bad_settings(monkeypatch, caplog) -> None:
    """A settings failure fails OPEN (still emits) and never raises."""

    def _boom(*_a, **_k):
        raise RuntimeError("settings unavailable")

    monkeypatch.setattr("litellm_llmrouter.settings.get_settings", _boom)
    caplog.set_level(logging.INFO, logger="routeiq.routing_decision")
    # Must not raise; fail-open default emits the line.
    returned = emit_routing_decision_log(selected_model="m")
    assert returned is not None


def test_manager_log_routing_decision_also_emits_structured_line(caplog) -> None:
    """ObservabilityManager.log_routing_decision delegates to the structured line.

    The existing human-readable call site now ALSO emits the flat JSON line the
    AWS stack consumes (no separate wiring needed at every call site).
    """
    caplog.set_level(logging.INFO, logger="routeiq.routing_decision")
    manager = observability.ObservabilityManager()
    manager.log_routing_decision(
        strategy="llmrouter-knn",
        selected_model="gpt-4",
        latency_ms=123.45,
    )
    records = [r for r in caplog.records if r.name == "routeiq.routing_decision"]
    assert len(records) == 1
    parsed = json.loads(records[0].getMessage())
    assert parsed["selected_model"] == "gpt-4"
    assert parsed[_GEN_AI_RESPONSE_MODEL] == "gpt-4"
    assert parsed["strategy"] == "llmrouter-knn"
