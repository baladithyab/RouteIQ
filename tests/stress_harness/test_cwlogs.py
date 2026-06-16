"""Tests for the routing_decision log enrichment (cwlogs.py) — RouteIQ-b0df.

The parser is exercised with fixture log lines (no AWS, no boto3). The enricher
is driven through a fake boto3-like client injected via monkeypatch so the lazy
boto3 import is never reached. cred-free.
"""

from __future__ import annotations

import json

from stress_harness import cwlogs
from stress_harness.cwlogs import (
    CloudWatchLogsEnricher,
    parse_decision_line,
    parse_lines,
)


def test_parse_flat_line_with_strategy():
    line = json.dumps(
        {
            "message": "routing_decision",
            "request_id": "chatcmpl-1",
            "model": "model-a",
            "strategy": "llmrouter-knn",
            "latency_ms": 42.0,
        }
    )
    decision = parse_decision_line(line)
    assert decision is not None
    assert decision.present is True
    assert decision.request_id == "chatcmpl-1"
    assert decision.model == "model-a"
    assert decision.strategy == "llmrouter-knn"
    assert decision.latency_ms == 42.0


def test_parse_nested_routing_decision_block():
    line = {
        "routing_decision": {
            "request_id": "chatcmpl-2",
            "model": "model-b",
            "strategy": "cost-aware",
        }
    }
    decision = parse_decision_line(line)
    assert decision.request_id == "chatcmpl-2"
    assert decision.strategy == "cost-aware"


def test_parse_dotted_keys():
    line = {
        "request_id": "chatcmpl-3",
        "routing_decision.model": "model-c",
        "routing_decision.strategy": "llmrouter-mlp",
    }
    decision = parse_decision_line(line)
    assert decision.model == "model-c"
    assert decision.strategy == "llmrouter-mlp"


def test_strategy_absent_does_not_raise():
    # older build: model but no strategy. Must parse with strategy=None.
    decision = parse_decision_line({"request_id": "x", "model": "model-a"})
    assert decision is not None
    assert decision.strategy is None
    assert decision.model == "model-a"


def test_no_request_id_returns_none():
    assert parse_decision_line({"model": "m", "strategy": "s"}) is None


def test_non_json_returns_none():
    assert parse_decision_line("}{ not json") is None


def test_id_used_as_join_key_fallback():
    decision = parse_decision_line({"id": "chatcmpl-z", "model": "m"})
    assert decision.request_id == "chatcmpl-z"


def test_parse_lines_keyed_by_request_id():
    lines = [
        {"request_id": "a", "model": "m1", "strategy": "knn"},
        {"request_id": "b", "model": "m2", "strategy": "knn"},
    ]
    out = parse_lines(lines)
    assert set(out) == {"a", "b"}
    assert out["a"].model == "m1"


def test_build_query_includes_request_ids():
    enricher = CloudWatchLogsEnricher(env="prod")
    q = enricher.build_query(["a", "b"])
    assert '"a"' in q and '"b"' in q
    assert "strategy" in q
    assert "routeiq" in enricher.log_group


def test_enricher_runs_against_fake_client(monkeypatch):
    # inject a fake boto3-style client so the lazy boto3 import is never reached.
    class _FakeClient:
        def start_query(self, **kwargs):
            return {"queryId": "q-1"}

        def get_query_results(self, **kwargs):
            return {
                "status": "Complete",
                "results": [
                    [
                        {"field": "request_id", "value": "chatcmpl-1"},
                        {"field": "model", "value": "model-a"},
                        {"field": "strategy", "value": "llmrouter-knn"},
                        {"field": "@ptr", "value": "skip-me"},
                    ]
                ],
            }

    enricher = CloudWatchLogsEnricher(env="prod", poll_delay_s=0)
    monkeypatch.setattr(enricher, "_client", lambda: _FakeClient())
    decisions = enricher.enrich(["chatcmpl-1"], 1000.0, 2000.0, sleep=lambda _s: None)
    assert "chatcmpl-1" in decisions
    assert decisions["chatcmpl-1"].strategy == "llmrouter-knn"


def test_enricher_empty_request_ids_returns_empty():
    enricher = CloudWatchLogsEnricher(env="prod")
    assert enricher.enrich([], 0.0, 1.0) == {}


def test_row_to_dict_skips_metadata_and_decodes_json():
    row = [
        {"field": "@timestamp", "value": "2026-06-16"},
        {"field": "request_id", "value": "a"},
        {"field": "cost_usd", "value": "0.0021"},
    ]
    out = cwlogs._row_to_dict(row)
    assert "@timestamp" not in out
    assert out["request_id"] == "a"
    assert out["cost_usd"] == "0.0021"
