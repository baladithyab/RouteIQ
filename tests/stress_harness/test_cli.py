"""Tests for the CLI (cli.py) — RouteIQ-efa4.

Covers --dry-run short-circuiting BEFORE any network (incl. the 10k headline
allocation), arg parsing, and the report-writing pipeline driven entirely
through mocked transports (no live endpoint, no credentials).
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from stress_harness.cli import build_parser, main

from .conftest import make_chat_response, make_stats_transport


def test_dry_run_10k_no_network(capsys):
    rc = main(
        ["--base-url", "http://routeiq.local", "--num-requests", "10000", "--dry-run"]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "DRY RUN (no requests fired)" in out
    assert "TOTAL requests    : 10000" in out
    # each bucket gets 2000.
    assert "2000" in out


def test_dry_run_with_conversations_reports_total_turns(capsys):
    rc = main(
        [
            "--base-url",
            "http://routeiq.local",
            "--num-requests",
            "10",
            "--num-conversations",
            "4",
            "--turn-lengths",
            "2,3",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "conversation plan" in out
    assert "conv turns" in out


def test_category_weights_parsing():
    args = build_parser().parse_args(
        ["--base-url", "x", "--category-weights", "math=2,code=1"]
    )
    assert args.category_weights == {"math": 2.0, "code": 1.0}


def test_turn_lengths_parsing_clamps_to_two():
    args = build_parser().parse_args(["--base-url", "x", "--turn-lengths", "1,5,8"])
    assert args.turn_lengths == (2, 5, 8)


def test_full_pipeline_through_mock_transports(monkeypatch, tmp_path, capsys):
    # Patch httpx.AsyncClient so BOTH the data-plane client and the stats client
    # use mock transports — exercises the whole CLI pipeline cred-free.
    def chat_handler(req: httpx.Request) -> httpx.Response:
        return make_chat_response(
            request_id=f"id-{req.read().decode()[:8]}", model="model-a"
        )

    stats_handler = make_stats_transport(
        active_strategy="llmrouter-knn",
        model_distribution={"model-a": 5},
        strategy_distribution={"llmrouter-knn": 5},
        total_decisions=5,
    )

    def route(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/v1/chat/completions"):
            return chat_handler(request)
        return stats_handler(request)

    real_init = httpx.AsyncClient.__init__

    def patched_init(self, *a, **kw):
        kw.setdefault("transport", httpx.MockTransport(route))
        # drop a real base_url/transport conflict by forcing our transport.
        kw["transport"] = httpx.MockTransport(route)
        real_init(self, *a, **kw)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)

    out_dir = tmp_path / "out"
    rc = main(
        [
            "--base-url",
            "http://routeiq.local",
            "--num-requests",
            "5",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    report = Path(out_dir) / "report.json"
    assert report.exists()
    data = json.loads(report.read_text())
    assert data["active_strategy"] == "llmrouter-knn"
    assert data["verdict_family"] == "consistency"
    assert data["successful_requests"] == 5
    md = (Path(out_dir) / "report.md").read_text()
    assert "Active routing strategy" in md
    assert "llmrouter-knn" in md


def test_pipeline_no_stats_flag(monkeypatch, tmp_path):
    def chat_handler(req: httpx.Request) -> httpx.Response:
        return make_chat_response(model="model-a")

    real_init = httpx.AsyncClient.__init__

    def patched_init(self, *a, **kw):
        kw["transport"] = httpx.MockTransport(chat_handler)
        real_init(self, *a, **kw)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)

    out_dir = tmp_path / "out2"
    rc = main(
        [
            "--base-url",
            "http://routeiq.local",
            "--num-requests",
            "3",
            "--no-stats",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    data = json.loads((Path(out_dir) / "report.json").read_text())
    # no stats fetched -> unknown strategy -> generic verdict, no crash.
    assert data["active_strategy"] is None
    assert data["verdict_family"] == "generic"


# ==========================================================================
# RouteIQ-3b18 — --base-url optional under --dry-run, required otherwise
# ==========================================================================


def test_dry_run_without_base_url_no_network(capsys):
    """--dry-run needs no target: a plan can be previewed with NO --base-url."""
    rc = main(["--num-requests", "10000", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "DRY RUN (no requests fired)" in out
    assert "TOTAL requests    : 10000" in out
    # base-url line shows the unset placeholder, never crashes on None.
    assert "<unset>" in out


def test_base_url_no_longer_required_at_argparse_level():
    """build_parser() accepts NO --base-url (validation deferred to main())."""
    args = build_parser().parse_args(["--num-requests", "5"])
    assert args.base_url is None


def test_real_run_without_base_url_errors():
    """A non-dry-run with no --base-url is rejected by main() (parser.error ->
    SystemExit(2))."""
    with pytest.raises(SystemExit) as exc:
        main(["--num-requests", "5"])
    assert exc.value.code == 2


# ==========================================================================
# RouteIQ-b0df — CW Logs enrichment wired through the CLI pipeline
# ==========================================================================


def test_enrich_cwlogs_attaches_decision_lines(monkeypatch, tmp_path):
    """--enrich-cwlogs flows the authoritative routing_decision line per request
    into the analysis: the per-request strategy from the decision line populates
    the enriched count + strategy_distribution. Driven through a fake enricher so
    NO boto3 / AWS creds are touched."""
    from stress_harness import cwlogs
    from stress_harness.models import RoutingDecisionLine

    # Deterministic request id per chat call so the fake enricher can key on it.
    counter = {"n": 0}

    def chat_handler(req: httpx.Request) -> httpx.Response:
        counter["n"] += 1
        return make_chat_response(
            request_id=f"chatcmpl-{counter['n']:04d}", model="model-a"
        )

    real_init = httpx.AsyncClient.__init__

    def patched_init(self, *a, **kw):
        kw["transport"] = httpx.MockTransport(chat_handler)
        real_init(self, *a, **kw)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)

    # Fake the CW Logs enricher: return a decision line (with strategy) for every
    # request id handed in. Never reaches the lazy boto3 import.
    class _FakeEnricher:
        def __init__(self, **kwargs):
            pass

        def enrich(self, request_ids, start_ts, end_ts):
            return {
                rid: RoutingDecisionLine(
                    present=True,
                    request_id=rid,
                    model="model-a",
                    strategy="kumaraswamy-thompson",
                )
                for rid in request_ids
            }

    monkeypatch.setattr(cwlogs, "CloudWatchLogsEnricher", _FakeEnricher)

    out_dir = tmp_path / "cw"
    rc = main(
        [
            "--base-url",
            "http://routeiq.local",
            "--num-requests",
            "4",
            "--no-stats",
            "--enrich-cwlogs",
            "--cwlogs-delay",
            "0",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    data = json.loads((Path(out_dir) / "report.json").read_text())
    assert data["enriched_requests"] == 4
    # the per-request strategy from the decision line populated the distribution.
    assert data["strategy_distribution"].get("kumaraswamy-thompson") == 4


def test_enrich_cwlogs_degrades_when_enricher_raises(monkeypatch, tmp_path):
    """A CW Logs enrichment failure must NOT abort the run — it proceeds body-
    only (RouteIQ-b0df: degrade, never abort)."""
    from stress_harness import cwlogs

    def chat_handler(req: httpx.Request) -> httpx.Response:
        return make_chat_response(model="model-a")

    real_init = httpx.AsyncClient.__init__

    def patched_init(self, *a, **kw):
        kw["transport"] = httpx.MockTransport(chat_handler)
        real_init(self, *a, **kw)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)

    class _BoomEnricher:
        def __init__(self, **kwargs):
            pass

        def enrich(self, request_ids, start_ts, end_ts):
            raise RuntimeError("insights query failed")

    monkeypatch.setattr(cwlogs, "CloudWatchLogsEnricher", _BoomEnricher)

    out_dir = tmp_path / "cw2"
    rc = main(
        [
            "--base-url",
            "http://routeiq.local",
            "--num-requests",
            "3",
            "--no-stats",
            "--enrich-cwlogs",
            "--cwlogs-delay",
            "0",
            "--out-dir",
            str(out_dir),
        ]
    )
    # run still succeeds; just no enrichment.
    assert rc == 0
    data = json.loads((Path(out_dir) / "report.json").read_text())
    assert data["enriched_requests"] == 0
    assert data["successful_requests"] == 3
