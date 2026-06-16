"""Tests for the CLI (cli.py) — RouteIQ-efa4.

Covers --dry-run short-circuiting BEFORE any network (incl. the 10k headline
allocation), arg parsing, and the report-writing pipeline driven entirely
through mocked transports (no live endpoint, no credentials).
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx

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
