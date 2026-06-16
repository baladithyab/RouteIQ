"""Tests for the OpenAI-compat client (client.py) via a mocked transport.

NO live endpoint, NO credentials: every request flows through an
httpx.MockTransport. Asserts the client parses id/model/usage/headers, sends the
load-bearing ``model:"auto"`` body with RouteIQ Bearer auth, and degrades on
errors. — RouteIQ-b245.
"""

from __future__ import annotations

import asyncio
import json

import httpx

from stress_harness.client import RouterClient
from stress_harness.models import RequestRecord

from .conftest import make_chat_response


def _run(coro):
    return asyncio.run(coro)


def test_parses_id_model_usage_and_headers(make_async_client):
    def handler(request: httpx.Request) -> httpx.Response:
        return make_chat_response(
            request_id="chatcmpl-abc",
            model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
            prompt_tokens=20,
            completion_tokens=80,
            headers={"x-request-id": "chatcmpl-abc"},
        )

    client = RouterClient(
        "http://routeiq.local", http_client=make_async_client(handler)
    )
    [rec] = _run(client.run([RequestRecord(my_category_tag="math", prompt="2+2?")]))

    assert rec.ok
    assert rec.request_id == "chatcmpl-abc"
    assert rec.body_model == "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"
    assert rec.prompt_tokens == 20
    assert rec.completion_tokens == 80
    assert rec.total_tokens == 100
    assert rec.header_request_id == "chatcmpl-abc"
    assert rec.http_status == 200
    assert rec.client_latency_ms is not None


def test_sends_model_auto_and_bearer(make_async_client):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        captured["auth"] = request.headers.get("authorization")
        captured["url"] = str(request.url)
        return make_chat_response()

    client = RouterClient(
        "http://routeiq.local/",
        token="test-api-key",
        http_client=make_async_client(handler),
    )
    _run(client.run([RequestRecord(my_category_tag="code", prompt="hi")]))

    # model:"auto" is load-bearing — it triggers RouteIQ's ML router.
    assert captured["body"]["model"] == "auto"
    assert captured["body"]["stream"] is False
    assert captured["body"]["messages"][0]["content"] == "hi"
    assert captured["auth"] == "Bearer test-api-key"
    assert captured["url"] == "http://routeiq.local/v1/chat/completions"


def test_per_record_user_id_header(make_async_client):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["user"] = request.headers.get("x-user-id")
        return make_chat_response()

    client = RouterClient(
        "http://routeiq.local", http_client=make_async_client(handler)
    )
    rec = RequestRecord(my_category_tag="math", prompt="x", user_id="user-007")
    _run(client.run([rec]))
    assert captured["user"] == "user-007"


def test_pinned_model_is_passed_through(make_async_client):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return make_chat_response()

    client = RouterClient(
        "http://routeiq.local",
        model="anthropic.claude-opus-4-8",
        http_client=make_async_client(handler),
    )
    _run(client.run([RequestRecord(my_category_tag="math", prompt="x")]))
    assert captured["body"]["model"] == "anthropic.claude-opus-4-8"


def test_http_error_status_recorded_not_raised(make_async_client):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    client = RouterClient(
        "http://routeiq.local", http_client=make_async_client(handler)
    )
    [rec] = _run(client.run([RequestRecord(my_category_tag="math", prompt="x")]))
    assert rec.http_status == 500
    assert not rec.ok


def test_network_error_recorded_as_error(make_async_client):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client = RouterClient(
        "http://routeiq.local", http_client=make_async_client(handler)
    )
    [rec] = _run(client.run([RequestRecord(my_category_tag="math", prompt="x")]))
    assert rec.error is not None
    assert "ConnectError" in rec.error
    assert not rec.ok


def test_non_httperror_escape_is_recorded_not_raised(make_async_client):
    def handler(request: httpx.Request) -> httpx.Response:
        raise ValueError("unexpected non-http error")

    client = RouterClient(
        "http://routeiq.local", http_client=make_async_client(handler)
    )
    out = _run(client.run([RequestRecord(my_category_tag="math", prompt="x")]))
    assert len(out) == 1
    assert out[0].error is not None
    assert "ValueError" in out[0].error
    assert not out[0].ok


def test_run_preserves_input_order(make_async_client):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return make_chat_response(request_id=body["messages"][-1]["content"])

    client = RouterClient(
        "http://routeiq.local", http_client=make_async_client(handler)
    )
    records = [RequestRecord(my_category_tag="math", prompt=f"p{i}") for i in range(10)]
    out = _run(client.run(records, concurrency=4))
    assert [r.request_id for r in out] == [f"p{i}" for i in range(10)]


def test_request_id_falls_back_to_header_echo(make_async_client):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"object": "chat.completion", "model": "m", "usage": {}},
            headers={"x-request-id": "from-header"},
        )

    client = RouterClient(
        "http://routeiq.local", http_client=make_async_client(handler)
    )
    [rec] = _run(client.run([RequestRecord(my_category_tag="math", prompt="x")]))
    assert rec.request_id == "from-header"


def test_empty_records_returns_empty(make_async_client):
    client = RouterClient(
        "http://routeiq.local", http_client=make_async_client(make_chat_response)
    )
    assert _run(client.run([])) == []


def test_multi_turn_threads_history(make_async_client):
    seen_lengths: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen_lengths.append(len(body["messages"]))
        return make_chat_response(request_id=f"turn-{len(seen_lengths)}")

    client = RouterClient(
        "http://routeiq.local", http_client=make_async_client(handler)
    )
    conv = [
        RequestRecord(
            my_category_tag="math",
            prompt="q0",
            conversation_id="c",
            turn_index=0,
            num_turns=3,
        ),
        RequestRecord(
            my_category_tag="math",
            prompt="q1",
            conversation_id="c",
            turn_index=1,
            num_turns=3,
        ),
        RequestRecord(
            my_category_tag="math",
            prompt="q2",
            conversation_id="c",
            turn_index=2,
            num_turns=3,
        ),
    ]
    [done] = _run(client.run_conversations([conv]))
    # turn 0 -> 1 message; turn 1 -> 3 (user, assistant, user); turn 2 -> 5.
    assert seen_lengths == [1, 3, 5]
    assert [r.request_id for r in done] == ["turn-1", "turn-2", "turn-3"]
