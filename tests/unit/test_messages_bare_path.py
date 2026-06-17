"""
Tests for the Anthropic Messages bare-path surface (RouteIQ-0e37).

Upstream LiteLLM registers the Anthropic Messages family ONLY at the
``/v1``-prefixed path, so mounting LiteLLM under ``/v1`` would expose it at
``/v1/v1/messages`` — a 404 for Anthropic SDK callers POSTing ``/v1/messages``.

``routes/messages.py`` registers a thin router on the RouteIQ-owned parent app
at the bare external paths ``/v1/messages`` and ``/v1/messages/count_tokens``,
delegating to the upstream LiteLLM handler.  These tests assert:

1. The router resolves at the bare external paths (route-table inspection).
2. The router does NOT introduce the doubled ``/v1/v1/messages`` path.
3. Through a TestClient, a POST to ``/v1/messages`` reaches the delegating
   handler even when a ``/v1`` mount is present (ordering wins over the mount),
   and Anthropic beta headers (extended-thinking + prompt-caching) pass through.
4. ``count_tokens`` resolves and delegates the same way.
5. ``register_messages_routes`` degrades gracefully (returns False) when the
   upstream handler import fails.

All tests are offline / credential-free: the upstream handlers are monkeypatched
so no real LiteLLM routing, network, or AWS credentials are exercised.
"""

import pytest
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient

from litellm_llmrouter.routes.messages import (
    build_messages_router,
    register_messages_routes,
)


def test_router_registers_bare_v1_paths():
    """The router exposes the bare external /v1/messages family, not doubled."""
    router = build_messages_router()
    paths = {route.path for route in router.routes}

    assert "/v1/messages" in paths
    assert "/v1/messages/count_tokens" in paths
    # The whole point: we must NOT reproduce the /v1/v1/... double-prefix.
    assert "/v1/v1/messages" not in paths
    assert "/v1/v1/messages/count_tokens" not in paths


def test_router_methods_are_post():
    """Both routes are POST-only, matching the Anthropic Messages spec."""
    router = build_messages_router()
    methods_by_path = {route.path: set(route.methods) for route in router.routes}

    assert methods_by_path["/v1/messages"] == {"POST"}
    assert methods_by_path["/v1/messages/count_tokens"] == {"POST"}


def _build_app_with_messages_and_v1_mount(monkeypatch):
    """Assemble a parent app that mirrors gateway/app.py composition order.

    Registers the bare-path messages router FIRST, then mounts a stub ``/v1``
    sub-app whose only ``/messages`` behaviour is a 404 — exactly the upstream
    situation this fix corrects.  The upstream handlers are monkeypatched.

    Returns (app, calls) where ``calls`` records delegation invocations.
    """
    calls: dict[str, dict] = {}

    async def _fake_anthropic_response(*, fastapi_response, request, user_api_key_dict):
        body = await request.json()
        calls["messages"] = {
            "model": body.get("model"),
            "anthropic_beta": request.headers.get("anthropic-beta"),
            "stream": body.get("stream"),
        }
        return JSONResponse(
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "model": body.get("model"),
            }
        )

    async def _fake_count_tokens(*, request, user_api_key_dict):
        body = await request.json()
        calls["count_tokens"] = {"model": body.get("model")}
        return {"input_tokens": 42}

    async def _fake_user_api_key_auth():
        # Stand-in for LiteLLM's auth dependency; returns a sentinel object.
        # No params so FastAPI doesn't treat *args/**kwargs as query fields.
        return object()

    import litellm_llmrouter.routes.messages as messages_mod

    # Patch the build to use our fakes (so no real LiteLLM import/handlers run).
    def _fake_build_router():
        from fastapi import APIRouter, Depends

        router = APIRouter(tags=["[beta] Anthropic `/v1/messages`"])

        @router.post("/v1/messages")
        async def messages(
            fastapi_response: Response,
            request: Request,
            user_api_key_dict=Depends(_fake_user_api_key_auth),
        ):
            return await _fake_anthropic_response(
                fastapi_response=fastapi_response,
                request=request,
                user_api_key_dict=user_api_key_dict,
            )

        @router.post("/v1/messages/count_tokens")
        async def count_tokens(
            request: Request,
            user_api_key_dict=Depends(_fake_user_api_key_auth),
        ):
            return await _fake_count_tokens(
                request=request,
                user_api_key_dict=user_api_key_dict,
            )

        return router

    monkeypatch.setattr(messages_mod, "build_messages_router", _fake_build_router)

    app = FastAPI()
    # Composition order mirrors create_gateway_app(): bare routes BEFORE the mount.
    registered = register_messages_routes(app)
    assert registered is True

    # Stub LiteLLM sub-app mounted at /v1.  It only knows /chat/completions;
    # /messages 404s here — exactly the broken upstream situation.
    litellm_stub = FastAPI()

    @litellm_stub.post("/chat/completions")
    async def _chat():
        return {"object": "chat.completion"}

    app.mount("/v1", litellm_stub)
    return app, calls


def test_post_v1_messages_resolves_to_delegating_handler(monkeypatch):
    """A POST to the bare /v1/messages reaches our handler, not the /v1 mount 404."""
    app, calls = _build_app_with_messages_and_v1_mount(monkeypatch)
    client = TestClient(app)

    resp = client.post(
        "/v1/messages",
        json={
            "model": "claude-sonnet-4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        },
        headers={
            # Extended-thinking + prompt-caching beta headers must pass through.
            "anthropic-beta": "interleaved-thinking-2025-05-14,prompt-caching-2024-07-31",
            "Authorization": "Bearer test-api-key",
        },
    )

    # NOT a 404 — the bare route matched before the /v1 mount catch-all.
    assert resp.status_code == 200, resp.text
    assert resp.json()["type"] == "message"
    # Delegation happened and beta headers reached the upstream handler intact.
    assert calls["messages"]["model"] == "claude-sonnet-4"
    assert "interleaved-thinking-2025-05-14" in calls["messages"]["anthropic_beta"]
    assert "prompt-caching-2024-07-31" in calls["messages"]["anthropic_beta"]


def test_post_v1_messages_count_tokens_resolves(monkeypatch):
    """count_tokens resolves at the bare path and delegates upstream."""
    app, calls = _build_app_with_messages_and_v1_mount(monkeypatch)
    client = TestClient(app)

    resp = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "claude-sonnet-4",
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers={"Authorization": "Bearer test-api-key"},
    )

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"input_tokens": 42}
    assert calls["count_tokens"]["model"] == "claude-sonnet-4"


def test_doubled_prefix_path_is_not_directly_served(monkeypatch):
    """Sanity: the parent app does not itself answer /v1/v1/messages.

    (Through the real /v1 LiteLLM mount that path WOULD be the broken one;
    here we just confirm our bare router didn't accidentally create it.)
    """
    app, _ = _build_app_with_messages_and_v1_mount(monkeypatch)
    client = TestClient(app)

    resp = client.post(
        "/v1/v1/messages",
        json={"model": "claude-sonnet-4", "messages": []},
        headers={"Authorization": "Bearer test-api-key"},
    )
    # The stub /v1 mount has no /v1/messages route -> 404 / 405, never our 200.
    assert resp.status_code != 200


def test_does_not_clobber_chat_completions_via_v1_mount(monkeypatch):
    """The /v1 mount's own routes (e.g. /v1/chat/completions) still resolve."""
    app, _ = _build_app_with_messages_and_v1_mount(monkeypatch)
    client = TestClient(app)

    resp = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4", "messages": []},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["object"] == "chat.completion"


def test_register_messages_routes_degrades_when_handler_unavailable(monkeypatch):
    """register_messages_routes returns False (no-op) if the build raises."""
    import litellm_llmrouter.routes.messages as messages_mod

    def _boom():
        raise ImportError("litellm anthropic endpoints unavailable")

    monkeypatch.setattr(messages_mod, "build_messages_router", _boom)

    app = FastAPI()
    n_routes_before = len(app.routes)
    result = register_messages_routes(app)

    assert result is False
    # No routes added when the upstream handler is missing.
    assert len(app.routes) == n_routes_before


def test_build_messages_router_imports_real_upstream_handler():
    """When LiteLLM is installed, the real router wires the upstream handlers."""
    pytest.importorskip("litellm.proxy.anthropic_endpoints.endpoints")

    router = build_messages_router()
    paths = {route.path for route in router.routes}
    assert paths == {"/v1/messages", "/v1/messages/count_tokens"}


# ==========================================================================
# RouteIQ-db1d — bare-path REAL wiring + streaming + auth-reject coverage
# ==========================================================================


def test_real_router_wires_upstream_handlers_and_auth():
    """The REAL router endpoints delegate to the upstream LiteLLM Anthropic
    handlers and depend on the upstream ``user_api_key_auth`` (so auth + beta-
    header passthrough behave identically to hitting LiteLLM directly)."""
    pytest.importorskip("litellm.proxy.anthropic_endpoints.endpoints")
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    router = build_messages_router()
    by_path = {route.path: route for route in router.routes}

    # every bare route depends on the upstream auth dependency.
    for path in ("/v1/messages", "/v1/messages/count_tokens"):
        deps = by_path[path].dependant.dependencies
        dep_calls = {d.call for d in deps}
        assert user_api_key_auth in dep_calls, f"{path} missing upstream auth dep"


def test_streaming_response_passes_through(monkeypatch):
    """An SSE streaming Anthropic response flows through the bare-path handler
    unchanged (the handler returns whatever the upstream handler returns, so a
    StreamingResponse streams)."""
    calls: dict[str, dict] = {}

    async def _sse_body():
        # minimal Anthropic SSE event stream.
        yield b'event: message_start\ndata: {"type":"message_start"}\n\n'
        yield b'event: content_block_delta\ndata: {"type":"content_block_delta"}\n\n'
        yield b"event: message_stop\ndata: {}\n\n"

    async def _fake_anthropic_response(*, fastapi_response, request, user_api_key_dict):
        body = await request.json()
        calls["messages"] = {"stream": body.get("stream")}
        return StreamingResponse(_sse_body(), media_type="text/event-stream")

    async def _fake_count_tokens(*, request, user_api_key_dict):
        return {"input_tokens": 1}

    async def _fake_user_api_key_auth():
        return object()

    import litellm_llmrouter.routes.messages as messages_mod

    def _fake_build_router():
        from fastapi import APIRouter, Depends

        router = APIRouter()

        @router.post("/v1/messages")
        async def messages(
            fastapi_response: Response,
            request: Request,
            user_api_key_dict=Depends(_fake_user_api_key_auth),
        ):
            return await _fake_anthropic_response(
                fastapi_response=fastapi_response,
                request=request,
                user_api_key_dict=user_api_key_dict,
            )

        return router

    monkeypatch.setattr(messages_mod, "build_messages_router", _fake_build_router)

    app = FastAPI()
    register_messages_routes(app)
    client = TestClient(app)

    with client.stream(
        "POST",
        "/v1/messages",
        json={
            "model": "claude-sonnet-4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
        headers={"Authorization": "Bearer test-api-key"},
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = b"".join(resp.iter_bytes())

    assert calls["messages"]["stream"] is True
    assert b"message_start" in body
    assert b"message_stop" in body


def test_auth_rejection_blocks_delegation(monkeypatch):
    """When the auth dependency REJECTS (raises 401), the request never reaches
    the upstream delegating handler — the bare path is fail-closed on auth."""
    delegated = {"called": False}

    async def _fake_anthropic_response(
        *, fastapi_response, request, user_api_key_dict
    ):  # pragma: no cover - must NOT be reached on auth failure
        delegated["called"] = True
        return JSONResponse({"id": "msg"})

    async def _rejecting_auth():
        raise HTTPException(status_code=401, detail="invalid api key")

    import litellm_llmrouter.routes.messages as messages_mod

    def _fake_build_router():
        from fastapi import APIRouter, Depends

        router = APIRouter()

        @router.post("/v1/messages")
        async def messages(
            fastapi_response: Response,
            request: Request,
            user_api_key_dict=Depends(_rejecting_auth),
        ):
            return await _fake_anthropic_response(
                fastapi_response=fastapi_response,
                request=request,
                user_api_key_dict=user_api_key_dict,
            )

        return router

    monkeypatch.setattr(messages_mod, "build_messages_router", _fake_build_router)

    app = FastAPI()
    register_messages_routes(app)
    client = TestClient(app)

    resp = client.post(
        "/v1/messages",
        json={"model": "claude-sonnet-4", "messages": []},
        headers={"Authorization": "Bearer bad-key"},
    )
    assert resp.status_code == 401
    # the upstream handler was never invoked (auth gated it).
    assert delegated["called"] is False


# ==========================================================================
# RouteIQ-b9fe — contract guard: detect upstream adding a bare /messages route
# ==========================================================================


def test_upstream_does_not_register_bare_messages_route():
    """Contract guard: RouteIQ registers the BARE ``/v1/messages`` family BECAUSE
    upstream LiteLLM registers the Anthropic Messages family ONLY at the
    ``/v1``-prefixed path inside its own app (so the /v1 mount doubles it to
    /v1/v1/messages).

    If a future LiteLLM adds the family at the BARE path too, mounting it under
    /v1 would no longer hide it and RouteIQ's parent-app registration would
    DOUBLE-register the route. This test fails loudly if upstream starts
    exposing a bare ``/messages`` / ``/v1/messages`` on its OWN app router, so the
    double-register is caught at CI time, not in production."""
    endpoints = pytest.importorskip("litellm.proxy.anthropic_endpoints.endpoints")

    router = getattr(endpoints, "router", None)
    assert router is not None, "upstream anthropic endpoints router not found"

    paths = {getattr(route, "path", None) for route in router.routes}
    # Upstream registers the Anthropic family at the /v1-prefixed path ONLY.
    # The contract RouteIQ relies on: upstream must NOT also expose a BARE
    # ``/messages`` on its own router (which, once mounted under /v1, would mean
    # RouteIQ's bare /v1/messages collides / double-registers).
    assert "/messages" not in paths, (
        "Upstream LiteLLM now registers a BARE /messages route on its own router. "
        "Mounting it under /v1 no longer doubles it, so RouteIQ's bare-path "
        "registration in routes/messages.py would DOUBLE-register. Re-evaluate "
        "register_messages_routes() (RouteIQ-b9fe contract guard)."
    )
    # The /v1-prefixed form upstream DOES register (the reason the doubling
    # happens) should still be present — if it vanished the fix's premise changed.
    assert "/v1/messages" in paths, (
        "Upstream LiteLLM no longer registers /v1/messages on its own router; "
        "the premise of routes/messages.py (doubled /v1/v1/messages) has changed "
        "(RouteIQ-b9fe contract guard)."
    )
