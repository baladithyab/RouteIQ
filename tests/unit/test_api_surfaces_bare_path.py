"""Tests for the first-class API-surface bare-path routes (RouteIQ-e48a).

The OpenAI/Cohere-style families (Responses, rerank, RAG, batches, vector-store,
audio, images) register their bare paths upstream, but RouteIQ mounts the whole
LiteLLM proxy under ``/v1`` -- so the documented bare path (``/rerank``,
``/responses``, ...) 404s for SDK callers and the upstream ``/v1`` path lands at
the doubled ``/v1/v1/...``.

``routes/api_surfaces.py`` registers thin delegating routes on the RouteIQ-owned
parent app at the bare external paths, mirroring ``routes/messages.py``.  These
tests assert:

1. The router resolves each surface at its BARE external path (not 404,
   route-table inspection) -- the core acceptance criterion.
2. The doubled ``/v1/...`` prefix is NOT reproduced by this router.
3. Through a TestClient, a POST to a bare surface reaches the delegating handler
   even when a ``/v1`` mount is present (ordering wins over the mount).
4. ``register_api_surface_routes`` degrades gracefully (returns False / no-op)
   when the upstream handlers cannot be imported.
5. A surface whose handler import fails is skipped individually (the others
   still register).

All tests are offline / credential-free.
"""

from __future__ import annotations

import pytest
from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from litellm_llmrouter.routes import api_surfaces
from litellm_llmrouter.routes.api_surfaces import (
    _SURFACES,
    build_api_surface_router,
    register_api_surface_routes,
)


def test_every_surface_resolves_at_its_bare_path():
    """Each configured surface resolves at its BARE external path (not /v1)."""
    router = build_api_surface_router()
    registered = {(route.path, frozenset(route.methods)) for route in router.routes}

    for surface in _SURFACES:
        # audio_speech lives in proxy_server.py whose import pulls optional deps
        # (boto3); it may be skipped in a standalone env.  Every OTHER surface
        # MUST resolve at its bare path.
        if surface.path == "/audio/speech":
            continue
        matched = any(
            path == surface.path and set(surface.methods).issubset(methods)
            for path, methods in registered
        )
        assert matched, f"surface {surface.methods} {surface.path} did not register"


def test_router_does_not_reproduce_doubled_v1_prefix():
    """The router exposes BARE paths, never the doubled /v1/v1/... form."""
    router = build_api_surface_router()
    paths = {route.path for route in router.routes}

    for path in paths:
        assert not path.startswith("/v1/v1/"), path
        # These are bare surfaces; they must not carry a /v1 prefix at all.
        assert not path.startswith("/v1/"), path


def test_core_surfaces_present():
    """Sanity: the headline surfaces named in RouteIQ-e48a are configured."""
    configured = {(s.path, s.methods) for s in _SURFACES}
    assert ("/responses", ("POST",)) in configured
    assert ("/rerank", ("POST",)) in configured
    assert ("/rag/ingest", ("POST",)) in configured
    assert ("/batches", ("POST",)) in configured
    assert ("/vector_stores", ("POST",)) in configured


def _build_app_with_surfaces_and_v1_mount(monkeypatch):
    """Assemble a parent app mirroring gateway/app.py composition order.

    Registers the bare-path surface router FIRST (with a fake build that wires
    stub handlers), then mounts a stub ``/v1`` sub-app whose ``/rerank`` 404s --
    exactly the broken upstream situation this fix corrects.

    Returns (app, calls).
    """
    calls: dict[str, dict] = {}

    async def _fake_user_api_key_auth():
        # Stand-in for LiteLLM's auth dependency.  No params so FastAPI doesn't
        # treat *args/**kwargs as query fields.
        return object()

    def _fake_build_router():
        from fastapi import Depends

        router = APIRouter()

        @router.post("/rerank")
        async def rerank(
            request: Request,
            fastapi_response: Response,
            user_api_key_dict=Depends(_fake_user_api_key_auth),
        ):
            body = await request.json()
            calls["rerank"] = {"model": body.get("model")}
            return JSONResponse({"results": [], "model": body.get("model")})

        @router.post("/responses")
        async def responses(
            request: Request,
            fastapi_response: Response,
            user_api_key_dict=Depends(_fake_user_api_key_auth),
        ):
            body = await request.json()
            calls["responses"] = {"model": body.get("model")}
            return JSONResponse({"id": "resp_test", "model": body.get("model")})

        return router

    monkeypatch.setattr(api_surfaces, "build_api_surface_router", _fake_build_router)

    app = FastAPI()
    registered = register_api_surface_routes(app)
    assert registered is True

    # Stub LiteLLM sub-app mounted at /v1.  It only knows /chat/completions;
    # /rerank + /responses 404 here -- exactly the broken upstream situation.
    litellm_stub = FastAPI()

    @litellm_stub.post("/chat/completions")
    async def _chat():
        return {"object": "chat.completion"}

    app.mount("/v1", litellm_stub)
    return app, calls


def test_post_bare_rerank_resolves_to_delegating_handler(monkeypatch):
    """A POST to the bare /rerank reaches our handler, not the /v1 mount 404."""
    app, calls = _build_app_with_surfaces_and_v1_mount(monkeypatch)
    client = TestClient(app)

    resp = client.post(
        "/rerank",
        json={"model": "rerank-v1", "query": "q", "documents": ["a", "b"]},
        headers={"Authorization": "Bearer test-api-key"},
    )

    # NOT a 404 -- the bare route matched before the /v1 mount catch-all.
    assert resp.status_code == 200, resp.text
    assert calls["rerank"]["model"] == "rerank-v1"


def test_post_bare_responses_resolves(monkeypatch):
    """The Responses surface resolves at the bare /responses path."""
    app, calls = _build_app_with_surfaces_and_v1_mount(monkeypatch)
    client = TestClient(app)

    resp = client.post(
        "/responses",
        json={"model": "gpt-4o", "input": "hi"},
        headers={"Authorization": "Bearer test-api-key"},
    )
    assert resp.status_code == 200, resp.text
    assert calls["responses"]["model"] == "gpt-4o"


def test_does_not_clobber_v1_mount(monkeypatch):
    """The /v1 mount's own routes still resolve (we only added bare paths)."""
    app, _ = _build_app_with_surfaces_and_v1_mount(monkeypatch)
    client = TestClient(app)

    resp = client.post("/v1/chat/completions", json={"model": "gpt-4", "messages": []})
    assert resp.status_code == 200, resp.text
    assert resp.json()["object"] == "chat.completion"


def test_register_degrades_when_handlers_unavailable(monkeypatch):
    """register_api_surface_routes returns False (no-op) if the build raises."""

    def _boom():
        raise ImportError("litellm endpoints unavailable")

    monkeypatch.setattr(api_surfaces, "build_api_surface_router", _boom)

    app = FastAPI()
    n_routes_before = len(app.routes)
    result = register_api_surface_routes(app)

    assert result is False
    assert len(app.routes) == n_routes_before


def test_build_skips_individual_failed_import(monkeypatch):
    """A single surface whose import fails is skipped; the others register."""
    good = _SURFACES[0]  # /responses POST

    def _ok():
        async def _h(request: Request):  # minimal valid endpoint
            return {}

        return _h

    def _bad():
        raise ImportError("nope")

    import dataclasses

    patched = (
        dataclasses.replace(good, import_handler=_ok),
        dataclasses.replace(good, path="/badone", import_handler=_bad),
    )
    monkeypatch.setattr(api_surfaces, "_SURFACES", patched)

    router = build_api_surface_router()
    paths = {route.path for route in router.routes}
    assert good.path in paths
    assert "/badone" not in paths


def test_build_imports_real_upstream_handlers():
    """When LiteLLM is installed, the real router wires the upstream handlers."""
    pytest.importorskip("litellm.proxy.rerank_endpoints.endpoints")

    router = build_api_surface_router()
    paths = {route.path for route in router.routes}
    # The dedicated-module surfaces import cleanly even without boto3.
    assert "/rerank" in paths
    assert "/responses" in paths
    assert "/rag/ingest" in paths
    assert "/batches" in paths
    assert "/vector_stores" in paths
