"""Credential-free unit tests for the pre-routing model-alias rewrite (RouteIQ-0dcb).

Claude Code pins a concrete model id; LiteLLM matches model_name exactly, so a
pinned id never hits a routing GROUP like ``claude-auto``. The model-alias layer
rewrites the request ``model`` field PRE-routing, at the RouteIQ app/route entry
layer (a raw-ASGI middleware -- NOT the broken ``on_llm_pre_call`` mutation
seam), so an unmodified Anthropic client is transparently routed through the
group.

Covers:
* the pure resolver (exact short-circuit, regex fullmatch, identity passthrough,
  fail-open on bad regex);
* the body-rewrite helper (byte-stable no-op, non-JSON / no-model passthrough);
* the ASGI middleware end-to-end through a TestClient (mapped id rewritten,
  unmapped id passes through, non-targeted paths untouched);
* the ``add_model_alias_middleware`` gating (off by default; no rules => no-op).

All offline / credential-free.
"""

from __future__ import annotations

import json

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from litellm_llmrouter.model_alias import (
    ModelAliasMiddleware,
    ModelAliasResolver,
    _rewrite_body_model,
    add_model_alias_middleware,
)
from litellm_llmrouter.settings import (
    ModelAliasSettings,
    reset_settings,
)


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


def test_resolver_exact_match_rewrites():
    r = ModelAliasResolver(exact={"claude-sonnet-4-20250514": "claude-auto"})
    assert r.resolve("claude-sonnet-4-20250514") == "claude-auto"


def test_resolver_unmapped_passes_through():
    r = ModelAliasResolver(exact={"claude-sonnet-4-20250514": "claude-auto"})
    assert r.resolve("gpt-4o") == "gpt-4o"


def test_resolver_regex_fullmatch_rewrites():
    r = ModelAliasResolver(regex={r"^claude-.*$": "claude-auto"})
    assert r.resolve("claude-opus-4-20250514") == "claude-auto"


def test_resolver_regex_is_fullmatch_not_search():
    r = ModelAliasResolver(regex={r"claude": "claude-auto"})
    # 'claude' does not fullmatch 'claude-opus-4' -> identity.
    assert r.resolve("claude-opus-4") == "claude-opus-4"


def test_resolver_exact_short_circuits_before_regex():
    r = ModelAliasResolver(
        exact={"claude-opus-4": "exact-target"},
        regex={r"^claude-.*$": "regex-target"},
    )
    assert r.resolve("claude-opus-4") == "exact-target"


def test_resolver_first_regex_rule_wins():
    r = ModelAliasResolver(
        regex={r"^claude-opus.*$": "opus-group", r"^claude-.*$": "all-claude"}
    )
    assert r.resolve("claude-opus-4") == "opus-group"
    assert r.resolve("claude-sonnet-4") == "all-claude"


def test_resolver_bad_regex_is_skipped_fail_open():
    # An uncompilable pattern is skipped, not raised; the good rule still applies.
    r = ModelAliasResolver(regex={r"[unclosed": "bad", r"^claude-.*$": "claude-auto"})
    assert r.resolve("claude-sonnet-4") == "claude-auto"


def test_resolver_empty_is_identity():
    r = ModelAliasResolver()
    assert not r.has_rules
    assert r.resolve("anything") == "anything"


def test_resolver_from_settings():
    s = ModelAliasSettings(
        enabled=True,
        exact={"a": "b"},
        regex={r"^x.*$": "y"},
    )
    r = ModelAliasResolver.from_settings(s)
    assert r.has_rules
    assert r.resolve("a") == "b"
    assert r.resolve("xyz") == "y"


# ---------------------------------------------------------------------------
# Body rewrite helper
# ---------------------------------------------------------------------------


def test_rewrite_body_changes_model():
    r = ModelAliasResolver(exact={"old": "new"})
    out = _rewrite_body_model(b'{"model": "old", "messages": []}', r)
    assert json.loads(out)["model"] == "new"


def test_rewrite_body_noop_is_byte_identical():
    r = ModelAliasResolver(exact={"old": "new"})
    raw = b'{"model": "unmapped", "messages": []}'
    # No rewrite => exact same bytes back (no re-serialization churn).
    assert _rewrite_body_model(raw, r) is raw


def test_rewrite_body_non_json_passthrough():
    r = ModelAliasResolver(exact={"old": "new"})
    raw = b"not json at all"
    assert _rewrite_body_model(raw, r) is raw


def test_rewrite_body_no_model_field_passthrough():
    r = ModelAliasResolver(exact={"old": "new"})
    raw = b'{"messages": []}'
    assert _rewrite_body_model(raw, r) is raw


def test_rewrite_body_non_string_model_passthrough():
    r = ModelAliasResolver(exact={"old": "new"})
    raw = b'{"model": 123}'
    assert _rewrite_body_model(raw, r) is raw


def test_rewrite_body_empty_passthrough():
    r = ModelAliasResolver(exact={"old": "new"})
    assert _rewrite_body_model(b"", r) == b""


# ---------------------------------------------------------------------------
# Middleware end-to-end (TestClient)
# ---------------------------------------------------------------------------


def _echo_app(resolver: ModelAliasResolver) -> FastAPI:
    """An app whose /v1/messages echoes the (post-middleware) model it received."""
    app = FastAPI()

    @app.post("/v1/messages")
    async def messages(request: Request):
        body = await request.json()
        return {"model": body.get("model")}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        return {"model": body.get("model")}

    @app.post("/other")
    async def other(request: Request):
        body = await request.json()
        return {"model": body.get("model")}

    app.add_middleware(ModelAliasMiddleware, resolver=resolver)
    return app


def test_middleware_rewrites_mapped_model_on_messages_path():
    resolver = ModelAliasResolver(exact={"claude-sonnet-4-20250514": "claude-auto"})
    client = TestClient(_echo_app(resolver))
    resp = client.post(
        "/v1/messages",
        json={"model": "claude-sonnet-4-20250514", "messages": []},
    )
    assert resp.status_code == 200, resp.text
    # The handler saw the REWRITTEN model -> the rewrite happened pre-routing.
    assert resp.json()["model"] == "claude-auto"


def test_middleware_passes_through_unmapped_model():
    resolver = ModelAliasResolver(exact={"claude-sonnet-4-20250514": "claude-auto"})
    client = TestClient(_echo_app(resolver))
    resp = client.post("/v1/messages", json={"model": "gpt-4o", "messages": []})
    assert resp.json()["model"] == "gpt-4o"


def test_middleware_ignores_non_targeted_paths():
    """A path outside the rewrite set is never touched even if mapped."""
    resolver = ModelAliasResolver(exact={"claude-sonnet-4-20250514": "claude-auto"})
    client = TestClient(_echo_app(resolver))
    resp = client.post(
        "/other", json={"model": "claude-sonnet-4-20250514", "messages": []}
    )
    # /other is not in _REWRITE_PATHS -> identity.
    assert resp.json()["model"] == "claude-sonnet-4-20250514"


def test_middleware_regex_rewrite_end_to_end():
    resolver = ModelAliasResolver(regex={r"^claude-.*$": "claude-auto"})
    client = TestClient(_echo_app(resolver))
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "claude-haiku-3-5", "messages": []},
    )
    assert resp.json()["model"] == "claude-auto"


# ---------------------------------------------------------------------------
# Gating: add_model_alias_middleware
# ---------------------------------------------------------------------------


def test_add_middleware_noop_when_disabled(monkeypatch):
    reset_settings()
    try:
        # Default settings: model_alias disabled -> not added.
        app = FastAPI()
        assert add_model_alias_middleware(app) is False
    finally:
        reset_settings()


def test_add_middleware_noop_when_enabled_but_no_rules(monkeypatch):
    monkeypatch.setenv("ROUTEIQ_MODEL_ALIAS__ENABLED", "true")
    reset_settings()
    try:
        app = FastAPI()
        # Enabled but no exact/regex rules -> identity, not added.
        assert add_model_alias_middleware(app) is False
    finally:
        reset_settings()


def test_add_middleware_added_when_enabled_with_rules(monkeypatch):
    monkeypatch.setenv("ROUTEIQ_MODEL_ALIAS__ENABLED", "true")
    monkeypatch.setenv(
        "ROUTEIQ_MODEL_ALIAS__EXACT",
        json.dumps({"claude-sonnet-4-20250514": "claude-auto"}),
    )
    reset_settings()
    try:
        app = FastAPI()
        assert add_model_alias_middleware(app) is True
    finally:
        reset_settings()


def test_settings_model_alias_defaults():
    s = ModelAliasSettings()
    assert s.enabled is False
    assert s.exact == {}
    assert s.regex == {}
