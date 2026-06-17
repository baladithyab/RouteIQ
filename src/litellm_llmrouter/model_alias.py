"""Pre-routing model-name alias/rewrite layer (RouteIQ-0dcb).

Claude Code -- and any unmodified Anthropic / OpenAI SDK client -- PINS a
concrete model id in the request body (``claude-sonnet-4-20250514``). LiteLLM
matches ``model_list`` rows by EXACT ``model_name``, so a pinned id never lands
on a synthesized routing GROUP such as ``claude-auto``.

This module rewrites the request ``model`` field BEFORE the request reaches
LiteLLM, at the RouteIQ-owned app/route entry layer. It deliberately does NOT
use the ``PluginCallbackBridge.on_llm_pre_call`` seam (RouteIQ-60e3: that is a
logging seam that does NOT reliably mutate the outbound litellm request).
Instead it is a raw-ASGI middleware that mutates the buffered JSON body where
the request FIRST lands -- the verified, request-entry path the upstream
``/v1/messages`` and ``/v1/chat/completions`` handlers read directly.

Two maps drive the rewrite (see :class:`ModelAliasSettings`):

* ``exact``  -- ``{requested_id: target_group}`` lookup (fast path, short-circuits).
* ``regex``  -- ordered ``{pattern: target_group}`` rules; first ``fullmatch`` wins.

Default disabled / empty => IDENTITY (no rewrite), byte-stable. The middleware
self-disables (passes the scope through untouched) unless the alias layer is
enabled AND has at least one rule, so an off configuration adds no body buffering.

This module has NO import side effects: building the resolver or middleware is a
pure construction; nothing scans settings or compiles patterns at import time.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import FastAPI
    from starlette.types import ASGIApp, Message, Receive, Scope, Send

    from litellm_llmrouter.settings import ModelAliasSettings

logger = logging.getLogger(__name__)

#: Request paths whose JSON body carries a top-level ``model`` field that the
#: rewrite targets. Kept conservative: the Anthropic Messages surface plus the
#: OpenAI chat/completions + responses surfaces an Anthropic-compat or generic
#: client would hit through RouteIQ. Matched as a prefix against the ASGI path.
_REWRITE_PATHS: tuple[str, ...] = (
    "/v1/messages",
    "/v1/chat/completions",
    "/chat/completions",
    "/v1/completions",
    "/v1/responses",
    "/responses",
)


class ModelAliasResolver:
    """Compiles an alias config into a single ``resolve(model) -> model`` map.

    The resolver is the pure, testable core: it owns the exact lookup and the
    ordered, pre-compiled regex rules. The middleware is a thin ASGI wrapper
    over it. An uncompilable regex is logged once and skipped (fail-open) so a
    bad operator rule never blocks traffic.
    """

    def __init__(
        self,
        *,
        exact: Optional[dict[str, str]] = None,
        regex: Optional[dict[str, str]] = None,
    ) -> None:
        self._exact: dict[str, str] = dict(exact or {})
        self._regex: list[tuple[re.Pattern[str], str]] = []
        for pattern, target in (regex or {}).items():
            try:
                self._regex.append((re.compile(pattern), target))
            except re.error as exc:
                logger.warning(
                    "model-alias: skipping uncompilable regex rule %r -> %r: %s",
                    pattern,
                    target,
                    exc,
                )

    @property
    def has_rules(self) -> bool:
        """True when at least one exact or regex rule is configured."""
        return bool(self._exact) or bool(self._regex)

    def resolve(self, model: str) -> str:
        """Return the rewritten model name, or ``model`` unchanged (identity).

        Order: exact map first (a hit short-circuits), then the regex rules in
        insertion order -- the first ``fullmatch`` wins. Never raises.
        """
        if not model:
            return model
        target = self._exact.get(model)
        if target is not None:
            return target
        for pattern, dest in self._regex:
            if pattern.fullmatch(model):
                return dest
        return model

    @classmethod
    def from_settings(cls, settings: "ModelAliasSettings") -> "ModelAliasResolver":
        """Build a resolver from a :class:`ModelAliasSettings` block."""
        return cls(exact=settings.exact, regex=settings.regex)


def _rewrite_body_model(raw: bytes, resolver: ModelAliasResolver) -> bytes:
    """Rewrite the top-level ``model`` field in a JSON request body.

    Returns the original bytes unchanged when the body is not a JSON object,
    carries no ``model``, or the resolver leaves the name unchanged -- so a
    no-op rewrite is byte-identical (no re-serialization churn).
    """
    if not raw:
        return raw
    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return raw
    if not isinstance(payload, dict):
        return raw
    model = payload.get("model")
    if not isinstance(model, str):
        return raw
    new_model = resolver.resolve(model)
    if new_model == model:
        return raw
    payload["model"] = new_model
    logger.debug("model-alias: rewrote model %r -> %r", model, new_model)
    return json.dumps(payload).encode("utf-8")


class ModelAliasMiddleware:
    """Raw-ASGI middleware that rewrites the request body's ``model`` field.

    Raw ASGI (the RequestIDMiddleware / BackpressureMiddleware pattern) rather
    than ``BaseHTTPMiddleware`` so it never buffers the RESPONSE (streaming
    safe). It buffers only the REQUEST body for the targeted POST paths, applies
    the resolver, and replays the (possibly rewritten) body downstream via a
    wrapped ``receive``. Non-targeted paths / methods pass straight through with
    zero buffering.
    """

    def __init__(self, app: "ASGIApp", resolver: ModelAliasResolver) -> None:
        self.app = app
        self.resolver = resolver

    async def __call__(self, scope: "Scope", receive: "Receive", send: "Send") -> None:
        if scope["type"] != "http" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        if not any(path.startswith(p) for p in _REWRITE_PATHS):
            await self.app(scope, receive, send)
            return

        # Buffer the full request body (the upstream handlers read it whole).
        body = b""
        more_body = True
        while more_body:
            message = await receive()
            if message["type"] != "http.request":
                # Forward any non-body message (e.g. http.disconnect) verbatim.
                await self.app(scope, _single_message_receive(message), send)
                return
            body += message.get("body", b"")
            more_body = message.get("more_body", False)

        try:
            body = _rewrite_body_model(body, self.resolver)
        except Exception as exc:  # pragma: no cover - defensive, must fail open
            logger.warning("model-alias: rewrite failed, passing through: %s", exc)

        await self.app(scope, _replay_body_receive(body), send)


def _single_message_receive(message: "Message") -> "Receive":
    """A receive() that yields one captured message, then disconnects."""
    sent = False

    async def _receive() -> "Message":
        nonlocal sent
        if not sent:
            sent = True
            return message
        return {"type": "http.disconnect"}

    return _receive


def _replay_body_receive(body: bytes) -> "Receive":
    """A receive() that replays the (rewritten) body as one final chunk."""
    sent = False

    async def _receive() -> "Message":
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    return _receive


def add_model_alias_middleware(app: "FastAPI") -> bool:
    """Register the model-alias rewrite middleware when enabled & non-empty.

    Reads ``settings.model_alias``; a no-op (returns ``False``) when the layer
    is disabled or has no rules, so the default config adds zero overhead.
    Idempotent at the call-site level (called once from ``create_gateway_app``).
    Never raises -- a config error degrades to identity (no middleware).

    Args:
        app: The RouteIQ-owned FastAPI application.

    Returns:
        ``True`` if the middleware was added, ``False`` otherwise.
    """
    try:
        from litellm_llmrouter.settings import get_settings

        cfg = getattr(get_settings(), "model_alias", None)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("model-alias: settings unavailable, skipping: %s", exc)
        return False

    if cfg is None or not getattr(cfg, "enabled", False):
        return False

    resolver = ModelAliasResolver.from_settings(cfg)
    if not resolver.has_rules:
        logger.info("model-alias: enabled but no rules configured; skipping (identity)")
        return False

    app.add_middleware(ModelAliasMiddleware, resolver=resolver)
    logger.info(
        "Added ModelAliasMiddleware (exact=%d rules, regex=%d rules)",
        len(cfg.exact),
        len(cfg.regex),
    )
    return True
