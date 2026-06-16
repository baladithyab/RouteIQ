"""
Anthropic Messages API surface at the BARE external path.

Why this module exists
----------------------
RouteIQ mounts the upstream LiteLLM proxy as a sub-application at ``/v1`` (see
``gateway/app.py``).  Starlette mounts *prepend* their mount path to every route
the sub-app registers.  Upstream LiteLLM registers the Anthropic Messages family
**only** at the ``/v1``-prefixed paths::

    POST /v1/messages
    POST /v1/messages/count_tokens

(unlike ``/responses`` and ``/rerank``, which register *both* the bare and the
``/v1`` variant).  Mounting that router under ``/v1`` therefore exposes the
Messages family at ``/v1/v1/messages`` externally — so an Anthropic SDK caller
POSTing to ``/v1/messages`` gets a 404.  (Upstream additionally lazy-loads the
Anthropic router only when it sees a path starting with ``/v1/messages`` *inside*
the sub-app, but the mount has already stripped the ``/v1`` prefix, so the lazy
trigger never fires either.)

The fix
-------
Register a thin router **on the RouteIQ-owned parent app** at the bare external
paths ``/v1/messages`` and ``/v1/messages/count_tokens``.  RouteIQ registers its
own routes before the ``/v1`` mount, so these match first and the doubled-prefix
mount path is never reached for them.

Each handler delegates to the upstream LiteLLM handler, forwarding the same
``Request`` / ``Response`` objects and the ``user_api_key_dict`` resolved by
LiteLLM's own ``user_api_key_auth`` dependency.  Because the upstream handlers
read the raw request body and headers off the ``Request`` directly, the
Anthropic beta headers (extended-thinking, prompt-caching) and SSE streaming
flow through unchanged — we re-implement nothing.

This module has **no import side effects**: ``register_messages_routes(app)``
imports the upstream handlers lazily and is a no-op when LiteLLM is not
installed (standalone / unit-test mode).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Request, Response

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import FastAPI

logger = logging.getLogger(__name__)


def build_messages_router() -> "APIRouter":
    """Build the bare-path Anthropic Messages router.

    The router is wired to the upstream LiteLLM Anthropic handlers and its own
    ``user_api_key_auth`` dependency (mirroring upstream) so authentication and
    beta-header passthrough behave identically to hitting LiteLLM directly.

    Returns:
        An :class:`fastapi.APIRouter` registering ``POST /v1/messages`` and
        ``POST /v1/messages/count_tokens`` at the bare external path.

    Raises:
        ImportError: If the upstream LiteLLM Anthropic endpoints are unavailable.
    """
    # Imported here (not at module import time) so this module stays free of
    # import side effects and importable without LiteLLM present.
    from fastapi import Depends

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.anthropic_endpoints.endpoints import (
        anthropic_response as _upstream_anthropic_response,
        count_tokens as _upstream_count_tokens,
    )
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    router = APIRouter(tags=["[beta] Anthropic `/v1/messages`"])

    @router.post("/v1/messages")
    async def messages(  # pyright: ignore[reportUnusedFunction]
        fastapi_response: Response,
        request: Request,
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    ) -> Any:
        """Anthropic Messages API at the bare external ``/v1/messages`` path.

        Delegates to the upstream LiteLLM handler; body, headers (including the
        ``anthropic-beta`` extended-thinking / prompt-caching headers), and SSE
        streaming pass through unchanged.
        """
        return await _upstream_anthropic_response(
            fastapi_response=fastapi_response,
            request=request,
            user_api_key_dict=user_api_key_dict,
        )

    @router.post("/v1/messages/count_tokens")
    async def messages_count_tokens(  # pyright: ignore[reportUnusedFunction]
        request: Request,
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    ) -> Any:
        """Anthropic token-counting at the bare ``/v1/messages/count_tokens`` path.

        Delegates to the upstream LiteLLM ``count_tokens`` handler.
        """
        return await _upstream_count_tokens(
            request=request,
            user_api_key_dict=user_api_key_dict,
        )

    return router


def register_messages_routes(app: "FastAPI") -> bool:
    """Register the bare-path Anthropic Messages routes on the parent app.

    Must be called **before** LiteLLM is mounted at ``/v1`` so these explicit
    routes win over the mount's catch-all.  No-op (returns ``False``) when the
    upstream LiteLLM Anthropic endpoints cannot be imported, so RouteIQ still
    boots in standalone / unit-test mode.

    Args:
        app: The RouteIQ-owned FastAPI application.

    Returns:
        ``True`` if the routes were registered, ``False`` otherwise.
    """
    try:
        router = build_messages_router()
    except Exception as exc:  # pragma: no cover - defensive (LiteLLM absent)
        logger.warning(
            "Anthropic Messages bare-path routes not registered "
            "(upstream LiteLLM handler unavailable): %s",
            exc,
        )
        return False

    app.include_router(router)
    logger.info(
        "Registered bare-path Anthropic Messages routes: "
        "POST /v1/messages, POST /v1/messages/count_tokens "
        "(avoids /v1/v1/messages double-prefix from the /v1 LiteLLM mount)"
    )
    return True
