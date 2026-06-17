"""First-class RouteIQ API surfaces at the BARE external paths (RouteIQ-e48a).

Why this module exists
----------------------
RouteIQ mounts the upstream LiteLLM proxy as a sub-application at ``/v1`` (see
``gateway/app.py``).  Starlette mounts *prepend* their mount path to every route
the sub-app registers, so an upstream route registered as ``/responses`` is
reachable externally only at ``/v1/responses`` -- and an upstream route
registered as ``/v1/responses`` ends up at the doubled ``/v1/v1/responses``.

The OpenAI / Cohere style API families (Responses, rerank, RAG, batches,
vector-store, audio, images) each register BOTH a bare path (``/responses``,
``/rerank``, ...) AND a ``/v1``-prefixed path upstream.  Because the whole
sub-app is mounted under ``/v1``:

* the bare upstream path ``/rerank`` is only reachable at ``/v1/rerank``
  (so a client POSTing the documented bare ``/rerank`` gets a 404), and
* the upstream ``/v1/rerank`` lands at the doubled ``/v1/v1/rerank``.

This mirrors exactly the problem ``routes/messages.py`` (RouteIQ-0e37) fixed for
the Anthropic Messages family: the bare external path 404s for SDK callers.

The fix
-------
Register thin delegating routes **on the RouteIQ-owned parent app** at the bare
external paths (``/responses``, ``/rerank``, ``/rag/ingest``, ``/batches``,
``/vector_stores``, ``/audio/speech``, ``/images/generations``, ...).  RouteIQ
registers its own routes BEFORE the ``/v1`` mount, so these match first and the
caller hits a first-class RouteIQ route that gets the full RouteIQ middleware
stack (governance / auth / telemetry) and is documented in the OpenAPI schema.

Each handler delegates to the upstream LiteLLM handler, forwarding the same
``Request`` / ``Response`` objects and the ``user_api_key_dict`` resolved by
LiteLLM's own ``user_api_key_auth`` dependency.  Because the upstream handlers
read the raw request body and headers off the ``Request`` directly, bodies,
headers, and SSE streaming flow through unchanged -- we re-implement nothing.

This module has **no import side effects**: :func:`register_api_surface_routes`
imports the upstream handlers lazily and is a no-op when LiteLLM is not installed
(standalone / unit-test mode).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from fastapi import APIRouter

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import FastAPI

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SurfaceRoute:
    """A single bare-path surface to register on the parent app.

    ``import_handler`` is a zero-arg callable that lazily imports and returns the
    upstream LiteLLM handler.  It is only called when LiteLLM is present, so the
    module stays importable standalone.  ``methods`` lists the HTTP verbs and
    ``path`` is the bare external path the delegator registers.
    """

    path: str
    methods: tuple[str, ...]
    import_handler: Callable[[], Callable[..., Any]]
    tags: tuple[str, ...] = field(default_factory=tuple)


def _responses_handler() -> Callable[..., Any]:
    from litellm.proxy.response_api_endpoints.endpoints import responses_api

    return responses_api


def _get_response_handler() -> Callable[..., Any]:
    from litellm.proxy.response_api_endpoints.endpoints import get_response

    return get_response


def _delete_response_handler() -> Callable[..., Any]:
    from litellm.proxy.response_api_endpoints.endpoints import delete_response

    return delete_response


def _rerank_handler() -> Callable[..., Any]:
    from litellm.proxy.rerank_endpoints.endpoints import rerank

    return rerank


def _rag_ingest_handler() -> Callable[..., Any]:
    from litellm.proxy.rag_endpoints.endpoints import rag_ingest

    return rag_ingest


def _rag_query_handler() -> Callable[..., Any]:
    from litellm.proxy.rag_endpoints.endpoints import rag_query

    return rag_query


def _create_batch_handler() -> Callable[..., Any]:
    from litellm.proxy.batches_endpoints.endpoints import create_batch

    return create_batch


def _list_batches_handler() -> Callable[..., Any]:
    from litellm.proxy.batches_endpoints.endpoints import list_batches

    return list_batches


def _vector_store_create_handler() -> Callable[..., Any]:
    from litellm.proxy.vector_store_endpoints.endpoints import vector_store_create

    return vector_store_create


def _vector_store_list_handler() -> Callable[..., Any]:
    from litellm.proxy.vector_store_endpoints.endpoints import vector_store_list

    return vector_store_list


def _vector_store_search_handler() -> Callable[..., Any]:
    from litellm.proxy.vector_store_endpoints.endpoints import vector_store_search

    return vector_store_search


def _audio_speech_handler() -> Callable[..., Any]:
    from litellm.proxy.proxy_server import audio_speech

    return audio_speech


def _image_generation_handler() -> Callable[..., Any]:
    from litellm.proxy.image_endpoints.endpoints import image_generation

    return image_generation


# The bare-path surfaces RouteIQ promotes to first-class routes.  Each delegates
# to the same upstream handler the /v1 mount would have served -- so behaviour is
# identical, but the route is now reachable at the documented bare path AND gets
# RouteIQ's middleware + OpenAPI documentation.  Path params use the SAME names
# the upstream handler declares so FastAPI binds them when delegating.
_SURFACES: tuple[_SurfaceRoute, ...] = (
    # Responses API (OpenAI Responses spec).
    _SurfaceRoute("/responses", ("POST",), _responses_handler, ("responses",)),
    _SurfaceRoute(
        "/responses/{response_id}", ("GET",), _get_response_handler, ("responses",)
    ),
    _SurfaceRoute(
        "/responses/{response_id}",
        ("DELETE",),
        _delete_response_handler,
        ("responses",),
    ),
    # Rerank (Cohere / Jina style).
    _SurfaceRoute("/rerank", ("POST",), _rerank_handler, ("rerank",)),
    # RAG ingest / query.
    _SurfaceRoute("/rag/ingest", ("POST",), _rag_ingest_handler, ("rag",)),
    _SurfaceRoute("/rag/query", ("POST",), _rag_query_handler, ("rag",)),
    # Batches.
    _SurfaceRoute("/batches", ("POST",), _create_batch_handler, ("batch",)),
    _SurfaceRoute("/batches", ("GET",), _list_batches_handler, ("batch",)),
    # Vector stores.
    _SurfaceRoute(
        "/vector_stores", ("POST",), _vector_store_create_handler, ("vector_store",)
    ),
    _SurfaceRoute(
        "/vector_stores", ("GET",), _vector_store_list_handler, ("vector_store",)
    ),
    _SurfaceRoute(
        "/vector_stores/{vector_store_id:path}/search",
        ("POST",),
        _vector_store_search_handler,
        ("vector_store",),
    ),
    # Audio (JSON-bodied text-to-speech).
    _SurfaceRoute("/audio/speech", ("POST",), _audio_speech_handler, ("audio",)),
    # Images (generations is JSON-bodied; edits is multipart and stays on /v1).
    _SurfaceRoute(
        "/images/generations", ("POST",), _image_generation_handler, ("images",)
    ),
)


def build_api_surface_router() -> "APIRouter":
    """Build the bare-path first-class API-surface router.

    Each route delegates to the matching upstream LiteLLM handler.  The handler's
    own ``Depends(user_api_key_auth)`` dependency is preserved because we register
    the *handler function itself* as the endpoint (it still carries its default
    parameters / dependencies), so authentication behaves identically to hitting
    LiteLLM directly.

    Returns:
        An :class:`fastapi.APIRouter` registering the bare external paths.  A
        surface whose upstream handler cannot be imported (e.g. an optional
        provider dep like ``boto3`` is absent in a standalone env) is skipped
        individually rather than aborting the whole router -- so the surfaces
        that DO import are still promoted.
    """
    router = APIRouter()

    for surface in _SURFACES:
        # Resolve the upstream handler lazily.  Skip (don't abort) on a per-
        # surface import failure so one optional dep never drops every surface.
        try:
            handler = surface.import_handler()
        except Exception as exc:  # pragma: no cover - optional-dep degradation
            logger.warning(
                "Skipping bare-path surface %s %s (upstream handler unavailable: %s)",
                "/".join(surface.methods),
                surface.path,
                exc,
            )
            continue
        router.add_api_route(
            surface.path,
            handler,
            methods=list(surface.methods),
            tags=list(surface.tags),
        )

    return router


def register_api_surface_routes(app: "FastAPI") -> bool:
    """Register the bare-path first-class API surfaces on the parent app.

    Must be called **before** LiteLLM is mounted at ``/v1`` so these explicit
    bare routes win over the mount's catch-all.  No-op (returns ``False``) when
    the upstream LiteLLM endpoints cannot be imported, so RouteIQ still boots in
    standalone / unit-test mode.

    Args:
        app: The RouteIQ-owned FastAPI application.

    Returns:
        ``True`` if the routes were registered, ``False`` otherwise.
    """
    try:
        router = build_api_surface_router()
    except Exception as exc:  # pragma: no cover - defensive (LiteLLM absent)
        logger.warning(
            "First-class API surface routes not registered "
            "(upstream LiteLLM handlers unavailable): %s",
            exc,
        )
        return False

    app.include_router(router)
    logger.info(
        "Registered %d first-class API surface routes at bare paths "
        "(responses, rerank, rag, batches, vector_stores, audio, images) "
        "so they get RouteIQ middleware + documentation (RouteIQ-e48a)",
        len(_SURFACES),
    )
    return True
