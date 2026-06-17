"""Multinode engine-affinity header passthrough (RouteIQ-bdd0 + RouteIQ-3316).

Disaggregated (prefill/decode-split) inference engines such as NVIDIA Dynamo and
vLLM-disagg route a request across separate prefill and decode workers and
coordinate KV-cache transfer between them. To make RouteIQ's session /
conversation affinity reach the *engine* decode worker -- not just RouteIQ's own
router -- the gateway must propagate a small set of passthrough signals to the
engine arm:

  * ``do_remote_prefill`` / ``do_remote_decode`` -- the disaggregation flags that
    tell the engine this request participates in remote prefill/decode (3316).
  * ``kv_transfer_params`` -- the KV-cache transfer descriptor the engine uses to
    move the prefill KV to the chosen decode worker (3316).
  * ``x-worker-instance-id`` -- the sticky decode-worker hint derived from the
    conversation/session affinity key, so a multi-turn conversation lands on the
    same decode worker and reuses its KV cache (bdd0: propagate the affinity key
    to the engine).

This module is a PURE passthrough helper at the engine-arm layer (it builds the
header/param dict the caller attaches to the outbound engine request); it does
NOT touch ``custom_routing_strategy`` / ``strategies`` (which are owned by other
waves). The routing decision -- which decode worker -- is made elsewhere; this
just carries the resulting affinity to the wire.

DEFAULT-OFF + byte-stable: every builder returns an EMPTY dict unless the
multinode passthrough is enabled (``ROUTEIQ_MULTINODE_AFFINITY_ENABLED=true``)
and the relevant input is present, so a default deployment attaches nothing new
to the outbound request.
"""

from __future__ import annotations

import os
from typing import Any, Optional

#: Header carrying the sticky decode-worker hint to the engine.
WORKER_INSTANCE_HEADER = "x-worker-instance-id"
#: Header carrying the opaque affinity/session key (echoed for engine stickiness).
AFFINITY_KEY_HEADER = "x-routeiq-affinity-key"


def multinode_affinity_enabled() -> bool:
    """Whether multinode engine-affinity passthrough is active (default OFF)."""
    return os.getenv("ROUTEIQ_MULTINODE_AFFINITY_ENABLED", "false").lower() == "true"


def build_affinity_headers(
    *,
    affinity_key: Optional[str] = None,
    worker_instance_id: Optional[str] = None,
) -> dict[str, str]:
    """Build the engine passthrough HEADERS for session/conversation affinity.

    Returns ``{x-worker-instance-id, x-routeiq-affinity-key}`` populated from the
    inputs that are present. Empty dict when the feature is off or no input is
    given, so the outbound request headers are unchanged (byte-stable).

    The ``worker_instance_id`` is the sticky decode-worker hint; when absent but
    an ``affinity_key`` is present, the affinity key alone is echoed so the
    engine layer can hash it to a worker itself.
    """
    if not multinode_affinity_enabled():
        return {}
    headers: dict[str, str] = {}
    if worker_instance_id:
        headers[WORKER_INSTANCE_HEADER] = str(worker_instance_id)
    if affinity_key:
        headers[AFFINITY_KEY_HEADER] = str(affinity_key)
    return headers


def build_kv_transfer_params(
    *,
    do_remote_prefill: bool = False,
    do_remote_decode: bool = False,
    kv_transfer_params: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build the engine passthrough PARAMS for disaggregated prefill/decode.

    Returns a dict suitable for merging into the outbound engine request body
    (e.g. an OpenAI-compatible ``extra_body`` / vLLM/Dynamo passthrough):

        {
          "do_remote_prefill": <bool>,   # only when True
          "do_remote_decode": <bool>,    # only when True
          "kv_transfer_params": {...},   # only when provided
        }

    Empty dict when the feature is off or no disaggregation signal is set, so the
    request body is unchanged (byte-stable). Only truthy flags are emitted so the
    common non-disagg path adds nothing.
    """
    if not multinode_affinity_enabled():
        return {}
    params: dict[str, Any] = {}
    if do_remote_prefill:
        params["do_remote_prefill"] = True
    if do_remote_decode:
        params["do_remote_decode"] = True
    if kv_transfer_params:
        params["kv_transfer_params"] = kv_transfer_params
    return params


def apply_engine_affinity(
    request_kwargs: dict[str, Any],
    *,
    affinity_key: Optional[str] = None,
    worker_instance_id: Optional[str] = None,
    do_remote_prefill: bool = False,
    do_remote_decode: bool = False,
    kv_transfer_params: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Merge engine-affinity headers + params into an outbound request kwargs dict.

    Returns a NEW dict (does not mutate the input) with the affinity headers
    merged under ``extra_headers`` and the disaggregation params merged at the
    top level. When the feature is off / nothing applies, returns a shallow copy
    of ``request_kwargs`` unchanged (byte-stable no-op).

    Existing caller-supplied ``extra_headers`` / params are never clobbered:
    affinity values fill only keys the caller has not already set.
    """
    out = dict(request_kwargs)
    headers = build_affinity_headers(
        affinity_key=affinity_key, worker_instance_id=worker_instance_id
    )
    if headers:
        existing = dict(out.get("extra_headers") or {})
        for k, v in headers.items():
            existing.setdefault(k, v)
        out["extra_headers"] = existing

    params = build_kv_transfer_params(
        do_remote_prefill=do_remote_prefill,
        do_remote_decode=do_remote_decode,
        kv_transfer_params=kv_transfer_params,
    )
    for k, v in params.items():
        out.setdefault(k, v)
    return out


__all__ = [
    "WORKER_INSTANCE_HEADER",
    "AFFINITY_KEY_HEADER",
    "multinode_affinity_enabled",
    "build_affinity_headers",
    "build_kv_transfer_params",
    "apply_engine_affinity",
]
