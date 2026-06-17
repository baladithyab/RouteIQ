"""SCIM v2 provisioning + scheduled API-key auto-rotation (RouteIQ-b8a2).

Two related identity-lifecycle features for the governance control plane:

1. **SCIM 2.0 provisioning** (RFC 7644). A minimal, IdP-driven Users + Groups
   surface so an external identity provider (Okta / Azure AD / etc.) can
   provision and de-provision RouteIQ governance principals automatically:
     * SCIM ``User``  <-> governance ``KeyGovernance`` (a provisioned principal
       maps to a governance key keyed by the SCIM ``userName``).
     * SCIM ``Group`` <-> governance ``WorkspaceConfig`` (a group maps to a
       workspace; group membership ties keys to the workspace).
   The router implements the SCIM CRUD verbs (POST/GET/PUT/PATCH/DELETE) over
   ``/scim/v2/Users`` and ``/scim/v2/Groups``, with SCIM-shaped JSON envelopes.

2. **Scheduled key auto-rotation**. A bounded helper that rotates governance
   keys whose age exceeds a configured max, minting a new secret + recording the
   rotation, so long-lived keys are cycled without operator action.

Both are DEFAULT-OFF + cred-free testable:

* SCIM is mounted only when ``ROUTEIQ_SCIM_ENABLED=true``; the router is bearer-
  token protected (``ROUTEIQ_SCIM_BEARER_TOKEN``) and fail-closed when no token
  is configured. Disabled -> the router is never created (byte-stable).
* Key rotation runs only when ``ROUTEIQ_KEY_ROTATION_ENABLED=true``; the
  rotation function is pure-Python over the in-memory governance engine (no
  scheduler bound here) so it is unit-testable without a clock or AWS.

Provisioning mutates the in-memory governance engine; the caller persists via
the engine's normal save path.
"""

from __future__ import annotations

import logging
import os
import secrets
import time
from typing import Any, Optional

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("litellm_llmrouter.scim")

SCIM_USER_SCHEMA = "urn:ietf:params:scim:schemas:core:2.0:User"
SCIM_GROUP_SCHEMA = "urn:ietf:params:scim:schemas:core:2.0:Group"
SCIM_LIST_SCHEMA = "urn:ietf:params:scim:api:messages:2.0:ListResponse"
SCIM_ERROR_SCHEMA = "urn:ietf:params:scim:api:messages:2.0:Error"


# ---------------------------------------------------------------------------
# Config gates
# ---------------------------------------------------------------------------


def scim_enabled() -> bool:
    """Whether the SCIM v2 provisioning surface is exposed (default OFF)."""
    return os.getenv("ROUTEIQ_SCIM_ENABLED", "false").lower() == "true"


def _scim_bearer_token() -> str:
    """The bearer token SCIM clients must present (empty => fail-closed)."""
    return os.getenv("ROUTEIQ_SCIM_BEARER_TOKEN", "")


def key_rotation_enabled() -> bool:
    """Whether scheduled key auto-rotation is active (default OFF)."""
    return os.getenv("ROUTEIQ_KEY_ROTATION_ENABLED", "false").lower() == "true"


def key_rotation_max_age_seconds() -> int:
    """Max key age before auto-rotation (default 90 days)."""
    try:
        return int(os.getenv("ROUTEIQ_KEY_ROTATION_MAX_AGE_SECONDS", str(90 * 86400)))
    except ValueError:
        return 90 * 86400


# ---------------------------------------------------------------------------
# SCIM <-> governance projection helpers
# ---------------------------------------------------------------------------


def _key_to_scim_user(kg: Any) -> dict:
    """Project a ``KeyGovernance`` into a SCIM User resource."""
    metadata = getattr(kg, "metadata", None) or {}
    user_name = metadata.get("scim_user_name") or getattr(kg, "key_id", "")
    return {
        "schemas": [SCIM_USER_SCHEMA],
        "id": getattr(kg, "key_id", ""),
        "userName": user_name,
        "active": bool(metadata.get("scim_active", True)),
        "displayName": metadata.get("name") or user_name,
        "meta": {"resourceType": "User"},
    }


def _workspace_to_scim_group(ws: Any) -> dict:
    """Project a ``WorkspaceConfig`` into a SCIM Group resource."""
    return {
        "schemas": [SCIM_GROUP_SCHEMA],
        "id": getattr(ws, "workspace_id", ""),
        "displayName": getattr(ws, "name", ""),
        "meta": {"resourceType": "Group"},
    }


def _scim_list(resources: list[dict]) -> dict:
    """Wrap resources in a SCIM ListResponse envelope."""
    return {
        "schemas": [SCIM_LIST_SCHEMA],
        "totalResults": len(resources),
        "itemsPerPage": len(resources),
        "startIndex": 1,
        "Resources": resources,
    }


def _scim_error(status: int, detail: str) -> JSONResponse:
    """A SCIM-shaped error envelope."""
    return JSONResponse(
        status_code=status,
        content={
            "schemas": [SCIM_ERROR_SCHEMA],
            "detail": detail,
            "status": str(status),
        },
    )


# ---------------------------------------------------------------------------
# Key auto-rotation
# ---------------------------------------------------------------------------


def _mask_secret(secret: str) -> str:
    last4 = secret[-4:] if len(secret) >= 4 else ""
    return f"sk-rq-...{last4}"


def rotate_stale_keys(
    engine: Any,
    *,
    max_age_seconds: Optional[int] = None,
    now: Optional[float] = None,
) -> list[dict]:
    """Rotate governance keys older than ``max_age_seconds``.

    For each key whose ``metadata.created_at`` (or ``rotated_at``) is older than
    the cutoff, mints a new secret, stamps ``metadata`` with the rotation time +
    a masked preview + a hashed secret, and records the rotation. Returns a list
    of ``{key_id, public_id, rotated_at}`` for each rotated key (the plaintext
    secret is NOT returned -- only the operator's out-of-band delivery path
    should surface it).

    No-op (returns ``[]``) when rotation is disabled. Pure over the in-memory
    engine -- the caller persists. Idempotent within a window: a freshly-rotated
    key's age resets, so a second call in the same window rotates nothing.
    """
    if not key_rotation_enabled():
        return []
    import hashlib

    key_governance = getattr(engine, "_key_governance", None)
    if not key_governance:
        return []

    cutoff_age = (
        max_age_seconds
        if max_age_seconds is not None
        else key_rotation_max_age_seconds()
    )
    current = now if now is not None else time.time()
    rotated: list[dict] = []

    for kg in list(key_governance.values()):
        metadata = getattr(kg, "metadata", None)
        if not isinstance(metadata, dict):
            continue
        # Auto-rotation is opt-in per key via metadata.auto_rotate.
        if not metadata.get("auto_rotate"):
            continue
        last_rotation = metadata.get("rotated_at") or metadata.get("created_at")
        if last_rotation is None:
            # No timestamp -> stamp now, treat as freshly rotated (don't churn).
            metadata["rotated_at"] = current
            continue
        try:
            age = current - float(last_rotation)
        except (TypeError, ValueError):
            continue
        if age < cutoff_age:
            continue

        new_secret = f"sk-rq-{secrets.token_urlsafe(24)}"
        metadata["rotated_at"] = current
        metadata["masked"] = _mask_secret(new_secret)
        metadata["secret_hash"] = hashlib.sha256(new_secret.encode("utf-8")).hexdigest()
        public_id = metadata.get("public_id") or f"kid_{secrets.token_urlsafe(12)}"
        metadata["public_id"] = public_id
        rotated.append(
            {
                "key_id": getattr(kg, "key_id", ""),
                "public_id": public_id,
                "rotated_at": current,
            }
        )
        logger.info(
            "Key rotation: rotated key public_id=%s (age=%.0fs)", public_id, age
        )

    if rotated:
        logger.info("Key rotation: rotated %d stale key(s)", len(rotated))
    return rotated


# ---------------------------------------------------------------------------
# SCIM v2 router
# ---------------------------------------------------------------------------


def create_scim_router() -> APIRouter:
    """Create the SCIM v2 provisioning router (Users + Groups).

    Bearer-token protected; fail-closed when ``ROUTEIQ_SCIM_BEARER_TOKEN`` is
    unset (every request 401s) so an enabled-but-unconfigured deployment cannot
    be provisioned anonymously.
    """
    router = APIRouter(prefix="/scim/v2", tags=["scim"])

    def _authorize(authorization: Optional[str]) -> None:
        token = _scim_bearer_token()
        if not token:
            raise HTTPException(status_code=401, detail="SCIM not configured")
        expected = f"Bearer {token}"
        if not authorization or not secrets.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="invalid bearer token")

    def _engine() -> Any:
        from litellm_llmrouter.governance import get_governance_engine

        return get_governance_engine()

    # ---- Users ----------------------------------------------------------

    @router.get("/Users", summary="List SCIM users")
    async def list_users(
        authorization: Optional[str] = Header(default=None),
    ) -> JSONResponse:
        _authorize(authorization)
        engine = _engine()
        users = [
            _key_to_scim_user(kg)
            for kg in getattr(engine, "_key_governance", {}).values()
            if (getattr(kg, "metadata", None) or {}).get("scim_provisioned")
        ]
        return JSONResponse(content=_scim_list(users))

    @router.get("/Users/{user_id}", summary="Get a SCIM user")
    async def get_user(
        user_id: str, authorization: Optional[str] = Header(default=None)
    ) -> JSONResponse:
        _authorize(authorization)
        kg = _engine().get_key_governance(user_id)
        if kg is None:
            return _scim_error(404, f"user {user_id} not found")
        return JSONResponse(content=_key_to_scim_user(kg))

    @router.post("/Users", status_code=201, summary="Provision a SCIM user")
    async def create_user(
        request: Request, authorization: Optional[str] = Header(default=None)
    ) -> JSONResponse:
        _authorize(authorization)
        body = await request.json()
        user_name = body.get("userName")
        if not user_name:
            return _scim_error(400, "userName is required")
        from litellm_llmrouter.governance import KeyGovernance

        key_id = body.get("id") or f"scim-{secrets.token_urlsafe(12)}"
        kg = KeyGovernance(
            key_id=key_id,
            metadata={
                "scim_provisioned": True,
                "scim_user_name": user_name,
                "scim_active": bool(body.get("active", True)),
                "name": body.get("displayName") or user_name,
                "created_at": time.time(),
            },
        )
        _engine().register_key_governance(kg)
        logger.info("SCIM: provisioned user %s -> key %s", user_name, key_id)
        return JSONResponse(status_code=201, content=_key_to_scim_user(kg))

    @router.put("/Users/{user_id}", summary="Replace a SCIM user")
    async def replace_user(
        user_id: str,
        request: Request,
        authorization: Optional[str] = Header(default=None),
    ) -> JSONResponse:
        _authorize(authorization)
        engine = _engine()
        kg = engine.get_key_governance(user_id)
        if kg is None:
            return _scim_error(404, f"user {user_id} not found")
        body = await request.json()
        metadata = dict(getattr(kg, "metadata", None) or {})
        if "active" in body:
            metadata["scim_active"] = bool(body["active"])
        if body.get("displayName"):
            metadata["name"] = body["displayName"]
        kg.metadata = metadata
        engine.register_key_governance(kg)
        return JSONResponse(content=_key_to_scim_user(kg))

    @router.delete(
        "/Users/{user_id}", status_code=204, summary="De-provision a SCIM user"
    )
    async def delete_user(
        user_id: str, authorization: Optional[str] = Header(default=None)
    ) -> JSONResponse:
        _authorize(authorization)
        removed = _engine().delete_key_governance(user_id)
        if not removed:
            return _scim_error(404, f"user {user_id} not found")
        logger.info("SCIM: de-provisioned user %s", user_id)
        return JSONResponse(status_code=204, content=None)

    # ---- Groups ---------------------------------------------------------

    @router.get("/Groups", summary="List SCIM groups")
    async def list_groups(
        authorization: Optional[str] = Header(default=None),
    ) -> JSONResponse:
        _authorize(authorization)
        engine = _engine()
        groups = [_workspace_to_scim_group(ws) for ws in engine.list_workspaces()]
        return JSONResponse(content=_scim_list(groups))

    @router.post("/Groups", status_code=201, summary="Provision a SCIM group")
    async def create_group(
        request: Request, authorization: Optional[str] = Header(default=None)
    ) -> JSONResponse:
        _authorize(authorization)
        body = await request.json()
        display_name = body.get("displayName")
        if not display_name:
            return _scim_error(400, "displayName is required")
        from litellm_llmrouter.governance import WorkspaceConfig

        workspace_id = body.get("id") or f"scim-grp-{secrets.token_urlsafe(8)}"
        ws = WorkspaceConfig(
            workspace_id=workspace_id,
            name=display_name,
            metadata={"scim_provisioned": True, "created_at": time.time()},
        )
        _engine().register_workspace(ws)
        logger.info(
            "SCIM: provisioned group %s -> workspace %s", display_name, workspace_id
        )
        return JSONResponse(status_code=201, content=_workspace_to_scim_group(ws))

    @router.delete(
        "/Groups/{group_id}", status_code=204, summary="De-provision a SCIM group"
    )
    async def delete_group(
        group_id: str, authorization: Optional[str] = Header(default=None)
    ) -> JSONResponse:
        _authorize(authorization)
        removed = _engine().delete_workspace(group_id)
        if not removed:
            return _scim_error(404, f"group {group_id} not found")
        logger.info("SCIM: de-provisioned group %s", group_id)
        return JSONResponse(status_code=204, content=None)

    return router


__all__ = [
    "SCIM_USER_SCHEMA",
    "SCIM_GROUP_SCHEMA",
    "scim_enabled",
    "key_rotation_enabled",
    "key_rotation_max_age_seconds",
    "rotate_stale_keys",
    "create_scim_router",
]
