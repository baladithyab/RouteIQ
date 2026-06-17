"""
Bedrock Request Levers Plugin
=============================

Three per-request Bedrock Converse levers that mutate the OUTBOUND request on
the correct litellm seam. The plugin's ``on_llm_pre_call`` returns a kwargs
override; that override is applied by :class:`PluginCallbackBridge` from
``CustomLogger.async_pre_call_deployment_hook`` (the documented "modify the
request AFTER a deployment is selected, but BEFORE the request is sent" seam,
``custom_logger.py:264``). litellm replaces the completion ``kwargs`` with the
hook's return value (``utils.py:1815``), so the mutations reach the wire.

The three levers, each independently gated and default-OFF (byte-stable when
off -- no kwarg is added):

RouteIQ-294a: ``requestMetadata`` tenant tags
    Forwards tenant identity (workspace_id / key_id, read from the governance
    stamp ``metadata["_governance_ctx"]``) as Bedrock ``requestMetadata`` tags.
    Lands as the TOP-LEVEL completion kwarg ``requestMetadata`` -- a litellm
    Bedrock ``supported_param`` that ``AmazonConverseConfig.map_openai_params``
    reads from ``non_default_params`` and writes onto the converse body
    (``converse_transformation.py:960,997,1688``). NOT a logging-kwarg.

RouteIQ-9cd8: per-team logging callback override
    Sets per-team logging sinks as the TOP-LEVEL request kwargs ``callbacks`` /
    ``success_callback`` / ``failure_callback`` -- the exact keys litellm sources
    dynamic per-request callbacks from (``utils.py:791,903,923,927``). NOT
    ``metadata["callbacks"]`` (never read). The global callback registration is
    untouched; this only ADDS per-request sinks for a mapped team.

RouteIQ-b9ee: Bedrock prompt-caching ``cachePoint``
    Drives caching via litellm's ``cache_control_injection_points`` mechanism
    (a top-level completion kwarg). litellm auto-wires the
    ``anthropic_cache_control_hook`` for ``message``-location points (system
    prefix), and the Bedrock converse transform appends a tool_config
    ``cachePoint`` for ``tool_config``-location points when the request carries
    tools (``converse_transformation.py:1647-1654``). NOT a bare ``tools[]``
    ``cachePoint`` entry, which the converse transform drops.

Config: :class:`litellm_llmrouter.settings.BedrockRequestLeversSettings`
(``settings.bedrock_levers``). Env prefix ``ROUTEIQ_BEDROCK_LEVERS__``.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from litellm_llmrouter.gateway.plugin_manager import (
    GatewayPlugin,
    PluginCapability,
    PluginContext,
    PluginMetadata,
)

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

__all__ = ["BedrockRequestLeversPlugin"]

# Bedrock requestMetadata keys/values are restricted to [a-zA-Z0-9 :_@$#=/+,.-]
# (converse_transformation._validate_request_metadata). Strip anything else so a
# stray character in a workspace id never 400s the request.
_BEDROCK_METADATA_SANITIZE = re.compile(r"[^a-zA-Z0-9 :_@$#=/+,.\-]")


class BedrockRequestLeversPlugin(GatewayPlugin):
    """Per-request Bedrock Converse levers (requestMetadata / team callbacks /
    cachePoint), applied on the pre-call request-mutation seam."""

    def __init__(self) -> None:
        self._request_metadata: bool = False
        self._metadata_prefix: str = "routeiq_"
        self._team_callbacks: bool = False
        self._team_callback_map: dict[str, dict[str, list[str]]] = {}
        self._cache_point: bool = False
        self._cache_system: bool = True
        self._cache_tools: bool = True

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="bedrock-request-levers",
            version="1.0.0",
            capabilities={PluginCapability.OBSERVABILITY_EXPORTER},
            priority=60,
            description=(
                "Per-request Bedrock requestMetadata / team callbacks / "
                "cachePoint on the pre-call mutation seam (default off)."
            ),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def startup(
        self, app: "FastAPI", context: PluginContext | None = None
    ) -> None:
        from litellm_llmrouter.settings import get_settings

        cfg = get_settings().bedrock_levers
        self._request_metadata = cfg.request_metadata
        self._metadata_prefix = cfg.metadata_prefix
        self._team_callbacks = cfg.team_callbacks
        self._team_callback_map = dict(cfg.team_callback_map or {})
        self._cache_point = cfg.cache_point
        self._cache_system = cfg.cache_system
        self._cache_tools = cfg.cache_tools

        if not (self._request_metadata or self._team_callbacks or self._cache_point):
            logger.info("Bedrock request levers plugin: all levers off (no-op)")
        else:
            logger.info(
                "Bedrock request levers active: requestMetadata=%s team_callbacks=%s "
                "cachePoint=%s",
                self._request_metadata,
                self._team_callbacks,
                self._cache_point,
            )

    async def shutdown(
        self, app: "FastAPI", context: PluginContext | None = None
    ) -> None:
        pass

    # ------------------------------------------------------------------
    # Request mutation seam
    # ------------------------------------------------------------------

    async def on_llm_pre_call(
        self, model: str, messages: list[Any], kwargs: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Build the kwargs override applied by the deployment hook.

        Returns ``None`` (no change, byte-stable) when every lever is off or
        produces no mutation. Otherwise returns a dict merged into the outbound
        completion kwargs by the callback bridge.
        """
        override: dict[str, Any] = {}

        if self._request_metadata:
            self._apply_request_metadata(kwargs, override)
        if self._team_callbacks:
            self._apply_team_callbacks(kwargs, override)
        if self._cache_point:
            self._apply_cache_point(kwargs, messages, override)

        return override or None

    # ------------------------------------------------------------------
    # Lever: requestMetadata (RouteIQ-294a)
    # ------------------------------------------------------------------

    def _apply_request_metadata(
        self, kwargs: dict[str, Any], override: dict[str, Any]
    ) -> None:
        """Forward tenant identity as Bedrock requestMetadata tags.

        Lands at the TOP-LEVEL ``requestMetadata`` kwarg (a Bedrock
        supported_param read from non_default_params), NOT the logging metadata.
        A caller-supplied ``requestMetadata`` is never clobbered.
        """
        if kwargs.get("requestMetadata"):
            return  # respect an explicit caller-supplied value

        ctx = self._governance_ctx(kwargs)
        tags: dict[str, str] = {}
        workspace_id = ctx.get("workspace_id")
        key_id = ctx.get("key_id")
        if workspace_id:
            tags[self._tag_key("workspace")] = self._tag_value(str(workspace_id))
        if key_id:
            tags[self._tag_key("key")] = self._tag_value(str(key_id))

        if tags:
            override["requestMetadata"] = tags

    def _tag_key(self, name: str) -> str:
        return self._sanitize(f"{self._metadata_prefix}{name}")[:256]

    @staticmethod
    def _tag_value(value: str) -> str:
        return BedrockRequestLeversPlugin._sanitize(value)[:256]

    @staticmethod
    def _sanitize(value: str) -> str:
        return _BEDROCK_METADATA_SANITIZE.sub("_", value)

    # ------------------------------------------------------------------
    # Lever: per-team callback override (RouteIQ-9cd8)
    # ------------------------------------------------------------------

    def _apply_team_callbacks(
        self, kwargs: dict[str, Any], override: dict[str, Any]
    ) -> None:
        """Set per-team logging sinks as TOP-LEVEL request kwargs.

        litellm sources dynamic per-request callbacks from the top-level
        ``callbacks`` / ``success_callback`` / ``failure_callback`` kwargs
        (utils.py:791/903/923/927), NOT from metadata. The global registration
        is left intact; we only ADD the team's sinks for this request.
        """
        team_id = self._team_id(kwargs)
        if not team_id:
            return
        team_cfg = self._team_callback_map.get(team_id)
        if not team_cfg:
            return

        for slot in ("callbacks", "success_callback", "failure_callback"):
            sinks = team_cfg.get(slot)
            if not sinks:
                continue
            # Merge with any caller/existing value (preserve order, dedupe).
            existing = kwargs.get(slot)
            merged: list[Any] = list(existing) if isinstance(existing, list) else []
            for sink in sinks:
                if sink not in merged:
                    merged.append(sink)
            override[slot] = merged

    # ------------------------------------------------------------------
    # Lever: Bedrock prompt caching cachePoint (RouteIQ-b9ee)
    # ------------------------------------------------------------------

    def _apply_cache_point(
        self,
        kwargs: dict[str, Any],
        messages: list[Any],
        override: dict[str, Any],
    ) -> None:
        """Drive cachePoint via litellm cache_control_injection_points.

        Builds a list of injection points and sets it as the top-level
        ``cache_control_injection_points`` completion kwarg. litellm wires the
        anthropic_cache_control_hook for the ``message`` (system prefix) point;
        the Bedrock converse transform appends a tool_config cachePoint for the
        ``tool_config`` point when the request carries tools. A caller-supplied
        value is never clobbered.
        """
        if kwargs.get("cache_control_injection_points"):
            return

        points: list[dict[str, Any]] = []
        if self._cache_system and self._has_role(messages, "system"):
            points.append(
                {
                    "location": "message",
                    "role": "system",
                    "control": {"type": "ephemeral"},
                }
            )
        # tool_config cachePoint only does anything when tools are present; the
        # converse transform guards on len(bedrock_tools) > 0 anyway, but skip
        # cleanly when there are no tools to keep the override empty/byte-stable.
        if self._cache_tools and kwargs.get("tools"):
            points.append({"location": "tool_config"})

        if points:
            override["cache_control_injection_points"] = points

    @staticmethod
    def _has_role(messages: list[Any], role: str) -> bool:
        for msg in messages or []:
            if isinstance(msg, dict) and msg.get("role") == role:
                return True
        return False

    # ------------------------------------------------------------------
    # Governance context resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _governance_ctx(kwargs: dict[str, Any]) -> dict[str, Any]:
        """Read the RouteIQ governance stamp from request metadata.

        The routing strategy stamps ``metadata["_governance_ctx"]`` =
        {workspace_id, key_id, effective_profile}. Checks both
        ``litellm_params.metadata`` and top-level ``metadata`` (the same dual
        lookup the callback bridge uses for workspace resolution).
        """
        for container in (kwargs.get("litellm_params"), kwargs):
            if not isinstance(container, dict):
                continue
            metadata = container.get("metadata")
            if isinstance(metadata, dict):
                ctx = metadata.get("_governance_ctx")
                if isinstance(ctx, dict):
                    return ctx
        return {}

    @classmethod
    def _team_id(cls, kwargs: dict[str, Any]) -> str | None:
        """Resolve the team id for the request.

        Prefers an explicit ``metadata["team_id"]`` (litellm's standard team
        field), then falls back to the governance workspace_id so an operator
        can key the callback map on workspace.
        """
        for container in (kwargs.get("litellm_params"), kwargs):
            if not isinstance(container, dict):
                continue
            metadata = container.get("metadata")
            if isinstance(metadata, dict):
                team_id = metadata.get("team_id")
                if isinstance(team_id, str) and team_id:
                    return team_id
        ctx = cls._governance_ctx(kwargs)
        workspace_id = ctx.get("workspace_id")
        return workspace_id if isinstance(workspace_id, str) and workspace_id else None

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def health_check(self) -> dict[str, Any]:
        any_on = self._request_metadata or self._team_callbacks or self._cache_point
        return {
            "status": "ok" if any_on else "disabled",
            "request_metadata": self._request_metadata,
            "team_callbacks": self._team_callbacks,
            "cache_point": self._cache_point,
        }
