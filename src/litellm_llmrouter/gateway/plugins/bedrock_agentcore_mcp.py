"""
Bedrock AgentCore MCP Connector Plugin
=======================================

Reference plugin that demonstrates the v0.2.0 Universal PluginContext by
connecting Amazon Bedrock AgentCore tool-use capabilities to the MCP gateway.

This plugin:
- Uses ``context.mcp`` to register/unregister MCP servers
- Uses ``context.metrics`` to emit custom counters
- Uses ``context.validate_outbound_url`` for SSRF protection
- Implements ``on_config_reload`` to refresh server list
- Implements ``health_check`` to report AgentCore connectivity
- Implements ``on_management_operation`` to observe key/model changes

Configuration via environment variables:
- ROUTEIQ_PLUGIN_BEDROCK_AGENTCORE_ENABLED: Enable this plugin (default: false)
- ROUTEIQ_PLUGIN_BEDROCK_AGENTCORE_REGION: AWS region (default: resolved from env)
- ROUTEIQ_PLUGIN_BEDROCK_AGENTCORE_AGENT_IDS: Comma-separated AgentCore Runtime
  identifiers — either bare runtime ids or full ``agentRuntimeArn`` values.
- ROUTEIQ_PLUGIN_BEDROCK_AGENTCORE_QUALIFIER: Runtime endpoint qualifier
  (default: ``DEFAULT``) — replaces the legacy hardcoded ``TSTALIASID``.

AgentCore retargeting (RouteIQ-e5a4):
- The legacy connector targeted the DEPRECATED ``bedrock-agent-runtime`` data
  plane (``/agents/{id}/agentAliases/TSTALIASID/sessions/test/text``) with a
  hardcoded test alias and NO SigV4 signing, registered (contradictorily) under
  the ``streamable_http`` transport.
- It now targets Amazon Bedrock AgentCore Runtime's ``InvokeAgentRuntime`` API
  on the ``bedrock-agentcore`` data-plane service. Invocation is SigV4-signed
  by a ``boto3.client("bedrock-agentcore")`` (botocore signs every request), so
  no manual credential handling is required.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, TYPE_CHECKING

from ..plugin_manager import (
    GatewayPlugin,
    PluginCapability,
    PluginContext,
    PluginMetadata,
)

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

# AgentCore Runtime data-plane service name (boto3/botocore). Distinct from the
# control-plane ``bedrock-agentcore-control`` service (see the IAM prefix split
# in agentcore-gotchas). InvokeAgentRuntime lives on the data plane.
AGENTCORE_RUNTIME_SERVICE = "bedrock-agentcore"


def _resolve_region(configured: str | None) -> str | None:
    """Resolve the AWS region for AgentCore calls.

    Mirrors the region-resolution discipline used elsewhere (bedrock_discovery /
    database): explicit config first, then ``AWS_REGION`` / ``AWS_DEFAULT_REGION``,
    then the boto3 session default. Returns None when none can be resolved (the
    caller then skips registration rather than signing ``Region=None``).
    """
    if configured and configured.strip():
        return configured.strip()
    for env_key in ("AWS_REGION", "AWS_DEFAULT_REGION"):
        val = os.getenv(env_key)
        if val and val.strip():
            return val.strip()
    try:
        import boto3

        return boto3.session.Session().region_name or None
    except Exception:
        return None


def _agentcore_runtime_endpoint(region: str) -> str:
    """Return the AgentCore Runtime data-plane endpoint for a region.

    Used both as the registered server URL (for discovery) and as the boto3
    ``endpoint_url``. Shape: ``https://bedrock-agentcore.{region}.amazonaws.com``.
    """
    return f"https://{AGENTCORE_RUNTIME_SERVICE}.{region}.amazonaws.com"


def _agent_runtime_arn(agent_id: str, region: str) -> str:
    """Normalize an agent identifier into a full ``agentRuntimeArn``.

    Accepts either a full ARN (passed through) or a bare runtime id (wrapped).
    The account id is resolved at invoke time by boto3 from the caller's
    credentials, so registration uses a best-effort ARN with an empty account
    segment when only a bare id is supplied; invocation always passes the
    operator-supplied identifier through to boto3 unchanged.
    """
    if agent_id.startswith("arn:"):
        return agent_id
    return f"arn:aws:{AGENTCORE_RUNTIME_SERVICE}:{region}::runtime/{agent_id}"


def _make_agentcore_client(region: str, endpoint_url: str | None = None) -> Any:
    """Construct a SigV4-signing ``boto3.client('bedrock-agentcore')``.

    boto3/botocore attaches SigV4 to every request automatically from the
    standard credential chain — no manual signing required. Built inline (no
    module-level cache => no reset obligation), following the
    ``bedrock_discovery._bedrock_control_client`` pattern.
    """
    import boto3

    return boto3.session.Session().client(
        AGENTCORE_RUNTIME_SERVICE,
        region_name=region,
        endpoint_url=endpoint_url,
    )


class BedrockAgentCoreMCPPlugin(GatewayPlugin):
    """
    Exposes Amazon Bedrock AgentCore agents as MCP tool servers.

    Each configured AgentCore agent is registered as an MCP server,
    making its tools discoverable and invokable through the MCP gateway.
    """

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="bedrock-agentcore-mcp",
            version="1.0.0rc1",
            capabilities={
                PluginCapability.TOOL_RUNTIME,
                PluginCapability.OBSERVABILITY_EXPORTER,
            },
            priority=500,
            description="Connects Bedrock AgentCore agents as MCP tool servers",
        )

    def __init__(self, client_factory: Any = None) -> None:
        # Region resolution discipline: explicit config -> AWS_REGION ->
        # boto3 session default. May be None until startup (we resolve again
        # lazily so a late-set AWS_REGION still works).
        self._region = _resolve_region(
            os.getenv("ROUTEIQ_PLUGIN_BEDROCK_AGENTCORE_REGION")
        )
        # Runtime endpoint qualifier replaces the legacy hardcoded TSTALIASID.
        self._qualifier = os.getenv(
            "ROUTEIQ_PLUGIN_BEDROCK_AGENTCORE_QUALIFIER", "DEFAULT"
        )
        self._agent_ids: list[str] = []
        raw_ids = os.getenv("ROUTEIQ_PLUGIN_BEDROCK_AGENTCORE_AGENT_IDS", "")
        if raw_ids.strip():
            self._agent_ids = [a.strip() for a in raw_ids.split(",") if a.strip()]

        # Injectable boto3 client factory: ``region -> client``. Tests pass a
        # MagicMock factory so no live AWS / real boto3 is needed.
        self._client_factory = client_factory or _make_agentcore_client

        self._registered_server_ids: list[str] = []
        self._context: PluginContext | None = None
        self._tools_registered_counter: Any = None
        self._invocations_counter: Any = None
        self._last_health_check: float = 0.0
        self._healthy = True

    async def startup(
        self, app: "FastAPI", context: PluginContext | None = None
    ) -> None:
        """Register AgentCore agents as MCP servers."""
        if context is None:
            logger.warning("BedrockAgentCoreMCPPlugin: No context provided, skipping")
            return

        self._context = context

        # Set up metrics
        if context.metrics:
            self._tools_registered_counter = context.metrics.create_counter(
                "routeiq.plugin.bedrock_agentcore.tools_registered",
                unit="{tool}",
                description="Number of AgentCore tools registered as MCP servers",
            )
            self._invocations_counter = context.metrics.create_counter(
                "routeiq.plugin.bedrock_agentcore.invocations",
                unit="{invocation}",
                description="Number of AgentCore tool invocations",
            )

        # Register agents as MCP servers
        await self._register_agents(context)

        logger.info(
            f"BedrockAgentCoreMCPPlugin started: "
            f"region={self._region}, "
            f"agents={len(self._agent_ids)}, "
            f"servers_registered={len(self._registered_server_ids)}"
        )

    async def shutdown(
        self, app: "FastAPI", context: PluginContext | None = None
    ) -> None:
        """Unregister all MCP servers."""
        ctx = context or self._context
        if ctx and ctx.mcp:
            for server_id in self._registered_server_ids:
                try:
                    await ctx.mcp.unregister_server(server_id)
                except Exception as e:
                    logger.warning(f"Failed to unregister MCP server {server_id}: {e}")

        self._registered_server_ids.clear()
        logger.info("BedrockAgentCoreMCPPlugin shut down")

    async def health_check(self) -> dict[str, Any]:
        """Report plugin health based on AgentCore connectivity."""
        now = time.monotonic()

        # Cache health check for 30 seconds
        if now - self._last_health_check < 30:
            return {
                "status": "ok" if self._healthy else "degraded",
                "region": self._region,
                "agents_configured": len(self._agent_ids),
                "servers_registered": len(self._registered_server_ids),
            }

        self._last_health_check = now

        # Basic connectivity check
        try:
            # We don't actually call AgentCore here (too expensive for health checks).
            # Instead, verify our MCP servers are still registered.
            if self._context and self._context.mcp:
                servers = await self._context.mcp.list_servers()
                registered = {s.get("server_id") for s in servers}
                missing = [
                    sid for sid in self._registered_server_ids if sid not in registered
                ]
                if missing:
                    self._healthy = False
                    return {
                        "status": "degraded",
                        "reason": f"MCP servers missing: {missing}",
                        "region": self._region,
                    }
            self._healthy = True
        except Exception as e:
            self._healthy = False
            return {
                "status": "degraded",
                "reason": str(e),
                "region": self._region,
            }

        return {
            "status": "ok",
            "region": self._region,
            "agents_configured": len(self._agent_ids),
            "servers_registered": len(self._registered_server_ids),
        }

    async def on_config_reload(
        self, old_config: dict[str, Any], new_config: dict[str, Any]
    ) -> None:
        """Re-read agent IDs from config and re-register MCP servers."""
        # Check if agent IDs changed in the new config
        new_agent_ids = new_config.get("bedrock_agentcore_agent_ids", [])
        if isinstance(new_agent_ids, str):
            new_agent_ids = [a.strip() for a in new_agent_ids.split(",") if a.strip()]

        if not new_agent_ids:
            return  # No change — keep using env var config

        if set(new_agent_ids) != set(self._agent_ids):
            logger.info(
                f"BedrockAgentCoreMCPPlugin: Agent IDs changed, re-registering. "
                f"Old: {self._agent_ids}, New: {new_agent_ids}"
            )
            self._agent_ids = new_agent_ids

            if self._context:
                # Unregister old servers
                if self._context.mcp:
                    for server_id in self._registered_server_ids:
                        try:
                            await self._context.mcp.unregister_server(server_id)
                        except Exception:
                            pass
                self._registered_server_ids.clear()

                # Register new servers
                await self._register_agents(self._context)

    async def on_management_operation(
        self,
        operation: str,
        resource_type: str,
        method: str,
        path: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Observe management operations for metrics and logging."""
        # Track key generation events (example of observing management ops)
        if operation == "key.generate":
            logger.debug(
                f"BedrockAgentCoreMCPPlugin: Key generated "
                f"(status={metadata.get('status_code') if metadata else 'unknown'})"
            )

    # =========================================================================
    # Internal helpers
    # =========================================================================

    async def _register_agents(self, context: PluginContext) -> None:
        """Register configured AgentCore Runtime agents as MCP servers."""
        if not context.mcp:
            logger.debug("MCP accessor not available, skipping agent registration")
            return

        if not context.mcp.is_enabled():
            logger.debug("MCP gateway disabled, skipping agent registration")
            return

        # Re-resolve the region lazily so a late-set AWS_REGION is honoured.
        region = self._region or _resolve_region(None)
        if not region:
            logger.warning(
                "BedrockAgentCoreMCPPlugin: no AWS region resolved; "
                "set ROUTEIQ_PLUGIN_BEDROCK_AGENTCORE_REGION or AWS_REGION. "
                "Skipping AgentCore registration."
            )
            return
        self._region = region

        # AgentCore Runtime data-plane endpoint (NOT the deprecated
        # bedrock-agent-runtime alias path). This is the SigV4 target.
        endpoint = _agentcore_runtime_endpoint(region)

        for agent_id in self._agent_ids:
            server_id = f"bedrock-agentcore-{agent_id}"
            runtime_arn = _agent_runtime_arn(agent_id, region)

            # Validate the endpoint against SSRF if available (registration-time,
            # no DNS). The endpoint, not a fabricated alias path, is what gets
            # called — so that is what we validate.
            if context.validate_outbound_url:
                try:
                    context.validate_outbound_url(endpoint)
                except Exception as e:
                    logger.warning(f"SSRF validation failed for AgentCore URL: {e}")
                    continue

            try:
                await context.mcp.register_server(
                    server_id=server_id,
                    name=f"Bedrock AgentCore ({agent_id})",
                    url=endpoint,
                    # AgentCore Runtime speaks MCP; register under the SSE/MCP
                    # transport, not the legacy contradictory streamable_http
                    # pointed at a REST alias path.
                    transport="sse",
                    metadata={
                        "provider": "bedrock-agentcore",
                        "region": region,
                        "agent_id": agent_id,
                        "agent_runtime_arn": runtime_arn,
                        "qualifier": self._qualifier,
                        "service": AGENTCORE_RUNTIME_SERVICE,
                        "signed": "sigv4",
                    },
                )
                self._registered_server_ids.append(server_id)

                # Emit metric
                if self._tools_registered_counter:
                    self._tools_registered_counter.add(
                        1,
                        {"agent_id": agent_id, "region": region},
                    )

                logger.info(f"Registered AgentCore agent as MCP server: {server_id}")

            except Exception as e:
                logger.warning(
                    f"Failed to register AgentCore agent {agent_id} as MCP server: {e}"
                )

    async def invoke_agent(
        self,
        agent_id: str,
        payload: dict[str, Any],
        *,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Invoke an AgentCore Runtime agent via the SigV4-signed data plane.

        Uses ``boto3.client('bedrock-agentcore').invoke_agent_runtime`` — botocore
        SigV4-signs the request automatically from the standard credential chain.
        This replaces the legacy unsigned POST to the deprecated
        ``bedrock-agent-runtime`` alias path.

        Args:
            agent_id: A configured agent id or full ``agentRuntimeArn``.
            payload: JSON-serializable request body for the runtime.
            session_id: Optional runtime session id for multi-turn continuity.

        Returns:
            The decoded response payload from the runtime.

        Raises:
            ValueError: when no region can be resolved.
        """
        region = self._region or _resolve_region(None)
        if not region:
            raise ValueError(
                "No AWS region resolved for AgentCore invocation; "
                "set ROUTEIQ_PLUGIN_BEDROCK_AGENTCORE_REGION or AWS_REGION."
            )

        endpoint = _agentcore_runtime_endpoint(region)
        runtime_arn = _agent_runtime_arn(agent_id, region)

        # Build a fresh SigV4-signing client (no module-level cache).
        client = self._client_factory(region, endpoint)

        kwargs: dict[str, Any] = {
            "agentRuntimeArn": runtime_arn,
            "qualifier": self._qualifier,
            "payload": json.dumps(payload).encode("utf-8"),
        }
        if session_id:
            kwargs["runtimeSessionId"] = session_id

        response = client.invoke_agent_runtime(**kwargs)

        if self._invocations_counter:
            self._invocations_counter.add(1, {"agent_id": agent_id, "region": region})

        return self._decode_invoke_response(response)

    @staticmethod
    def _decode_invoke_response(response: dict[str, Any]) -> dict[str, Any]:
        """Decode the InvokeAgentRuntime response body into a dict.

        The runtime returns a streaming/bytes ``response`` body carrying JSON.
        Tolerates botocore StreamingBody (``.read()``), raw bytes, str, or an
        already-decoded dict.
        """
        body = response.get("response") if isinstance(response, dict) else None
        if body is None:
            return response if isinstance(response, dict) else {}

        raw: Any = body
        if hasattr(body, "read"):
            raw = body.read()
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="replace")
        if isinstance(raw, str):
            try:
                decoded: Any = json.loads(raw)
            except json.JSONDecodeError:
                return {"output": raw}
            if isinstance(decoded, dict):
                return decoded
            return {"output": decoded}
        if isinstance(raw, dict):
            return raw
        return {"output": str(raw)}
