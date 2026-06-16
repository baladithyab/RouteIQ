"""
Bedrock Guardrails Plugin
=========================

Example plugin demonstrating integration with AWS Bedrock Guardrails.
Calls bedrock-runtime:ApplyGuardrail API to evaluate content against
a configured guardrail.

Guardrails run as input filters (``on_llm_pre_call``) blocking or
anonymising content before it reaches the model.

Optionally, the plugin can also *natively activate* the guardrail on the
outbound Bedrock model request: when ``BEDROCK_GUARDRAIL_NATIVE_ACTIVATION``
is enabled, ``on_llm_pre_call`` injects a ``guardrailConfig`` block (carrying
``guardrailIdentifier`` + ``guardrailVersion``) into the request kwargs so the
Bedrock ``Converse`` / ``InvokeModel`` arm evaluates the guardrail inline. This
is distinct from (and composable with) the ``ApplyGuardrail`` input filter.

Configuration (environment variables):
    BEDROCK_GUARDRAIL_ID:                  The guardrail identifier (required)
    BEDROCK_GUARDRAIL_VERSION:             Guardrail version (default: "DRAFT")
    BEDROCK_GUARDRAIL_ENABLED:             Enable/disable (default: false)
    BEDROCK_GUARDRAIL_NATIVE_ACTIVATION:   Inject guardrailConfig into the
                                           outbound Bedrock request (default:
                                           false). When false the request is
                                           left byte-for-byte unchanged.
    BEDROCK_GUARDRAIL_TRACE:               Guardrail trace mode for the native
                                           arm, "enabled" or "disabled"
                                           (default: "disabled").
    AWS_REGION:                            AWS region for Bedrock (default:
                                           us-east-1)

Hook points:
    on_llm_pre_call  -> evaluate last user message, block if BLOCKED; and,
                        when native activation is on, attach guardrailConfig
                        to the outbound request kwargs.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from litellm_llmrouter.gateway.plugin_manager import (
    GatewayPlugin,
    PluginCapability,
    PluginContext,
    PluginMetadata,
)
from litellm_llmrouter.gateway.plugins.guardrails_base import (
    GuardrailBlockError,
    record_guardrail_check_metric,
)

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

__all__ = ["BedrockGuardrailsPlugin"]


class BedrockGuardrailsPlugin(GatewayPlugin):
    """Plugin that evaluates content using AWS Bedrock Guardrails.

    Uses the ``ApplyGuardrail`` API to check content against a configured
    guardrail.  Supports ``BLOCKED`` and ``ANONYMIZED`` actions from
    Bedrock.
    """

    def __init__(self) -> None:
        self._guardrail_id: str = ""
        self._guardrail_version: str = "DRAFT"
        self._enabled: bool = False
        self._region: str = "us-east-1"
        self._client: Any = None  # boto3 bedrock-runtime client
        # Native activation: inject guardrailConfig into the outbound Bedrock
        # request. Default-OFF -> outbound request kwargs are left untouched.
        self._native_activation: bool = False
        self._trace: str = "disabled"

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="bedrock-guardrails",
            version="1.0.0",
            capabilities={PluginCapability.GUARDRAIL},
            priority=50,
            description="AWS Bedrock Guardrails content evaluation",
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def startup(
        self, app: "FastAPI", context: PluginContext | None = None
    ) -> None:
        """Read configuration from env vars and optionally validate."""
        self._enabled = (
            os.getenv("BEDROCK_GUARDRAIL_ENABLED", "false").lower() == "true"
        )
        if not self._enabled:
            logger.info("Bedrock Guardrails plugin disabled")
            return

        self._guardrail_id = os.getenv("BEDROCK_GUARDRAIL_ID", "")
        self._guardrail_version = os.getenv("BEDROCK_GUARDRAIL_VERSION", "DRAFT")
        self._region = os.getenv("AWS_REGION", "us-east-1")
        self._native_activation = (
            os.getenv("BEDROCK_GUARDRAIL_NATIVE_ACTIVATION", "false").lower() == "true"
        )
        trace = os.getenv("BEDROCK_GUARDRAIL_TRACE", "disabled").lower()
        self._trace = "enabled" if trace == "enabled" else "disabled"

        if not self._guardrail_id:
            logger.warning(
                "BEDROCK_GUARDRAIL_ID not set, disabling Bedrock Guardrails plugin"
            )
            self._enabled = False
            return

        logger.info(
            "Bedrock Guardrails plugin enabled "
            f"(guardrail={self._guardrail_id}, version={self._guardrail_version}, "
            f"native_activation={self._native_activation})"
        )

    async def shutdown(
        self, app: "FastAPI", context: PluginContext | None = None
    ) -> None:
        """Release boto3 client."""
        self._client = None

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    async def evaluate(self, content: str, source: str = "INPUT") -> dict[str, Any]:
        """Evaluate *content* against the configured Bedrock guardrail.

        Args:
            content: Text content to evaluate.
            source:  ``"INPUT"`` for user input, ``"OUTPUT"`` for model output.

        Returns:
            Dict with ``action`` (``NONE`` / ``BLOCKED`` / ``ANONYMIZED``)
            and additional details from the Bedrock response.
        """
        if not self._enabled:
            return {"action": "NONE", "reason": "plugin_disabled"}

        try:
            import boto3  # noqa: F811 - optional dependency
        except ImportError:
            logger.warning("boto3 not installed, cannot use Bedrock Guardrails")
            return {"action": "NONE", "reason": "boto3_not_installed"}

        try:
            if self._client is None:
                self._client = boto3.client("bedrock-runtime", region_name=self._region)

            response = self._client.apply_guardrail(
                guardrailIdentifier=self._guardrail_id,
                guardrailVersion=self._guardrail_version,
                source=source,
                content=[{"text": {"text": content}}],
            )

            action = response.get("action", "NONE")

            return {
                "action": action,
                "outputs": response.get("outputs", []),
                "assessments": response.get("assessments", []),
                "guardrail_id": self._guardrail_id,
            }

        except Exception as e:
            logger.error(f"Bedrock Guardrail evaluation failed: {e}")
            # Fail-open: allow the request through on errors
            return {"action": "NONE", "reason": f"error: {e}"}

    # ------------------------------------------------------------------
    # LLM lifecycle hooks
    # ------------------------------------------------------------------

    async def on_llm_pre_call(
        self, model: str, messages: list[Any], kwargs: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Evaluate the last user message before the LLM call.

        Raises ``GuardrailBlockError`` when the Bedrock guardrail returns
        ``BLOCKED``.

        When native activation is enabled, returns a ``{"guardrailConfig": ...}``
        override so the Bedrock arm evaluates the guardrail inline on the
        outbound model request. When native activation is off (default), the
        outbound request is left untouched.
        """
        if not self._enabled:
            return None

        content = self._extract_last_user_text(messages)
        if content:
            result = await self.evaluate(content, source="INPUT")

            # Telemetry: BLOCKED -> deny, ANONYMIZED -> redact, NONE -> pass.
            action = {"BLOCKED": "deny", "ANONYMIZED": "redact"}.get(
                result["action"], "pass"
            )
            record_guardrail_check_metric("bedrock", action)

            if result["action"] == "BLOCKED":
                raise GuardrailBlockError(
                    guardrail_name="bedrock-guardrails",
                    category="bedrock",
                    message="Content blocked by Bedrock Guardrail",
                    score=1.0,
                )

        # Native activation: attach guardrailConfig to the outbound request so
        # the Bedrock model arm applies the guardrail inline.
        return self._native_guardrail_override(kwargs)

    def _native_guardrail_override(
        self, kwargs: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Return a ``guardrailConfig`` override for the outbound request.

        Returns ``None`` (no change) unless native activation is enabled, a
        guardrail id is configured, and the request does not already carry a
        ``guardrailConfig`` (a caller-supplied config is never clobbered).
        """
        if not self._native_activation or not self._guardrail_id:
            return None
        if kwargs.get("guardrailConfig"):
            # Respect an explicit caller-supplied guardrailConfig.
            return None
        return {
            "guardrailConfig": {
                "guardrailIdentifier": self._guardrail_id,
                "guardrailVersion": self._guardrail_version,
                "trace": self._trace,
            }
        }

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def health_check(self) -> dict[str, Any]:
        return {
            "status": "ok" if self._enabled else "disabled",
            "enabled": self._enabled,
            "guardrail_id": self._guardrail_id,
            "guardrail_version": self._guardrail_version,
            "native_activation": self._native_activation,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_last_user_text(messages: list[Any]) -> str:
        """Return the text content of the last ``user`` message."""
        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
        return ""
