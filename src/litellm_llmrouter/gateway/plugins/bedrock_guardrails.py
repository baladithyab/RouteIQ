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
    BEDROCK_GUARDRAIL_REGION:              Region the guardrail lives in, when
                                           it differs from AWS_REGION
                                           (cross-region; default: unset ->
                                           AWS_REGION).
    BEDROCK_GUARDRAIL_ROLE_ARN:            IAM role to STS-assume so the
                                           guardrail can live in a DIFFERENT
                                           account (cross-account; default:
                                           unset -> same-account client).

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
        # Cross-account / cross-region per-arm guardrail (RouteIQ-3024). When a
        # guardrail lives in a different account/region than the gateway, the
        # ApplyGuardrail client must target that account+region. Default-OFF:
        # both unset -> single-region same-account client (byte-stable).
        self._guardrail_account_role_arn: str = ""
        self._guardrail_region: str = ""
        # Cache of (region, role_arn) -> bedrock-runtime client so a per-arm
        # client is built once.
        self._clients_by_arm: dict[tuple[str, str], Any] = {}

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
        # Cross-account / cross-region per-arm guardrail (RouteIQ-3024).
        # BEDROCK_GUARDRAIL_REGION overrides the ApplyGuardrail call region (the
        # guardrail can live in a different region than the gateway);
        # BEDROCK_GUARDRAIL_ROLE_ARN, when set, makes the plugin assume that
        # role (STS) so the guardrail can live in a different ACCOUNT.
        self._guardrail_region = os.getenv("BEDROCK_GUARDRAIL_REGION", "").strip()
        self._guardrail_account_role_arn = os.getenv(
            "BEDROCK_GUARDRAIL_ROLE_ARN", ""
        ).strip()
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
        """Release boto3 client(s)."""
        self._client = None
        self._clients_by_arm.clear()

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
            client = self._resolve_guardrail_client(boto3)

            response = client.apply_guardrail(
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
    # Cross-account / cross-region client resolution (RouteIQ-3024)
    # ------------------------------------------------------------------

    def _resolve_guardrail_client(self, boto3: Any) -> Any:
        """Resolve the bedrock-runtime client for the guardrail arm.

        Default path (no per-arm overrides): same-account client in
        ``AWS_REGION`` -- identical to the historical single client (and cached
        on ``self._client`` for backward compatibility).

        Cross-region (``BEDROCK_GUARDRAIL_REGION``): the ApplyGuardrail call is
        made against the guardrail's region rather than the gateway region.

        Cross-account (``BEDROCK_GUARDRAIL_ROLE_ARN``): the plugin assumes the
        given role via STS and builds the client from the temporary
        credentials, so the guardrail can live in another account.

        Per-arm clients are cached by ``(region, role_arn)`` so the assume-role
        round-trip happens once.
        """
        region = self._guardrail_region or self._region
        role_arn = self._guardrail_account_role_arn

        # Fast path: no overrides -> the original cached single client.
        if not self._guardrail_region and not role_arn:
            if self._client is None:
                self._client = boto3.client("bedrock-runtime", region_name=region)
            return self._client

        cache_key = (region, role_arn)
        cached = self._clients_by_arm.get(cache_key)
        if cached is not None:
            return cached

        if role_arn:
            creds = self._assume_role(boto3, role_arn, region)
            client = boto3.client(
                "bedrock-runtime",
                region_name=region,
                aws_access_key_id=creds["AccessKeyId"],
                aws_secret_access_key=creds["SecretAccessKey"],
                aws_session_token=creds["SessionToken"],
            )
        else:
            client = boto3.client("bedrock-runtime", region_name=region)

        self._clients_by_arm[cache_key] = client
        return client

    @staticmethod
    def _assume_role(boto3: Any, role_arn: str, region: str) -> dict[str, Any]:
        """Assume ``role_arn`` via STS and return its temporary credentials."""
        sts = boto3.client("sts", region_name=region)
        resp = sts.assume_role(
            RoleArn=role_arn,
            RoleSessionName="routeiq-bedrock-guardrail",
        )
        return resp["Credentials"]

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
            "guardrail_region": self._guardrail_region or self._region,
            "cross_account": bool(self._guardrail_account_role_arn),
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
