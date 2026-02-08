"""
LlamaGuard Safety Plugin
=========================

Example plugin demonstrating integration with a self-hosted LlamaGuard model.
Sends chat messages to a LlamaGuard endpoint for safety classification.

LlamaGuard returns a single-token classification:

    safe       -> request allowed
    unsafe O1  -> BLOCKED (with category code)

The plugin translates this into a ``GuardrailBlockError`` when content is
classified as unsafe.

Configuration (environment variables):
    LLAMAGUARD_ENDPOINT:  URL of the LlamaGuard inference endpoint (required)
    LLAMAGUARD_ENABLED:   Enable/disable (default: false)
    LLAMAGUARD_MODEL:     Model name (default: "meta-llama/LlamaGuard-7b")
    LLAMAGUARD_TIMEOUT:   Request timeout in seconds (default: 10)

Hook points:
    on_llm_pre_call  -> classify last user message, block if unsafe
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
)

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

__all__ = ["LlamaGuardPlugin"]


class LlamaGuardPlugin(GatewayPlugin):
    """Plugin that classifies content using a self-hosted LlamaGuard model.

    Calls an HTTP endpoint (e.g. vLLM, TGI, or SageMaker) hosting a
    LlamaGuard model and interprets the ``safe`` / ``unsafe`` output.
    """

    def __init__(self) -> None:
        self._endpoint: str = ""
        self._model: str = "meta-llama/LlamaGuard-7b"
        self._enabled: bool = False
        self._timeout: float = 10.0
        self._http_client: Any = None  # httpx.AsyncClient or None

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="llamaguard",
            version="1.0.0",
            capabilities={PluginCapability.GUARDRAIL},
            priority=55,
            description="LlamaGuard safety classification guardrail",
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def startup(
        self, app: "FastAPI", context: PluginContext | None = None
    ) -> None:
        """Read configuration and optionally acquire shared HTTP client."""
        self._enabled = os.getenv("LLAMAGUARD_ENABLED", "false").lower() == "true"
        if not self._enabled:
            logger.info("LlamaGuard plugin disabled")
            return

        self._endpoint = os.getenv("LLAMAGUARD_ENDPOINT", "")
        self._model = os.getenv("LLAMAGUARD_MODEL", "meta-llama/LlamaGuard-7b")
        self._timeout = float(os.getenv("LLAMAGUARD_TIMEOUT", "10"))

        if not self._endpoint:
            logger.warning("LLAMAGUARD_ENDPOINT not set, disabling LlamaGuard plugin")
            self._enabled = False
            return

        # Try to use the shared HTTP client pool; fall back to a local client
        try:
            from litellm_llmrouter.http_client_pool import get_http_client

            self._http_client = get_http_client()
        except Exception:
            logger.debug("Shared HTTP client pool unavailable, will create per-call")

        logger.info(
            f"LlamaGuard plugin enabled (endpoint={self._endpoint}, "
            f"model={self._model})"
        )

    async def shutdown(
        self, app: "FastAPI", context: PluginContext | None = None
    ) -> None:
        """Release resources (shared client lifecycle is managed elsewhere)."""
        self._http_client = None

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    async def evaluate(self, content: str) -> dict[str, Any]:
        """Classify *content* using the LlamaGuard endpoint.

        Returns:
            Dict with ``safe`` (bool), ``raw`` (response text), and
            optionally ``category`` (e.g. ``"O1"``).
        """
        if not self._enabled:
            return {"safe": True, "raw": "", "reason": "plugin_disabled"}

        try:
            import httpx
        except ImportError:  # pragma: no cover
            logger.warning("httpx not installed, cannot use LlamaGuard plugin")
            return {"safe": True, "raw": "", "reason": "httpx_not_installed"}

        payload = self._build_payload(content)

        try:
            client = self._http_client
            if client is None:
                client = httpx.AsyncClient()

            response = await client.post(
                self._endpoint,
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
            body = response.json()

            return self._parse_response(body)

        except Exception as e:
            logger.error(f"LlamaGuard evaluation failed: {e}")
            # Fail-open: allow the request through on errors
            return {"safe": True, "raw": "", "reason": f"error: {e}"}

    # ------------------------------------------------------------------
    # LLM lifecycle hooks
    # ------------------------------------------------------------------

    async def on_llm_pre_call(
        self, model: str, messages: list[Any], kwargs: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Classify the last user message before the LLM call.

        Raises ``GuardrailBlockError`` when LlamaGuard classifies the
        content as unsafe.
        """
        if not self._enabled:
            return None

        content = self._extract_last_user_text(messages)
        if not content:
            return None

        result = await self.evaluate(content)

        if not result.get("safe", True):
            category = result.get("category", "unknown")
            raise GuardrailBlockError(
                guardrail_name="llamaguard",
                category=f"llamaguard:{category}",
                message=f"Content classified as unsafe by LlamaGuard ({category})",
                score=1.0,
            )

        return None

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def health_check(self) -> dict[str, Any]:
        return {
            "status": "ok" if self._enabled else "disabled",
            "enabled": self._enabled,
            "endpoint": self._endpoint,
            "model": self._model,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_payload(self, content: str) -> dict[str, Any]:
        """Build the JSON payload expected by a LlamaGuard serving endpoint.

        Compatible with vLLM / TGI ``/v1/completions``-style API.
        """
        # LlamaGuard expects a conversation formatted with special tokens.
        # We use the simple single-turn format:
        #   [INST] Task: ... <conversation> ... [/INST]
        prompt = (
            "[INST] Task: Check if there is unsafe content in "
            "'User' messages in conversations according to our safety policy.\n\n"
            "<BEGIN CONVERSATION>\n\n"
            f"User: {content}\n\n"
            "<END CONVERSATION>\n\n"
            "Provide your safety assessment for User in the above conversation:\n"
            "- First line must read 'safe' or 'unsafe'.\n"
            "- If unsafe, a second line must include a comma-separated list "
            "of violated categories. [/INST]"
        )
        return {
            "model": self._model,
            "prompt": prompt,
            "max_tokens": 32,
            "temperature": 0.0,
        }

    @staticmethod
    def _parse_response(body: dict[str, Any]) -> dict[str, Any]:
        """Parse the LlamaGuard model output.

        Expected response (vLLM ``/v1/completions``):
            { "choices": [{ "text": " safe" }] }
        or:
            { "choices": [{ "text": " unsafe\\nO1" }] }
        """
        text = ""
        choices = body.get("choices", [])
        if choices:
            text = choices[0].get("text", "").strip()

        if not text:
            # Fallback: some endpoints return "generated_text" directly
            text = body.get("generated_text", "").strip()

        lines = text.split("\n")
        classification = lines[0].strip().lower()
        category = lines[1].strip() if len(lines) > 1 else ""

        is_safe = classification == "safe"
        return {
            "safe": is_safe,
            "raw": text,
            "category": category if not is_safe else "",
        }

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
