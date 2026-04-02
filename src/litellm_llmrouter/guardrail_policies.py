"""Guardrail Policy Engine for RouteIQ.

Provides config-driven guardrail management with:
- Named, versioned guardrail definitions (CRUD API)
- Input/output guardrail hooks (pre-request and post-response)
- Workspace and org-level enforcement
- Actions: deny (block), log (allow + log), alert (allow + notify)
- Integration with RouteIQ's existing plugin guardrails
- Custom status codes: 446 (guardrail denied), 246 (guardrail warning)

Architecture:
  Guardrails are defined as policy objects stored in memory/Redis.
  Each policy specifies: check type, parameters, action on failure,
  and scope (workspace/org/global).

  The engine evaluates input_guardrails before routing and
  output_guardrails after response, integrating with the existing
  plugin callback bridge.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("litellm_llmrouter.guardrail_policies")

# Custom HTTP status codes
HTTP_446_GUARDRAIL_DENIED = 446
HTTP_246_GUARDRAIL_WARNING = 246


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class GuardrailPhase(str, Enum):
    INPUT = "input"  # Pre-request
    OUTPUT = "output"  # Post-response


class GuardrailAction(str, Enum):
    DENY = "deny"  # Block request (446)
    LOG = "log"  # Allow but log violation (246)
    ALERT = "alert"  # Allow + trigger alert


class GuardrailType(str, Enum):
    # Deterministic checks (no LLM cost)
    REGEX_MATCH = "regex_match"
    REGEX_DENY = "regex_deny"
    WORD_COUNT_MIN = "word_count_min"
    WORD_COUNT_MAX = "word_count_max"
    JSON_SCHEMA = "json_schema"
    MODEL_ALLOWLIST = "model_allowlist"
    REQUIRED_METADATA = "required_metadata"
    CONTAINS_CODE = "contains_code"
    MAX_TOKENS = "max_tokens"

    # Plugin-based checks (delegates to RouteIQ plugins)
    PII_DETECTION = "pii_detection"
    CONTENT_FILTER = "content_filter"
    PROMPT_INJECTION = "prompt_injection"

    # External checks
    WEBHOOK = "webhook"  # Call external URL for validation

    # Custom (user-defined function)
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# Policy model
# ---------------------------------------------------------------------------


class GuardrailPolicy(BaseModel):
    """A guardrail policy definition."""

    guardrail_id: str
    name: str
    description: str = ""
    enabled: bool = True
    version: int = 1

    # When to run
    phase: GuardrailPhase = GuardrailPhase.INPUT

    # What to check
    check_type: GuardrailType
    parameters: Dict[str, Any] = Field(default_factory=dict)
    # e.g., for REGEX_DENY: {"patterns": ["(?i)ignore.*instructions"]}
    # e.g., for MODEL_ALLOWLIST: {"models": ["gpt-4o*", "@anthropic/*"]}
    # e.g., for WEBHOOK: {"url": "https://guard.example.com/check", "timeout": 5}

    # What to do on failure
    action: GuardrailAction = GuardrailAction.DENY

    # Scope
    workspace_id: Optional[str] = None  # None = global
    org_id: Optional[str] = None

    # Execution
    async_execution: bool = False  # Run alongside LLM call (non-blocking)
    priority: int = 100  # Lower = runs first

    # Metadata
    created_at: Optional[float] = None
    updated_at: Optional[float] = None


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass
class GuardrailResult:
    """Result of evaluating a guardrail."""

    guardrail_id: str
    guardrail_name: str
    passed: bool
    action: GuardrailAction
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class GuardrailPolicyEngine:
    """Evaluates guardrail policies against requests and responses."""

    def __init__(self) -> None:
        self._policies: Dict[str, GuardrailPolicy] = {}
        self._check_handlers: Dict[
            GuardrailType,
            Callable[..., Awaitable[GuardrailResult]],
        ] = {}
        self._custom_handlers: Dict[
            str,
            Callable[..., Awaitable[GuardrailResult]],
        ] = {}
        self._register_builtin_checks()

    def _register_builtin_checks(self) -> None:
        """Register built-in guardrail check handlers."""
        self._check_handlers[GuardrailType.REGEX_MATCH] = self._check_regex_match
        self._check_handlers[GuardrailType.REGEX_DENY] = self._check_regex_deny
        self._check_handlers[GuardrailType.WORD_COUNT_MIN] = self._check_word_count_min
        self._check_handlers[GuardrailType.WORD_COUNT_MAX] = self._check_word_count_max
        self._check_handlers[GuardrailType.MODEL_ALLOWLIST] = (
            self._check_model_allowlist
        )
        self._check_handlers[GuardrailType.REQUIRED_METADATA] = (
            self._check_required_metadata
        )
        self._check_handlers[GuardrailType.MAX_TOKENS] = self._check_max_tokens
        self._check_handlers[GuardrailType.JSON_SCHEMA] = self._check_json_schema
        self._check_handlers[GuardrailType.CONTAINS_CODE] = self._check_contains_code
        self._check_handlers[GuardrailType.WEBHOOK] = self._check_webhook
        self._check_handlers[GuardrailType.PII_DETECTION] = self._check_pii_detection
        self._check_handlers[GuardrailType.CONTENT_FILTER] = self._check_content_filter
        self._check_handlers[GuardrailType.PROMPT_INJECTION] = (
            self._check_prompt_injection
        )
        self._check_handlers[GuardrailType.CUSTOM] = self._check_custom

    # =====================================================================
    # CRUD
    # =====================================================================

    def add_policy(self, policy: GuardrailPolicy) -> None:
        """Add or update a guardrail policy."""
        now = time.time()
        if policy.guardrail_id in self._policies:
            policy.updated_at = now
            if policy.created_at is None:
                policy.created_at = self._policies[policy.guardrail_id].created_at
        else:
            if policy.created_at is None:
                policy.created_at = now
            policy.updated_at = now
        self._policies[policy.guardrail_id] = policy
        logger.info(
            "Guardrail policy '%s' (%s) registered: phase=%s check=%s action=%s",
            policy.name,
            policy.guardrail_id,
            policy.phase.value,
            policy.check_type.value,
            policy.action.value,
        )

    def remove_policy(self, guardrail_id: str) -> bool:
        """Remove a guardrail policy. Returns True if it existed."""
        removed = self._policies.pop(guardrail_id, None)
        if removed:
            logger.info("Guardrail policy '%s' removed", guardrail_id)
        return removed is not None

    def get_policy(self, guardrail_id: str) -> Optional[GuardrailPolicy]:
        """Get a guardrail policy by ID."""
        return self._policies.get(guardrail_id)

    def list_policies(
        self,
        phase: Optional[GuardrailPhase] = None,
        workspace_id: Optional[str] = None,
    ) -> List[GuardrailPolicy]:
        """List guardrail policies, optionally filtered by phase and workspace."""
        policies = list(self._policies.values())
        if phase is not None:
            policies = [p for p in policies if p.phase == phase]
        if workspace_id is not None:
            policies = [
                p
                for p in policies
                if p.workspace_id is None or p.workspace_id == workspace_id
            ]
        return sorted(policies, key=lambda p: p.priority)

    def register_custom_handler(
        self,
        handler_id: str,
        handler: Callable[..., Awaitable[GuardrailResult]],
    ) -> None:
        """Register a custom check handler for CUSTOM type guardrails."""
        self._custom_handlers[handler_id] = handler
        logger.info("Custom guardrail handler '%s' registered", handler_id)

    # =====================================================================
    # Evaluation
    # =====================================================================

    async def evaluate_input(
        self,
        request_data: Dict[str, Any],
        workspace_id: Optional[str] = None,
    ) -> List[GuardrailResult]:
        """Evaluate all input guardrails against the request.

        Args:
            request_data: The incoming request dict. Expected keys:
                - messages: list of message dicts
                - model: model name string
                - metadata: optional metadata dict
                - max_tokens: optional int
            workspace_id: Scope to this workspace (+ global policies).

        Returns:
            List of GuardrailResult for all evaluated policies.
        """
        policies = self._matching_policies(GuardrailPhase.INPUT, workspace_id)
        content = self._extract_content_from_messages(request_data.get("messages", []))
        return await self._evaluate_policies(policies, content, request_data)

    async def evaluate_output(
        self,
        response_data: Dict[str, Any],
        workspace_id: Optional[str] = None,
    ) -> List[GuardrailResult]:
        """Evaluate all output guardrails against the response.

        Args:
            response_data: The outgoing response dict. Expected keys:
                - content: response text
                - model: model name string
                - metadata: optional metadata dict
            workspace_id: Scope to this workspace (+ global policies).

        Returns:
            List of GuardrailResult for all evaluated policies.
        """
        policies = self._matching_policies(GuardrailPhase.OUTPUT, workspace_id)
        content = response_data.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        return await self._evaluate_policies(policies, content, response_data)

    def has_deny_result(self, results: List[GuardrailResult]) -> bool:
        """Check if any result has a DENY action and failed."""
        return any(not r.passed and r.action == GuardrailAction.DENY for r in results)

    def get_deny_results(self, results: List[GuardrailResult]) -> List[GuardrailResult]:
        """Get all results that are DENY failures."""
        return [r for r in results if not r.passed and r.action == GuardrailAction.DENY]

    def get_warning_results(
        self, results: List[GuardrailResult]
    ) -> List[GuardrailResult]:
        """Get all results that are LOG/ALERT failures (warnings)."""
        return [
            r
            for r in results
            if not r.passed and r.action in (GuardrailAction.LOG, GuardrailAction.ALERT)
        ]

    # =====================================================================
    # Internal: matching + evaluation pipeline
    # =====================================================================

    def _matching_policies(
        self,
        phase: GuardrailPhase,
        workspace_id: Optional[str],
    ) -> List[GuardrailPolicy]:
        """Get enabled policies matching phase and scope, sorted by priority."""
        matching = []
        for p in self._policies.values():
            if not p.enabled:
                continue
            if p.phase != phase:
                continue
            # Scope filtering: global policies always apply;
            # workspace-scoped policies only apply if workspace matches
            if p.workspace_id is not None and p.workspace_id != workspace_id:
                continue
            matching.append(p)
        return sorted(matching, key=lambda p: p.priority)

    async def _evaluate_policies(
        self,
        policies: List[GuardrailPolicy],
        content: str,
        data: Dict[str, Any],
    ) -> List[GuardrailResult]:
        """Run each policy's check handler and collect results."""
        results: List[GuardrailResult] = []
        for policy in policies:
            handler = self._check_handlers.get(policy.check_type)
            if handler is None:
                logger.warning(
                    "No handler for guardrail type '%s' (policy '%s'), skipping",
                    policy.check_type.value,
                    policy.guardrail_id,
                )
                continue

            start = time.perf_counter()
            try:
                result = await handler(policy, content, data)
                result.latency_ms = (time.perf_counter() - start) * 1000
            except Exception as exc:
                elapsed = (time.perf_counter() - start) * 1000
                logger.error(
                    "Guardrail '%s' (%s) raised exception: %s",
                    policy.guardrail_id,
                    policy.check_type.value,
                    exc,
                    exc_info=True,
                )
                result = GuardrailResult(
                    guardrail_id=policy.guardrail_id,
                    guardrail_name=policy.name,
                    passed=True,  # fail-open on handler error
                    action=policy.action,
                    message=f"Handler error: {exc}",
                    details={"error": str(exc)},
                    latency_ms=elapsed,
                )

            results.append(result)

            # Log non-pass results
            if not result.passed:
                level = (
                    logging.WARNING
                    if result.action == GuardrailAction.DENY
                    else logging.INFO
                )
                logger.log(
                    level,
                    "Guardrail '%s' FAILED: action=%s message=%s latency=%.1fms",
                    result.guardrail_name,
                    result.action.value,
                    result.message,
                    result.latency_ms,
                )

        return results

    # =====================================================================
    # Content extraction helpers
    # =====================================================================

    @staticmethod
    def _extract_content_from_messages(
        messages: List[Dict[str, Any]],
    ) -> str:
        """Concatenate text content from all messages (focus on user role)."""
        parts: List[str] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                parts.append(content)
            elif isinstance(content, list):
                # Multi-part messages (vision, etc.)
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(part.get("text", ""))
        return " ".join(parts)

    # =====================================================================
    # Built-in check implementations
    # =====================================================================

    async def _check_regex_deny(
        self,
        policy: GuardrailPolicy,
        content: str,
        data: Dict[str, Any],
    ) -> GuardrailResult:
        """Deny if content matches any of the specified regex patterns.

        Parameters:
            patterns: list of regex pattern strings
        """
        patterns = policy.parameters.get("patterns", [])
        if not patterns:
            return GuardrailResult(
                guardrail_id=policy.guardrail_id,
                guardrail_name=policy.name,
                passed=True,
                action=policy.action,
                message="No patterns configured",
            )

        matched: List[str] = []
        for pat_str in patterns:
            try:
                if re.search(pat_str, content):
                    matched.append(pat_str)
            except re.error as exc:
                logger.warning(
                    "Invalid regex in guardrail '%s': %s (%s)",
                    policy.guardrail_id,
                    pat_str,
                    exc,
                )

        if matched:
            return GuardrailResult(
                guardrail_id=policy.guardrail_id,
                guardrail_name=policy.name,
                passed=False,
                action=policy.action,
                message=f"Content matched denied pattern(s): {matched}",
                details={"matched_patterns": matched, "pattern_count": len(matched)},
            )

        return GuardrailResult(
            guardrail_id=policy.guardrail_id,
            guardrail_name=policy.name,
            passed=True,
            action=policy.action,
        )

    async def _check_regex_match(
        self,
        policy: GuardrailPolicy,
        content: str,
        data: Dict[str, Any],
    ) -> GuardrailResult:
        """Deny if content does NOT match at least one required pattern.

        Parameters:
            patterns: list of regex pattern strings (at least one must match)
        """
        patterns = policy.parameters.get("patterns", [])
        if not patterns:
            return GuardrailResult(
                guardrail_id=policy.guardrail_id,
                guardrail_name=policy.name,
                passed=True,
                action=policy.action,
                message="No patterns configured",
            )

        for pat_str in patterns:
            try:
                if re.search(pat_str, content):
                    return GuardrailResult(
                        guardrail_id=policy.guardrail_id,
                        guardrail_name=policy.name,
                        passed=True,
                        action=policy.action,
                        details={"matched_pattern": pat_str},
                    )
            except re.error as exc:
                logger.warning(
                    "Invalid regex in guardrail '%s': %s (%s)",
                    policy.guardrail_id,
                    pat_str,
                    exc,
                )

        return GuardrailResult(
            guardrail_id=policy.guardrail_id,
            guardrail_name=policy.name,
            passed=False,
            action=policy.action,
            message="Content did not match any required pattern",
            details={"required_patterns": patterns},
        )

    async def _check_word_count_min(
        self,
        policy: GuardrailPolicy,
        content: str,
        data: Dict[str, Any],
    ) -> GuardrailResult:
        """Deny if content has fewer words than the minimum.

        Parameters:
            min_words: int
        """
        min_words = policy.parameters.get("min_words", 1)
        word_count = len(content.split()) if content.strip() else 0
        passed = word_count >= min_words

        return GuardrailResult(
            guardrail_id=policy.guardrail_id,
            guardrail_name=policy.name,
            passed=passed,
            action=policy.action,
            message="" if passed else f"Word count {word_count} < minimum {min_words}",
            details={"word_count": word_count, "min_words": min_words},
        )

    async def _check_word_count_max(
        self,
        policy: GuardrailPolicy,
        content: str,
        data: Dict[str, Any],
    ) -> GuardrailResult:
        """Deny if content has more words than the maximum.

        Parameters:
            max_words: int
        """
        max_words = policy.parameters.get("max_words", 10000)
        word_count = len(content.split()) if content.strip() else 0
        passed = word_count <= max_words

        return GuardrailResult(
            guardrail_id=policy.guardrail_id,
            guardrail_name=policy.name,
            passed=passed,
            action=policy.action,
            message="" if passed else f"Word count {word_count} > maximum {max_words}",
            details={"word_count": word_count, "max_words": max_words},
        )

    async def _check_json_schema(
        self,
        policy: GuardrailPolicy,
        content: str,
        data: Dict[str, Any],
    ) -> GuardrailResult:
        """Deny if content is not valid JSON or doesn't match schema keys.

        Parameters:
            required_keys: list of top-level keys that must be present
            must_be_json: bool (default True) - whether content must parse as JSON
        """
        must_be_json = policy.parameters.get("must_be_json", True)
        required_keys = policy.parameters.get("required_keys", [])

        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            if must_be_json:
                return GuardrailResult(
                    guardrail_id=policy.guardrail_id,
                    guardrail_name=policy.name,
                    passed=False,
                    action=policy.action,
                    message="Content is not valid JSON",
                    details={"must_be_json": True},
                )
            return GuardrailResult(
                guardrail_id=policy.guardrail_id,
                guardrail_name=policy.name,
                passed=True,
                action=policy.action,
                message="Content is not JSON but must_be_json is False",
            )

        # Check required keys
        if required_keys and isinstance(parsed, dict):
            missing = [k for k in required_keys if k not in parsed]
            if missing:
                return GuardrailResult(
                    guardrail_id=policy.guardrail_id,
                    guardrail_name=policy.name,
                    passed=False,
                    action=policy.action,
                    message=f"Missing required JSON keys: {missing}",
                    details={"missing_keys": missing, "required_keys": required_keys},
                )

        return GuardrailResult(
            guardrail_id=policy.guardrail_id,
            guardrail_name=policy.name,
            passed=True,
            action=policy.action,
        )

    async def _check_model_allowlist(
        self,
        policy: GuardrailPolicy,
        content: str,
        data: Dict[str, Any],
    ) -> GuardrailResult:
        """Deny if the requested model is not in the allowlist.

        Parameters:
            models: list of allowed model patterns (supports fnmatch globs)
                    e.g., ["gpt-4o*", "claude-*", "anthropic/*"]
        """
        allowed_models = policy.parameters.get("models", [])
        if not allowed_models:
            return GuardrailResult(
                guardrail_id=policy.guardrail_id,
                guardrail_name=policy.name,
                passed=True,
                action=policy.action,
                message="No model allowlist configured",
            )

        requested_model = data.get("model", "")
        if not requested_model:
            return GuardrailResult(
                guardrail_id=policy.guardrail_id,
                guardrail_name=policy.name,
                passed=False,
                action=policy.action,
                message="No model specified in request",
                details={"allowed_models": allowed_models},
            )

        for pattern in allowed_models:
            if fnmatch.fnmatch(requested_model, pattern):
                return GuardrailResult(
                    guardrail_id=policy.guardrail_id,
                    guardrail_name=policy.name,
                    passed=True,
                    action=policy.action,
                    details={
                        "model": requested_model,
                        "matched_pattern": pattern,
                    },
                )

        return GuardrailResult(
            guardrail_id=policy.guardrail_id,
            guardrail_name=policy.name,
            passed=False,
            action=policy.action,
            message=f"Model '{requested_model}' not in allowlist",
            details={
                "model": requested_model,
                "allowed_models": allowed_models,
            },
        )

    async def _check_required_metadata(
        self,
        policy: GuardrailPolicy,
        content: str,
        data: Dict[str, Any],
    ) -> GuardrailResult:
        """Deny if required metadata keys are missing from the request.

        Parameters:
            required_keys: list of metadata key names that must be present
        """
        required_keys = policy.parameters.get("required_keys", [])
        if not required_keys:
            return GuardrailResult(
                guardrail_id=policy.guardrail_id,
                guardrail_name=policy.name,
                passed=True,
                action=policy.action,
                message="No required metadata keys configured",
            )

        metadata = data.get("metadata", {}) or {}
        missing = [k for k in required_keys if k not in metadata]

        if missing:
            return GuardrailResult(
                guardrail_id=policy.guardrail_id,
                guardrail_name=policy.name,
                passed=False,
                action=policy.action,
                message=f"Missing required metadata keys: {missing}",
                details={
                    "missing_keys": missing,
                    "required_keys": required_keys,
                    "present_keys": list(metadata.keys()),
                },
            )

        return GuardrailResult(
            guardrail_id=policy.guardrail_id,
            guardrail_name=policy.name,
            passed=True,
            action=policy.action,
        )

    async def _check_contains_code(
        self,
        policy: GuardrailPolicy,
        content: str,
        data: Dict[str, Any],
    ) -> GuardrailResult:
        """Deny if content contains code-like patterns.

        Parameters:
            deny_code: bool (default True) - if True, deny when code is found
            languages: list of language hints (optional, for logging)
        """
        deny_code = policy.parameters.get("deny_code", True)

        # Heuristic code detection patterns
        code_patterns = [
            r"```[\w]*\n",  # Markdown code fences
            r"(?:def|class|function|const|let|var|import|from)\s+\w+",  # Declarations
            r"(?:if|for|while|try|catch|except)\s*[\(\{:]",  # Control flow
            r"(?:=>|->)\s*\{",  # Arrow functions
            r"^\s*(?:#include|#import|#define)\b",  # C/C++ preprocessor
            r"(?:SELECT|INSERT|UPDATE|DELETE|CREATE)\s+(?:FROM|INTO|TABLE)",  # SQL
        ]

        found_patterns: List[str] = []
        for pat in code_patterns:
            try:
                if re.search(pat, content, re.MULTILINE | re.IGNORECASE):
                    found_patterns.append(pat)
            except re.error:
                pass

        contains_code = len(found_patterns) > 0

        if deny_code and contains_code:
            return GuardrailResult(
                guardrail_id=policy.guardrail_id,
                guardrail_name=policy.name,
                passed=False,
                action=policy.action,
                message="Content contains code-like patterns",
                details={
                    "patterns_found": len(found_patterns),
                    "deny_code": deny_code,
                },
            )

        if not deny_code and not contains_code:
            # Require code but none found
            return GuardrailResult(
                guardrail_id=policy.guardrail_id,
                guardrail_name=policy.name,
                passed=False,
                action=policy.action,
                message="Content does not contain expected code patterns",
                details={"deny_code": deny_code},
            )

        return GuardrailResult(
            guardrail_id=policy.guardrail_id,
            guardrail_name=policy.name,
            passed=True,
            action=policy.action,
        )

    async def _check_max_tokens(
        self,
        policy: GuardrailPolicy,
        content: str,
        data: Dict[str, Any],
    ) -> GuardrailResult:
        """Deny if requested max_tokens exceeds the limit.

        Parameters:
            max_tokens_limit: int - maximum allowed max_tokens value
        """
        limit = policy.parameters.get("max_tokens_limit", 4096)
        requested = data.get("max_tokens")

        if requested is None:
            return GuardrailResult(
                guardrail_id=policy.guardrail_id,
                guardrail_name=policy.name,
                passed=True,
                action=policy.action,
                message="No max_tokens in request",
            )

        try:
            requested_int = int(requested)
        except (TypeError, ValueError):
            return GuardrailResult(
                guardrail_id=policy.guardrail_id,
                guardrail_name=policy.name,
                passed=False,
                action=policy.action,
                message=f"Invalid max_tokens value: {requested}",
                details={"max_tokens": requested},
            )

        passed = requested_int <= limit
        return GuardrailResult(
            guardrail_id=policy.guardrail_id,
            guardrail_name=policy.name,
            passed=passed,
            action=policy.action,
            message=(
                "" if passed else f"max_tokens {requested_int} exceeds limit {limit}"
            ),
            details={
                "requested_max_tokens": requested_int,
                "max_tokens_limit": limit,
            },
        )

    async def _check_webhook(
        self,
        policy: GuardrailPolicy,
        content: str,
        data: Dict[str, Any],
    ) -> GuardrailResult:
        """Call an external webhook for validation.

        Parameters:
            url: str - webhook URL
            timeout: int (default 5) - timeout in seconds
            method: str (default "POST") - HTTP method
            headers: dict (optional) - additional headers
        """
        url = policy.parameters.get("url", "")
        timeout = policy.parameters.get("timeout", 5)
        method = policy.parameters.get("method", "POST").upper()
        extra_headers = policy.parameters.get("headers", {})

        if not url:
            return GuardrailResult(
                guardrail_id=policy.guardrail_id,
                guardrail_name=policy.name,
                passed=True,
                action=policy.action,
                message="No webhook URL configured",
            )

        # SSRF protection
        try:
            from litellm_llmrouter.url_security import validate_outbound_url

            validate_outbound_url(url)
        except Exception as exc:
            return GuardrailResult(
                guardrail_id=policy.guardrail_id,
                guardrail_name=policy.name,
                passed=True,  # fail-open on security check
                action=policy.action,
                message=f"Webhook URL blocked by SSRF protection: {exc}",
                details={"url": url, "error": str(exc)},
            )

        try:
            from litellm_llmrouter.http_client_pool import get_http_client

            client = get_http_client()
            payload = {
                "guardrail_id": policy.guardrail_id,
                "content": content[:10000],  # Truncate for safety
                "phase": policy.phase.value,
                "metadata": data.get("metadata", {}),
                "model": data.get("model", ""),
            }

            headers = {"Content-Type": "application/json"}
            headers.update(extra_headers)

            if method == "POST":
                resp = await client.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=timeout,
                )
            else:
                resp = await client.get(
                    url,
                    params={"guardrail_id": policy.guardrail_id},
                    headers=headers,
                    timeout=timeout,
                )

            if resp.status_code == 200:
                body = (
                    resp.json()
                    if resp.headers.get("content-type", "").startswith(
                        "application/json"
                    )
                    else {}
                )
                return GuardrailResult(
                    guardrail_id=policy.guardrail_id,
                    guardrail_name=policy.name,
                    passed=body.get("passed", True),
                    action=policy.action,
                    message=body.get("message", ""),
                    details=body.get("details", {}),
                )
            else:
                return GuardrailResult(
                    guardrail_id=policy.guardrail_id,
                    guardrail_name=policy.name,
                    passed=True,  # fail-open
                    action=policy.action,
                    message=f"Webhook returned status {resp.status_code}",
                    details={"status_code": resp.status_code},
                )

        except ImportError:
            return GuardrailResult(
                guardrail_id=policy.guardrail_id,
                guardrail_name=policy.name,
                passed=True,
                action=policy.action,
                message="HTTP client pool not available",
            )
        except Exception as exc:
            return GuardrailResult(
                guardrail_id=policy.guardrail_id,
                guardrail_name=policy.name,
                passed=True,  # fail-open
                action=policy.action,
                message=f"Webhook call failed: {exc}",
                details={"error": str(exc)},
            )

    # =====================================================================
    # Plugin-delegated checks
    # =====================================================================

    async def _check_pii_detection(
        self,
        policy: GuardrailPolicy,
        content: str,
        data: Dict[str, Any],
    ) -> GuardrailResult:
        """Delegate PII detection to the PIIGuard plugin scanner.

        Parameters:
            entity_types: list of PII types (optional, default all)
        """
        try:
            from litellm_llmrouter.gateway.plugins.pii_guard import (
                PIIGuard,
                ALL_ENTITY_TYPES,
            )

            guard = PIIGuard()
            # Override entity types if specified in policy
            requested_types = policy.parameters.get("entity_types")
            if requested_types:
                valid = {t.upper() for t in requested_types} & ALL_ENTITY_TYPES
                if valid:
                    guard._entity_types = valid

            findings = guard.scan_pii(content)
            if findings:
                entity_types = sorted({f.entity_type for f in findings})
                return GuardrailResult(
                    guardrail_id=policy.guardrail_id,
                    guardrail_name=policy.name,
                    passed=False,
                    action=policy.action,
                    message=f"PII detected: {entity_types}",
                    details={
                        "entity_types": entity_types,
                        "count": len(findings),
                    },
                )

            return GuardrailResult(
                guardrail_id=policy.guardrail_id,
                guardrail_name=policy.name,
                passed=True,
                action=policy.action,
            )
        except ImportError:
            return GuardrailResult(
                guardrail_id=policy.guardrail_id,
                guardrail_name=policy.name,
                passed=True,
                action=policy.action,
                message="PIIGuard plugin not available",
            )

    async def _check_content_filter(
        self,
        policy: GuardrailPolicy,
        content: str,
        data: Dict[str, Any],
    ) -> GuardrailResult:
        """Delegate content filtering to the ContentFilterPlugin scorer.

        Parameters:
            categories: list of categories to check (default all)
            threshold: float score threshold (default 0.7)
        """
        try:
            from litellm_llmrouter.gateway.plugins.content_filter import (
                ContentFilterPlugin,
                ALL_CATEGORIES,
                CategoryConfig,
                _DEFAULT_KEYWORDS,
                _DEFAULT_PATTERNS,
            )

            plugin = ContentFilterPlugin()
            threshold = policy.parameters.get("threshold", 0.7)
            requested_cats = policy.parameters.get("categories")
            active_cats = (
                set(requested_cats) & ALL_CATEGORIES
                if requested_cats
                else ALL_CATEGORIES
            )

            # Build category configs
            import re as _re

            for cat in active_cats:
                keywords = _DEFAULT_KEYWORDS.get(cat, [])
                raw_patterns = _DEFAULT_PATTERNS.get(cat, [])
                compiled = []
                for p in raw_patterns:
                    try:
                        compiled.append(_re.compile(p))
                    except _re.error:
                        pass
                plugin._categories[cat] = CategoryConfig(
                    keywords=keywords,
                    patterns=compiled,
                    threshold=threshold,
                    action="block",
                )

            violations: List[str] = []
            max_score = 0.0
            for cat in active_cats:
                score_result = plugin._score_content(content, cat)
                if score_result.score >= threshold:
                    violations.append(f"{cat}={score_result.score:.2f}")
                    max_score = max(max_score, score_result.score)

            if violations:
                return GuardrailResult(
                    guardrail_id=policy.guardrail_id,
                    guardrail_name=policy.name,
                    passed=False,
                    action=policy.action,
                    message=f"Content violations: {violations}",
                    details={
                        "violations": violations,
                        "max_score": max_score,
                        "threshold": threshold,
                    },
                )

            return GuardrailResult(
                guardrail_id=policy.guardrail_id,
                guardrail_name=policy.name,
                passed=True,
                action=policy.action,
            )
        except ImportError:
            return GuardrailResult(
                guardrail_id=policy.guardrail_id,
                guardrail_name=policy.name,
                passed=True,
                action=policy.action,
                message="ContentFilterPlugin not available",
            )

    async def _check_prompt_injection(
        self,
        policy: GuardrailPolicy,
        content: str,
        data: Dict[str, Any],
    ) -> GuardrailResult:
        """Delegate prompt injection detection to PromptInjectionGuard scanner.

        Parameters:
            extra_patterns: list of additional regex patterns (optional)
        """
        try:
            from litellm_llmrouter.gateway.plugins.prompt_injection_guard import (
                PromptInjectionGuard,
            )

            guard = PromptInjectionGuard()
            score, label = guard.scan(content)

            if score > 0.0:
                return GuardrailResult(
                    guardrail_id=policy.guardrail_id,
                    guardrail_name=policy.name,
                    passed=False,
                    action=policy.action,
                    message=f"Prompt injection detected: {label}",
                    details={
                        "pattern_matched": label,
                        "score": score,
                    },
                )

            # Check extra patterns if provided
            extra = policy.parameters.get("extra_patterns", [])
            for pat_str in extra:
                try:
                    if re.search(pat_str, content, re.IGNORECASE):
                        return GuardrailResult(
                            guardrail_id=policy.guardrail_id,
                            guardrail_name=policy.name,
                            passed=False,
                            action=policy.action,
                            message=f"Prompt injection (custom pattern): {pat_str}",
                            details={
                                "pattern_matched": pat_str,
                                "source": "extra_patterns",
                            },
                        )
                except re.error:
                    pass

            return GuardrailResult(
                guardrail_id=policy.guardrail_id,
                guardrail_name=policy.name,
                passed=True,
                action=policy.action,
            )
        except ImportError:
            return GuardrailResult(
                guardrail_id=policy.guardrail_id,
                guardrail_name=policy.name,
                passed=True,
                action=policy.action,
                message="PromptInjectionGuard plugin not available",
            )

    async def _check_custom(
        self,
        policy: GuardrailPolicy,
        content: str,
        data: Dict[str, Any],
    ) -> GuardrailResult:
        """Invoke a user-registered custom handler.

        Parameters:
            handler_id: str - ID of the registered custom handler
        """
        handler_id = policy.parameters.get("handler_id", "")
        handler = self._custom_handlers.get(handler_id)
        if handler is None:
            return GuardrailResult(
                guardrail_id=policy.guardrail_id,
                guardrail_name=policy.name,
                passed=True,
                action=policy.action,
                message=f"Custom handler '{handler_id}' not registered",
            )

        return await handler(policy, content, data)

    # =====================================================================
    # Status
    # =====================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get engine status summary."""
        policies = list(self._policies.values())
        enabled = [p for p in policies if p.enabled]
        return {
            "total_policies": len(policies),
            "enabled_policies": len(enabled),
            "input_policies": len(
                [p for p in enabled if p.phase == GuardrailPhase.INPUT]
            ),
            "output_policies": len(
                [p for p in enabled if p.phase == GuardrailPhase.OUTPUT]
            ),
            "check_types": sorted({p.check_type.value for p in enabled}),
            "custom_handlers": sorted(self._custom_handlers.keys()),
        }


# ---------------------------------------------------------------------------
# File-Based Persistence
# ---------------------------------------------------------------------------

_GUARDRAIL_POLICIES_STATE_PATH = os.getenv("ROUTEIQ_GUARDRAIL_POLICIES_STATE_PATH", "")


def save_guardrail_policies_state(engine: GuardrailPolicyEngine) -> None:
    """Save guardrail policy definitions to file for persistence across restarts."""
    if not _GUARDRAIL_POLICIES_STATE_PATH:
        return
    try:
        import json as _json

        state = {gid: p.model_dump() for gid, p in engine._policies.items()}
        with open(_GUARDRAIL_POLICIES_STATE_PATH, "w") as f:
            _json.dump(state, f, indent=2, default=str)
        logger.debug(
            "Guardrail policies state saved to %s", _GUARDRAIL_POLICIES_STATE_PATH
        )
    except Exception as exc:
        logger.warning("Failed to save guardrail policies state: %s", exc)


def load_guardrail_policies_state(engine: GuardrailPolicyEngine) -> int:
    """Load guardrail policy definitions from file. Returns count loaded."""
    if not _GUARDRAIL_POLICIES_STATE_PATH or not os.path.exists(
        _GUARDRAIL_POLICIES_STATE_PATH
    ):
        return 0
    try:
        import json as _json

        with open(_GUARDRAIL_POLICIES_STATE_PATH) as f:
            state = _json.load(f)
        count = 0
        for policy_data in state.values():
            engine.add_policy(GuardrailPolicy(**policy_data))
            count += 1
        logger.info(
            "Loaded %d guardrail policies from %s",
            count,
            _GUARDRAIL_POLICIES_STATE_PATH,
        )
        return count
    except Exception as exc:
        logger.warning("Failed to load guardrail policies state: %s", exc)
        return 0


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_engine: Optional[GuardrailPolicyEngine] = None


def get_guardrail_policy_engine() -> GuardrailPolicyEngine:
    """Get or create the global guardrail policy engine singleton."""
    global _engine
    if _engine is None:
        _engine = GuardrailPolicyEngine()
    return _engine


def reset_guardrail_policy_engine() -> None:
    """Reset the global guardrail policy engine (for testing)."""
    global _engine
    _engine = None
