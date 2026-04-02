"""
Content/Toxicity Filter Plugin
===============================

Keyword and pattern-based content scoring across configurable categories.

Categories:
- violence: violent language, threats, weapons
- hate_speech: slurs, discriminatory language
- sexual: explicit sexual content
- self_harm: self-harm, suicide references
- illegal_activity: drug manufacturing, hacking instructions, etc.

Each category has:
- Configurable keyword/pattern lists
- Score threshold (0.0-1.0)
- Action: block, warn, log (configurable per category)

Hook points:
- on_llm_pre_call: Input filtering (scan user messages, block if threshold exceeded)
- on_llm_success: Output filtering (scan response, log warning if threshold exceeded)

Configuration via environment variables:
- CONTENT_FILTER_ENABLED: Enable/disable the filter (default: false)
- CONTENT_FILTER_CATEGORIES: Comma-separated list of active categories
- CONTENT_FILTER_THRESHOLD: Default score threshold 0.0-1.0 (default: 0.7)
- CONTENT_FILTER_ACTION: Default action block|warn|log (default: block)
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
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

# Re-export so existing imports (tests, etc.) keep working
__all__ = ["ContentFilterPlugin", "GuardrailBlockError", "CategoryConfig"]


@dataclass
class CategoryConfig:
    """Configuration for a single content category."""

    keywords: list[str] = field(default_factory=list)
    patterns: list[re.Pattern[str]] = field(default_factory=list)
    threshold: float = 0.7
    action: str = "block"  # block | warn | log


@dataclass
class ContentScore:
    """Score result for a content scan."""

    category: str
    score: float
    matched_keywords: list[str] = field(default_factory=list)
    matched_patterns: list[str] = field(default_factory=list)


# Default keyword and pattern definitions per category
_DEFAULT_KEYWORDS: dict[str, list[str]] = {
    "violence": [
        "kill",
        "murder",
        "attack",
        "assault",
        "bomb",
        "shoot",
        "stab",
        "weapon",
        "gun",
        "knife",
        "explode",
        "execute",
        "torture",
        "massacre",
        "slaughter",
        "decapitate",
        "dismember",
        "bloodbath",
        "carnage",
        "maim",
    ],
    "hate_speech": [
        "inferior race",
        "ethnic cleansing",
        "white supremacy",
        "racial purity",
        "subhuman",
        "genocide",
        "hate all",
        "exterminate",
        "master race",
        "racial superiority",
    ],
    "sexual": [
        "explicit sexual",
        "pornographic",
        "sexually explicit",
        "nude images",
        "sexual act",
        "sexual content",
        "erotic content",
        "adult content",
        "sexual fantasy",
        "sexual exploitation",
    ],
    "self_harm": [
        "kill myself",
        "suicide method",
        "how to end my life",
        "self-harm",
        "cut myself",
        "overdose method",
        "suicidal",
        "end it all",
        "ways to die",
        "hang myself",
    ],
    "illegal_activity": [
        "how to make a bomb",
        "synthesize drugs",
        "hack into",
        "bypass security",
        "manufacture methamphetamine",
        "create malware",
        "pick a lock",
        "forge documents",
        "counterfeit money",
        "exploit vulnerability",
    ],
}

_DEFAULT_PATTERNS: dict[str, list[str]] = {
    "violence": [
        r"(?i)\b(?:kill|murder|attack|shoot|stab)\s+(?:everyone|them all|people)\b",
        r"(?i)\bhow\s+to\s+(?:kill|murder|assassinate|poison)\b",
        r"(?i)\b(?:build|make|construct)\s+(?:a\s+)?(?:bomb|weapon|explosive)\b",
    ],
    "hate_speech": [
        r"(?i)\b(?:all|every)\s+\w+\s+(?:should\s+die|are\s+(?:inferior|subhuman))\b",
        r"(?i)\b(?:death\s+to|eliminate)\s+(?:all\s+)?\w+\b",
    ],
    "sexual": [
        r"(?i)\b(?:explicit|graphic)\s+(?:sexual|erotic)\s+(?:content|material|scene)\b",
        r"(?i)\b(?:generate|create|write)\s+(?:pornographic|sexually\s+explicit)\b",
    ],
    "self_harm": [
        r"(?i)\bhow\s+(?:to|can\s+i)\s+(?:kill\s+myself|end\s+my\s+life|commit\s+suicide)\b",
        r"(?i)\bbest\s+(?:way|method)\s+to\s+(?:die|end\s+it)\b",
    ],
    "illegal_activity": [
        r"(?i)\bhow\s+to\s+(?:hack|break\s+into|bypass)\b",
        r"(?i)\b(?:synthesize|manufacture|produce)\s+(?:drugs|methamphetamine|cocaine)\b",
        r"(?i)\bcreate\s+(?:a\s+)?(?:virus|malware|ransomware|trojan)\b",
    ],
}

ALL_CATEGORIES = frozenset(
    ["violence", "hate_speech", "sexual", "self_harm", "illegal_activity"]
)


class ContentFilterPlugin(GatewayPlugin):
    """
    Content/Toxicity filtering plugin for the RouteIQ gateway.

    Scans LLM request messages and response content for category violations
    using keyword density and regex pattern matching. Configurable thresholds
    and actions per category.
    """

    def __init__(self) -> None:
        self._enabled: bool = False
        self._categories: dict[str, CategoryConfig] = {}
        self._default_threshold: float = 0.7
        self._default_action: str = "block"

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="content-filter",
            version="1.0.0",
            capabilities={PluginCapability.GUARDRAIL},
            priority=70,
            description="Content/toxicity filtering for LLM requests and responses",
        )

    async def startup(
        self, app: "FastAPI", context: PluginContext | None = None
    ) -> None:
        """Load configuration from environment variables, with settings fallback."""
        # These env vars are NOT ROUTEIQ_-prefixed, so read them directly
        # and fall back to settings for defaults.
        env_enabled = os.getenv("CONTENT_FILTER_ENABLED")
        if env_enabled is not None:
            self._enabled = env_enabled.lower() == "true"
        else:
            try:
                from litellm_llmrouter.settings import get_settings

                self._enabled = get_settings().content_filter.enabled
            except Exception:
                self._enabled = False

        if not self._enabled:
            logger.info("Content filter plugin disabled (CONTENT_FILTER_ENABLED=false)")
            return

        try:
            from litellm_llmrouter.settings import get_settings

            cf_cfg = get_settings().content_filter
            self._default_threshold = float(
                os.getenv("CONTENT_FILTER_THRESHOLD", str(cf_cfg.threshold))
            )
            self._default_action = os.getenv(
                "CONTENT_FILTER_ACTION", cf_cfg.action
            ).lower()
            categories_str = os.getenv("CONTENT_FILTER_CATEGORIES", cf_cfg.categories)
        except Exception:
            self._default_threshold = float(
                os.getenv("CONTENT_FILTER_THRESHOLD", "0.7")
            )
            self._default_action = os.getenv("CONTENT_FILTER_ACTION", "block").lower()
            categories_str = os.getenv(
                "CONTENT_FILTER_CATEGORIES",
                "violence,hate_speech,sexual,self_harm,illegal_activity",
            )
        active_categories = {c.strip() for c in categories_str.split(",") if c.strip()}

        for cat in active_categories:
            if cat not in ALL_CATEGORIES:
                logger.warning(f"Unknown content filter category: {cat}, skipping")
                continue

            keywords = _DEFAULT_KEYWORDS.get(cat, [])
            raw_patterns = _DEFAULT_PATTERNS.get(cat, [])
            compiled_patterns = []
            for p in raw_patterns:
                try:
                    compiled_patterns.append(re.compile(p))
                except re.error as e:
                    logger.warning(f"Invalid regex pattern for {cat}: {e}")

            self._categories[cat] = CategoryConfig(
                keywords=keywords,
                patterns=compiled_patterns,
                threshold=self._default_threshold,
                action=self._default_action,
            )

        logger.info(
            f"Content filter plugin started with categories: "
            f"{sorted(self._categories.keys())}"
        )

    async def shutdown(
        self, app: "FastAPI", context: PluginContext | None = None
    ) -> None:
        """Clean up resources."""
        self._categories.clear()
        logger.info("Content filter plugin shut down")

    def _score_content(self, text: str, category: str) -> ContentScore:
        """
        Score text content against a category.

        Uses keyword density and pattern matching to produce a 0.0-1.0 score.
        """
        config = self._categories.get(category)
        if not config:
            return ContentScore(category=category, score=0.0)

        text_lower = text.lower()
        words = text_lower.split()
        word_count = max(len(words), 1)

        # Keyword matching
        matched_keywords: list[str] = []
        for keyword in config.keywords:
            if keyword.lower() in text_lower:
                matched_keywords.append(keyword)

        # Pattern matching
        matched_patterns: list[str] = []
        for pattern in config.patterns:
            if pattern.search(text):
                matched_patterns.append(pattern.pattern)

        # Score calculation:
        # - Keyword density contributes up to 0.5
        # - Pattern matches contribute up to 0.5 each (capped at 0.5 total)
        keyword_score = min(len(matched_keywords) / max(word_count * 0.1, 1), 0.5)
        pattern_score = min(len(matched_patterns) * 0.25, 0.5)
        total_score = min(keyword_score + pattern_score, 1.0)

        return ContentScore(
            category=category,
            score=total_score,
            matched_keywords=matched_keywords,
            matched_patterns=matched_patterns,
        )

    def _extract_user_text(self, messages: list[Any]) -> str:
        """Extract text content from user messages."""
        parts: list[str] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            if role == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    # Multi-part messages (e.g. vision)
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            parts.append(part.get("text", ""))
        return " ".join(parts)

    def _extract_response_text(self, response: Any) -> str:
        """Extract text content from an LLM response object."""
        if response is None:
            return ""

        # LiteLLM ModelResponse
        if hasattr(response, "choices"):
            parts: list[str] = []
            for choice in response.choices:
                if hasattr(choice, "message") and hasattr(choice.message, "content"):
                    content = choice.message.content
                    if isinstance(content, str):
                        parts.append(content)
            return " ".join(parts)

        # Plain string fallback
        if isinstance(response, str):
            return response

        return ""

    async def on_llm_pre_call(
        self, model: str, messages: list[Any], kwargs: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Scan input messages for content violations."""
        if not self._enabled or not self._categories:
            return None

        text = self._extract_user_text(messages)
        if not text:
            return None

        for category, config in self._categories.items():
            score_result = self._score_content(text, category)
            if score_result.score >= config.threshold:
                if config.action == "block":
                    raise GuardrailBlockError(
                        guardrail_name="content-filter",
                        category=category,
                        message=(
                            f"Content violation: {category} "
                            f"(score: {score_result.score:.2f}, "
                            f"threshold: {config.threshold})"
                        ),
                        score=score_result.score,
                    )
                elif config.action == "warn":
                    logger.warning(
                        f"Content filter warning: {category} "
                        f"score={score_result.score:.2f} "
                        f"for model={model}"
                    )
                else:
                    # action == "log"
                    logger.info(
                        f"Content filter log: {category} "
                        f"score={score_result.score:.2f} "
                        f"for model={model}"
                    )

        return None

    async def on_llm_success(
        self, model: str, response: Any, kwargs: dict[str, Any]
    ) -> None:
        """Scan output content for category violations."""
        if not self._enabled or not self._categories:
            return

        text = self._extract_response_text(response)
        if not text:
            return

        warnings: list[str] = []
        for category, config in self._categories.items():
            score_result = self._score_content(text, category)
            if score_result.score >= config.threshold:
                warnings.append(f"{category}={score_result.score:.2f}")
                logger.warning(
                    f"Content filter output warning: {category} "
                    f"score={score_result.score:.2f} "
                    f"for model={model}"
                )

        # Add warning metadata to kwargs for downstream consumers
        if warnings:
            metadata = kwargs.setdefault("metadata", {})
            metadata["X-Content-Filter-Warning"] = "; ".join(warnings)

    async def health_check(self) -> dict[str, Any]:
        """Report plugin health."""
        return {
            "status": "ok" if self._enabled else "disabled",
            "enabled": self._enabled,
            "categories": sorted(self._categories.keys()),
        }
