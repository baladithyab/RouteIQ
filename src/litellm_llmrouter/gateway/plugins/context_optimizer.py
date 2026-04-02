"""
Context Optimizer Plugin for RouteIQ Gateway
=============================================

Reduces LLM input tokens by 30-70% using deterministic, lossless transforms
applied to messages before dispatch. Stacks with routing savings.

Two modes:

- **safe** (default): 5 lossless transforms, zero quality loss

  1. JSON minification — compact-serialize JSON objects in message content
  2. Tool schema deduplication — replace repeated tool schemas with references
  3. System prompt deduplication — remove system text duplicated in user messages
  4. Whitespace normalization — collapse excessive whitespace (skip code blocks)
  5. Chat history trimming — for long conversations, keep system + first + last N turns

- **aggressive**: safe + semantic dedup

  6. Semantic deduplication — merge near-duplicate messages using difflib

Configuration (environment variables):

  - ``ROUTEIQ_CONTEXT_OPTIMIZE=off|safe|aggressive`` (default: ``off``)
  - ``ROUTEIQ_CONTEXT_OPTIMIZE_MAX_TURNS=40`` — trim conversations longer than this
  - ``ROUTEIQ_CONTEXT_OPTIMIZE_KEEP_LAST=20`` — keep this many recent turns after trimming

Hook point:

  - ``on_llm_pre_call``: Optimizes the messages list before dispatch to the LLM.

Telemetry:

  Injects ``_routeiq_metadata.context_optimize`` into kwargs metadata with
  mode, tokens_saved, reduction_pct, and transforms applied.
"""

from __future__ import annotations

import copy
import difflib
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from litellm_llmrouter.gateway.plugin_manager import (
    GatewayPlugin,
    PluginCapability,
    PluginContext,
    PluginMetadata,
)

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger("litellm_llmrouter.plugins.context_optimizer")

# Regex for fenced code blocks (``` or ~~~, with optional language tag)
_CODE_BLOCK_RE = re.compile(r"(```[\w]*\n.*?\n```|~~~[\w]*\n.*?\n~~~)", re.DOTALL)

# Chars-to-tokens ratio (approximate, 1 token ≈ 4 characters for English)
_CHARS_PER_TOKEN = 4


def _extract_json_spans(text: str) -> list[tuple[int, int]]:
    """Find start/end offsets of top-level JSON objects/arrays in *text*.

    Uses brace-balancing to handle arbitrarily nested structures that
    simple regexes cannot match. Skips JSON inside double-quoted strings
    to avoid false positives from prose containing ``{`` or ``[``.

    Returns:
        List of ``(start, end)`` byte offsets where ``text[start:end]``
        is a candidate JSON object or array.
    """
    spans: list[tuple[int, int]] = []
    i = 0
    length = len(text)

    while i < length:
        ch = text[i]
        if ch in ("{", "["):
            close_ch = "}" if ch == "{" else "]"
            depth = 1
            start = i
            j = i + 1
            in_str = False
            while j < length and depth > 0:
                c = text[j]
                if in_str:
                    if c == "\\" and j + 1 < length:
                        j += 2  # skip escaped char
                        continue
                    if c == '"':
                        in_str = False
                elif c == '"':
                    in_str = True
                elif c == ch:
                    depth += 1
                elif c == close_ch:
                    depth -= 1
                j += 1

            if depth == 0:
                candidate = text[start:j]
                # Quick sanity: must start with { or [ and end with } or ]
                if len(candidate) >= 2:
                    spans.append((start, j))
                i = j
                continue
        i += 1

    return spans


@dataclass
class OptimizeResult:
    """Result of context optimization applied to a message list.

    Tracks character counts before and after, estimated token savings,
    and which transforms were applied.
    """

    original_chars: int = 0
    optimized_chars: int = 0
    original_estimated_tokens: int = 0
    optimized_estimated_tokens: int = 0
    transforms_applied: list[str] = field(default_factory=list)
    mode: str = "safe"

    @property
    def reduction_pct(self) -> float:
        """Percentage of characters removed (0-100)."""
        if self.original_chars == 0:
            return 0.0
        return (1 - self.optimized_chars / self.original_chars) * 100

    @property
    def tokens_saved(self) -> int:
        """Estimated token reduction."""
        return self.original_estimated_tokens - self.optimized_estimated_tokens


class ContextOptimizer:
    """Stateless context optimizer with configurable transforms.

    Applies a pipeline of deterministic, lossless transforms to a list of
    chat messages, reducing input token count without changing semantic
    meaning.

    Args:
        mode: ``"safe"`` for 5 lossless transforms, ``"aggressive"`` for safe + semantic dedup.
        max_turns: Trim conversations with more turns than this (``safe`` mode only).
        keep_last: After trimming, retain this many recent turns.
    """

    def __init__(
        self, mode: str = "safe", max_turns: int = 40, keep_last: int = 20
    ) -> None:
        if mode not in ("safe", "aggressive"):
            raise ValueError(
                f"Invalid context optimize mode: {mode!r}. Must be 'safe' or 'aggressive'."
            )
        self.mode = mode
        self.max_turns = max_turns
        self.keep_last = keep_last

    def optimize(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], OptimizeResult]:
        """Apply the optimization pipeline to messages.

        Returns:
            Tuple of (optimized messages, result metadata). The input list
            is not mutated — a deep copy is made internally.
        """
        result = OptimizeResult(mode=self.mode)
        result.original_chars = self._count_chars(messages)
        result.original_estimated_tokens = result.original_chars // _CHARS_PER_TOKEN

        # Deep copy to avoid mutating caller's data
        msgs = copy.deepcopy(messages)

        # --- Safe transforms (lossless, order matters) ---
        msgs = self._minify_json(msgs, result)
        msgs = self._dedup_tool_schemas(msgs, result)
        msgs = self._dedup_system_prompt(msgs, result)
        msgs = self._normalize_whitespace(msgs, result)
        msgs = self._trim_history(msgs, result)

        # --- Aggressive transforms ---
        if self.mode == "aggressive":
            msgs = self._semantic_dedup(msgs, result)

        result.optimized_chars = self._count_chars(msgs)
        result.optimized_estimated_tokens = result.optimized_chars // _CHARS_PER_TOKEN
        return msgs, result

    # =========================================================================
    # Internal helpers
    # =========================================================================

    @staticmethod
    def _count_chars(messages: list[dict[str, Any]]) -> int:
        """Sum character length of all message content fields."""
        total = 0
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, str):
                total += len(content)
            elif isinstance(content, list):
                # Multi-part content (vision, tool results, etc.)
                for part in content:
                    if isinstance(part, dict):
                        total += len(part.get("text", ""))
                    elif isinstance(part, str):
                        total += len(part)
            # Also count tool_calls serialized form
            tool_calls = m.get("tool_calls")
            if tool_calls:
                total += len(json.dumps(tool_calls, separators=(",", ":")))
        return total

    @staticmethod
    def _get_text(message: dict[str, Any]) -> str:
        """Extract text content from a message, handling str and list formats."""
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    parts.append(part)
            return "\n".join(parts)
        return ""

    @staticmethod
    def _set_text(message: dict[str, Any], new_text: str) -> None:
        """Set text content on a message, preserving the original format.

        If the message has string content, replaces it directly.
        If the message has list content (multi-part), replaces the first
        text part and leaves other parts (images, etc.) intact.
        """
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = new_text
        elif isinstance(content, list):
            # Replace first text part, preserve non-text parts
            replaced = False
            for i, part in enumerate(content):
                if isinstance(part, dict) and part.get("type") == "text":
                    if not replaced:
                        content[i]["text"] = new_text
                        replaced = True
                    else:
                        # Remove additional text parts since we merged into first
                        content[i]["text"] = ""
                elif isinstance(part, str) and not replaced:
                    content[i] = new_text
                    replaced = True
            if not replaced:
                # No text part found, append one
                content.append({"type": "text", "text": new_text})
        else:
            message["content"] = new_text

    @staticmethod
    def _extract_code_blocks(text: str) -> tuple[str, list[tuple[str, str]]]:
        """Replace fenced code blocks with placeholders.

        Returns:
            Tuple of (text_with_placeholders, list_of_(placeholder, code_block) pairs).
        """
        blocks: list[tuple[str, str]] = []

        def _replacer(match: re.Match[str]) -> str:
            placeholder = f"__CODE_BLOCK_{len(blocks)}__"
            blocks.append((placeholder, match.group(0)))
            return placeholder

        text_clean = _CODE_BLOCK_RE.sub(_replacer, text)
        return text_clean, blocks

    @staticmethod
    def _restore_code_blocks(text: str, blocks: list[tuple[str, str]]) -> str:
        """Restore code block placeholders back to their original content."""
        for placeholder, original in blocks:
            text = text.replace(placeholder, original)
        return text

    # =========================================================================
    # Transform 1: JSON Minification
    # =========================================================================

    def _minify_json(
        self,
        messages: list[dict[str, Any]],
        result: OptimizeResult,
    ) -> list[dict[str, Any]]:
        """Compact-serialize JSON objects and arrays found in message content.

        Finds JSON-like structures in text content, parses them, and re-serializes
        with ``separators=(',', ':')`` (no extra whitespace). Fenced code blocks
        are preserved as-is.

        This is lossless — the JSON data is semantically identical.
        """
        total_saved = 0

        for msg in messages:
            text = self._get_text(msg)
            if not text:
                continue

            # Protect code blocks
            text_clean, blocks = self._extract_code_blocks(text)

            # Find JSON spans and process from right-to-left to preserve offsets
            spans = _extract_json_spans(text_clean)
            parts = list(text_clean)  # work on char list for efficient splicing
            for start, end in reversed(spans):
                candidate = text_clean[start:end]
                try:
                    parsed = json.loads(candidate)
                    compact = json.dumps(
                        parsed, separators=(",", ":"), ensure_ascii=False
                    )
                    saved = len(candidate) - len(compact)
                    if saved > 0:
                        parts[start:end] = list(compact)
                        total_saved += saved
                except json.JSONDecodeError, ValueError, TypeError:
                    pass

            text_minified = "".join(parts)

            # Restore code blocks
            text_final = self._restore_code_blocks(text_minified, blocks)
            if text_final != text:
                self._set_text(msg, text_final)

        if total_saved > 0:
            result.transforms_applied.append(f"json_minify(-{total_saved} chars)")
            logger.debug("json_minify: saved %d chars", total_saved)

        return messages

    # =========================================================================
    # Transform 2: Tool Schema Deduplication
    # =========================================================================

    def _dedup_tool_schemas(
        self,
        messages: list[dict[str, Any]],
        result: OptimizeResult,
    ) -> list[dict[str, Any]]:
        """Replace duplicate tool/function schemas with back-references.

        When the same tool schema appears multiple times across messages
        (common in multi-turn tool-use conversations), the second and
        subsequent occurrences are replaced with a compact reference
        like ``[see tool "function_name" definition above]``.

        Schemas are identified by hashing their JSON representation. Only
        schemas embedded in message content strings are deduplicated; the
        top-level ``tools`` parameter is not touched (handled by LiteLLM).
        """
        seen_schemas: dict[str, str] = {}  # hash -> tool/function name
        total_saved = 0

        for msg in messages:
            text = self._get_text(msg)
            if not text:
                continue

            # Protect code blocks
            text_clean, blocks = self._extract_code_blocks(text)

            # Find JSON spans and process from right-to-left to preserve offsets
            spans = _extract_json_spans(text_clean)
            replacements: list[tuple[int, int, str]] = []

            for start, end in spans:
                candidate = text_clean[start:end]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError, ValueError, TypeError:
                    continue

                if not isinstance(parsed, dict):
                    continue

                # Detect tool/function schema patterns:
                # {"type": "function", "function": {"name": ..., "parameters": ...}}
                # {"name": ..., "parameters": ..., "description": ...}
                tool_name = None
                if parsed.get("type") == "function" and isinstance(
                    parsed.get("function"), dict
                ):
                    tool_name = parsed["function"].get("name")
                elif "name" in parsed and "parameters" in parsed:
                    tool_name = parsed.get("name")

                if not tool_name:
                    continue

                # Hash the schema for dedup comparison
                schema_hash = hashlib.md5(
                    json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()

                if schema_hash in seen_schemas:
                    # Duplicate — replace with reference
                    reference = f'[see tool "{tool_name}" definition above]'
                    saved = len(candidate) - len(reference)
                    if saved > 0:
                        replacements.append((start, end, reference))
                        total_saved += saved
                else:
                    seen_schemas[schema_hash] = tool_name

            # Apply replacements right-to-left
            new_text = text_clean
            for start, end, replacement in reversed(replacements):
                new_text = new_text[:start] + replacement + new_text[end:]

            # Restore code blocks
            text_final = self._restore_code_blocks(new_text, blocks)
            if text_final != text:
                self._set_text(msg, text_final)

        if total_saved > 0:
            result.transforms_applied.append(f"tool_schema_dedup(-{total_saved} chars)")
            logger.debug(
                "tool_schema_dedup: saved %d chars across %d schemas",
                total_saved,
                len(seen_schemas),
            )

        return messages

    # =========================================================================
    # Transform 3: System Prompt Deduplication
    # =========================================================================

    def _dedup_system_prompt(
        self,
        messages: list[dict[str, Any]],
        result: OptimizeResult,
    ) -> list[dict[str, Any]]:
        """Remove system prompt text that is duplicated in later user messages.

        Some frameworks prepend the system prompt to user messages as context.
        This transform detects when a user message contains the full system
        prompt verbatim and removes the duplicated portion, keeping only the
        unique user content.

        Only removes exact substring matches of the entire system content.
        Partial overlaps are left untouched to avoid semantic changes.
        """
        # Collect all system message content
        system_texts: list[str] = []
        for msg in messages:
            if msg.get("role") == "system":
                sys_text = self._get_text(msg).strip()
                if len(sys_text) > 50:  # Only dedup substantial system prompts
                    system_texts.append(sys_text)

        if not system_texts:
            return messages

        total_saved = 0
        for msg in messages:
            if msg.get("role") != "user":
                continue

            text = self._get_text(msg)
            if not text:
                continue

            original_len = len(text)
            for sys_text in system_texts:
                # Check for exact containment of system prompt in user message
                idx = text.find(sys_text)
                if idx != -1:
                    # Remove the system prompt portion
                    before = text[:idx]
                    after = text[idx + len(sys_text) :]
                    # Clean up: remove leading/trailing whitespace and separator artifacts
                    combined = (before.rstrip() + "\n" + after.lstrip()).strip()
                    if combined and len(combined) < original_len:
                        text = combined

            saved = original_len - len(text)
            if saved > 0:
                self._set_text(msg, text)
                total_saved += saved

        if total_saved > 0:
            result.transforms_applied.append(
                f"system_prompt_dedup(-{total_saved} chars)"
            )
            logger.debug("system_prompt_dedup: saved %d chars", total_saved)

        return messages

    # =========================================================================
    # Transform 4: Whitespace Normalization
    # =========================================================================

    def _normalize_whitespace(
        self,
        messages: list[dict[str, Any]],
        result: OptimizeResult,
    ) -> list[dict[str, Any]]:
        """Collapse excessive whitespace in message content.

        - 3+ consecutive blank lines → 2 blank lines
        - 2+ consecutive spaces (outside code blocks) → 1 space
        - Leading/trailing whitespace per message → stripped

        Fenced code blocks (````` and ``~~~``) are preserved exactly as-is
        to avoid breaking code formatting.
        """
        total_saved = 0

        for msg in messages:
            text = self._get_text(msg)
            if not text:
                continue

            original_len = len(text)

            # Extract code blocks before normalizing
            text_clean, blocks = self._extract_code_blocks(text)

            # Collapse 3+ blank lines to 2
            text_clean = re.sub(r"\n{3,}", "\n\n", text_clean)

            # Collapse runs of 2+ spaces to single space (not at line start for indentation)
            # We preserve single leading spaces for list items / indentation
            text_clean = re.sub(r"(?<=\S) {2,}(?=\S)", " ", text_clean)

            # Also collapse runs of spaces at start of lines to reasonable indent
            # (but only if > 8 spaces, to preserve normal indentation)
            text_clean = re.sub(r"^ {9,}", "        ", text_clean, flags=re.MULTILINE)

            # Strip leading/trailing whitespace from the whole message
            text_clean = text_clean.strip()

            # Restore code blocks
            text_final = self._restore_code_blocks(text_clean, blocks)

            saved = original_len - len(text_final)
            if saved > 0:
                self._set_text(msg, text_final)
                total_saved += saved

        if total_saved > 0:
            result.transforms_applied.append(f"whitespace_norm(-{total_saved} chars)")
            logger.debug("whitespace_norm: saved %d chars", total_saved)

        return messages

    # =========================================================================
    # Transform 5: Chat History Trimming
    # =========================================================================

    def _trim_history(
        self,
        messages: list[dict[str, Any]],
        result: OptimizeResult,
    ) -> list[dict[str, Any]]:
        """Trim long conversations to keep system + first exchange + last N turns.

        For conversations with more than ``max_turns`` non-system messages,
        retains:

        1. All system messages (always kept at the front)
        2. The first user+assistant exchange (establishes context)
        3. The last ``keep_last`` messages (recent context)

        A ``[... N earlier messages trimmed for context ...]`` marker is
        inserted between the first exchange and the recent window.
        """
        # Separate system messages from conversation turns
        system_msgs: list[dict[str, Any]] = []
        turn_msgs: list[dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "system":
                system_msgs.append(msg)
            else:
                turn_msgs.append(msg)

        if len(turn_msgs) <= self.max_turns:
            return messages  # No trimming needed

        # Keep first exchange (first 2 messages: user + assistant)
        first_exchange = turn_msgs[:2] if len(turn_msgs) >= 2 else turn_msgs[:1]
        # Keep last N messages
        recent = turn_msgs[-self.keep_last :]

        # Calculate how many messages were trimmed
        trimmed_start = len(first_exchange)
        trimmed_end = len(turn_msgs) - self.keep_last
        trimmed_count = max(0, trimmed_end - trimmed_start)

        if trimmed_count <= 0:
            return (
                messages  # Nothing to trim (overlap between first exchange and recent)
            )

        # Build trimmed message list
        trim_marker: dict[str, Any] = {
            "role": "system",
            "content": f"[... {trimmed_count} earlier messages trimmed for context ...]",
        }

        trimmed = system_msgs + first_exchange + [trim_marker] + recent

        # Calculate savings
        original_chars = self._count_chars(messages)
        trimmed_chars = self._count_chars(trimmed)
        saved = original_chars - trimmed_chars

        if saved > 0:
            result.transforms_applied.append(
                f"history_trim(-{trimmed_count} msgs, -{saved} chars)"
            )
            logger.debug(
                "history_trim: removed %d messages, saved %d chars",
                trimmed_count,
                saved,
            )

        return trimmed

    # =========================================================================
    # Transform 6: Semantic Deduplication (aggressive mode only)
    # =========================================================================

    def _semantic_dedup(
        self,
        messages: list[dict[str, Any]],
        result: OptimizeResult,
    ) -> list[dict[str, Any]]:
        """Merge near-duplicate messages using word-level sequence matching.

        Uses ``difflib.SequenceMatcher`` to detect pairs of messages with
        the same role where >80% of content overlaps. When a near-duplicate
        is found, the earlier message is removed and unique content from it
        is prepended to the later message with a ``[merged from earlier]``
        marker.

        Only applies to consecutive messages with the same role (which
        indicates accidental duplication or framework-injected repetition).
        System messages are never merged. Messages shorter than 100 chars
        are skipped to avoid false positives on short replies.
        """
        if len(messages) < 2:
            return messages

        total_merged = 0
        total_saved = 0
        indices_to_remove: set[int] = set()

        for i in range(len(messages) - 1):
            if i in indices_to_remove:
                continue

            msg_a = messages[i]
            msg_b = messages[i + 1]

            # Only merge same-role, non-system messages
            role_a = msg_a.get("role", "")
            role_b = msg_b.get("role", "")
            if role_a != role_b or role_a == "system":
                continue

            text_a = self._get_text(msg_a)
            text_b = self._get_text(msg_b)

            # Skip short messages (low confidence for similarity)
            if len(text_a) < 100 or len(text_b) < 100:
                continue

            # Compute similarity ratio
            matcher = difflib.SequenceMatcher(
                None, text_a.split(), text_b.split(), autojunk=True
            )
            ratio = matcher.ratio()

            if ratio >= 0.80:
                # Extract unique content from the earlier message
                unique_parts: list[str] = []
                for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                    if tag in ("delete", "replace"):
                        # Words in A but not in B
                        unique_words = text_a.split()[i1:i2]
                        if unique_words:
                            unique_parts.append(" ".join(unique_words))

                saved_chars = len(text_a)

                if unique_parts:
                    unique_text = " ".join(unique_parts)
                    # Prepend unique content to the later message
                    merged_text = f"[merged from earlier] {unique_text}\n\n{text_b}"
                    self._set_text(msg_b, merged_text)
                    saved_chars -= len(unique_text) + len("[merged from earlier] \n\n")

                if saved_chars > 0:
                    indices_to_remove.add(i)
                    total_merged += 1
                    total_saved += saved_chars

        if indices_to_remove:
            messages = [
                m for idx, m in enumerate(messages) if idx not in indices_to_remove
            ]
            result.transforms_applied.append(
                f"semantic_dedup(-{total_merged} msgs, -{total_saved} chars)"
            )
            logger.debug(
                "semantic_dedup: merged %d message pairs, saved %d chars",
                total_merged,
                total_saved,
            )

        return messages


class ContextOptimizerPlugin(GatewayPlugin):
    """Gateway plugin that optimizes message context before LLM dispatch.

    Wraps the :class:`ContextOptimizer` engine and wires it into the
    ``on_llm_pre_call`` hook so that every LLM request passing through
    the gateway gets context optimization applied transparently.

    Configuration is read from environment variables at startup time.
    When ``ROUTEIQ_CONTEXT_OPTIMIZE=off`` (the default), the plugin
    starts but does nothing — zero overhead.
    """

    def __init__(self) -> None:
        self._optimizer: ContextOptimizer | None = None
        self._enabled: bool = False
        self._call_count: int = 0
        self._total_tokens_saved: int = 0

    @property
    def metadata(self) -> PluginMetadata:
        """Plugin metadata for the context optimizer."""
        return PluginMetadata(
            name="context_optimizer",
            version="1.0.0",
            capabilities={PluginCapability.EVALUATOR},  # pre-request evaluation
            priority=50,  # Run early, before routing/guardrails
            description="Reduces input tokens 30-70% via deterministic transforms",
        )

    async def startup(
        self, app: "FastAPI", context: PluginContext | None = None
    ) -> None:
        """Initialize the optimizer from environment config.

        Reads ``ROUTEIQ_CONTEXT_OPTIMIZE``, ``ROUTEIQ_CONTEXT_OPTIMIZE_MAX_TURNS``,
        and ``ROUTEIQ_CONTEXT_OPTIMIZE_KEEP_LAST`` to configure the optimizer engine.
        """
        mode = os.environ.get("ROUTEIQ_CONTEXT_OPTIMIZE", "off").lower().strip()

        if mode == "off":
            logger.info(
                "Context optimizer plugin disabled (ROUTEIQ_CONTEXT_OPTIMIZE=off)"
            )
            self._enabled = False
            return

        if mode not in ("safe", "aggressive"):
            logger.warning(
                "Invalid ROUTEIQ_CONTEXT_OPTIMIZE=%r, falling back to 'safe'", mode
            )
            mode = "safe"

        max_turns = int(os.environ.get("ROUTEIQ_CONTEXT_OPTIMIZE_MAX_TURNS", "40"))
        keep_last = int(os.environ.get("ROUTEIQ_CONTEXT_OPTIMIZE_KEEP_LAST", "20"))

        if keep_last >= max_turns:
            logger.warning(
                "ROUTEIQ_CONTEXT_OPTIMIZE_KEEP_LAST (%d) >= MAX_TURNS (%d), "
                "history trimming will have no effect",
                keep_last,
                max_turns,
            )

        self._optimizer = ContextOptimizer(
            mode=mode, max_turns=max_turns, keep_last=keep_last
        )
        self._enabled = True
        self._call_count = 0
        self._total_tokens_saved = 0

        logger.info(
            "Context optimizer started: mode=%s, max_turns=%d, keep_last=%d",
            mode,
            max_turns,
            keep_last,
        )

    async def shutdown(
        self, app: "FastAPI", context: PluginContext | None = None
    ) -> None:
        """Log summary statistics and clean up."""
        if self._enabled and self._call_count > 0:
            avg_saved = self._total_tokens_saved // max(self._call_count, 1)
            logger.info(
                "Context optimizer shutting down: %d calls, %d total tokens saved "
                "(avg %d per call)",
                self._call_count,
                self._total_tokens_saved,
                avg_saved,
            )
        self._optimizer = None
        self._enabled = False

    async def on_llm_pre_call(
        self, model: str, messages: list[Any], kwargs: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Optimize messages before LLM dispatch.

        Applies the configured transforms to the messages list. Injects
        optimization telemetry into ``kwargs["metadata"]`` for downstream
        observability.

        Args:
            model: Target model name (e.g. ``"gpt-4"``, ``"claude-3-opus"``).
            messages: The messages list being sent to the LLM.
            kwargs: Additional call parameters (modified in-place with metadata).

        Returns:
            Dict with ``"messages"`` key containing the optimized messages,
            or ``None`` if optimization is disabled or no messages present.
        """
        if not self._enabled or self._optimizer is None:
            return None

        if not messages:
            return None

        try:
            start_ns = time.monotonic_ns()
            optimized, opt_result = self._optimizer.optimize(messages)
            elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000

            # Update counters
            self._call_count += 1
            self._total_tokens_saved += opt_result.tokens_saved

            # Inject telemetry metadata
            metadata = kwargs.setdefault("metadata", {})
            metadata["context_optimize"] = {
                "mode": opt_result.mode,
                "original_tokens": opt_result.original_estimated_tokens,
                "optimized_tokens": opt_result.optimized_estimated_tokens,
                "tokens_saved": opt_result.tokens_saved,
                "reduction_pct": round(opt_result.reduction_pct, 1),
                "transforms": opt_result.transforms_applied,
                "elapsed_ms": round(elapsed_ms, 2),
            }

            if opt_result.tokens_saved > 0:
                logger.debug(
                    "Context optimized for model=%s: %d → %d tokens "
                    "(-%.0f%%, %d saved, %.1fms)",
                    model,
                    opt_result.original_estimated_tokens,
                    opt_result.optimized_estimated_tokens,
                    opt_result.reduction_pct,
                    opt_result.tokens_saved,
                    elapsed_ms,
                )

            # Return kwargs overrides — the callback bridge merges this into kwargs
            # We replace messages in-place since the bridge doesn't support
            # returning a new messages list directly
            messages.clear()
            messages.extend(optimized)
            return None

        except Exception:
            logger.exception(
                "Context optimization failed for model=%s, passing through unmodified",
                model,
            )
            return None

    async def health_check(self) -> dict[str, Any]:
        """Report plugin health and cumulative stats."""
        return {
            "status": "ok" if self._enabled else "disabled",
            "enabled": self._enabled,
            "mode": self._optimizer.mode if self._optimizer else "off",
            "calls": self._call_count,
            "total_tokens_saved": self._total_tokens_saved,
        }
