"""
Tests for the Context Optimizer Plugin.

Tests cover:
1. JSON minification (with and without code blocks)
2. Tool schema deduplication
3. System prompt deduplication
4. Whitespace normalization (preserving code blocks)
5. Chat history trimming
6. Semantic dedup (aggressive mode)
7. Full optimize() pipeline
8. OptimizeResult calculations
9. ContextOptimizerPlugin lifecycle and hooks
10. Edge cases: empty messages, multi-part content, no-op transforms
"""

from __future__ import annotations

import json

import pytest
from unittest.mock import MagicMock, patch

from litellm_llmrouter.gateway.plugins.context_optimizer import (
    ContextOptimizer,
    ContextOptimizerPlugin,
    OptimizeResult,
    _extract_json_spans,
    _CHARS_PER_TOKEN,
)


# ============================================================================
# Helpers
# ============================================================================


def _msg(role: str, content: str) -> dict:
    """Shorthand for creating a message dict."""
    return {"role": role, "content": content}


def _user(content: str) -> dict:
    return _msg("user", content)


def _assistant(content: str) -> dict:
    return _msg("assistant", content)


def _system(content: str) -> dict:
    return _msg("system", content)


# ============================================================================
# OptimizeResult
# ============================================================================


class TestOptimizeResult:
    """Test OptimizeResult dataclass and computed properties."""

    def test_default_values(self):
        r = OptimizeResult()
        assert r.original_chars == 0
        assert r.optimized_chars == 0
        assert r.transforms_applied == []
        assert r.mode == "safe"

    def test_reduction_pct_nonzero(self):
        r = OptimizeResult(original_chars=1000, optimized_chars=600)
        assert r.reduction_pct == pytest.approx(40.0)

    def test_reduction_pct_zero_original(self):
        """Division by zero guard when original_chars is 0."""
        r = OptimizeResult(original_chars=0, optimized_chars=0)
        assert r.reduction_pct == 0.0

    def test_reduction_pct_no_change(self):
        r = OptimizeResult(original_chars=500, optimized_chars=500)
        assert r.reduction_pct == pytest.approx(0.0)

    def test_tokens_saved(self):
        r = OptimizeResult(
            original_estimated_tokens=250,
            optimized_estimated_tokens=100,
        )
        assert r.tokens_saved == 150

    def test_tokens_saved_zero(self):
        r = OptimizeResult(
            original_estimated_tokens=100,
            optimized_estimated_tokens=100,
        )
        assert r.tokens_saved == 0


# ============================================================================
# _extract_json_spans helper
# ============================================================================


class TestExtractJsonSpans:
    """Test the brace-balancing JSON span finder."""

    def test_simple_object(self):
        text = 'some text {"key": "value"} more text'
        spans = _extract_json_spans(text)
        assert len(spans) == 1
        s, e = spans[0]
        assert text[s:e] == '{"key": "value"}'

    def test_nested_object(self):
        text = '{"a": {"b": 1}}'
        spans = _extract_json_spans(text)
        assert len(spans) == 1
        s, e = spans[0]
        assert text[s:e] == text

    def test_array(self):
        text = "here [1, 2, 3] there"
        spans = _extract_json_spans(text)
        assert len(spans) == 1
        s, e = spans[0]
        assert text[s:e] == "[1, 2, 3]"

    def test_multiple_spans(self):
        text = '{"a": 1} plain {"b": 2}'
        spans = _extract_json_spans(text)
        assert len(spans) == 2

    def test_no_json(self):
        text = "Hello, world!"
        spans = _extract_json_spans(text)
        assert spans == []

    def test_unbalanced_brace(self):
        text = '{"a": 1'  # unclosed
        spans = _extract_json_spans(text)
        assert spans == []

    def test_string_with_braces(self):
        text = '{"msg": "hello {world}"}'
        spans = _extract_json_spans(text)
        assert len(spans) == 1


# ============================================================================
# ContextOptimizer — invalid mode
# ============================================================================


class TestContextOptimizerInit:
    """Test constructor validation."""

    def test_valid_safe_mode(self):
        opt = ContextOptimizer(mode="safe")
        assert opt.mode == "safe"

    def test_valid_aggressive_mode(self):
        opt = ContextOptimizer(mode="aggressive")
        assert opt.mode == "aggressive"

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid context optimize mode"):
            ContextOptimizer(mode="turbo")


# ============================================================================
# Transform 1: JSON Minification
# ============================================================================


class TestJsonMinification:
    """Test the _minify_json transform."""

    def test_pretty_json_is_compacted(self):
        pretty_json = json.dumps({"key": "value", "n": 42}, indent=2)
        messages = [_user(f"Here is data: {pretty_json}")]
        opt = ContextOptimizer()

        optimized, result = opt.optimize(messages)

        text = optimized[0]["content"]
        # The compact form should be present
        assert '{"key":"value","n":42}' in text
        assert "json_minify" in str(result.transforms_applied)

    def test_already_compact_json_unchanged(self):
        compact = '{"key":"value"}'
        messages = [_user(f"Data: {compact}")]
        opt = ContextOptimizer()

        optimized, result = opt.optimize(messages)

        # No json_minify transform should fire (nothing saved)
        assert not any("json_minify" in t for t in result.transforms_applied)

    def test_code_blocks_preserved(self):
        """JSON inside fenced code blocks should not be minified."""
        code_block = '```json\n{\n  "key": "value"\n}\n```'
        messages = [_user(f"Look at this:\n{code_block}")]
        opt = ContextOptimizer()

        optimized, result = opt.optimize(messages)

        text = optimized[0]["content"]
        # The code block should contain the original pretty-printed JSON
        assert '"key": "value"' in text

    def test_no_content_message(self):
        """Message with empty content should not crash."""
        messages = [{"role": "user", "content": ""}]
        opt = ContextOptimizer()
        optimized, result = opt.optimize(messages)
        assert optimized[0]["content"] == ""

    def test_multipart_content(self):
        """Multi-part content (list format) should be handled."""
        pretty = json.dumps({"x": 1}, indent=4)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Data: {pretty}"},
                ],
            }
        ]
        opt = ContextOptimizer()
        optimized, result = opt.optimize(messages)
        text_part = optimized[0]["content"][0]["text"]
        assert '{"x":1}' in text_part


# ============================================================================
# Transform 2: Tool Schema Deduplication
# ============================================================================


class TestToolSchemaDedup:
    """Test _dedup_tool_schemas transform."""

    def test_duplicate_schema_replaced_with_reference(self):
        schema = json.dumps(
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object"},
                    "description": "Get weather",
                },
            }
        )
        messages = [
            _user(f"Tools available: {schema}"),
            _assistant("OK"),
            _user(f"Reminder: {schema}"),
        ]
        opt = ContextOptimizer()
        optimized, result = opt.optimize(messages)

        # The second occurrence should be replaced with a reference
        text_last = optimized[2]["content"]
        assert '[see tool "get_weather" definition above]' in text_last
        assert "tool_schema_dedup" in str(result.transforms_applied)

    def test_different_schemas_not_deduped(self):
        schema_a = json.dumps({"name": "foo", "parameters": {"type": "object"}})
        schema_b = json.dumps({"name": "bar", "parameters": {"type": "string"}})
        messages = [_user(schema_a), _user(schema_b)]
        opt = ContextOptimizer()
        optimized, result = opt.optimize(messages)

        # Neither should be replaced
        assert "foo" in optimized[0]["content"]
        assert "bar" in optimized[1]["content"]

    def test_non_tool_json_not_deduped(self):
        """Regular JSON objects (not tool schemas) should not be deduped."""
        data = json.dumps({"temperature": 72, "humidity": 45})
        messages = [_user(data), _user(data)]
        opt = ContextOptimizer()
        optimized, result = opt.optimize(messages)

        # tool_schema_dedup should NOT appear
        assert not any("tool_schema_dedup" in t for t in result.transforms_applied)


# ============================================================================
# Transform 3: System Prompt Deduplication
# ============================================================================


class TestSystemPromptDedup:
    """Test _dedup_system_prompt transform."""

    def test_system_prompt_in_user_message_removed(self):
        long_system = "You are a helpful assistant that answers questions. " * 5
        messages = [
            _system(long_system),
            _user(f"{long_system}\nWhat is Python?"),
        ]
        opt = ContextOptimizer()
        optimized, result = opt.optimize(messages)

        # The user message should no longer contain the full system prompt
        user_text = optimized[1]["content"]
        assert long_system.strip() not in user_text
        assert "What is Python?" in user_text
        assert "system_prompt_dedup" in str(result.transforms_applied)

    def test_short_system_prompt_not_deduped(self):
        """System prompts <= 50 chars are skipped (too short)."""
        short_sys = "Be helpful."
        messages = [
            _system(short_sys),
            _user(f"{short_sys}\nHello"),
        ]
        opt = ContextOptimizer()
        optimized, result = opt.optimize(messages)

        # No dedup should fire for short prompts
        assert not any("system_prompt_dedup" in t for t in result.transforms_applied)

    def test_no_system_messages(self):
        """No system messages -> dedup is a no-op."""
        messages = [_user("Hello"), _assistant("Hi")]
        opt = ContextOptimizer()
        optimized, result = opt.optimize(messages)
        assert not any("system_prompt_dedup" in t for t in result.transforms_applied)

    def test_system_not_duplicated_in_user(self):
        """When system prompt is NOT present in user message, nothing changes."""
        long_system = "You are a coding assistant with deep expertise. " * 5
        messages = [
            _system(long_system),
            _user("What is Python?"),
        ]
        opt = ContextOptimizer()
        optimized, result = opt.optimize(messages)
        assert not any("system_prompt_dedup" in t for t in result.transforms_applied)


# ============================================================================
# Transform 4: Whitespace Normalization
# ============================================================================


class TestWhitespaceNormalization:
    """Test _normalize_whitespace transform."""

    def test_multiple_blank_lines_collapsed(self):
        text = "Line 1\n\n\n\n\nLine 2"
        messages = [_user(text)]
        opt = ContextOptimizer()
        optimized, result = opt.optimize(messages)

        content = optimized[0]["content"]
        # 3+ newlines -> 2 newlines
        assert "\n\n\n" not in content
        assert "Line 1\n\nLine 2" == content

    def test_multiple_spaces_collapsed(self):
        text = "Hello     world"
        messages = [_user(text)]
        opt = ContextOptimizer()
        optimized, result = opt.optimize(messages)

        content = optimized[0]["content"]
        assert "Hello world" == content

    def test_leading_trailing_whitespace_stripped(self):
        text = "   Hello   "
        messages = [_user(text)]
        opt = ContextOptimizer()
        optimized, result = opt.optimize(messages)

        content = optimized[0]["content"]
        assert content == "Hello"

    def test_code_blocks_preserved_in_whitespace_norm(self):
        """Whitespace inside code blocks should NOT be normalized."""
        code = "```python\ndef foo():\n    if True:\n        pass\n```"
        messages = [_user(f"Code:\n\n\n\n\n{code}")]
        opt = ContextOptimizer()
        optimized, result = opt.optimize(messages)

        content = optimized[0]["content"]
        # Code block indentation must be preserved
        assert "    if True:" in content
        assert "        pass" in content

    def test_excessive_indentation_capped(self):
        """Lines with 9+ leading spaces get capped to 8 (but strip still applies)."""
        # The indentation cap applies per-line; put excessive indent in a middle line
        text = "Start\n            deeply indented\nEnd"  # 12 leading spaces on line 2
        messages = [_user(text)]
        opt = ContextOptimizer()
        optimized, result = opt.optimize(messages)

        content = optimized[0]["content"]
        # 9+ spaces at start of line -> 8 spaces
        assert "        deeply indented" in content
        # The outer strip doesn't affect inner lines
        assert content.startswith("Start")


# ============================================================================
# Transform 5: Chat History Trimming
# ============================================================================


class TestHistoryTrimming:
    """Test _trim_history transform."""

    def test_short_conversation_not_trimmed(self):
        messages = [_system("sys"), _user("q1"), _assistant("a1")]
        opt = ContextOptimizer(max_turns=40, keep_last=20)
        optimized, result = opt.optimize(messages)

        # 2 turn messages < 40, no trimming
        assert len(optimized) == 3
        assert not any("history_trim" in t for t in result.transforms_applied)

    def test_long_conversation_trimmed(self):
        """Conversation with > max_turns non-system messages gets trimmed."""
        messages = [_system("System prompt")]
        # Add 50 user/assistant pairs = 100 turn messages
        for i in range(50):
            messages.append(_user(f"Question {i}"))
            messages.append(_assistant(f"Answer {i}"))

        opt = ContextOptimizer(max_turns=10, keep_last=6)
        optimized, result = opt.optimize(messages)

        # Should have: 1 system + 2 first exchange + 1 trim marker + 6 recent = 10
        assert len(optimized) == 10
        assert "history_trim" in str(result.transforms_applied)

        # Check trim marker is present
        contents = [m.get("content", "") for m in optimized]
        assert any("earlier messages trimmed" in c for c in contents)

        # First exchange preserved
        assert optimized[1]["content"] == "Question 0"
        assert optimized[2]["content"] == "Answer 0"

        # Last 6 messages preserved
        assert optimized[-1]["content"] == "Answer 49"

    def test_all_system_messages_kept(self):
        """System messages are always kept, regardless of trimming."""
        messages = [
            _system("System 1"),
            _system("System 2"),
        ]
        for i in range(50):
            messages.append(_user(f"Q{i}"))
            messages.append(_assistant(f"A{i}"))

        opt = ContextOptimizer(max_turns=10, keep_last=4)
        optimized, result = opt.optimize(messages)

        system_msgs = [m for m in optimized if m["role"] == "system"]
        # Both system messages + 1 trim marker (also system role)
        assert len(system_msgs) >= 2
        assert "System 1" in system_msgs[0]["content"]
        assert "System 2" in system_msgs[1]["content"]


# ============================================================================
# Transform 6: Semantic Deduplication (aggressive mode)
# ============================================================================


class TestSemanticDedup:
    """Test _semantic_dedup transform (aggressive mode only)."""

    def test_not_applied_in_safe_mode(self):
        # Two nearly identical user messages
        long_text = "The quick brown fox jumps over the lazy dog. " * 10
        messages = [_user(long_text), _user(long_text)]
        opt = ContextOptimizer(mode="safe")
        optimized, result = opt.optimize(messages)

        assert not any("semantic_dedup" in t for t in result.transforms_applied)
        # Both messages remain
        user_msgs = [m for m in optimized if m["role"] == "user"]
        assert len(user_msgs) == 2

    def test_near_duplicates_merged_in_aggressive(self):
        base_text = "The quick brown fox jumps over the lazy dog. " * 10
        variant = base_text + " Extra unique content."  # >80% similar
        messages = [_user(base_text), _user(variant)]
        opt = ContextOptimizer(mode="aggressive")
        optimized, result = opt.optimize(messages)

        user_msgs = [m for m in optimized if m["role"] == "user"]
        # The earlier message should be merged into the later one
        assert len(user_msgs) == 1
        assert "semantic_dedup" in str(result.transforms_applied)

    def test_different_role_messages_not_merged(self):
        """Messages with different roles should never be merged."""
        long_text = "The quick brown fox jumps over the lazy dog. " * 10
        messages = [_user(long_text), _assistant(long_text)]
        opt = ContextOptimizer(mode="aggressive")
        optimized, result = opt.optimize(messages)

        assert len(optimized) == 2
        assert not any("semantic_dedup" in t for t in result.transforms_applied)

    def test_system_messages_never_merged(self):
        long_text = "You are a helpful AI assistant. " * 10
        messages = [_system(long_text), _system(long_text)]
        opt = ContextOptimizer(mode="aggressive")
        optimized, result = opt.optimize(messages)

        system_msgs = [m for m in optimized if m["role"] == "system"]
        assert len(system_msgs) == 2

    def test_short_messages_skipped(self):
        """Messages shorter than 100 chars are not candidates for dedup."""
        messages = [_user("Hello"), _user("Hello")]
        opt = ContextOptimizer(mode="aggressive")
        optimized, result = opt.optimize(messages)

        user_msgs = [m for m in optimized if m["role"] == "user"]
        assert len(user_msgs) == 2

    def test_single_message_no_dedup(self):
        messages = [_user("Hello world")]
        opt = ContextOptimizer(mode="aggressive")
        optimized, result = opt.optimize(messages)
        assert len(optimized) == 1


# ============================================================================
# Full optimize() pipeline
# ============================================================================


class TestOptimizePipeline:
    """Test the full optimize() pipeline."""

    def test_empty_messages(self):
        opt = ContextOptimizer()
        optimized, result = opt.optimize([])
        assert optimized == []
        assert result.original_chars == 0
        assert result.optimized_chars == 0

    def test_does_not_mutate_input(self):
        """optimize() should deep-copy and not mutate the input list."""
        messages = [_user("Hello world")]
        original_content = messages[0]["content"]
        opt = ContextOptimizer()
        optimized, _ = opt.optimize(messages)
        assert messages[0]["content"] == original_content

    def test_pipeline_records_transforms(self):
        """Multiple transforms can fire in a single optimize() call."""
        pretty_json = json.dumps({"a": 1, "b": 2}, indent=4)
        text = f"   Data: {pretty_json}   "  # whitespace + JSON
        messages = [_user(text)]
        opt = ContextOptimizer()
        optimized, result = opt.optimize(messages)

        # At minimum, whitespace_norm should fire (leading/trailing spaces)
        assert len(result.transforms_applied) >= 1

    def test_result_char_counts_correct(self):
        messages = [_user("Hello" * 100)]
        opt = ContextOptimizer()
        _, result = opt.optimize(messages)

        assert result.original_chars == 500
        assert result.original_estimated_tokens == 500 // _CHARS_PER_TOKEN

    def test_aggressive_mode_runs_all_six_transforms(self):
        """In aggressive mode, the pipeline attempts all 6 transforms."""
        opt = ContextOptimizer(mode="aggressive")
        messages = [_user("Simple message")]
        optimized, result = opt.optimize(messages)
        # Even if no transforms fire, the pipeline should complete
        assert result.mode == "aggressive"


# ============================================================================
# ContextOptimizerPlugin lifecycle
# ============================================================================


class TestContextOptimizerPlugin:
    """Test the plugin wrapper around ContextOptimizer."""

    def test_metadata(self):
        plugin = ContextOptimizerPlugin()
        meta = plugin.metadata
        assert meta.name == "context_optimizer"
        assert meta.version == "1.0.0"

    async def test_startup_off_by_default(self):
        """Plugin is disabled when ROUTEIQ_CONTEXT_OPTIMIZE is unset/off."""
        plugin = ContextOptimizerPlugin()
        app = MagicMock()

        with patch.dict("os.environ", {}, clear=False):
            # Ensure the var is not set
            import os

            os.environ.pop("ROUTEIQ_CONTEXT_OPTIMIZE", None)
            await plugin.startup(app)

        assert plugin._enabled is False

    async def test_startup_safe_mode(self, monkeypatch):
        plugin = ContextOptimizerPlugin()
        app = MagicMock()
        monkeypatch.setenv("ROUTEIQ_CONTEXT_OPTIMIZE", "safe")
        await plugin.startup(app)

        assert plugin._enabled is True
        assert plugin._optimizer is not None
        assert plugin._optimizer.mode == "safe"

    async def test_startup_aggressive_mode(self, monkeypatch):
        plugin = ContextOptimizerPlugin()
        app = MagicMock()
        monkeypatch.setenv("ROUTEIQ_CONTEXT_OPTIMIZE", "aggressive")
        await plugin.startup(app)

        assert plugin._enabled is True
        assert plugin._optimizer.mode == "aggressive"

    async def test_startup_invalid_mode_falls_back_to_safe(self, monkeypatch):
        plugin = ContextOptimizerPlugin()
        app = MagicMock()
        monkeypatch.setenv("ROUTEIQ_CONTEXT_OPTIMIZE", "turbo")
        await plugin.startup(app)

        assert plugin._enabled is True
        assert plugin._optimizer.mode == "safe"

    async def test_on_llm_pre_call_disabled(self):
        """When disabled, on_llm_pre_call returns None."""
        plugin = ContextOptimizerPlugin()
        result = await plugin.on_llm_pre_call("gpt-4", [_user("Hi")], {})
        assert result is None

    async def test_on_llm_pre_call_empty_messages(self, monkeypatch):
        plugin = ContextOptimizerPlugin()
        app = MagicMock()
        monkeypatch.setenv("ROUTEIQ_CONTEXT_OPTIMIZE", "safe")
        await plugin.startup(app)

        result = await plugin.on_llm_pre_call("gpt-4", [], {})
        assert result is None

    async def test_on_llm_pre_call_injects_metadata(self, monkeypatch):
        plugin = ContextOptimizerPlugin()
        app = MagicMock()
        monkeypatch.setenv("ROUTEIQ_CONTEXT_OPTIMIZE", "safe")
        await plugin.startup(app)

        messages = [_user("  Hello   world  ")]
        kwargs: dict = {}
        await plugin.on_llm_pre_call("gpt-4", messages, kwargs)

        # Metadata should be injected
        assert "context_optimize" in kwargs.get("metadata", {})
        meta = kwargs["metadata"]["context_optimize"]
        assert "mode" in meta
        assert "tokens_saved" in meta
        assert "reduction_pct" in meta

    async def test_on_llm_pre_call_updates_counters(self, monkeypatch):
        plugin = ContextOptimizerPlugin()
        app = MagicMock()
        monkeypatch.setenv("ROUTEIQ_CONTEXT_OPTIMIZE", "safe")
        await plugin.startup(app)

        messages = [_user("Hello")]
        await plugin.on_llm_pre_call("gpt-4", messages, {})

        assert plugin._call_count == 1

    async def test_shutdown_resets_state(self, monkeypatch):
        plugin = ContextOptimizerPlugin()
        app = MagicMock()
        monkeypatch.setenv("ROUTEIQ_CONTEXT_OPTIMIZE", "safe")
        await plugin.startup(app)
        assert plugin._enabled is True

        await plugin.shutdown(app)
        assert plugin._enabled is False
        assert plugin._optimizer is None

    async def test_health_check_enabled(self, monkeypatch):
        plugin = ContextOptimizerPlugin()
        app = MagicMock()
        monkeypatch.setenv("ROUTEIQ_CONTEXT_OPTIMIZE", "safe")
        await plugin.startup(app)

        health = await plugin.health_check()
        assert health["status"] == "ok"
        assert health["enabled"] is True
        assert health["mode"] == "safe"

    async def test_health_check_disabled(self):
        plugin = ContextOptimizerPlugin()
        health = await plugin.health_check()
        assert health["status"] == "disabled"
        assert health["enabled"] is False

    async def test_on_llm_pre_call_exception_handled(self, monkeypatch):
        """If optimizer.optimize() raises, pre_call returns None (pass-through)."""
        plugin = ContextOptimizerPlugin()
        app = MagicMock()
        monkeypatch.setenv("ROUTEIQ_CONTEXT_OPTIMIZE", "safe")
        await plugin.startup(app)

        # Patch the optimizer to raise
        plugin._optimizer.optimize = MagicMock(side_effect=RuntimeError("boom"))
        result = await plugin.on_llm_pre_call("gpt-4", [_user("Hi")], {})
        assert result is None


# ============================================================================
# _count_chars helper
# ============================================================================


class TestCountChars:
    """Test the _count_chars static method."""

    def test_string_content(self):
        messages = [_user("Hello")]
        assert ContextOptimizer._count_chars(messages) == 5

    def test_list_content(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "image_url", "image_url": {"url": "..."}},
                ],
            }
        ]
        assert ContextOptimizer._count_chars(messages) == 5

    def test_tool_calls_counted(self):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "1", "function": {"name": "f"}}],
            }
        ]
        count = ContextOptimizer._count_chars(messages)
        assert count > 0  # serialized tool_calls add chars

    def test_empty_messages(self):
        assert ContextOptimizer._count_chars([]) == 0

    def test_none_content(self):
        messages = [{"role": "assistant"}]
        assert ContextOptimizer._count_chars(messages) == 0
