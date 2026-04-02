"""Unit tests for the prompt management & versioning system."""

import json

import pytest

from litellm_llmrouter.prompt_management import (
    PromptManager,
    get_prompt_manager,
    reset_prompt_manager,
    _validate_prompt_name,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def _clean_manager():
    """Ensure a fresh manager for each test."""
    reset_prompt_manager()
    yield
    reset_prompt_manager()


@pytest.fixture
def manager() -> PromptManager:
    return get_prompt_manager()


# ============================================================================
# Name Validation
# ============================================================================


class TestNameValidation:
    def test_valid_names(self):
        assert _validate_prompt_name("a") == "a"
        assert _validate_prompt_name("code-review") == "code-review"
        assert _validate_prompt_name("my-prompt-v2") == "my-prompt-v2"
        assert _validate_prompt_name("x1") == "x1"
        assert _validate_prompt_name("abc123") == "abc123"

    def test_invalid_empty(self):
        with pytest.raises(ValueError, match="Invalid prompt name"):
            _validate_prompt_name("")

    def test_invalid_starts_with_hyphen(self):
        with pytest.raises(ValueError, match="Invalid prompt name"):
            _validate_prompt_name("-bad")

    def test_invalid_ends_with_hyphen(self):
        with pytest.raises(ValueError, match="Invalid prompt name"):
            _validate_prompt_name("bad-")

    def test_invalid_uppercase(self):
        with pytest.raises(ValueError, match="Invalid prompt name"):
            _validate_prompt_name("BadName")

    def test_invalid_spaces(self):
        with pytest.raises(ValueError, match="Invalid prompt name"):
            _validate_prompt_name("bad name")

    def test_invalid_special_chars(self):
        with pytest.raises(ValueError, match="Invalid prompt name"):
            _validate_prompt_name("bad_name")


# ============================================================================
# CRUD Operations
# ============================================================================


class TestCreatePrompt:
    def test_create_basic(self, manager: PromptManager):
        prompt = manager.create_prompt(
            name="hello",
            template="Say hello to {{name}}",
        )
        assert prompt.name == "hello"
        assert prompt.active_version == 1
        assert len(prompt.versions) == 1
        assert prompt.versions[1].template == "Say hello to {{name}}"
        assert prompt.versions[1].change_note == "Initial version"
        assert prompt.created_at > 0

    def test_create_with_system_template(self, manager: PromptManager):
        prompt = manager.create_prompt(
            name="assistant",
            template="{{question}}",
            system_template="You are a helpful {{role}}.",
            model="gpt-4o",
            description="General assistant",
        )
        assert prompt.versions[1].system_template == "You are a helpful {{role}}."
        assert prompt.versions[1].model == "gpt-4o"
        assert prompt.description == "General assistant"

    def test_create_with_metadata(self, manager: PromptManager):
        prompt = manager.create_prompt(
            name="tagged",
            template="test",
            tags=["dev", "test"],
            metadata={"team": "engineering"},
        )
        assert prompt.tags == ["dev", "test"]
        assert prompt.metadata == {"team": "engineering"}

    def test_create_with_workspace(self, manager: PromptManager):
        prompt = manager.create_prompt(
            name="scoped",
            template="test",
            workspace_id="ws-acme",
        )
        assert prompt.workspace_id == "ws-acme"

    def test_create_duplicate_raises(self, manager: PromptManager):
        manager.create_prompt(name="dup", template="first")
        with pytest.raises(ValueError, match="already exists"):
            manager.create_prompt(name="dup", template="second")

    def test_create_same_name_different_workspace(self, manager: PromptManager):
        p1 = manager.create_prompt(name="shared", template="v1", workspace_id="ws-a")
        p2 = manager.create_prompt(name="shared", template="v2", workspace_id="ws-b")
        assert p1.workspace_id == "ws-a"
        assert p2.workspace_id == "ws-b"

    def test_create_invalid_name(self, manager: PromptManager):
        with pytest.raises(ValueError, match="Invalid prompt name"):
            manager.create_prompt(name="BAD NAME!", template="test")

    def test_create_with_temperature_and_max_tokens(self, manager: PromptManager):
        prompt = manager.create_prompt(
            name="precise",
            template="test",
            temperature=0.1,
            max_tokens=500,
        )
        assert prompt.versions[1].temperature == 0.1
        assert prompt.versions[1].max_tokens == 500


class TestGetPrompt:
    def test_get_existing(self, manager: PromptManager):
        manager.create_prompt(name="exists", template="test")
        prompt = manager.get_prompt("exists")
        assert prompt is not None
        assert prompt.name == "exists"

    def test_get_nonexistent(self, manager: PromptManager):
        assert manager.get_prompt("nope") is None

    def test_get_with_workspace(self, manager: PromptManager):
        manager.create_prompt(name="scoped", template="ws", workspace_id="ws-1")
        assert manager.get_prompt("scoped", workspace_id="ws-1") is not None
        assert manager.get_prompt("scoped", workspace_id="ws-2") is None
        assert manager.get_prompt("scoped") is None  # no workspace = different key

    def test_get_returns_deep_copy(self, manager: PromptManager):
        manager.create_prompt(name="copy-test", template="original")
        prompt = manager.get_prompt("copy-test")
        assert prompt is not None
        prompt.description = "mutated"
        fresh = manager.get_prompt("copy-test")
        assert fresh is not None
        assert fresh.description == ""  # original unchanged


class TestUpdatePrompt:
    def test_update_creates_new_version(self, manager: PromptManager):
        manager.create_prompt(name="evolve", template="v1")
        updated = manager.update_prompt(
            name="evolve",
            template="v2",
            change_note="Improved clarity",
        )
        assert updated.active_version == 2
        assert len(updated.versions) == 2
        assert updated.versions[1].template == "v1"
        assert updated.versions[2].template == "v2"
        assert updated.versions[2].change_note == "Improved clarity"

    def test_update_nonexistent_raises(self, manager: PromptManager):
        with pytest.raises(ValueError, match="not found"):
            manager.update_prompt(name="ghost", template="test")

    def test_update_preserves_old_versions(self, manager: PromptManager):
        manager.create_prompt(name="multi", template="v1")
        manager.update_prompt(name="multi", template="v2")
        manager.update_prompt(name="multi", template="v3")
        prompt = manager.get_prompt("multi")
        assert prompt is not None
        assert len(prompt.versions) == 3
        assert prompt.active_version == 3

    def test_update_with_metadata_changes(self, manager: PromptManager):
        manager.create_prompt(
            name="meta",
            template="v1",
            description="old",
            tags=["old"],
        )
        updated = manager.update_prompt(
            name="meta",
            template="v2",
            description="new",
            tags=["new"],
            metadata={"key": "value"},
        )
        assert updated.description == "new"
        assert updated.tags == ["new"]
        assert updated.metadata == {"key": "value"}


class TestDeletePrompt:
    def test_delete_existing(self, manager: PromptManager):
        manager.create_prompt(name="doomed", template="test")
        assert manager.delete_prompt("doomed") is True
        assert manager.get_prompt("doomed") is None

    def test_delete_nonexistent(self, manager: PromptManager):
        assert manager.delete_prompt("nope") is False

    def test_delete_with_workspace(self, manager: PromptManager):
        manager.create_prompt(name="scoped", template="t", workspace_id="ws-1")
        assert manager.delete_prompt("scoped", workspace_id="ws-2") is False
        assert manager.delete_prompt("scoped", workspace_id="ws-1") is True


class TestListPrompts:
    def test_list_empty(self, manager: PromptManager):
        assert manager.list_prompts() == []

    def test_list_all(self, manager: PromptManager):
        manager.create_prompt(name="b-prompt", template="b")
        manager.create_prompt(name="a-prompt", template="a")
        prompts = manager.list_prompts()
        assert len(prompts) == 2
        # Sorted by name
        assert prompts[0].name == "a-prompt"
        assert prompts[1].name == "b-prompt"

    def test_list_by_workspace(self, manager: PromptManager):
        manager.create_prompt(name="ws1", template="t", workspace_id="ws-a")
        manager.create_prompt(name="ws2", template="t", workspace_id="ws-b")
        manager.create_prompt(name="global", template="t")

        assert len(manager.list_prompts(workspace_id="ws-a")) == 1
        assert len(manager.list_prompts(workspace_id="ws-b")) == 1
        assert len(manager.list_prompts()) == 3

    def test_list_by_tags(self, manager: PromptManager):
        manager.create_prompt(name="p1", template="t", tags=["dev", "test"])
        manager.create_prompt(name="p2", template="t", tags=["prod"])
        manager.create_prompt(name="p3", template="t", tags=["dev", "prod"])

        assert len(manager.list_prompts(tags=["dev"])) == 2
        assert len(manager.list_prompts(tags=["prod"])) == 2
        assert len(manager.list_prompts(tags=["dev", "prod"])) == 1


# ============================================================================
# Version Management
# ============================================================================


class TestRollback:
    def test_rollback_to_v1(self, manager: PromptManager):
        manager.create_prompt(name="roll", template="v1")
        manager.update_prompt(name="roll", template="v2")
        manager.update_prompt(name="roll", template="v3")

        rolled = manager.rollback("roll", 1)
        assert rolled.active_version == 1

    def test_rollback_nonexistent_prompt(self, manager: PromptManager):
        with pytest.raises(ValueError, match="not found"):
            manager.rollback("ghost", 1)

    def test_rollback_nonexistent_version(self, manager: PromptManager):
        manager.create_prompt(name="roll", template="v1")
        with pytest.raises(ValueError, match="Version 99 not found"):
            manager.rollback("roll", 99)


# ============================================================================
# A/B Testing
# ============================================================================


class TestABTesting:
    def test_set_ab_test(self, manager: PromptManager):
        manager.create_prompt(name="ab", template="v1")
        manager.update_prompt(name="ab", template="v2")

        result = manager.set_ab_test("ab", {1: 0.5, 2: 0.5})
        assert result.ab_enabled is True
        assert result.ab_versions == {1: 0.5, 2: 0.5}

    def test_ab_test_invalid_version(self, manager: PromptManager):
        manager.create_prompt(name="ab", template="v1")
        with pytest.raises(ValueError, match="Version 99 not found"):
            manager.set_ab_test("ab", {1: 0.5, 99: 0.5})

    def test_ab_test_weights_must_sum_to_one(self, manager: PromptManager):
        manager.create_prompt(name="ab", template="v1")
        manager.update_prompt(name="ab", template="v2")
        with pytest.raises(ValueError, match="must sum to"):
            manager.set_ab_test("ab", {1: 0.3, 2: 0.3})

    def test_ab_test_negative_weight(self, manager: PromptManager):
        manager.create_prompt(name="ab", template="v1")
        with pytest.raises(ValueError, match="between 0 and 1"):
            manager.set_ab_test("ab", {1: -0.5})

    def test_ab_test_empty_versions(self, manager: PromptManager):
        manager.create_prompt(name="ab", template="v1")
        with pytest.raises(ValueError, match="at least one version"):
            manager.set_ab_test("ab", {})

    def test_stop_ab_test(self, manager: PromptManager):
        manager.create_prompt(name="ab", template="v1")
        manager.update_prompt(name="ab", template="v2")
        manager.set_ab_test("ab", {1: 0.5, 2: 0.5})

        result = manager.stop_ab_test("ab")
        assert result.ab_enabled is False
        assert result.ab_versions == {}

    def test_stop_ab_test_with_winner(self, manager: PromptManager):
        manager.create_prompt(name="ab", template="v1")
        manager.update_prompt(name="ab", template="v2")
        manager.set_ab_test("ab", {1: 0.5, 2: 0.5})

        result = manager.stop_ab_test("ab", winner=1)
        assert result.ab_enabled is False
        assert result.active_version == 1

    def test_stop_ab_test_invalid_winner(self, manager: PromptManager):
        manager.create_prompt(name="ab", template="v1")
        manager.set_ab_test("ab", {1: 1.0})
        with pytest.raises(ValueError, match="Winner version 99 not found"):
            manager.stop_ab_test("ab", winner=99)

    def test_select_version_without_ab(self, manager: PromptManager):
        """Without A/B testing, always returns active version."""
        manager.create_prompt(name="no-ab", template="v1")
        manager.update_prompt(name="no-ab", template="v2")
        prompt = manager.get_prompt("no-ab")
        assert prompt is not None
        assert manager._select_version(prompt) == 2

    def test_select_version_with_ab(self, manager: PromptManager):
        """With A/B testing, returns a valid version."""
        manager.create_prompt(name="ab", template="v1")
        manager.update_prompt(name="ab", template="v2")
        manager.set_ab_test("ab", {1: 0.5, 2: 0.5})

        prompt = manager.get_prompt("ab")
        assert prompt is not None
        # Run many times, should hit both versions
        versions_seen: set[int] = set()
        for _ in range(100):
            v = manager._select_version(prompt)
            versions_seen.add(v)
        assert versions_seen == {1, 2}


# ============================================================================
# Resolution
# ============================================================================


class TestResolvePrompt:
    def test_resolve_basic(self, manager: PromptManager):
        manager.create_prompt(
            name="greet",
            template="Hello, {{name}}! Welcome to {{place}}.",
        )
        messages, model, meta = manager.resolve_prompt(
            "greet", {"name": "Alice", "place": "Wonderland"}
        )
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello, Alice! Welcome to Wonderland."
        assert model is None
        assert meta["prompt_name"] == "greet"
        assert meta["prompt_version"] == 1

    def test_resolve_with_system(self, manager: PromptManager):
        manager.create_prompt(
            name="sys",
            template="{{question}}",
            system_template="You are a {{role}}.",
        )
        messages, _, _ = manager.resolve_prompt(
            "sys", {"question": "What is 2+2?", "role": "math tutor"}
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a math tutor."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "What is 2+2?"

    def test_resolve_with_model_override(self, manager: PromptManager):
        manager.create_prompt(
            name="pinned",
            template="test",
            model="claude-3-opus",
        )
        _, model, _ = manager.resolve_prompt("pinned")
        assert model == "claude-3-opus"

    def test_resolve_missing_variables_preserved(self, manager: PromptManager):
        manager.create_prompt(
            name="partial",
            template="Hello {{name}}, your id is {{id}}.",
        )
        messages, _, _ = manager.resolve_prompt("partial", {"name": "Bob"})
        assert messages[0]["content"] == "Hello Bob, your id is {{id}}."

    def test_resolve_nonexistent_raises(self, manager: PromptManager):
        with pytest.raises(ValueError, match="Prompt not found"):
            manager.resolve_prompt("ghost")

    def test_resolve_with_ab_test(self, manager: PromptManager):
        manager.create_prompt(name="ab", template="Version 1")
        manager.update_prompt(name="ab", template="Version 2")
        manager.set_ab_test("ab", {1: 0.5, 2: 0.5})

        # Should resolve successfully with either version
        messages, _, meta = manager.resolve_prompt("ab")
        assert messages[0]["content"] in ("Version 1", "Version 2")
        assert meta["ab_variant"] is not None

    def test_resolve_includes_temperature(self, manager: PromptManager):
        manager.create_prompt(
            name="temp",
            template="test",
            temperature=0.7,
            max_tokens=100,
        )
        _, _, meta = manager.resolve_prompt("temp")
        assert meta["temperature"] == 0.7
        assert meta["max_tokens"] == 100


# ============================================================================
# Template Rendering
# ============================================================================


class TestRendering:
    def test_render_simple(self, manager: PromptManager):
        result = manager._render("Hello {{name}}", {"name": "World"})
        assert result == "Hello World"

    def test_render_multiple_vars(self, manager: PromptManager):
        result = manager._render(
            "{{a}} + {{b}} = {{c}}",
            {"a": "1", "b": "2", "c": "3"},
        )
        assert result == "1 + 2 = 3"

    def test_render_no_vars(self, manager: PromptManager):
        result = manager._render("No variables here", {})
        assert result == "No variables here"

    def test_render_missing_var_preserved(self, manager: PromptManager):
        result = manager._render("Hello {{name}}", {})
        assert result == "Hello {{name}}"

    def test_render_extra_vars_ignored(self, manager: PromptManager):
        result = manager._render("Hello {{name}}", {"name": "X", "extra": "Y"})
        assert result == "Hello X"


# ============================================================================
# Import/Export
# ============================================================================


class TestImportExport:
    def test_export_empty(self, manager: PromptManager):
        exported = manager.export_prompts()
        assert json.loads(exported) == []

    def test_export_roundtrip(self, manager: PromptManager):
        manager.create_prompt(name="p1", template="t1", tags=["dev"])
        manager.create_prompt(name="p2", template="t2", model="gpt-4")

        exported = manager.export_prompts()
        data = json.loads(exported)
        assert len(data) == 2

        # Reset and reimport
        reset_prompt_manager()
        fresh = get_prompt_manager()
        count = fresh.import_prompts(data)
        assert count == 2
        assert fresh.get_prompt("p1") is not None
        assert fresh.get_prompt("p2") is not None

    def test_import_with_workspace_override(self, manager: PromptManager):
        manager.create_prompt(name="p1", template="t1")
        exported = json.loads(manager.export_prompts())

        reset_prompt_manager()
        fresh = get_prompt_manager()
        count = fresh.import_prompts(exported, workspace_id="ws-new")
        assert count == 1
        # Should be findable under the new workspace
        assert fresh.get_prompt("p1", workspace_id="ws-new") is not None

    def test_import_skips_invalid(self, manager: PromptManager):
        data = [
            {"name": "good", "template": "t", "versions": {}, "active_version": 1},
            {"invalid": "data"},  # missing required fields
        ]
        count = manager.import_prompts(data)
        # At least one should import (the valid one)
        assert count >= 1

    def test_export_by_workspace(self, manager: PromptManager):
        manager.create_prompt(name="p1", template="t1", workspace_id="ws-a")
        manager.create_prompt(name="p2", template="t2", workspace_id="ws-b")

        exported = json.loads(manager.export_prompts(workspace_id="ws-a"))
        assert len(exported) == 1
        assert exported[0]["name"] == "p1"


# ============================================================================
# Singleton Pattern
# ============================================================================


class TestSingleton:
    def test_get_returns_same_instance(self):
        m1 = get_prompt_manager()
        m2 = get_prompt_manager()
        assert m1 is m2

    def test_reset_creates_new_instance(self):
        m1 = get_prompt_manager()
        reset_prompt_manager()
        m2 = get_prompt_manager()
        assert m1 is not m2

    def test_reset_clears_data(self):
        m = get_prompt_manager()
        m.create_prompt(name="test", template="t")
        reset_prompt_manager()
        m2 = get_prompt_manager()
        assert m2.list_prompts() == []
