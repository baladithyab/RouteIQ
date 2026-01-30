"""
Tests for the Skills Discovery Plugin.

These tests verify:
1. Skills index endpoint returns correct format
2. SKILL.md endpoint returns markdown content
3. File endpoint returns skill files
4. Path traversal attacks are blocked
5. Name validation enforces constraints
6. Cache refresh works correctly
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


class TestSkillNameValidation:
    """Test skill name validation."""

    def test_valid_skill_names(self):
        """Test that valid skill names pass validation."""
        from litellm_llmrouter.gateway.plugins.skills_discovery import SkillsStore

        store = SkillsStore()

        valid_names = [
            "my-skill",
            "skill1",
            "a",
            "my-cool-skill-123",
            "abcdefghijklmnopqrstuvwxyz0123456789-abcdefghijklmnopqrstuvwxy",  # 64 chars
        ]

        for name in valid_names:
            assert store._validate_skill_name(name), f"Should be valid: {name}"

    def test_invalid_skill_names(self):
        """Test that invalid skill names fail validation."""
        from litellm_llmrouter.gateway.plugins.skills_discovery import SkillsStore

        store = SkillsStore()

        invalid_names = [
            "MySkill",  # uppercase
            "my_skill",  # underscore
            "1skill",  # starts with digit
            "-skill",  # starts with hyphen
            "",  # empty
            "a" * 65,  # too long (65 chars)
            "skill!",  # special char
            "skill.name",  # dot
            "skill/name",  # slash
            "../secret",  # path traversal attempt
        ]

        for name in invalid_names:
            assert not store._validate_skill_name(name), f"Should be invalid: {name}"


class TestPathTraversalProtection:
    """Test path traversal attack prevention."""

    def test_safe_path_within_base(self):
        """Test that paths within base are allowed."""
        from litellm_llmrouter.gateway.plugins.skills_discovery import SkillsStore

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            store = SkillsStore(base)

            # Create a test structure
            skill_dir = base / "my-skill"
            skill_dir.mkdir()
            (skill_dir / "file.txt").write_text("content")
            (skill_dir / "sub").mkdir()
            (skill_dir / "sub" / "nested.txt").write_text("nested")

            # These should all be safe
            assert store._is_safe_path(skill_dir, skill_dir / "file.txt")
            assert store._is_safe_path(skill_dir, skill_dir / "sub" / "nested.txt")
            assert store._is_safe_path(base, skill_dir)

    def test_unsafe_path_traversal(self):
        """Test that path traversal attempts are blocked."""
        from litellm_llmrouter.gateway.plugins.skills_discovery import SkillsStore

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            store = SkillsStore(base)

            # Create test structure
            (base / "my-skill").mkdir()
            (base / "secret").mkdir()
            (base / "secret" / "data.txt").write_text("secret")

            skill_dir = base / "my-skill"

            # These should all be blocked (path traversal attempts)
            assert not store._is_safe_path(skill_dir, base / "secret" / "data.txt")
            assert not store._is_safe_path(skill_dir, skill_dir / ".." / "secret")
            assert not store._is_safe_path(
                skill_dir, skill_dir / "sub" / ".." / ".." / "secret"
            )

    def test_read_file_blocks_traversal(self):
        """Test that read_file blocks path traversal."""
        from litellm_llmrouter.gateway.plugins.skills_discovery import SkillsStore

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            store = SkillsStore(base)

            # Create test structure
            skill_dir = base / "my-skill"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text("# My Skill")

            # Create a secret file outside skill directory
            (base / "secret.txt").write_text("secret data")

            # Normal access should work
            content = store.read_file("my-skill", "SKILL.md")
            assert content == b"# My Skill"

            # Path traversal should be blocked
            assert store.read_file("my-skill", "../secret.txt") is None
            assert store.read_file("my-skill", "../../etc/passwd") is None
            assert store.read_file("../other", "file.txt") is None


class TestSkillsStore:
    """Test the SkillsStore class."""

    def test_skills_dir_from_env(self):
        """Test that skills dir can be set via environment variable."""
        from litellm_llmrouter.gateway.plugins.skills_discovery import SkillsStore

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"ROUTEIQ_SKILLS_DIR": tmpdir}):
                store = SkillsStore()
                assert store.skills_dir == Path(tmpdir).resolve()

    def test_skills_dir_explicit_override(self):
        """Test that explicit skills dir takes precedence."""
        from litellm_llmrouter.gateway.plugins.skills_discovery import SkillsStore

        with tempfile.TemporaryDirectory() as tmpdir:
            override = Path(tmpdir)
            with patch.dict(os.environ, {"ROUTEIQ_SKILLS_DIR": "/some/other/path"}):
                store = SkillsStore(override)
                assert store.skills_dir == override.resolve()

    def test_scan_skills_finds_valid_skills(self):
        """Test that skill scanning finds valid skills."""
        from litellm_llmrouter.gateway.plugins.skills_discovery import SkillsStore

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)

            # Create a valid skill
            skill1 = base / "my-skill"
            skill1.mkdir()
            (skill1 / "SKILL.md").write_text(
                "# My Skill\n\nThis is a test skill for demo purposes."
            )
            (skill1 / "helper.py").write_text("# helper code")

            # Create another valid skill
            skill2 = base / "another-skill"
            skill2.mkdir()
            (skill2 / "SKILL.md").write_text("# Another Skill\n\nSecond description.")

            # Create invalid skill (no SKILL.md)
            invalid = base / "invalid-skill"
            invalid.mkdir()
            (invalid / "README.md").write_text("Not a skill")

            # Create invalid name
            bad_name = base / "BadName"
            bad_name.mkdir()
            (bad_name / "SKILL.md").write_text("# Bad Name")

            store = SkillsStore(base)
            index = store.get_index()

            # Should find 2 valid skills
            assert len(index) == 2
            names = {s["name"] for s in index}
            assert "my-skill" in names
            assert "another-skill" in names
            assert "invalid-skill" not in names
            assert "BadName" not in names

    def test_get_index_includes_files(self):
        """Test that index includes file listings."""
        from litellm_llmrouter.gateway.plugins.skills_discovery import SkillsStore

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)

            skill = base / "my-skill"
            skill.mkdir()
            (skill / "SKILL.md").write_text("# My Skill\n\nDescription here.")
            (skill / "helper.py").write_text("code")
            (skill / "sub").mkdir()
            (skill / "sub" / "nested.txt").write_text("nested")

            store = SkillsStore(base)
            index = store.get_index()

            assert len(index) == 1
            files = index[0]["files"]
            assert "SKILL.md" in files
            assert "helper.py" in files
            assert "sub/nested.txt" in files or "sub\\nested.txt" in files

    def test_description_extraction(self):
        """Test that descriptions are extracted from SKILL.md."""
        from litellm_llmrouter.gateway.plugins.skills_discovery import SkillsStore

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)

            skill = base / "my-skill"
            skill.mkdir()
            (skill / "SKILL.md").write_text(
                """# My Awesome Skill

This is the first paragraph that should become the description.
It spans multiple lines.

## Features

- Feature 1
- Feature 2
"""
            )

            store = SkillsStore(base)
            index = store.get_index()

            description = index[0]["description"]
            assert "first paragraph" in description
            assert "multiple lines" in description
            assert "## Features" not in description

    def test_cache_refresh(self):
        """Test that cache is refreshed when directory changes."""
        from litellm_llmrouter.gateway.plugins.skills_discovery import SkillsStore

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)

            # Create initial skill
            skill1 = base / "skill-one"
            skill1.mkdir()
            (skill1 / "SKILL.md").write_text("# Skill One")

            store = SkillsStore(base)
            index1 = store.get_index()
            assert len(index1) == 1

            # Force cache to be "old" by manipulating internal state
            if store._cache:
                store._cache.last_scan = 0

            # Add another skill
            skill2 = base / "skill-two"
            skill2.mkdir()
            (skill2 / "SKILL.md").write_text("# Skill Two")

            # Should refresh and find both
            index2 = store.get_index()
            assert len(index2) == 2


class TestSkillsDiscoveryRoutes:
    """Test the HTTP routes."""

    @pytest.fixture
    def app_with_plugin(self):
        """Create a FastAPI app with the skills discovery plugin."""
        from litellm_llmrouter.gateway.plugins.skills_discovery import (
            SkillsStore,
            create_skills_router,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)

            # Create test skill
            skill = base / "test-skill"
            skill.mkdir()
            (skill / "SKILL.md").write_text(
                "# Test Skill\n\nA skill for testing purposes."
            )
            (skill / "helper.py").write_text("# Helper code\nprint('hello')")
            (skill / "data").mkdir()
            (skill / "data" / "config.json").write_text('{"key": "value"}')

            store = SkillsStore(base)
            app = FastAPI()
            app.include_router(create_skills_router(store))

            yield app, base

    def test_get_index_json(self, app_with_plugin):
        """Test GET /.well-known/skills/index.json."""
        app, _ = app_with_plugin
        client = TestClient(app)

        response = client.get("/.well-known/skills/index.json")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")

        data = response.json()
        assert "skills" in data
        assert len(data["skills"]) == 1

        skill = data["skills"][0]
        assert skill["name"] == "test-skill"
        assert "description" in skill
        assert "files" in skill
        assert "SKILL.md" in skill["files"]

    def test_get_skill_md(self, app_with_plugin):
        """Test GET /.well-known/skills/{skill}/SKILL.md."""
        app, _ = app_with_plugin
        client = TestClient(app)

        response = client.get("/.well-known/skills/test-skill/SKILL.md")
        assert response.status_code == 200
        assert "text/markdown" in response.headers["content-type"]
        assert b"# Test Skill" in response.content

    def test_get_skill_md_not_found(self, app_with_plugin):
        """Test GET /.well-known/skills/{skill}/SKILL.md for nonexistent skill."""
        app, _ = app_with_plugin
        client = TestClient(app)

        response = client.get("/.well-known/skills/nonexistent/SKILL.md")
        assert response.status_code == 404

    def test_get_skill_file(self, app_with_plugin):
        """Test GET /.well-known/skills/{skill}/{path}."""
        app, _ = app_with_plugin
        client = TestClient(app)

        # Get Python file
        response = client.get("/.well-known/skills/test-skill/helper.py")
        assert response.status_code == 200
        assert "print('hello')" in response.text

        # Get nested JSON file
        response = client.get("/.well-known/skills/test-skill/data/config.json")
        assert response.status_code == 200
        assert response.json() == {"key": "value"}

    def test_get_file_traversal_blocked(self, app_with_plugin):
        """Test that path traversal is blocked in file endpoint."""
        app, base = app_with_plugin

        # Create a secret file outside skill dir
        (base / "secret.txt").write_text("secret data")

        client = TestClient(app)

        # Try various traversal attacks
        attacks = [
            "/.well-known/skills/test-skill/../secret.txt",
            "/.well-known/skills/test-skill/../../secret.txt",
            "/.well-known/skills/test-skill/data/../../secret.txt",
        ]

        for attack in attacks:
            response = client.get(attack)
            # Should either 404 or be sanitized
            assert response.status_code in (
                404,
                400,
            ), f"Traversal not blocked: {attack}"
            if response.status_code == 200:
                # If somehow 200, ensure it's not the secret
                assert b"secret data" not in response.content

    def test_invalid_skill_name_rejected(self, app_with_plugin):
        """Test that invalid skill names are rejected."""
        app, _ = app_with_plugin
        client = TestClient(app)

        invalid_names = [
            "BadName",
            "skill_with_underscore",
            "../etc",
            "skill%00name",
        ]

        for name in invalid_names:
            response = client.get(f"/.well-known/skills/{name}/SKILL.md")
            assert response.status_code == 404, f"Should reject: {name}"


class TestSkillsDiscoveryPlugin:
    """Test the plugin class."""

    @pytest.mark.asyncio
    async def test_plugin_metadata(self):
        """Test that plugin has correct metadata."""
        from litellm_llmrouter.gateway.plugins.skills_discovery import (
            SkillsDiscoveryPlugin,
        )
        from litellm_llmrouter.gateway.plugin_manager import PluginCapability

        plugin = SkillsDiscoveryPlugin()
        meta = plugin.metadata

        assert meta.name == "skills-discovery"
        assert meta.version == "1.0.0"
        assert PluginCapability.ROUTES in meta.capabilities

    @pytest.mark.asyncio
    async def test_plugin_startup_registers_routes(self):
        """Test that plugin startup registers routes."""
        from litellm_llmrouter.gateway.plugins.skills_discovery import (
            SkillsDiscoveryPlugin,
        )
        from litellm_llmrouter.gateway.plugin_manager import PluginContext

        with tempfile.TemporaryDirectory() as tmpdir:
            app = FastAPI()
            plugin = SkillsDiscoveryPlugin(Path(tmpdir))
            context = PluginContext()

            await plugin.startup(app, context)

            # Check that routes were registered
            routes = [r.path for r in app.routes]
            assert "/.well-known/skills/index.json" in routes
            assert "/.well-known/skills/{skill_name}/SKILL.md" in routes
            assert "/.well-known/skills/{skill_name}/{path:path}" in routes

    @pytest.mark.asyncio
    async def test_plugin_shutdown_cleanup(self):
        """Test that plugin shutdown cleans up."""
        from litellm_llmrouter.gateway.plugins.skills_discovery import (
            SkillsDiscoveryPlugin,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            app = FastAPI()
            plugin = SkillsDiscoveryPlugin(Path(tmpdir))

            await plugin.startup(app, None)
            assert plugin._store is not None

            await plugin.shutdown(app, None)
            assert plugin._store is None


class TestDescriptionExtraction:
    """Test description extraction edge cases."""

    def test_empty_skill_md(self):
        """Test handling of empty SKILL.md."""
        from litellm_llmrouter.gateway.plugins.skills_discovery import SkillsStore

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            skill = base / "empty-skill"
            skill.mkdir()
            (skill / "SKILL.md").write_text("")

            store = SkillsStore(base)
            index = store.get_index()

            assert len(index) == 1
            assert index[0]["description"] == "No description available"

    def test_header_only_skill_md(self):
        """Test SKILL.md with only header."""
        from litellm_llmrouter.gateway.plugins.skills_discovery import SkillsStore

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            skill = base / "header-only"
            skill.mkdir()
            (skill / "SKILL.md").write_text("# Just a Header\n")

            store = SkillsStore(base)
            index = store.get_index()

            assert len(index) == 1
            assert index[0]["description"] == "No description available"

    def test_long_description_truncated(self):
        """Test that long descriptions are truncated."""
        from litellm_llmrouter.gateway.plugins.skills_discovery import SkillsStore

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            skill = base / "long-desc"
            skill.mkdir()
            # Create a description longer than 200 chars
            long_text = "This is a very long description. " * 20
            (skill / "SKILL.md").write_text(f"# Long Skill\n\n{long_text}")

            store = SkillsStore(base)
            index = store.get_index()

            description = index[0]["description"]
            assert len(description) <= 200
            assert description.endswith("...")
