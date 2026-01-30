"""
Skills Discovery Plugin for Well-Known Skills Index.

This plugin implements a discovery surface for skills compatible with the
well-known index pattern (e.g., `/.well-known/skills/index.json`) with
progressive disclosure.

Endpoints:
    - GET /.well-known/skills/index.json → returns list of skills
    - GET /.well-known/skills/{skill}/SKILL.md → returns markdown body
    - GET /.well-known/skills/{skill}/{path} → returns referenced files

Configuration:
    - ROUTEIQ_SKILLS_DIR: Directory to load skills from (default: ./skills or ./docs/skills)

Usage:
    export LLMROUTER_PLUGINS=litellm_llmrouter.gateway.plugins.skills_discovery.SkillsDiscoveryPlugin
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import JSONResponse

from litellm_llmrouter.gateway.plugin_manager import (
    GatewayPlugin,
    PluginCapability,
    PluginContext,
    PluginMetadata,
)

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

# Skill name validation pattern: lowercase, hyphens, 1-64 chars
SKILL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9-]{0,63}$")

# Cache refresh interval (seconds)
CACHE_REFRESH_SECONDS = 5.0


@dataclass
class SkillInfo:
    """Information about a discovered skill."""

    name: str
    description: str
    files: list[str]
    mtime: float


@dataclass
class SkillsCache:
    """In-memory cache for skills index."""

    skills: dict[str, SkillInfo]
    last_scan: float
    dir_mtime: float


class SkillsStore:
    """
    Thread-safe store for loading and caching skills from a directory.

    Skills are loaded from a configurable directory (ROUTEIQ_SKILLS_DIR).
    Each skill is a subdirectory containing at minimum a SKILL.md file.

    The store caches the index and refreshes based on directory mtime.
    """

    def __init__(self, skills_dir: Path | None = None) -> None:
        """
        Initialize the skills store.

        Args:
            skills_dir: Optional override for skills directory. If None, uses
                       ROUTEIQ_SKILLS_DIR env var or defaults to ./skills.
        """
        self._skills_dir = self._resolve_skills_dir(skills_dir)
        self._cache: SkillsCache | None = None
        logger.info(f"Skills store initialized with directory: {self._skills_dir}")

    def _resolve_skills_dir(self, override: Path | None) -> Path:
        """Resolve the skills directory from config or defaults."""
        if override:
            return override.resolve()

        # Check env var
        env_dir = os.getenv("ROUTEIQ_SKILLS_DIR")
        if env_dir:
            return Path(env_dir).resolve()

        # Check default locations
        cwd = Path.cwd()
        for default in ["skills", "docs/skills"]:
            candidate = cwd / default
            if candidate.exists() and candidate.is_dir():
                return candidate.resolve()

        # Fall back to ./skills (will be created if needed)
        return (cwd / "skills").resolve()

    @property
    def skills_dir(self) -> Path:
        """Return the skills directory path."""
        return self._skills_dir

    def _validate_skill_name(self, name: str) -> bool:
        """
        Validate skill name against constraints.

        Requirements:
        - Lowercase letters, digits, hyphens only
        - Must start with a letter
        - 1-64 characters
        """
        return bool(SKILL_NAME_PATTERN.match(name))

    def _is_safe_path(self, base: Path, target: Path) -> bool:
        """
        Check if target path is safely within base directory.

        Prevents path traversal attacks by ensuring the resolved path
        is a descendant of the base directory.
        """
        try:
            resolved_base = base.resolve()
            resolved_target = target.resolve()
            # Check that target starts with base path
            return (
                str(resolved_target).startswith(str(resolved_base) + os.sep)
                or resolved_target == resolved_base
            )
        except (OSError, ValueError):
            return False

    def _get_dir_mtime(self) -> float:
        """Get the modification time of the skills directory."""
        try:
            if self._skills_dir.exists():
                return self._skills_dir.stat().st_mtime
        except OSError:
            pass
        return 0.0

    def _should_refresh_cache(self) -> bool:
        """Check if the cache needs refreshing."""
        if self._cache is None:
            return True

        now = time.monotonic()
        if now - self._cache.last_scan < CACHE_REFRESH_SECONDS:
            return False

        # Check if directory has been modified
        current_mtime = self._get_dir_mtime()
        return current_mtime > self._cache.dir_mtime

    def _scan_skills(self) -> dict[str, SkillInfo]:
        """Scan the skills directory for valid skills."""
        skills: dict[str, SkillInfo] = {}

        if not self._skills_dir.exists():
            logger.warning(f"Skills directory does not exist: {self._skills_dir}")
            return skills

        for entry in self._skills_dir.iterdir():
            if not entry.is_dir():
                continue

            skill_name = entry.name
            if not self._validate_skill_name(skill_name):
                logger.debug(f"Skipping invalid skill name: {skill_name}")
                continue

            # Check for SKILL.md
            skill_md = entry / "SKILL.md"
            if not skill_md.exists():
                logger.debug(f"Skipping skill without SKILL.md: {skill_name}")
                continue

            # Parse skill info
            try:
                skill_info = self._parse_skill(entry, skill_md)
                skills[skill_name] = skill_info
            except Exception as e:
                logger.warning(f"Failed to parse skill {skill_name}: {e}")

        return skills

    def _parse_skill(self, skill_dir: Path, skill_md: Path) -> SkillInfo:
        """Parse skill information from SKILL.md."""
        content = skill_md.read_text(encoding="utf-8")

        # Extract description (first paragraph after title or first line)
        description = self._extract_description(content)

        # List all files in the skill directory
        files = []
        for file_path in skill_dir.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(skill_dir)
                files.append(str(rel_path))

        return SkillInfo(
            name=skill_dir.name,
            description=description,
            files=sorted(files),
            mtime=skill_md.stat().st_mtime,
        )

    def _extract_description(self, content: str) -> str:
        """Extract description from markdown content."""
        lines = content.strip().split("\n")
        description_lines = []

        in_header = True
        for line in lines:
            stripped = line.strip()

            # Skip empty lines at start
            if not stripped and in_header:
                continue

            # Skip markdown headers
            if stripped.startswith("#"):
                in_header = False
                continue

            if in_header:
                in_header = False

            # Stop at next header or empty line after content
            if stripped.startswith("#") or (not stripped and description_lines):
                break

            if stripped:
                description_lines.append(stripped)

        description = " ".join(description_lines)
        # Truncate to reasonable length
        if len(description) > 200:
            description = description[:197] + "..."

        return description or "No description available"

    def get_index(self) -> list[dict[str, Any]]:
        """
        Get the skills index.

        Returns a list of skill objects with name, description, and files.
        Uses cached data if available and not stale.
        """
        if self._should_refresh_cache():
            skills = self._scan_skills()
            self._cache = SkillsCache(
                skills=skills,
                last_scan=time.monotonic(),
                dir_mtime=self._get_dir_mtime(),
            )
            logger.debug(f"Skills cache refreshed, found {len(skills)} skills")

        return [
            {
                "name": skill.name,
                "description": skill.description,
                "files": skill.files,
            }
            for skill in sorted(self._cache.skills.values(), key=lambda s: s.name)
        ]

    def get_skill(self, skill_name: str) -> SkillInfo | None:
        """Get information about a specific skill."""
        # Refresh cache if needed
        self.get_index()

        if self._cache is None:
            return None

        return self._cache.skills.get(skill_name)

    def read_file(self, skill_name: str, file_path: str) -> bytes | None:
        """
        Read a file from a skill directory.

        Args:
            skill_name: Name of the skill
            file_path: Relative path to the file within the skill directory

        Returns:
            File contents as bytes, or None if not found or access denied
        """
        # Validate skill name
        if not self._validate_skill_name(skill_name):
            logger.warning(f"Invalid skill name requested: {skill_name}")
            return None

        skill_dir = self._skills_dir / skill_name
        target_file = skill_dir / file_path

        # Security check: ensure path is within skill directory
        if not self._is_safe_path(skill_dir, target_file):
            logger.warning(
                f"Path traversal attempt blocked: skill={skill_name}, path={file_path}"
            )
            return None

        if not target_file.exists() or not target_file.is_file():
            return None

        try:
            return target_file.read_bytes()
        except OSError as e:
            logger.warning(f"Failed to read file {target_file}: {e}")
            return None


def create_skills_router(store: SkillsStore) -> APIRouter:
    """
    Create the FastAPI router for skills discovery endpoints.

    Args:
        store: The skills store to use for loading skills

    Returns:
        Configured APIRouter
    """
    router = APIRouter(tags=["Skills Discovery"])

    @router.get("/.well-known/skills/index.json")
    async def get_skills_index() -> JSONResponse:
        """
        Get the skills index.

        Returns a JSON array of skill objects, each containing:
        - name: Skill identifier (lowercase, hyphenated)
        - description: Brief description extracted from SKILL.md
        - files: List of files in the skill directory
        """
        skills = store.get_index()
        return JSONResponse(
            content={"skills": skills},
            headers={
                "Cache-Control": "public, max-age=60",
            },
        )

    @router.get("/.well-known/skills/{skill_name}/SKILL.md")
    async def get_skill_markdown(skill_name: str) -> Response:
        """
        Get the SKILL.md file for a specific skill.

        Args:
            skill_name: The skill identifier

        Returns:
            The markdown content of SKILL.md
        """
        content = store.read_file(skill_name, "SKILL.md")
        if content is None:
            raise HTTPException(
                status_code=404, detail=f"Skill not found: {skill_name}"
            )

        return Response(
            content=content,
            media_type="text/markdown; charset=utf-8",
            headers={
                "Cache-Control": "public, max-age=300",
            },
        )

    @router.get("/.well-known/skills/{skill_name}/{path:path}")
    async def get_skill_file(skill_name: str, path: str) -> Response:
        """
        Get a file from a skill directory.

        Args:
            skill_name: The skill identifier
            path: Path to the file relative to the skill directory

        Returns:
            The file contents with appropriate content type
        """
        content = store.read_file(skill_name, path)
        if content is None:
            raise HTTPException(
                status_code=404, detail=f"File not found: {skill_name}/{path}"
            )

        # Determine content type based on extension
        media_type = _get_media_type(path)

        return Response(
            content=content,
            media_type=media_type,
            headers={
                "Cache-Control": "public, max-age=300",
            },
        )

    return router


def _get_media_type(path: str) -> str:
    """Get MIME type for a file path."""
    ext = Path(path).suffix.lower()
    media_types = {
        ".md": "text/markdown; charset=utf-8",
        ".txt": "text/plain; charset=utf-8",
        ".json": "application/json",
        ".yaml": "application/x-yaml",
        ".yml": "application/x-yaml",
        ".py": "text/x-python; charset=utf-8",
        ".js": "text/javascript; charset=utf-8",
        ".ts": "text/typescript; charset=utf-8",
        ".html": "text/html; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".svg": "image/svg+xml",
    }
    return media_types.get(ext, "application/octet-stream")


class SkillsDiscoveryPlugin(GatewayPlugin):
    """
    Plugin that provides well-known skills discovery endpoints.

    This plugin registers routes for:
    - GET /.well-known/skills/index.json
    - GET /.well-known/skills/{skill}/SKILL.md
    - GET /.well-known/skills/{skill}/{path}

    Enable by adding to LLMROUTER_PLUGINS:
        export LLMROUTER_PLUGINS=litellm_llmrouter.gateway.plugins.skills_discovery.SkillsDiscoveryPlugin
    """

    def __init__(self, skills_dir: Path | None = None) -> None:
        """
        Initialize the plugin.

        Args:
            skills_dir: Optional override for skills directory
        """
        self._skills_dir = skills_dir
        self._store: SkillsStore | None = None

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="skills-discovery",
            version="1.0.0",
            capabilities={PluginCapability.ROUTES},
            priority=500,  # Load before most user plugins
            description="Well-known skills discovery endpoints (/.well-known/skills/)",
        )

    async def startup(
        self, app: "FastAPI", context: PluginContext | None = None
    ) -> None:
        """
        Register skills discovery routes with the app.

        Args:
            app: The FastAPI application instance
            context: Plugin context (optional)
        """
        self._store = SkillsStore(self._skills_dir)
        router = create_skills_router(self._store)
        app.include_router(router)

        if context and context.logger:
            context.logger.info(
                f"Skills discovery plugin enabled, "
                f"serving from: {self._store.skills_dir}"
            )
        else:
            logger.info(
                f"Skills discovery plugin enabled, "
                f"serving from: {self._store.skills_dir}"
            )

    async def shutdown(
        self, app: "FastAPI", context: PluginContext | None = None
    ) -> None:
        """
        Cleanup on shutdown.

        Args:
            app: The FastAPI application instance
            context: Plugin context (optional)
        """
        self._store = None
        logger.debug("Skills discovery plugin shutdown complete")
