"""Prompt Management & Versioning for RouteIQ Gateway.

Enables deploying, versioning, and A/B testing prompts without code changes.
Prompts are named templates with variables, stored in-memory with optional
Redis persistence for multi-worker deployments.

Features:
- Named prompt templates with Jinja2-style variable substitution
- Version history with rollback capability
- A/B testing: split traffic between prompt versions
- Per-workspace prompt isolation
- Audit logging of prompt changes
- Import/export (JSON format)

Usage:
  # Create a prompt
  POST /api/v1/routeiq/prompts
  {
    "name": "code-review",
    "template": "Review this {{language}} code for bugs:\n\n{{code}}",
    "model": "gpt-4o",
    "metadata": {"team": "engineering", "category": "dev-tools"}
  }

  # Use in a request via header or metadata
  POST /v1/chat/completions
  Headers: X-RouteIQ-Prompt: code-review
  Body: {
    "messages": [...],
    "metadata": {
      "_prompt": "code-review",
      "_prompt_vars": {"language": "python", "code": "..."}
    }
  }

Configuration:
  ROUTEIQ_PROMPT_MANAGEMENT=true
"""

import copy
import json
import logging
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger("litellm_llmrouter.prompt_management")

# ============================================================================
# Name validation
# ============================================================================

_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9\-]{0,62}[a-z0-9]$|^[a-z0-9]$")


def _validate_prompt_name(name: str) -> str:
    """Validate prompt name: lowercase alphanumeric with hyphens, 1-64 chars."""
    if not name or not _NAME_PATTERN.match(name):
        raise ValueError(
            f"Invalid prompt name '{name}'. "
            "Must be 1-64 chars, lowercase alphanumeric with hyphens, "
            "cannot start/end with hyphen."
        )
    return name


# ============================================================================
# Models
# ============================================================================


class PromptVersion(BaseModel):
    """A specific version of a prompt template."""

    version: int
    template: str  # The prompt template with {{variable}} placeholders
    system_template: Optional[str] = None  # Optional system message template
    model: Optional[str] = None  # Pin to specific model (overrides routing)
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    created_at: float = 0.0
    created_by: Optional[str] = None
    change_note: str = ""


class PromptDefinition(BaseModel):
    """A named prompt with version history."""

    name: str  # Unique identifier (slug format: lowercase, hyphens)
    description: str = ""
    workspace_id: Optional[str] = None

    # Current active version
    active_version: int = 1

    # A/B testing
    ab_enabled: bool = False
    ab_versions: Dict[int, float] = Field(default_factory=dict)
    # e.g., {1: 0.9, 2: 0.1} — 90% v1, 10% v2

    # Version history
    versions: Dict[int, PromptVersion] = Field(default_factory=dict)

    # Tags for organization
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Timestamps
    created_at: float = 0.0
    updated_at: float = 0.0


# ============================================================================
# Request/response models for API routes
# ============================================================================


class CreatePromptRequest(BaseModel):
    """Request body for creating a new prompt."""

    name: str
    template: str
    system_template: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    description: str = ""
    workspace_id: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UpdatePromptRequest(BaseModel):
    """Request body for updating (new version of) an existing prompt."""

    template: str
    system_template: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    change_note: str = ""
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class RollbackRequest(BaseModel):
    """Request body for rolling back to a specific version."""

    version: int


class ABTestRequest(BaseModel):
    """Request body for configuring an A/B test."""

    versions: Dict[int, float]
    # e.g., {1: 0.9, 2: 0.1}


class ABTestStopRequest(BaseModel):
    """Request body for stopping an A/B test."""

    winner: Optional[int] = None


class ImportPromptsRequest(BaseModel):
    """Request body for importing prompts."""

    prompts: List[Dict[str, Any]]
    workspace_id: Optional[str] = None


# ============================================================================
# Prompt Manager
# ============================================================================


class PromptManager:
    """Manages prompt templates with versioning and A/B testing."""

    def __init__(self) -> None:
        self._prompts: Dict[str, PromptDefinition] = {}  # name -> definition
        self._var_pattern = re.compile(r"\{\{(\w+)\}\}")

    # ── CRUD ──────────────────────────────────────────────

    def create_prompt(
        self,
        name: str,
        template: str,
        system_template: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        description: str = "",
        workspace_id: Optional[str] = None,
        created_by: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PromptDefinition:
        """Create a new named prompt.

        Raises ValueError if name is invalid or already exists.
        """
        name = _validate_prompt_name(name)

        key = self._storage_key(name, workspace_id)
        if key in self._prompts:
            raise ValueError(f"Prompt '{name}' already exists")

        now = time.time()
        version = PromptVersion(
            version=1,
            template=template,
            system_template=system_template,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            created_at=now,
            created_by=created_by,
            change_note="Initial version",
        )

        prompt = PromptDefinition(
            name=name,
            description=description,
            workspace_id=workspace_id,
            active_version=1,
            versions={1: version},
            tags=tags or [],
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
        )

        self._prompts[key] = prompt
        logger.info(
            "Created prompt '%s' (workspace=%s, by=%s)",
            name,
            workspace_id,
            created_by,
        )
        return copy.deepcopy(prompt)

    def update_prompt(
        self,
        name: str,
        template: str,
        system_template: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        change_note: str = "",
        created_by: Optional[str] = None,
        workspace_id: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PromptDefinition:
        """Create a new version of an existing prompt.

        Raises ValueError if prompt does not exist.
        """
        key = self._storage_key(name, workspace_id)
        prompt = self._prompts.get(key)
        if prompt is None:
            raise ValueError(f"Prompt '{name}' not found")

        now = time.time()
        new_version_num = max(prompt.versions.keys()) + 1

        version = PromptVersion(
            version=new_version_num,
            template=template,
            system_template=system_template,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            created_at=now,
            created_by=created_by,
            change_note=change_note,
        )

        prompt.versions[new_version_num] = version
        prompt.active_version = new_version_num
        prompt.updated_at = now

        # Update optional metadata fields if provided
        if description is not None:
            prompt.description = description
        if tags is not None:
            prompt.tags = tags
        if metadata is not None:
            prompt.metadata = metadata

        logger.info(
            "Updated prompt '%s' to v%d (by=%s, note='%s')",
            name,
            new_version_num,
            created_by,
            change_note,
        )
        return copy.deepcopy(prompt)

    def get_prompt(
        self, name: str, workspace_id: Optional[str] = None
    ) -> Optional[PromptDefinition]:
        """Get a prompt definition by name."""
        key = self._storage_key(name, workspace_id)
        prompt = self._prompts.get(key)
        if prompt is not None:
            return copy.deepcopy(prompt)
        return None

    def list_prompts(
        self,
        workspace_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[PromptDefinition]:
        """List all prompts, optionally filtered by workspace and/or tags."""
        results: List[PromptDefinition] = []
        for prompt in self._prompts.values():
            # Workspace filter
            if workspace_id is not None and prompt.workspace_id != workspace_id:
                continue
            # Tag filter (all specified tags must be present)
            if tags:
                if not all(t in prompt.tags for t in tags):
                    continue
            results.append(copy.deepcopy(prompt))

        results.sort(key=lambda p: p.name)
        return results

    def delete_prompt(self, name: str, workspace_id: Optional[str] = None) -> bool:
        """Delete a prompt and all its versions. Returns True if deleted."""
        key = self._storage_key(name, workspace_id)
        if key in self._prompts:
            del self._prompts[key]
            logger.info("Deleted prompt '%s' (workspace=%s)", name, workspace_id)
            return True
        return False

    # ── Version Management ────────────────────────────────

    def rollback(
        self, name: str, version: int, workspace_id: Optional[str] = None
    ) -> PromptDefinition:
        """Set active version to a previous version.

        Raises ValueError if prompt or version doesn't exist.
        """
        key = self._storage_key(name, workspace_id)
        prompt = self._prompts.get(key)
        if prompt is None:
            raise ValueError(f"Prompt '{name}' not found")

        if version not in prompt.versions:
            raise ValueError(
                f"Version {version} not found for prompt '{name}'. "
                f"Available: {sorted(prompt.versions.keys())}"
            )

        prompt.active_version = version
        prompt.updated_at = time.time()
        logger.info("Rolled back prompt '%s' to v%d", name, version)
        return copy.deepcopy(prompt)

    def set_ab_test(
        self,
        name: str,
        versions: Dict[int, float],
        workspace_id: Optional[str] = None,
    ) -> PromptDefinition:
        """Configure A/B testing between prompt versions.

        versions: {version_number: weight} e.g., {1: 0.9, 2: 0.1}
        Weights must sum to ~1.0 (tolerance: 0.01).
        Raises ValueError on invalid configuration.
        """
        key = self._storage_key(name, workspace_id)
        prompt = self._prompts.get(key)
        if prompt is None:
            raise ValueError(f"Prompt '{name}' not found")

        if not versions:
            raise ValueError("A/B test must have at least one version")

        # Validate all referenced versions exist
        for ver in versions:
            if ver not in prompt.versions:
                raise ValueError(
                    f"Version {ver} not found for prompt '{name}'. "
                    f"Available: {sorted(prompt.versions.keys())}"
                )

        # Validate weights
        for ver, weight in versions.items():
            if weight < 0 or weight > 1:
                raise ValueError(
                    f"Weight for version {ver} must be between 0 and 1, got {weight}"
                )

        weight_sum = sum(versions.values())
        if abs(weight_sum - 1.0) > 0.01:
            raise ValueError(f"A/B test weights must sum to ~1.0, got {weight_sum:.4f}")

        prompt.ab_enabled = True
        prompt.ab_versions = dict(versions)
        prompt.updated_at = time.time()
        logger.info(
            "Started A/B test for prompt '%s': %s",
            name,
            {f"v{k}": f"{v:.0%}" for k, v in versions.items()},
        )
        return copy.deepcopy(prompt)

    def stop_ab_test(
        self,
        name: str,
        winner: Optional[int] = None,
        workspace_id: Optional[str] = None,
    ) -> PromptDefinition:
        """Stop A/B test. If winner specified, promote that version.

        Raises ValueError if prompt doesn't exist or winner version is invalid.
        """
        key = self._storage_key(name, workspace_id)
        prompt = self._prompts.get(key)
        if prompt is None:
            raise ValueError(f"Prompt '{name}' not found")

        if winner is not None:
            if winner not in prompt.versions:
                raise ValueError(
                    f"Winner version {winner} not found for prompt '{name}'"
                )
            prompt.active_version = winner

        prompt.ab_enabled = False
        prompt.ab_versions = {}
        prompt.updated_at = time.time()
        logger.info(
            "Stopped A/B test for prompt '%s' (winner=v%s)",
            name,
            winner if winner is not None else "none",
        )
        return copy.deepcopy(prompt)

    # ── Resolution ────────────────────────────────────────

    def resolve_prompt(
        self,
        name: str,
        variables: Optional[Dict[str, str]] = None,
        workspace_id: Optional[str] = None,
    ) -> Tuple[List[Dict[str, str]], Optional[str], Dict[str, Any]]:
        """Resolve a named prompt into messages + optional model override.

        Returns: (messages, model_override, metadata)
        - messages: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        - model_override: model name if pinned, None otherwise
        - metadata: {"prompt_name": "...", "prompt_version": N, "ab_variant": "..."}

        Raises ValueError if prompt or selected version is not found.
        """
        prompt = self.get_prompt(name, workspace_id)
        if prompt is None:
            raise ValueError(f"Prompt not found: {name}")

        # Select version (A/B testing or active)
        version_num = self._select_version(prompt)
        version = prompt.versions.get(version_num)
        if version is None:
            raise ValueError(f"Version {version_num} not found for prompt {name}")

        # Render templates with variables
        vars_ = variables or {}
        messages: List[Dict[str, str]] = []
        if version.system_template:
            system_content = self._render(version.system_template, vars_)
            messages.append({"role": "system", "content": system_content})

        user_content = self._render(version.template, vars_)
        messages.append({"role": "user", "content": user_content})

        model_override = version.model

        meta: Dict[str, Any] = {
            "prompt_name": name,
            "prompt_version": version_num,
            "ab_variant": f"v{version_num}" if prompt.ab_enabled else None,
        }

        # Include temperature/max_tokens overrides if set
        if version.temperature is not None:
            meta["temperature"] = version.temperature
        if version.max_tokens is not None:
            meta["max_tokens"] = version.max_tokens

        return messages, model_override, meta

    def _select_version(self, prompt: PromptDefinition) -> int:
        """Select version based on A/B test weights or active version."""
        if not prompt.ab_enabled or not prompt.ab_versions:
            return prompt.active_version

        r = random.random()
        cumulative = 0.0
        for ver, weight in sorted(prompt.ab_versions.items()):
            cumulative += weight
            if r < cumulative:
                return ver
        # Fallback to active version (shouldn't reach here with valid weights)
        return prompt.active_version

    def _render(self, template: str, variables: Dict[str, str]) -> str:
        """Render a template with {{variable}} substitution.

        Unresolved variables are preserved as-is (e.g., {{unknown}} stays).
        """

        def replace(match: re.Match) -> str:
            var_name = match.group(1)
            return str(variables.get(var_name, f"{{{{{var_name}}}}}"))

        return self._var_pattern.sub(replace, template)

    # ── Import/Export ─────────────────────────────────────

    def export_prompts(self, workspace_id: Optional[str] = None) -> str:
        """Export all prompts as JSON string."""
        prompts = self.list_prompts(workspace_id=workspace_id)
        return json.dumps([p.model_dump() for p in prompts], indent=2)

    def import_prompts(
        self, data: List[Dict[str, Any]], workspace_id: Optional[str] = None
    ) -> int:
        """Import prompts from a list of prompt dicts. Returns count imported.

        Existing prompts with the same name are overwritten.
        If workspace_id is provided, it overrides the workspace in the data.
        """
        count = 0
        for item in data:
            try:
                prompt = PromptDefinition.model_validate(item)
                if workspace_id is not None:
                    prompt.workspace_id = workspace_id

                # Reconstruct versions dict with int keys (JSON may have string keys)
                fixed_versions: Dict[int, PromptVersion] = {}
                for k, v in prompt.versions.items():
                    int_key = int(k)
                    if isinstance(v, dict):
                        fixed_versions[int_key] = PromptVersion.model_validate(v)
                    else:
                        fixed_versions[int_key] = v
                prompt.versions = fixed_versions

                # Fix ab_versions keys too
                if prompt.ab_versions:
                    prompt.ab_versions = {
                        int(k): v for k, v in prompt.ab_versions.items()
                    }

                key = self._storage_key(prompt.name, prompt.workspace_id)
                self._prompts[key] = prompt
                count += 1
            except Exception as exc:
                logger.warning(
                    "Skipping invalid prompt during import: %s",
                    exc,
                )
        logger.info("Imported %d prompts (workspace=%s)", count, workspace_id)
        return count

    # ── Internal ──────────────────────────────────────────

    def _storage_key(self, name: str, workspace_id: Optional[str] = None) -> str:
        """Create a composite storage key for workspace-scoped prompts."""
        if workspace_id:
            return f"{workspace_id}::{name}"
        return name


# ============================================================================
# Singleton
# ============================================================================

_manager: Optional[PromptManager] = None


def get_prompt_manager() -> PromptManager:
    """Get or create the singleton PromptManager."""
    global _manager
    if _manager is None:
        _manager = PromptManager()
    return _manager


def reset_prompt_manager() -> None:
    """Reset the singleton PromptManager. Used in tests."""
    global _manager
    _manager = None


def is_prompt_management_enabled() -> bool:
    """Check if prompt management feature is enabled."""
    try:
        from litellm_llmrouter.settings import get_settings

        return get_settings().prompt_management.enabled
    except Exception:
        return os.getenv("ROUTEIQ_PROMPT_MANAGEMENT", "false").lower() in (
            "true",
            "1",
            "yes",
        )
