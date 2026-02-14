"""
Management Endpoint Classifier
===============================

Maps LiteLLM management endpoint paths to structured operation metadata.
This enables the management middleware to apply RBAC, audit, and telemetry
without hard-coding individual endpoint logic.

Classification structure:
- operation: Dotted operation name (e.g., "key.generate", "team.create")
- resource_type: Category of resource (e.g., "key", "team", "model")
- sensitivity: "read" or "write" (for RBAC permission mapping)

Unknown paths return None (middleware skips classification for non-management
paths like /chat/completions, /embeddings, /_health/*, etc.).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ManagementOperation:
    """Classified management operation."""

    operation: str
    """Dotted operation name (e.g., 'key.generate')."""

    resource_type: str
    """Resource category (e.g., 'key', 'team')."""

    sensitivity: str
    """'read' or 'write'."""


# =============================================================================
# Classification Table
# =============================================================================

# Mapping: (method, path_pattern) -> ManagementOperation
# path_pattern is a regex string. Method is uppercase.
# Order matters: first match wins.

_RULES: list[tuple[str, str, ManagementOperation]] = [
    # ── Key Management ──
    ("POST", r"^/key/generate$", ManagementOperation("key.generate", "key", "write")),
    (
        "POST",
        r"^/key/regenerate$",
        ManagementOperation("key.regenerate", "key", "write"),
    ),
    ("POST", r"^/key/update$", ManagementOperation("key.update", "key", "write")),
    ("POST", r"^/key/delete$", ManagementOperation("key.delete", "key", "write")),
    ("POST", r"^/key/block$", ManagementOperation("key.block", "key", "write")),
    ("POST", r"^/key/unblock$", ManagementOperation("key.unblock", "key", "write")),
    ("GET", r"^/key/info$", ManagementOperation("key.info", "key", "read")),
    ("GET", r"^/key/aliases$", ManagementOperation("key.list_aliases", "key", "read")),
    ("GET", r"^/keys$", ManagementOperation("key.list", "key", "read")),
    ("POST", r"^/key/health$", ManagementOperation("key.health", "key", "read")),
    # ── Team Management ──
    ("POST", r"^/team/new$", ManagementOperation("team.create", "team", "write")),
    ("POST", r"^/team/update$", ManagementOperation("team.update", "team", "write")),
    ("POST", r"^/team/delete$", ManagementOperation("team.delete", "team", "write")),
    ("POST", r"^/team/block$", ManagementOperation("team.block", "team", "write")),
    ("POST", r"^/team/unblock$", ManagementOperation("team.unblock", "team", "write")),
    ("GET", r"^/team/info$", ManagementOperation("team.info", "team", "read")),
    ("GET", r"^/team$", ManagementOperation("team.list", "team", "read")),
    (
        "POST",
        r"^/team/member/add$",
        ManagementOperation("team.member_add", "team", "write"),
    ),
    (
        "POST",
        r"^/team/member/delete$",
        ManagementOperation("team.member_delete", "team", "write"),
    ),
    (
        "POST",
        r"^/team/member/update$",
        ManagementOperation("team.member_update", "team", "write"),
    ),
    (
        "POST",
        r"^/team/model/add$",
        ManagementOperation("team.model_add", "team", "write"),
    ),
    (
        "POST",
        r"^/team/model/delete$",
        ManagementOperation("team.model_delete", "team", "write"),
    ),
    # ── Organization Management ──
    (
        "POST",
        r"^/organization/new$",
        ManagementOperation("organization.create", "organization", "write"),
    ),
    (
        "PATCH",
        r"^/organization/update$",
        ManagementOperation("organization.update", "organization", "write"),
    ),
    (
        "DELETE",
        r"^/organization/[^/]+$",
        ManagementOperation("organization.delete", "organization", "write"),
    ),
    (
        "GET",
        r"^/organization/[^/]+$",
        ManagementOperation("organization.info", "organization", "read"),
    ),
    (
        "GET",
        r"^/organization$",
        ManagementOperation("organization.list", "organization", "read"),
    ),
    # ── Model Management ──
    ("POST", r"^/model/new$", ManagementOperation("model.add", "model", "write")),
    ("POST", r"^/model/update$", ManagementOperation("model.update", "model", "write")),
    ("POST", r"^/model/delete$", ManagementOperation("model.delete", "model", "write")),
    ("GET", r"^/model/info$", ManagementOperation("model.info", "model", "read")),
    ("GET", r"^/models$", ManagementOperation("model.list", "model", "read")),
    ("GET", r"^/v1/models$", ManagementOperation("model.list", "model", "read")),
    (
        "GET",
        r"^/model_group/info$",
        ManagementOperation("model.group_info", "model", "read"),
    ),
    # ── Budget Management ──
    ("POST", r"^/budgets$", ManagementOperation("budget.create", "budget", "write")),
    (
        "POST",
        r"^/budgets/update$",
        ManagementOperation("budget.update", "budget", "write"),
    ),
    (
        "POST",
        r"^/budgets/delete$",
        ManagementOperation("budget.delete", "budget", "write"),
    ),
    ("POST", r"^/budgets/info$", ManagementOperation("budget.info", "budget", "read")),
    ("GET", r"^/budgets$", ManagementOperation("budget.list", "budget", "read")),
    (
        "GET",
        r"^/budgets/settings$",
        ManagementOperation("budget.settings", "budget", "read"),
    ),
    # ── Spend Tracking ──
    ("GET", r"^/spend/", ManagementOperation("spend.read", "spend", "read")),
    ("POST", r"^/spend/", ManagementOperation("spend.write", "spend", "write")),
    # ── Credential Management ──
    (
        "POST",
        r"^/credentials$",
        ManagementOperation("credential.create", "credential", "write"),
    ),
    (
        "GET",
        r"^/credentials$",
        ManagementOperation("credential.list", "credential", "read"),
    ),
    (
        "GET",
        r"^/credentials/[^/]+$",
        ManagementOperation("credential.info", "credential", "read"),
    ),
    (
        "DELETE",
        r"^/credentials/[^/]+$",
        ManagementOperation("credential.delete", "credential", "write"),
    ),
    (
        "PATCH",
        r"^/credentials/[^/]+$",
        ManagementOperation("credential.update", "credential", "write"),
    ),
    # ── Guardrails ──
    (
        "GET",
        r"^/guardrails$",
        ManagementOperation("guardrail.list", "guardrail", "read"),
    ),
    (
        "POST",
        r"^/guardrails/new$",
        ManagementOperation("guardrail.create", "guardrail", "write"),
    ),
    (
        "PUT",
        r"^/guardrails/[^/]+$",
        ManagementOperation("guardrail.update", "guardrail", "write"),
    ),
    (
        "DELETE",
        r"^/guardrails/[^/]+$",
        ManagementOperation("guardrail.delete", "guardrail", "write"),
    ),
    (
        "GET",
        r"^/guardrails/[^/]+$",
        ManagementOperation("guardrail.info", "guardrail", "read"),
    ),
    # ── Cache Settings ──
    (
        "GET",
        r"^/cache/settings$",
        ManagementOperation("cache.settings", "cache", "read"),
    ),
    (
        "POST",
        r"^/cache/settings/",
        ManagementOperation("cache.settings_write", "cache", "write"),
    ),
    # ── Callbacks ──
    (
        "GET",
        r"^/get/config/callbacks$",
        ManagementOperation("callback.list", "callback", "read"),
    ),
    (
        "POST",
        r"^/config/callbacks$",
        ManagementOperation("callback.update", "callback", "write"),
    ),
    # ── User Management ──
    ("POST", r"^/user/new$", ManagementOperation("user.create", "user", "write")),
    ("POST", r"^/user/update$", ManagementOperation("user.update", "user", "write")),
    ("POST", r"^/user/delete$", ManagementOperation("user.delete", "user", "write")),
    ("GET", r"^/user/info$", ManagementOperation("user.info", "user", "read")),
    # ── Global Settings ──
    (
        "GET",
        r"^/global/spend/",
        ManagementOperation("global.spend_read", "global", "read"),
    ),
    (
        "POST",
        r"^/global/spend/",
        ManagementOperation("global.spend_write", "global", "write"),
    ),
    (
        "GET",
        r"^/config/yaml$",
        ManagementOperation("config.yaml_read", "config", "read"),
    ),
    # ── Router Settings ──
    (
        "GET",
        r"^/router/settings$",
        ManagementOperation("router.settings_read", "router", "read"),
    ),
    (
        "POST",
        r"^/router/settings$",
        ManagementOperation("router.settings_write", "router", "write"),
    ),
    # ── Health & Config Reload ──
    (
        "POST",
        r"^/config/reload$",
        ManagementOperation("config.reload", "config", "write"),
    ),
]

# Compile regexes once at import time
_COMPILED_RULES: list[tuple[str, re.Pattern[str], ManagementOperation]] = [
    (method, re.compile(pattern), op) for method, pattern, op in _RULES
]


def classify(method: str, path: str) -> ManagementOperation | None:
    """
    Classify an HTTP request as a management operation.

    Args:
        method: HTTP method (GET, POST, PUT, DELETE, PATCH).
        path: Request URL path (without query string).

    Returns:
        ManagementOperation if the path is a known management endpoint,
        None otherwise (data-plane paths like /chat/completions).
    """
    method_upper = method.upper()
    for rule_method, pattern, op in _COMPILED_RULES:
        if rule_method == method_upper and pattern.search(path):
            return op
    return None


# =============================================================================
# RBAC Permission Mapping
# =============================================================================

# Maps resource_type + sensitivity to the RBAC permission string.
# These extend the existing permissions in rbac.py.

MANAGEMENT_PERMISSIONS: dict[tuple[str, str], str] = {
    ("key", "write"): "key.write",
    ("key", "read"): "key.read",
    ("team", "write"): "team.write",
    ("team", "read"): "team.read",
    ("organization", "write"): "organization.write",
    ("organization", "read"): "organization.read",
    ("model", "write"): "model.write",
    ("model", "read"): "model.read",
    ("budget", "write"): "budget.write",
    ("budget", "read"): "budget.read",
    ("spend", "read"): "spend.read",
    ("spend", "write"): "spend.write",
    ("credential", "write"): "credential.write",
    ("credential", "read"): "credential.read",
    ("guardrail", "write"): "guardrail.write",
    ("guardrail", "read"): "guardrail.read",
    ("cache", "read"): "cache.read",
    ("cache", "write"): "cache.write",
    ("callback", "read"): "callback.read",
    ("callback", "write"): "callback.write",
    ("user", "write"): "user.write",
    ("user", "read"): "user.read",
    ("global", "read"): "global.read",
    ("global", "write"): "global.write",
    ("config", "read"): "config.read",
    ("config", "write"): "config.write",
    ("router", "read"): "router.read",
    ("router", "write"): "router.write",
}


def get_required_permission(op: ManagementOperation) -> str | None:
    """
    Get the RBAC permission string required for a management operation.

    Returns:
        Permission string (e.g., "key.write"), or None if no mapping exists.
    """
    return MANAGEMENT_PERMISSIONS.get((op.resource_type, op.sensitivity))


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ManagementOperation",
    "classify",
    "get_required_permission",
    "MANAGEMENT_PERMISSIONS",
]
