"""
OPA-Style Policy Engine for RouteIQ Gateway
============================================

This module provides a pre-request policy evaluation engine that can allow/deny
requests based on:
- Subject (team/user/api-key)
- Route path and method
- Model (for /chat/completions, /v1/completions, etc.)
- Request metadata (headers, source IP)

The engine implements:
1. PolicyEvaluator - Evaluates policies against request context
2. PolicyMiddleware - ASGI middleware for enforcement
3. PolicyDecision - Auditable decision records with deny reasons
4. Built-in rule engines - Config-driven allow/deny rules

Configuration:
    Environment variables:
    - POLICY_ENGINE_ENABLED: Enable/disable policy engine (default: false)
    - POLICY_ENGINE_FAIL_MODE: "open" (default) or "closed"
      - fail-open: Policy errors logged, request allowed
      - fail-closed: Policy errors return 503
    - POLICY_CONFIG_PATH: Path to policy YAML config (optional)

Usage:
    from litellm_llmrouter.policy_engine import (
        PolicyMiddleware,
        get_policy_engine,
        PolicyDecision,
    )

    # ASGI middleware - add early to catch all routes
    app.app = PolicyMiddleware(app.app)

    # Or manual evaluation:
    engine = get_policy_engine()
    decision = await engine.evaluate(context)
    if not decision.allowed:
        return 403, decision.reason
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import yaml
from starlette.types import ASGIApp, Receive, Scope, Send

from .auth import get_request_id

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


def is_policy_engine_enabled() -> bool:
    """Check if policy engine is enabled."""
    env_val = os.getenv("POLICY_ENGINE_ENABLED", "false").lower().strip()
    return env_val in ("true", "1", "yes", "on")


def is_policy_fail_closed() -> bool:
    """Check if policy engine uses fail-closed mode."""
    env_val = os.getenv("POLICY_ENGINE_FAIL_MODE", "open").lower().strip()
    return env_val == "closed"


def get_policy_config_path() -> str | None:
    """Get the path to policy configuration file."""
    return os.getenv("POLICY_CONFIG_PATH")


# =============================================================================
# Policy Decision Types
# =============================================================================


class PolicyAction(str, Enum):
    """Policy evaluation result actions."""
    ALLOW = "allow"
    DENY = "deny"
    ERROR = "error"  # Policy evaluation error


@dataclass
class PolicyDecision:
    """
    Auditable decision record from policy evaluation.
    
    Contains all information needed for audit logging and error responses.
    """
    allowed: bool
    action: PolicyAction
    reason: str | None = None
    policy_name: str | None = None  # Which policy made the decision
    evaluation_time_ms: float = 0.0
    request_id: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Context captured for audit
    subject: dict[str, Any] = field(default_factory=dict)
    route: str = ""
    method: str = ""
    model: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for audit logging."""
        return {
            "allowed": self.allowed,
            "action": self.action.value,
            "reason": self.reason,
            "policy_name": self.policy_name,
            "evaluation_time_ms": round(self.evaluation_time_ms, 3),
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "subject": self.subject,
            "route": self.route,
            "method": self.method,
            "model": self.model,
        }


@dataclass
class PolicyContext:
    """
    Request context for policy evaluation.
    
    Contains all information needed to evaluate a policy.
    """
    # Subject identification
    team_id: str | None = None
    user_id: str | None = None
    api_key_subject: str | None = None  # Masked key identifier
    api_key_hash: str | None = None
    
    # Request routing
    route: str = ""
    method: str = "GET"
    
    # Model (for LLM routes)
    model: str | None = None
    
    # Request metadata
    source_ip: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    
    # Request ID for correlation
    request_id: str | None = None
    
    def get_subject_dict(self) -> dict[str, Any]:
        """Get subject as a dictionary for audit logging."""
        return {
            "team_id": self.team_id,
            "user_id": self.user_id,
            "api_key_subject": self.api_key_subject,
        }


# =============================================================================
# Policy Rules
# =============================================================================


@dataclass
class PolicyRule:
    """
    A single policy rule.
    
    Rules are evaluated in order. First matching rule wins.
    """
    name: str
    action: PolicyAction  # allow or deny
    
    # Matchers (all must match if specified - logical AND)
    # Use None to mean "any" / "not specified"
    teams: list[str] | None = None  # Team IDs or patterns
    users: list[str] | None = None  # User IDs or patterns
    api_keys: list[str] | None = None  # API key patterns
    routes: list[str] | None = None  # Route patterns (glob)
    methods: list[str] | None = None  # HTTP methods
    models: list[str] | None = None  # Model patterns
    source_ips: list[str] | None = None  # Source IP patterns/CIDRs
    
    # Optional reason for denials
    reason: str | None = None
    
    # Priority (lower = evaluated first)
    priority: int = 100
    
    def matches(self, context: PolicyContext) -> bool:
        """
        Check if this rule matches the given context.
        
        All specified matchers must match (logical AND).
        An unspecified matcher (None) matches anything.
        """
        # Check teams
        if self.teams is not None:
            if not context.team_id:
                return False
            if not self._matches_patterns(context.team_id, self.teams):
                return False
        
        # Check users
        if self.users is not None:
            if not context.user_id:
                return False
            if not self._matches_patterns(context.user_id, self.users):
                return False
        
        # Check API keys
        if self.api_keys is not None:
            if not context.api_key_subject:
                return False
            if not self._matches_patterns(context.api_key_subject, self.api_keys):
                return False
        
        # Check routes
        if self.routes is not None:
            if not self._matches_patterns(context.route, self.routes):
                return False
        
        # Check methods
        if self.methods is not None:
            method_upper = context.method.upper()
            if method_upper not in [m.upper() for m in self.methods]:
                return False
        
        # Check models
        if self.models is not None:
            if not context.model:
                # No model specified in request but rule requires model match
                return False
            if not self._matches_patterns(context.model, self.models):
                return False
        
        # Check source IPs
        if self.source_ips is not None:
            if not context.source_ip:
                return False
            if not self._matches_ip_patterns(context.source_ip, self.source_ips):
                return False
        
        return True
    
    def _matches_patterns(self, value: str, patterns: list[str]) -> bool:
        """Check if value matches any of the patterns (glob-style)."""
        for pattern in patterns:
            # Exact match or glob pattern
            if fnmatch.fnmatch(value, pattern):
                return True
            # Also try regex if pattern starts with ^
            if pattern.startswith("^"):
                try:
                    if re.match(pattern, value):
                        return True
                except re.error:
                    pass
        return False
    
    def _matches_ip_patterns(self, ip: str, patterns: list[str]) -> bool:
        """Check if IP matches any of the patterns (CIDR or exact)."""
        for pattern in patterns:
            if "/" in pattern:
                # CIDR notation - check network match
                try:
                    import ipaddress
                    network = ipaddress.ip_network(pattern, strict=False)
                    addr = ipaddress.ip_address(ip)
                    if addr in network:
                        return True
                except ValueError:
                    pass
            else:
                # Exact match or glob
                if fnmatch.fnmatch(ip, pattern):
                    return True
        return False


# =============================================================================
# Policy Configuration
# =============================================================================


@dataclass
class PolicyConfig:
    """
    Policy engine configuration.
    
    Contains the ordered list of policy rules and default action.
    """
    rules: list[PolicyRule] = field(default_factory=list)
    default_action: PolicyAction = PolicyAction.ALLOW  # Default when no rule matches
    default_reason: str = "No matching policy rule"
    
    # Excluded paths (always allowed, bypass policy)
    excluded_paths: set[str] = field(default_factory=lambda: {
        "/_health/live",
        "/_health/ready",
        "/health/liveliness",
        "/health/readiness",
        "/health",
    })
    
    def sort_rules(self) -> None:
        """Sort rules by priority (ascending)."""
        self.rules.sort(key=lambda r: r.priority)
    
    @classmethod
    def from_yaml(cls, yaml_content: str) -> "PolicyConfig":
        """Load policy configuration from YAML string."""
        data = yaml.safe_load(yaml_content)
        return cls.from_dict(data or {})
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PolicyConfig":
        """Load policy configuration from dictionary."""
        config = cls()
        
        # Load default action
        default = data.get("default_action", "allow").lower()
        config.default_action = PolicyAction(default) if default in ("allow", "deny") else PolicyAction.ALLOW
        config.default_reason = data.get("default_reason", "No matching policy rule")
        
        # Load excluded paths
        excluded = data.get("excluded_paths", [])
        if excluded:
            config.excluded_paths = set(excluded)
        
        # Load rules
        rules_data = data.get("rules", [])
        for rule_data in rules_data:
            action_str = rule_data.get("action", "allow").lower()
            action = PolicyAction(action_str) if action_str in ("allow", "deny") else PolicyAction.ALLOW
            
            rule = PolicyRule(
                name=rule_data.get("name", "unnamed"),
                action=action,
                teams=rule_data.get("teams"),
                users=rule_data.get("users"),
                api_keys=rule_data.get("api_keys"),
                routes=rule_data.get("routes"),
                methods=rule_data.get("methods"),
                models=rule_data.get("models"),
                source_ips=rule_data.get("source_ips"),
                reason=rule_data.get("reason"),
                priority=rule_data.get("priority", 100),
            )
            config.rules.append(rule)
        
        config.sort_rules()
        return config
    
    @classmethod
    def from_file(cls, file_path: str) -> "PolicyConfig":
        """Load policy configuration from YAML file."""
        with open(file_path, "r") as f:
            return cls.from_yaml(f.read())


# =============================================================================
# Policy Engine
# =============================================================================


class PolicyEngine:
    """
    OPA-style policy engine for request authorization.
    
    Evaluates policies against request context and returns auditable decisions.
    """
    
    def __init__(
        self,
        config: PolicyConfig | None = None,
        enabled: bool | None = None,
        fail_closed: bool | None = None,
    ):
        self._config = config or PolicyConfig()
        self._enabled = enabled if enabled is not None else is_policy_engine_enabled()
        self._fail_closed = fail_closed if fail_closed is not None else is_policy_fail_closed()
        self._lock = asyncio.Lock()
        logger.info(
            f"PolicyEngine initialized (enabled={self._enabled}, "
            f"fail_closed={self._fail_closed}, rules={len(self._config.rules)})"
        )
    
    @property
    def is_enabled(self) -> bool:
        """Check if policy engine is enabled."""
        return self._enabled
    
    @property
    def is_fail_closed(self) -> bool:
        """Check if fail-closed mode is enabled."""
        return self._fail_closed
    
    @property
    def config(self) -> PolicyConfig:
        """Get the current policy configuration."""
        return self._config
    
    async def reload_config(self, config: PolicyConfig) -> None:
        """Reload policy configuration atomically."""
        async with self._lock:
            config.sort_rules()
            self._config = config
            logger.info(f"PolicyEngine configuration reloaded ({len(config.rules)} rules)")
    
    async def reload_from_file(self, file_path: str) -> None:
        """Reload policy configuration from file."""
        config = PolicyConfig.from_file(file_path)
        await self.reload_config(config)
    
    def is_path_excluded(self, path: str) -> bool:
        """Check if a path is excluded from policy evaluation."""
        return path in self._config.excluded_paths
    
    async def evaluate(self, context: PolicyContext) -> PolicyDecision:
        """
        Evaluate a policy against the given context.
        
        Args:
            context: Request context containing subject, route, model, etc.
            
        Returns:
            PolicyDecision with allow/deny result and audit information
        """
        import time
        start_time = time.monotonic()
        
        request_id = context.request_id or get_request_id() or str(uuid.uuid4())
        
        # If policy engine is disabled, allow everything
        if not self._enabled:
            return PolicyDecision(
                allowed=True,
                action=PolicyAction.ALLOW,
                reason="Policy engine disabled",
                policy_name="__disabled__",
                request_id=request_id,
                subject=context.get_subject_dict(),
                route=context.route,
                method=context.method,
                model=context.model,
            )
        
        # Check excluded paths
        if self.is_path_excluded(context.route):
            return PolicyDecision(
                allowed=True,
                action=PolicyAction.ALLOW,
                reason="Path excluded from policy",
                policy_name="__excluded__",
                request_id=request_id,
                subject=context.get_subject_dict(),
                route=context.route,
                method=context.method,
                model=context.model,
            )
        
        try:
            # Evaluate rules in priority order
            for rule in self._config.rules:
                if rule.matches(context):
                    elapsed_ms = (time.monotonic() - start_time) * 1000
                    
                    decision = PolicyDecision(
                        allowed=(rule.action == PolicyAction.ALLOW),
                        action=rule.action,
                        reason=rule.reason,
                        policy_name=rule.name,
                        evaluation_time_ms=elapsed_ms,
                        request_id=request_id,
                        subject=context.get_subject_dict(),
                        route=context.route,
                        method=context.method,
                        model=context.model,
                    )
                    
                    # Log denials at warning level
                    if not decision.allowed:
                        logger.warning(
                            f"Policy DENY: rule={rule.name} route={context.route} "
                            f"reason={rule.reason} request_id={request_id}"
                        )
                    else:
                        logger.debug(
                            f"Policy ALLOW: rule={rule.name} route={context.route} "
                            f"request_id={request_id}"
                        )
                    
                    return decision
            
            # No rule matched - use default action
            elapsed_ms = (time.monotonic() - start_time) * 1000
            allowed = self._config.default_action == PolicyAction.ALLOW
            
            decision = PolicyDecision(
                allowed=allowed,
                action=self._config.default_action,
                reason=self._config.default_reason,
                policy_name="__default__",
                evaluation_time_ms=elapsed_ms,
                request_id=request_id,
                subject=context.get_subject_dict(),
                route=context.route,
                method=context.method,
                model=context.model,
            )
            
            logger.debug(
                f"Policy {'ALLOW' if allowed else 'DENY'} (default): "
                f"route={context.route} request_id={request_id}"
            )
            
            return decision
            
        except Exception as e:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.error(f"Policy evaluation error: {e}", exc_info=True)
            
            # In fail-closed mode, deny on errors
            if self._fail_closed:
                return PolicyDecision(
                    allowed=False,
                    action=PolicyAction.ERROR,
                    reason=f"Policy evaluation error (fail-closed): {str(e)}",
                    policy_name="__error__",
                    evaluation_time_ms=elapsed_ms,
                    request_id=request_id,
                    subject=context.get_subject_dict(),
                    route=context.route,
                    method=context.method,
                    model=context.model,
                )
            else:
                # Fail-open: allow on errors
                return PolicyDecision(
                    allowed=True,
                    action=PolicyAction.ALLOW,
                    reason=f"Policy evaluation error (fail-open): {str(e)}",
                    policy_name="__error__",
                    evaluation_time_ms=elapsed_ms,
                    request_id=request_id,
                    subject=context.get_subject_dict(),
                    route=context.route,
                    method=context.method,
                    model=context.model,
                )
    
    def get_status(self) -> dict[str, Any]:
        """Get policy engine status for health checks."""
        return {
            "enabled": self._enabled,
            "fail_closed": self._fail_closed,
            "rules_count": len(self._config.rules),
            "default_action": self._config.default_action.value,
            "excluded_paths_count": len(self._config.excluded_paths),
        }


# =============================================================================
# ASGI Policy Middleware
# =============================================================================


class PolicyMiddleware:
    """
    ASGI middleware for policy enforcement.
    
    Evaluates policies BEFORE the request reaches any FastAPI routes.
    This ensures:
    - Policy enforcement for all routes including provider proxies
    - Streaming responses work correctly (enforcement before streaming begins)
    - No response buffering
    
    Denied requests get a JSON 403 response immediately.
    """
    
    def __init__(
        self,
        app: ASGIApp,
        engine: PolicyEngine | None = None,
    ):
        self.app = app
        self._engine = engine or get_policy_engine()
        logger.info(
            f"PolicyMiddleware initialized (enabled={self._engine.is_enabled})"
        )
    
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """ASGI entry point."""
        # Only apply to HTTP requests
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        path = scope.get("path", "")
        scope.get("method", "GET")
        
        # Skip if policy engine is disabled
        if not self._engine.is_enabled:
            await self.app(scope, receive, send)
            return
        
        # Skip excluded paths
        if self._engine.is_path_excluded(path):
            await self.app(scope, receive, send)
            return
        
        # Build context from ASGI scope
        context = self._build_context(scope)
        
        # Evaluate policy
        decision = await self._engine.evaluate(context)
        
        # If denied, return 403 immediately (before streaming begins)
        if not decision.allowed:
            await self._send_policy_denied_response(send, decision)
            # Log audit event for the denial
            await self._audit_denial(decision)
            return
        
        # Policy allowed - proceed with request
        await self.app(scope, receive, send)
    
    def _build_context(self, scope: Scope) -> PolicyContext:
        """Build PolicyContext from ASGI scope."""
        path = scope.get("path", "")
        method = scope.get("method", "GET")
        
        # Extract headers
        headers = {}
        for name, value in scope.get("headers", []):
            headers[name.decode("latin1").lower()] = value.decode("latin1")
        
        # Extract request ID
        request_id = headers.get("x-request-id")
        
        # Extract source IP (from X-Forwarded-For or client address)
        source_ip = headers.get("x-forwarded-for")
        if source_ip:
            source_ip = source_ip.split(",")[0].strip()
        else:
            client = scope.get("client")
            if client:
                source_ip = client[0]
        
        # Extract subject from common auth headers
        # These would typically be set by auth middleware before policy
        api_key_subject = None
        team_id = None
        user_id = None
        
        # Try to extract from LiteLLM's auth info if attached to scope/state
        litellm_info = scope.get("state", {}).get("litellm_user_info", {})
        if litellm_info:
            team_id = litellm_info.get("team_id")
            user_id = litellm_info.get("user_id")
            api_key_subject = litellm_info.get("api_key_subject")
        
        # Also check for headers that might carry subject info
        if not team_id:
            team_id = headers.get("x-team-id")
        if not user_id:
            user_id = headers.get("x-user-id")
        
        # Extract model from request body for LLM routes
        # NOTE: For streaming, we don't want to buffer the body
        # Could use Content-Type check and read if it's JSON, but this
        # requires care. For now, model extraction from body is optional.
        model = headers.get("x-model")  # Optional header hint
        
        return PolicyContext(
            team_id=team_id,
            user_id=user_id,
            api_key_subject=api_key_subject,
            route=path,
            method=method,
            model=model,
            source_ip=source_ip,
            headers=headers,
            request_id=request_id,
        )
    
    async def _send_policy_denied_response(
        self,
        send: Send,
        decision: PolicyDecision,
    ) -> None:
        """Send a 403 JSON response for policy denial."""
        body = {
            "error": "policy_denied",
            "message": decision.reason or "Request denied by policy",
            "policy_name": decision.policy_name,
            "request_id": decision.request_id,
        }
        
        body_bytes = json.dumps(body).encode("utf-8")
        
        await send({
            "type": "http.response.start",
            "status": 403,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body_bytes)).encode()),
            ],
        })
        await send({
            "type": "http.response.body",
            "body": body_bytes,
            "more_body": False,
        })
    
    async def _audit_denial(self, decision: PolicyDecision) -> None:
        """Log policy denial to audit log."""
        try:
            from .audit import audit_denied
            
            # Create a new audit action for policy denials
            # We'll use a custom action string
            await audit_denied(
                action="policy.evaluation.deny",
                resource_type="policy_decision",
                resource_id=decision.policy_name,
                reason=decision.reason,
                metadata={
                    "route": decision.route,
                    "method": decision.method,
                    "model": decision.model,
                    "subject": decision.subject,
                    "evaluation_time_ms": decision.evaluation_time_ms,
                },
            )
        except Exception as e:
            # Audit failure should not block the denial
            logger.error(f"Failed to audit policy denial: {e}")


# =============================================================================
# Global Singleton Management
# =============================================================================


_policy_engine: PolicyEngine | None = None


def get_policy_engine() -> PolicyEngine:
    """Get or create the global policy engine singleton."""
    global _policy_engine
    if _policy_engine is None:
        config = PolicyConfig()
        
        # Load from file if configured
        config_path = get_policy_config_path()
        if config_path and os.path.exists(config_path):
            try:
                config = PolicyConfig.from_file(config_path)
                logger.info(f"Loaded policy config from {config_path}")
            except Exception as e:
                logger.error(f"Failed to load policy config from {config_path}: {e}")
        
        _policy_engine = PolicyEngine(config=config)
    return _policy_engine


def reset_policy_engine() -> None:
    """Reset the global policy engine (for testing)."""
    global _policy_engine
    _policy_engine = None


def set_policy_engine(engine: PolicyEngine) -> None:
    """Set the global policy engine (for testing)."""
    global _policy_engine
    _policy_engine = engine


# =============================================================================
# Helper Functions for App Integration
# =============================================================================


def add_policy_middleware(app: Any) -> bool:
    """
    Add policy middleware to a FastAPI/Starlette app.
    
    This should be called early in app setup, before routers are registered.
    The middleware wraps the ASGI app to enforce policies.
    
    Returns:
        True if middleware was added, False if disabled by config
    """
    engine = get_policy_engine()
    
    if not engine.is_enabled:
        logger.info(
            "Policy middleware disabled (POLICY_ENGINE_ENABLED not set or false)"
        )
        return False
    
    # Add as ASGI middleware by wrapping the app
    original_app = getattr(app, "app", None)
    if original_app:
        app.app = PolicyMiddleware(original_app, engine)
    else:
        logger.warning("Could not wrap app for policy middleware")
        return False
    
    logger.info(
        f"Policy middleware enabled ({engine.config.rules_count} rules, "
        f"fail_closed={engine.is_fail_closed})"
    )
    return True


# =============================================================================
# Audit Action Extension
# =============================================================================


# Add policy audit action to the AuditAction enum
# This is done via monkey-patching since we can't modify the original enum
# Alternatively, we just use string actions

AUDIT_ACTION_POLICY_DENY = "policy.evaluation.deny"
AUDIT_ACTION_POLICY_ALLOW = "policy.evaluation.allow"
AUDIT_ACTION_POLICY_ERROR = "policy.evaluation.error"


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Config functions
    "is_policy_engine_enabled",
    "is_policy_fail_closed",
    "get_policy_config_path",
    # Types
    "PolicyAction",
    "PolicyDecision",
    "PolicyContext",
    "PolicyRule",
    "PolicyConfig",
    # Engine
    "PolicyEngine",
    "get_policy_engine",
    "reset_policy_engine",
    "set_policy_engine",
    # Middleware
    "PolicyMiddleware",
    "add_policy_middleware",
    # Audit actions
    "AUDIT_ACTION_POLICY_DENY",
    "AUDIT_ACTION_POLICY_ALLOW",
    "AUDIT_ACTION_POLICY_ERROR",
]
