"""
Plugin Manager for Gateway Extensibility
=========================================

This module provides a production-ready plugin lifecycle interface for extending the gateway.

Plugins can hook into:
- startup(app, context): Called after the FastAPI app is created but before server starts
- shutdown(app, context): Called during application shutdown

Features:
- **Capabilities**: Plugins declare what they provide (ROUTES, ROUTING_STRATEGY, etc.)
- **Dependencies**: Plugins can declare depends_on for load ordering
- **Priority**: Numeric priority for tie-breaking (lower = earlier)
- **Failure Modes**: Per-plugin behavior on failure (continue, abort, quarantine)
- **Security Policy**: Only plugins with allowed capabilities can load
- **Allowlist**: Optional plugin allowlist for explicit control

Usage:
    from litellm_llmrouter.gateway.plugin_manager import (
        GatewayPlugin,
        PluginCapability,
        PluginManager,
        PluginMetadata,
        FailureMode,
    )

    class MyPlugin(GatewayPlugin):
        @property
        def metadata(self) -> PluginMetadata:
            return PluginMetadata(
                name="my-plugin",
                version="1.0.0",
                capabilities={PluginCapability.ROUTES},
                priority=100,
            )

        async def startup(self, app, context):
            print("Plugin starting up!")

        async def shutdown(self, app, context):
            print("Plugin shutting down!")

Configuration (Environment Variables):
    - LLMROUTER_PLUGINS: Comma-separated list of plugin module paths to load
      Example: LLMROUTER_PLUGINS=mypackage.plugin1.MyPlugin,mypackage.plugin2.AnotherPlugin

    - LLMROUTER_PLUGINS_ALLOWLIST: Comma-separated list of allowed plugin paths
      If set, only plugins in this list can be loaded. Empty means all are allowed.
      Example: LLMROUTER_PLUGINS_ALLOWLIST=mypackage.plugin1.MyPlugin,mypackage.plugin2.AnotherPlugin

    - LLMROUTER_PLUGINS_ALLOWED_CAPABILITIES: Comma-separated list of allowed capabilities
      If set, plugins requesting disallowed capabilities are rejected.
      Example: LLMROUTER_PLUGINS_ALLOWED_CAPABILITIES=ROUTES,OBSERVABILITY_EXPORTER

    - LLMROUTER_PLUGINS_FAILURE_MODE: Global default failure mode (continue|abort|quarantine)
      Default: continue (plugin failure doesn't stop startup)
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from fastapi import FastAPI
    from .plugin_middleware import PluginRequest, PluginResponse, ResponseMetadata
    from .plugin_protocols import (
        MCPGatewayAccessor,
        A2AGatewayAccessor,
        ConfigSyncAccessor,
        RoutingAccessor,
        ResilienceAccessor,
        ModelsAccessor,
        MetricsAccessor,
    )

logger = logging.getLogger(__name__)

# Singleton instance
_plugin_manager: "PluginManager | None" = None


class PluginCapability(Enum):
    """
    Capabilities that plugins can provide.

    These are used for:
    - Security policy: deny loading plugins with disallowed capabilities
    - Documentation: understand what a plugin does
    - Future: conditional loading based on what's needed
    """

    ROUTES = "routes"
    """Plugin registers HTTP routes with the FastAPI app."""

    ROUTING_STRATEGY = "routing_strategy"
    """Plugin provides a custom routing strategy."""

    TOOL_RUNTIME = "tool_runtime"
    """Plugin provides tool execution runtime (e.g., MCP tools)."""

    EVALUATOR = "evaluator"
    """Plugin provides request/response evaluation hooks."""

    OBSERVABILITY_EXPORTER = "observability_exporter"
    """Plugin exports telemetry data (metrics, traces, logs)."""

    MIDDLEWARE = "middleware"
    """Plugin adds FastAPI middleware."""

    AUTH_PROVIDER = "auth_provider"
    """Plugin provides authentication/authorization."""

    STORAGE_BACKEND = "storage_backend"
    """Plugin provides storage capabilities."""

    GUARDRAIL = "guardrail"
    """Plugin provides content security guardrails."""

    CACHE = "cache"
    """Plugin provides response caching."""

    COST_TRACKER = "cost_tracker"
    """Plugin provides cost tracking and reconciliation."""


class FailureMode(Enum):
    """
    How to handle plugin failures during startup/shutdown.
    """

    CONTINUE = "continue"
    """Log error and continue with other plugins (default)."""

    ABORT = "abort"
    """Raise exception and stop startup."""

    QUARANTINE = "quarantine"
    """Disable the plugin and continue. Plugin won't be called again."""


@dataclass
class PluginMetadata:
    """
    Metadata describing a plugin's identity and capabilities.

    All fields have safe defaults for backwards compatibility with
    legacy plugins that don't override the metadata property.
    """

    name: str = ""
    """Unique plugin name. Defaults to class name if empty."""

    version: str = "0.0.0"
    """Semantic version of the plugin."""

    capabilities: set[PluginCapability] = field(default_factory=set)
    """Set of capabilities this plugin provides."""

    depends_on: list[str] = field(default_factory=list)
    """List of plugin names this plugin depends on (for ordering)."""

    priority: int = 1000
    """Load priority. Lower values load first. Default 1000 for user plugins."""

    failure_mode: FailureMode = FailureMode.CONTINUE
    """How to handle failures for this plugin."""

    description: str = ""
    """Human-readable description of what the plugin does."""


@dataclass
class PluginContext:
    """
    Context object passed to plugin startup/shutdown.

    Provides controlled access to gateway resources without
    exposing raw internals.

    The original 3 fields (settings, logger, validate_outbound_url) remain
    for backwards compatibility. Seven new subsystem accessor fields are
    added, all defaulting to ``None`` so existing plugins are unaffected.
    """

    settings: dict[str, Any] = field(default_factory=dict)
    """Read-only settings dict. Plugins should not modify this."""

    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))
    """Logger instance for the plugin to use."""

    validate_outbound_url: Callable[[str], str] | None = None
    """
    URL validation function for SSRF prevention.
    Plugins making outbound HTTP requests SHOULD use this.
    Raises SSRFBlockedError if URL is dangerous.
    """

    # =========================================================================
    # Subsystem accessors (v0.2.0) — all optional for backwards compat
    # =========================================================================

    mcp: "MCPGatewayAccessor | None" = None
    """MCP Gateway accessor: register/list servers, list/invoke tools."""

    a2a: "A2AGatewayAccessor | None" = None
    """A2A Gateway accessor: register/list agents, invoke agents."""

    config_sync: "ConfigSyncAccessor | None" = None
    """Config sync accessor: force sync, get status, get config."""

    routing: "RoutingAccessor | None" = None
    """Routing accessor: list/set strategies, register strategies, get weights."""

    resilience: "ResilienceAccessor | None" = None
    """Resilience accessor: circuit breaker status, force open/close, is_degraded."""

    models: "ModelsAccessor | None" = None
    """Models accessor: list/get LLM model deployments."""

    metrics: "MetricsAccessor | None" = None
    """Metrics accessor: get meter, create counters/histograms."""


class GatewayPlugin(ABC):
    """
    Abstract base class for gateway plugins.

    Plugins must implement startup() and shutdown() hooks.
    Both methods receive the FastAPI application instance and a PluginContext.

    Optional ASGI-level hooks (wired via PluginMiddleware):
    - on_request(request): Called before each HTTP request is processed.
      Return None to pass through, or PluginResponse to short-circuit.
    - on_response(request, response_meta): Called after each response is sent.
      Receives status + headers only (no body) to preserve streaming.

    Optional LLM lifecycle hooks (wired via PluginCallbackBridge):
    - on_llm_pre_call(model, messages, kwargs): Before LLM API call
    - on_llm_success(model, response, kwargs): After successful LLM call
    - on_llm_failure(model, exception, kwargs): After failed LLM call

    Other optional hooks:
    - health_check(): Called during readiness checks to report plugin health

    Legacy Compatibility:
    - Plugins that don't override `metadata` get safe defaults
    - Plugins using the old startup(app) signature still work
    """

    @property
    def metadata(self) -> PluginMetadata:
        """
        Return plugin metadata.

        Override this to declare capabilities, dependencies, priority, etc.
        Default implementation provides safe backwards-compatible defaults.
        """
        return PluginMetadata(
            name=self.__class__.__name__,
            version="0.0.0",
            capabilities=set(),  # No declared capabilities = all checks pass
            depends_on=[],
            priority=1000,
            failure_mode=FailureMode.CONTINUE,
        )

    @property
    def name(self) -> str:
        """Return the plugin name (from metadata or class name)."""
        meta = self.metadata
        return meta.name if meta.name else self.__class__.__name__

    @abstractmethod
    async def startup(
        self, app: "FastAPI", context: PluginContext | None = None
    ) -> None:
        """
        Called during application startup.

        Use this to register routes, middleware, or initialize resources.

        Args:
            app: The FastAPI application instance
            context: Plugin context with settings, logger, validators (optional for backwards compat)
        """
        pass

    @abstractmethod
    async def shutdown(
        self, app: "FastAPI", context: PluginContext | None = None
    ) -> None:
        """
        Called during application shutdown.

        Use this to clean up resources, close connections, etc.

        Args:
            app: The FastAPI application instance
            context: Plugin context with settings, logger, validators (optional for backwards compat)
        """
        pass

    async def health_check(self) -> dict[str, Any]:
        """
        Optional health check hook called during readiness probes.

        Override to report plugin-specific health status.

        Returns:
            Dict with at least {"status": "ok"|"degraded"|"unhealthy"}
            and any additional health metadata.
        """
        return {"status": "ok"}

    # =========================================================================
    # ASGI-level request/response hooks (wired via PluginMiddleware)
    # =========================================================================

    async def on_request(self, request: "PluginRequest") -> "PluginResponse | None":
        """
        Called before each HTTP request is processed.

        Override to inspect requests, enforce policies, or short-circuit responses.

        Args:
            request: Parsed request metadata (method, path, headers, etc.)

        Returns:
            None to pass the request through to the next plugin/handler,
            or a PluginResponse to short-circuit with a direct response.
        """
        return None

    async def on_response(
        self,
        request: "PluginRequest",
        response: "ResponseMetadata",
    ) -> None:
        """
        Called after each HTTP response is fully sent.

        Override to log, track costs, update caches, emit metrics, etc.
        Called in reverse plugin order (symmetric wrapping with on_request).

        Note: Only receives status + headers, not body, to preserve streaming.

        Args:
            request: The original request metadata
            response: Response metadata (status_code, headers, duration_ms)
        """
        pass

    # =========================================================================
    # LLM lifecycle hooks (wired via PluginCallbackBridge)
    # =========================================================================

    async def on_llm_pre_call(
        self, model: str, messages: list[Any], kwargs: dict[str, Any]
    ) -> dict[str, Any] | None:
        """
        Called before an LLM API call is made.

        Override to inspect/modify call parameters, add metadata, or block calls.

        Args:
            model: The model being called (e.g., 'gpt-4', 'claude-3-opus')
            messages: The messages list being sent
            kwargs: Additional call parameters

        Returns:
            None to pass through unchanged, or a dict of kwargs overrides
            to merge into the call parameters.
        """
        return None

    async def on_llm_success(
        self, model: str, response: Any, kwargs: dict[str, Any]
    ) -> None:
        """
        Called after a successful LLM API call.

        Override to log responses, track costs, update metrics, cache results, etc.

        Args:
            model: The model that was called
            response: The LLM response object
            kwargs: The call parameters that were used
        """
        pass

    async def on_llm_failure(
        self, model: str, exception: Exception, kwargs: dict[str, Any]
    ) -> None:
        """
        Called after a failed LLM API call.

        Override to log errors, update circuit breakers, emit alerts, etc.

        Args:
            model: The model that was called
            exception: The exception that occurred
            kwargs: The call parameters that were used
        """
        pass

    # =========================================================================
    # Configuration and infrastructure hooks
    # =========================================================================

    async def on_config_reload(
        self, old_config: dict[str, Any], new_config: dict[str, Any]
    ) -> None:
        """
        Called when gateway configuration is reloaded.

        Override to react to configuration changes (e.g., update thresholds,
        reload rules, adjust behavior).

        Args:
            old_config: The previous configuration dict
            new_config: The new configuration dict
        """
        pass

    async def on_route_register(self, route_path: str, methods: list[str]) -> None:
        """
        Called when a new route is registered.

        Override to track registered routes, add route-specific behavior, etc.

        Args:
            route_path: The URL path of the registered route
            methods: HTTP methods for the route (e.g., ["GET", "POST"])
        """
        pass

    async def on_model_health_change(
        self, model: str, healthy: bool, reason: str
    ) -> None:
        """
        Called when a model's health status changes.

        Override to react to model health transitions (e.g., adjust routing,
        emit alerts, update dashboards).

        Args:
            model: The model identifier
            healthy: Whether the model is now healthy
            reason: Human-readable reason for the health change
        """
        pass

    async def on_circuit_breaker_change(
        self, breaker_name: str, old_state: str, new_state: str
    ) -> None:
        """
        Called when a circuit breaker changes state.

        Override to react to circuit breaker transitions (e.g., emit alerts,
        adjust routing weights, update dashboards).

        Args:
            breaker_name: Name of the circuit breaker (e.g., "database", "redis")
            old_state: Previous state ("closed", "open", "half_open")
            new_state: New state ("closed", "open", "half_open")
        """
        pass

    async def on_management_operation(
        self,
        operation: str,
        resource_type: str,
        method: str,
        path: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Called when a LiteLLM management endpoint is invoked.

        Override to observe, audit, or react to management operations
        (key generation, team updates, model changes, etc.).

        Args:
            operation: Classified operation name (e.g., "key.generate",
                       "team.create", "model.add").
            resource_type: Resource category (e.g., "key", "team", "model").
            method: HTTP method (GET, POST, PUT, DELETE).
            path: Request URL path.
            metadata: Optional dict with additional context (actor, outcome).
        """
        pass


class NoOpPlugin(GatewayPlugin):
    """
    A no-op plugin that does nothing.

    Useful as a placeholder or for testing.
    """

    async def startup(
        self, app: "FastAPI", context: PluginContext | None = None
    ) -> None:
        """No-op startup."""
        pass

    async def shutdown(
        self, app: "FastAPI", context: PluginContext | None = None
    ) -> None:
        """No-op shutdown."""
        pass


class PluginLoadError(Exception):
    """Raised when a plugin fails to load."""

    def __init__(self, plugin_path: str, reason: str):
        self.plugin_path = plugin_path
        self.reason = reason
        super().__init__(f"Failed to load plugin {plugin_path}: {reason}")


class PluginSecurityError(PluginLoadError):
    """Raised when a plugin is blocked by security policy."""

    pass


class PluginDependencyError(Exception):
    """Raised when plugin dependencies cannot be resolved."""

    def __init__(self, message: str, cycle: list[str] | None = None):
        self.cycle = cycle
        super().__init__(message)


class PluginManager:
    """
    Manages the lifecycle of gateway plugins.

    The plugin manager:
    - Loads plugins from configuration (env var LLMROUTER_PLUGINS)
    - Validates plugins against allowlist and capability policy
    - Orders plugins by dependencies and priority (topological sort)
    - Calls startup hooks with PluginContext
    - Calls shutdown hooks in reverse order
    - Handles failures according to per-plugin and global failure modes
    """

    def __init__(self) -> None:
        self._plugins: list[GatewayPlugin] = []
        self._sorted_plugins: list[GatewayPlugin] | None = None  # Cached topo order
        self._quarantined: set[str] = set()  # Plugin names that failed and are disabled
        self._started = False
        self._context: PluginContext | None = None

        # Load configuration from environment
        self._allowlist = self._load_allowlist()
        self._allowed_capabilities = self._load_allowed_capabilities()
        self._global_failure_mode = self._load_global_failure_mode()

    def _load_allowlist(self) -> set[str] | None:
        """
        Load plugin allowlist from environment.

        Returns None if no allowlist is configured (all plugins allowed).
        Returns empty set if explicitly set to empty string (no plugins allowed).
        """
        value = os.getenv("LLMROUTER_PLUGINS_ALLOWLIST")
        if value is None:
            return None  # No allowlist = all allowed

        if not value.strip():
            return set()  # Empty string = none allowed

        return {p.strip() for p in value.split(",") if p.strip()}

    def _load_allowed_capabilities(self) -> set[PluginCapability] | None:
        """
        Load allowed capabilities from environment.

        Returns None if no restriction is configured (all capabilities allowed).
        """
        value = os.getenv("LLMROUTER_PLUGINS_ALLOWED_CAPABILITIES")
        if value is None:
            return None  # No restriction

        if not value.strip():
            return set()  # Empty = no capabilities allowed

        allowed = set()
        for cap_str in value.split(","):
            cap_str = cap_str.strip().upper()
            try:
                allowed.add(PluginCapability[cap_str])
            except KeyError:
                logger.warning(
                    f"Unknown capability in LLMROUTER_PLUGINS_ALLOWED_CAPABILITIES: {cap_str}"
                )

        return allowed

    def _load_global_failure_mode(self) -> FailureMode:
        """Load global default failure mode from environment."""
        value = os.getenv("LLMROUTER_PLUGINS_FAILURE_MODE", "continue").lower()
        try:
            return FailureMode(value)
        except ValueError:
            logger.warning(
                f"Invalid LLMROUTER_PLUGINS_FAILURE_MODE: {value}, using 'continue'"
            )
            return FailureMode.CONTINUE

    def _validate_allowlist(self, plugin_path: str) -> None:
        """
        Check if plugin path is in the allowlist.

        Raises PluginSecurityError if not allowed.
        """
        if self._allowlist is None:
            return  # No allowlist = all allowed

        if plugin_path not in self._allowlist:
            raise PluginSecurityError(
                plugin_path,
                f"Plugin not in allowlist. Allowed: {sorted(self._allowlist) if self._allowlist else 'none'}",
            )

    def _validate_capabilities(self, plugin: GatewayPlugin, plugin_path: str) -> None:
        """
        Check if plugin's capabilities are allowed.

        Raises PluginSecurityError if any capability is disallowed.
        """
        if self._allowed_capabilities is None:
            return  # No restriction

        meta = plugin.metadata
        disallowed = meta.capabilities - self._allowed_capabilities
        if disallowed:
            disallowed_names = sorted(c.value for c in disallowed)
            allowed_names = sorted(c.value for c in self._allowed_capabilities)
            raise PluginSecurityError(
                plugin_path,
                f"Plugin requests disallowed capabilities: {disallowed_names}. "
                f"Allowed capabilities: {allowed_names}",
            )

    def _topological_sort(self, plugins: list[GatewayPlugin]) -> list[GatewayPlugin]:
        """
        Sort plugins by dependencies using topological sort, then by priority.

        Args:
            plugins: List of plugins to sort

        Returns:
            Sorted list of plugins

        Raises:
            PluginDependencyError: If circular dependencies are detected or
                                   a dependency is missing
        """
        # Build name -> plugin map
        plugin_map: dict[str, GatewayPlugin] = {p.name: p for p in plugins}

        # Check for missing dependencies
        for plugin in plugins:
            for dep in plugin.metadata.depends_on:
                if dep not in plugin_map:
                    raise PluginDependencyError(
                        f"Plugin '{plugin.name}' depends on '{dep}' which is not loaded"
                    )

        # Kahn's algorithm for topological sort
        # Build in-degree map and adjacency list
        in_degree: dict[str, int] = {p.name: 0 for p in plugins}
        dependents: dict[str, list[str]] = {p.name: [] for p in plugins}

        for plugin in plugins:
            for dep in plugin.metadata.depends_on:
                dependents[dep].append(plugin.name)
                in_degree[plugin.name] += 1

        # Start with plugins that have no dependencies
        # Sort by priority for deterministic ordering among equals
        queue = [name for name, degree in in_degree.items() if degree == 0]
        queue.sort(key=lambda n: plugin_map[n].metadata.priority)

        result: list[GatewayPlugin] = []
        while queue:
            # Pop the plugin with lowest priority
            name = queue.pop(0)
            result.append(plugin_map[name])

            # Update dependents
            for dependent in dependents[name]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    # Insert maintaining priority order
                    queue.append(dependent)
                    queue.sort(key=lambda n: plugin_map[n].metadata.priority)

        # Check for cycles
        if len(result) != len(plugins):
            # Find the cycle for a useful error message
            remaining = [p.name for p in plugins if p not in result]
            raise PluginDependencyError(
                f"Circular dependency detected among plugins: {remaining}",
                cycle=remaining,
            )

        return result

    def register(self, plugin: GatewayPlugin) -> None:
        """
        Register a plugin with the manager.

        Args:
            plugin: The plugin instance to register

        Raises:
            RuntimeError: If called after startup() has been invoked
        """
        if self._started:
            raise RuntimeError(
                f"Cannot register plugin {plugin.name} after startup() has been called"
            )
        self._plugins.append(plugin)
        logger.info(f"Registered plugin: {plugin.name}")

    def load_from_config(self) -> int:
        """
        Load plugins from the LLMROUTER_PLUGINS environment variable.

        Returns:
            Number of plugins successfully loaded

        Note:
            Plugin paths should be fully qualified module paths containing
            a class that inherits from GatewayPlugin.
            Example: LLMROUTER_PLUGINS=mypackage.myplugin.MyPlugin
        """
        plugins_str = os.getenv("LLMROUTER_PLUGINS", "").strip()
        if not plugins_str:
            logger.debug("No plugins configured via LLMROUTER_PLUGINS")
            return 0

        loaded = 0
        for plugin_path in plugins_str.split(","):
            plugin_path = plugin_path.strip()
            if not plugin_path:
                continue

            try:
                plugin = self._load_plugin(plugin_path)
                if plugin:
                    self.register(plugin)
                    loaded += 1
            except PluginSecurityError as e:
                logger.error(
                    f"Security policy blocked plugin {plugin_path}: {e.reason}"
                )
            except PluginLoadError as e:
                logger.error(f"Failed to load plugin {plugin_path}: {e.reason}")
            except Exception as e:
                logger.error(f"Failed to load plugin {plugin_path}: {e}")

        return loaded

    def _load_plugin(self, plugin_path: str) -> GatewayPlugin | None:
        """
        Load a plugin from a module path.

        Args:
            plugin_path: Fully qualified path to the plugin class
                        (e.g., 'mypackage.myplugin.MyPlugin')

        Returns:
            Plugin instance or None if loading failed

        Raises:
            PluginSecurityError: If plugin is blocked by security policy
            PluginLoadError: If plugin cannot be imported or instantiated
        """
        # Check allowlist FIRST (before importing)
        self._validate_allowlist(plugin_path)

        # Split module path and class name
        if "." not in plugin_path:
            raise PluginLoadError(plugin_path, "Invalid path format (no class name)")

        module_path, class_name = plugin_path.rsplit(".", 1)

        try:
            # Import the module
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise PluginLoadError(plugin_path, f"Could not import module: {e}") from e

        try:
            # Get the class
            plugin_class = getattr(module, class_name)
        except AttributeError as e:
            raise PluginLoadError(plugin_path, f"Class not found in module: {e}") from e

        # Verify it's a GatewayPlugin subclass
        if not issubclass(plugin_class, GatewayPlugin):
            raise PluginLoadError(plugin_path, "Class is not a GatewayPlugin subclass")

        # Instantiate
        try:
            plugin = plugin_class()
        except Exception as e:
            raise PluginLoadError(plugin_path, f"Failed to instantiate: {e}") from e

        # Validate capabilities AFTER instantiation (to read metadata)
        self._validate_capabilities(plugin, plugin_path)

        return plugin

    def _create_context(self) -> PluginContext:
        """Create the PluginContext for plugin startup/shutdown."""
        # Import here to avoid circular imports
        try:
            from litellm_llmrouter.url_security import validate_outbound_url

            url_validator = validate_outbound_url
        except ImportError:
            url_validator = None
            logger.warning(
                "Could not import validate_outbound_url, plugins won't have URL validation"
            )

        # Populate settings from environment for plugin consumption
        settings = self._collect_plugin_settings()

        # Create subsystem adapters (lazy imports inside each adapter)
        try:
            from .plugin_adapters import create_all_adapters

            adapters = create_all_adapters()
        except Exception as e:
            logger.warning(f"Could not create plugin adapters: {e}")
            adapters = {}

        return PluginContext(
            settings=settings,
            logger=logger,
            validate_outbound_url=url_validator,
            mcp=adapters.get("mcp"),
            a2a=adapters.get("a2a"),
            config_sync=adapters.get("config_sync"),
            routing=adapters.get("routing"),
            resilience=adapters.get("resilience"),
            models=adapters.get("models"),
            metrics=adapters.get("metrics"),
        )

    def _collect_plugin_settings(self) -> dict[str, Any]:
        """
        Collect gateway settings relevant to plugins.

        Returns a read-only snapshot of environment-driven configuration
        that plugins may need during startup/shutdown.
        """
        settings: dict[str, Any] = {}

        # Gateway feature flags
        settings["mcp_gateway_enabled"] = (
            os.getenv("MCP_GATEWAY_ENABLED", "false").lower() == "true"
        )
        settings["a2a_gateway_enabled"] = (
            os.getenv("A2A_GATEWAY_ENABLED", "false").lower() == "true"
        )
        settings["otel_enabled"] = os.getenv("OTEL_ENABLED", "true").lower() != "false"
        settings["policy_engine_enabled"] = (
            os.getenv("POLICY_ENGINE_ENABLED", "false").lower() == "true"
        )

        # Config paths
        settings["config_path"] = os.getenv("LITELLM_CONFIG_PATH", "")
        settings["policy_config_path"] = os.getenv("POLICY_CONFIG_PATH", "")

        # Service info
        settings["service_name"] = os.getenv("OTEL_SERVICE_NAME", "litellm-gateway")

        # Plugin-specific settings (prefixed with ROUTEIQ_PLUGIN_)
        for key, value in os.environ.items():
            if key.startswith("ROUTEIQ_PLUGIN_"):
                # Strip prefix and lowercase for plugin consumption
                setting_key = key[len("ROUTEIQ_PLUGIN_") :].lower()
                settings[setting_key] = value

        return settings

    def _get_failure_mode(self, plugin: GatewayPlugin) -> FailureMode:
        """Get the effective failure mode for a plugin."""
        meta = plugin.metadata
        # Plugin-specific mode overrides global default
        if meta.failure_mode != FailureMode.CONTINUE:
            return meta.failure_mode
        return self._global_failure_mode

    def _handle_failure(
        self, plugin: GatewayPlugin, phase: str, error: Exception
    ) -> None:
        """
        Handle a plugin failure according to its failure mode.

        Args:
            plugin: The plugin that failed
            phase: "startup" or "shutdown"
            error: The exception that was raised

        Raises:
            Exception: Re-raises the error if failure_mode is ABORT
        """
        mode = self._get_failure_mode(plugin)

        if mode == FailureMode.ABORT:
            logger.error(
                f"Plugin {plugin.name} {phase} failed (failure_mode=abort): {error}"
            )
            raise error

        if mode == FailureMode.QUARANTINE:
            logger.error(f"Plugin {plugin.name} {phase} failed, quarantining: {error}")
            self._quarantined.add(plugin.name)
        else:
            # CONTINUE
            logger.error(f"Plugin {plugin.name} {phase} failed (continuing): {error}")

    async def startup(self, app: "FastAPI") -> None:
        """
        Call startup hooks on all registered plugins.

        Plugins are started in dependency order (topological sort + priority).

        Args:
            app: The FastAPI application instance

        Raises:
            PluginDependencyError: If plugin dependencies cannot be resolved
            Exception: If any plugin with failure_mode=abort fails
        """
        if self._started:
            logger.warning("Plugin manager startup() called multiple times")
            return

        self._started = True

        # Sort plugins by dependencies and priority, cache for get_*_plugins()
        try:
            sorted_plugins = self._topological_sort(self._plugins)
            self._sorted_plugins = sorted_plugins
        except PluginDependencyError as e:
            logger.error(f"Plugin dependency resolution failed: {e}")
            raise

        # Create context
        self._context = self._create_context()

        startup_timeout = float(os.getenv("ROUTEIQ_PLUGIN_STARTUP_TIMEOUT", "30"))

        for plugin in sorted_plugins:
            if plugin.name in self._quarantined:
                logger.debug(f"Skipping quarantined plugin: {plugin.name}")
                continue

            try:
                logger.debug(
                    f"Starting plugin: {plugin.name} "
                    f"(priority={plugin.metadata.priority})"
                )
                await asyncio.wait_for(
                    plugin.startup(app, self._context),
                    timeout=startup_timeout,
                )
                logger.info(f"Plugin started: {plugin.name}")
            except asyncio.TimeoutError:
                error = TimeoutError(
                    f"Plugin {plugin.name} startup timed out after {startup_timeout}s"
                )
                self._handle_failure(plugin, "startup", error)
            except Exception as e:
                self._handle_failure(plugin, "startup", e)

    async def shutdown(self, app: "FastAPI") -> None:
        """
        Call shutdown hooks on all registered plugins.

        Plugins are shut down in reverse startup order.

        Args:
            app: The FastAPI application instance
        """
        if not self._started:
            logger.warning("Plugin manager shutdown() called before startup()")
            return

        # Shutdown in reverse order (sorted order, reversed)
        try:
            sorted_plugins = self._topological_sort(self._plugins)
        except PluginDependencyError:
            # If sorting fails, just use registration order
            sorted_plugins = self._plugins

        for plugin in reversed(sorted_plugins):
            if plugin.name in self._quarantined:
                logger.debug(f"Skipping quarantined plugin: {plugin.name}")
                continue

            try:
                logger.debug(f"Shutting down plugin: {plugin.name}")
                await plugin.shutdown(app, self._context)
                logger.info(f"Plugin shut down: {plugin.name}")
            except Exception as e:
                self._handle_failure(plugin, "shutdown", e)

        self._started = False

    @property
    def plugins(self) -> list[GatewayPlugin]:
        """Return a copy of the registered plugins list."""
        return list(self._plugins)

    @property
    def quarantined_plugins(self) -> set[str]:
        """Return the set of quarantined plugin names."""
        return set(self._quarantined)

    @property
    def is_started(self) -> bool:
        """Return whether startup() has been called."""
        return self._started

    def get_middleware_plugins(self) -> list[GatewayPlugin]:
        """
        Return plugins that override on_request or on_response.

        These are the plugins that should be wired into PluginMiddleware.
        Plugins are returned in topological sort order (dependency + priority).

        Returns:
            List of plugins with middleware hooks, in dependency order
        """
        source = (
            self._sorted_plugins if self._sorted_plugins is not None else self._plugins
        )
        middleware_plugins = []
        for plugin in source:
            if plugin.name in self._quarantined:
                continue
            # Check if the plugin overrides on_request or on_response
            has_on_request = type(plugin).on_request is not GatewayPlugin.on_request
            has_on_response = type(plugin).on_response is not GatewayPlugin.on_response
            if has_on_request or has_on_response:
                middleware_plugins.append(plugin)
        return middleware_plugins

    def get_callback_plugins(self) -> list[GatewayPlugin]:
        """
        Return plugins that override LLM lifecycle hooks.

        These are the plugins that should be wired into PluginCallbackBridge.
        Plugins are returned in topological sort order (dependency + priority).

        Returns:
            List of plugins with LLM callback hooks, in dependency order
        """
        source = (
            self._sorted_plugins if self._sorted_plugins is not None else self._plugins
        )
        callback_plugins = []
        for plugin in source:
            if plugin.name in self._quarantined:
                continue
            has_pre = type(plugin).on_llm_pre_call is not GatewayPlugin.on_llm_pre_call
            has_success = (
                type(plugin).on_llm_success is not GatewayPlugin.on_llm_success
            )
            has_failure = (
                type(plugin).on_llm_failure is not GatewayPlugin.on_llm_failure
            )
            if has_pre or has_success or has_failure:
                callback_plugins.append(plugin)
        return callback_plugins

    def get_guardrail_plugins(self) -> list[GatewayPlugin]:
        """
        Get plugins with GUARDRAIL capability.

        Returns:
            List of plugins declaring the GUARDRAIL capability,
            excluding quarantined plugins, in topological sort order.
        """
        source = (
            self._sorted_plugins if self._sorted_plugins is not None else self._plugins
        )
        return [
            p
            for p in source
            if p.name not in self._quarantined
            and PluginCapability.GUARDRAIL in p.metadata.capabilities
        ]

    async def notify_config_reload(
        self, old_config: dict[str, Any], new_config: dict[str, Any]
    ) -> None:
        """Dispatch on_config_reload to all active plugins."""
        source = (
            self._sorted_plugins if self._sorted_plugins is not None else self._plugins
        )
        for plugin in source:
            if plugin.name in self._quarantined:
                continue
            try:
                await plugin.on_config_reload(old_config, new_config)
            except Exception as e:
                logger.warning(f"Plugin {plugin.name} on_config_reload failed: {e}")

    async def notify_route_registered(
        self, route_path: str, methods: list[str]
    ) -> None:
        """Dispatch on_route_register to all active plugins."""
        source = (
            self._sorted_plugins if self._sorted_plugins is not None else self._plugins
        )
        for plugin in source:
            if plugin.name in self._quarantined:
                continue
            try:
                await plugin.on_route_register(route_path, methods)
            except Exception as e:
                logger.warning(f"Plugin {plugin.name} on_route_register failed: {e}")

    async def notify_circuit_breaker_change(
        self, breaker_name: str, old_state: str, new_state: str
    ) -> None:
        """Dispatch on_circuit_breaker_change to all active plugins."""
        source = (
            self._sorted_plugins if self._sorted_plugins is not None else self._plugins
        )
        for plugin in source:
            if plugin.name in self._quarantined:
                continue
            try:
                await plugin.on_circuit_breaker_change(
                    breaker_name, old_state, new_state
                )
            except Exception as e:
                logger.warning(
                    f"Plugin {plugin.name} on_circuit_breaker_change failed: {e}"
                )

    async def notify_management_operation(
        self,
        operation: str,
        resource_type: str,
        method: str,
        path: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Dispatch on_management_operation to all active plugins."""
        source = (
            self._sorted_plugins if self._sorted_plugins is not None else self._plugins
        )
        for plugin in source:
            if plugin.name in self._quarantined:
                continue
            # Only dispatch to plugins that override the hook
            if (
                type(plugin).on_management_operation
                is GatewayPlugin.on_management_operation
            ):
                continue
            try:
                await plugin.on_management_operation(
                    operation, resource_type, method, path, metadata=metadata
                )
            except Exception as e:
                logger.warning(
                    f"Plugin {plugin.name} on_management_operation failed: {e}"
                )

    async def health_checks(self) -> dict[str, dict[str, Any]]:
        """
        Run health checks on all active plugins.

        Returns:
            Dict mapping plugin name to health status dict.
            Each status contains at least {"status": "ok"|"degraded"|"unhealthy"}.
        """
        results: dict[str, dict[str, Any]] = {}

        for plugin in self._plugins:
            if plugin.name in self._quarantined:
                results[plugin.name] = {"status": "quarantined"}
                continue

            try:
                results[plugin.name] = await plugin.health_check()
            except Exception as e:
                logger.warning(f"Plugin {plugin.name} health check failed: {e}")
                results[plugin.name] = {"status": "unhealthy", "error": str(e)}

        return results


def get_plugin_manager() -> PluginManager:
    """
    Get the global plugin manager instance.

    Creates the instance on first call (lazy initialization).

    Returns:
        The global PluginManager instance
    """
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager


def reset_plugin_manager() -> None:
    """
    Reset the global plugin manager instance.

    Primarily useful for testing.
    """
    global _plugin_manager
    _plugin_manager = None
