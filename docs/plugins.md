# Gateway Plugin System

The RouteIQ gateway provides a production-ready plugin system for extending gateway functionality without modifying core code.

## Overview

Plugins can:
- Register HTTP routes
- Add middleware
- Implement custom routing strategies
- Export observability data
- Provide authentication/authorization

## Quick Start

### 1. Create a Plugin

```python
from litellm_llmrouter.gateway.plugin_manager import (
    GatewayPlugin,
    PluginMetadata,
    PluginCapability,
    PluginContext,
)

class MyPlugin(GatewayPlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="my-plugin",
            version="1.0.0",
            capabilities={PluginCapability.ROUTES},
            description="My custom plugin",
        )

    async def startup(self, app, context: PluginContext):
        # Register routes, initialize resources
        @app.get("/my-plugin/hello")
        async def hello():
            return {"message": "Hello from my plugin!"}

    async def shutdown(self, app, context: PluginContext):
        # Cleanup resources
        pass
```

### 2. Enable the Plugin

Set the environment variable:

```bash
export LLMROUTER_PLUGINS=mypackage.myplugin.MyPlugin
```

Multiple plugins can be loaded by comma-separating:

```bash
export LLMROUTER_PLUGINS=mypackage.plugin1.Plugin1,mypackage.plugin2.Plugin2
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLMROUTER_PLUGINS` | Comma-separated list of plugin module paths | (empty) |
| `LLMROUTER_PLUGINS_ALLOWLIST` | Comma-separated allowlist of plugin paths | (none - all allowed) |
| `LLMROUTER_PLUGINS_ALLOWED_CAPABILITIES` | Comma-separated allowed capabilities | (none - all allowed) |
| `LLMROUTER_PLUGINS_FAILURE_MODE` | Global default failure mode | `continue` |

### Plugin Allowlist

For production deployments, you can restrict which plugins can load:

```bash
# Only allow these specific plugins
export LLMROUTER_PLUGINS_ALLOWLIST=myorg.approved.Plugin1,myorg.approved.Plugin2
```

If not set, any plugin can load. If set to empty string, no plugins can load.

### Capability Security Policy

Restrict what capabilities plugins can use:

```bash
# Only allow plugins that register routes or export metrics
export LLMROUTER_PLUGINS_ALLOWED_CAPABILITIES=ROUTES,OBSERVABILITY_EXPORTER
```

Available capabilities:
- `ROUTES` - Register HTTP routes
- `ROUTING_STRATEGY` - Provide routing strategies
- `TOOL_RUNTIME` - Provide tool execution
- `EVALUATOR` - Request/response evaluation
- `OBSERVABILITY_EXPORTER` - Export telemetry
- `MIDDLEWARE` - Add FastAPI middleware
- `AUTH_PROVIDER` - Provide authentication
- `STORAGE_BACKEND` - Provide storage

## Plugin Metadata

Plugins declare their identity and capabilities via metadata:

```python
from litellm_llmrouter.gateway.plugin_manager import (
    PluginMetadata,
    PluginCapability,
    FailureMode,
)

@property
def metadata(self) -> PluginMetadata:
    return PluginMetadata(
        name="my-plugin",           # Unique identifier
        version="1.0.0",            # Semantic version
        capabilities={              # What this plugin provides
            PluginCapability.ROUTES,
            PluginCapability.MIDDLEWARE,
        },
        depends_on=["other-plugin"],  # Dependencies (for ordering)
        priority=100,                  # Lower = loads earlier (default: 1000)
        failure_mode=FailureMode.CONTINUE,  # What to do on failure
        description="Does something useful",
    )
```

### Dependencies

Plugins can declare dependencies to control load order:

```python
# This plugin loads AFTER "base-plugin"
PluginMetadata(
    name="my-plugin",
    depends_on=["base-plugin"],
)
```

The plugin manager uses topological sorting to resolve dependencies.
Circular dependencies are detected and raise `PluginDependencyError`.

### Priority

When plugins have no dependencies between them, priority determines order:

```python
# This loads before plugins with higher priority numbers
PluginMetadata(
    name="early-plugin",
    priority=10,  # Default is 1000
)
```

## Failure Modes

Control what happens when a plugin fails during startup/shutdown:

| Mode | Behavior |
|------|----------|
| `continue` | Log error, continue with other plugins (default) |
| `abort` | Raise exception, stop startup |
| `quarantine` | Disable the plugin, continue with others |

Per-plugin:
```python
PluginMetadata(
    name="critical-plugin",
    failure_mode=FailureMode.ABORT,  # Stop if this fails
)
```

Global default:
```bash
export LLMROUTER_PLUGINS_FAILURE_MODE=abort
```

## Plugin Context

Plugins receive a `PluginContext` object with utilities:

```python
async def startup(self, app, context: PluginContext):
    # Use the provided logger
    context.logger.info("Plugin starting")

    # Access settings (read-only)
    debug_mode = context.settings.get("debug", False)

    # Validate URLs for SSRF prevention (when making outbound requests)
    if context.validate_outbound_url:
        safe_url = context.validate_outbound_url(user_provided_url)
```

### SSRF Prevention

Plugins making outbound HTTP requests **should** use the provided URL validator:

```python
async def call_external_api(self, url: str, context: PluginContext):
    if context.validate_outbound_url:
        # Raises SSRFBlockedError if URL is dangerous
        context.validate_outbound_url(url)

    # Now safe to make the request
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
```

## Backwards Compatibility

Plugins written for older versions continue to work:

```python
# Legacy plugin - still works!
class LegacyPlugin(GatewayPlugin):
    async def startup(self, app):  # No context parameter
        pass

    async def shutdown(self, app):
        pass
```

Legacy plugins get default metadata:
- `name`: Class name
- `version`: "0.0.0"
- `capabilities`: Empty set (passes all capability checks)
- `priority`: 1000
- `failure_mode`: `continue`

## Load Order

Plugins are loaded in this order:

1. **Discovery**: Load from `LLMROUTER_PLUGINS` env var
2. **Allowlist check**: Reject plugins not in allowlist
3. **Capability check**: Reject plugins with disallowed capabilities
4. **Dependency resolution**: Topological sort + priority
5. **Startup**: Call `startup()` in resolved order
6. **Shutdown**: Call `shutdown()` in reverse order

## Best Practices

1. **Declare capabilities**: Always declare what your plugin does
2. **Use context.logger**: Consistent logging with the gateway
3. **Validate URLs**: Use `context.validate_outbound_url` for user-provided URLs
4. **Handle failures gracefully**: Use appropriate `failure_mode`
5. **Specify dependencies**: If your plugin needs another, declare it
6. **Keep priority high**: Use default (1000) unless you need earlier loading

## Troubleshooting

### Plugin not loading

Check:
1. Is the plugin path correct? (e.g., `mypackage.module.ClassName`)
2. Is the plugin in the allowlist (if set)?
3. Does the plugin request allowed capabilities (if restricted)?

### Startup order issues

Use dependencies:
```python
PluginMetadata(
    name="my-plugin",
    depends_on=["required-plugin"],  # Ensures required-plugin loads first
)
```

### Plugin fails silently

Check the global failure mode:
```bash
# See detailed errors
export LLMROUTER_PLUGINS_FAILURE_MODE=abort
```

Or set per-plugin:
```python
PluginMetadata(
    failure_mode=FailureMode.ABORT,
)
```
