# Gateway Plugin System

The RouteIQ gateway provides a production-ready plugin system for extending gateway functionality without modifying core code.

## Overview

Plugins can:
- Register HTTP routes
- Add middleware
- Implement custom routing strategies
- Export observability data
- Provide authentication/authorization

### What Plugins Can Do (Capabilities)

The plugin system is designed around specific capabilities. Plugins must declare which capabilities they require in their metadata.

| Capability | Description | Stability |
|------------|-------------|-----------|
| `ROUTES` | Register new FastAPI routes/endpoints | Stable |
| `MIDDLEWARE` | Add global or per-route middleware | Stable |
| `ROUTING_STRATEGY` | Add custom ML or logic-based routing strategies | Stable |
| `OBSERVABILITY_EXPORTER` | Export traces/metrics to custom backends | Stable |
| `EVALUATOR` | Score/evaluate MCP and Agent interactions | Beta |
| `TOOL_RUNTIME` | Custom execution environments for tools | Alpha |
| `AUTH_PROVIDER` | Custom authentication logic | Alpha |
| `STORAGE_BACKEND` | Custom storage for state/config | Alpha |

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

## Evaluator Plugins

Evaluator plugins provide post-execution scoring for MCP tool invocations and A2A agent calls. They emit OTEL metrics for observability.

### Evaluator Plugin Contract

Evaluator plugins extend `EvaluatorPlugin` and implement two methods:

```python
from litellm_llmrouter.gateway.plugins.evaluator import (
    EvaluatorPlugin,
    EvaluationResult,
    MCPInvocationContext,
    A2AInvocationContext,
)

class MyEvaluator(EvaluatorPlugin):
    async def evaluate_mcp_result(
        self, context: MCPInvocationContext
    ) -> EvaluationResult:
        # Score the MCP tool invocation
        score = 1.0 if context.success else 0.0
        return EvaluationResult(
            score=score,
            status="success" if context.success else "error",
            metadata={"tool_name": context.tool_name},
        )

    async def evaluate_a2a_result(
        self, context: A2AInvocationContext
    ) -> EvaluationResult:
        # Score the A2A agent invocation
        return EvaluationResult(
            score=0.9,
            status="success",
            metadata={"agent_id": context.agent_id},
        )
```

### OTEL Attributes

Evaluator plugins emit the following OTEL span attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `eval.plugin` | string | Name of the evaluator plugin |
| `eval.score` | float | Numeric score (0.0-1.0) |
| `eval.status` | string | Status: success, error, skipped |
| `eval.duration_ms` | float | Evaluation duration in ms |
| `eval.error` | string | Error message (if status is error) |
| `eval.invocation_type` | string | Type: "mcp" or "a2a" |

### Enabling Evaluator Hooks

Evaluator hooks are **disabled by default**. Enable them with:

```bash
export ROUTEIQ_EVALUATOR_ENABLED=true
```

### UpskillEvaluatorPlugin (Reference Implementation)

A reference evaluator plugin is provided that demonstrates:
- Basic success/failure scoring
- Optional integration with `upskill` CLI
- SSRF-protected endpoint calls
- OTEL metric emission

#### Enable the Plugin

```bash
# Enable the plugin
export LLMROUTER_PLUGINS=litellm_llmrouter.gateway.plugins.upskill_evaluator.UpskillEvaluatorPlugin

# Enable evaluator hooks
export ROUTEIQ_EVALUATOR_ENABLED=true
```

#### Optional Upskill Integration

The plugin can optionally use the `upskill` CLI or service for advanced scoring:

```bash
# Enable upskill integration
export ROUTEIQ_UPSKILL_ENABLED=true

# Optional: Specify upskill service endpoint
export ROUTEIQ_UPSKILL_ENDPOINT=http://upskill-service:8080/evaluate

# Optional: Timeout for upskill calls (default: 5 seconds)
export ROUTEIQ_UPSKILL_TIMEOUT=10
```

**Note**: The plugin has no hard dependency on upskill. If upskill is not available, it falls back to basic scoring.

#### How It Works

1. **Basic Mode** (default): Scores 1.0 for success, 0.0 for failure
2. **Upskill CLI Mode**: Shells out to `upskill` binary if found in PATH
3. **Upskill Service Mode**: Calls HTTP endpoint (with SSRF protection)

#### Example OTEL Output

When evaluator hooks are enabled, spans will include attributes like:

```
eval.plugin: upskill-evaluator
eval.score: 0.95
eval.status: success
eval.duration_ms: 12.5
eval.invocation_type: mcp
```

### Creating Custom Evaluators

1. Create your evaluator class:

```python
# mypackage/my_evaluator.py
from litellm_llmrouter.gateway.plugins.evaluator import (
    EvaluatorPlugin,
    EvaluationResult,
    MCPInvocationContext,
    A2AInvocationContext,
    register_evaluator,
)
from litellm_llmrouter.gateway.plugin_manager import (
    PluginMetadata,
    PluginCapability,
    PluginContext,
)

class MyEvaluator(EvaluatorPlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="my-evaluator",
            version="1.0.0",
            capabilities={PluginCapability.EVALUATOR},
            description="Custom evaluation logic",
        )

    async def startup(self, app, context: PluginContext | None = None):
        # Register as an evaluator
        register_evaluator(self)

    async def evaluate_mcp_result(
        self, context: MCPInvocationContext
    ) -> EvaluationResult:
        # Your custom scoring logic
        score = self._calculate_score(context)
        return EvaluationResult(score=score, status="success")

    async def evaluate_a2a_result(
        self, context: A2AInvocationContext
    ) -> EvaluationResult:
        score = self._calculate_score_a2a(context)
        return EvaluationResult(score=score, status="success")

    def _calculate_score(self, context):
        # Implement your scoring logic
        return 0.9 if context.success else 0.1

    def _calculate_score_a2a(self, context):
        return 0.85 if context.success else 0.15
```

2. Enable your evaluator:

```bash
export LLMROUTER_PLUGINS=mypackage.my_evaluator.MyEvaluator
export ROUTEIQ_EVALUATOR_ENABLED=true
```

### Security Considerations

1. **SSRF Protection**: Always use `context.validate_outbound_url` when making outbound HTTP requests
2. **Timeouts**: Set appropriate timeouts for external evaluation services
3. **Error Handling**: Evaluator failures are logged but don't block the request
4. **Resource Limits**: Be mindful of evaluation overhead on request latency

## Custom Vector Database Resources

Plugins can extend the gateway's RAG capabilities by providing custom vector store implementations.

### Registering a Custom Vector Store

Currently, the internal hook for registering a fully custom vector store backend is **experimental**. However, you can implement a plugin that provides vector search endpoints and integrates with the existing RAG system via the `vector_store` interface.

**Roadmap Item**: A stable `PluginCapability.VECTOR_STORE` API is planned for Milestone D+.

#### Implementation Pattern

1.  **Define the Store**: Implement the `VectorStore` interface.
2.  **Expose Endpoints**: Use `PluginCapability.ROUTES` to expose management endpoints.
3.  **Monkey-patch (Temporary)**: Until the stable API is ready, you may need to register your store in the global registry during startup.

```python
from litellm_llmrouter.gateway.plugin_manager import GatewayPlugin, PluginMetadata, PluginCapability
from litellm_llmrouter.gateway.vector_stores import register_vector_store

class MyVectorStorePlugin(GatewayPlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="my-vector-store",
            version="0.1.0",
            capabilities={PluginCapability.ROUTES},
        )

    async def startup(self, app, context):
        # Register the custom backend
        # Note: This uses an internal API that may change
        register_vector_store("my_custom_db", MyCustomVectorStoreClass)
        
        context.logger.info("Registered custom vector store: my_custom_db")
```

## Overriding and Patching Built-in Behavior

Plugins may need to modify or extend existing gateway functionality. It is critical to do this safely to maintain system stability and upgradeability.

### Recommended Approaches

1.  **Routing Strategy Plugins**: Instead of modifying the core router, register a new strategy and configure the gateway to use it.
2.  **Middleware Hooks**: Use `PluginCapability.MIDDLEWARE` to intercept requests/responses globally.
3.  **Additional Routes**: Add new specific routes that take precedence over generic ones (FastAPI matches specific routes first).

### Discouraged: Monkey Patching

**Avoid import-time monkey patching.** Modifying global state or replacing functions at import time can lead to unpredictable behavior, especially with hot-reloading and testing.

**If you must patch core functionality:**
1.  Do it inside `startup()`.
2.  Ensure it is idempotent (safe to run multiple times).
3.  Log a warning so operators know standard behavior is altered.

```python
# Discouraged but sometimes necessary pattern
async def startup(self, app, context):
    import litellm_llmrouter.some_module as target_module
    
    # Save original
    self._original_func = target_module.some_function
    
    # Define patch
    def patched_function(*args, **kwargs):
        context.logger.warning("Using patched function")
        return self._original_func(*args, **kwargs)
        
    # Apply patch
    target_module.some_function = patched_function
    
async def shutdown(self, app, context):
    # Restore original
    if hasattr(self, '_original_func'):
        import litellm_llmrouter.some_module as target_module
        target_module.some_function = self._original_func
```

### Route Ordering

FastAPI evaluates routes in order of registration.
- **Built-in routes** are registered before plugins.
- **Plugin routes** are registered during startup.

To "override" a built-in route (e.g., `/v1/chat/completions`), you cannot simply register a new route with the same path, as the first one will always match.

**Workaround**:
1.  Use Middleware to intercept requests to the target path.
2.  Redirect or handle the request within the middleware.
3.  Return a response directly, bypassing the router.

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
