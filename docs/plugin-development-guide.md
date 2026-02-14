# RouteIQ Plugin Development Guide

## Overview

RouteIQ v0.2.0 introduces a Universal PluginContext that gives plugins typed access to every gateway subsystem. This guide shows how to build plugins that interact with MCP, A2A, routing, metrics, and more.

## PluginContext Fields

| Field | Type | Purpose |
|-------|------|---------|
| `settings` | `dict[str, Any]` | Read-only gateway configuration |
| `logger` | `logging.Logger` | Plugin logger |
| `validate_outbound_url` | `Callable` | SSRF protection |
| `mcp` | `MCPGatewayAccessor` | MCP server/tool management |
| `a2a` | `A2AGatewayAccessor` | A2A agent management |
| `config_sync` | `ConfigSyncAccessor` | Config sync status/control |
| `routing` | `RoutingAccessor` | Strategy inspection/control |
| `resilience` | `ResilienceAccessor` | Circuit breaker status |
| `models` | `ModelsAccessor` | LLM model deployment info |
| `metrics` | `MetricsAccessor` | OTel metric creation |

All subsystem fields default to `None` for backwards compatibility.

## Quick Start

```python
from litellm_llmrouter.gateway.plugin_manager import (
    GatewayPlugin,
    PluginCapability,
    PluginContext,
    PluginMetadata,
)

class MyPlugin(GatewayPlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="my-plugin",
            version="1.0.0",
            capabilities={PluginCapability.TOOL_RUNTIME},
            priority=500,
        )

    async def startup(self, app, context=None):
        if context and context.mcp and context.mcp.is_enabled():
            await context.mcp.register_server(
                server_id="my-server",
                name="My Tool Server",
                url="https://my-tools.example.com/mcp",
            )

    async def shutdown(self, app, context=None):
        if context and context.mcp:
            await context.mcp.unregister_server("my-server")
```

## Available Hooks

### ASGI-level hooks
- `on_request(request)` - Intercept HTTP requests
- `on_response(request, response)` - Observe HTTP responses

### LLM lifecycle hooks
- `on_llm_pre_call(model, messages, kwargs)` - Before LLM API call
- `on_llm_success(model, response, kwargs)` - After successful call
- `on_llm_failure(model, exception, kwargs)` - After failed call

### Infrastructure hooks
- `on_config_reload(old_config, new_config)` - Config changes
- `on_route_register(route_path, methods)` - Route registration
- `on_model_health_change(model, healthy, reason)` - Model health
- `on_circuit_breaker_change(breaker_name, old_state, new_state)` - CB state
- `on_management_operation(operation, resource_type, method, path)` - Management ops
- `health_check()` - Readiness probes

## Using Metrics

```python
async def startup(self, app, context=None):
    if context and context.metrics:
        self._counter = context.metrics.create_counter(
            "myplugin.requests",
            unit="{request}",
            description="Requests handled by my plugin",
        )

async def on_request(self, request):
    if self._counter:
        self._counter.add(1, {"path": request.path})
    return None
```

## Using Routing

```python
async def startup(self, app, context=None):
    if context and context.routing:
        strategies = context.routing.list_strategies()
        active = context.routing.get_active_strategy()
        self.logger.info(f"Active strategy: {active}, all: {strategies}")
```

## Configuration

Enable your plugin via environment variable:

```bash
LLMROUTER_PLUGINS=mypackage.plugins.MyPlugin
```

Plugin-specific settings use the `ROUTEIQ_PLUGIN_` prefix:

```bash
ROUTEIQ_PLUGIN_MY_SETTING=value
```

These are accessible via `context.settings["my_setting"]`.

## Reference Implementation

See `src/litellm_llmrouter/gateway/plugins/bedrock_agentcore_mcp.py` for a complete reference plugin that uses all subsystem accessors.
