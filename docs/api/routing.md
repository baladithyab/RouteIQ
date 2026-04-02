# Routing API

APIs for managing routing strategies and A/B testing.

## Router Info

```http
GET /router/info
Authorization: Bearer <api_key>
```

Returns current routing strategy, model list, and health status.

## Hot Reload

```http
POST /router/reload
X-Admin-API-Key: <admin_key>
```

Reloads routing strategies and model artifacts without restart.

## Strategy Configuration

Routing strategies are configured via `config.yaml`:

```yaml
router_settings:
  routing_strategy: llmrouter-knn
  routing_strategy_args:
    model_path: /app/models/knn_router.pt
```

## A/B Testing

Configure traffic splits between strategies:

```python
from litellm_llmrouter import get_routing_registry

registry = get_routing_registry()
registry.set_weights({"baseline": 90, "candidate": 10})
```

## MCP Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/llmrouter/mcp/servers` | Admin | List MCP servers |
| POST | `/llmrouter/mcp/servers` | Admin | Register server |
| GET | `/llmrouter/mcp/tools` | User | List all tools |
| POST | `/llmrouter/mcp/tools/call` | Admin | Invoke a tool |

## A2A Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/a2a/agents` | User | List agents |
| POST | `/a2a/agents` | Admin | Register agent |
| POST | `/a2a/{agent_id}` | User | Invoke agent |
| DELETE | `/a2a/agents/{id}` | Admin | Remove agent |
