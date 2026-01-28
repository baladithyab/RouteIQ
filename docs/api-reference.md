# API Reference

The LiteLLM + LLMRouter gateway exposes an OpenAI-compatible API.

## Base URL

```
http://localhost:4000
```

## Authentication

Include your API key in the Authorization header:

```bash
Authorization: Bearer sk-your-api-key
```

## Endpoints

### Chat Completions

```http
POST /chat/completions
```

**Request:**
```json
{
  "model": "gpt-4",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7
}
```

**Response:**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "model": "gpt-4",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      }
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 15,
    "total_tokens": 25
  }
}
```

### Completions

```http
POST /completions
```

### Embeddings

```http
POST /embeddings
```

### Models

```http
GET /models
```

### Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.56.5"
}
```

### Cloud-Native Probes

RouteIQ Gateway exposes standard liveness and readiness probes for Kubernetes/cloud environments:

```http
GET /_health/live
```
*Returns 200 OK if the service is running.*

```http
GET /_health/ready
```
*Returns 200 OK if the service is ready to accept traffic (DB/Redis connected).*

## Router-Specific Endpoints

### Get Current Routing Strategy

```http
GET /router/info
Authorization: Bearer sk-master-key
```

**Response:**
```json
{
  "registered_strategies": ["llmrouter-knn", "llmrouter-mlp"],
  "strategy_count": 2,
  "hot_reload_enabled": true
}
```

### Reload Router Model

```http
POST /router/reload
Authorization: Bearer sk-master-key
```

### Get Routing Decision (Debug)

> **Note:** This endpoint is not yet implemented. Planned for future release.

```http
POST /router/route
Authorization: Bearer sk-master-key
Content-Type: application/json

{
  "query": "Explain quantum entanglement"
}
```

**Response (planned):**
```json
{
  "selected_model": "claude-3-opus",
  "confidence": 0.87,
  "alternatives": [
    {"model": "gpt-4", "score": 0.82},
    {"model": "claude-3-sonnet", "score": 0.75}
  ]
}
```

## A2A Gateway Endpoints

> Requires `A2A_GATEWAY_ENABLED=true`

### Register A2A Agent

```http
POST /a2a/agents
Authorization: Bearer sk-master-key
Content-Type: application/json

{
  "agent_name": "my-agent",
  "description": "Agent for customer support",
  "url": "http://agent-service:8000/a2a",
  "capabilities": ["chat", "support"],
  "agent_card_params": {},
  "litellm_params": {}
}
```

### List A2A Agents

```http
GET /a2a/agents
Authorization: Bearer sk-master-key
```

**Response:**
```json
{
  "agents": [
    {
      "agent_id": "abc123...",
      "agent_name": "my-agent",
      "description": "Agent for customer support",
      "url": "http://agent-service:8000/a2a"
    }
  ]
}
```

### Discover Agents by Capability

```http
GET /a2a/agents?capability=chat
Authorization: Bearer sk-master-key
```

### Get Agent Card

```http
GET /a2/a/agents/{agent_id}/card
```

### Unregister Agent

```http
DELETE /a2a/agents/{agent_id}
Authorization: Bearer sk-master-key
```

### Streaming A2A Message

```http
POST /a2a/{agent_id}/message/stream
Authorization: Bearer sk-master-key
Content-Type: application/json

{
  "jsonrpc": "2.0",
  "method": "message/send",
  "id": "1",
  "params": { ... }
}
```

## MCP Gateway Endpoints

> Requires `MCP_GATEWAY_ENABLED=true` and `LITELLM_ROUTER_MCP_MODE=true`
> as MCP Gateway endpoints are not backwards-compatible with LiteLLM
> native `/mcp` endpoints [using `/llmrouter/mcp`].
> See README for details.

> Note: These REST endpoints are prefixed with `/llmrouter/mcp` to avoid conflicts
> with LiteLLM's native `/mcp` endpoint (which uses JSON-RPC over SSE).

### Register MCP Server

```http
POST /llmrouter/mcp/servers
Authorization: Bearer sk-master-key
Content-Type: application/json

{
  "server_id": "my-mcp-server",
  "name": "My MCP Server",
  "url": "http://mcp-service:8080/mcp",
  "transport": "streamable_http",
  "tools": ["search", "fetch"]
}
```

### List MCP Servers

```http
GET /llmrouter/mcp/servers
Authorization: Bearer sk-master-key
```

### Get MCP Server

```http
GET /llmrouter/mcp/servers/{server_id}
Authorization: Bearer sk-master-key
```

### Update MCP Server

```http
PUT /llmrouter/mcp/servers/{server_id}
Authorization: Bearer sk-master-key
Content-Type: application/json

{
  "server_id": "my-mcp-server",
  "name": "Updated MCP Server",
  "url": "http://mcp-service:8080/mcp",
  "transport": "streamable_http",
  "tools": ["search", "fetch", "store"]
}
```

### Unregister MCP Server

```http
DELETE /llmrouter/mcp/servers/{server_id}
Authorization: Bearer sk-master-key
```

### List Available Tools

```http
GET /llmrouter/mcp/tools
Authorization: Bearer sk-master-key
```

### List Available Tools (Detailed)

```http
GET /llmrouter/mcp/tools/list
Authorization: Bearer sk-master-key
```

### Get Tool Details

```http
GET /llmrouter/mcp/tools/{tool_name}
Authorization: Bearer sk-master-key
```

### Call MCP Tool

```http
POST /llmrouter/mcp/tools/call
Authorization: Bearer sk-master-key
Content-Type: application/json

{
  "tool_name": "search",
  "arguments": { "query": "example" }
}
```

### Register MCP Server Tool

```http
POST /llmrouter/mcp/servers/{server_id}/tools
Authorization: Bearer sk-master-key
Content-Type: application/json

{
  "name": "custom_tool",
  "description": "A custom tool",
  "input_schema": { "type": "object", "properties": {} }
}
```

### List Available Resources

```http
GET /llmrouter/mcp/resources
Authorization: Bearer sk-master-key
```

### MCP Server Health

```http
GET /v1/llmrouter/mcp/server/health
Authorization: Bearer sk-master-key
```

### Single MCP Server Health

```http
GET /v1/llmrouter/mcp/server/{server_id}/health
Authorization: Bearer sk-master-key
```

### MCP Registry Document

```http
GET /v1/llmrouter/mcp/registry.json
Authorization: Bearer sk-master-key
```

### MCP Access Groups

```http
GET /v1/llmrouter/mcp/access_groups
Authorization: Bearer sk-master-key
```

## Skills Gateway Endpoints

> See [Skills Gateway Guide](skills-gateway.md) for full details.
> These endpoints are inherited from LiteLLM Proxy.

### List Skills

```http
GET /v1/skills
Authorization: Bearer sk-proxy-key
```

### Create/Invoke Skill

```http
POST /v1/skills
Authorization: Bearer sk-proxy-key
Content-Type: application/json

{
  "skill_name": "computer_use",
  "parameters": { ... }
}
```

### Get Skill Details

```http
GET /v1/skills/{skill_id}
Authorization: Bearer sk-proxy-key
```

## Inherited API Families

RouteIQ Gateway inherits the full suite of OpenAI-compatible endpoints from LiteLLM. For a complete analysis of supported endpoints, see [API Parity Analysis](api-parity-analysis.md).

Key supported families include:

- **Assistants**: `/v1/assistants*`, `/v1/threads*`, `/v1/runs*`
- **Files**: `/v1/files*`
- **Vector Stores**: `/v1/vector_stores*` (OpenAI-compatible file search)
- **Responses**: `/v1/responses` (if configured)

## Config Sync Endpoints

### Get Sync Status

```http
GET /config/sync/status
Authorization: Bearer sk-master-key
```

**Response:**
```json
{
  "enabled": true,
  "hot_reload_enabled": true,
  "sync_interval_seconds": 60,
  "s3": {
    "enabled": true,
    "bucket": "my-config-bucket",
    "key": "config.yaml",
    "last_etag": "abc123..."
  },
  "reload_count": 3,
  "last_sync_time": 1704067200.0,
  "running": true
}
```

### Reload Configuration

```http
POST /config/reload
Authorization: Bearer sk-master-key
Content-Type: application/json

{
  "force_sync": true
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Config reload triggered",
  "synced_from_remote": true
}
```

## Error Responses

```json
{
  "error": {
    "message": "Invalid API key",
    "type": "authentication_error",
    "code": "invalid_api_key"
  }
}
```

## Rate Limiting

Rate limit headers:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1704067200
```

## Streaming

Enable streaming with `stream: true`:

```bash
curl -X POST http://localhost:4000/chat/completions \
  -H "Authorization: Bearer sk-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

## See Also

- [A2A Gateway](a2a-gateway.md) - Agent-to-Agent protocol
- [MCP Gateway](mcp-gateway.md) - Model Context Protocol
- [Vector Stores](vector-stores.md) - Vector database integrations (coming soon)
- [Hot Reloading](hot-reloading.md) - Dynamic configuration updates
