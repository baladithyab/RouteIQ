# MCP Gateway

The MCP (Model Context Protocol) Gateway allows LLMs to access external tools,
data sources, and services through a standardized protocol.

## Overview

The MCP Gateway acts as a centralized hub for Model Context Protocol interactions,
enabling LLMs to:

- Access external tools (search, file operations, APIs)
- Query data sources and databases
- Interact with custom services
- Use resources from MCP servers

## Surface Contract

| Surface | Base Path | Protocol | Consumers |
|---------|-----------|----------|-----------|
| RouteIQ Management | `/llmrouter/mcp/*` | REST (JSON) | Admins, CI/CD |
| Upstream Parity | `/v1/mcp/*` | REST (JSON) | LiteLLM-compatible clients |
| Native MCP | `/mcp/*` | JSON-RPC / SSE | Claude Desktop, IDEs |

## Enabling MCP Gateway

```bash
MCP_GATEWAY_ENABLED=true
```

## Feature Flags

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_GATEWAY_ENABLED` | `false` | Enable MCP gateway |
| `MCP_SSE_TRANSPORT_ENABLED` | `false` | Enable SSE transport |
| `MCP_SSE_LEGACY_MODE` | `false` | Legacy SSE mode |
| `MCP_PROTOCOL_PROXY_ENABLED` | `false` | Protocol-level proxy (admin) |
| `MCP_OAUTH_ENABLED` | `false` | OAuth for MCP |

## Registering an MCP Server

```bash
curl -X POST http://localhost:4000/llmrouter/mcp/servers \
  -H "X-Admin-API-Key: $ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-tools",
    "url": "http://my-mcp-server:8080",
    "transport": "sse"
  }'
```

## Tool Discovery

```bash
# List all available tools
curl http://localhost:4000/llmrouter/mcp/tools \
  -H "Authorization: Bearer $API_KEY"
```

## Tool Invocation

Requires `LLMROUTER_ENABLE_MCP_TOOL_INVOCATION=true`:

```bash
curl -X POST http://localhost:4000/llmrouter/mcp/tools/call \
  -H "X-Admin-API-Key: $ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "stub.echo",
    "arguments": {"message": "hello"}
  }'
```

## Security

- **SSRF Protection**: All registered server URLs are validated against SSRF attacks
- **Dual validation**: URLs checked at registration time (no DNS) and invocation time (with DNS)
- **Admin auth required**: Server management endpoints require admin API key
- **Audit logging**: All MCP operations are logged for audit compliance

!!! note
    Skills (Anthropic Computer Use, Bash, Text Editor) are distinct from MCP.
    See [Skills Gateway](skills-gateway.md) for details.
