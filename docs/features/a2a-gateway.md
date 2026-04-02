# A2A Protocol

RouteIQ implements Google's [A2A (Agent-to-Agent)](https://google.github.io/A2A/)
protocol for standardized communication between AI agents.

## Overview

The A2A Gateway enables:

- Registering AI agents with their capabilities
- Discovering agents based on capabilities
- Routing requests to appropriate agents
- Building multi-agent systems with orchestration

## Enabling A2A Gateway

```bash
A2A_GATEWAY_ENABLED=true
```

## API Endpoints

### Register an Agent

```bash
curl -X POST http://localhost:4000/a2a/agents \
  -H "X-Admin-API-Key: $ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "my-agent",
    "name": "My Agent",
    "description": "A helpful assistant agent",
    "capabilities": ["chat", "code-review"]
  }'
```

### Invoke an Agent

```bash
curl -X POST http://localhost:4000/a2a/my-agent \
  -H "Authorization: Bearer $MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "message/send",
    "id": "1",
    "params": {
      "message": {
        "role": "user",
        "content": "Hello, agent!"
      }
    }
  }'
```

### List Agents

```bash
curl http://localhost:4000/a2a/agents \
  -H "Authorization: Bearer $API_KEY"
```

### Streaming

```bash
curl -X POST http://localhost:4000/a2a/my-agent/message/stream \
  -H "Authorization: Bearer $MASTER_KEY" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "jsonrpc": "2.0",
    "method": "message/send",
    "id": "1",
    "params": {
      "message": {
        "role": "user",
        "content": "Stream a response"
      }
    }
  }'
```

## Observability

A2A operations are instrumented with OpenTelemetry spans via `a2a_tracing.py`,
providing visibility into agent invocations, latency, and errors.
