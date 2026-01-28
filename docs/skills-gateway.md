# Skills Gateway

**RouteIQ Gateway** features a Skills Gateway that allows you to register and execute Python functions as "skills" that can be invoked by models or other agents.

## Overview

The Skills Gateway provides
This guide covers the **Skills Gateway** functionality, which exposes Anthropic's "Skills" (Computer Use, Bash, Text Editor) via standard endpoints.

## Overview

The Skills Gateway allows you to use Anthropic's [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) precursors—specifically "Computer Use" and other agentic tools—through a standardized API.

> **Note:** Skills are distinct from the [MCP Gateway](mcp-gateway.md).
> - **Skills**: Anthropic-specific agentic capabilities (Computer Use, Bash, Text Editor).
> - **MCP**: An open standard for connecting any LLM to any data source or tool.
>
> This project supports **both** simultaneously.

## Supported Endpoints

These endpoints are inherited directly from the underlying LiteLLM Proxy. We do not fork or modify them, ensuring full compatibility with upstream LiteLLM updates.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/skills` | Create/invoke a skill |
| `GET` | `/v1/skills` | List available skills |
| `GET` | `/v1/skills/{id}` | Get details for a specific skill |
| `DELETE` | `/v1/skills/{id}` | Remove a skill |

## Authentication

The Skills Gateway uses the same authentication as the rest of the proxy.

```bash
Authorization: Bearer sk-your-proxy-key
```

## Configuration

To use Skills, you must configure your Anthropic API key and ensure your model supports the required beta features.

### 1. Environment Variables

Ensure your `ANTHROPIC_API_KEY` is set in your environment or `.env` file.

```bash
ANTHROPIC_API_KEY=sk-ant-...
```

### 2. Model Configuration

In your `config.yaml`, configure a model that supports Anthropic's beta features (e.g., `claude-3-5-sonnet-20241022`).

```yaml
model_list:
  - model_name: claude-3-5-sonnet
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY
```

### 3. Usage with `beta=true`

When making requests to the Skills endpoints, you may need to enable beta features depending on the specific skill you are accessing.

## Moat-Mode & Database Backing

For production deployments ("Moat Mode"), you can configure LiteLLM to back these skills with a database (PostgreSQL) instead of in-memory storage. This ensures persistence across container restarts.

Refer to the [LiteLLM Documentation](https://docs.litellm.ai/docs/proxy/skills) for details on setting up the `litellm_proxy` database connection.

## Routing to Multiple Accounts

You can route Skills requests to different Anthropic accounts by defining multiple models in your `config.yaml` and using specific model names or headers in your requests.

```yaml
model_list:
  - model_name: team-a-claude
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/TEAM_A_KEY

  - model_name: team-b-claude
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/TEAM_B_KEY
```

## See Also

- [MCP Gateway](mcp-gateway.md) - For standard Model Context Protocol support
- [Moat Mode](moat-mode.md) - For production security and isolation
- [LiteLLM Skills Docs](https://docs.litellm.ai/docs/proxy/skills) - Upstream documentation
