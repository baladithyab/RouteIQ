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

## Skills Discovery

You can discover the available skills using the following endpoints:

| Method | Endpoint | Description |
|-------|---------|-------------|
| `GET` | /skills/{namespace}/index.json | Gets the available skills in a single namespace, respecting the cache size and mutual exclusivity rules. |
| `GET` | /skills/{namespace}__all.json | Gets a flat list of all skills in a single namespace across all available functions and classes. |
| `GET` | /_skill/discovery.json | Gets the available functions and classes across all loaded Python say modules as a JSON array, along with their namespaces and paths. |

The JSON responses for these endpoints are of the format:

     {
       "namespace": "<namespace>",
       "name": "<function name>",
       "docstring": "<Optional: function docstring>",
       "args": [
         {
           "name": "<arg name>",
           "type": "<arg type>"
         },
         ...
       ],
       "returns": "<return type>"
     }

Where `<namespace>` is the module path, `<function name>` is the name of the function, and any relative imports are * referenced, e.g., `routeiq.say.compose.*`. The endpoints align with litellm's invocation patterns—so for example, `routeiq.say.compose.transform_text.*`:
​​​​​- `routeiq.say.compose.transform_text.TextTransformer.transform` * `POST /v1/skills/TextTransformer/transform`
​​​​​- `routeiq.say.compose.transform_text.TextTransformer`

**Note:**
* The Skill Discovery endpoints retrieve available functions from registered skills via `highly_authenticated_client.function_list_client.get_skills_names_by_module_path`.
* The cache size on `say_module` has a default value of 20 in Moat Mode.

## Well-Known Skills Index (Plugin)

RouteIQ provides an optional plugin for publishing a discoverable skills index at the well-known path `/.well-known/skills/`. This follows the progressive disclosure pattern for skill catalogs.

### Enabling the Plugin

Add the plugin to your `LLMROUTER_PLUGINS` environment variable:

```bash
export LLMROUTER_PLUGINS=litellm_llmrouter.gateway.plugins.skills_discovery.SkillsDiscoveryPlugin
```

### Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `ROUTEIQ_SKILLS_DIR` | Directory containing skill definitions | `./skills` or `./docs/skills` |

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/.well-known/skills/index.json` | List all available skills with metadata |
| `GET` | `/.well-known/skills/{skill}/SKILL.md` | Get the markdown body for a skill |
| `GET` | `/.well-known/skills/{skill}/{path}` | Get any file from a skill directory |

### Skill Directory Structure

Each skill is a subdirectory in the skills directory:

```
skills/
├── my-skill/
│   ├── SKILL.md          # Required: skill description
│   ├── helper.py         # Optional: implementation files
│   └── data/
│       └── config.json   # Optional: nested files
└── another-skill/
    └── SKILL.md
```

### Skill Naming Constraints

Skill directory names must follow these rules:
- Lowercase letters, digits, and hyphens only
- Must start with a letter
- 1-64 characters in length

Examples: `my-skill`, `code-analyzer`, `data-processor-v2`

### Index Response Format

```json
{
  "skills": [
    {
      "name": "my-skill",
      "description": "First paragraph from SKILL.md",
      "files": ["SKILL.md", "helper.py", "data/config.json"]
    }
  ]
}
```

### Security

- **Path Traversal Protection**: All file access is validated to prevent directory traversal attacks.
- **Read-Only**: The plugin only publishes skills; it does not execute them.
- **Opt-In**: The plugin is disabled by default and must be explicitly enabled.

### Caching

The skills index is cached in memory and refreshes automatically when the skills directory is modified. The cache checks for changes every 5 seconds.

## See Also

- [MCP Gateway](mcp-gateway.md) - For standard Model Context Protocol support
- [Moat Mode](moat-mode.md) - For production security and isolation
- [LiteLLM Skills Docs](https://docs.litellm.ai/docs/proxy/skills) - Upstream documentation
