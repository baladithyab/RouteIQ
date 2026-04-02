# Skills Gateway

The Skills Gateway exposes Anthropic's agentic capabilities (Computer Use, Bash,
Text Editor) through standardized API endpoints.

## Overview

Skills are distinct from [MCP](mcp-gateway.md):

- **Skills**: Anthropic-specific agentic capabilities
- **MCP**: Open standard for connecting LLMs to tools and data sources

RouteIQ supports both simultaneously.

## Supported Skills

| Skill | Description |
|-------|-------------|
| Computer Use | Control a virtual desktop environment |
| Bash | Execute shell commands |
| Text Editor | View and edit files programmatically |

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/skills` | Create/invoke a skill |
| `GET` | `/v1/skills` | List available skills |
| `GET` | `/v1/skills/{id}` | Get skill details |
| `DELETE` | `/v1/skills/{id}` | Remove a skill |

## Configuration

### 1. Set API Key

```bash
ANTHROPIC_API_KEY=sk-ant-...
```

### 2. Configure Model

```yaml
model_list:
  - model_name: claude-3-5-sonnet
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY
```

## Authentication

```bash
Authorization: Bearer sk-your-proxy-key
```

## Plugin

The Skills Discovery plugin registers and manages Anthropic skills as MCP-compatible tools:

```bash
LLMROUTER_PLUGINS=litellm_llmrouter.gateway.plugins.skills_discovery.SkillsDiscoveryPlugin
```
