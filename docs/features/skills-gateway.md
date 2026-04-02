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

## Routing to Multiple Accounts

Route Skills requests to different Anthropic accounts:

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

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/skills/{namespace}/index.json` | Skills in a namespace |
| `GET` | `/skills/{namespace}__all.json` | All skills in a namespace |
| `GET` | `/_skill/discovery.json` | All available functions/classes |

## Well-Known Skills Index (Plugin)

The Skills Discovery plugin publishes a discoverable skills index at
`/.well-known/skills/`.

### Enabling

```bash
LLMROUTER_PLUGINS=litellm_llmrouter.gateway.plugins.skills_discovery.SkillsDiscoveryPlugin
```

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ROUTEIQ_SKILLS_DIR` | `./skills` or `./docs/skills` | Skill definitions directory |

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/.well-known/skills/index.json` | List all skills with metadata |
| `GET` | `/.well-known/skills/{skill}/SKILL.md` | Get skill markdown body |
| `GET` | `/.well-known/skills/{skill}/{path}` | Get any file from a skill |

### Skill Directory Structure

```
skills/
  my-skill/
    SKILL.md          # Required: skill description
    helper.py         # Optional: implementation files
  another-skill/
    SKILL.md
```

Skill names: lowercase letters, digits, hyphens only. Must start with a letter. 1-64 chars.

### Security

- **Path Traversal Protection**: All file access validated
- **Read-Only**: Plugin only publishes skills; does not execute them
- **Opt-In**: Disabled by default

### Caching

Skills index is cached in memory and refreshes automatically when the skills
directory is modified. Cache checks every 5 seconds.

## Moat Mode (Database Backing)

For production, back skills state with PostgreSQL instead of memory:

```bash
DATABASE_URL=postgresql://user:pass@host:5432/litellm
```

See [LiteLLM Skills Docs](https://docs.litellm.ai/docs/proxy/skills) for details.
