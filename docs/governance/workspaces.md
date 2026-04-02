# Workspaces

Workspaces provide isolated environments for teams to use AI services
with independent budgets, model access, and configuration.

## Overview

Each workspace can have:

- Its own API keys and authentication
- Model access lists (which models the team can use)
- Budget limits (monthly/daily spend caps)
- Rate limits (requests per minute/hour)
- Custom routing profiles

## Configuration

Workspaces are managed through the admin API:

```bash
curl -X POST http://localhost:4000/workspace/new \
  -H "Authorization: Bearer $MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "workspace_name": "engineering",
    "models": ["gpt-4", "claude-3-opus"],
    "max_budget": 1000.00,
    "budget_duration": "monthly"
  }'
```

## Key Management

Generate API keys scoped to a workspace:

```bash
curl -X POST http://localhost:4000/key/generate \
  -H "Authorization: Bearer $MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "team_id": "engineering",
    "max_budget": 100.00,
    "models": ["gpt-4"]
  }'
```
