# Governance API

APIs for managing workspaces, keys, and budgets.

## Workspace Management

### Create Workspace

```http
POST /workspace/new
Authorization: Bearer <master_key>
```

```json
{
  "workspace_name": "engineering",
  "models": ["gpt-4", "claude-3-opus"],
  "max_budget": 1000.00
}
```

### List Workspaces

```http
GET /workspace/list
Authorization: Bearer <master_key>
```

## Key Management

### Generate Key

```http
POST /key/generate
Authorization: Bearer <master_key>
```

```json
{
  "team_id": "engineering",
  "max_budget": 100.00,
  "models": ["gpt-4"],
  "duration": "30d"
}
```

### List Keys

```http
GET /key/list
Authorization: Bearer <master_key>
```

### Delete Key

```http
POST /key/delete
Authorization: Bearer <master_key>
```

## Budget and Usage

### Get Spend

```http
GET /spend/logs
Authorization: Bearer <master_key>
```

### Get Budget

```http
GET /budget/info
Authorization: Bearer <master_key>
```
