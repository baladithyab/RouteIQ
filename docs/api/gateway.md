# Gateway API

RouteIQ exposes an OpenAI-compatible API for LLM inference.

## Base URL

```
http://localhost:4000
```

## Authentication

```bash
Authorization: Bearer sk-your-api-key
```

## Inference Endpoints

### Chat Completions

```http
POST /chat/completions
```

```json
{
  "model": "gpt-4",
  "messages": [{"role": "user", "content": "Hello!"}],
  "temperature": 0.7
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

## Health Endpoints

```http
GET /_health/live     # Liveness probe (unauthenticated)
GET /_health/ready    # Readiness probe (unauthenticated)
GET /health           # Basic health check
```

## Admin Endpoints

Require `X-Admin-API-Key` or admin bearer token:

```http
POST /router/reload        # Hot reload routing strategies
POST /config/reload        # Reload configuration
GET  /router/info          # Router status and strategy info
GET  /config/sync/status   # Config sync status
```
