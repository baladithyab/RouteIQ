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

## Router-Specific Endpoints

### Get Current Routing Strategy

```http
GET /router/info
Authorization: Bearer sk-master-key
```

**Response:**
```json
{
  "routing_strategy": "llmrouter-knn",
  "model_path": "/app/models/knn_router.pt",
  "hot_reload": true,
  "last_reload": "2024-01-15T10:30:00Z"
}
```

### Reload Router Model

```http
POST /router/reload
Authorization: Bearer sk-master-key
```

### Get Routing Decision (Debug)

```http
POST /router/route
Authorization: Bearer sk-master-key
Content-Type: application/json

{
  "query": "Explain quantum entanglement"
}
```

**Response:**
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
