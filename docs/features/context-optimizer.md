# Context Optimization

RouteIQ provides context optimization capabilities to reduce token usage
while preserving the semantic content of LLM requests.

## Overview

Context optimization applies lossless transforms to reduce the number of tokens
sent to LLM providers, resulting in:

- Lower costs per request
- Faster response times (fewer tokens to process)
- Ability to fit more context within model limits

## Optimization Strategies

### Prompt Deduplication

Automatically detects and removes duplicate content in conversation history,
particularly useful for long multi-turn conversations.

### Whitespace Normalization

Compresses unnecessary whitespace and formatting that consumes tokens
without adding semantic value.

### Context Window Management

Intelligently truncates and summarizes conversation history when approaching
model context limits, preserving the most relevant information.

## Configuration

Context optimization is configured via the gateway settings:

```yaml
general_settings:
  context_optimization:
    enabled: true
    strategies:
      - deduplication
      - whitespace_normalization
```

## Metrics

Token savings are tracked via OpenTelemetry metrics:

- `routeiq.context.tokens_saved` - Total tokens saved per request
- `routeiq.context.compression_ratio` - Compression ratio achieved
