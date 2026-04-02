# Prompt Management

RouteIQ supports prompt management capabilities for organizing, versioning,
and deploying prompts across your LLM applications.

## Overview

Prompt management in RouteIQ enables:

- **Prompt Templates**: Define reusable prompt templates with variable substitution
- **Version Control**: Track prompt versions and roll back when needed
- **A/B Testing**: Test prompt variations against each other
- **Audit Trail**: Track which prompts are used for each request

## Prompt Templates

Define prompts in your configuration:

```yaml
prompts:
  summarize:
    template: |
      Summarize the following text in {style} style:
      
      {text}
    variables:
      style:
        type: string
        default: concise
      text:
        type: string
        required: true
```

## Using Prompts

Reference prompts in API requests:

```bash
curl -X POST http://localhost:4000/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [
      {
        "role": "user",
        "content": "{{summarize:style=bullet,text=Your long text here}}"
      }
    ]
  }'
```

## Integration with Routing

Prompt metadata can influence routing decisions. For example, prompts tagged
as `reasoning` can be automatically routed to models with stronger reasoning
capabilities when using the `auto` routing profile.
