# Guardrails

RouteIQ ships with multiple guardrail plugins for content safety.

## Built-in Guardrail Plugins

| Plugin | Description |
|--------|-------------|
| `content_filter` | Block harmful content against configurable policies |
| `pii_guard` | Detect and redact PII (SSN, email, phone, etc.) |
| `prompt_injection_guard` | Detect prompt injection patterns |
| `llamaguard_plugin` | Meta's LlamaGuard safety classification |
| `bedrock_guardrails` | AWS Bedrock Guardrails integration |

## Enabling Guardrails

```bash
# Content filter
LLMROUTER_PLUGINS=litellm_llmrouter.gateway.plugins.content_filter.ContentFilterPlugin

# PII guard
LLMROUTER_PLUGINS=litellm_llmrouter.gateway.plugins.pii_guard.PIIGuardPlugin

# Prompt injection
LLMROUTER_PLUGINS=litellm_llmrouter.gateway.plugins.prompt_injection_guard.PromptInjectionGuardPlugin
```

Multiple plugins can be comma-separated.

## PII Detection

The PII Guard scans messages for:

- Social Security Numbers
- Email addresses
- Phone numbers
- Credit card numbers
- IP addresses

Detected PII can be **redacted** (replaced with placeholders) or **blocked** (request rejected).

## AWS Bedrock Guardrails

Integrate with AWS-managed guardrails:

```bash
LLMROUTER_PLUGINS=litellm_llmrouter.gateway.plugins.bedrock_guardrails.BedrockGuardrailsPlugin
```

Requires AWS credentials and a Bedrock Guardrail ID.
