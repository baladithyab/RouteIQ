# ADR-0023: Guardrail Policy Enforcement via Callback Bridge

**Status**: Accepted
**Date**: 2026-04-02

## Context
The GuardrailPolicyEngine (ADR-0020) defines 14 check types with CRUD API.
Input guardrails must block requests before they reach the LLM (saving cost).
Output guardrails must inspect responses (logging only — response already sent).

## Decision
Wire guardrails into the PluginCallbackBridge:
- Input: `async_log_pre_api_call()` — after plugin hooks, before LLM call
- Output: `async_log_success_event()` — after plugin hooks, in response path

Status codes:
- 446: Input guardrail denied (custom, signals "guardrail block" to clients)
- Response-path violations are logged but never block (too late)

## Implementation
- `_evaluate_input_guardrails()` resolves workspace from governance context
- Builds request_data dict from kwargs (model, messages, tools, metadata)
- Calls `GuardrailPolicyEngine.evaluate_input()`
- DENY results → HTTPException(446) with structured guardrail detail
- LOG/ALERT results → logger.info
- `_evaluate_output_guardrails()` extracts response content, logs violations
- `_extract_response_content()` handles ModelResponse, dict, and string types

## Consequences
### Positive
- Input guardrails save cost by blocking before LLM call
- 446 status code enables client-side retry/fallback logic
- Workspace scoping inherited from governance context (ADR-0022)
- Plugin-delegated checks (PII, content filter, prompt injection) reuse existing plugins

### Negative
- Output guardrails are advisory only (response already committed)
- Regex checks on large responses could add latency (mitigated by admin-only patterns)
