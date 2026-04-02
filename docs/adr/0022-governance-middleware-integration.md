# ADR-0022: Governance Enforcement via Routing Strategy Hook

**Status**: Accepted
**Date**: 2026-04-02

## Context
The GovernanceEngine (ADR-0020) was initially only accessible via CRUD API.
It needed to be wired into the actual request pipeline to enforce workspace
isolation, budgets, rate limits, and model access during LLM API calls.

Three integration points were considered:
1. ASGI middleware (like policy_engine.py)
2. LiteLLM callback (async_log_pre_api_call)  
3. Custom routing strategy (async_get_available_deployment Step 0)

## Decision
Wire governance enforcement into `RouteIQRoutingStrategy.async_get_available_deployment()`
as Step 0, before any routing logic. This was chosen because:
- The routing strategy has access to the parsed model name and API key
- It runs after auth but before model selection
- It can raise HTTPException to short-circuit the request
- It integrates naturally with the routing pipeline

## Implementation
- `_enforce_governance()` extracts API key, calls `GovernanceEngine.enforce()`
- Raises HTTPException(403) for model access denied
- Raises HTTPException(429) for budget/rate limit exceeded
- Stores governance context in metadata for downstream guardrails
- Fail-open: any unexpected exception logs and passes through

## Consequences
### Positive
- Zero overhead when no governance rules are configured (fast-path skip)
- Governance context propagated to guardrails and telemetry
- Effective routing profile from workspace applied to routing decision

### Negative
- Only works on async path (sync `get_available_deployment` skips governance)
- Budget checks use Redis read (not atomic check-and-decrement)
