# ADR-0020: Governance Layer — Workspace Isolation & Dynamic Policies

**Status**: Accepted
**Date**: 2026-04-02

## Context
Enterprise deployments need multi-tenant isolation, per-key budgets, rate limits, 
model access control, and dynamic policy management. Portkey gates all governance 
behind Enterprise pricing. RouteIQ can be the first open-source AI gateway with 
production-grade governance.

## Decision
Implement a three-tier governance system:
1. **Workspace isolation**: Org → Workspace → API Key hierarchy
2. **Key governance**: Per-key scopes, budgets, rate limits, model access, config enforcement
3. **Dynamic policies**: Condition-based policies with group-by dimensions (CRUD API)

### Architecture
- `governance.py` — workspace config, key governance, governance engine
- `usage_policies.py` — dynamic policy engine with Redis-backed counters
- Routes in `routes/config.py` — admin CRUD API
- Integrates with existing `rbac.py`, `quota.py`, `policy_engine.py`

## Consequences
### Positive
- First open-source AI gateway with Portkey-class governance
- Foundation for enterprise self-service
- Compatible with existing LiteLLM team/key management

### Negative
- Adds complexity to the request path
- Requires Redis for counter storage (optional — degrades gracefully)
