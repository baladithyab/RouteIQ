# ADR-0003: Delete Redundant MCP Transport Code

**Status**: Accepted
**Date**: 2026-04-02
**Decision Makers**: RouteIQ Core Team

## Context

### MCP Surface Proliferation

RouteIQ v0.1.0 implemented a comprehensive Model Context Protocol (MCP) gateway
with multiple transport surfaces. Over time, as both RouteIQ and LiteLLM evolved,
this resulted in six competing URL hierarchies for MCP:

| Surface | Endpoint | Module | Lines | Source |
|---------|----------|--------|------:|--------|
| JSON-RPC | `POST /mcp` | `mcp_jsonrpc.py` | 868 | RouteIQ |
| SSE | `/mcp/sse` | `mcp_sse_transport.py` | 1,439 | RouteIQ |
| REST | `/mcp-rest/*` | `mcp_gateway.py` | (shared) | RouteIQ |
| Parity | `/v1/mcp/*` | `mcp_parity.py` | 1,072 | RouteIQ |
| Proxy | `/mcp-proxy/*` | routes/mcp.py | (shared) | RouteIQ |
| Native | `/mcp/*` | LiteLLM upstream | N/A | LiteLLM |

### LiteLLM v1.81+ MCP Capabilities

Starting with LiteLLM v1.81, the upstream project provides native, comprehensive
MCP support including:

- **JSON-RPC 2.0**: Full MCP protocol implementation with `initialize`,
  `tools/list`, `tools/call`, `resources/list`, `resources/read`,
  `prompts/list`, `prompts/get`
- **SSE transport**: Server-Sent Events for real-time streaming
- **REST API**: RESTful endpoints for MCP server CRUD operations
- **OAuth 2.1**: MCP authentication via OAuth flows
- **Management CRUD**: Database-backed `MCPServerTable` with health checks
- **Tool aggregation**: Automatic tool schema merging across registered servers

This means RouteIQ's custom MCP implementations were redundant with upstream.

### The Redundancy Cost

Maintaining parallel MCP implementations created real problems:

1. **Confusion**: Users didn't know which MCP endpoint to use. The 6 URL
   hierarchies served overlapping functions with subtly different behaviors.

2. **Bug surface**: Each transport needed independent security auditing, error
   handling, and testing. Bugs fixed in one surface weren't automatically fixed
   in others.

3. **Upgrade friction**: When LiteLLM improved its MCP support, RouteIQ's
   parallel implementations needed manual alignment, often lagging behind.

4. **Code volume**: 3,379 lines of redundant transport code that provided no
   value over upstream.

## Decision

### Delete Redundant Modules

Remove three MCP transport modules that duplicate LiteLLM's native capabilities:

| Module | Lines | Reason for Deletion |
|--------|------:|--------------------|
| `mcp_parity.py` | 1,072 | Upstream-compatible aliases now served natively by LiteLLM |
| `mcp_jsonrpc.py` | 868 | LiteLLM provides native JSON-RPC 2.0 MCP handling |
| `mcp_sse_transport.py` | 1,439 | LiteLLM provides native SSE transport |
| **Total** | **3,379** | |

### Retain RouteIQ Value-Add Modules

Keep modules that provide genuine value over upstream:

| Module | Lines | Reason for Retention |
|--------|------:|---------------------|
| `mcp_gateway.py` | ~1,292 | SSRF-validated server registration, HA-safe Redis-backed sync, tool invocation security gates |
| `mcp_tracing.py` | ~300 | OpenTelemetry instrumentation with RouteIQ's versioned telemetry contracts |
| `routes/mcp.py` | ~200 | Audit logging and RBAC wrappers around LiteLLM's native endpoints |

### Migration Path

1. **Feature flags**: `MCP_GATEWAY_ENABLED` continues to gate RouteIQ's MCP
   enhancements. When disabled, only LiteLLM's native MCP is available.

2. **URL consolidation**: After deletion, the MCP URL hierarchy simplifies to:
   - `/mcp` — LiteLLM native JSON-RPC (for Claude Desktop, IDE clients)
   - `/mcp/sse` — LiteLLM native SSE transport
   - `/v1/mcp/*` — LiteLLM native REST management
   - `/llmrouter/mcp/*` — RouteIQ enhancements (SSRF, HA sync, tracing)

3. **SSRF wrapping**: RouteIQ's SSRF validation (`url_security.py`) is applied
   as middleware/hooks on LiteLLM's native endpoints, not as a parallel
   implementation.

## Consequences

### Positive

- **-3,379 lines of code**: Direct reduction in maintenance burden, attack surface,
  and cognitive load.

- **No more URL confusion**: Users have a clear, single set of MCP endpoints
  backed by LiteLLM's well-tested implementation.

- **Automatic upstream improvements**: LiteLLM's MCP fixes and features are
  immediately available without RouteIQ needing to re-implement them.

- **Reduced test burden**: Three fewer transport implementations to test,
  covering security, error handling, and protocol compliance.

- **Simpler security auditing**: One MCP implementation to audit instead of four.

### Negative

- **Feature gap risk**: If LiteLLM's MCP implementation lacks a feature that
  RouteIQ's custom code provided, users may experience regressions. Mitigated
  by retaining `mcp_gateway.py` for RouteIQ-specific enhancements.

- **Breaking change**: Deployments using RouteIQ-specific MCP endpoints
  (`/mcp-rest/*`, custom JSON-RPC behaviors) need to migrate to LiteLLM's
  native endpoints. This requires documentation and a migration guide.

- **Loss of customization**: RouteIQ's custom SSE transport had some features
  (e.g., async queue-based message delivery with 202 responses) that may not
  exist in LiteLLM's SSE implementation. These are edge cases unlikely to
  affect most users.

## Alternatives Considered

### Alternative A: Keep All MCP Surfaces

Maintain both RouteIQ's and LiteLLM's MCP implementations side by side.

- **Pros**: No breaking changes; users can use whichever endpoint they prefer.
- **Cons**: Doubles maintenance cost; perpetuates URL confusion; 3,379 lines
  of code providing zero marginal value over upstream.
- **Rejected**: The maintenance cost and user confusion outweigh the
  compatibility benefit.

### Alternative B: Replace LiteLLM's MCP with RouteIQ's

Disable LiteLLM's native MCP and use only RouteIQ's implementation.

- **Pros**: Single implementation; RouteIQ has full control.
- **Cons**: Fights upstream; loses access to LiteLLM's MCP improvements;
  RouteIQ must maintain all MCP features indefinitely; contrary to the
  three-layer architecture principle of leveraging upstream.
- **Rejected**: Violates [ADR-0001](0001-three-layer-architecture.md) and
  [ADR-0017](0017-leverage-litellm-upstream.md).

### Alternative C: Gradual Deprecation with Feature Flags

Keep all modules but deprecate them behind feature flags, removing one at a
time over several releases.

- **Pros**: Gentler migration; users have time to adapt.
- **Cons**: Extends the maintenance burden; the redundancy problem persists
  during the deprecation period; feature flags add complexity.
- **Partially adopted**: Feature flags exist (`MCP_SSE_LEGACY_MODE`,
  `MCP_PROTOCOL_PROXY_ENABLED`) for the transition period, but the deletions
  are executed in a single release to minimize the dual-maintenance window.

## References

- `src/litellm_llmrouter/mcp_gateway.py` — Retained (SSRF, HA sync)
- `src/litellm_llmrouter/mcp_tracing.py` — Retained (OTel instrumentation)
- `src/litellm_llmrouter/routes/mcp.py` — Retained (audit/RBAC wrappers)
- [ADR-0001: Three-Layer Architecture](0001-three-layer-architecture.md)
- [ADR-0017: Leverage LiteLLM Upstream](0017-leverage-litellm-upstream.md)
- [TG3 Alternative Patterns Analysis](../architecture/tg3-alternative-patterns.md)
- LiteLLM MCP documentation: `reference/litellm/docs/my-website/docs/mcp.md`
