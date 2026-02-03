# MCP Gateway - Model Context Protocol

> **Attribution**:
> RouteIQ is built on top of upstream [LiteLLM](https://github.com/BerriAI/litellm) for proxy/API compatibility and [LLMRouter](https://github.com/ulab-uiuc/LLMRouter) for ML routing.

**RouteIQ Gateway** includes a fully integrated MCP (Model Context Protocol) Gateway, allowing you to bridge MCP servers to LLMs and expose tools and resources securely.

This guide covers the MCP (Model Context Protocol) gateway functionality for extending LLMs with external tools and data sources.

## Overview

The MCP Gateway acts as a centralized hub for Model Context Protocol interactions. It allows LLMs to:

- Access external tools (search, file operations, APIs)
- Query data sources and databases
- Interact with custom services
- Use resources from MCP servers

## Surface Contract

RouteIQ exposes three distinct surfaces for MCP interaction, each serving different needs:

| Surface | Base Path | Protocol | Intended Consumers | Auth Requirements | Status |
|---------|-----------|----------|--------------------|-------------------|--------|
| **RouteIQ Management** | `/llmrouter/mcp/*` | REST (JSON) | Admins, Internal Tools, CI/CD | Bearer Token (Master/Admin Key) | ✅ **Production** |
| **Upstream Parity** | `/v1/mcp/*` | REST (JSON) | LiteLLM-compatible Clients | Bearer Token (User/Admin Key) | ✅ **Production** |
| **Native MCP** | `/mcp/*` | JSON-RPC / SSE | Native MCP Clients (Claude Desktop, IDEs) | Bearer Token (User Key) | 🚧 **Experimental** |

### 1. RouteIQ Management API (`/llmrouter/mcp/*`)
The primary interface for registering servers, managing configuration, and debugging. It uses standard REST conventions and is the most stable surface.

### 2. Upstream Parity API (`/v1/mcp/*`)
Provides compatibility with the upstream LiteLLM API contract. Use this if you are migrating existing LiteLLM integrations or using tools built for the standard LiteLLM proxy.

### 3. Native MCP Protocol (`/mcp/*`)
**Status: Experimental / In-Progress**
Exposes native MCP endpoints (JSON-RPC over SSE or HTTP) for direct connection by MCP clients like Claude Desktop. This surface is feature-flagged via `MCP_PROTOCOL_PROXY_ENABLED`.

> **Note on Anthropic Skills:**
> If you are looking for Anthropic's "Computer Use" or "Bash" skills, those are distinct from MCP. See the [Skills Gateway Guide](skills-gateway.md) for details. We support both protocols.

## Local Validation

We provide a comprehensive validation script to verify the MCP Gateway end-to-end using a local test stack.

### Prerequisites
- Docker and Docker Compose
- `curl` and `jq` installed

### Running Validation
The validation script [`scripts/validate_mcp_gateway_curl.sh`](scripts/validate_mcp_gateway_curl.sh) runs against the `local-test` environment, which includes an MCP stub server for deterministic testing.

1. **Start the local test stack:**
   ```bash
   docker compose -f docker-compose.local-test.yml up -d
   ```

2. **Run the validation script:**
   ```bash
   # Run against local-test (single node)
   LB_URL=http://localhost:4010 \
   MASTER_KEY=sk-test-master-key \
   ADMIN_API_KEY=sk-test-admin-key \
   HA_MODE=false \
   ./scripts/validate_mcp_gateway_curl.sh
   ```

### What it Tests
- **Discovery:** Verifies the registry endpoint returns valid JSON.
- **Registration:** Registers the internal `mcp-stub-server`.
- **Tool Aggregation:** Checks that stub tools (`stub.echo`, `stub.sum`) appear in the global tool list.
- **Invocation:** Calls `stub.echo` via the REST API and verifies the response.
- **HA Sync:** (If HA mode enabled) Verifies registration propagates to replicas.

## Enabling MCP Gateway
