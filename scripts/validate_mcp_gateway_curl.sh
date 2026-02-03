#!/usr/bin/env bash
#
# Gate 6: MCP Gateway End-to-End Validation
# Black-box testing via curl against HA stack or local-test environment
#
# NOTE: LLMRouter MCP REST endpoints use /llmrouter/mcp/* prefix
# to avoid conflicts with LiteLLM's native /mcp (JSON-RPC/SSE) endpoint.
#
# Environment Variables (all optional, with sensible defaults):
#   LB_URL          - Load balancer or single gateway URL (default: http://localhost:8080 for HA, http://localhost:4010 for local-test)
#   REPLICA1_URL    - First replica URL for HA mode (default: http://localhost:4000)
#   REPLICA2_URL    - Second replica URL for HA mode (default: http://localhost:4001)
#   MASTER_KEY      - Master API key for user-level auth (default: sk-master-key-change-me)
#   ADMIN_API_KEY   - Admin API key for control-plane auth (default: uses MASTER_KEY)
#   MCP_STUB_URL    - MCP stub server URL as seen by containers (default: http://mcp-stub-server:9100/mcp)
#   MCP_STUB_LOCAL_URL - MCP stub server URL for local testing (default: http://localhost:9100/mcp)
#   HA_MODE         - Set to "true" for HA testing (3 URLs), "false" for single-node (default: auto-detect)
#
# Usage Examples:
#   # HA mode (default):
#   ./scripts/validate_mcp_gateway_curl.sh
#
#   # Local-test mode (single node on port 4010):
#   LB_URL=http://localhost:4010 MASTER_KEY=sk-test-master-key ADMIN_API_KEY=sk-test-admin-key HA_MODE=false ./scripts/validate_mcp_gateway_curl.sh
#
#   # Custom HA configuration:
#   LB_URL=http://lb.example.com:8080 REPLICA1_URL=http://node1:4000 REPLICA2_URL=http://node2:4000 MASTER_KEY=sk-prod-key ./scripts/validate_mcp_gateway_curl.sh
#

set -euo pipefail

# Configuration with environment variable overrides
# Auto-detect HA_MODE based on whether REPLICA1_URL is set or LB_URL contains non-HA port
_detect_ha_mode() {
    if [[ -n "${HA_MODE:-}" ]]; then
        [[ "${HA_MODE}" == "true" ]] && echo "true" || echo "false"
    elif [[ -n "${REPLICA1_URL:-}" ]] || [[ -n "${REPLICA2_URL:-}" ]]; then
        echo "true"
    elif [[ "${LB_URL:-}" == *":4010"* ]]; then
        echo "false"
    else
        echo "true"  # Default to HA mode
    fi
}

HA_MODE=$(_detect_ha_mode)

# URL defaults depend on mode
if [[ "$HA_MODE" == "true" ]]; then
    LB_URL="${LB_URL:-http://localhost:8080}"
    REPLICA1_URL="${REPLICA1_URL:-http://localhost:4000}"
    REPLICA2_URL="${REPLICA2_URL:-http://localhost:4001}"
else
    LB_URL="${LB_URL:-http://localhost:4010}"
    REPLICA1_URL="${LB_URL}"  # In single-node mode, all URLs point to the same place
    REPLICA2_URL="${LB_URL}"
fi

# Authentication keys
MASTER_KEY="${MASTER_KEY:-sk-master-key-change-me}"
ADMIN_API_KEY="${ADMIN_API_KEY:-${MASTER_KEY}}"

# MCP stub server URL as seen by containers (service name in Docker network)
STUB_CONTAINER_URL="${MCP_STUB_URL:-http://mcp-stub-server:9100/mcp}"
STUB_LOCAL_URL="${MCP_STUB_LOCAL_URL:-http://localhost:9100/mcp}"

# Auth headers - use Bearer token that also works as admin key
AUTH_HEADER="Authorization: Bearer ${MASTER_KEY}"
# X-Admin-API-Key header for explicit admin auth (preferred for control plane operations)
ADMIN_HEADER="X-Admin-API-Key: ${ADMIN_API_KEY}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
PASS_COUNT=0
FAIL_COUNT=0

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $*"
    ((PASS_COUNT++))
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $*"
    ((FAIL_COUNT++))
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

# Test helper that expects JSON response
# For POST/DELETE methods, also sends X-Admin-API-Key header for control-plane auth
test_json_endpoint() {
    local name="$1"
    local method="$2"
    local url="$3"
    local payload="$4"
    local expected_status="$5"

    log_info "Testing: $name"
    echo "  curl -sS -X $method '$url'"

    # Build common headers
    local headers=(-H "$AUTH_HEADER" -H "Content-Type: application/json" -H "Accept: application/json")
    
    # Add admin header for mutating operations (POST, PUT, DELETE)
    if [[ "$method" == "POST" || "$method" == "PUT" || "$method" == "DELETE" || "$method" == "PATCH" ]]; then
        headers+=(-H "$ADMIN_HEADER")
    fi

    if [ -n "$payload" ]; then
        response=$(curl -sS -w "\n%{http_code}" -X "$method" \
            "${headers[@]}" \
            -d "$payload" \
            "$url" 2>&1 || echo "CURL_ERROR")
    else
        response=$(curl -sS -w "\n%{http_code}" -X "$method" \
            "${headers[@]}" \
            "$url" 2>&1 || echo "CURL_ERROR")
    fi

    if [[ "$response" == "CURL_ERROR" ]]; then
        log_fail "$name - curl command failed"
        echo "  Error: curl execution failed"
        return 1
    fi

    # Extract status code (last line) and body (all but last line)
    # Using portable approach that works on both GNU and BSD (macOS)
    status_code=$(echo "$response" | tail -n 1)
    body=$(echo "$response" | sed '$d')

    echo "  Status: $status_code"
    echo "  Body: $body"

    if [ "$status_code" = "$expected_status" ]; then
        log_pass "$name (status=$status_code)"
        return 0
    else
        log_fail "$name (expected=$expected_status, got=$status_code)"
        return 1
    fi
}

# Main validation
main() {
    echo "=========================================="
    echo "  MCP Gateway E2E Validation (Gate 6)"
    echo "=========================================="
    echo "Mode:          ${HA_MODE} (HA_MODE=${HA_MODE})"
    echo "Gateway URL:   $LB_URL"
    if [[ "$HA_MODE" == "true" ]]; then
        echo "Replica 1:     $REPLICA1_URL"
        echo "Replica 2:     $REPLICA2_URL"
    fi
    echo "Master Key:    ${MASTER_KEY:0:8}..."
    echo "Admin Key:     ${ADMIN_API_KEY:0:8}..."
    echo "Stub (container): $STUB_CONTAINER_URL"
    echo "Stub (local):     $STUB_LOCAL_URL"
    echo ""
    echo "NOTE: Using /llmrouter/mcp/* REST endpoints"
    echo "      (avoiding LiteLLM's native /mcp JSON-RPC endpoint)"
    echo ""

    # ==========================================
    # 1. MCP Enablement + Discovery
    # ==========================================
    echo -e "\n${BLUE}=== 1. MCP Enablement + Discovery ===${NC}\n"

    test_json_endpoint \
        "GET /v1/llmrouter/mcp/registry.json" \
        "GET" \
        "$LB_URL/v1/llmrouter/mcp/registry.json" \
        "" \
        "200" || true

    test_json_endpoint \
        "GET /llmrouter/mcp/servers (gateway)" \
        "GET" \
        "$LB_URL/llmrouter/mcp/servers" \
        "" \
        "200" || true

    # ==========================================
    # 2. Verify Stub Server is Running
    # ==========================================
    echo -e "\n${BLUE}=== 2. Verify Stub MCP Server ===${NC}\n"

    # Note: This test is optional - stub server may not be running in local-test
    if curl -sS --connect-timeout 2 "$STUB_LOCAL_URL/tools" > /dev/null 2>&1; then
        test_json_endpoint \
            "GET stub /mcp/tools (local)" \
            "GET" \
            "$STUB_LOCAL_URL/tools" \
            "" \
            "200" || true
    else
        log_warn "Stub MCP server not reachable at $STUB_LOCAL_URL - skipping stub tests"
    fi

    # ==========================================
    # 3. Register Stub Server
    # ==========================================
    echo -e "\n${BLUE}=== 3. Register Stub Server ===${NC}\n"

    # LLMRouter MCP endpoints use REST API (not JSON-RPC)
    register_payload='{
      "server_id": "stub-mcp-1",
      "name": "test-stub",
      "url": "'"$STUB_CONTAINER_URL"'",
      "transport": "streamable_http",
      "tools": ["stub.echo", "stub.sum"],
      "resources": ["stub://resource/demo"],
      "metadata": {"environment": "e2e"}
    }'

    test_json_endpoint \
        "POST /llmrouter/mcp/servers (register stub)" \
        "POST" \
        "$LB_URL/llmrouter/mcp/servers" \
        "$register_payload" \
        "200" || true

    # Wait for propagation
    sleep 2

    # Verify registration on replicas (HA mode only)
    if [[ "$HA_MODE" == "true" ]]; then
        test_json_endpoint \
            "GET /llmrouter/mcp/servers (replica1)" \
            "GET" \
            "$REPLICA1_URL/llmrouter/mcp/servers" \
            "" \
            "200" || true

        test_json_endpoint \
            "GET /llmrouter/mcp/servers (replica2)" \
            "GET" \
            "$REPLICA2_URL/llmrouter/mcp/servers" \
            "" \
            "200" || true
    else
        log_info "Skipping replica verification (single-node mode)"
    fi

    # ==========================================
    # 4. Tools/Resources Aggregation
    # ==========================================
    echo -e "\n${BLUE}=== 4. Tools/Resources Aggregation ===${NC}\n"

    test_json_endpoint \
        "GET /llmrouter/mcp/tools (should include stub tools)" \
        "GET" \
        "$LB_URL/llmrouter/mcp/tools" \
        "" \
        "200" || true

    test_json_endpoint \
        "GET /llmrouter/mcp/resources (should include stub resources)" \
        "GET" \
        "$LB_URL/llmrouter/mcp/resources" \
        "" \
        "200" || true

    # ==========================================
    # 5. Tool Invocation
    # ==========================================
    echo -e "\n${BLUE}=== 5. Tool Invocation ===${NC}\n"

    # LLMRouter MCP tool call uses REST API format
    # Required fields: tool_name, arguments
    tool_call_payload='{
      "tool_name": "stub.echo",
      "arguments": {
        "text": "Hello from validation"
      }
    }'

    test_json_endpoint \
        "POST /llmrouter/mcp/tools/call (stub.echo)" \
        "POST" \
        "$LB_URL/llmrouter/mcp/tools/call" \
        "$tool_call_payload" \
        "200" || true

    # Error propagation - unknown tool (expects 404)
    unknown_tool_payload='{
      "tool_name": "stub.nonexistent",
      "arguments": {}
    }'

    test_json_endpoint \
        "POST /llmrouter/mcp/tools/call (unknown tool)" \
        "POST" \
        "$LB_URL/llmrouter/mcp/tools/call" \
        "$unknown_tool_payload" \
        "404" || true

    # ==========================================
    # 6. HA Sync Verification
    # ==========================================
    echo -e "\n${BLUE}=== 6. HA Sync Verification ===${NC}\n"

    if [[ "$HA_MODE" == "true" ]]; then
        log_info "Verifying server registered via LB is visible on both replicas..."

        # Check that both replicas see the server
        test_json_endpoint \
            "GET /llmrouter/mcp/servers/stub-mcp-1 (replica1)" \
            "GET" \
            "$REPLICA1_URL/llmrouter/mcp/servers/stub-mcp-1" \
            "" \
            "200" || true

        test_json_endpoint \
            "GET /llmrouter/mcp/servers/stub-mcp-1 (replica2)" \
            "GET" \
            "$REPLICA2_URL/llmrouter/mcp/servers/stub-mcp-1" \
            "" \
            "200" || true
    else
        log_info "Skipping HA sync verification (single-node mode)"
        # In single-node mode, just verify the server is registered on the gateway
        test_json_endpoint \
            "GET /llmrouter/mcp/servers/stub-mcp-1 (gateway)" \
            "GET" \
            "$LB_URL/llmrouter/mcp/servers/stub-mcp-1" \
            "" \
            "200" || true
    fi

    # ==========================================
    # 7. Performance Sanity Check
    # ==========================================
    echo -e "\n${BLUE}=== 7. Performance Sanity Check ===${NC}\n"

    log_info "Running 20 tool calls with concurrency 5..."

    perf_start=$(date +%s.%N)

    # Use GNU parallel if available, otherwise sequential
    if command -v parallel &> /dev/null; then
        log_info "Using GNU parallel for concurrent requests"
        seq 1 20 | parallel -j 5 --bar \
            "curl -sS -X POST -H '$AUTH_HEADER' -H 'Content-Type: application/json' \
             -d '$tool_call_payload' '$LB_URL/llmrouter/mcp/tools/call' > /dev/null && echo OK || echo FAIL" \
             2>/dev/null | tee /tmp/perf_results.txt

        success_count=$(grep -c "OK" /tmp/perf_results.txt || echo 0)
        fail_count=$(grep -c "FAIL" /tmp/perf_results.txt || echo 0)
    else
        log_warn "GNU parallel not found, running sequential test instead"
        success_count=0
        fail_count=0
        for _ in $(seq 1 20); do
            if curl -sS -X POST \
                -H "$AUTH_HEADER" \
                -H "Content-Type: application/json" \
                -d "$tool_call_payload" \
                "$LB_URL/llmrouter/mcp/tools/call" > /dev/null 2>&1; then
                ((success_count++))
            else
                ((fail_count++))
            fi
        done
    fi

    perf_end=$(date +%s.%N)
    perf_duration=$(echo "$perf_end - $perf_start" | bc)

    echo "  Total requests: 20"
    echo "  Success: $success_count"
    echo "  Failed: $fail_count"
    echo "  Duration: ${perf_duration}s"

    success_rate=$(echo "scale=1; $success_count * 100 / 20" | bc)
    echo "  Success rate: ${success_rate}%"

    if [ "$success_count" -ge 18 ]; then
        log_pass "Performance sanity check (success rate >= 90%)"
    else
        log_fail "Performance sanity check (success rate < 90%)"
    fi

    # ==========================================
    # 8. Cleanup
    # ==========================================
    echo -e "\n${BLUE}=== 8. Cleanup ===${NC}\n"

    test_json_endpoint \
        "DELETE /llmrouter/mcp/servers/stub-mcp-1 (cleanup)" \
        "DELETE" \
        "$LB_URL/llmrouter/mcp/servers/stub-mcp-1" \
        "" \
        "200" || true

    # ==========================================
    # Summary
    # ==========================================
    echo ""
    echo "=========================================="
    echo "  Validation Summary"
    echo "=========================================="
    echo -e "${GREEN}Passed: $PASS_COUNT${NC}"
    echo -e "${RED}Failed: $FAIL_COUNT${NC}"
    echo ""

    if [ "$FAIL_COUNT" -eq 0 ]; then
        echo -e "${GREEN}All validations passed!${NC}"
        exit 0
    else
        echo -e "${RED}Some validations failed.${NC}"
        exit 1
    fi
}

main "$@"
