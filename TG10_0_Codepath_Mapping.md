# TG10.0 Codepath Mapping & Baseline Notes

## 1. Streaming Proxy Codepaths
**Goal:** Identify where line-buffering or blocking occurs in the streaming path.

*   **Entry Point:** `reference/litellm/litellm/proxy/proxy_server.py`
    *   `chat_completion` (line 5494) calls `base_llm_response_processor.base_process_llm_request`.
*   **Processing Logic:** `reference/litellm/litellm/proxy/common_request_processing.py`
    *   `base_process_llm_request` (line 625) handles the request.
    *   **Hotspot:** `create_response` (line 141) consumes the first chunk to check for errors:
        ```python
        first_chunk_value = await generator.__anext__()
        ```
        This delays the first byte and potentially buffers if the generator is not truly streaming.
*   **Generator:** `reference/litellm/litellm/proxy/proxy_server.py`
    *   `async_data_generator` (line 4574) iterates over `proxy_logging_obj.async_post_call_streaming_iterator_hook`.
    *   **Hotspot:** `str_so_far_parts` (line 4580) accumulates response segments.
*   **Passthrough:** `reference/litellm/litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py`
    *   `is_passthrough_request_streaming` (line 82) detects streaming.

**Implementation Sketch (TG10.1):**
*   Modify `create_response` to peek at the first chunk without consuming it if possible, or ensure it's yielded immediately.
*   Ensure `async_data_generator` yields chunks as soon as they are received, minimizing accumulation.

## 2. SSRF Validation & DNS Resolution
**Goal:** Identify blocking DNS calls or synchronous validation.

*   **Observation:** Explicit `validate_url` or `is_safe_url` functions were not found in the codebase search.
*   **Implicit Hotspot:** `reference/litellm/litellm/proxy/pass_through_endpoints/pass_through_endpoints.py`
    *   `response = await async_client.send(req, stream=stream)` (line 802).
    *   If `async_client` is not configured with a custom transport that handles DNS resolution asynchronously and safely, it might rely on default behavior which can be blocking or unsafe.
*   **Recommendation:** Implement explicit, async-safe URL validation before making requests. Use `aiohttp` or `httpx` with a custom resolver if needed.

**Implementation Sketch (TG10.2):**
*   Add a `validate_url` function in `reference/litellm/litellm/proxy/common_utils/http_parsing_utils.py`.
*   Use `socket.getaddrinfo` in a thread pool or use an async DNS resolver to check for private IPs.

## 3. Outbound HTTP Client Instantiation
**Goal:** Identify per-request client creation (no pooling).

*   **Hotspot 1:** `reference/litellm/litellm/integrations/custom_secret_manager.py`
    *   Line 68: `async with httpx.AsyncClient() as client:` - Created per request.
*   **Hotspot 2:** `reference/litellm/litellm/experimental_mcp_client/client.py`
    *   Line 238: `return httpx.AsyncClient(...)` - Created in factory method.
*   **Hotspot 3:** `reference/litellm/litellm/llms/openai/common_utils.py`
    *   Line 233: `return httpx.AsyncClient(...)` - Created if `litellm.aclient_session` is None.
*   **Shared Session:** `reference/litellm/litellm/proxy/proxy_server.py`
    *   `_initialize_shared_aiohttp_session` (line 647) creates a shared `aiohttp.ClientSession`.
    *   `shared_aiohttp_session` (line 1213) is the global variable.

**Implementation Sketch (TG10.3):**
*   Refactor Hotspots 1 & 2 to use `litellm.aclient_session` or `shared_aiohttp_session` where appropriate.
*   Ensure `AsyncHTTPHandler` in `reference/litellm/litellm/llms/custom_httpx/http_handler.py` reuses the global client.

## 4. MCP Transport Selection & SSE
**Goal:** Identify MCP transport logic and SSE gaps.

*   **MCP Server:** `reference/litellm/litellm/proxy/_experimental/mcp_server/server.py`
    *   `handle_streamable_http_mcp` (line 1843) handles HTTP/SSE.
*   **SSE Transport:** `reference/litellm/litellm/proxy/_experimental/mcp_server/sse_transport.py`
    *   `handle_sse_mcp` (line 1995 in `server.py` mounts it).
    *   Uses `anyio.create_memory_object_stream`.

**Implementation Sketch (TG10.4):**
*   Ensure `handle_streamable_http_mcp` correctly negotiates SSE vs HTTP.
*   Verify `sse_transport.py` handles connection keep-alive and proper event formatting.

## Test Suggestions (TG10.5)
*   **Streaming:** Create a test that measures TTFB (Time To First Byte) and inter-chunk latency. Ensure TTFB is < 50ms for local mock.
*   **SSRF:** Create a test case with a local private IP (e.g., `127.0.0.1`, `10.0.0.1`) and ensure it is blocked *asynchronously*.
*   **Connection Pooling:** Use `netstat` or similar to verify connection reuse during load testing.

## Rollback/Feature Flags
*   Use `general_settings` in `config.yaml` to toggle new behaviors.
*   Example: `disable_strict_ssrf_check: true` to disable the new SSRF validation.
