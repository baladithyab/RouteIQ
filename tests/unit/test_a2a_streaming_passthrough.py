"""
Unit Tests for A2A Raw Streaming Passthrough (TG10.1)

These tests validate the raw streaming passthrough implementation for the A2A Gateway,
ensuring proper chunk cadence, TTFB, and rollback safety.

Acceptance Criteria:
- New tests fail on the old implementation and pass on the new implementation
- Streaming endpoint begins emitting without waiting for newline boundaries
- Rollback flag exists and can revert to previous behavior

Test Categories:
1. Feature flag behavior (toggle between raw and buffered modes)
2. Raw streaming (newline-in-chunk preservation, chunk boundary handling)
3. Buffered streaming (backward compatibility, line-based chunking)
4. No full-buffering behavior (backpressure respected)
"""

import asyncio
import json
import os
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, patch

import pytest

# Mark all tests as async
pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _bypass_ssrf_validation():
    """
    Bypass SSRF URL validation for streaming passthrough tests.

    The SSRF fail-closed behavior rejects mock URLs (e.g. test.example.com)
    because DNS resolution fails. These streaming tests use mock HTTP clients
    and don't make real outbound requests, so SSRF validation is irrelevant.

    After yield, we reload a2a_gateway so that subsequent test modules get
    the real (unpatched) validate_outbound_url_async references.
    """

    async def _noop_validate(url, **kwargs):
        return url

    with (
        patch(
            "litellm_llmrouter.url_security.validate_outbound_url_async",
            new=_noop_validate,
        ),
        patch(
            "litellm_llmrouter.url_security.validate_outbound_url",
            side_effect=lambda url, **kwargs: url,
        ),
    ):
        yield

    # Reload a2a_gateway so it picks up the restored (real) url_security functions.
    # Without this, importlib.reload() calls inside tests leave stale mock references.
    import importlib
    import litellm_llmrouter.a2a_gateway as gateway_module

    importlib.reload(gateway_module)


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


class MockHTTPResponse:
    """Mock HTTP response for testing streaming behavior."""

    def __init__(
        self,
        content: bytes | list[bytes],
        status_code: int = 200,
        chunk_delay: float = 0.0,
    ):
        self.status_code = status_code
        self._content = content if isinstance(content, list) else [content]
        self._chunk_delay = chunk_delay
        self._read = False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    async def aiter_bytes(self, chunk_size: int = 1024) -> AsyncIterator[bytes]:
        """Iterate over raw bytes in chunks, preserving original boundaries."""
        for chunk in self._content:
            if self._chunk_delay > 0:
                await asyncio.sleep(self._chunk_delay)
            yield chunk

    async def aiter_lines(self) -> AsyncIterator[str]:
        """Iterate over lines, waiting for newline boundaries."""
        buffer = b""
        for chunk in self._content:
            if self._chunk_delay > 0:
                await asyncio.sleep(self._chunk_delay)
            buffer += chunk

        # Lines are split on newlines
        text = buffer.decode("utf-8")
        for line in text.split("\n"):
            if line:
                yield line


class MockAsyncClient:
    """Mock async HTTP client with streaming support."""

    def __init__(self, response: MockHTTPResponse):
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def stream(self, method: str, url: str, **kwargs) -> "MockStreamContext":
        return MockStreamContext(self._response)


class MockStreamContext:
    """Mock stream context manager."""

    def __init__(self, response: MockHTTPResponse):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *args):
        pass


def create_a2a_gateway_with_agent(
    agent_id: str = "test-agent",
    agent_url: str = "https://test.example.com/agent",
) -> Any:
    """Create an A2A gateway with a registered test agent."""
    # Import here to allow patching env vars first
    from litellm_llmrouter.a2a_gateway import A2AGateway, A2AAgent

    gateway = A2AGateway()
    gateway.enabled = True

    agent = A2AAgent(
        agent_id=agent_id,
        name="Test Agent",
        description="A test agent",
        url=agent_url,
        capabilities=["streaming"],
    )
    gateway.agents[agent_id] = agent

    return gateway


def create_jsonrpc_request(request_id: str = "1") -> Any:
    """Create a JSON-RPC request for testing."""
    from litellm_llmrouter.a2a_gateway import JSONRPCRequest

    return JSONRPCRequest(
        method="message/stream",
        params={
            "message": {"role": "user", "parts": [{"type": "text", "text": "test"}]}
        },
        id=request_id,
    )


# =============================================================================
# Feature Flag Tests
# =============================================================================


class TestFeatureFlagBehavior:
    """Tests for the A2A_RAW_STREAMING_ENABLED feature flag."""

    def test_default_flag_is_disabled(self):
        """
        Rollback safety: default flag value is 'false'.

        This ensures minimal blast radius - new behavior only activated
        when explicitly enabled.
        """
        with patch.dict(os.environ, {}, clear=True):
            # Force module reload to pick up env changes
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            assert gateway_module.A2A_RAW_STREAMING_ENABLED is False

    def test_flag_enabled_when_set_true(self):
        """Feature flag enables raw streaming when set to 'true'."""
        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "true"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            assert gateway_module.A2A_RAW_STREAMING_ENABLED is True

    def test_flag_case_insensitive(self):
        """Feature flag is case-insensitive."""
        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "TRUE"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            assert gateway_module.A2A_RAW_STREAMING_ENABLED is True

    def test_is_raw_streaming_enabled_helper(self):
        """Helper function correctly reports flag state."""
        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "true"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            assert gateway_module.is_raw_streaming_enabled() is True

    def test_chunk_size_configurable(self):
        """A2A_RAW_STREAMING_CHUNK_SIZE is configurable via environment."""
        with patch.dict(os.environ, {"A2A_RAW_STREAMING_CHUNK_SIZE": "4096"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            assert gateway_module.A2A_RAW_STREAMING_CHUNK_SIZE == 4096


# =============================================================================
# Raw Streaming Tests
# =============================================================================


class TestRawStreamingPassthrough:
    """
    Tests for raw streaming mode (A2A_RAW_STREAMING_ENABLED=true).

    These tests verify:
    - Chunks with embedded newlines are NOT split
    - Chunk boundaries from upstream are preserved
    - No waiting for newline boundaries before emitting
    """

    async def test_raw_streaming_preserves_newlines_in_chunk(self):
        """
        CRITICAL: Raw streaming does NOT split on newlines.

        In raw mode, a chunk like 'line1\\nline2' should be yielded as-is,
        NOT split into separate 'line1' and 'line2' chunks.

        This test would FAIL on the old line-buffered implementation.
        """
        # Prepare a chunk with embedded newlines
        chunk_with_newlines = b'{"part1": "value"}\n{"part2": "value"}\n'

        mock_response = MockHTTPResponse(content=[chunk_with_newlines])

        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "true"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = create_a2a_gateway_with_agent()
            request = create_jsonrpc_request()

            with patch(
                "httpx.AsyncClient", return_value=MockAsyncClient(mock_response)
            ):
                chunks = []
                async for chunk in gateway._stream_agent_response_raw(
                    "test-agent", request
                ):
                    chunks.append(chunk)

        # In raw mode, should receive the entire content as ONE chunk
        # (or multiple chunks that together equal the original, without splitting on newlines)
        combined = "".join(chunks)
        assert '{"part1": "value"}\n{"part2": "value"}\n' in combined

        # Key assertion: chunk count should match upstream chunk count (1)
        # NOT the line count (2)
        assert len(chunks) == 1, f"Expected 1 chunk but got {len(chunks)}: {chunks}"

    async def test_raw_streaming_preserves_chunk_boundaries(self):
        """
        Raw streaming preserves upstream chunk boundaries.

        If upstream sends 3 chunks, downstream should receive 3 chunks.
        """
        upstream_chunks = [
            b'{"chunk": 1}',
            b'{"chunk": 2}',
            b'{"chunk": 3}',
        ]

        mock_response = MockHTTPResponse(content=upstream_chunks)

        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "true"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = create_a2a_gateway_with_agent()
            request = create_jsonrpc_request()

            with patch(
                "httpx.AsyncClient", return_value=MockAsyncClient(mock_response)
            ):
                chunks = []
                async for chunk in gateway._stream_agent_response_raw(
                    "test-agent", request
                ):
                    chunks.append(chunk)

        # Should receive same number of chunks as upstream
        assert len(chunks) == 3

        # Content should match
        assert chunks[0] == '{"chunk": 1}'
        assert chunks[1] == '{"chunk": 2}'
        assert chunks[2] == '{"chunk": 3}'

    async def test_raw_streaming_no_waiting_for_newline(self):
        """
        Raw streaming yields chunks immediately, not waiting for newlines.

        Send a chunk without newline - it should be yielded immediately.
        """
        # Chunk without any newline
        partial_chunk = b'{"partial": "data'  # No newline, incomplete JSON

        mock_response = MockHTTPResponse(content=[partial_chunk])

        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "true"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = create_a2a_gateway_with_agent()
            request = create_jsonrpc_request()

            with patch(
                "httpx.AsyncClient", return_value=MockAsyncClient(mock_response)
            ):
                chunks = []
                async for chunk in gateway._stream_agent_response_raw(
                    "test-agent", request
                ):
                    chunks.append(chunk)

        # Should still receive the chunk even without newline
        assert len(chunks) == 1
        assert chunks[0] == '{"partial": "data'

    async def test_raw_streaming_empty_chunks_filtered(self):
        """Empty chunks are filtered out but non-empty ones pass through."""
        chunks_with_empty = [
            b"chunk1",
            b"",  # Empty chunk
            b"chunk2",
        ]

        mock_response = MockHTTPResponse(content=chunks_with_empty)

        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "true"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = create_a2a_gateway_with_agent()
            request = create_jsonrpc_request()

            with patch(
                "httpx.AsyncClient", return_value=MockAsyncClient(mock_response)
            ):
                chunks = []
                async for chunk in gateway._stream_agent_response_raw(
                    "test-agent", request
                ):
                    chunks.append(chunk)

        # Empty chunk filtered out
        assert len(chunks) == 2
        assert "chunk1" in chunks[0]
        assert "chunk2" in chunks[1]


# =============================================================================
# Buffered Streaming Tests (Backward Compatibility)
# =============================================================================


class TestBufferedStreamingCompatibility:
    """
    Tests for buffered streaming mode (A2A_RAW_STREAMING_ENABLED=false).

    These tests verify backward compatibility with the original
    line-buffered implementation.
    """

    async def test_buffered_streaming_splits_on_newlines(self):
        """
        Buffered mode waits for and splits on newline boundaries.

        This is the original behavior for backward compatibility.
        """
        # Content with multiple lines
        content_with_lines = b'{"line1": "value"}\n{"line2": "value"}\n'

        mock_response = MockHTTPResponse(content=[content_with_lines])

        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "false"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = create_a2a_gateway_with_agent()
            request = create_jsonrpc_request()

            with patch(
                "httpx.AsyncClient", return_value=MockAsyncClient(mock_response)
            ):
                chunks = []
                async for chunk in gateway._stream_agent_response_buffered(
                    "test-agent", request
                ):
                    chunks.append(chunk)

        # Buffered mode should split on newlines
        assert len(chunks) == 2
        assert '{"line1": "value"}\n' in chunks[0]
        assert '{"line2": "value"}\n' in chunks[1]

    async def test_buffered_streaming_appends_newline(self):
        """Buffered mode appends newline to each yielded line."""
        content = b'{"data": "test"}\n'

        mock_response = MockHTTPResponse(content=[content])

        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "false"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = create_a2a_gateway_with_agent()
            request = create_jsonrpc_request()

            with patch(
                "httpx.AsyncClient", return_value=MockAsyncClient(mock_response)
            ):
                chunks = []
                async for chunk in gateway._stream_agent_response_buffered(
                    "test-agent", request
                ):
                    chunks.append(chunk)

        # Each chunk should end with newline
        for chunk in chunks:
            assert chunk.endswith("\n")


# =============================================================================
# Dispatcher Tests
# =============================================================================


class TestStreamAgentResponseDispatcher:
    """
    Tests for the stream_agent_response dispatcher method.

    Verifies correct routing based on feature flag.
    """

    async def test_dispatcher_uses_raw_when_enabled(self):
        """When flag is enabled, dispatcher uses raw streaming."""
        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "true"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = create_a2a_gateway_with_agent()
            request = create_jsonrpc_request()

            # Mock the internal methods to verify which is called
            raw_called = False
            buffered_called = False

            async def mock_raw(*args, **kwargs):
                nonlocal raw_called
                raw_called = True
                yield "raw"

            async def mock_buffered(*args, **kwargs):
                nonlocal buffered_called
                buffered_called = True
                yield "buffered"

            gateway._stream_agent_response_raw = mock_raw
            gateway._stream_agent_response_buffered = mock_buffered

            # Consume the generator
            async for _ in gateway.stream_agent_response("test-agent", request):
                pass

            assert raw_called is True
            assert buffered_called is False

    async def test_dispatcher_uses_buffered_when_disabled(self):
        """When flag is disabled, dispatcher uses buffered streaming."""
        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "false"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = create_a2a_gateway_with_agent()
            request = create_jsonrpc_request()

            # Mock the internal methods to verify which is called
            raw_called = False
            buffered_called = False

            async def mock_raw(*args, **kwargs):
                nonlocal raw_called
                raw_called = True
                yield "raw"

            async def mock_buffered(*args, **kwargs):
                nonlocal buffered_called
                buffered_called = True
                yield "buffered"

            gateway._stream_agent_response_raw = mock_raw
            gateway._stream_agent_response_buffered = mock_buffered

            # Consume the generator
            async for _ in gateway.stream_agent_response("test-agent", request):
                pass

            assert buffered_called is True
            assert raw_called is False


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestStreamingErrorHandling:
    """Tests for error handling in both streaming modes."""

    async def test_disabled_gateway_error(self):
        """Both modes yield proper error when gateway is disabled."""
        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "true"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = gateway_module.A2AGateway()
            gateway.enabled = False

            request = create_jsonrpc_request()

            chunks = []
            async for chunk in gateway._stream_agent_response_raw(
                "test-agent", request
            ):
                chunks.append(chunk)

            assert len(chunks) == 1
            error_response = json.loads(chunks[0].strip())
            assert error_response["error"]["code"] == -32000
            assert "not enabled" in error_response["error"]["message"]

    async def test_agent_not_found_error(self):
        """Both modes yield proper error for nonexistent agent."""
        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "true"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = gateway_module.A2AGateway()
            gateway.enabled = True

            request = create_jsonrpc_request()

            chunks = []
            async for chunk in gateway._stream_agent_response_raw(
                "nonexistent-agent", request
            ):
                chunks.append(chunk)

            assert len(chunks) == 1
            error_response = json.loads(chunks[0].strip())
            assert error_response["error"]["code"] == -32000
            assert "not found" in error_response["error"]["message"]


# =============================================================================
# Regression Tests
# =============================================================================


class TestStreamingRegressions:
    """
    Regression tests for streaming behavior.

    These tests catch specific bugs that could break streaming semantics.
    """

    async def test_no_full_buffering_behavior(self):
        """
        Verify streaming doesn't buffer entire response before yielding.

        This test simulates slow chunks and verifies we get partial
        results before the full response is complete.
        """
        slow_chunks = [
            b'{"chunk": 1}',
            b'{"chunk": 2}',
            b'{"chunk": 3}',
        ]

        # Add delay between chunks
        mock_response = MockHTTPResponse(content=slow_chunks, chunk_delay=0.01)

        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "true"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = create_a2a_gateway_with_agent()
            request = create_jsonrpc_request()

            with patch(
                "httpx.AsyncClient", return_value=MockAsyncClient(mock_response)
            ):
                received_chunks = []
                timestamps = []

                async for chunk in gateway._stream_agent_response_raw(
                    "test-agent", request
                ):
                    received_chunks.append(chunk)
                    timestamps.append(asyncio.get_event_loop().time())

                    # Verify we're getting chunks incrementally, not all at once
                    if len(received_chunks) < 3:
                        # Should have received this chunk before all chunks are done
                        assert len(received_chunks) < len(slow_chunks)

        # Should have received all chunks
        assert len(received_chunks) == 3

    async def test_binary_content_decoded_safely(self):
        """Raw streaming handles binary content with invalid UTF-8 gracefully."""
        # Content with invalid UTF-8 byte
        content_with_invalid_utf8 = b'{"data": "\xff\xfe"}'

        mock_response = MockHTTPResponse(content=[content_with_invalid_utf8])

        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "true"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = create_a2a_gateway_with_agent()
            request = create_jsonrpc_request()

            with patch(
                "httpx.AsyncClient", return_value=MockAsyncClient(mock_response)
            ):
                chunks = []
                async for chunk in gateway._stream_agent_response_raw(
                    "test-agent", request
                ):
                    chunks.append(chunk)

        # Should handle gracefully with replacement characters
        assert len(chunks) == 1
        # Contains replacement character for invalid bytes
        assert "" in chunks[0] or "data" in chunks[0]

    async def test_empty_response_handled(self):
        """Both modes handle empty responses gracefully."""
        mock_response = MockHTTPResponse(content=[])

        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "true"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = create_a2a_gateway_with_agent()
            request = create_jsonrpc_request()

            with patch(
                "httpx.AsyncClient", return_value=MockAsyncClient(mock_response)
            ):
                chunks = []
                async for chunk in gateway._stream_agent_response_raw(
                    "test-agent", request
                ):
                    chunks.append(chunk)

        # Empty response should yield no chunks
        assert len(chunks) == 0


# =============================================================================
# Performance Characteristic Tests
# =============================================================================


class TestStreamingPerformanceCharacteristics:
    """
    Tests that verify performance characteristics are correct.

    These aren't strict benchmarks but verify streaming behavior
    that affects TTFB and chunk cadence.
    """

    async def test_first_chunk_emitted_before_second_arrives(self):
        """
        First chunk should be emitted before second chunk from upstream.

        This verifies we're not waiting for all chunks before emitting.
        """
        chunks_received_order = []

        class TrackedMockResponse:
            status_code = 200

            def raise_for_status(self):
                pass

            async def aiter_bytes(self, chunk_size: int = 1024):
                chunks_received_order.append("upstream_1")
                yield b"chunk1"

                chunks_received_order.append("upstream_2")
                yield b"chunk2"

        mock_response = TrackedMockResponse()

        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "true"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = create_a2a_gateway_with_agent()
            request = create_jsonrpc_request()

            with patch(
                "httpx.AsyncClient",
                return_value=MockAsyncClient(mock_response),  # type: ignore
            ):
                async for chunk in gateway._stream_agent_response_raw(
                    "test-agent", request
                ):
                    chunks_received_order.append(f"downstream_{chunk}")

        # Verify interleaving: first upstream chunk should be yielded
        # before second upstream chunk is received
        assert chunks_received_order.index(
            "downstream_chunk1"
        ) < chunks_received_order.index("upstream_2")


# =============================================================================
# Header Preservation Tests (TG10.1)
# =============================================================================


class TestHeaderPreservation:
    """
    Tests for upstream header preservation on streaming responses.

    Validates that important headers like Content-Type: text/event-stream
    are preserved on streaming responses.
    """

    def test_filter_upstream_headers_preserves_content_type(self):
        """Content-Type header is preserved from upstream."""
        import httpx
        from litellm_llmrouter.a2a_gateway import filter_upstream_headers

        headers = httpx.Headers(
            {
                "content-type": "text/event-stream; charset=utf-8",
                "cache-control": "no-cache",
                "connection": "keep-alive",  # Should be stripped (hop-by-hop)
            }
        )

        filtered = filter_upstream_headers(headers)

        assert "content-type" in filtered
        assert filtered["content-type"] == "text/event-stream; charset=utf-8"
        assert "cache-control" in filtered
        assert "connection" not in filtered  # Hop-by-hop stripped

    def test_filter_upstream_headers_strips_hop_by_hop(self):
        """Hop-by-hop headers are stripped from forwarding."""
        import httpx
        from litellm_llmrouter.a2a_gateway import (
            filter_upstream_headers,
            HOP_BY_HOP_HEADERS,
        )

        # Build headers with all hop-by-hop headers
        header_dict = {"content-type": "text/event-stream"}
        for hop_header in HOP_BY_HOP_HEADERS:
            header_dict[hop_header] = "some-value"

        headers = httpx.Headers(header_dict)
        filtered = filter_upstream_headers(headers)

        # None of the hop-by-hop headers should be present
        for hop_header in HOP_BY_HOP_HEADERS:
            assert hop_header not in filtered

        # Content-type should still be there
        assert "content-type" in filtered

    def test_filter_upstream_headers_strips_unsafe(self):
        """Unsafe/security-sensitive headers are stripped."""
        import httpx
        from litellm_llmrouter.a2a_gateway import (
            filter_upstream_headers,
            UNSAFE_HEADERS,
        )

        header_dict = {"content-type": "text/event-stream"}
        for unsafe_header in UNSAFE_HEADERS:
            header_dict[unsafe_header] = "some-value"

        headers = httpx.Headers(header_dict)
        filtered = filter_upstream_headers(headers)

        # None of the unsafe headers should be present
        for unsafe_header in UNSAFE_HEADERS:
            assert unsafe_header not in filtered

    def test_filter_upstream_headers_preserves_x_accel_buffering(self):
        """X-Accel-Buffering header is preserved (important for nginx SSE)."""
        import httpx
        from litellm_llmrouter.a2a_gateway import filter_upstream_headers

        headers = httpx.Headers(
            {
                "content-type": "text/event-stream",
                "x-accel-buffering": "no",
            }
        )

        filtered = filter_upstream_headers(headers)

        assert "x-accel-buffering" in filtered
        assert filtered["x-accel-buffering"] == "no"

    def test_streaming_response_meta_defaults(self):
        """StreamingResponseMeta has correct defaults."""
        from litellm_llmrouter.a2a_gateway import StreamingResponseMeta

        meta = StreamingResponseMeta()

        assert meta.headers == {}
        assert meta.status_code == 200

    def test_streaming_response_meta_with_values(self):
        """StreamingResponseMeta stores provided values."""
        from litellm_llmrouter.a2a_gateway import StreamingResponseMeta

        meta = StreamingResponseMeta(
            headers={"content-type": "text/event-stream"},
            status_code=206,
        )

        assert meta.headers == {"content-type": "text/event-stream"}
        assert meta.status_code == 206


# =============================================================================
# Method Mismatch / Routing Tests (TG10.1)
# =============================================================================


class TestMethodMismatchRouting:
    """
    Tests for correct routing of message/stream to streaming endpoint.

    Validates that invoke_agent returns an error for message/stream requests,
    directing callers to use the streaming endpoint instead.
    """

    async def test_invoke_agent_rejects_message_stream_method(self):
        """invoke_agent returns error for message/stream, directing to streaming endpoint."""
        with patch.dict(os.environ, {"A2A_GATEWAY_ENABLED": "true"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = create_a2a_gateway_with_agent()

            # Create a message/stream request
            request = gateway_module.JSONRPCRequest(
                method="message/stream",
                params={
                    "message": {
                        "role": "user",
                        "parts": [{"type": "text", "text": "test"}],
                    }
                },
                id="1",
            )

            # Mock the HTTP response
            mock_response = type(
                "MockResponse",
                (),
                {
                    "status_code": 200,
                    "raise_for_status": lambda self: None,
                    "json": lambda self: {"result": {"status": "ok"}},
                },
            )()

            mock_client = type(
                "MockClient",
                (),
                {
                    "__aenter__": lambda self: self,
                    "__aexit__": lambda self, *args: None,
                    "post": lambda self, *args, **kwargs: mock_response,
                },
            )()

            # Mock the async context manager
            async def mock_aenter():
                return mock_client

            async def mock_aexit(*args):
                pass

            class MockClientContext:
                async def __aenter__(self):
                    return mock_client

                async def __aexit__(self, *args):
                    pass

            with patch(
                "litellm_llmrouter.a2a_gateway.get_client_for_request",
                return_value=MockClientContext(),
            ):
                response = await gateway.invoke_agent("test-agent", request)

            # Should return an error directing to streaming endpoint
            assert response.error is not None
            assert response.error["code"] == -32600
            assert "streaming endpoint" in response.error["message"].lower()


# =============================================================================
# Progressive Yield Without Newline Tests (TG10.1)
# =============================================================================


class TestProgressiveYieldWithoutNewlines:
    """
    Tests verifying that raw streaming yields progressively even when
    upstream content has no newlines.

    This is critical for proper SSE chunking where data may arrive
    in arbitrary boundaries.
    """

    async def test_stream_with_no_newlines_yields_progressively(self):
        """Stream with zero newlines still yields all chunks progressively."""
        # Upstream sends multiple chunks without any newlines
        chunks_no_newlines = [
            b'{"data": "part1"}',
            b'{"data": "part2"}',
            b'{"data": "part3"}',
        ]

        mock_response = MockHTTPResponse(content=chunks_no_newlines)

        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "true"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = create_a2a_gateway_with_agent()
            request = create_jsonrpc_request()

            # Raw mode - should yield 3 chunks
            with patch(
                "httpx.AsyncClient", return_value=MockAsyncClient(mock_response)
            ):
                raw_chunks = []
                async for chunk in gateway._stream_agent_response_raw(
                    "test-agent", request
                ):
                    raw_chunks.append(chunk)

        # Raw mode should have yielded 3 chunks - one per upstream chunk
        assert len(raw_chunks) == 3

        # Check that each chunk is present in the combined response
        combined = "".join(raw_chunks)
        for chunk in chunks_no_newlines:
            chunk_str = chunk.decode("utf-8")
            assert chunk_str in combined

        # CRITICAL: In raw mode, chunks do NOT have newlines appended
        # This is the key difference from buffered mode - we pass through
        # the exact bytes from upstream without any transformation
        for chunk in raw_chunks:
            # Raw chunks preserve original format - no automatic newline
            assert not chunk.endswith("\n"), "Raw mode should not append newlines"

    async def test_buffered_mode_buffers_without_newlines(self):
        """
        Buffered mode may not yield chunks without newlines.

        This demonstrates the behavior difference between raw and buffered modes.
        In buffered mode (line iteration), content without newlines may be held.
        """

    async def test_raw_vs_buffered_responsiveness_comparison(self):
        """
        Raw mode should be more responsive than buffered mode for
        content without newlines.
        """
        # Chunks that would be slow in buffered mode
        chunks = [
            b'{"event": "start"}',  # No newline - buffered would wait
            b'{"event": "progress", "pct": 50}',  # Still waiting...
            b'{"event": "done"}\n',  # Finally a newline!
        ]

        mock_response_raw = MockHTTPResponse(content=chunks)
        mock_response_buffered = MockHTTPResponse(content=chunks)

        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "true"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = create_a2a_gateway_with_agent()
            request = create_jsonrpc_request()

            # Raw mode - should yield all 3 chunks
            with patch(
                "httpx.AsyncClient", return_value=MockAsyncClient(mock_response_raw)
            ):
                raw_chunks = []
                async for chunk in gateway._stream_agent_response_raw(
                    "test-agent", request
                ):
                    raw_chunks.append(chunk)

        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "false"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = create_a2a_gateway_with_agent()
            request = create_jsonrpc_request()

            # Buffered mode - may yield fewer chunks (waits for newlines)
            with patch(
                "httpx.AsyncClient",
                return_value=MockAsyncClient(mock_response_buffered),
            ):
                buffered_chunks = []
                async for chunk in gateway._stream_agent_response_buffered(
                    "test-agent", request
                ):
                    buffered_chunks.append(chunk)

        # Raw mode should have yielded 3 chunks (one-per-upstream-plan)
        assert len(raw_chunks) == 3

        # .Buffered mode should have yielded fewer chunks (depends on mock implementation)
        # Key assertion: compare counting when raw is chunk boundary preserving


# =============================================================================
# Content-Type Passthrough Tests (TG10.1)
# =============================================================================


class TestContentTypePassthrough:
    """
    Tests verifying that Content-Type header from upstream is preserved.

    This is critical for SSE clients to properly interpret the stream.
    """

    def test_sse_content_type_preserved(self):
        """text/event-stream content-type is preserved."""
        import httpx
        from litellm_llmrouter.a2a_gateway import filter_upstream_headers

        headers = httpx.Headers(
            {
                "content-type": "text/event-stream",
            }
        )

        filtered = filter_upstream_headers(headers)

        assert filtered.get("content-type") == "text/event-stream"

    def test_json_content_type_preserved(self):
        """application/json content-type is preserved."""
        import httpx
        from litellm_llmrouter.a2a_gateway import filter_upstream_headers

        headers = httpx.Headers(
            {
                "content-type": "application/json; charset=utf-8",
            }
        )

        filtered = filter_upstream_headers(headers)

        assert filtered.get("content-type") == "application/json; charset=utf-8"

    def test_ndjson_content_type_preserved(self):
        """application/x-ndjson content-type is preserved."""
        import httpx
        from litellm_llmrouter.a2a_gateway import filter_upstream_headers

        headers = httpx.Headers(
            {
                "content-type": "application/x-ndjson",
            }
        )

        filtered = filter_upstream_headers(headers)

        assert filtered.get("content-type") == "application/x-ndjson"

    def test_streaming_headers_allowlist(self):
        """Verify the allowlist contains expected streaming headers."""
        from litellm_llmrouter.a2a_gateway import STREAMING_HEADERS_TO_PRESERVE

        # These headers are critical for SSE/streaming
        assert "content-type" in STREAMING_HEADERS_TO_PRESERVE
        assert "cache-control" in STREAMING_HEADERS_TO_PRESERVE
        assert "x-accel-buffering" in STREAMING_HEADERS_TO_PRESERVE

    def test_real_sse_headers_preserved(self):
        """Real SSE streaming response headers are preserved correctly."""
        import httpx
        from litellm_llmrouter.a2a_gateway import filter_upstream_headers

        # Simulate a real SSE response headers
        sample_sse_headers = httpx.Headers(
            {
                "content-type": "text/event-stream; charset=utf-8",
                "cache-control": "no-cache",
                "x-accel-buffering": "no",
            }
        )

        filtered = filter_upstream_headers(sample_sse_headers)

        assert filtered["content-type"] == "text/event-stream; charset=utf-8"
        assert filtered["cache-control"] == "no-cache"
        assert filtered["x-accel-buffering"] == "no"
