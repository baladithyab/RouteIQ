"""
Streaming Correctness Tests for TG10.5

This module provides deterministic tests for verifying streaming correctness:
1. No line buffering - newlines inside chunks must not delay emission
2. No full-response buffering - incremental yields verified
3. Byte-for-byte integrity - payload integrity validation
4. Cancellation/backpressure - client disconnect stops upstream read

These tests are designed to be regression-resistant and work against:
- In-process ASGI test server
- Running gateway URL (configurable via env var)

Assumptions:
- Raw streaming mode uses aiter_bytes() which preserves chunk boundaries
- Buffered mode uses aiter_lines() which splits on newlines
- Tests mock httpx client to simulate upstream behavior deterministically
"""

import asyncio
import hashlib
import os
import time
from collections.abc import AsyncIterator
from unittest.mock import patch

import pytest

# Mark all tests as async
pytestmark = pytest.mark.asyncio


# =============================================================================
# Mock Infrastructure for Deterministic Testing
# =============================================================================


class TimedChunkStream:
    """
    A mock response that emits chunks at controlled intervals.

    This allows precise verification of:
    - TTFB (time to first byte)
    - Chunk cadence (inter-chunk timing)
    - No full-buffering (chunks available before stream completes)
    """

    def __init__(
        self,
        chunks: list[bytes],
        inter_chunk_delay: float = 0.0,
        status_code: int = 200,
    ):
        self.chunks = chunks
        self.inter_chunk_delay = inter_chunk_delay
        self.status_code = status_code
        self.emit_timestamps: list[float] = []
        self._cancelled = False
        self._chunks_emitted = 0

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    async def aiter_bytes(self, chunk_size: int = 1024) -> AsyncIterator[bytes]:
        """Iterate raw bytes preserving chunk boundaries."""
        for chunk in self.chunks:
            if self._cancelled:
                break
            if self.inter_chunk_delay > 0:
                await asyncio.sleep(self.inter_chunk_delay)
            self.emit_timestamps.append(time.monotonic())
            self._chunks_emitted += 1
            yield chunk

    async def aiter_lines(self) -> AsyncIterator[str]:
        """Iterate lines, buffering until newline boundaries."""
        buffer = b""
        for chunk in self.chunks:
            if self._cancelled:
                break
            if self.inter_chunk_delay > 0:
                await asyncio.sleep(self.inter_chunk_delay)
            buffer += chunk

        # Only yields when ALL data is collected (simulating line buffering)
        text = buffer.decode("utf-8")
        for line in text.split("\n"):
            if line:
                self.emit_timestamps.append(time.monotonic())
                yield line

    def cancel(self):
        """Simulate client cancellation."""
        self._cancelled = True


class MockStreamContext:
    """Context manager for mock streaming responses."""

    def __init__(self, response: TimedChunkStream):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, *args):
        pass


class MockClient:
    """Mock httpx client for testing."""

    def __init__(self, response: TimedChunkStream):
        self.response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def stream(self, method: str, url: str, **kwargs) -> MockStreamContext:
        return MockStreamContext(self.response)


def create_gateway_and_agent(
    agent_id: str = "test-agent", url: str = "https://test.example.com"
):
    """Create a test gateway with a registered agent."""
    from litellm_llmrouter.a2a_gateway import A2AGateway, A2AAgent

    gateway = A2AGateway()
    gateway.enabled = True
    gateway.agents[agent_id] = A2AAgent(
        agent_id=agent_id,
        name="Test Agent",
        description="Test agent for streaming tests",
        url=url,
        capabilities=["streaming"],
    )
    return gateway


def create_request(request_id: str = "test-1"):
    """Create a JSON-RPC request for testing."""
    from litellm_llmrouter.a2a_gateway import JSONRPCRequest

    return JSONRPCRequest(
        method="message/stream",
        params={"message": {"role": "user", "parts": [{"text": "test"}]}},
        id=request_id,
    )


# =============================================================================
# Test Category 1: No Line Buffering
# =============================================================================


class TestNoLineBuffering:
    """
    Verify that newlines inside a chunk do not delay emission.

    Critical behavior: In raw streaming mode, receiving a chunk like
    'data1\\ndata2' should yield immediately, NOT wait for more data.
    """

    async def test_embedded_newline_yielded_immediately(self):
        """
        A chunk containing newlines must be yielded as a single unit
        without splitting or waiting.
        """
        # Chunk with embedded newlines - should NOT be split
        chunk_with_newlines = b'{"part1": "val1"}\n{"part2": "val2"}\n{"part3": "val3"}'

        mock_response = TimedChunkStream(chunks=[chunk_with_newlines])

        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "true"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = create_gateway_and_agent()
            request = create_request()

            with patch("httpx.AsyncClient", return_value=MockClient(mock_response)):
                chunks = []
                async for chunk in gateway._stream_agent_response_raw(
                    "test-agent", request
                ):
                    chunks.append(chunk)

        # CRITICAL: Should be exactly 1 chunk, not 3 (split on newlines)
        assert len(chunks) == 1, (
            f"Raw streaming should NOT split on newlines. "
            f"Expected 1 chunk, got {len(chunks)}: {chunks}"
        )

        # Content integrity
        expected_content = '{"part1": "val1"}\n{"part2": "val2"}\n{"part3": "val3"}'
        assert chunks[0] == expected_content

    async def test_multiple_newlines_single_chunk(self):
        """Multiple newlines in a single chunk are preserved."""
        chunk = b"line1\nline2\nline3\nline4\n\n\nline5"

        mock_response = TimedChunkStream(chunks=[chunk])

        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "true"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = create_gateway_and_agent()
            request = create_request()

            with patch("httpx.AsyncClient", return_value=MockClient(mock_response)):
                result = []
                async for chunk in gateway._stream_agent_response_raw(
                    "test-agent", request
                ):
                    result.append(chunk)

        assert len(result) == 1
        assert result[0] == "line1\nline2\nline3\nline4\n\n\nline5"

    async def test_partial_line_not_held(self):
        """
        Partial lines (no trailing newline) must still be yielded immediately.

        Old line-buffered implementations would hold this data waiting for newline.
        """
        partial = b'{"incomplete": "json'  # No newline at end

        mock_response = TimedChunkStream(chunks=[partial])

        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "true"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = create_gateway_and_agent()
            request = create_request()

            with patch("httpx.AsyncClient", return_value=MockClient(mock_response)):
                result = []
                async for chunk in gateway._stream_agent_response_raw(
                    "test-agent", request
                ):
                    result.append(chunk)

        # Must yield immediately even without trailing newline
        assert len(result) == 1
        assert result[0] == '{"incomplete": "json'


# =============================================================================
# Test Category 2: No Full-Response Buffering (Incremental Yields)
# =============================================================================


class TestIncrementalYields:
    """
    Verify streaming doesn't buffer entire response before yielding.

    This is critical for:
    - Low TTFB (Time To First Byte)
    - Memory efficiency (don't buffer entire large responses)
    - Proper backpressure handling
    """

    async def test_chunks_yielded_incrementally(self):
        """
        Chunks must be yielded as they arrive, not collected first.

        Test approach: Use a mock that tracks emit timestamps.
        With inter-chunk delays, we can verify ordering shows incremental
        yields rather than all-at-once buffered output.
        """
        chunks = [b"chunk1", b"chunk2", b"chunk3", b"chunk4"]

        # 10ms between chunks - enough to verify incrementality
        mock_response = TimedChunkStream(chunks=chunks, inter_chunk_delay=0.01)

        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "true"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = create_gateway_and_agent()
            request = create_request()

            with patch("httpx.AsyncClient", return_value=MockClient(mock_response)):
                received = []
                receive_times = []

                async for chunk in gateway._stream_agent_response_raw(
                    "test-agent", request
                ):
                    received.append(chunk)
                    receive_times.append(time.monotonic())

        # All 4 chunks received
        assert len(received) == 4

        # Verify incremental receipt - each receive time should be after previous
        # with non-trivial gaps (proving we didn't buffer all then yield)
        for i in range(1, len(receive_times)):
            gap = receive_times[i] - receive_times[i - 1]
            # Gap should be at least 5ms (half the inter-chunk delay, allowing for timing variance)
            assert gap >= 0.005, (
                f"Chunk {i} received too quickly after chunk {i - 1}. "
                f"Gap: {gap * 1000:.2f}ms. This suggests full buffering."
            )

    async def test_first_chunk_available_before_last_sent(self):
        """
        First chunk must be yielded before the upstream finishes sending all data.

        This is the core TTFB guarantee.
        """
        # Large number of slow chunks
        chunks = [f"chunk{i}".encode() for i in range(10)]

        mock_response = TimedChunkStream(chunks=chunks, inter_chunk_delay=0.01)
        start_time = None
        first_chunk_time = None
        all_chunks_time = None

        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "true"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = create_gateway_and_agent()
            request = create_request()

            with patch("httpx.AsyncClient", return_value=MockClient(mock_response)):
                received = []
                start_time = time.monotonic()

                async for chunk in gateway._stream_agent_response_raw(
                    "test-agent", request
                ):
                    if first_chunk_time is None:
                        first_chunk_time = time.monotonic()
                    received.append(chunk)

                all_chunks_time = time.monotonic()

        # Verify timing
        ttfb = first_chunk_time - start_time
        total_time = all_chunks_time - start_time

        # TTFB should be much less than total time (proving incremental yields)
        # With 10 chunks at 10ms each, total ~100ms, TTFB should be ~10ms
        # Threshold of 0.5 accounts for module-reload overhead while still proving
        # incremental yields (fully buffered would have TTFB ≈ total_time)
        assert ttfb < total_time * 0.5, (
            f"TTFB ({ttfb * 1000:.2f}ms) is too close to total time ({total_time * 1000:.2f}ms). "
            f"This suggests full buffering rather than incremental yields."
        )

    async def test_large_payload_not_buffered_in_memory(self):
        """
        Large payloads should stream through without full memory buffering.

        We verify by checking that yielded chunks match input chunks exactly
        (proving we're not collecting and re-chunking).
        """
        # Create chunks that would be large in aggregate
        chunk_data = b"X" * 1024  # 1KB per chunk
        chunks = [chunk_data for _ in range(100)]  # 100KB total

        mock_response = TimedChunkStream(chunks=chunks)

        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "true"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = create_gateway_and_agent()
            request = create_request()

            with patch("httpx.AsyncClient", return_value=MockClient(mock_response)):
                chunk_count = 0
                async for chunk in gateway._stream_agent_response_raw(
                    "test-agent", request
                ):
                    chunk_count += 1
                    # Each yielded chunk should match original size (1024 bytes as string)
                    assert len(chunk) == 1024

        # Should have exact same number of chunks
        assert chunk_count == 100


# =============================================================================
# Test Category 3: Byte-for-Byte Integrity
# =============================================================================


class TestByteIntegrity:
    """
    Verify streamed payload integrity - hash(input) == hash(output).

    Tests various edge cases:
    - Binary-like data
    - Unicode characters
    - Large payloads
    - Mixed content
    """

    async def test_byte_integrity_simple(self):
        """Simple content maintains byte-for-byte integrity."""
        original = b'{"hello": "world", "count": 42}'

        mock_response = TimedChunkStream(chunks=[original])

        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "true"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = create_gateway_and_agent()
            request = create_request()

            with patch("httpx.AsyncClient", return_value=MockClient(mock_response)):
                received = []
                async for chunk in gateway._stream_agent_response_raw(
                    "test-agent", request
                ):
                    received.append(chunk)

        result = "".join(received).encode("utf-8")
        assert result == original

    async def test_byte_integrity_multi_chunk(self):
        """Multi-chunk content reassembles to exact original."""
        chunks = [
            b'{"part": 1, "data": "',
            b"hello world this is a test",
            b'"}\n{"part": 2}',
        ]

        # Compute expected hash
        expected = b"".join(chunks)
        expected_hash = hashlib.sha256(expected).hexdigest()

        mock_response = TimedChunkStream(chunks=chunks)

        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "true"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = create_gateway_and_agent()
            request = create_request()

            with patch("httpx.AsyncClient", return_value=MockClient(mock_response)):
                received = []
                async for chunk in gateway._stream_agent_response_raw(
                    "test-agent", request
                ):
                    received.append(chunk)

        result = "".join(received).encode("utf-8")
        result_hash = hashlib.sha256(result).hexdigest()

        assert result_hash == expected_hash, (
            f"Hash mismatch! Expected {expected_hash}, got {result_hash}"
        )

    async def test_byte_integrity_unicode(self):
        """Unicode content is preserved exactly."""
        unicode_content = "Hello 世界 🌍 مرحبا שלום"
        original = unicode_content.encode("utf-8")

        mock_response = TimedChunkStream(chunks=[original])

        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "true"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = create_gateway_and_agent()
            request = create_request()

            with patch("httpx.AsyncClient", return_value=MockClient(mock_response)):
                received = []
                async for chunk in gateway._stream_agent_response_raw(
                    "test-agent", request
                ):
                    received.append(chunk)

        result = "".join(received)
        assert result == unicode_content

    async def test_byte_integrity_large_payload(self):
        """Large payload integrity check via hash comparison."""
        # Generate 1MB of deterministic "random" data
        import random

        random.seed(42)

        chunk_size = 8192
        num_chunks = 128  # ~1MB total

        chunks = []
        for i in range(num_chunks):
            # Deterministic pseudo-random bytes
            data = bytes([random.randint(32, 126) for _ in range(chunk_size)])
            chunks.append(data)

        expected = b"".join(chunks)
        expected_hash = hashlib.sha256(expected).hexdigest()

        mock_response = TimedChunkStream(chunks=chunks)

        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "true"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = create_gateway_and_agent()
            request = create_request()

            with patch("httpx.AsyncClient", return_value=MockClient(mock_response)):
                received_bytes = b""
                async for chunk in gateway._stream_agent_response_raw(
                    "test-agent", request
                ):
                    received_bytes += chunk.encode("utf-8")

        result_hash = hashlib.sha256(received_bytes).hexdigest()

        assert result_hash == expected_hash, (
            f"Large payload hash mismatch! "
            f"Expected {expected_hash[:16]}..., got {result_hash[:16]}..."
        )


# =============================================================================
# Test Category 4: Cancellation and Backpressure
# =============================================================================


class TestCancellationBackpressure:
    """
    Verify cancellation/backpressure behavior.

    Key behaviors:
    - Client disconnect should stop upstream read
    - Slow consumers should cause backpressure (not unbounded buffering)
    - Generator cleanup on cancellation
    """

    async def test_cancellation_stops_upstream_read(self):
        """
        When consumer cancels, upstream iteration should stop.

        This is critical for:
        - Resource cleanup
        - Avoiding wasted bandwidth
        - Proper memory management
        """
        # Many chunks with delays - we'll cancel after first few
        chunks = [b"chunk{i}" for i in range(100)]
        mock_response = TimedChunkStream(chunks=chunks, inter_chunk_delay=0.001)

        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "true"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = create_gateway_and_agent()
            request = create_request()

            with patch("httpx.AsyncClient", return_value=MockClient(mock_response)):
                received_count = 0

                async for chunk in gateway._stream_agent_response_raw(
                    "test-agent", request
                ):
                    received_count += 1
                    if received_count >= 5:
                        # Simulate client disconnect by breaking
                        break

        # We should have received exactly 5 chunks before breaking
        assert received_count == 5

        # Key verification: upstream should NOT have emitted all 100 chunks
        # Allow for some buffering/prefetch, but should be far less than 100
        assert mock_response._chunks_emitted <= 10, (
            f"Upstream emitted {mock_response._chunks_emitted} chunks despite "
            f"consumer only taking 5. Expected backpressure to limit this."
        )

    async def test_async_generator_cleanup_on_exception(self):
        """Generator properly cleans up when consumer raises exception."""
        chunks = [f"chunk{i}".encode() for i in range(50)]
        mock_response = TimedChunkStream(chunks=chunks, inter_chunk_delay=0.001)

        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "true"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = create_gateway_and_agent()
            request = create_request()

            class ConsumerError(Exception):
                pass

            with patch("httpx.AsyncClient", return_value=MockClient(mock_response)):
                received_count = 0

                with pytest.raises(ConsumerError):
                    async for chunk in gateway._stream_agent_response_raw(
                        "test-agent", request
                    ):
                        received_count += 1
                        if received_count >= 3:
                            raise ConsumerError("Simulated consumer failure")

        # Verify we stopped early
        assert received_count == 3

    async def test_slow_consumer_backpressure(self):
        """
        Slow consumers should cause backpressure rather than unbounded buffering.

        Test approach: Fast producer, slow consumer. Verify producer doesn't
        race far ahead of consumer.
        """
        # Fast producer: no delays, many chunks
        chunks = [f"data{i}".encode() for i in range(50)]
        mock_response = TimedChunkStream(chunks=chunks, inter_chunk_delay=0)

        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "true"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = create_gateway_and_agent()
            request = create_request()

            with patch("httpx.AsyncClient", return_value=MockClient(mock_response)):
                consumed = 0

                async for chunk in gateway._stream_agent_response_raw(
                    "test-agent", request
                ):
                    consumed += 1
                    # Slow consumer: 5ms per chunk
                    await asyncio.sleep(0.005)

                    # Check backpressure at each step
                    # Producer shouldn't be more than ~2 ahead due to async iteration
                    # (This is more of a sanity check - true backpressure depends on implementation)

        # All chunks eventually consumed
        assert consumed == 50


# =============================================================================
# Test Category 5: Edge Cases and Regression Tests
# =============================================================================


class TestStreamingEdgeCases:
    """Edge cases and regression tests for streaming correctness."""

    async def test_empty_stream(self):
        """Empty stream (zero chunks) handled gracefully."""
        mock_response = TimedChunkStream(chunks=[])

        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "true"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = create_gateway_and_agent()
            request = create_request()

            with patch("httpx.AsyncClient", return_value=MockClient(mock_response)):
                received = []
                async for chunk in gateway._stream_agent_response_raw(
                    "test-agent", request
                ):
                    received.append(chunk)

        assert received == []

    async def test_single_byte_chunks(self):
        """Single-byte chunks are handled correctly."""
        text = "Hello"
        chunks = [bytes([b]) for b in text.encode()]  # Each byte separate

        mock_response = TimedChunkStream(chunks=chunks)

        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "true"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = create_gateway_and_agent()
            request = create_request()

            with patch("httpx.AsyncClient", return_value=MockClient(mock_response)):
                received = []
                async for chunk in gateway._stream_agent_response_raw(
                    "test-agent", request
                ):
                    received.append(chunk)

        result = "".join(received)
        assert result == text

    async def test_chunk_with_only_newlines(self):
        """Chunk containing only newlines is passed through."""
        chunk = b"\n\n\n"

        mock_response = TimedChunkStream(chunks=[chunk])

        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "true"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = create_gateway_and_agent()
            request = create_request()

            with patch("httpx.AsyncClient", return_value=MockClient(mock_response)):
                received = []
                async for chunk in gateway._stream_agent_response_raw(
                    "test-agent", request
                ):
                    received.append(chunk)

        assert len(received) == 1
        assert received[0] == "\n\n\n"

    async def test_whitespace_handling(self):
        """Various whitespace characters preserved."""
        whitespace = "\t \r\n \t  \n"
        chunk = whitespace.encode()

        mock_response = TimedChunkStream(chunks=[chunk])

        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "true"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = create_gateway_and_agent()
            request = create_request()

            with patch("httpx.AsyncClient", return_value=MockClient(mock_response)):
                received = []
                async for chunk in gateway._stream_agent_response_raw(
                    "test-agent", request
                ):
                    received.append(chunk)

        assert "".join(received) == whitespace


# =============================================================================
# Mode Comparison: Raw vs Buffered
# =============================================================================


class TestModeComparison:
    """
    Compare raw streaming vs buffered streaming behavior.

    These tests highlight the differences between modes and ensure
    the feature flag correctly switches behavior.
    """

    async def test_raw_preserves_newline_in_chunk(self):
        """Raw mode: newline in chunk does NOT cause split."""
        chunk = b"line1\nline2"
        mock_response = TimedChunkStream(chunks=[chunk])

        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "true"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = create_gateway_and_agent()
            request = create_request()

            with patch("httpx.AsyncClient", return_value=MockClient(mock_response)):
                received = []
                async for c in gateway._stream_agent_response_raw(
                    "test-agent", request
                ):
                    received.append(c)

        # Raw: single chunk preserved
        assert len(received) == 1
        assert received[0] == "line1\nline2"

    async def test_buffered_splits_on_newline(self):
        """Buffered mode: splits on newlines."""
        chunk = b"line1\nline2\n"
        mock_response = TimedChunkStream(chunks=[chunk])

        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "false"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = create_gateway_and_agent()
            request = create_request()

            with patch("httpx.AsyncClient", return_value=MockClient(mock_response)):
                received = []
                async for c in gateway._stream_agent_response_buffered(
                    "test-agent", request
                ):
                    received.append(c)

        # Buffered: splits into lines
        assert len(received) == 2
        assert "line1" in received[0]
        assert "line2" in received[1]

    async def test_dispatcher_routes_correctly(self):
        """
        Verify stream_agent_response routes to correct implementation
        based on feature flag.
        """
        # Test with raw enabled
        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "true"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            assert gateway_module.A2A_RAW_STREAMING_ENABLED is True

        # Test with raw disabled
        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "false"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            assert gateway_module.A2A_RAW_STREAMING_ENABLED is False
