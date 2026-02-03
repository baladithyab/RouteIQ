#!/usr/bin/env python3
"""
Deterministic Streaming Stub Server for Performance Testing

This server provides a streaming endpoint that emits precisely-timed
markers for validating TTFB and chunk cadence through the gateway.

Features:
- Configurable marker count and interval
- First marker emitted immediately (for TTFB validation)
- Subsequent markers at fixed intervals
- JSON-RPC 2.0 compatible responses
- SSE and raw streaming support

Usage:
    uv run python scripts/streaming_stub_server.py

Environment Variables:
    STREAMING_STUB_PORT: Port to listen on (default: 9200)
    STREAMING_STUB_MARKER_COUNT: Number of markers to emit (default: 20)
    STREAMING_STUB_INTERVAL_MS: Interval between markers in ms (default: 100)
"""

import asyncio
import json
import os
import time
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request

try:
    from fastapi.responses import StreamingResponse
except ImportError:
    from starlette.responses import StreamingResponse

import uvicorn

app = FastAPI(
    title="Streaming Stub Server",
    description="Deterministic streaming endpoint for TTFB/cadence testing",
    version="1.0.0",
)

# Configuration from environment
STUB_PORT = int(os.getenv("STREAMING_STUB_PORT", "9200"))
MARKER_COUNT = int(os.getenv("STREAMING_STUB_MARKER_COUNT", "20"))
INTERVAL_MS = float(os.getenv("STREAMING_STUB_INTERVAL_MS", "100"))

# Convert interval to seconds
INTERVAL_S = INTERVAL_MS / 1000.0


async def generate_markers(
    request_id: str | int | None = None,
    marker_count: int = MARKER_COUNT,
    interval_s: float = INTERVAL_S,
) -> AsyncIterator[bytes]:
    """
    Generate streaming markers at precise intervals.

    First marker is emitted immediately for accurate TTFB measurement.
    Subsequent markers are emitted at the specified interval.

    Each marker is a JSON-RPC 2.0 notification with:
    - method: "stream/marker"
    - params: {marker_index, timestamp_ms, elapsed_ms}
    """
    start_time = time.monotonic()

    for i in range(marker_count):
        current_time = time.monotonic()
        elapsed_ms = (current_time - start_time) * 1000

        # Create marker payload
        marker = {
            "jsonrpc": "2.0",
            "method": "stream/marker",
            "params": {
                "marker_index": i,
                "total_markers": marker_count,
                "timestamp_ms": int(time.time() * 1000),
                "elapsed_ms": round(elapsed_ms, 3),
                "request_id": request_id,
            },
        }

        # Yield as newline-delimited JSON
        yield (json.dumps(marker) + "\n").encode("utf-8")

        # Wait for next interval (but NOT before first marker for TTFB accuracy)
        if i < marker_count - 1:
            await asyncio.sleep(interval_s)


async def generate_sse_markers(
    request_id: str | int | None = None,
    marker_count: int = MARKER_COUNT,
    interval_s: float = INTERVAL_S,
) -> AsyncIterator[bytes]:
    """
    Generate SSE-formatted streaming markers.

    Each marker is sent as an SSE "data" event with JSON payload.
    """
    start_time = time.monotonic()

    for i in range(marker_count):
        current_time = time.monotonic()
        elapsed_ms = (current_time - start_time) * 1000

        # Create marker payload
        marker = {
            "marker_index": i,
            "total_markers": marker_count,
            "timestamp_ms": int(time.time() * 1000),
            "elapsed_ms": round(elapsed_ms, 3),
            "request_id": request_id,
        }

        # SSE format: data: <json>\n\n
        yield f"data: {json.dumps(marker)}\n\n".encode("utf-8")

        # Wait for next interval
        if i < marker_count - 1:
            await asyncio.sleep(interval_s)

    # Send end event
    yield b"event: end\ndata: {}\n\n"


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "streaming-stub-server",
        "config": {
            "marker_count": MARKER_COUNT,
            "interval_ms": INTERVAL_MS,
        },
    }


@app.get("/stream")
async def stream_get(
    markers: int | None = None,
    interval_ms: float | None = None,
) -> StreamingResponse:
    """
    GET streaming endpoint for raw marker stream.

    Query params:
        markers: Number of markers (default: MARKER_COUNT)
        interval_ms: Interval between markers (default: INTERVAL_MS)
    """
    marker_count = markers or MARKER_COUNT
    interval_s = (interval_ms or INTERVAL_MS) / 1000.0

    return StreamingResponse(
        generate_markers(
            request_id="GET",
            marker_count=marker_count,
            interval_s=interval_s,
        ),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/stream")
async def stream_post(request: Request) -> StreamingResponse:
    """
    POST streaming endpoint for JSON-RPC style requests.

    Accepts JSON-RPC 2.0 requests with optional params:
    - marker_count: Number of markers to emit
    - interval_ms: Interval between markers
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    # Extract request ID and params
    request_id = body.get("id", "POST")
    params = body.get("params", {})
    marker_count = params.get("marker_count", MARKER_COUNT)
    interval_ms = params.get("interval_ms", INTERVAL_MS)
    interval_s = interval_ms / 1000.0

    return StreamingResponse(
        generate_markers(
            request_id=request_id,
            marker_count=marker_count,
            interval_s=interval_s,
        ),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/stream/sse")
async def stream_sse(
    markers: int | None = None,
    interval_ms: float | None = None,
) -> StreamingResponse:
    """
    SSE streaming endpoint.

    Returns Server-Sent Events formatted stream.
    """
    marker_count = markers or MARKER_COUNT
    interval_s = (interval_ms or INTERVAL_MS) / 1000.0

    return StreamingResponse(
        generate_sse_markers(
            request_id="SSE",
            marker_count=marker_count,
            interval_s=interval_s,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# A2A-style endpoint for gateway passthrough testing
@app.post("/a2a")
async def a2a_stream(request: Request) -> StreamingResponse:
    """
    A2A-style streaming endpoint.

    Mimics an A2A agent that streams JSON-RPC 2.0 responses.
    Used for gateway passthrough testing.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    # Extract request ID
    request_id = body.get("id", "a2a-request")
    params = body.get("params", {})
    marker_count = params.get("marker_count", MARKER_COUNT)
    interval_ms = params.get("interval_ms", INTERVAL_MS)
    interval_s = interval_ms / 1000.0

    async def a2a_markers() -> AsyncIterator[bytes]:
        """Generate A2A-style streaming response."""
        start_time = time.monotonic()

        for i in range(marker_count):
            current_time = time.monotonic()
            elapsed_ms = (current_time - start_time) * 1000

            # A2A JSON-RPC notification format
            notification = {
                "jsonrpc": "2.0",
                "method": "tasks/update",
                "params": {
                    "id": request_id,
                    "type": "streaming",
                    "marker": i,
                    "total": marker_count,
                    "elapsed_ms": round(elapsed_ms, 3),
                    "timestamp": int(time.time() * 1000),
                },
            }

            yield (json.dumps(notification) + "\n").encode("utf-8")

            if i < marker_count - 1:
                await asyncio.sleep(interval_s)

        # Final result
        final_result = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "status": "completed",
                "markers_sent": marker_count,
                "total_time_ms": round((time.monotonic() - start_time) * 1000, 3),
            },
        }
        yield (json.dumps(final_result) + "\n").encode("utf-8")

    return StreamingResponse(
        a2a_markers(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    print(f"🚀 Starting Streaming Stub Server on port {STUB_PORT}")
    print(f"   Markers: {MARKER_COUNT}, Interval: {INTERVAL_MS}ms")
    print(f"   GET /stream - Raw JSON-RPC stream")
    print(f"   POST /stream - JSON-RPC streaming")
    print(f"   GET /stream/sse - SSE stream")
    print(f"   POST /a2a - A2A-style streaming")
    uvicorn.run(app, host="0.0.0.0", port=STUB_PORT)
