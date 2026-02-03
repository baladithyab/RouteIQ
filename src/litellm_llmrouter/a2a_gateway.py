"""
A2A Gateway - Agent-to-Agent Protocol Support
==============================================

Provides A2A (Agent-to-Agent) protocol gateway functionality for LiteLLM.
A2A is a protocol for agent-to-agent communication, allowing AI agents
to discover and communicate with each other.

Security Notes:
- Outbound URLs are validated against SSRF attacks before making requests
- See url_security.py for details on blocked targets

Thread Safety:
- Singleton initialization is protected by a module-level lock
- Registry mutations are protected by a reentrant lock
- Read operations return snapshots to avoid stale-read issues

Streaming Modes:
- A2A_RAW_STREAMING_ENABLED=true: True raw streaming passthrough using aiter_bytes()
  for minimal TTFB and proper chunk cadence. Does not wait for newline boundaries.
- A2A_RAW_STREAMING_ENABLED=false (default): Line-buffered streaming using aiter_lines()
  for backward compatibility. This is the rollback-safe default.

See: https://google.github.io/A2A/
"""

import json
import os
import threading
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, AsyncIterator

import httpx

from litellm._logging import verbose_proxy_logger

# Import SSRF protection utilities
try:
    from .url_security import validate_outbound_url, SSRFBlockedError

    SSRF_PROTECTION_AVAILABLE = True
except ImportError:
    SSRF_PROTECTION_AVAILABLE = False
    SSRFBlockedError = Exception  # Fallback type

    def validate_outbound_url(url: str, **kwargs) -> str:
        """No-op fallback when url_security module is not available."""
        return url


# Import tracing utilities for W3C trace context propagation
try:
    from .a2a_tracing import inject_trace_headers

    TRACING_AVAILABLE = True
except ImportError:
    TRACING_AVAILABLE = False

    def inject_trace_headers(headers: dict[str, str]) -> dict[str, str]:
        """No-op fallback when tracing module is not available."""
        return headers


# =============================================================================
# Feature Flags for Streaming Behavior
# =============================================================================

# A2A_RAW_STREAMING_ENABLED: When true, uses raw byte streaming (aiter_bytes)
# for true passthrough semantics. When false (default), uses line-buffered
# streaming (aiter_lines) for backward compatibility.
#
# Rollback Safety: Default is False to minimize blast radius. Set to "true"
# to enable raw streaming passthrough after validation in staging.
#
# Toggle: Set environment variable A2A_RAW_STREAMING_ENABLED=true to enable.
A2A_RAW_STREAMING_ENABLED = (
    os.getenv("A2A_RAW_STREAMING_ENABLED", "false").lower() == "true"
)

# Default chunk size for raw streaming (8KB balances latency vs overhead)
A2A_RAW_STREAMING_CHUNK_SIZE = int(
    os.getenv("A2A_RAW_STREAMING_CHUNK_SIZE", "8192")
)


@dataclass
class A2AAgent:
    """Represents an A2A agent registration."""

    agent_id: str
    name: str
    description: str
    url: str
    capabilities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class JSONRPCRequest:
    """JSON-RPC 2.0 request."""

    method: str
    params: dict[str, Any]
    id: str | int | None = None
    jsonrpc: str = "2.0"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "jsonrpc": self.jsonrpc,
            "method": self.method,
            "params": self.params,
            "id": self.id,
        }


@dataclass
class JSONRPCResponse:
    """JSON-RPC 2.0 response."""

    id: str | int | None
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    jsonrpc: str = "2.0"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        response = {"jsonrpc": self.jsonrpc, "id": self.id}
        if self.error is not None:
            response["error"] = self.error
        else:
            response["result"] = self.result
        return response

    @classmethod
    def error_response(
        cls, request_id: str | int | None, code: int, message: str
    ) -> "JSONRPCResponse":
        """Create an error response."""
        return cls(id=request_id, error={"code": code, "message": message})

    @classmethod
    def success_response(
        cls, request_id: str | int | None, result: dict[str, Any]
    ) -> "JSONRPCResponse":
        """Create a success response."""
        return cls(id=request_id, result=result)


class A2AGateway:
    """
    A2A Gateway for managing agent registrations and discovery.

    This gateway allows:
    - Registering AI agents with their capabilities
    - Discovering available agents
    - Routing requests to appropriate agents

    Thread Safety:
    All registry mutations are protected by a reentrant lock. Read operations
    return immutable snapshots to avoid stale-read issues and allow iteration
    without holding locks.

    Streaming Modes:
    - Raw streaming (A2A_RAW_STREAMING_ENABLED=true): True passthrough using
      aiter_bytes() for minimal TTFB and chunk cadence preservation.
    - Line-buffered (default): Uses aiter_lines() for backward compatibility.
    """

    def __init__(self):
        # Thread safety: RLock for registry mutations (reentrant to allow nested calls)
        self._lock = threading.RLock()

        self.agents: dict[str, A2AAgent] = {}
        self.enabled = os.getenv("A2A_GATEWAY_ENABLED", "false").lower() == "true"

    def is_enabled(self) -> bool:
        """Check if A2A gateway is enabled."""
        return self.enabled

    def register_agent(self, agent: A2AAgent) -> None:
        """
        Register an agent with the gateway.

        Security: Agent URLs are validated against SSRF attacks before registration.
        Thread Safety: Registry mutation is protected by lock.
        """
        if not self.enabled:
            verbose_proxy_logger.warning("A2A Gateway is not enabled")
            return

        # Security: Validate URL against SSRF attacks at registration time
        # Done outside lock to avoid holding lock during validation
        if agent.url:
            try:
                validate_outbound_url(
                    agent.url, resolve_dns=False
                )  # Don't resolve during registration
            except SSRFBlockedError as e:
                verbose_proxy_logger.warning(
                    f"A2A: SSRF blocked for agent '{agent.agent_id}': {e}"
                )
                raise ValueError(f"Agent URL blocked for security reasons: {e.reason}")
            except ValueError as e:
                verbose_proxy_logger.warning(
                    f"A2A: Invalid URL for agent '{agent.agent_id}': {e}"
                )
                raise ValueError(f"Agent URL is invalid: {str(e)}")

        # Thread-safe registry update
        with self._lock:
            self.agents[agent.agent_id] = agent

        verbose_proxy_logger.info(
            f"A2A: Registered agent {agent.name} ({agent.agent_id})"
        )

    def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent from the gateway.

        Thread Safety: Registry mutation is protected by lock.
        """
        with self._lock:
            if agent_id in self.agents:
                del self.agents[agent_id]
                found = True
            else:
                found = False

        if found:
            verbose_proxy_logger.info(f"A2A: Unregistered agent {agent_id}")
            return True
        return False

    def get_agent(self, agent_id: str) -> A2AAgent | None:
        """Get an agent by ID. Thread-safe."""
        with self._lock:
            return self.agents.get(agent_id)

    def get_agents_snapshot(self) -> MappingProxyType[str, A2AAgent]:
        """
        Get an immutable snapshot of the agents registry.

        Returns:
            Read-only view of current agents dict.
        """
        with self._lock:
            # Return immutable proxy to a copy to prevent mutation
            return MappingProxyType(dict(self.agents))

    def list_agents(self, capability: str | None = None) -> list[A2AAgent]:
        """Discover agents, optionally filtered by capability. Thread-safe."""
        with self._lock:
            agents_snapshot = list(self.agents.values())

        if capability is None:
            return agents_snapshot
        return [a for a in agents_snapshot if capability in a.capabilities]

    def get_agent_card(self, agent_id: str) -> dict[str, Any] | None:
        """Get the A2A agent card for an agent. Thread-safe."""
        with self._lock:
            agent = self.agents.get(agent_id)

        if not agent:
            return None

        return {
            "name": agent.name,
            "description": agent.description,
            "url": agent.url,
            "capabilities": {
                "streaming": "streaming" in agent.capabilities,
                "pushNotifications": "push_notifications" in agent.capabilities,
                "stateTransitionHistory": "state_history" in agent.capabilities,
            },
            "skills": [
                {"id": cap, "name": cap.replace("_", " ").title()}
                for cap in agent.capabilities
            ],
        }

    async def invoke_agent(
        self, agent_id: str, request: JSONRPCRequest
    ) -> JSONRPCResponse:
        """
        Invoke an agent using JSON-RPC 2.0 protocol.

        Supports methods:
        - message/send: Send a message and get a response
        - message/stream: Send a message and stream the response (returns first chunk)

        Security: Agent URLs are validated against SSRF attacks before making requests.
        Thread Safety: Agent lookup is protected by lock.

        Args:
            agent_id: The ID of the agent to invoke
            request: The JSON-RPC 2.0 request

        Returns:
            JSONRPCResponse with the result or error
        """
        if not self.enabled:
            return JSONRPCResponse.error_response(
                request.id, -32000, "A2A Gateway is not enabled"
            )

        # Get agent under lock
        with self._lock:
            agent = self.agents.get(agent_id)

        if not agent:
            return JSONRPCResponse.error_response(
                request.id, -32000, f"Agent '{agent_id}' not found"
            )

        if not agent.url:
            return JSONRPCResponse.error_response(
                request.id, -32000, f"Agent '{agent_id}' has no URL configured"
            )

        # Security: Validate URL against SSRF attacks (outside lock)
        try:
            validate_outbound_url(agent.url)
        except SSRFBlockedError as e:
            verbose_proxy_logger.warning(
                f"A2A: SSRF blocked for agent '{agent_id}': {e}"
            )
            return JSONRPCResponse.error_response(
                request.id,
                -32000,
                f"Agent URL blocked for security reasons: {e.reason}",
            )
        except ValueError as e:
            verbose_proxy_logger.warning(
                f"A2A: Invalid URL for agent '{agent_id}': {e}"
            )
            return JSONRPCResponse.error_response(
                request.id, -32000, f"Agent URL is invalid: {str(e)}"
            )

        # Validate JSON-RPC format
        if request.jsonrpc != "2.0":
            return JSONRPCResponse.error_response(
                request.id, -32600, "Invalid Request: jsonrpc must be '2.0'"
            )

        method = request.method
        if method not in ("message/send", "message/stream"):
            return JSONRPCResponse.error_response(
                request.id, -32601, f"Method '{method}' not found"
            )

        verbose_proxy_logger.info(
            f"A2A: Invoking agent '{agent_id}' with method '{method}'"
        )

        try:
            # Forward the request to the agent backend
            # Inject W3C trace context headers for distributed tracing
            headers = {"Content-Type": "application/json"}
            headers = inject_trace_headers(headers)

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    agent.url,
                    json=request.to_dict(),
                    headers=headers,
                )
                response.raise_for_status()
                result = response.json()

                # Return the response from the agent
                if "error" in result:
                    return JSONRPCResponse(
                        id=request.id,
                        error=result["error"],
                    )
                return JSONRPCResponse(
                    id=request.id,
                    result=result.get("result", result),
                )

        except httpx.TimeoutException:
            verbose_proxy_logger.error(f"A2A: Timeout invoking agent '{agent_id}'")
            return JSONRPCResponse.error_response(
                request.id, -32000, f"Timeout invoking agent '{agent_id}'"
            )
        except httpx.HTTPStatusError as e:
            verbose_proxy_logger.error(
                f"A2A: HTTP error invoking agent '{agent_id}': {e}"
            )
            return JSONRPCResponse.error_response(
                request.id, -32000, f"HTTP error: {e.response.status_code}"
            )
        except Exception as e:
            verbose_proxy_logger.exception(
                f"A2A: Error invoking agent '{agent_id}': {e}"
            )
            return JSONRPCResponse.error_response(
                request.id, -32603, f"Internal error: {str(e)}"
            )

    async def stream_agent_response(
        self, agent_id: str, request: JSONRPCRequest
    ) -> AsyncIterator[str]:
        """
        Stream response from an agent using Server-Sent Events.

        This method dispatches to either raw streaming (aiter_bytes) or
        line-buffered streaming (aiter_lines) based on the A2A_RAW_STREAMING_ENABLED
        feature flag.

        Security: Agent URLs are validated against SSRF attacks before making requests.
        Thread Safety: Agent lookup is protected by lock.

        Streaming Modes:
        - A2A_RAW_STREAMING_ENABLED=true: Raw byte streaming for true passthrough.
          Emits chunks as they arrive without waiting for newline boundaries.
        - A2A_RAW_STREAMING_ENABLED=false (default): Line-buffered streaming for
          backward compatibility. Waits for complete lines before yielding.

        Args:
            agent_id: The ID of the agent to invoke
            request: The JSON-RPC 2.0 request with method 'message/stream'

        Yields:
            Response chunks (bytes decoded as UTF-8 for raw mode, lines for buffered mode)
        """
        if A2A_RAW_STREAMING_ENABLED:
            async for chunk in self._stream_agent_response_raw(agent_id, request):
                yield chunk
        else:
            async for chunk in self._stream_agent_response_buffered(agent_id, request):
                yield chunk

    async def _stream_agent_response_raw(
        self, agent_id: str, request: JSONRPCRequest
    ) -> AsyncIterator[str]:
        """
        True raw streaming passthrough using aiter_bytes().

        This implementation:
        - Does NOT wait for newline boundaries (true passthrough)
        - Preserves upstream chunk cadence as much as possible
        - Respects backpressure (async iteration)
        - Supports cancellation (no full buffering)

        Args:
            agent_id: The ID of the agent to invoke
            request: The JSON-RPC 2.0 request with method 'message/stream'

        Yields:
            Raw bytes decoded as UTF-8 strings, preserving original chunk boundaries
        """
        if not self.enabled:
            yield (
                json.dumps(
                    JSONRPCResponse.error_response(
                        request.id, -32000, "A2A Gateway is not enabled"
                    ).to_dict()
                )
                + "\n"
            )
            return

        # Get agent under lock
        with self._lock:
            agent = self.agents.get(agent_id)

        if not agent:
            yield (
                json.dumps(
                    JSONRPCResponse.error_response(
                        request.id, -32000, f"Agent '{agent_id}' not found"
                    ).to_dict()
                )
                + "\n"
            )
            return

        if not agent.url:
            yield (
                json.dumps(
                    JSONRPCResponse.error_response(
                        request.id, -32000, f"Agent '{agent_id}' has no URL configured"
                    ).to_dict()
                )
                + "\n"
            )
            return

        # Security: Validate URL against SSRF attacks (outside lock)
        try:
            validate_outbound_url(agent.url)
        except SSRFBlockedError as e:
            verbose_proxy_logger.warning(
                f"A2A: SSRF blocked for agent '{agent_id}': {e}"
            )
            yield (
                json.dumps(
                    JSONRPCResponse.error_response(
                        request.id,
                        -32000,
                        f"Agent URL blocked for security reasons: {e.reason}",
                    ).to_dict()
                )
                + "\n"
            )
            return
        except ValueError as e:
            verbose_proxy_logger.warning(
                f"A2A: Invalid URL for agent '{agent_id}': {e}"
            )
            yield (
                json.dumps(
                    JSONRPCResponse.error_response(
                        request.id, -32000, f"Agent URL is invalid: {str(e)}"
                    ).to_dict()
                )
                + "\n"
            )
            return

        verbose_proxy_logger.info(
            f"A2A: Raw streaming from agent '{agent_id}' (chunk_size={A2A_RAW_STREAMING_CHUNK_SIZE})"
        )

        try:
            # Inject W3C trace context headers for distributed tracing
            headers = {"Content-Type": "application/json"}
            headers = inject_trace_headers(headers)

            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    agent.url,
                    json=request.to_dict(),
                    headers=headers,
                ) as response:
                    response.raise_for_status()

                    # True raw streaming: yield bytes as they arrive
                    # without waiting for newline boundaries
                    async for chunk in response.aiter_bytes(
                        chunk_size=A2A_RAW_STREAMING_CHUNK_SIZE
                    ):
                        if chunk:
                            # Decode bytes to string for consistency with API contract
                            # Note: This preserves chunk boundaries but decodes to UTF-8
                            yield chunk.decode("utf-8", errors="replace")

        except httpx.TimeoutException:
            verbose_proxy_logger.error(
                f"A2A: Timeout streaming from agent '{agent_id}'"
            )
            yield (
                json.dumps(
                    JSONRPCResponse.error_response(
                        request.id, -32000, f"Timeout streaming from agent '{agent_id}'"
                    ).to_dict()
                )
                + "\n"
            )
        except httpx.HTTPStatusError as e:
            verbose_proxy_logger.error(
                f"A2A: HTTP error streaming from agent '{agent_id}': {e}"
            )
            yield (
                json.dumps(
                    JSONRPCResponse.error_response(
                        request.id, -32000, f"HTTP error: {e.response.status_code}"
                    ).to_dict()
                )
                + "\n"
            )
        except Exception as e:
            verbose_proxy_logger.exception(
                f"A2A: Error streaming from agent '{agent_id}': {e}"
            )
            yield (
                json.dumps(
                    JSONRPCResponse.error_response(
                        request.id, -32603, f"Streaming error: {str(e)}"
                    ).to_dict()
                )
                + "\n"
            )

    async def _stream_agent_response_buffered(
        self, agent_id: str, request: JSONRPCRequest
    ) -> AsyncIterator[str]:
        """
        Line-buffered streaming using aiter_lines() - the original implementation.

        This is the backward-compatible implementation that waits for complete
        lines (newline boundaries) before yielding. Preserved for rollback safety.

        Args:
            agent_id: The ID of the agent to invoke
            request: The JSON-RPC 2.0 request with method 'message/stream'

        Yields:
            JSON-encoded response chunks as newline-delimited JSON
        """
        if not self.enabled:
            yield (
                json.dumps(
                    JSONRPCResponse.error_response(
                        request.id, -32000, "A2A Gateway is not enabled"
                    ).to_dict()
                )
                + "\n"
            )
            return

        # Get agent under lock
        with self._lock:
            agent = self.agents.get(agent_id)

        if not agent:
            yield (
                json.dumps(
                    JSONRPCResponse.error_response(
                        request.id, -32000, f"Agent '{agent_id}' not found"
                    ).to_dict()
                )
                + "\n"
            )
            return

        if not agent.url:
            yield (
                json.dumps(
                    JSONRPCResponse.error_response(
                        request.id, -32000, f"Agent '{agent_id}' has no URL configured"
                    ).to_dict()
                )
                + "\n"
            )
            return

        # Security: Validate URL against SSRF attacks (outside lock)
        try:
            validate_outbound_url(agent.url)
        except SSRFBlockedError as e:
            verbose_proxy_logger.warning(
                f"A2A: SSRF blocked for agent '{agent_id}': {e}"
            )
            yield (
                json.dumps(
                    JSONRPCResponse.error_response(
                        request.id,
                        -32000,
                        f"Agent URL blocked for security reasons: {e.reason}",
                    ).to_dict()
                )
                + "\n"
            )
            return
        except ValueError as e:
            verbose_proxy_logger.warning(
                f"A2A: Invalid URL for agent '{agent_id}': {e}"
            )
            yield (
                json.dumps(
                    JSONRPCResponse.error_response(
                        request.id, -32000, f"Agent URL is invalid: {str(e)}"
                    ).to_dict()
                )
                + "\n"
            )
            return

        verbose_proxy_logger.info(f"A2A: Streaming from agent '{agent_id}'")

        try:
            # Inject W3C trace context headers for distributed tracing
            headers = {"Content-Type": "application/json"}
            headers = inject_trace_headers(headers)

            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    agent.url,
                    json=request.to_dict(),
                    headers=headers,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line:
                            yield line + "\n"

        except httpx.TimeoutException:
            verbose_proxy_logger.error(
                f"A2A: Timeout streaming from agent '{agent_id}'"
            )
            yield (
                json.dumps(
                    JSONRPCResponse.error_response(
                        request.id, -32000, f"Timeout streaming from agent '{agent_id}'"
                    ).to_dict()
                )
                + "\n"
            )
        except httpx.HTTPStatusError as e:
            verbose_proxy_logger.error(
                f"A2A: HTTP error streaming from agent '{agent_id}': {e}"
            )
            yield (
                json.dumps(
                    JSONRPCResponse.error_response(
                        request.id, -32000, f"HTTP error: {e.response.status_code}"
                    ).to_dict()
                )
                + "\n"
            )
        except Exception as e:
            verbose_proxy_logger.exception(
                f"A2A: Error streaming from agent '{agent_id}': {e}"
            )
            yield (
                json.dumps(
                    JSONRPCResponse.error_response(
                        request.id, -32603, f"Streaming error: {str(e)}"
                    ).to_dict()
                )
                + "\n"
            )


# Singleton instance and lock for thread-safe initialization
_a2a_gateway: A2AGateway | None = None
_a2a_gateway_lock = threading.Lock()


def get_a2a_gateway() -> A2AGateway:
    """
    Get the global A2A gateway instance.

    Thread-safe: Uses double-checked locking pattern for efficient
    singleton initialization.
    """
    global _a2a_gateway
    if _a2a_gateway is None:
        with _a2a_gateway_lock:
            # Double-check after acquiring lock
            if _a2a_gateway is None:
                _a2a_gateway = A2AGateway()
    return _a2a_gateway


def reset_a2a_gateway() -> None:
    """
    Reset the global A2A gateway instance.

    WARNING: For testing purposes only. Not safe to call while
    requests are in flight.
    """
    global _a2a_gateway
    with _a2a_gateway_lock:
        _a2a_gateway = None


def is_raw_streaming_enabled() -> bool:
    """
    Check if raw streaming passthrough is enabled.

    Returns:
        True if A2A_RAW_STREAMING_ENABLED environment variable is set to "true".

    Usage:
        from litellm_llmrouter.a2a_gateway import is_raw_streaming_enabled
        if is_raw_streaming_enabled():
            print("Raw streaming mode active")
    """
    return A2A_RAW_STREAMING_ENABLED
