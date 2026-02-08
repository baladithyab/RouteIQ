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

HTTP Client Pooling:
- Uses shared HTTP client pool by default (HTTP_CLIENT_POOLING_ENABLED=true)
- Falls back to per-request clients when pooling is disabled
- See http_client_pool.py for configuration and lifecycle

See: https://google.github.io/A2A/
"""

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, AsyncIterator

import httpx

from litellm._logging import verbose_proxy_logger

# Import shared HTTP client pool
from .http_client_pool import get_client_for_request

# Import SSRF protection utilities
try:
    from .url_security import (
        validate_outbound_url,
        validate_outbound_url_async,
        SSRFBlockedError,
    )

    SSRF_PROTECTION_AVAILABLE = True
except ImportError:
    SSRF_PROTECTION_AVAILABLE = False
    SSRFBlockedError = Exception  # Fallback type

    def validate_outbound_url(url: str, **kwargs) -> str:
        """No-op fallback when url_security module is not available."""
        return url

    async def validate_outbound_url_async(url: str, **kwargs) -> str:
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
A2A_RAW_STREAMING_CHUNK_SIZE = int(os.getenv("A2A_RAW_STREAMING_CHUNK_SIZE", "8192"))

# =============================================================================
# Header Preservation for Streaming Responses
# =============================================================================

# Headers to preserve from upstream responses (case-insensitive matching)
# These are important for SSE/chunked streaming to work correctly
STREAMING_HEADERS_TO_PRESERVE = frozenset(
    {
        "content-type",
        "cache-control",
        "x-accel-buffering",
        "x-request-id",
        "x-trace-id",
    }
)

# Hop-by-hop headers that MUST NOT be forwarded (RFC 2616 Section 13.5.1)
HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)

# Unsafe headers that should not be forwarded for security reasons
UNSAFE_HEADERS = frozenset(
    {
        "set-cookie",
        "set-cookie2",
        "authorization",
        "www-authenticate",
        "proxy-connection",
    }
)


def filter_upstream_headers(
    headers: httpx.Headers,
    preserve_list: frozenset[str] = STREAMING_HEADERS_TO_PRESERVE,
) -> dict[str, str]:
    """
    Filter upstream response headers for safe forwarding.

    Preserves headers important for streaming (Content-Type, Cache-Control, etc.)
    while stripping hop-by-hop and security-sensitive headers.

    Args:
        headers: The upstream response headers (httpx.Headers)
        preserve_list: Set of header names to preserve (lowercase)

    Returns:
        Dictionary of safe headers to forward downstream
    """
    filtered: dict[str, str] = {}

    for name, value in headers.items():
        name_lower = name.lower()

        # Skip hop-by-hop headers (MUST NOT forward)
        if name_lower in HOP_BY_HOP_HEADERS:
            continue

        # Skip unsafe headers
        if name_lower in UNSAFE_HEADERS:
            continue

        # Only preserve headers in the allow-list
        if name_lower in preserve_list:
            filtered[name] = value

    return filtered


def format_sse_event(event_type: str, data: dict[str, Any]) -> str:
    """
    Format an SSE event string per the A2A specification.

    Args:
        event_type: The SSE event type (e.g., 'task-status-update')
        data: The event data to serialize as JSON

    Returns:
        Formatted SSE event string: ``event: <type>\\ndata: <json>\\n\\n``
    """
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


@dataclass
class StreamingResponseMeta:
    """
    Metadata for streaming responses including preserved upstream headers.

    This is returned alongside the chunk generator to allow callers to
    construct proper streaming responses with correct headers.
    """

    headers: dict[str, str] = field(default_factory=dict)
    status_code: int = 200


# =============================================================================
# A2A Task Model and State Machine (A2A Spec Compliance)
# =============================================================================

# Default TTL for tasks in memory (1 hour)
A2A_TASK_TTL_SECONDS = int(os.getenv("A2A_TASK_TTL_SECONDS", "3600"))

# Maximum tasks in store (prevent memory exhaustion DoS)
A2A_TASK_STORE_MAX_TASKS = int(os.getenv("A2A_TASK_STORE_MAX_TASKS", "10000"))

# Per-agent rate limit: max task creates per minute
A2A_TASK_RATE_LIMIT = int(os.getenv("A2A_TASK_RATE_LIMIT", "100"))


class TaskState(str, Enum):
    """A2A task lifecycle states per the A2A specification."""

    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    UNKNOWN = "unknown"


# Valid state transitions per the A2A spec state machine:
#   submitted -> working | completed | failed | canceled
#   working -> completed | failed | canceled | input-required
#   input-required -> working | completed | failed | canceled
#   completed -> (terminal, no transitions)
#   failed -> (terminal, no transitions)
#   canceled -> (terminal, no transitions)
VALID_TRANSITIONS: dict[TaskState, set[TaskState]] = {
    TaskState.SUBMITTED: {
        TaskState.WORKING,
        TaskState.COMPLETED,
        TaskState.FAILED,
        TaskState.CANCELED,
    },
    TaskState.WORKING: {
        TaskState.COMPLETED,
        TaskState.FAILED,
        TaskState.CANCELED,
        TaskState.INPUT_REQUIRED,
    },
    TaskState.INPUT_REQUIRED: {
        TaskState.WORKING,
        TaskState.COMPLETED,
        TaskState.FAILED,
        TaskState.CANCELED,
    },
    TaskState.COMPLETED: set(),
    TaskState.FAILED: set(),
    TaskState.CANCELED: set(),
    TaskState.UNKNOWN: {TaskState.SUBMITTED, TaskState.WORKING},
}

# String-based alias for external callers that don't use the TaskState enum
VALID_STATE_TRANSITIONS: dict[str, set[str]] = {
    state.value: {t.value for t in targets}
    for state, targets in VALID_TRANSITIONS.items()
}


class InvalidTaskTransitionError(ValueError):
    """Raised when an invalid task state transition is attempted."""

    def __init__(self, current: TaskState, target: TaskState):
        self.current = current
        self.target = target
        valid = VALID_TRANSITIONS.get(current, set())
        valid_names = {s.value for s in valid} if valid else set()
        super().__init__(
            f"Invalid state transition: {current.value} -> {target.value}. "
            f"Valid transitions from '{current.value}': {valid_names}"
        )


def validate_state_transition(current_state: str, new_state: str) -> bool:
    """Validate an A2A task state transition using string state names.

    Returns True if the transition is valid per the A2A spec.
    Raises ValueError with JSON-RPC -32602 error details if invalid.

    Args:
        current_state: Current state name (e.g., 'submitted', 'working').
        new_state: Target state name (e.g., 'working', 'completed').

    Returns:
        True if transition is valid.

    Raises:
        ValueError: If the transition is invalid, with details about valid targets.
    """
    valid_targets = VALID_STATE_TRANSITIONS.get(current_state)
    if valid_targets is None:
        raise ValueError(
            f"Unknown current state: '{current_state}'. "
            f"Valid states: {set(VALID_STATE_TRANSITIONS.keys())}"
        )
    if new_state not in valid_targets:
        raise ValueError(
            f"Invalid state transition: {current_state} -> {new_state}. "
            f"Valid transitions from '{current_state}': {valid_targets}"
        )
    return True


class A2AError:
    """Standard A2A JSON-RPC error responses.

    Provides predefined error codes and a factory method for creating
    JSON-RPC 2.0 error response dictionaries.

    Error codes follow JSON-RPC 2.0 conventions:
    - -32600 to -32699: Standard JSON-RPC errors
    - -32000 to -32099: Server/implementation errors
    - -32001 to -32003: A2A-specific application errors
    """

    # A2A application errors
    TASK_NOT_FOUND = {"code": -32001, "message": "Task not found"}
    AGENT_NOT_FOUND = {"code": -32002, "message": "Agent not found"}
    RATE_LIMITED = {"code": -32003, "message": "Rate limited"}

    # JSON-RPC standard errors
    INVALID_PARAMS = {"code": -32602, "message": "Invalid params"}
    INVALID_TRANSITION = {"code": -32602, "message": "Invalid state transition"}
    METHOD_NOT_FOUND = {"code": -32601, "message": "Method not found"}
    INVALID_REQUEST = {"code": -32600, "message": "Invalid Request"}
    INTERNAL_ERROR = {"code": -32603, "message": "Internal error"}

    @staticmethod
    def make_error(
        code: int, message: str, data: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Create a JSON-RPC 2.0 error response envelope.

        Args:
            code: JSON-RPC error code.
            message: Human-readable error message.
            data: Optional additional error data.

        Returns:
            Complete JSON-RPC 2.0 error response dict.
        """
        error: dict[str, Any] = {"code": code, "message": message}
        if data:
            error["data"] = data
        return {"jsonrpc": "2.0", "error": error, "id": None}

    @classmethod
    def task_not_found(cls, task_id: str) -> dict[str, Any]:
        """Create a task-not-found error response."""
        return cls.make_error(
            cls.TASK_NOT_FOUND["code"],
            f"Task '{task_id}' not found",
        )

    @classmethod
    def agent_not_found(cls, agent_id: str) -> dict[str, Any]:
        """Create an agent-not-found error response."""
        return cls.make_error(
            cls.AGENT_NOT_FOUND["code"],
            f"Agent '{agent_id}' not found",
        )

    @classmethod
    def invalid_transition(cls, current_state: str, new_state: str) -> dict[str, Any]:
        """Create an invalid-state-transition error response."""
        valid = VALID_STATE_TRANSITIONS.get(current_state, set())
        return cls.make_error(
            cls.INVALID_TRANSITION["code"],
            f"Invalid state transition: {current_state} -> {new_state}. "
            f"Valid transitions from '{current_state}': {valid}",
            data={"current_state": current_state, "requested_state": new_state},
        )


@dataclass
class A2ATask:
    """
    Represents an A2A task with lifecycle state management.

    Per the A2A specification, a task is the unit of work with a
    well-defined state machine governing its lifecycle.
    """

    id: str
    state: TaskState
    agent_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def transition(self, new_state: TaskState) -> None:
        """
        Transition the task to a new state with validation.

        Raises:
            InvalidTaskTransitionError: If the transition is not allowed.
        """
        allowed = VALID_TRANSITIONS.get(self.state, set())
        if new_state not in allowed:
            raise InvalidTaskTransitionError(self.state, new_state)
        old_state = self.state
        self.state = new_state
        self.updated_at = time.time()
        self.history.append(
            {
                "from": old_state.value,
                "to": new_state.value,
                "timestamp": self.updated_at,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize task to dictionary for JSON-RPC responses."""
        return {
            "id": self.id,
            "status": {"state": self.state.value},
            "agent_id": self.agent_id,
            "messages": self.messages,
            "artifacts": self.artifacts,
            "history": self.history,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class A2ATaskStore:
    """
    Thread-safe in-memory store for A2A tasks with TTL-based cleanup.

    Tasks are automatically removed after their TTL expires.
    Enforces max_tasks limit and per-agent rate limiting to prevent DoS.
    """

    def __init__(
        self,
        ttl_seconds: int = A2A_TASK_TTL_SECONDS,
        max_tasks: int = A2A_TASK_STORE_MAX_TASKS,
        rate_limit_per_minute: int = A2A_TASK_RATE_LIMIT,
    ):
        self._tasks: dict[str, A2ATask] = {}
        self._lock = threading.RLock()
        self._ttl = ttl_seconds
        self._max_tasks = max_tasks
        self._rate_limit = rate_limit_per_minute
        # Per-agent rate tracking: {agent_id: [timestamps]}
        self._agent_creates: dict[str, list[float]] = {}

    def _check_rate_limit(self, agent_id: str) -> bool:
        """Check per-agent rate limit. Returns True if allowed."""
        now = time.time()
        window_start = now - 60.0

        timestamps = self._agent_creates.get(agent_id, [])
        # Prune old timestamps outside the 1-minute window
        timestamps = [t for t in timestamps if t > window_start]
        self._agent_creates[agent_id] = timestamps

        if len(timestamps) >= self._rate_limit:
            return False
        timestamps.append(now)
        return True

    def create_task(self, agent_id: str, messages: list[dict] | None = None) -> A2ATask:
        """
        Create a new task in SUBMITTED state.

        Raises:
            ValueError: If store is at capacity or rate limit exceeded.
        """
        with self._lock:
            # Check per-agent rate limit
            if not self._check_rate_limit(agent_id):
                raise ValueError(
                    f"Rate limit exceeded for agent '{agent_id}': "
                    f"max {self._rate_limit} creates/minute"
                )

            # If at capacity, try cleanup first
            if len(self._tasks) >= self._max_tasks:
                self._cleanup_expired_unlocked()

            # Still at capacity after cleanup — reject
            if len(self._tasks) >= self._max_tasks:
                raise ValueError(
                    f"Task store at capacity ({self._max_tasks}). "
                    f"Cannot create new task."
                )

            task = A2ATask(
                id=str(uuid.uuid4()),
                state=TaskState.SUBMITTED,
                agent_id=agent_id,
                messages=messages or [],
            )
            self._tasks[task.id] = task
        return task

    def get_task(self, task_id: str) -> A2ATask | None:
        """Get a task by ID. Returns None if not found or expired."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            if time.time() - task.created_at > self._ttl:
                del self._tasks[task_id]
                return None
            return task

    def cleanup_expired(self) -> int:
        """Remove expired tasks. Returns number of tasks removed."""
        with self._lock:
            return self._cleanup_expired_unlocked()

    def _cleanup_expired_unlocked(self) -> int:
        """Remove expired tasks (must hold lock). Returns count removed."""
        now = time.time()
        expired_ids = [
            tid
            for tid, task in self._tasks.items()
            if now - task.created_at > self._ttl
        ]
        for tid in expired_ids:
            del self._tasks[tid]
        return len(expired_ids)

    def count(self) -> int:
        """Return current number of tasks (including possibly expired)."""
        with self._lock:
            return len(self._tasks)

    @property
    def max_tasks(self) -> int:
        """Configured max tasks limit."""
        return self._max_tasks

    def clear(self) -> None:
        """Remove all tasks. For testing only."""
        with self._lock:
            self._tasks.clear()
            self._agent_creates.clear()


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

    # Method aliases: A2A spec names -> internal handler names
    METHOD_ALIASES: dict[str, str] = {
        "tasks/send": "message/send",
        "tasks/sendSubscribe": "message/stream",
    }

    # Methods that require streaming transport
    STREAMING_METHODS: frozenset[str] = frozenset(
        {"message/stream", "tasks/sendSubscribe"}
    )

    def __init__(self):
        # Thread safety: RLock for registry mutations (reentrant to allow nested calls)
        self._lock = threading.RLock()

        self.agents: dict[str, A2AAgent] = {}
        self.enabled = os.getenv("A2A_GATEWAY_ENABLED", "false").lower() == "true"
        self.task_store = A2ATaskStore()

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
        """Get the A2A agent card for an agent per the A2A spec. Thread-safe."""
        with self._lock:
            agent = self.agents.get(agent_id)

        if not agent:
            return None

        return {
            "name": agent.name,
            "description": agent.description,
            "url": agent.url,
            "version": "0.2.0",
            "capabilities": {
                "streaming": "streaming" in agent.capabilities,
                "pushNotifications": False,
                "stateTransitionHistory": True,
            },
            "skills": [
                {"id": cap, "name": cap.replace("_", " ").title()}
                for cap in agent.capabilities
            ],
            "defaultInputModes": ["application/json"],
            "defaultOutputModes": ["application/json"],
            "authentication": {
                "schemes": ["apiKey"],
            },
        }

    def get_gateway_agent_card(self, base_url: str = "") -> dict[str, Any]:
        """
        Get the gateway-level A2A Agent Card.

        Returns a card describing the gateway itself, with skills
        derived from all registered agents.

        Args:
            base_url: The base URL of the gateway (e.g., 'https://gateway.example.com')

        Returns:
            Agent Card dictionary per the A2A specification.
        """
        agents_snapshot = self.list_agents()
        skills = []
        for agent in agents_snapshot:
            skills.append(
                {
                    "id": agent.agent_id,
                    "name": agent.name,
                    "description": agent.description,
                }
            )

        return {
            "name": "RouteIQ A2A Gateway",
            "description": "AI Gateway with A2A protocol support for agent-to-agent communication",
            "url": f"{base_url}/a2a",
            "version": "0.2.0",
            "capabilities": {
                "streaming": True,
                "pushNotifications": False,
                "stateTransitionHistory": True,
            },
            "skills": skills,
            "defaultInputModes": ["application/json"],
            "defaultOutputModes": ["application/json"],
            "authentication": {
                "schemes": ["apiKey"],
                "credentials": "X-Admin-API-Key header or Authorization: Bearer <key>",
            },
        }

    def _resolve_method(self, method: str) -> str:
        """Resolve A2A spec method aliases to internal handler names."""
        return self.METHOD_ALIASES.get(method, method)

    def _is_streaming_method(self, method: str) -> bool:
        """Check if a method requires streaming transport."""
        return method in self.STREAMING_METHODS

    async def handle_task_get(self, request: JSONRPCRequest) -> JSONRPCResponse:
        """
        Handle tasks/get: retrieve task state by ID.

        Params must include 'id' (task ID).
        """
        task_id = request.params.get("id") if request.params else None
        if not task_id:
            return JSONRPCResponse.error_response(
                request.id, -32602, "Missing required param: id"
            )

        task = self.task_store.get_task(task_id)
        if not task:
            return JSONRPCResponse.error_response(
                request.id, -32001, f"Task '{task_id}' not found"
            )

        return JSONRPCResponse.success_response(request.id, task.to_dict())

    async def handle_task_cancel(self, request: JSONRPCRequest) -> JSONRPCResponse:
        """
        Handle tasks/cancel: cancel a running or submitted task.

        Params must include 'id' (task ID).
        """
        task_id = request.params.get("id") if request.params else None
        if not task_id:
            return JSONRPCResponse.error_response(
                request.id, -32602, "Missing required param: id"
            )

        task = self.task_store.get_task(task_id)
        if not task:
            return JSONRPCResponse.error_response(
                request.id, -32001, f"Task '{task_id}' not found"
            )

        try:
            task.transition(TaskState.CANCELED)
        except InvalidTaskTransitionError as e:
            return JSONRPCResponse.error_response(
                request.id,
                A2AError.INVALID_TRANSITION["code"],
                str(e),
            )

        return JSONRPCResponse.success_response(request.id, task.to_dict())

    async def invoke_agent(
        self, agent_id: str, request: JSONRPCRequest
    ) -> JSONRPCResponse:
        """
        Invoke an agent using JSON-RPC 2.0 protocol.

        Supports methods (both A2A spec and legacy names):
        - message/send | tasks/send: Send a message and get a response
        - message/stream | tasks/sendSubscribe: Streaming (returns error directing to streaming endpoint)
        - tasks/get: Retrieve task state
        - tasks/cancel: Cancel a running task

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

        # Handle task-level methods that don't require agent lookup
        if request.method == "tasks/get":
            return await self.handle_task_get(request)
        if request.method == "tasks/cancel":
            return await self.handle_task_cancel(request)

        # Get agent under lock
        with self._lock:
            agent = self.agents.get(agent_id)

        if not agent:
            return JSONRPCResponse.error_response(
                request.id,
                A2AError.AGENT_NOT_FOUND["code"],
                f"Agent '{agent_id}' not found",
            )

        if not agent.url:
            return JSONRPCResponse.error_response(
                request.id,
                A2AError.AGENT_NOT_FOUND["code"],
                f"Agent '{agent_id}' has no URL configured",
            )

        # Security: Validate URL against SSRF attacks (outside lock)
        # Use async version to avoid blocking the event loop
        try:
            await validate_outbound_url_async(agent.url)
        except SSRFBlockedError as e:
            verbose_proxy_logger.warning(
                f"A2A: SSRF blocked for agent '{agent_id}': {e}"
            )
            return JSONRPCResponse.error_response(
                request.id,
                A2AError.INTERNAL_ERROR["code"],
                f"Agent URL blocked for security reasons: {e.reason}",
            )
        except ValueError as e:
            verbose_proxy_logger.warning(
                f"A2A: Invalid URL for agent '{agent_id}': {e}"
            )
            return JSONRPCResponse.error_response(
                request.id,
                A2AError.INVALID_PARAMS["code"],
                f"Agent URL is invalid: {str(e)}",
            )

        # Validate JSON-RPC format
        if request.jsonrpc != "2.0":
            return JSONRPCResponse.error_response(
                request.id,
                A2AError.INVALID_REQUEST["code"],
                "Invalid Request: jsonrpc must be '2.0'",
            )

        # Resolve method aliases (tasks/send -> message/send, etc.)
        resolved_method = self._resolve_method(request.method)

        # Route streaming methods to the streaming endpoint
        if self._is_streaming_method(request.method):
            return JSONRPCResponse.error_response(
                request.id,
                A2AError.INVALID_REQUEST["code"],
                "Use streaming endpoint for streaming methods. "
                "POST to /a2a/{agent_id} with Accept: text/event-stream header.",
            )

        if resolved_method != "message/send":
            return JSONRPCResponse.error_response(
                request.id,
                A2AError.METHOD_NOT_FOUND["code"],
                f"Method '{request.method}' not found",
            )

        verbose_proxy_logger.info(
            f"A2A: Invoking agent '{agent_id}' with method '{request.method}'"
        )

        # Create task for lifecycle tracking
        messages = []
        if request.params and "message" in request.params:
            messages = [request.params["message"]]
        task = self.task_store.create_task(agent_id=agent_id, messages=messages)
        task.transition(TaskState.WORKING)

        try:
            # Forward the request to the agent backend
            # Inject W3C trace context headers for distributed tracing
            headers = {"Content-Type": "application/json"}
            headers = inject_trace_headers(headers)

            # Build the forwarded request using resolved method name
            forwarded_request = JSONRPCRequest(
                method=resolved_method,
                params=request.params,
                id=request.id,
            )

            async with get_client_for_request(timeout=60.0) as client:
                response = await client.post(
                    agent.url,
                    json=forwarded_request.to_dict(),
                    headers=headers,
                )
                response.raise_for_status()
                result = response.json()

                # Return the response from the agent
                if "error" in result:
                    task.transition(TaskState.FAILED)
                    resp = JSONRPCResponse(
                        id=request.id,
                        error=result["error"],
                    )
                else:
                    task.transition(TaskState.COMPLETED)
                    agent_result = result.get("result", result)
                    # Store artifacts if present in result
                    if isinstance(agent_result, dict) and "artifacts" in agent_result:
                        task.artifacts = agent_result["artifacts"]
                    resp = JSONRPCResponse(
                        id=request.id,
                        result=agent_result,
                    )

                # Embed task info in the response result
                if resp.result is not None and isinstance(resp.result, dict):
                    resp.result["_task"] = task.to_dict()
                return resp

        except httpx.TimeoutException:
            verbose_proxy_logger.error(f"A2A: Timeout invoking agent '{agent_id}'")
            task.transition(TaskState.FAILED)
            return JSONRPCResponse.error_response(
                request.id,
                A2AError.INTERNAL_ERROR["code"],
                f"Timeout invoking agent '{agent_id}'",
            )
        except httpx.HTTPStatusError as e:
            verbose_proxy_logger.error(
                f"A2A: HTTP error invoking agent '{agent_id}': {e}"
            )
            task.transition(TaskState.FAILED)
            return JSONRPCResponse.error_response(
                request.id,
                A2AError.INTERNAL_ERROR["code"],
                f"HTTP error: {e.response.status_code}",
            )
        except Exception as e:
            verbose_proxy_logger.exception(
                f"A2A: Error invoking agent '{agent_id}': {e}"
            )
            task.transition(TaskState.FAILED)
            return JSONRPCResponse.error_response(
                request.id,
                A2AError.INTERNAL_ERROR["code"],
                f"Internal error: {str(e)}",
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

    async def stream_agent_response_sse(
        self, agent_id: str, request: JSONRPCRequest
    ) -> AsyncIterator[str]:
        """
        Stream response with A2A-compliant SSE event framing.

        Wraps the underlying stream with proper SSE event types:
        - ``event: task-status-update`` for state transitions
        - ``event: task-artifact-update`` for content chunks
        - Final event includes ``"final": true``

        Creates a task, transitions through submitted -> working -> completed/failed,
        and yields SSE-formatted events.

        Args:
            agent_id: The ID of the agent to invoke
            request: The JSON-RPC 2.0 request

        Yields:
            SSE-formatted event strings (``event: ...\ndata: ...\n\n``)
        """
        # Resolve method alias for the forwarded request
        resolved_method = self._resolve_method(request.method)

        # Create task for lifecycle tracking
        messages = []
        if request.params and "message" in request.params:
            messages = [request.params["message"]]
        task = self.task_store.create_task(agent_id=agent_id, messages=messages)

        # Emit initial submitted status
        yield format_sse_event(
            "task-status-update",
            {"id": task.id, "status": {"state": TaskState.SUBMITTED.value}},
        )

        # Transition to working
        task.transition(TaskState.WORKING)
        yield format_sse_event(
            "task-status-update",
            {"id": task.id, "status": {"state": TaskState.WORKING.value}},
        )

        # Build a request with the resolved method name for forwarding
        forwarded_request = JSONRPCRequest(
            method=resolved_method,
            params=request.params,
            id=request.id,
        )

        try:
            async for chunk in self.stream_agent_response(agent_id, forwarded_request):
                # Wrap each chunk as an artifact update
                yield format_sse_event(
                    "task-artifact-update",
                    {
                        "id": task.id,
                        "artifact": {
                            "parts": [{"type": "text", "text": chunk}],
                        },
                    },
                )

            # Completed
            task.transition(TaskState.COMPLETED)
            yield format_sse_event(
                "task-status-update",
                {
                    "id": task.id,
                    "status": {"state": TaskState.COMPLETED.value},
                    "final": True,
                },
            )
        except Exception as e:
            task.transition(TaskState.FAILED)
            yield format_sse_event(
                "task-status-update",
                {
                    "id": task.id,
                    "status": {
                        "state": TaskState.FAILED.value,
                        "message": str(e),
                    },
                    "final": True,
                },
            )

    async def stream_agent_response_with_headers(
        self, agent_id: str, request: JSONRPCRequest
    ) -> tuple[StreamingResponseMeta, AsyncIterator[bytes]]:
        """
        Stream response from an agent with preserved upstream headers.

        This method returns both the streaming iterator and metadata including
        preserved upstream headers (Content-Type, Cache-Control, etc.).

        Use this method when you need to preserve upstream headers on the
        streaming response (e.g., for proper SSE handling).

        Returns:
            Tuple of (StreamingResponseMeta, AsyncIterator[bytes])
            - meta: Contains preserved headers and status code
            - iterator: Raw byte iterator for streaming response

        Raises:
            ValueError: If gateway is disabled or agent not found
        """
        if not self.enabled:
            raise ValueError("A2A Gateway is not enabled")

        # Get agent under lock
        with self._lock:
            agent = self.agents.get(agent_id)

        if not agent:
            raise ValueError(f"Agent '{agent_id}' not found")

        if not agent.url:
            raise ValueError(f"Agent '{agent_id}' has no URL configured")

        # Security: Validate URL against SSRF attacks
        try:
            await validate_outbound_url_async(agent.url)
        except SSRFBlockedError as e:
            verbose_proxy_logger.warning(
                f"A2A: SSRF blocked for agent '{agent_id}': {e}"
            )
            raise ValueError(f"Agent URL blocked for security reasons: {e.reason}")
        except ValueError as e:
            verbose_proxy_logger.warning(
                f"A2A: Invalid URL for agent '{agent_id}': {e}"
            )
            raise ValueError(f"Agent URL is invalid: {str(e)}")

        verbose_proxy_logger.info(
            f"A2A: Raw streaming with headers from agent '{agent_id}'"
        )

        # Inject W3C trace context headers for distributed tracing
        headers = {"Content-Type": "application/json"}
        headers = inject_trace_headers(headers)

        # Create client and stream - caller is responsible for consuming
        client = await get_client_for_request(timeout=120.0).__aenter__()

        try:
            # Start the streaming request
            response = await client.send(
                client.build_request(
                    "POST",
                    agent.url,
                    json=request.to_dict(),
                    headers=headers,
                ),
                stream=True,
            )
            response.raise_for_status()

            # Extract preserved headers
            preserved_headers = filter_upstream_headers(response.headers)
            meta = StreamingResponseMeta(
                headers=preserved_headers,
                status_code=response.status_code,
            )

            async def chunk_iterator() -> AsyncIterator[bytes]:
                """Iterate over raw bytes and cleanup when done."""
                try:
                    async for chunk in response.aiter_bytes(
                        chunk_size=A2A_RAW_STREAMING_CHUNK_SIZE
                    ):
                        if chunk:
                            yield chunk
                finally:
                    await response.aclose()
                    await client.aclose()

            return meta, chunk_iterator()

        except Exception:
            # Cleanup on error
            await client.aclose()
            raise

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
        # Use async version to avoid blocking the event loop
        try:
            await validate_outbound_url_async(agent.url)
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

            async with get_client_for_request(timeout=120.0) as client:
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
        # Use async version to avoid blocking the event loop
        try:
            await validate_outbound_url_async(agent.url)
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

            async with get_client_for_request(timeout=120.0) as client:
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
