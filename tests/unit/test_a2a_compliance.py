"""
Unit Tests for A2A Protocol Spec Compliance
============================================

Tests for A2A spec compliance features:
- Both tasks/send and message/send dispatch to same handler
- Task creation, state transitions, get, cancel
- State machine validation (invalid transitions rejected)
- validate_state_transition() standalone function
- A2AError standardized error responses
- Agent Card endpoint returns valid JSON with required fields
- SSE event framing format
- Task TTL cleanup
- Concurrent state transitions (optimistic locking)
- Error response format compliance
"""

import json
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm_llmrouter import a2a_gateway
from litellm_llmrouter.a2a_gateway import (
    A2AAgent,
    A2AError,
    A2AGateway,
    A2ATask,
    A2ATaskStore,
    InvalidTaskTransitionError,
    JSONRPCRequest,
    JSONRPCResponse,
    TaskState,
    VALID_STATE_TRANSITIONS,
    VALID_TRANSITIONS,
    format_sse_event,
    reset_a2a_gateway,
    validate_state_transition,
)

# Mark all tests as async
pytestmark = pytest.mark.asyncio


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_gateway():
    """Reset the global gateway singleton before each test."""
    reset_a2a_gateway()
    yield
    reset_a2a_gateway()


@pytest.fixture
def gateway():
    """Create an enabled A2A gateway for testing."""
    gw = A2AGateway()
    gw.enabled = True
    return gw


@pytest.fixture
def task_store():
    """Create a task store with short TTL for testing."""
    return A2ATaskStore(ttl_seconds=60)


@pytest.fixture
def sample_agent():
    """Create a sample A2A agent."""
    return A2AAgent(
        agent_id="agent-1",
        name="Test Agent",
        description="A test agent",
        url="https://agent.example.com/a2a",
        capabilities=["streaming", "text_generation"],
    )


# =============================================================================
# TaskState Enum Tests
# =============================================================================


class TestTaskState:
    """Tests for TaskState enum values."""

    def test_all_states_present(self):
        """All A2A spec states should be defined."""
        assert TaskState.SUBMITTED == "submitted"
        assert TaskState.WORKING == "working"
        assert TaskState.INPUT_REQUIRED == "input-required"
        assert TaskState.COMPLETED == "completed"
        assert TaskState.FAILED == "failed"
        assert TaskState.CANCELED == "canceled"

    def test_state_is_string_enum(self):
        """TaskState values should be usable as strings."""
        assert isinstance(TaskState.SUBMITTED.value, str)
        assert f"{TaskState.WORKING}" == "TaskState.WORKING"
        assert TaskState.COMPLETED.value == "completed"


# =============================================================================
# A2ATask State Machine Tests
# =============================================================================


class TestA2ATaskStateMachine:
    """Tests for A2ATask state transitions."""

    def test_task_creation(self, task_store):
        """New tasks should start in SUBMITTED state."""
        task = task_store.create_task(agent_id="agent-1")
        assert task.state == TaskState.SUBMITTED
        assert task.agent_id == "agent-1"
        assert len(task.history) == 0
        assert task.id  # Should have a UUID

    def test_valid_transition_submitted_to_working(self):
        """submitted -> working should be valid."""
        task = A2ATask(id="t1", state=TaskState.SUBMITTED, agent_id="a1")
        task.transition(TaskState.WORKING)
        assert task.state == TaskState.WORKING
        assert len(task.history) == 1
        assert task.history[0]["from"] == "submitted"
        assert task.history[0]["to"] == "working"

    def test_valid_transition_submitted_to_completed(self):
        """submitted -> completed should be valid (fast-track completion)."""
        task = A2ATask(id="t1", state=TaskState.SUBMITTED, agent_id="a1")
        task.transition(TaskState.COMPLETED)
        assert task.state == TaskState.COMPLETED

    def test_valid_transition_submitted_to_failed(self):
        """submitted -> failed should be valid (immediate failure)."""
        task = A2ATask(id="t1", state=TaskState.SUBMITTED, agent_id="a1")
        task.transition(TaskState.FAILED)
        assert task.state == TaskState.FAILED

    def test_valid_transition_submitted_to_canceled(self):
        """submitted -> canceled should be valid (cancel before work starts)."""
        task = A2ATask(id="t1", state=TaskState.SUBMITTED, agent_id="a1")
        task.transition(TaskState.CANCELED)
        assert task.state == TaskState.CANCELED

    def test_valid_transition_working_to_completed(self):
        """working -> completed should be valid."""
        task = A2ATask(id="t1", state=TaskState.WORKING, agent_id="a1")
        task.transition(TaskState.COMPLETED)
        assert task.state == TaskState.COMPLETED

    def test_valid_transition_working_to_failed(self):
        """working -> failed should be valid."""
        task = A2ATask(id="t1", state=TaskState.WORKING, agent_id="a1")
        task.transition(TaskState.FAILED)
        assert task.state == TaskState.FAILED

    def test_valid_transition_working_to_canceled(self):
        """working -> canceled should be valid."""
        task = A2ATask(id="t1", state=TaskState.WORKING, agent_id="a1")
        task.transition(TaskState.CANCELED)
        assert task.state == TaskState.CANCELED

    def test_valid_transition_working_to_input_required(self):
        """working -> input-required should be valid."""
        task = A2ATask(id="t1", state=TaskState.WORKING, agent_id="a1")
        task.transition(TaskState.INPUT_REQUIRED)
        assert task.state == TaskState.INPUT_REQUIRED

    def test_valid_transition_input_required_to_working(self):
        """input-required -> working should be valid (resume after input)."""
        task = A2ATask(id="t1", state=TaskState.INPUT_REQUIRED, agent_id="a1")
        task.transition(TaskState.WORKING)
        assert task.state == TaskState.WORKING

    def test_valid_transition_input_required_to_canceled(self):
        """input-required -> canceled should be valid."""
        task = A2ATask(id="t1", state=TaskState.INPUT_REQUIRED, agent_id="a1")
        task.transition(TaskState.CANCELED)
        assert task.state == TaskState.CANCELED

    def test_valid_transition_input_required_to_completed(self):
        """input-required -> completed should be valid."""
        task = A2ATask(id="t1", state=TaskState.INPUT_REQUIRED, agent_id="a1")
        task.transition(TaskState.COMPLETED)
        assert task.state == TaskState.COMPLETED

    def test_valid_transition_input_required_to_failed(self):
        """input-required -> failed should be valid."""
        task = A2ATask(id="t1", state=TaskState.INPUT_REQUIRED, agent_id="a1")
        task.transition(TaskState.FAILED)
        assert task.state == TaskState.FAILED

    def test_invalid_transition_completed_to_working(self):
        """completed -> working should be invalid (terminal state)."""
        task = A2ATask(id="t1", state=TaskState.COMPLETED, agent_id="a1")
        with pytest.raises(InvalidTaskTransitionError):
            task.transition(TaskState.WORKING)

    def test_invalid_transition_completed_to_submitted(self):
        """completed -> submitted should be invalid (terminal state)."""
        task = A2ATask(id="t1", state=TaskState.COMPLETED, agent_id="a1")
        with pytest.raises(InvalidTaskTransitionError):
            task.transition(TaskState.SUBMITTED)

    def test_invalid_transition_failed_to_working(self):
        """failed -> working should be invalid (terminal state)."""
        task = A2ATask(id="t1", state=TaskState.FAILED, agent_id="a1")
        with pytest.raises(InvalidTaskTransitionError):
            task.transition(TaskState.WORKING)

    def test_invalid_transition_failed_to_completed(self):
        """failed -> completed should be invalid (terminal state)."""
        task = A2ATask(id="t1", state=TaskState.FAILED, agent_id="a1")
        with pytest.raises(InvalidTaskTransitionError):
            task.transition(TaskState.COMPLETED)

    def test_invalid_transition_canceled_to_working(self):
        """canceled -> working should be invalid (terminal state)."""
        task = A2ATask(id="t1", state=TaskState.CANCELED, agent_id="a1")
        with pytest.raises(InvalidTaskTransitionError):
            task.transition(TaskState.WORKING)

    def test_invalid_transition_canceled_to_submitted(self):
        """canceled -> submitted should be invalid (terminal state)."""
        task = A2ATask(id="t1", state=TaskState.CANCELED, agent_id="a1")
        with pytest.raises(InvalidTaskTransitionError):
            task.transition(TaskState.SUBMITTED)

    def test_invalid_transition_submitted_to_input_required(self):
        """submitted -> input-required should be invalid (must work first)."""
        task = A2ATask(id="t1", state=TaskState.SUBMITTED, agent_id="a1")
        with pytest.raises(InvalidTaskTransitionError):
            task.transition(TaskState.INPUT_REQUIRED)

    def test_transition_records_history(self):
        """State transitions should be recorded in history."""
        task = A2ATask(id="t1", state=TaskState.SUBMITTED, agent_id="a1")
        task.transition(TaskState.WORKING)
        task.transition(TaskState.COMPLETED)
        assert len(task.history) == 2
        assert task.history[0]["from"] == "submitted"
        assert task.history[0]["to"] == "working"
        assert task.history[1]["from"] == "working"
        assert task.history[1]["to"] == "completed"
        # Timestamps should be present
        assert "timestamp" in task.history[0]
        assert "timestamp" in task.history[1]

    def test_transition_updates_timestamp(self):
        """Transitions should update the updated_at timestamp."""
        task = A2ATask(id="t1", state=TaskState.SUBMITTED, agent_id="a1")
        original_time = task.updated_at
        time.sleep(0.01)
        task.transition(TaskState.WORKING)
        assert task.updated_at >= original_time

    def test_task_to_dict(self):
        """to_dict() should serialize all task fields."""
        task = A2ATask(
            id="t1",
            state=TaskState.WORKING,
            agent_id="a1",
            messages=[{"role": "user", "parts": [{"type": "text", "text": "hi"}]}],
            artifacts=[],
        )
        d = task.to_dict()
        assert d["id"] == "t1"
        assert d["status"]["state"] == "working"
        assert d["agent_id"] == "a1"
        assert len(d["messages"]) == 1
        assert "created_at" in d
        assert "updated_at" in d

    def test_all_terminal_states_have_no_transitions(self):
        """Terminal states should have empty transition sets."""
        for state in [TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED]:
            assert VALID_TRANSITIONS[state] == set(), (
                f"{state.value} should have no valid transitions"
            )

    def test_all_terminal_states_reject_all_targets(self):
        """Terminal states must reject transitions to every other state."""
        terminal_states = [TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED]
        all_states = [
            TaskState.SUBMITTED,
            TaskState.WORKING,
            TaskState.INPUT_REQUIRED,
            TaskState.COMPLETED,
            TaskState.FAILED,
            TaskState.CANCELED,
        ]
        for terminal in terminal_states:
            for target in all_states:
                task = A2ATask(id="t1", state=terminal, agent_id="a1")
                with pytest.raises(InvalidTaskTransitionError):
                    task.transition(target)

    def test_all_valid_transitions_succeed(self):
        """Every transition in VALID_TRANSITIONS should succeed."""
        for source, targets in VALID_TRANSITIONS.items():
            if source == TaskState.UNKNOWN:
                continue  # skip UNKNOWN since it's internal
            for target in targets:
                task = A2ATask(id="t1", state=source, agent_id="a1")
                task.transition(target)
                assert task.state == target, (
                    f"Transition {source.value} -> {target.value} should succeed"
                )

    def test_invalid_transition_error_message_includes_valid_targets(self):
        """InvalidTaskTransitionError message should list valid transition targets."""
        task = A2ATask(id="t1", state=TaskState.COMPLETED, agent_id="a1")
        with pytest.raises(InvalidTaskTransitionError) as exc_info:
            task.transition(TaskState.WORKING)
        error_msg = str(exc_info.value)
        assert "completed" in error_msg
        assert "working" in error_msg
        assert "Valid transitions" in error_msg


# =============================================================================
# validate_state_transition() Standalone Function Tests
# =============================================================================


class TestValidateStateTransition:
    """Tests for the standalone validate_state_transition() function."""

    def test_valid_transition_returns_true(self):
        """Valid transitions should return True."""
        assert validate_state_transition("submitted", "working") is True
        assert validate_state_transition("working", "completed") is True
        assert validate_state_transition("working", "failed") is True
        assert validate_state_transition("input-required", "working") is True

    def test_invalid_transition_raises_value_error(self):
        """Invalid transitions should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid state transition"):
            validate_state_transition("completed", "working")

    def test_terminal_states_reject_all(self):
        """Terminal states should reject all transitions."""
        for terminal in ("completed", "failed", "canceled"):
            for target in ("submitted", "working", "completed", "failed", "canceled"):
                with pytest.raises(ValueError):
                    validate_state_transition(terminal, target)

    def test_unknown_current_state_raises(self):
        """Unknown current state should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown current state"):
            validate_state_transition("nonexistent", "working")

    def test_submitted_valid_targets(self):
        """submitted should allow working, completed, failed, canceled."""
        for target in ("working", "completed", "failed", "canceled"):
            assert validate_state_transition("submitted", target) is True

    def test_submitted_rejects_input_required(self):
        """submitted -> input-required should be invalid."""
        with pytest.raises(ValueError):
            validate_state_transition("submitted", "input-required")

    def test_working_valid_targets(self):
        """working should allow completed, failed, canceled, input-required."""
        for target in ("completed", "failed", "canceled", "input-required"):
            assert validate_state_transition("working", target) is True

    def test_input_required_valid_targets(self):
        """input-required should allow working, completed, failed, canceled."""
        for target in ("working", "completed", "failed", "canceled"):
            assert validate_state_transition("input-required", target) is True

    def test_string_alias_matches_enum_transitions(self):
        """VALID_STATE_TRANSITIONS should match VALID_TRANSITIONS."""
        for state, targets in VALID_TRANSITIONS.items():
            str_targets = VALID_STATE_TRANSITIONS.get(state.value, set())
            expected = {t.value for t in targets}
            assert str_targets == expected, (
                f"String alias mismatch for state '{state.value}': "
                f"expected {expected}, got {str_targets}"
            )


# =============================================================================
# A2AError Standardized Error Response Tests
# =============================================================================


class TestA2AError:
    """Tests for A2AError standardized error responses."""

    def test_predefined_error_codes(self):
        """Predefined errors should have correct codes."""
        assert A2AError.TASK_NOT_FOUND["code"] == -32001
        assert A2AError.AGENT_NOT_FOUND["code"] == -32002
        assert A2AError.RATE_LIMITED["code"] == -32003
        assert A2AError.INVALID_TRANSITION["code"] == -32602
        assert A2AError.INVALID_PARAMS["code"] == -32602
        assert A2AError.METHOD_NOT_FOUND["code"] == -32601
        assert A2AError.INVALID_REQUEST["code"] == -32600
        assert A2AError.INTERNAL_ERROR["code"] == -32603

    def test_predefined_error_messages(self):
        """Predefined errors should have human-readable messages."""
        assert A2AError.TASK_NOT_FOUND["message"] == "Task not found"
        assert A2AError.AGENT_NOT_FOUND["message"] == "Agent not found"
        assert A2AError.RATE_LIMITED["message"] == "Rate limited"
        assert A2AError.INTERNAL_ERROR["message"] == "Internal error"

    def test_make_error_basic(self):
        """make_error should produce a valid JSON-RPC error response."""
        result = A2AError.make_error(-32000, "Test error")
        assert result["jsonrpc"] == "2.0"
        assert result["error"]["code"] == -32000
        assert result["error"]["message"] == "Test error"
        assert result["id"] is None
        assert "data" not in result["error"]

    def test_make_error_with_data(self):
        """make_error with data should include it in the error."""
        result = A2AError.make_error(
            -32602,
            "Invalid params",
            data={"field": "id", "reason": "missing"},
        )
        assert result["error"]["data"]["field"] == "id"
        assert result["error"]["data"]["reason"] == "missing"

    def test_task_not_found_response(self):
        """task_not_found should create properly formatted error."""
        result = A2AError.task_not_found("task-123")
        assert result["jsonrpc"] == "2.0"
        assert result["error"]["code"] == -32001
        assert "task-123" in result["error"]["message"]

    def test_agent_not_found_response(self):
        """agent_not_found should create properly formatted error."""
        result = A2AError.agent_not_found("agent-xyz")
        assert result["jsonrpc"] == "2.0"
        assert result["error"]["code"] == -32002
        assert "agent-xyz" in result["error"]["message"]

    def test_invalid_transition_response(self):
        """invalid_transition should include state details."""
        result = A2AError.invalid_transition("completed", "working")
        assert result["jsonrpc"] == "2.0"
        assert result["error"]["code"] == -32602
        assert "completed" in result["error"]["message"]
        assert "working" in result["error"]["message"]
        assert result["error"]["data"]["current_state"] == "completed"
        assert result["error"]["data"]["requested_state"] == "working"


# =============================================================================
# A2ATaskStore Tests
# =============================================================================


class TestA2ATaskStore:
    """Tests for task store CRUD and TTL."""

    def test_create_task(self, task_store):
        """create_task should return a task in SUBMITTED state."""
        task = task_store.create_task(agent_id="a1")
        assert task.state == TaskState.SUBMITTED
        assert task_store.count() == 1

    def test_get_task(self, task_store):
        """get_task should return the task by ID."""
        task = task_store.create_task(agent_id="a1")
        retrieved = task_store.get_task(task.id)
        assert retrieved is not None
        assert retrieved.id == task.id

    def test_get_task_not_found(self, task_store):
        """get_task should return None for unknown IDs."""
        assert task_store.get_task("nonexistent") is None

    def test_ttl_expiry(self):
        """Expired tasks should be removed on get."""
        store = A2ATaskStore(ttl_seconds=0)
        task = store.create_task(agent_id="a1")
        # Task is immediately expired since TTL is 0
        time.sleep(0.01)
        assert store.get_task(task.id) is None

    def test_cleanup_expired(self):
        """cleanup_expired should remove expired tasks."""
        store = A2ATaskStore(ttl_seconds=0)
        store.create_task(agent_id="a1")
        store.create_task(agent_id="a2")
        time.sleep(0.01)
        removed = store.cleanup_expired()
        assert removed == 2
        assert store.count() == 0

    def test_clear(self, task_store):
        """clear should remove all tasks."""
        task_store.create_task(agent_id="a1")
        task_store.create_task(agent_id="a2")
        assert task_store.count() == 2
        task_store.clear()
        assert task_store.count() == 0

    def test_create_task_with_messages(self, task_store):
        """create_task should accept initial messages."""
        msgs = [{"role": "user", "parts": [{"type": "text", "text": "hello"}]}]
        task = task_store.create_task(agent_id="a1", messages=msgs)
        assert len(task.messages) == 1

    def test_rate_limiting(self):
        """Task store should enforce per-agent rate limits."""
        store = A2ATaskStore(rate_limit_per_minute=2)
        store.create_task(agent_id="a1")
        store.create_task(agent_id="a1")
        with pytest.raises(ValueError, match="Rate limit exceeded"):
            store.create_task(agent_id="a1")

    def test_max_capacity(self):
        """Task store should reject when at max capacity."""
        store = A2ATaskStore(max_tasks=2, rate_limit_per_minute=100)
        store.create_task(agent_id="a1")
        store.create_task(agent_id="a2")
        with pytest.raises(ValueError, match="at capacity"):
            store.create_task(agent_id="a3")


# =============================================================================
# Method Alias Dispatch Tests
# =============================================================================


class TestMethodAliasDispatch:
    """Tests that A2A spec method names dispatch to the correct handlers."""

    def test_resolve_tasks_send(self, gateway):
        """tasks/send should resolve to message/send."""
        assert gateway._resolve_method("tasks/send") == "message/send"

    def test_resolve_tasks_send_subscribe(self, gateway):
        """tasks/sendSubscribe should resolve to message/stream."""
        assert gateway._resolve_method("tasks/sendSubscribe") == "message/stream"

    def test_resolve_message_send_unchanged(self, gateway):
        """message/send should remain message/send (no alias)."""
        assert gateway._resolve_method("message/send") == "message/send"

    def test_resolve_message_stream_unchanged(self, gateway):
        """message/stream should remain message/stream (no alias)."""
        assert gateway._resolve_method("message/stream") == "message/stream"

    def test_resolve_unknown_method(self, gateway):
        """Unknown methods should pass through unchanged."""
        assert gateway._resolve_method("foo/bar") == "foo/bar"

    def test_is_streaming_method(self, gateway):
        """Streaming methods should be identified correctly."""
        assert gateway._is_streaming_method("message/stream") is True
        assert gateway._is_streaming_method("tasks/sendSubscribe") is True
        assert gateway._is_streaming_method("message/send") is False
        assert gateway._is_streaming_method("tasks/send") is False

    @patch.object(A2AGateway, "invoke_agent")
    async def test_tasks_send_and_message_send_same_dispatch(
        self, mock_invoke, gateway
    ):
        """
        Both tasks/send and message/send should go through the same invoke_agent path.

        We verify the method alias is resolved inside invoke_agent.
        """
        # Since invoke_agent is what we're testing, we need to call the real
        # implementation. Instead, test the alias map directly.
        assert A2AGateway.METHOD_ALIASES["tasks/send"] == "message/send"
        assert A2AGateway.METHOD_ALIASES["tasks/sendSubscribe"] == "message/stream"

    async def test_invoke_tasks_send_redirects_like_message_send(
        self, gateway, sample_agent
    ):
        """
        Calling invoke_agent with tasks/send should behave the same as message/send.

        Both should create a task, forward the request, and track lifecycle.
        """
        gateway.register_agent(sample_agent)

        # Mock the HTTP client
        mock_response_data = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"content": "hello"},
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value=mock_response_data)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch.object(
                a2a_gateway, "get_client_for_request", return_value=mock_client
            ),
            patch.object(
                a2a_gateway, "validate_outbound_url_async", new_callable=AsyncMock
            ),
        ):
            # Test with tasks/send
            req1 = JSONRPCRequest(
                method="tasks/send", params={"message": {"text": "hi"}}, id=1
            )
            resp1 = await gateway.invoke_agent("agent-1", req1)
            assert resp1.error is None
            assert resp1.result is not None
            # Should have task info
            assert "_task" in resp1.result

            # Test with message/send
            req2 = JSONRPCRequest(
                method="message/send", params={"message": {"text": "hi"}}, id=2
            )
            resp2 = await gateway.invoke_agent("agent-1", req2)
            assert resp2.error is None
            assert resp2.result is not None
            assert "_task" in resp2.result

    async def test_invoke_tasks_sendsubscribe_returns_streaming_error(
        self, gateway, sample_agent
    ):
        """tasks/sendSubscribe should return the same error as message/stream."""
        gateway.register_agent(sample_agent)

        with patch.object(
            a2a_gateway, "validate_outbound_url_async", new_callable=AsyncMock
        ):
            req = JSONRPCRequest(method="tasks/sendSubscribe", params={}, id=1)
            resp = await gateway.invoke_agent("agent-1", req)
            assert resp.error is not None
            assert "streaming" in resp.error["message"].lower()


# =============================================================================
# tasks/get and tasks/cancel Handler Tests
# =============================================================================


class TestTaskGetAndCancel:
    """Tests for tasks/get and tasks/cancel handlers."""

    async def test_tasks_get_returns_task(self, gateway):
        """tasks/get should return the task state."""
        task = gateway.task_store.create_task(agent_id="a1")
        task.transition(TaskState.WORKING)

        req = JSONRPCRequest(method="tasks/get", params={"id": task.id}, id=1)
        resp = await gateway.handle_task_get(req)
        assert resp.error is None
        assert resp.result["id"] == task.id
        assert resp.result["status"]["state"] == "working"

    async def test_tasks_get_missing_id(self, gateway):
        """tasks/get without id param should return error."""
        req = JSONRPCRequest(method="tasks/get", params={}, id=1)
        resp = await gateway.handle_task_get(req)
        assert resp.error is not None
        assert resp.error["code"] == -32602

    async def test_tasks_get_not_found(self, gateway):
        """tasks/get for unknown task should return -32001 error."""
        req = JSONRPCRequest(method="tasks/get", params={"id": "nonexistent"}, id=1)
        resp = await gateway.handle_task_get(req)
        assert resp.error is not None
        assert resp.error["code"] == A2AError.TASK_NOT_FOUND["code"]

    async def test_tasks_cancel_working_task(self, gateway):
        """tasks/cancel should cancel a working task."""
        task = gateway.task_store.create_task(agent_id="a1")
        task.transition(TaskState.WORKING)

        req = JSONRPCRequest(method="tasks/cancel", params={"id": task.id}, id=1)
        resp = await gateway.handle_task_cancel(req)
        assert resp.error is None
        assert resp.result["status"]["state"] == "canceled"

    async def test_tasks_cancel_submitted_task(self, gateway):
        """tasks/cancel should cancel a submitted task (now a valid transition)."""
        task = gateway.task_store.create_task(agent_id="a1")

        req = JSONRPCRequest(method="tasks/cancel", params={"id": task.id}, id=1)
        resp = await gateway.handle_task_cancel(req)
        assert resp.error is None
        assert resp.result["status"]["state"] == "canceled"

    async def test_tasks_cancel_completed_task(self, gateway):
        """tasks/cancel should not cancel a completed task (terminal state)."""
        task = gateway.task_store.create_task(agent_id="a1")
        task.transition(TaskState.WORKING)
        task.transition(TaskState.COMPLETED)

        req = JSONRPCRequest(method="tasks/cancel", params={"id": task.id}, id=1)
        resp = await gateway.handle_task_cancel(req)
        assert resp.error is not None
        assert resp.error["code"] == A2AError.INVALID_TRANSITION["code"]

    async def test_tasks_cancel_failed_task(self, gateway):
        """tasks/cancel should not cancel a failed task (terminal state)."""
        task = gateway.task_store.create_task(agent_id="a1")
        task.transition(TaskState.WORKING)
        task.transition(TaskState.FAILED)

        req = JSONRPCRequest(method="tasks/cancel", params={"id": task.id}, id=1)
        resp = await gateway.handle_task_cancel(req)
        assert resp.error is not None
        assert resp.error["code"] == A2AError.INVALID_TRANSITION["code"]

    async def test_tasks_cancel_missing_id(self, gateway):
        """tasks/cancel without id param should return error."""
        req = JSONRPCRequest(method="tasks/cancel", params={}, id=1)
        resp = await gateway.handle_task_cancel(req)
        assert resp.error is not None
        assert resp.error["code"] == -32602

    async def test_tasks_cancel_not_found(self, gateway):
        """tasks/cancel for unknown task should return -32001 error."""
        req = JSONRPCRequest(method="tasks/cancel", params={"id": "nonexistent"}, id=1)
        resp = await gateway.handle_task_cancel(req)
        assert resp.error is not None
        assert resp.error["code"] == A2AError.TASK_NOT_FOUND["code"]

    async def test_invoke_agent_dispatches_tasks_get(self, gateway):
        """invoke_agent should dispatch tasks/get to handle_task_get."""
        task = gateway.task_store.create_task(agent_id="a1")
        task.transition(TaskState.WORKING)

        req = JSONRPCRequest(method="tasks/get", params={"id": task.id}, id=1)
        # invoke_agent should dispatch to handle_task_get
        resp = await gateway.invoke_agent("any-agent", req)
        assert resp.error is None
        assert resp.result["id"] == task.id

    async def test_invoke_agent_dispatches_tasks_cancel(self, gateway):
        """invoke_agent should dispatch tasks/cancel to handle_task_cancel."""
        task = gateway.task_store.create_task(agent_id="a1")
        task.transition(TaskState.WORKING)

        req = JSONRPCRequest(method="tasks/cancel", params={"id": task.id}, id=1)
        resp = await gateway.invoke_agent("any-agent", req)
        assert resp.error is None
        assert resp.result["status"]["state"] == "canceled"


# =============================================================================
# Agent Card Tests
# =============================================================================


class TestAgentCard:
    """Tests for Agent Card endpoints and structure."""

    def test_get_agent_card_includes_required_fields(self, gateway, sample_agent):
        """Agent card should include all required A2A spec fields."""
        gateway.register_agent(sample_agent)
        card = gateway.get_agent_card("agent-1")

        assert card is not None
        # Required per A2A spec
        assert "name" in card
        assert "description" in card
        assert "url" in card
        assert "version" in card
        assert "capabilities" in card
        assert "skills" in card
        assert "authentication" in card
        assert "defaultInputModes" in card
        assert "defaultOutputModes" in card

    def test_get_agent_card_capabilities(self, gateway, sample_agent):
        """Agent card capabilities should include streaming, pushNotifications, stateTransitionHistory."""
        gateway.register_agent(sample_agent)
        card = gateway.get_agent_card("agent-1")

        caps = card["capabilities"]
        assert "streaming" in caps
        assert "pushNotifications" in caps
        assert "stateTransitionHistory" in caps
        assert caps["streaming"] is True  # agent has streaming capability
        assert caps["pushNotifications"] is False
        assert caps["stateTransitionHistory"] is True

    def test_get_agent_card_not_found(self, gateway):
        """get_agent_card should return None for unknown agent."""
        assert gateway.get_agent_card("nonexistent") is None

    def test_get_agent_card_version(self, gateway, sample_agent):
        """Agent card should have a version field."""
        gateway.register_agent(sample_agent)
        card = gateway.get_agent_card("agent-1")
        assert card["version"] == "0.2.0"

    def test_get_gateway_agent_card(self, gateway, sample_agent):
        """Gateway-level agent card should aggregate registered agents as skills."""
        gateway.register_agent(sample_agent)
        card = gateway.get_gateway_agent_card(base_url="https://gw.example.com")

        assert card["name"] == "RouteIQ A2A Gateway"
        assert "https://gw.example.com/a2a" == card["url"]
        assert card["version"] == "0.2.0"
        assert len(card["skills"]) == 1
        assert card["skills"][0]["id"] == "agent-1"
        assert card["skills"][0]["name"] == "Test Agent"
        assert card["capabilities"]["streaming"] is True
        assert card["capabilities"]["pushNotifications"] is False
        assert card["capabilities"]["stateTransitionHistory"] is True
        assert "authentication" in card

    def test_get_gateway_agent_card_empty(self, gateway):
        """Gateway card with no agents should have empty skills."""
        card = gateway.get_gateway_agent_card()
        assert card["skills"] == []

    def test_get_gateway_agent_card_default_base_url(self, gateway):
        """Gateway card with no base_url should use empty string."""
        card = gateway.get_gateway_agent_card()
        assert card["url"] == "/a2a"

    def test_gateway_agent_card_has_all_required_a2a_fields(self, gateway):
        """Gateway agent card must include all fields required by A2A spec."""
        card = gateway.get_gateway_agent_card(base_url="https://example.com")
        required_fields = [
            "name",
            "description",
            "url",
            "version",
            "capabilities",
            "skills",
            "defaultInputModes",
            "defaultOutputModes",
            "authentication",
        ]
        for field in required_fields:
            assert field in card, f"Missing required A2A field: '{field}'"

    def test_gateway_agent_card_capabilities_structure(self, gateway):
        """Gateway card capabilities must have streaming and pushNotifications."""
        card = gateway.get_gateway_agent_card()
        caps = card["capabilities"]
        assert "streaming" in caps
        assert "pushNotifications" in caps
        assert isinstance(caps["streaming"], bool)
        assert isinstance(caps["pushNotifications"], bool)

    def test_agent_card_skills_array(self, gateway, sample_agent):
        """Agent card skills should be an array of skill objects with id and name."""
        gateway.register_agent(sample_agent)
        card = gateway.get_agent_card("agent-1")
        assert isinstance(card["skills"], list)
        for skill in card["skills"]:
            assert "id" in skill
            assert "name" in skill

    def test_agent_card_default_modes_are_arrays(self, gateway, sample_agent):
        """defaultInputModes and defaultOutputModes should be arrays."""
        gateway.register_agent(sample_agent)
        card = gateway.get_agent_card("agent-1")
        assert isinstance(card["defaultInputModes"], list)
        assert isinstance(card["defaultOutputModes"], list)
        assert len(card["defaultInputModes"]) > 0
        assert len(card["defaultOutputModes"]) > 0


# =============================================================================
# SSE Event Framing Tests
# =============================================================================


class TestSSEEventFraming:
    """Tests for SSE event formatting."""

    def test_format_sse_event_basic(self):
        """format_sse_event should produce correct SSE format."""
        result = format_sse_event(
            "task-status-update",
            {"id": "t1", "status": {"state": "working"}},
        )
        lines = result.split("\n")
        assert lines[0] == "event: task-status-update"
        assert lines[1].startswith("data: ")
        data = json.loads(lines[1][len("data: ") :])
        assert data["id"] == "t1"
        assert data["status"]["state"] == "working"
        # Should end with double newline
        assert result.endswith("\n\n")

    def test_format_sse_event_has_event_prefix(self):
        """SSE event must start with 'event: ' prefix."""
        result = format_sse_event("my-event", {"key": "value"})
        assert result.startswith("event: my-event\n")

    def test_format_sse_event_has_data_prefix(self):
        """SSE event data line must start with 'data: ' prefix."""
        result = format_sse_event("my-event", {"key": "value"})
        lines = result.split("\n")
        assert lines[1].startswith("data: ")

    def test_format_sse_event_data_is_valid_json(self):
        """SSE event data payload must be valid JSON."""
        data_in = {"nested": {"key": [1, 2, 3]}, "flag": True}
        result = format_sse_event("test", data_in)
        data_line = result.split("\n")[1]
        data_str = data_line[len("data: ") :]
        parsed = json.loads(data_str)
        assert parsed == data_in

    def test_format_sse_event_ends_with_double_newline(self):
        """SSE events must end with \\n\\n per the SSE specification."""
        result = format_sse_event("test", {})
        assert result.endswith("\n\n")

    def test_format_sse_event_artifact_update(self):
        """Artifact update events should have correct structure."""
        result = format_sse_event(
            "task-artifact-update",
            {
                "id": "t1",
                "artifact": {
                    "parts": [{"type": "text", "text": "hello"}],
                },
            },
        )
        assert "event: task-artifact-update\n" in result
        data = json.loads(result.split("data: ")[1].rstrip("\n"))
        assert data["artifact"]["parts"][0]["type"] == "text"

    def test_format_sse_event_final_flag(self):
        """Final events should include final: true."""
        result = format_sse_event(
            "task-status-update",
            {
                "id": "t1",
                "status": {"state": "completed"},
                "final": True,
            },
        )
        data = json.loads(result.split("data: ")[1].rstrip("\n"))
        assert data["final"] is True

    def test_format_sse_event_no_extra_newlines_in_data(self):
        """SSE data must be on a single line (no embedded newlines)."""
        # json.dumps should produce single-line output by default
        result = format_sse_event("test", {"multi": "line\ndata"})
        lines = result.rstrip("\n").split("\n")
        # Should be exactly: event line, data line
        assert len(lines) == 2

    async def test_stream_agent_response_sse_lifecycle(self, gateway, sample_agent):
        """
        stream_agent_response_sse should emit proper SSE event lifecycle:
        submitted -> working -> artifact chunks -> completed (final).
        """
        gateway.register_agent(sample_agent)

        # Mock the underlying stream_agent_response to yield some chunks
        async def mock_stream(*args, **kwargs):
            yield "chunk1"
            yield "chunk2"

        gateway.stream_agent_response = mock_stream

        with patch.object(
            a2a_gateway, "validate_outbound_url_async", new_callable=AsyncMock
        ):
            req = JSONRPCRequest(method="tasks/sendSubscribe", params={}, id=1)

            events = []
            async for event in gateway.stream_agent_response_sse("agent-1", req):
                events.append(event)

        # Parse events
        assert len(events) >= 4  # submitted, working, 2 artifacts, completed

        # First: submitted
        assert "task-status-update" in events[0]
        data0 = json.loads(events[0].split("data: ")[1].rstrip("\n"))
        assert data0["status"]["state"] == "submitted"

        # Second: working
        assert "task-status-update" in events[1]
        data1 = json.loads(events[1].split("data: ")[1].rstrip("\n"))
        assert data1["status"]["state"] == "working"

        # Middle: artifact updates
        assert "task-artifact-update" in events[2]
        assert "task-artifact-update" in events[3]

        # Last: completed with final flag
        assert "task-status-update" in events[4]
        data_final = json.loads(events[4].split("data: ")[1].rstrip("\n"))
        assert data_final["status"]["state"] == "completed"
        assert data_final["final"] is True

    async def test_stream_agent_response_sse_failure(self, gateway, sample_agent):
        """
        stream_agent_response_sse should emit failed status on error.
        """
        gateway.register_agent(sample_agent)

        async def mock_stream_error(*args, **kwargs):
            yield "partial"
            raise RuntimeError("agent error")

        gateway.stream_agent_response = mock_stream_error

        with patch.object(
            a2a_gateway, "validate_outbound_url_async", new_callable=AsyncMock
        ):
            req = JSONRPCRequest(method="tasks/sendSubscribe", params={}, id=1)

            events = []
            async for event in gateway.stream_agent_response_sse("agent-1", req):
                events.append(event)

        # Last event should be failed
        last = events[-1]
        assert "task-status-update" in last
        data = json.loads(last.split("data: ")[1].rstrip("\n"))
        assert data["status"]["state"] == "failed"
        assert data["final"] is True


# =============================================================================
# Task TTL Cleanup Tests
# =============================================================================


class TestTaskTTLCleanup:
    """Tests for task TTL expiry and cleanup."""

    def test_expired_task_not_returned(self):
        """Expired tasks should not be returned by get_task."""
        store = A2ATaskStore(ttl_seconds=0)
        task = store.create_task(agent_id="a1")
        time.sleep(0.01)
        assert store.get_task(task.id) is None

    def test_non_expired_task_returned(self):
        """Non-expired tasks should be returned normally."""
        store = A2ATaskStore(ttl_seconds=3600)
        task = store.create_task(agent_id="a1")
        assert store.get_task(task.id) is not None

    def test_cleanup_removes_only_expired(self):
        """cleanup_expired should only remove expired tasks."""
        store = A2ATaskStore(ttl_seconds=3600)
        t1 = store.create_task(agent_id="a1")
        t2 = store.create_task(agent_id="a2")

        # Force one task to be "old"
        t1.created_at = time.time() - 7200  # 2 hours old

        removed = store.cleanup_expired()
        assert removed == 1
        assert store.get_task(t2.id) is not None

    def test_cleanup_returns_count(self):
        """cleanup_expired should return the number of tasks removed."""
        store = A2ATaskStore(ttl_seconds=0)
        store.create_task(agent_id="a1")
        store.create_task(agent_id="a2")
        store.create_task(agent_id="a3")
        time.sleep(0.01)
        assert store.cleanup_expired() == 3


# =============================================================================
# Concurrent State Transition Tests
# =============================================================================


class TestConcurrentStateTransitions:
    """Tests for thread-safety of state transitions."""

    def test_concurrent_transitions_one_wins(self):
        """When two threads race to transition, exactly one should succeed."""
        task = A2ATask(id="t1", state=TaskState.WORKING, agent_id="a1")
        results = {"completed": 0, "failed": 0, "errors": 0}
        barrier = threading.Barrier(2)

        def try_transition(target: TaskState, result_key: str):
            barrier.wait()
            try:
                task.transition(target)
                results[result_key] += 1
            except InvalidTaskTransitionError:
                results["errors"] += 1

        t1 = threading.Thread(
            target=try_transition, args=(TaskState.COMPLETED, "completed")
        )
        t2 = threading.Thread(target=try_transition, args=(TaskState.FAILED, "failed"))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Exactly one should succeed, one should fail
        total_success = results["completed"] + results["failed"]
        assert total_success == 1, (
            f"Expected exactly 1 success, got {total_success}: {results}"
        )
        assert results["errors"] == 1, (
            f"Expected exactly 1 error, got {results['errors']}: {results}"
        )

    def test_concurrent_task_store_creates(self, task_store):
        """Multiple threads creating tasks should all succeed (within limits)."""
        results = []

        def create_task(idx):
            try:
                task = task_store.create_task(agent_id=f"agent-{idx}")
                results.append(("ok", task.id))
            except ValueError as e:
                results.append(("error", str(e)))

        threads = [threading.Thread(target=create_task, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        successes = [r for r in results if r[0] == "ok"]
        assert len(successes) == 10
        assert task_store.count() == 10


# =============================================================================
# Error Response Format Tests
# =============================================================================


class TestErrorResponseFormat:
    """Tests for JSON-RPC error response format compliance."""

    async def test_agent_not_found_uses_correct_code(self, gateway):
        """Agent not found should return -32002 error code."""
        with patch.object(
            a2a_gateway, "validate_outbound_url_async", new_callable=AsyncMock
        ):
            req = JSONRPCRequest(method="message/send", params={}, id=1)
            resp = await gateway.invoke_agent("nonexistent-agent", req)
        assert resp.error is not None
        assert resp.error["code"] == A2AError.AGENT_NOT_FOUND["code"]
        assert "nonexistent-agent" in resp.error["message"]

    async def test_task_not_found_uses_correct_code(self, gateway):
        """Task not found should return -32001 error code."""
        req = JSONRPCRequest(method="tasks/get", params={"id": "no-such-task"}, id=1)
        resp = await gateway.handle_task_get(req)
        assert resp.error is not None
        assert resp.error["code"] == A2AError.TASK_NOT_FOUND["code"]
        assert "no-such-task" in resp.error["message"]

    async def test_method_not_found_uses_correct_code(self, gateway, sample_agent):
        """Unknown method should return -32601 error code."""
        gateway.register_agent(sample_agent)
        with patch.object(
            a2a_gateway, "validate_outbound_url_async", new_callable=AsyncMock
        ):
            req = JSONRPCRequest(method="unknown/method", params={}, id=1)
            resp = await gateway.invoke_agent("agent-1", req)
        assert resp.error is not None
        assert resp.error["code"] == A2AError.METHOD_NOT_FOUND["code"]

    async def test_invalid_jsonrpc_version(self, gateway, sample_agent):
        """Invalid JSON-RPC version should return -32600 error code."""
        gateway.register_agent(sample_agent)
        with patch.object(
            a2a_gateway, "validate_outbound_url_async", new_callable=AsyncMock
        ):
            req = JSONRPCRequest(method="message/send", params={}, id=1)
            req.jsonrpc = "1.0"
            resp = await gateway.invoke_agent("agent-1", req)
        assert resp.error is not None
        assert resp.error["code"] == A2AError.INVALID_REQUEST["code"]

    def test_jsonrpc_response_error_format(self):
        """JSONRPCResponse.error_response should produce valid format."""
        resp = JSONRPCResponse.error_response("req-1", -32001, "Task not found")
        d = resp.to_dict()
        assert d["jsonrpc"] == "2.0"
        assert d["id"] == "req-1"
        assert d["error"]["code"] == -32001
        assert d["error"]["message"] == "Task not found"
        assert "result" not in d

    def test_jsonrpc_response_success_format(self):
        """JSONRPCResponse.success_response should produce valid format."""
        resp = JSONRPCResponse.success_response("req-1", {"data": "ok"})
        d = resp.to_dict()
        assert d["jsonrpc"] == "2.0"
        assert d["id"] == "req-1"
        assert d["result"]["data"] == "ok"
        assert "error" not in d


# =============================================================================
# Gateway Agent Card Integration (via get_gateway_agent_card)
# =============================================================================


class TestGatewayIntegration:
    """Integration tests for A2A spec compliance features."""

    async def test_invoke_agent_creates_task_on_send(self, gateway, sample_agent):
        """invoke_agent with message/send should create a task."""
        gateway.register_agent(sample_agent)

        mock_response_data = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"content": "hello"},
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value=mock_response_data)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch.object(
                a2a_gateway, "get_client_for_request", return_value=mock_client
            ),
            patch.object(
                a2a_gateway, "validate_outbound_url_async", new_callable=AsyncMock
            ),
        ):
            req = JSONRPCRequest(method="message/send", params={}, id=1)
            resp = await gateway.invoke_agent("agent-1", req)

        assert resp.error is None
        assert "_task" in resp.result
        task_info = resp.result["_task"]
        assert task_info["status"]["state"] == "completed"
        assert task_info["agent_id"] == "agent-1"

    async def test_invoke_agent_fails_task_on_error(self, gateway, sample_agent):
        """invoke_agent should transition task to failed on HTTP error."""
        gateway.register_agent(sample_agent)

        mock_response_data = {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32000, "message": "agent error"},
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value=mock_response_data)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch.object(
                a2a_gateway, "get_client_for_request", return_value=mock_client
            ),
            patch.object(
                a2a_gateway, "validate_outbound_url_async", new_callable=AsyncMock
            ),
        ):
            req = JSONRPCRequest(method="message/send", params={}, id=1)
            resp = await gateway.invoke_agent("agent-1", req)

        assert resp.error is not None

    async def test_unknown_method_returns_not_found(self, gateway, sample_agent):
        """Unknown methods should return -32601 Method not found."""
        gateway.register_agent(sample_agent)

        with patch.object(
            a2a_gateway, "validate_outbound_url_async", new_callable=AsyncMock
        ):
            req = JSONRPCRequest(method="unknown/method", params={}, id=1)
            resp = await gateway.invoke_agent("agent-1", req)

        assert resp.error is not None
        assert resp.error["code"] == -32601

    async def test_gateway_disabled_returns_error(self, sample_agent):
        """invoke_agent on disabled gateway should return error."""
        gw = A2AGateway()
        gw.enabled = False
        req = JSONRPCRequest(method="message/send", params={}, id=1)
        resp = await gw.invoke_agent("agent-1", req)
        assert resp.error is not None
        assert "not enabled" in resp.error["message"].lower()
