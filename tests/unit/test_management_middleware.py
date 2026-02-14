"""Tests for the management enhancement middleware."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm_llmrouter.management_middleware import (
    ManagementMiddleware,
    MGMT_OPERATION_ATTR,
    MGMT_RESOURCE_TYPE_ATTR,
    _get_header,
)


# =============================================================================
# ASGI helpers
# =============================================================================


def _make_scope(method: str = "POST", path: str = "/key/generate") -> dict:
    """Build a minimal ASGI HTTP scope."""
    return {
        "type": "http",
        "method": method,
        "path": path,
        "headers": [],
    }


async def _noop_receive():
    return {"type": "http.request", "body": b""}


class _ResponseCapture:
    """Captures ASGI send messages."""

    def __init__(self):
        self.messages: list[dict] = []

    async def __call__(self, message: dict) -> None:
        self.messages.append(message)

    @property
    def status_code(self) -> int:
        for msg in self.messages:
            if msg["type"] == "http.response.start":
                return msg.get("status", 0)
        return 0


async def _passthrough_app(scope, receive, send):
    """Minimal ASGI app that always returns 200."""
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [[b"content-type", b"application/json"]],
        }
    )
    await send({"type": "http.response.body", "body": b'{"ok": true}'})


# =============================================================================
# Tests
# =============================================================================


class TestManagementMiddleware:
    """Test the ManagementMiddleware ASGI layer."""

    @pytest.mark.asyncio
    async def test_non_http_scope_passes_through(self):
        """Non-HTTP scopes (websocket, lifespan) pass through unchanged."""
        app_called = False

        async def mock_app(scope, receive, send):
            nonlocal app_called
            app_called = True

        mw = ManagementMiddleware(mock_app)
        await mw({"type": "lifespan"}, _noop_receive, AsyncMock())
        assert app_called

    @pytest.mark.asyncio
    async def test_non_management_path_passes_through(self):
        """Data-plane paths like /chat/completions pass through."""
        capture = _ResponseCapture()
        mw = ManagementMiddleware(_passthrough_app)
        scope = _make_scope("POST", "/chat/completions")
        await mw(scope, _noop_receive, capture)
        assert capture.status_code == 200

    @pytest.mark.asyncio
    @patch(
        "litellm_llmrouter.management_middleware._is_management_otel_enabled",
        return_value=False,
    )
    async def test_management_path_passes_through_and_dispatches(self, _):
        """Management paths pass through to the inner app and dispatch to plugins."""
        capture = _ResponseCapture()
        mw = ManagementMiddleware(_passthrough_app)
        mw._otel_enabled = False  # Disable OTel for test isolation

        scope = _make_scope("POST", "/key/generate")

        with patch(
            "litellm_llmrouter.gateway.plugin_manager.get_plugin_manager"
        ) as mock_pm:
            mock_manager = MagicMock()
            mock_manager.is_started = True
            mock_manager.notify_management_operation = AsyncMock()
            mock_pm.return_value = mock_manager

            await mw(scope, _noop_receive, capture)

        assert capture.status_code == 200
        # Plugin manager was notified
        mock_manager.notify_management_operation.assert_called_once()
        call_kwargs = mock_manager.notify_management_operation.call_args
        assert call_kwargs[1]["metadata"]["status_code"] == 200

    @pytest.mark.asyncio
    async def test_otel_attributes_set_on_management_path(self):
        """OTel span attributes are set for management operations."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        capture = _ResponseCapture()
        mw = ManagementMiddleware(_passthrough_app)

        scope = _make_scope("POST", "/key/generate")

        with patch("opentelemetry.trace.get_current_span", return_value=mock_span):
            with patch(
                "litellm_llmrouter.gateway.plugin_manager.get_plugin_manager"
            ) as mock_pm:
                mock_manager = MagicMock()
                mock_manager.is_started = False
                mock_pm.return_value = mock_manager
                await mw(scope, _noop_receive, capture)

        # Verify OTel attributes were set
        calls = mock_span.set_attribute.call_args_list
        attr_names = [c[0][0] for c in calls]
        assert MGMT_OPERATION_ATTR in attr_names
        assert MGMT_RESOURCE_TYPE_ATTR in attr_names


class TestGetHeader:
    def test_extracts_header(self):
        headers = {b"content-type": b"application/json"}
        assert _get_header(headers, b"content-type") == "application/json"

    def test_missing_header_returns_empty(self):
        assert _get_header({}, b"x-missing") == ""

    def test_empty_value(self):
        headers = {b"x-empty": b""}
        assert _get_header(headers, b"x-empty") == ""
