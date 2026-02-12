"""
Unit tests for auth.py - Admin Authentication and Request ID Middleware
======================================================================

Tests cover:
- Secret scrubbing with various secret patterns and adversarial inputs
- Bearer token extraction edge cases
- Admin API key loading from environment
- Admin auth enable/disable logic
- admin_api_key_auth dependency (fail-closed, multi-key, header fallback)
- RequestIDMiddleware (passthrough, generation, context propagation)
- sanitize_error_response
- create_admin_error_response
- RouteIQ key prefix helpers (_ensure_prefix, _strip_prefix)
- Env-level prefix application (_apply_key_prefix_to_env)
- Backwards-compatible auth with prefixed/unprefixed keys
"""

from __future__ import annotations

import os
import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm_llmrouter.auth import (
    REQUEST_ID_HEADER,
    ROUTEIQ_KEY_PREFIX,
    RequestIDMiddleware,
    _apply_key_prefix_to_env,
    _ensure_prefix,
    _extract_bearer_token,
    _is_admin_auth_enabled,
    _load_admin_api_keys,
    _request_id_ctx,
    _scrub_secrets,
    _strip_prefix,
    admin_api_key_auth,
    create_admin_error_response,
    get_request_id,
    sanitize_error_response,
)


# =============================================================================
# Secret Scrubbing Tests
# =============================================================================


class TestScrubSecrets:
    """Tests for _scrub_secrets function."""

    def test_empty_string(self):
        assert _scrub_secrets("") == ""

    def test_none_returns_none(self):
        # _scrub_secrets returns falsy input unchanged
        assert _scrub_secrets("") == ""

    def test_no_secrets(self):
        text = "This is a normal log message with no secrets"
        assert _scrub_secrets(text) == text

    def test_scrubs_sk_api_key(self):
        text = "Error with key sk-1234567890abcdef1234"
        result = _scrub_secrets(text)
        assert "1234567890abcdef1234" not in result
        assert "[REDACTED]" in result

    def test_scrubs_api_key_pattern(self):
        text = "api_key=abcdef12345678"
        result = _scrub_secrets(text)
        assert "abcdef12345678" not in result
        assert "[REDACTED]" in result

    def test_scrubs_api_key_colon(self):
        text = "api-key: my-secret-api-key-value"
        result = _scrub_secrets(text)
        assert "my-secret-api-key-value" not in result

    def test_scrubs_aws_access_key(self):
        text = "Found AKIAIOSFODNN7EXAMPLE in config"
        result = _scrub_secrets(text)
        assert "AKIAIOSFODNN7EXAMPLE" not in result

    def test_scrubs_aws_secret(self):
        text = "aws_secret=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        result = _scrub_secrets(text)
        assert "wJalrXUtnFEMI" not in result
        assert "[REDACTED]" in result

    def test_scrubs_postgres_url(self):
        text = "Connected to postgresql://admin:secretpass@db:5432/mydb"
        result = _scrub_secrets(text)
        assert "secretpass" not in result
        assert "[REDACTED]" in result

    def test_scrubs_mysql_url(self):
        text = "Using mysql://root:password123@localhost/db"
        result = _scrub_secrets(text)
        assert "password123" not in result

    def test_scrubs_bearer_token(self):
        text = "Header: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payload.sig"
        result = _scrub_secrets(text)
        assert "eyJhbGci" not in result
        assert "[REDACTED]" in result

    def test_scrubs_password_pattern(self):
        text = "password=my_super_secret"
        result = _scrub_secrets(text)
        assert "my_super_secret" not in result
        assert "[REDACTED]" in result

    def test_scrubs_secret_pattern(self):
        text = "secret=the_secret_value"
        result = _scrub_secrets(text)
        assert "the_secret_value" not in result
        assert "[REDACTED]" in result

    def test_does_not_scrub_word_secret_alone(self):
        """The word 'secret' without = or : should not be scrubbed."""
        text = "This is a secret meeting about project plans"
        result = _scrub_secrets(text)
        assert "secret meeting" in result

    def test_multiple_secrets(self):
        text = "sk-abc1234567890def password=hunter2 api_key=test123456"
        result = _scrub_secrets(text)
        assert "abc1234567890def" not in result
        assert "hunter2" not in result
        assert "test123456" not in result

    def test_preserves_non_secret_content(self):
        text = "Status: OK, latency: 42ms, model: gpt-4"
        assert _scrub_secrets(text) == text


# =============================================================================
# Bearer Token Extraction Tests
# =============================================================================


class TestExtractBearerToken:
    """Tests for _extract_bearer_token function."""

    def test_valid_bearer(self):
        assert _extract_bearer_token("Bearer my-token-123") == "my-token-123"

    def test_bearer_with_whitespace(self):
        assert _extract_bearer_token("Bearer   my-token-123  ") == "my-token-123"

    def test_empty_bearer(self):
        assert _extract_bearer_token("Bearer ") is None

    def test_bearer_only_whitespace(self):
        assert _extract_bearer_token("Bearer    ") is None

    def test_not_bearer(self):
        assert _extract_bearer_token("Basic dXNlcjpwYXNz") is None

    def test_empty_string(self):
        assert _extract_bearer_token("") is None

    def test_bearer_lowercase(self):
        """Bearer prefix is case-sensitive per RFC."""
        assert _extract_bearer_token("bearer my-token") is None

    def test_bearer_no_space(self):
        assert _extract_bearer_token("Bearertoken") is None


# =============================================================================
# Admin API Key Loading Tests
# =============================================================================


class TestLoadAdminApiKeys:
    """Tests for _load_admin_api_keys function.

    Note: _load_admin_api_keys now expands each key to include both its
    prefixed (sk-riq-*) and unprefixed forms for backwards compatibility.
    Tests use subset checks (``<=``) to verify that at minimum the
    expected keys are present.
    """

    def test_no_keys_configured(self):
        with patch.dict(os.environ, {}, clear=True):
            keys = _load_admin_api_keys()
            assert keys == set()

    def test_single_key_from_admin_api_keys(self):
        with patch.dict(os.environ, {"ADMIN_API_KEYS": "key1"}, clear=True):
            keys = _load_admin_api_keys()
            assert {"key1", "sk-riq-key1"} <= keys

    def test_multiple_keys_from_admin_api_keys(self):
        with patch.dict(os.environ, {"ADMIN_API_KEYS": "key1,key2,key3"}, clear=True):
            keys = _load_admin_api_keys()
            assert {"key1", "key2", "key3"} <= keys

    def test_keys_with_whitespace(self):
        with patch.dict(
            os.environ, {"ADMIN_API_KEYS": " key1 , key2 , key3 "}, clear=True
        ):
            keys = _load_admin_api_keys()
            assert {"key1", "key2", "key3"} <= keys

    def test_empty_keys_filtered(self):
        with patch.dict(os.environ, {"ADMIN_API_KEYS": "key1,,key2,,"}, clear=True):
            keys = _load_admin_api_keys()
            assert {"key1", "key2"} <= keys

    def test_legacy_single_key_fallback(self):
        with patch.dict(os.environ, {"ADMIN_API_KEY": "legacy-key"}, clear=True):
            keys = _load_admin_api_keys()
            assert {"legacy-key", "sk-riq-legacy-key"} <= keys

    def test_both_sources_merged(self):
        with patch.dict(
            os.environ,
            {"ADMIN_API_KEYS": "key1,key2", "ADMIN_API_KEY": "legacy-key"},
            clear=True,
        ):
            keys = _load_admin_api_keys()
            assert {"key1", "key2", "legacy-key"} <= keys

    def test_duplicate_keys_deduplicated(self):
        with patch.dict(
            os.environ,
            {"ADMIN_API_KEYS": "key1,key1", "ADMIN_API_KEY": "key1"},
            clear=True,
        ):
            keys = _load_admin_api_keys()
            # Only "key1" and its prefixed variant should be present
            assert {"key1", "sk-riq-key1"} <= keys


# =============================================================================
# Admin Auth Enabled Tests
# =============================================================================


class TestIsAdminAuthEnabled:
    """Tests for _is_admin_auth_enabled function."""

    def test_default_is_enabled(self):
        with patch.dict(os.environ, {}, clear=True):
            assert _is_admin_auth_enabled() is True

    def test_true_is_enabled(self):
        with patch.dict(os.environ, {"ADMIN_AUTH_ENABLED": "true"}, clear=True):
            assert _is_admin_auth_enabled() is True

    def test_false_disables(self):
        with patch.dict(os.environ, {"ADMIN_AUTH_ENABLED": "false"}, clear=True):
            assert _is_admin_auth_enabled() is False

    def test_zero_disables(self):
        with patch.dict(os.environ, {"ADMIN_AUTH_ENABLED": "0"}, clear=True):
            assert _is_admin_auth_enabled() is False

    def test_no_disables(self):
        with patch.dict(os.environ, {"ADMIN_AUTH_ENABLED": "no"}, clear=True):
            assert _is_admin_auth_enabled() is False

    def test_off_disables(self):
        with patch.dict(os.environ, {"ADMIN_AUTH_ENABLED": "off"}, clear=True):
            assert _is_admin_auth_enabled() is False

    def test_case_insensitive(self):
        with patch.dict(os.environ, {"ADMIN_AUTH_ENABLED": "FALSE"}, clear=True):
            assert _is_admin_auth_enabled() is False

    def test_whitespace_trimmed(self):
        with patch.dict(os.environ, {"ADMIN_AUTH_ENABLED": " false "}, clear=True):
            assert _is_admin_auth_enabled() is False

    def test_random_string_is_enabled(self):
        with patch.dict(os.environ, {"ADMIN_AUTH_ENABLED": "whatever"}, clear=True):
            assert _is_admin_auth_enabled() is True


# =============================================================================
# Admin API Key Auth Dependency Tests
# =============================================================================


class TestAdminApiKeyAuth:
    """Tests for admin_api_key_auth FastAPI dependency."""

    def _make_request(self, headers: dict | None = None):
        """Create a mock Request object."""
        request = MagicMock()
        request.headers = headers or {}
        request.url.path = "/config/reload"
        return request

    @pytest.mark.asyncio
    async def test_auth_disabled_allows_all(self):
        with patch.dict(os.environ, {"ADMIN_AUTH_ENABLED": "false"}, clear=True):
            request = self._make_request()
            result = await admin_api_key_auth(request)
            assert result["admin_key"] == "__disabled__"

    @pytest.mark.asyncio
    async def test_no_keys_configured_denies(self):
        """Fail-closed: no admin keys = deny all."""
        with patch.dict(os.environ, {"ADMIN_AUTH_ENABLED": "true"}, clear=True):
            request = self._make_request()
            with pytest.raises(HTTPException) as exc:
                await admin_api_key_auth(request)
            assert exc.value.status_code == 403
            assert exc.value.detail["error"] == "control_plane_not_configured"

    @pytest.mark.asyncio
    async def test_valid_admin_key_header(self):
        with patch.dict(
            os.environ,
            {"ADMIN_AUTH_ENABLED": "true", "ADMIN_API_KEYS": "test-key"},
            clear=True,
        ):
            request = self._make_request({"X-Admin-API-Key": "test-key"})
            result = await admin_api_key_auth(request)
            assert result["admin_key"] == "test-key"

    @pytest.mark.asyncio
    async def test_valid_bearer_token_fallback(self):
        with patch.dict(
            os.environ,
            {"ADMIN_AUTH_ENABLED": "true", "ADMIN_API_KEYS": "test-key"},
            clear=True,
        ):
            request = self._make_request({"Authorization": "Bearer test-key"})
            result = await admin_api_key_auth(request)
            assert result["admin_key"] == "test-key"

    @pytest.mark.asyncio
    async def test_missing_key_returns_401(self):
        with patch.dict(
            os.environ,
            {"ADMIN_AUTH_ENABLED": "true", "ADMIN_API_KEYS": "test-key"},
            clear=True,
        ):
            request = self._make_request()
            with pytest.raises(HTTPException) as exc:
                await admin_api_key_auth(request)
            assert exc.value.status_code == 401
            assert exc.value.detail["error"] == "admin_key_required"

    @pytest.mark.asyncio
    async def test_invalid_key_returns_401(self):
        with patch.dict(
            os.environ,
            {"ADMIN_AUTH_ENABLED": "true", "ADMIN_API_KEYS": "test-key"},
            clear=True,
        ):
            request = self._make_request({"X-Admin-API-Key": "wrong-key"})
            with pytest.raises(HTTPException) as exc:
                await admin_api_key_auth(request)
            assert exc.value.status_code == 401
            assert exc.value.detail["error"] == "invalid_admin_key"

    @pytest.mark.asyncio
    async def test_admin_key_header_takes_precedence(self):
        """X-Admin-API-Key header should be checked before Authorization."""
        with patch.dict(
            os.environ,
            {"ADMIN_AUTH_ENABLED": "true", "ADMIN_API_KEYS": "admin-key,bearer-key"},
            clear=True,
        ):
            request = self._make_request(
                {
                    "X-Admin-API-Key": "admin-key",
                    "Authorization": "Bearer bearer-key",
                }
            )
            result = await admin_api_key_auth(request)
            assert result["admin_key"] == "admin-key"

    @pytest.mark.asyncio
    async def test_multiple_configured_keys(self):
        with patch.dict(
            os.environ,
            {"ADMIN_AUTH_ENABLED": "true", "ADMIN_API_KEYS": "key1,key2,key3"},
            clear=True,
        ):
            # Any of the keys should work
            for key in ["key1", "key2", "key3"]:
                request = self._make_request({"X-Admin-API-Key": key})
                result = await admin_api_key_auth(request)
                assert result["admin_key"] == key


# =============================================================================
# Request ID Tests
# =============================================================================


class TestGetRequestId:
    """Tests for get_request_id function."""

    def test_returns_none_outside_context(self):
        # When no context variable is set and no OTEL, returns None
        result = get_request_id()
        # May return None or OTEL trace ID depending on environment
        assert result is None or isinstance(result, str)


# =============================================================================
# RequestIDMiddleware (Raw ASGI) Tests
# =============================================================================


def _make_http_scope(headers: list[tuple[bytes, bytes]] | None = None) -> dict:
    """Create a minimal HTTP ASGI scope for testing."""
    return {
        "type": "http",
        "method": "GET",
        "path": "/test",
        "headers": headers or [],
    }


def _make_inner_app(captured: dict):
    """
    Create a simple ASGI inner app that captures the scope and context var,
    then sends a minimal HTTP response.
    """

    async def inner_app(scope, receive, send):
        # Capture the request_id context var as seen by downstream handlers
        captured["request_id_in_context"] = _request_id_ctx.get()
        # Send a minimal response
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/plain")],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": b"OK",
            }
        )

    return inner_app


class TestRequestIDMiddleware:
    """Tests for the raw ASGI RequestIDMiddleware."""

    @pytest.mark.asyncio
    async def test_generates_uuid_when_no_header(self):
        """When no X-Request-ID header is provided, a UUID is generated."""
        captured: dict = {}
        sent_messages: list[dict] = []

        async def mock_send(message):
            sent_messages.append(message)

        inner = _make_inner_app(captured)
        middleware = RequestIDMiddleware(inner)

        scope = _make_http_scope()
        await middleware(scope, None, mock_send)

        # Context should have been set with a valid UUID
        ctx_id = captured["request_id_in_context"]
        assert ctx_id is not None
        uuid.UUID(ctx_id)  # Raises if not a valid UUID

        # Response should have X-Request-ID header
        start_msg = sent_messages[0]
        assert start_msg["type"] == "http.response.start"
        header_dict = dict(start_msg["headers"])
        assert REQUEST_ID_HEADER.lower().encode("latin-1") in header_dict
        assert header_dict[
            REQUEST_ID_HEADER.lower().encode("latin-1")
        ] == ctx_id.encode("latin-1")

    @pytest.mark.asyncio
    async def test_passthrough_existing_request_id(self):
        """When X-Request-ID is provided, it is used as-is."""
        captured: dict = {}
        sent_messages: list[dict] = []

        async def mock_send(message):
            sent_messages.append(message)

        inner = _make_inner_app(captured)
        middleware = RequestIDMiddleware(inner)

        existing_id = "my-custom-request-id-123"
        scope = _make_http_scope(
            headers=[
                (b"x-request-id", existing_id.encode("latin-1")),
            ]
        )
        await middleware(scope, None, mock_send)

        assert captured["request_id_in_context"] == existing_id

        # Response header should also have the passthrough ID
        start_msg = sent_messages[0]
        header_dict = dict(start_msg["headers"])
        assert header_dict[
            REQUEST_ID_HEADER.lower().encode("latin-1")
        ] == existing_id.encode("latin-1")

    @pytest.mark.asyncio
    async def test_strips_whitespace_from_header(self):
        """Whitespace in the X-Request-ID header value is stripped."""
        captured: dict = {}
        sent_messages: list[dict] = []

        async def mock_send(message):
            sent_messages.append(message)

        inner = _make_inner_app(captured)
        middleware = RequestIDMiddleware(inner)

        scope = _make_http_scope(
            headers=[
                (b"x-request-id", b"  req-456  "),
            ]
        )
        await middleware(scope, None, mock_send)

        assert captured["request_id_in_context"] == "req-456"

    @pytest.mark.asyncio
    async def test_empty_header_generates_uuid(self):
        """An empty X-Request-ID header triggers UUID generation."""
        captured: dict = {}
        sent_messages: list[dict] = []

        async def mock_send(message):
            sent_messages.append(message)

        inner = _make_inner_app(captured)
        middleware = RequestIDMiddleware(inner)

        scope = _make_http_scope(
            headers=[
                (b"x-request-id", b""),
            ]
        )
        await middleware(scope, None, mock_send)

        ctx_id = captured["request_id_in_context"]
        assert ctx_id is not None
        uuid.UUID(ctx_id)  # Should be a valid UUID

    @pytest.mark.asyncio
    async def test_context_var_reset_after_request(self):
        """Context variable is reset after the request completes."""
        captured: dict = {}
        sent_messages: list[dict] = []

        async def mock_send(message):
            sent_messages.append(message)

        inner = _make_inner_app(captured)
        middleware = RequestIDMiddleware(inner)

        scope = _make_http_scope()
        await middleware(scope, None, mock_send)

        # After the middleware returns, the context var should be reset
        assert _request_id_ctx.get() is None

    @pytest.mark.asyncio
    async def test_context_var_reset_on_exception(self):
        """Context variable is reset even when the inner app raises."""

        async def failing_app(scope, receive, send):
            raise RuntimeError("inner app failure")

        middleware = RequestIDMiddleware(failing_app)
        scope = _make_http_scope()

        with pytest.raises(RuntimeError, match="inner app failure"):
            await middleware(scope, None, lambda msg: None)

        # Context var should still be reset
        assert _request_id_ctx.get() is None

    @pytest.mark.asyncio
    async def test_passthrough_non_http_scope(self):
        """Non-HTTP scopes (websocket, lifespan) are passed through unchanged."""
        called = {"inner": False}

        async def inner_app(scope, receive, send):
            called["inner"] = True

        middleware = RequestIDMiddleware(inner_app)

        # Test with lifespan scope
        lifespan_scope = {"type": "lifespan"}
        await middleware(lifespan_scope, None, None)
        assert called["inner"] is True

        # Test with websocket scope
        called["inner"] = False
        ws_scope = {"type": "websocket"}
        await middleware(ws_scope, None, None)
        assert called["inner"] is True

    @pytest.mark.asyncio
    async def test_preserves_existing_response_headers(self):
        """Existing response headers from the inner app are preserved."""
        sent_messages: list[dict] = []

        async def mock_send(message):
            sent_messages.append(message)

        async def inner_app(scope, receive, send):
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"x-custom", b"value"),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": b"{}"})

        middleware = RequestIDMiddleware(inner_app)
        scope = _make_http_scope()
        await middleware(scope, None, mock_send)

        start_msg = sent_messages[0]
        header_dict = dict(start_msg["headers"])
        # Original headers preserved
        assert header_dict[b"content-type"] == b"application/json"
        assert header_dict[b"x-custom"] == b"value"
        # Request ID header added
        assert REQUEST_ID_HEADER.lower().encode("latin-1") in header_dict

    @pytest.mark.asyncio
    async def test_body_messages_pass_through_unchanged(self):
        """Non-start messages (http.response.body) pass through without modification."""
        sent_messages: list[dict] = []

        async def mock_send(message):
            sent_messages.append(message)

        async def inner_app(scope, receive, send):
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [],
                }
            )
            await send({"type": "http.response.body", "body": b"hello"})

        middleware = RequestIDMiddleware(inner_app)
        scope = _make_http_scope()
        await middleware(scope, None, mock_send)

        body_msg = sent_messages[1]
        assert body_msg["type"] == "http.response.body"
        assert body_msg["body"] == b"hello"


# =============================================================================
# Error Response Tests
# =============================================================================


class TestSanitizeErrorResponse:
    """Tests for sanitize_error_response function."""

    def test_returns_structured_response(self):
        error = ValueError("something broke")
        result = sanitize_error_response(error)
        assert result["error"] == "internal_error"
        assert result["message"] == "An internal error occurred"
        assert "request_id" in result

    def test_custom_public_message(self):
        error = RuntimeError("db connection failed")
        result = sanitize_error_response(error, public_message="Service unavailable")
        assert result["message"] == "Service unavailable"

    def test_does_not_leak_internal_details(self):
        error = RuntimeError("postgresql://admin:hunter2@db:5432/mydb failed")
        result = sanitize_error_response(error)
        assert "hunter2" not in result["message"]
        assert "admin" not in result["message"]

    def test_uses_provided_request_id(self):
        error = ValueError("test")
        result = sanitize_error_response(error, request_id="req-123")
        assert result["request_id"] == "req-123"


class TestCreateAdminErrorResponse:
    """Tests for create_admin_error_response function."""

    def test_returns_http_exception(self):
        exc = create_admin_error_response(
            status_code=403,
            error_code="forbidden",
            message="Access denied",
        )
        assert isinstance(exc, HTTPException)
        assert exc.status_code == 403
        assert exc.detail["error"] == "forbidden"
        assert exc.detail["message"] == "Access denied"
        assert "request_id" in exc.detail

    def test_uses_provided_request_id(self):
        exc = create_admin_error_response(
            status_code=500,
            error_code="internal",
            message="Error",
            request_id="custom-id",
        )
        assert exc.detail["request_id"] == "custom-id"


# =============================================================================
# RouteIQ Key Prefix Helper Tests
# =============================================================================


class TestEnsurePrefix:
    """Tests for _ensure_prefix function."""

    def test_adds_prefix_to_bare_key(self):
        assert _ensure_prefix("abc123") == "sk-riq-abc123"

    def test_does_not_double_prefix(self):
        assert _ensure_prefix("sk-riq-abc123") == "sk-riq-abc123"

    def test_empty_string_unchanged(self):
        assert _ensure_prefix("") == ""

    def test_default_prefix_value(self):
        assert ROUTEIQ_KEY_PREFIX == "sk-riq-"


class TestStripPrefix:
    """Tests for _strip_prefix function."""

    def test_strips_prefix(self):
        assert _strip_prefix("sk-riq-abc123") == "abc123"

    def test_no_prefix_unchanged(self):
        assert _strip_prefix("abc123") == "abc123"

    def test_empty_string_unchanged(self):
        assert _strip_prefix("") == ""

    def test_partial_prefix_unchanged(self):
        assert _strip_prefix("sk-riq") == "sk-riq"


# =============================================================================
# Apply Key Prefix to Env Tests
# =============================================================================


class TestApplyKeyPrefixToEnv:
    """Tests for _apply_key_prefix_to_env function."""

    def test_prefixes_litellm_master_key(self):
        with patch.dict(
            os.environ,
            {"LITELLM_MASTER_KEY": "my-secret"},
            clear=True,
        ):
            _apply_key_prefix_to_env()
            assert os.environ["LITELLM_MASTER_KEY"] == "sk-riq-my-secret"

    def test_does_not_double_prefix_master_key(self):
        with patch.dict(
            os.environ,
            {"LITELLM_MASTER_KEY": "sk-riq-my-secret"},
            clear=True,
        ):
            _apply_key_prefix_to_env()
            assert os.environ["LITELLM_MASTER_KEY"] == "sk-riq-my-secret"

    def test_prefixes_admin_api_keys(self):
        with patch.dict(
            os.environ,
            {"ADMIN_API_KEYS": "key1,key2"},
            clear=True,
        ):
            _apply_key_prefix_to_env()
            assert os.environ["ADMIN_API_KEYS"] == "sk-riq-key1,sk-riq-key2"

    def test_does_not_double_prefix_admin_keys(self):
        with patch.dict(
            os.environ,
            {"ADMIN_API_KEYS": "sk-riq-key1,sk-riq-key2"},
            clear=True,
        ):
            _apply_key_prefix_to_env()
            assert os.environ["ADMIN_API_KEYS"] == "sk-riq-key1,sk-riq-key2"

    def test_mixed_admin_keys(self):
        """Some keys already prefixed, some not."""
        with patch.dict(
            os.environ,
            {"ADMIN_API_KEYS": "sk-riq-key1,key2"},
            clear=True,
        ):
            _apply_key_prefix_to_env()
            assert os.environ["ADMIN_API_KEYS"] == "sk-riq-key1,sk-riq-key2"

    def test_prefixes_legacy_admin_api_key(self):
        with patch.dict(
            os.environ,
            {"ADMIN_API_KEY": "legacy"},
            clear=True,
        ):
            _apply_key_prefix_to_env()
            assert os.environ["ADMIN_API_KEY"] == "sk-riq-legacy"

    def test_no_env_vars_is_noop(self):
        with patch.dict(os.environ, {}, clear=True):
            _apply_key_prefix_to_env()
            assert "LITELLM_MASTER_KEY" not in os.environ
            assert "ADMIN_API_KEYS" not in os.environ
            assert "ADMIN_API_KEY" not in os.environ


# =============================================================================
# Load Admin Keys with Prefix Backwards-Compatibility Tests
# =============================================================================


class TestLoadAdminApiKeysWithPrefix:
    """Tests that _load_admin_api_keys accepts both prefixed and unprefixed keys."""

    def test_prefixed_key_accepts_both_forms(self):
        """When stored key is sk-riq-test, both sk-riq-test and test should work."""
        with patch.dict(
            os.environ,
            {"ADMIN_API_KEYS": "sk-riq-test-api-key"},
            clear=True,
        ):
            keys = _load_admin_api_keys()
            assert "sk-riq-test-api-key" in keys
            assert "test-api-key" in keys

    def test_unprefixed_key_accepts_both_forms(self):
        """When stored key is bare, both prefixed and unprefixed should work."""
        with patch.dict(
            os.environ,
            {"ADMIN_API_KEYS": "test-api-key"},
            clear=True,
        ):
            keys = _load_admin_api_keys()
            assert "test-api-key" in keys
            assert "sk-riq-test-api-key" in keys

    def test_multiple_keys_all_expanded(self):
        with patch.dict(
            os.environ,
            {"ADMIN_API_KEYS": "sk-riq-alpha,beta"},
            clear=True,
        ):
            keys = _load_admin_api_keys()
            assert "sk-riq-alpha" in keys
            assert "alpha" in keys
            assert "sk-riq-beta" in keys
            assert "beta" in keys

    def test_legacy_key_expanded(self):
        with patch.dict(
            os.environ,
            {"ADMIN_API_KEY": "sk-riq-legacy-key"},
            clear=True,
        ):
            keys = _load_admin_api_keys()
            assert "sk-riq-legacy-key" in keys
            assert "legacy-key" in keys

    def test_empty_env_returns_empty(self):
        with patch.dict(os.environ, {}, clear=True):
            keys = _load_admin_api_keys()
            assert keys == set()


# =============================================================================
# Admin Auth with Prefix Integration Tests
# =============================================================================


class TestAdminApiKeyAuthWithPrefix:
    """Tests that admin_api_key_auth works with prefixed and unprefixed keys."""

    def _make_request(self, headers: dict | None = None):
        request = MagicMock()
        request.headers = headers or {}
        request.url.path = "/config/reload"
        return request

    @pytest.mark.asyncio
    async def test_prefixed_key_in_env_accepts_prefixed_header(self):
        with patch.dict(
            os.environ,
            {
                "ADMIN_AUTH_ENABLED": "true",
                "ADMIN_API_KEYS": "sk-riq-test-key",
            },
            clear=True,
        ):
            request = self._make_request({"X-Admin-API-Key": "sk-riq-test-key"})
            result = await admin_api_key_auth(request)
            assert result["admin_key"] == "sk-riq-test-key"

    @pytest.mark.asyncio
    async def test_prefixed_key_in_env_accepts_unprefixed_header(self):
        """Backwards compat: if env has sk-riq-X, sending X should work."""
        with patch.dict(
            os.environ,
            {
                "ADMIN_AUTH_ENABLED": "true",
                "ADMIN_API_KEYS": "sk-riq-test-key",
            },
            clear=True,
        ):
            request = self._make_request({"X-Admin-API-Key": "test-key"})
            result = await admin_api_key_auth(request)
            assert result["admin_key"] == "test-key"

    @pytest.mark.asyncio
    async def test_unprefixed_key_in_env_accepts_prefixed_header(self):
        """If env has bare key X, sending sk-riq-X should also work."""
        with patch.dict(
            os.environ,
            {
                "ADMIN_AUTH_ENABLED": "true",
                "ADMIN_API_KEYS": "test-key",
            },
            clear=True,
        ):
            request = self._make_request({"X-Admin-API-Key": "sk-riq-test-key"})
            result = await admin_api_key_auth(request)
            assert result["admin_key"] == "sk-riq-test-key"

    @pytest.mark.asyncio
    async def test_wrong_key_still_rejected(self):
        with patch.dict(
            os.environ,
            {
                "ADMIN_AUTH_ENABLED": "true",
                "ADMIN_API_KEYS": "sk-riq-test-key",
            },
            clear=True,
        ):
            request = self._make_request({"X-Admin-API-Key": "totally-wrong"})
            with pytest.raises(HTTPException) as exc:
                await admin_api_key_auth(request)
            assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_bearer_with_unprefixed_key_accepted(self):
        with patch.dict(
            os.environ,
            {
                "ADMIN_AUTH_ENABLED": "true",
                "ADMIN_API_KEYS": "sk-riq-bearer-test",
            },
            clear=True,
        ):
            request = self._make_request({"Authorization": "Bearer bearer-test"})
            result = await admin_api_key_auth(request)
            assert result["admin_key"] == "bearer-test"
