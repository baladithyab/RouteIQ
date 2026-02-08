"""
Admin Authentication and Request ID Middleware for LLMRouter Control-Plane
===========================================================================

This module provides:
1. Admin API key authentication for control-plane operations (MCP, A2A, hot-reload)
2. Request correlation ID middleware for tracing and error responses
3. Sanitized error responses that don't leak internal exception details
4. Secret scrubbing for logged error messages

Usage:
    from litellm_llmrouter.auth import admin_api_key_auth, get_request_id

    # Apply admin auth to control-plane endpoints
    @router.post("/config/reload", dependencies=[Depends(admin_api_key_auth)])
    async def reload_config():
        ...

Configuration:
    Environment variables:
    - ADMIN_API_KEYS: Comma-separated list of admin API keys
    - ADMIN_API_KEY: Single admin API key (legacy, use ADMIN_API_KEYS instead)
    - ADMIN_AUTH_ENABLED: Set to "false" to disable admin auth (NOT recommended)

    When no admin keys are configured and ADMIN_AUTH_ENABLED is not explicitly "false",
    control-plane endpoints will deny all requests (fail-closed).
"""

import logging
import os
import re
import uuid
from contextvars import ContextVar
from typing import Optional

from fastapi import HTTPException, Request
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)

# Context variable to store request ID for the current request
_request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)

# Header names
REQUEST_ID_HEADER = "X-Request-ID"
ADMIN_API_KEY_HEADER = "X-Admin-API-Key"
AUTHORIZATION_HEADER = "Authorization"

# Patterns for secret scrubbing in error logs
# These patterns match common secret formats to prevent them from being logged
SECRET_PATTERNS = [
    # API keys (various formats)
    re.compile(r"(sk[-_])[a-zA-Z0-9]{16,}", re.IGNORECASE),
    re.compile(r"(api[-_]?key[=:])\s*[a-zA-Z0-9\-_]{10,}", re.IGNORECASE),
    # AWS keys
    re.compile(r"(AKIA)[A-Z0-9]{16}"),
    re.compile(r"(aws[-_]?secret)[=:]?\s*[a-zA-Z0-9/+]{40}", re.IGNORECASE),
    # Database connection strings
    re.compile(r"(postgres(?:ql)?|mysql|mongodb)://[^@]+:[^@]+@", re.IGNORECASE),
    # Bearer tokens in headers
    re.compile(r"(Bearer\s+)[a-zA-Z0-9\-_\.]{10,}", re.IGNORECASE),
    # Generic password patterns - require = or : followed by the value
    re.compile(r"(password[=:])\s*[^\s,;\"']+", re.IGNORECASE),
    # Secret patterns - require = or : followed by the value (not just the word 'secret')
    re.compile(r"(secret[=:])\s*[^\s,;\"']+", re.IGNORECASE),
]


def _scrub_secrets(text: str) -> str:
    """
    Scrub obvious secret patterns from log text.

    Args:
        text: The text to scrub

    Returns:
        Text with secrets replaced by [REDACTED]
    """
    if not text:
        return text

    scrubbed = text
    for pattern in SECRET_PATTERNS:
        scrubbed = pattern.sub(r"\1[REDACTED]", scrubbed)
    return scrubbed


def _get_otel_trace_id() -> Optional[str]:
    """
    Extract the current OTEL trace ID if available.

    Returns:
        Hex string of trace ID, or None if not in a traced context
    """
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if span and span.is_recording():
            ctx = span.get_span_context()
            if ctx and ctx.is_valid:
                # Return trace_id in hex format (32 chars)
                return format(ctx.trace_id, "032x")
    except ImportError:
        # OpenTelemetry not installed
        pass
    except Exception:
        # Any other error - fail silently
        pass
    return None


def get_request_id() -> Optional[str]:
    """
    Get the current request's correlation ID.

    Priority order:
    1. Context variable (set by middleware)
    2. OTEL trace ID (if available)
    3. None

    Returns:
        The request ID from the current context, or None if not in a request context.
    """
    # First try context variable (most reliable, set by middleware)
    ctx_id = _request_id_ctx.get()
    if ctx_id:
        return ctx_id

    # Fallback to OTEL trace ID if available
    return _get_otel_trace_id()


def _load_admin_api_keys() -> set[str]:
    """
    Load admin API keys from environment configuration.

    Returns:
        Set of valid admin API keys. Empty set if none configured.

    Configuration sources (in order of precedence):
    1. ADMIN_API_KEYS: Comma-separated list of keys
    2. ADMIN_API_KEY: Single key (legacy fallback)
    """
    keys: set[str] = set()

    # Primary: comma-separated list
    keys_str = os.getenv("ADMIN_API_KEYS", "").strip()
    if keys_str:
        for key in keys_str.split(","):
            key = key.strip()
            if key:
                keys.add(key)

    # Fallback: single key
    single_key = os.getenv("ADMIN_API_KEY", "").strip()
    if single_key:
        keys.add(single_key)

    return keys


def _is_admin_auth_enabled() -> bool:
    """
    Check if admin authentication is enabled.

    Returns:
        True if admin auth is enabled (default), False if explicitly disabled.

    Note:
        Setting ADMIN_AUTH_ENABLED=false is NOT recommended for production.
        When disabled, control-plane endpoints are only protected by user API key auth.
    """
    env_val = os.getenv("ADMIN_AUTH_ENABLED", "true").lower().strip()
    return env_val not in ("false", "0", "no", "off")


def _extract_bearer_token(auth_header: str) -> Optional[str]:
    """Extract token from Bearer authorization header."""
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        return token if token else None
    return None


async def admin_api_key_auth(request: Request) -> dict:
    """
    FastAPI dependency for admin API key authentication.

    Validates that the request contains a valid admin API key.
    Keys can be provided via:
    - X-Admin-API-Key header (preferred)
    - Authorization: Bearer <key> header (fallback, checks against admin keys)

    Returns:
        dict with 'admin_key' on success

    Raises:
        HTTPException 401: Missing or invalid admin key
        HTTPException 403: Admin auth disabled or no admin keys configured (fail-closed)
    """
    request_id = get_request_id() or "unknown"

    # Check if admin auth is enabled
    if not _is_admin_auth_enabled():
        logger.warning(
            "Admin auth disabled via ADMIN_AUTH_ENABLED=false",
            extra={"request_id": request_id},
        )
        # Still allow request through but log warning
        return {"admin_key": "__disabled__"}

    # Load configured admin keys
    admin_keys = _load_admin_api_keys()

    # Fail-closed: if no admin keys configured, deny all requests
    if not admin_keys:
        logger.error(
            "Admin auth configured but no ADMIN_API_KEYS set - denying request",
            extra={"request_id": request_id},
        )
        raise HTTPException(
            status_code=403,
            detail={
                "error": "control_plane_not_configured",
                "message": "Control-plane access denied. Admin API keys not configured.",
                "request_id": request_id,
            },
        )

    # Try X-Admin-API-Key header first
    admin_key = request.headers.get(ADMIN_API_KEY_HEADER, "").strip()

    # Fallback to Authorization header
    if not admin_key:
        auth_header = request.headers.get(AUTHORIZATION_HEADER, "")
        admin_key = _extract_bearer_token(auth_header) or ""

    # Validate the key
    if not admin_key:
        logger.warning(
            "Missing admin API key in request",
            extra={"request_id": request_id, "path": str(request.url.path)},
        )
        raise HTTPException(
            status_code=401,
            detail={
                "error": "admin_key_required",
                "message": "Admin API key required. Provide via X-Admin-API-Key header.",
                "request_id": request_id,
            },
        )

    if admin_key not in admin_keys:
        logger.warning(
            "Invalid admin API key in request",
            extra={"request_id": request_id, "path": str(request.url.path)},
        )
        raise HTTPException(
            status_code=401,
            detail={
                "error": "invalid_admin_key",
                "message": "Invalid admin API key.",
                "request_id": request_id,
            },
        )

    logger.debug(
        "Admin authentication successful",
        extra={"request_id": request_id, "path": str(request.url.path)},
    )
    return {"admin_key": admin_key}


def sanitize_error_response(
    error: Exception,
    request_id: Optional[str] = None,
    public_message: str = "An internal error occurred",
) -> dict:
    """
    Create a sanitized error response that doesn't leak internal details.

    Args:
        error: The exception that occurred
        request_id: Correlation ID for the request (auto-fetched if None)
        public_message: User-facing message to return

    Returns:
        dict suitable for JSON response with 'error', 'message', and 'request_id'
    """
    # Get request ID with OTEL trace ID fallback
    req_id = request_id or get_request_id() or str(uuid.uuid4())

    # Scrub secrets from the error message before logging
    error_msg = _scrub_secrets(str(error))

    # Log the full error server-side with request ID for debugging
    logger.error(
        f"Request error (request_id={req_id}): {type(error).__name__}: {error_msg}",
        extra={"request_id": req_id, "error_type": type(error).__name__},
        exc_info=True,
    )

    return {
        "error": "internal_error",
        "message": public_message,
        "request_id": req_id,
    }


class RequestIDMiddleware:
    """
    Raw ASGI middleware to inject request correlation IDs.

    Uses the raw ASGI pattern (same as BackpressureMiddleware and
    RouterDecisionMiddleware) instead of BaseHTTPMiddleware, which buffers
    the entire response and breaks streaming.

    - Reads X-Request-ID from incoming request headers (for passthrough)
    - Generates a UUID if not provided
    - Sets the request ID in context for access throughout the request lifecycle
    - Injects X-Request-ID into response headers via send wrapper
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """ASGI entry point."""
        # Only process HTTP requests; pass through websocket/lifespan
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Extract X-Request-ID from ASGI scope headers
        request_id = ""
        header_name = REQUEST_ID_HEADER.lower().encode("latin-1")
        for header_key, header_value in scope.get("headers", []):
            if header_key == header_name:
                request_id = header_value.decode("latin-1").strip()
                break

        if not request_id:
            request_id = str(uuid.uuid4())

        # Set in context for access throughout request handling
        token = _request_id_ctx.set(request_id)

        try:
            # Wrap send to inject X-Request-ID into response headers
            request_id_header_pair = (
                REQUEST_ID_HEADER.lower().encode("latin-1"),
                request_id.encode("latin-1"),
            )

            async def send_with_request_id(message: dict) -> None:
                if message["type"] == "http.response.start":
                    headers = list(message.get("headers", []))
                    headers.append(request_id_header_pair)
                    message = {**message, "headers": headers}
                await send(message)

            await self.app(scope, receive, send_with_request_id)
        finally:
            # Reset context
            _request_id_ctx.reset(token)


def create_admin_error_response(
    status_code: int,
    error_code: str,
    message: str,
    request_id: Optional[str] = None,
) -> HTTPException:
    """
    Create an HTTPException with a standardized error response format.

    Args:
        status_code: HTTP status code
        error_code: Machine-readable error code
        message: Human-readable error message
        request_id: Correlation ID (auto-fetched if None)

    Returns:
        HTTPException with structured detail
    """
    req_id = request_id or get_request_id() or "unknown"
    return HTTPException(
        status_code=status_code,
        detail={
            "error": error_code,
            "message": message,
            "request_id": req_id,
        },
    )
