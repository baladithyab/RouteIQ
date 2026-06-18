"""
MCP JSON-RPC Client - Real Model Context Protocol Transport
===========================================================

A real MCP client speaking JSON-RPC 2.0 over HTTP, with optional SSE
(Server-Sent Events) response framing per the MCP "Streamable HTTP" transport.

This replaces the legacy custom REST stub (``POST {url}/mcp/tools/call`` with a
``{tool_name, arguments}`` body) used by :mod:`mcp_gateway`. Real MCP servers do
not understand that shape — they expect JSON-RPC 2.0 envelopes:

- ``initialize``      — handshake; negotiates protocol version + capabilities
- ``tools/list``      — enumerate available tools (with input schemas)
- ``tools/call``      — invoke a tool by ``name`` with ``arguments``

Per the MCP spec (https://modelcontextprotocol.io/specification), a Streamable
HTTP endpoint accepts ``POST`` with both ``application/json`` and
``text/event-stream`` in ``Accept``. The server may answer with either:

- ``Content-Type: application/json``       — a single JSON-RPC response object
- ``Content-Type: text/event-stream``      — one or more SSE ``data:`` frames,
  each carrying a JSON-RPC message; the response is the final ``result``/``error``

Security:
- The caller (mcp_gateway) is responsible for SSRF-validating the URL. This
  module makes no DNS/connection decisions of its own beyond the httpx client.

This module has NO import side effects and creates no singletons.

See: https://modelcontextprotocol.io/specification
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from typing import Any

import httpx
from litellm._logging import verbose_proxy_logger

from .http_client_pool import get_client_for_request

# MCP protocol version this client advertises during ``initialize``.
# Servers echo back their own supported version; we accept whatever they send.
MCP_PROTOCOL_VERSION = "2025-06-18"

# Default identity advertised in the initialize handshake.
_CLIENT_INFO = {"name": "RouteIQ-MCP-Gateway", "version": "1.0.0"}

# Accept both transports: a server may reply with plain JSON or an SSE stream.
_DEFAULT_ACCEPT = "application/json, text/event-stream"


class MCPProtocolError(Exception):
    """Raised when an MCP server returns a JSON-RPC error or a malformed reply."""

    def __init__(self, message: str, code: int | None = None) -> None:
        self.code = code
        super().__init__(message)


@dataclass
class MCPToolCallResult:
    """Parsed result of a ``tools/call`` invocation.

    ``content`` is the MCP ``content`` list (text/image/resource blocks).
    ``is_error`` mirrors the MCP ``isError`` flag (tool-level, not transport-level).
    ``raw`` is the full JSON-RPC ``result`` object for callers that need more.
    """

    content: list[dict[str, Any]]
    is_error: bool = False
    raw: dict[str, Any] | None = None


def _normalize_endpoint(url: str) -> str:
    """Return the JSON-RPC endpoint URL.

    Real MCP Streamable-HTTP servers expose a single endpoint that handles all
    methods (no per-method path). We treat the registered ``url`` as that
    endpoint verbatim. A trailing slash is stripped for consistency, but no
    ``/tools/call``-style suffix is appended (that was the legacy REST stub bug).
    """
    return url.rstrip("/") if url else url


def _build_headers(
    auth_type: str,
    metadata: dict[str, Any] | None,
    session_id: str | None = None,
) -> dict[str, str]:
    """Build request headers including content negotiation + optional auth."""
    headers = {
        "Content-Type": "application/json",
        "Accept": _DEFAULT_ACCEPT,
    }
    meta = metadata or {}
    # Provider-key resolution (RouteIQ-1786): route the upstream MCP server
    # credential through the secrets vault so an operator may store it as an
    # ``aws-secrets://<id>[#key]`` reference. Byte-stable when the vault is
    # disabled (resolve_provider_value returns the value unchanged).
    from .secrets_vault import resolve_provider_value

    if auth_type == "bearer_token" and meta.get("auth_token"):
        token = resolve_provider_value(meta["auth_token"])
        if token:
            headers["Authorization"] = f"Bearer {token}"
    elif auth_type == "api_key" and meta.get("api_key"):
        key = resolve_provider_value(meta["api_key"])
        if key:
            headers["X-API-Key"] = key
    # MCP session continuity: the server hands out an Mcp-Session-Id on
    # initialize; subsequent requests echo it back.
    if session_id:
        headers["Mcp-Session-Id"] = session_id
    return headers


def _parse_sse_payload(body: str) -> dict[str, Any]:
    """Extract the JSON-RPC message from an SSE ``text/event-stream`` body.

    SSE frames are ``data: <json>`` lines separated by blank lines. A single
    logical event may span multiple ``data:`` lines (concatenated with ``\\n``).
    We return the LAST JSON-RPC object that carries an ``id`` (the response to
    our request), tolerating interleaved notifications.
    """
    last_message: dict[str, Any] | None = None
    data_lines: list[str] = []

    def _flush() -> None:
        nonlocal last_message
        if not data_lines:
            return
        raw = "\n".join(data_lines)
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return
        if isinstance(obj, dict) and ("result" in obj or "error" in obj):
            last_message = obj

    for line in body.splitlines():
        if line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())
        elif line.strip() == "":
            _flush()
            data_lines = []
    _flush()

    if last_message is None:
        raise MCPProtocolError("No JSON-RPC response found in SSE stream")
    return last_message


def _parse_response_body(response: httpx.Response) -> dict[str, Any]:
    """Parse a JSON-RPC response from either a JSON or SSE HTTP response."""
    content_type = response.headers.get("content-type", "")
    if "text/event-stream" in content_type:
        return _parse_sse_payload(response.text)
    try:
        obj = response.json()
    except json.JSONDecodeError as exc:
        raise MCPProtocolError(f"Invalid JSON response from MCP server: {exc}") from exc
    if not isinstance(obj, dict):
        raise MCPProtocolError("MCP response was not a JSON object")
    return obj


class MCPJSONRPCClient:
    """A minimal, dependency-light MCP client over JSON-RPC 2.0 + HTTP/SSE.

    One client instance maps to one MCP server URL. The client is stateless
    between calls except for an optional negotiated session id; create one per
    invocation (cheap — it reuses the shared httpx pool).

    Thread-safety: not shared across coroutines; create per-use.
    """

    def __init__(
        self,
        url: str,
        *,
        auth_type: str = "none",
        metadata: dict[str, Any] | None = None,
        connect_timeout: float = 10.0,
        read_timeout: float = 30.0,
    ) -> None:
        self.endpoint = _normalize_endpoint(url)
        self.auth_type = auth_type
        self.metadata = metadata or {}
        self._id_counter = itertools.count(1)
        self._session_id: str | None = None
        self._timeout = httpx.Timeout(
            connect=connect_timeout,
            read=read_timeout,
            write=connect_timeout,
            pool=connect_timeout,
        )

    def _next_id(self) -> int:
        return next(self._id_counter)

    async def _post_rpc(
        self, method: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Send a single JSON-RPC request and return its ``result`` object.

        Raises:
            MCPProtocolError: on a JSON-RPC ``error`` or malformed response.
            httpx.HTTPError: on transport failures (caller maps to MCPToolResult).
        """
        request_id = self._next_id()
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params is not None:
            payload["params"] = params

        headers = _build_headers(self.auth_type, self.metadata, self._session_id)

        async with get_client_for_request(timeout=self._timeout) as client:
            response = await client.post(
                self.endpoint,
                json=payload,
                headers=headers,
                timeout=self._timeout,
            )

            # Capture a server-issued session id for subsequent calls.
            new_session = response.headers.get("mcp-session-id")
            if new_session:
                self._session_id = new_session

            if response.status_code >= 400:
                detail = (
                    response.text[:500]
                    if response.text
                    else f"HTTP {response.status_code}"
                )
                raise MCPProtocolError(
                    f"HTTP {response.status_code} from MCP server: {detail}"
                )

            message = _parse_response_body(response)

        if "error" in message and message["error"] is not None:
            err = message["error"]
            code = err.get("code") if isinstance(err, dict) else None
            msg = (
                err.get("message", "unknown error")
                if isinstance(err, dict)
                else str(err)
            )
            raise MCPProtocolError(f"MCP server error: {msg}", code=code)

        result = message.get("result")
        if not isinstance(result, dict):
            # tools/call results are objects; an absent/null result is a protocol
            # violation for the methods we send.
            raise MCPProtocolError(
                f"MCP response for '{method}' missing 'result' object"
            )
        return result

    async def initialize(self) -> dict[str, Any]:
        """Perform the MCP ``initialize`` handshake.

        Returns the server's reported capabilities + serverInfo. Records the
        negotiated session id (if any) for subsequent calls.
        """
        params = {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": _CLIENT_INFO,
        }
        result = await self._post_rpc("initialize", params)
        verbose_proxy_logger.debug(
            f"MCP: initialized with server "
            f"{result.get('serverInfo', {}).get('name', 'unknown')}"
        )
        return result

    async def list_tools(self) -> list[dict[str, Any]]:
        """Call ``tools/list`` and return the list of tool descriptors."""
        result = await self._post_rpc("tools/list", {})
        tools = result.get("tools", [])
        if not isinstance(tools, list):
            raise MCPProtocolError("tools/list returned a non-list 'tools'")
        return tools

    async def call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> MCPToolCallResult:
        """Call ``tools/call`` with the given tool name + arguments.

        Returns the parsed MCP tool result. ``is_error`` reflects a tool-level
        failure (the JSON-RPC call itself succeeded). Transport / JSON-RPC
        errors raise :class:`MCPProtocolError`.
        """
        params = {"name": name, "arguments": arguments}
        result = await self._post_rpc("tools/call", params)
        content = result.get("content", [])
        if not isinstance(content, list):
            content = [content] if content is not None else []
        return MCPToolCallResult(
            content=content,
            is_error=bool(result.get("isError", False)),
            raw=result,
        )

    async def initialize_and_call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> MCPToolCallResult:
        """Convenience: ``initialize`` (best-effort) then ``tools/call``.

        The handshake is best-effort — some servers accept ``tools/call``
        without it. If initialize fails we still attempt the call so that
        servers that skip the handshake remain reachable.
        """
        try:
            await self.initialize()
        except (MCPProtocolError, httpx.HTTPError) as exc:
            verbose_proxy_logger.debug(
                f"MCP: initialize handshake skipped/failed (continuing): {exc}"
            )
        return await self.call_tool(name, arguments)
