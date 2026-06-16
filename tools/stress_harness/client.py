"""Async OpenAI-compatible client for the RouteIQ gateway (RouteIQ-b245).

Fires ``POST {base_url}/v1/chat/completions`` and captures, per request:

  * the response-body ``id`` (== request_id, the join key for the optional
    ``routing_decision`` log enrichment),
  * the response-body ``model`` (the CONCRETE backend model RouteIQ routed the
    ``model: "auto"`` request to),
  * ``usage.{prompt,completion,total}_tokens``,
  * an optional echoed request-id response header (correlation only).

LOAD-BEARING request detail: the body must send ``"model": "auto"`` so RouteIQ's
ML router picks the backend. A pinned model name takes the specified-model path
and bypasses routing. ``model`` is configurable but defaults to ``"auto"``.

Auth is RouteIQ's data-plane key via ``Authorization: Bearer <token>`` (the
user-tier auth the data-plane ``/v1/chat/completions`` endpoint expects). An
optional ``x-user-id`` header is sent when ``user_id`` is set (per-conversation
synthetic users for the personalized-routing seam) but is never required.

The client takes an injectable ``httpx.AsyncClient`` so tests drive the whole
request/response-parsing path through an ``httpx.MockTransport`` with NO live
endpoint, no sockets, and no credentials.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from .models import RequestRecord

DEFAULT_REQUEST_ID_HEADER = "x-request-id"
DEFAULT_TIMEOUT_S = 120.0


class RouterClient:
    """Thin async wrapper over ``/v1/chat/completions`` that returns fully
    populated ``RequestRecord``s.

    The client never raises on a per-request HTTP error — it records
    ``http_status``/``error`` on the ``RequestRecord`` so a single failure can't
    abort a whole stress run (the analysis layer filters on ``RequestRecord.ok``).
    """

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        model: str = "auto",
        user_id: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
        request_id_header: str = DEFAULT_REQUEST_ID_HEADER,
        http_client: httpx.AsyncClient | None = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._model = model
        self._user_id = user_id
        self._timeout = timeout
        self._request_id_header = request_id_header.lower()
        # Injected client (tests pass one wired to a MockTransport). When None,
        # each run() builds and tears down its own client.
        self._injected = http_client

    # ----- header / body construction -------------------------------------

    def _headers(self, record: RequestRecord | None = None) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if self._token:
            # RouteIQ data-plane auth: Authorization: Bearer <user-tier key>.
            headers["authorization"] = f"Bearer {self._token}"
        # Per-request user id (the record's synthetic user) wins over the
        # client-wide default, so multi-tenant personalized routing can be
        # exercised; both are optional and never required by the gateway.
        user_id = (record.user_id if record else None) or self._user_id
        if user_id:
            headers["x-user-id"] = user_id
        return headers

    def _body(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        # "model":"auto" is load-bearing — keep it the default. ``messages`` is
        # the full running history so a multi-turn request carries prior turns.
        return {
            "model": self._model,
            "messages": messages,
            "stream": False,
        }

    @property
    def endpoint(self) -> str:
        return f"{self._base_url}/v1/chat/completions"

    # ----- response parsing -------------------------------------------------

    def _parse_response(
        self, record: RequestRecord, resp: httpx.Response
    ) -> str | None:
        """Populate ``record`` from a completed HTTP response (mutates in place).

        Reads the request-id echo header regardless of status (correlation), and
        the body JSON only on a parseable body. Returns the assistant message
        content (or None) so a multi-turn driver can thread it into the next
        turn's history.
        """
        record.http_status = resp.status_code
        record.header_request_id = resp.headers.get(self._request_id_header)

        assistant_content: str | None = None
        try:
            data = resp.json()
        except (ValueError, UnicodeDecodeError):
            data = None
        if isinstance(data, dict):
            rid = data.get("id")
            if isinstance(rid, str):
                record.request_id = rid
            model = data.get("model")
            if isinstance(model, str):
                record.body_model = model
            usage = data.get("usage")
            if isinstance(usage, dict):
                record.prompt_tokens = _as_int(usage.get("prompt_tokens"))
                record.completion_tokens = _as_int(usage.get("completion_tokens"))
                record.total_tokens = _as_int(usage.get("total_tokens"))
            assistant_content = _extract_assistant_content(data)

        # Fall back to the header echo for the join key if the body lacked one.
        if record.request_id is None and record.header_request_id:
            record.request_id = record.header_request_id
        return assistant_content

    # ----- single request ---------------------------------------------------

    async def _send_one(
        self,
        client: httpx.AsyncClient,
        record: RequestRecord,
        messages: list[dict[str, str]] | None = None,
    ) -> tuple[RequestRecord, str | None]:
        """Fire one chat-completions call for ``record``.

        ``messages`` is the full running history (prior turns + this turn). When
        None — the single-turn path — a fresh one-message history is built.
        Returns ``(record, assistant_content)``. On a network error the content
        is None and the error is recorded, never raised.
        """
        if messages is None:
            messages = [{"role": "user", "content": record.prompt}]
        record.sent_ts = time.time()
        start = time.perf_counter()
        try:
            resp = await client.post(
                self.endpoint,
                json=self._body(messages),
                headers=self._headers(record),
                timeout=self._timeout,
            )
        except asyncio.CancelledError:
            # cooperative cancellation must propagate, never be swallowed.
            raise
        except Exception as exc:  # noqa: BLE001
            # ANY per-request failure (httpx network/timeout AND rarer escapes
            # like httpx.InvalidURL from a malformed base_url, or a busted
            # transport raising a plain Exception) is recorded on the record,
            # never propagated — otherwise asyncio.gather aborts the whole batch
            # and discards every sibling's results.
            record.client_latency_ms = (time.perf_counter() - start) * 1000.0
            record.error = f"{type(exc).__name__}: {exc}"
            return record, None
        record.client_latency_ms = (time.perf_counter() - start) * 1000.0
        assistant_content = self._parse_response(record, resp)
        return record, assistant_content

    # ----- batch driver ------------------------------------------------------

    async def run(
        self,
        records: list[RequestRecord],
        *,
        concurrency: int = 4,
    ) -> list[RequestRecord]:
        """Fire all ``records`` with bounded ``concurrency`` and return them
        populated, in the SAME order they were supplied (input order is the time
        axis the analysis uses).
        """
        if not records:
            return []
        sem = asyncio.Semaphore(max(1, concurrency))

        owns_client = self._injected is None
        client = self._injected or httpx.AsyncClient()
        try:

            async def _guarded(rec: RequestRecord) -> RequestRecord:
                async with sem:
                    result, _ = await self._send_one(client, rec)
                    return result

            return await asyncio.gather(*(_guarded(r) for r in records))
        finally:
            if owns_client:
                await client.aclose()

    # ----- multi-turn conversation driver -----------------------------------

    # Cap how much assistant text we feed back into the next turn's history. The
    # routing decision keys off the latest user turn, so a truncated assistant
    # turn is plenty of context and keeps request bodies bounded over a long
    # conversation.
    _MAX_ASSISTANT_CHARS = 4000

    async def _run_one_conversation(
        self, client: httpx.AsyncClient, turns: list[RequestRecord]
    ) -> list[RequestRecord]:
        """Drive ONE conversation SEQUENTIALLY: each turn carries the accumulated
        history (prior user prompts + assistant replies + this turn's prompt).
        Turn N+1 needs turn N's reply, so turns within a conversation are NOT
        parallelised.

        If a turn errors or yields no assistant content, an empty-string
        assistant turn is appended so the history shape stays valid and the
        conversation still proceeds (the failed turn keeps its error; the
        analysis filters on ``ok``).
        """
        history: list[dict[str, str]] = []
        completed: list[RequestRecord] = []
        for rec in turns:
            history.append({"role": "user", "content": rec.prompt})
            _, assistant_content = await self._send_one(client, rec, list(history))
            completed.append(rec)
            reply = (assistant_content or "")[: self._MAX_ASSISTANT_CHARS]
            history.append({"role": "assistant", "content": reply})
        return completed

    async def run_conversations(
        self,
        conversations: list[list[RequestRecord]],
        *,
        concurrency: int = 4,
    ) -> list[list[RequestRecord]]:
        """Fire ``conversations`` with bounded CONVERSATION-level ``concurrency``.

        Each conversation runs its turns sequentially (history threading), but up
        to ``concurrency`` conversations run at once. Returns conversations in
        input order, each a list of populated turn records (in turn order). A
        single failed turn never aborts the batch.
        """
        if not conversations:
            return []
        sem = asyncio.Semaphore(max(1, concurrency))

        owns_client = self._injected is None
        client = self._injected or httpx.AsyncClient()
        try:

            async def _guarded(turns: list[RequestRecord]) -> list[RequestRecord]:
                async with sem:
                    return await self._run_one_conversation(client, turns)

            return await asyncio.gather(*(_guarded(c) for c in conversations))
        finally:
            if owns_client:
                await client.aclose()


def _as_int(value: Any) -> int | None:
    """Coerce a usage field to int, tolerating None / non-numeric junk."""
    if isinstance(value, bool):  # bool is an int subclass — reject explicitly
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _extract_assistant_content(data: dict[str, Any]) -> str | None:
    """Pull the assistant message text from an OpenAI-compat chat completion.

    Tolerates the standard ``choices[0].message.content`` shape, a list-of-blocks
    content, and a missing / malformed body (returns None).
    """
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    message = first.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ]
        return "".join(parts) if parts else None
    return None
