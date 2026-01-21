import asyncio
import json
import os
from typing import Any, AsyncIterator
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()


def _normalize_parts(message: dict[str, Any]) -> list[dict[str, Any]]:
    parts = message.get("parts") or []
    normalized: list[dict[str, Any]] = []
    if isinstance(parts, list):
        for part in parts:
            if isinstance(part, dict):
                kind = part.get("kind") or part.get("type") or "text"
                text = part.get("text", "")
                normalized.append({"kind": kind, "text": text})
    return normalized


def _extract_text(message: dict[str, Any]) -> str:
    parts = _normalize_parts(message)
    if not parts:
        return ""
    return " ".join(str(part.get("text", "")) for part in parts).strip()


def _error_response(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def _public_base_url(request: Request) -> str:
    env_url = os.getenv("A2A_STUB_PUBLIC_URL", "").strip()
    if env_url:
        return env_url.rstrip("/")
    return str(request.base_url).rstrip("/")


def _agent_card_payload(base_url: str) -> dict[str, Any]:
    name = os.getenv("A2A_STUB_NAME", "Stub Agent")
    description = os.getenv("A2A_STUB_DESCRIPTION", "Local A2A stub agent")
    version = os.getenv("A2A_STUB_VERSION", "0.1.0")
    return {
        "protocolVersion": "1.0",
        "name": name,
        "description": description,
        "url": f"{base_url}/a2a",
        "version": version,
        "capabilities": {"streaming": True},
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
        "skills": [
            {
                "id": "echo",
                "name": "Echo",
                "description": "Echo user input",
                "tags": ["echo"],
                "examples": ["hello"],
            }
        ],
    }


@app.get("/.well-known/agent-card.json")
async def get_agent_card_root(request: Request):
    base_url = _public_base_url(request)
    return JSONResponse(_agent_card_payload(base_url))


@app.get("/a2a/.well-known/agent-card.json")
async def get_agent_card_a2a(request: Request):
    base_url = _public_base_url(request)
    return JSONResponse(_agent_card_payload(base_url))


@app.post("/a2a")
async def handle_a2a(request: Request):
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(
            _error_response(None, -32700, "Parse error: Invalid JSON"), status_code=400
        )

    request_id = payload.get("id")
    if payload.get("jsonrpc") != "2.0":
        return JSONResponse(
            _error_response(
                request_id, -32600, "Invalid Request: jsonrpc must be '2.0'"
            ),
            status_code=400,
        )

    method = payload.get("method")
    if not method:
        return JSONResponse(
            _error_response(request_id, -32600, "Invalid Request: method is required"),
            status_code=400,
        )

    params = payload.get("params") or {}
    message = params.get("message") or {}
    message_parts = _normalize_parts(message)
    if not message_parts:
        message_parts = [{"kind": "text", "text": ""}]
    input_message_id = (
        message.get("messageId") or message.get("message_id") or uuid4().hex
    )
    input_role = message.get("role") or "user"
    user_text = _extract_text(message)

    if method == "message/send":
        response_text = f"echo: {user_text}".strip()
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "role": "agent",
                    "parts": [
                        {
                            "kind": "text",
                            "text": response_text,
                        }
                    ],
                    "messageId": uuid4().hex,
                },
            }
        )

    if method == "message/stream":

        async def _stream() -> AsyncIterator[str]:
            context_id = uuid4().hex
            task_id = uuid4().hex
            response_text = f"echo: {user_text}".strip()
            chunks = [
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "contextId": context_id,
                        "history": [
                            {
                                "contextId": context_id,
                                "kind": "message",
                                "messageId": input_message_id,
                                "parts": message_parts,
                                "role": input_role,
                                "taskId": task_id,
                            }
                        ],
                        "id": task_id,
                        "kind": "task",
                        "status": {"state": "submitted"},
                    },
                },
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "contextId": context_id,
                        "final": False,
                        "kind": "status-update",
                        "status": {"state": "working"},
                        "taskId": task_id,
                    },
                },
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "contextId": context_id,
                        "kind": "artifact-update",
                        "taskId": task_id,
                        "artifact": {
                            "artifactId": uuid4().hex,
                            "name": "response",
                            "parts": [{"kind": "text", "text": response_text}],
                        },
                    },
                },
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "contextId": context_id,
                        "final": True,
                        "kind": "status-update",
                        "status": {"state": "completed"},
                        "taskId": task_id,
                    },
                },
            ]
            for chunk in chunks:
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.05)

        return StreamingResponse(_stream(), media_type="text/event-stream")

    return JSONResponse(
        _error_response(request_id, -32601, f"Method '{method}' not found"),
        status_code=400,
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("A2A_STUB_PORT", "9000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
