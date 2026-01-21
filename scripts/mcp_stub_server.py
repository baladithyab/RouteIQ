import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

STUB_NAME = os.getenv("MCP_STUB_NAME", "Stub MCP Server")
STUB_VERSION = os.getenv("MCP_STUB_VERSION", "0.1.0")

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "stub.echo",
        "description": "Echo arguments back to the caller",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "stub.sum",
        "description": "Sum a list of numbers",
        "input_schema": {
            "type": "object",
            "properties": {"values": {"type": "array", "items": {"type": "number"}}},
            "required": ["values"],
        },
    },
]

RESOURCE_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "stub://resource/demo",
        "description": "Demo resource exposed by the stub",
    }
]


class ToolCall(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = {}


@app.get("/mcp")
async def get_stub_metadata() -> dict[str, Any]:
    return {
        "name": STUB_NAME,
        "version": STUB_VERSION,
        "tools": [tool["name"] for tool in TOOL_DEFINITIONS],
        "resources": [resource["name"] for resource in RESOURCE_DEFINITIONS],
    }


@app.get("/mcp/health")
async def get_stub_health() -> dict[str, Any]:
    return {"status": "ok", "name": STUB_NAME, "version": STUB_VERSION}


@app.get("/mcp/tools")
async def list_stub_tools() -> dict[str, Any]:
    return {"tools": TOOL_DEFINITIONS}


@app.get("/mcp/resources")
async def list_stub_resources() -> dict[str, Any]:
    return {"resources": RESOURCE_DEFINITIONS}


@app.post("/mcp/tools/call")
async def call_stub_tool(request: ToolCall) -> dict[str, Any]:
    if request.tool_name == "stub.echo":
        return {
            "status": "success",
            "tool_name": request.tool_name,
            "result": {"echo": request.arguments},
        }

    if request.tool_name == "stub.sum":
        values = request.arguments.get("values")
        if not isinstance(values, list):
            raise HTTPException(status_code=400, detail="values must be a list")
        total = 0.0
        for value in values:
            try:
                total += float(value)
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="values must be numbers")
        return {
            "status": "success",
            "tool_name": request.tool_name,
            "result": {"sum": total},
        }

    if request.tool_name == "stub.fail":
        raise HTTPException(status_code=500, detail="stub.fail forced error")

    raise HTTPException(status_code=404, detail=f"Unknown tool: {request.tool_name}")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("MCP_STUB_PORT", "9100"))
    uvicorn.run(app, host="0.0.0.0", port=port)
