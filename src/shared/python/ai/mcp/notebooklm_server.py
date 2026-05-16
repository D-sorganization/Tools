"""NotebookLM MCP shim.

There is no first-party Google NotebookLM MCP server at the time of writing
(2026-05) — the official NotebookLM API is still gated. This module therefore
implements a small **Python shim** that respects the MCP wire format and can
be invoked as ``python -m src.shared.python.ai.mcp.notebooklm_server``.

The shim advertises two tools and one resource family:

- ``search_notebook(query: str, notebook_id: str) -> {hits: [...]}``
- ``summarize_notebook(notebook_id: str) -> {summary: str}``
- Resource ``notebook://{id}`` (read-only metadata).

Dependency choice:
    We deliberately avoid adding a heavy upstream dependency (e.g.
    ``notebooklm-python``, which is unmaintained as of 2026-05). The shim
    delegates the actual NotebookLM API calls to a pluggable
    ``NotebookBackend`` Protocol. The default backend returns deterministic
    stub data so the surface is testable and useful for local development
    without credentials.

The shim is intentionally small and synchronous on its inner backend;
``main()`` runs the stdin/stdout JSON-RPC loop asynchronously.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any, Protocol

_LOG = logging.getLogger(__name__)

PROTOCOL_VERSION = "2024-11-05"


class NotebookBackend(Protocol):
    """Pluggable backend for the NotebookLM shim."""

    def search(self, notebook_id: str, query: str) -> list[dict[str, Any]]: ...
    def summarize(self, notebook_id: str) -> str: ...
    def metadata(self, notebook_id: str) -> dict[str, Any]: ...


class StubNotebookBackend:
    """Deterministic in-memory backend used when no real API is configured."""

    def search(self, notebook_id: str, query: str) -> list[dict[str, Any]]:
        return [
            {
                "notebook_id": notebook_id,
                "snippet": f"stub hit for {query!r}",
                "score": 1.0,
            }
        ]

    def summarize(self, notebook_id: str) -> str:
        return f"Stub summary for notebook {notebook_id}."

    def metadata(self, notebook_id: str) -> dict[str, Any]:
        return {"id": notebook_id, "title": f"Notebook {notebook_id}"}


_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "search_notebook",
        "description": "Search within a NotebookLM notebook for a query string.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "notebook_id": {"type": "string"},
                "query": {"type": "string"},
            },
            "required": ["notebook_id", "query"],
        },
    },
    {
        "name": "summarize_notebook",
        "description": "Summarize the contents of a NotebookLM notebook.",
        "inputSchema": {
            "type": "object",
            "properties": {"notebook_id": {"type": "string"}},
            "required": ["notebook_id"],
        },
    },
]


def _resource_for(notebook_id: str) -> dict[str, Any]:
    return {
        "uri": f"notebook://{notebook_id}",
        "name": f"Notebook {notebook_id}",
        "mimeType": "application/json",
    }


def handle_request(request: dict[str, Any], backend: NotebookBackend) -> dict[str, Any]:
    """Dispatch a single JSON-RPC request and return the response dict."""
    method = request.get("method")
    params = request.get("params") or {}
    request_id = request.get("id")
    try:
        result = _dispatch(method, params, backend)
        return {"jsonrpc": "2.0", "id": request_id, "result": result}
    except Exception as exc:  # noqa: BLE001 — JSON-RPC error envelope
        _LOG.exception("notebooklm shim: %s failed", method)
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32000, "message": str(exc)},
        }


def _dispatch(
    method: str | None, params: dict[str, Any], backend: NotebookBackend
) -> dict[str, Any]:
    if method == "initialize":
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}, "resources": {}},
            "serverInfo": {"name": "notebooklm-shim", "version": "0.1.0"},
        }
    if method == "tools/list":
        return {"tools": _TOOL_DEFINITIONS}
    if method == "resources/list":
        # Default stub advertises a single placeholder resource.
        return {"resources": [_resource_for("default")]}
    if method == "tools/call":
        return _handle_tool_call(params, backend)
    raise ValueError(f"unknown method: {method!r}")


def _handle_tool_call(
    params: dict[str, Any], backend: NotebookBackend
) -> dict[str, Any]:
    name = params.get("name")
    arguments = params.get("arguments") or {}
    if name == "search_notebook":
        notebook_id = str(arguments.get("notebook_id", ""))
        query = str(arguments.get("query", ""))
        if not notebook_id or not query:
            raise ValueError("notebook_id and query are required")
        hits = backend.search(notebook_id, query)
        return {"content": [{"type": "text", "text": json.dumps({"hits": hits})}]}
    if name == "summarize_notebook":
        notebook_id = str(arguments.get("notebook_id", ""))
        if not notebook_id:
            raise ValueError("notebook_id is required")
        summary = backend.summarize(notebook_id)
        return {
            "content": [{"type": "text", "text": summary}],
        }
    raise ValueError(f"unknown tool: {name!r}")


async def _run_loop(
    backend: NotebookBackend,
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    while True:
        line = await reader.readline()
        if not line:
            return
        try:
            request = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            _LOG.warning("notebooklm shim: malformed JSON-RPC frame; ignoring")
            continue
        response = handle_request(request, backend)
        writer.write((json.dumps(response) + "\n").encode("utf-8"))
        await writer.drain()


def main() -> None:  # pragma: no cover — integration entrypoint
    """Run the shim's stdin/stdout JSON-RPC loop."""
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    backend: NotebookBackend = StubNotebookBackend()

    async def _amain() -> None:
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader()
        await loop.connect_read_pipe(
            lambda: asyncio.StreamReaderProtocol(reader), sys.stdin
        )
        writer_transport, writer_protocol = await loop.connect_write_pipe(
            asyncio.streams.FlowControlMixin, sys.stdout
        )
        writer = asyncio.StreamWriter(writer_transport, writer_protocol, None, loop)
        await _run_loop(backend, reader, writer)

    asyncio.run(_amain())


if __name__ == "__main__":  # pragma: no cover
    main()
