"""Phase 2 tests for the NotebookLM MCP shim.

Covers the tools added in Tools #2900:
- list_notebooks
- create_notebook
- add_source_to_notebook
- generate_audio_overview
- follow_citation
- attach_to_chat

Plus extended ``notebook://{id}`` metadata and the new
``notebook://{id}/source/{source_id}`` resource.

All tests run against ``StubNotebookBackend`` for determinism.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.shared.python.ai.mcp.notebooklm_server import (
    StubNotebookBackend,
    handle_request,
)


def _call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    backend = StubNotebookBackend()
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }
    return handle_request(request, backend)


def _result_payload(response: dict[str, Any]) -> dict[str, Any]:
    assert "result" in response, f"unexpected error response: {response}"
    content = response["result"]["content"]
    assert content and content[0]["type"] == "text"
    return json.loads(content[0]["text"])


# ---------------------------------------------------------------------------
# tools/list — Phase 2 surface advertised alongside Phase 1
# ---------------------------------------------------------------------------


def test_tools_list_advertises_phase2_tools() -> None:
    backend = StubNotebookBackend()
    response = handle_request(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, backend
    )
    names = {tool["name"] for tool in response["result"]["tools"]}
    # Phase 1 tools still present
    assert {"search_notebook", "summarize_notebook"} <= names
    # Phase 2 additions
    assert {
        "list_notebooks",
        "create_notebook",
        "add_source_to_notebook",
        "generate_audio_overview",
        "follow_citation",
        "attach_to_chat",
    } <= names


def test_phase2_tools_have_input_schemas() -> None:
    backend = StubNotebookBackend()
    response = handle_request(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, backend
    )
    for tool in response["result"]["tools"]:
        assert "inputSchema" in tool, f"{tool['name']} missing inputSchema"
        schema = tool["inputSchema"]
        assert schema.get("type") == "object"
        assert "properties" in schema


def test_phase2_confirmation_tools_have_metadata() -> None:
    """generate_audio_overview and attach_to_chat carry requires_confirmation."""
    backend = StubNotebookBackend()
    response = handle_request(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, backend
    )
    tools_by_name = {tool["name"]: tool for tool in response["result"]["tools"]}
    for needs_confirm in ("generate_audio_overview", "attach_to_chat"):
        meta = tools_by_name[needs_confirm].get("metadata") or {}
        assert meta.get("requires_confirmation") is True, (
            f"{needs_confirm} must declare requires_confirmation=True"
        )


# ---------------------------------------------------------------------------
# list_notebooks
# ---------------------------------------------------------------------------


def test_list_notebooks_returns_descriptors() -> None:
    response = _call_tool("list_notebooks", {})
    payload = _result_payload(response)
    notebooks = payload["notebooks"]
    assert isinstance(notebooks, list)
    assert notebooks, "stub backend should return at least one notebook"
    for nb in notebooks:
        assert {"id", "title", "modified_at"} <= set(nb.keys())


def test_list_notebooks_is_deterministic() -> None:
    first = _result_payload(_call_tool("list_notebooks", {}))
    second = _result_payload(_call_tool("list_notebooks", {}))
    assert first == second


# ---------------------------------------------------------------------------
# create_notebook
# ---------------------------------------------------------------------------


def test_create_notebook_happy_path() -> None:
    response = _call_tool(
        "create_notebook",
        {"title": "Hydrolysis Kinetics", "sources": ["https://example.com/paper.pdf"]},
    )
    payload = _result_payload(response)
    assert payload["id"]
    assert payload["url"].startswith("notebook://")


def test_create_notebook_empty_sources_ok() -> None:
    response = _call_tool("create_notebook", {"title": "Empty Notebook", "sources": []})
    payload = _result_payload(response)
    assert payload["id"]


def test_create_notebook_rejects_empty_title() -> None:
    response = _call_tool("create_notebook", {"title": "", "sources": []})
    assert "error" in response
    assert response["error"]["code"] == -32000


def test_create_notebook_rejects_non_list_sources() -> None:
    response = _call_tool("create_notebook", {"title": "X", "sources": "not-a-list"})
    assert "error" in response


# ---------------------------------------------------------------------------
# add_source_to_notebook
# ---------------------------------------------------------------------------


def test_add_source_to_notebook_url() -> None:
    response = _call_tool(
        "add_source_to_notebook",
        {"notebook_id": "nb-1", "source_url_or_path": "https://example.com/a.pdf"},
    )
    payload = _result_payload(response)
    assert payload["source_id"]


def test_add_source_rejects_path_traversal() -> None:
    response = _call_tool(
        "add_source_to_notebook",
        {"notebook_id": "nb-1", "source_url_or_path": "../../../etc/passwd"},
    )
    assert "error" in response
    assert (
        "traversal" in response["error"]["message"].lower()
        or "reject" in response["error"]["message"].lower()
    )


def test_add_source_rejects_empty_notebook_id() -> None:
    response = _call_tool(
        "add_source_to_notebook",
        {"notebook_id": "", "source_url_or_path": "https://example.com/x"},
    )
    assert "error" in response


# ---------------------------------------------------------------------------
# generate_audio_overview
# ---------------------------------------------------------------------------


def test_generate_audio_overview_default_voice() -> None:
    response = _call_tool("generate_audio_overview", {"notebook_id": "nb-1"})
    payload = _result_payload(response)
    assert payload["audio_url"]
    assert isinstance(payload["duration_seconds"], (int, float))
    assert payload["duration_seconds"] > 0


def test_generate_audio_overview_custom_voice() -> None:
    response = _call_tool(
        "generate_audio_overview",
        {"notebook_id": "nb-1", "voice": "conversational"},
    )
    payload = _result_payload(response)
    assert payload["audio_url"]


def test_generate_audio_overview_requires_notebook_id() -> None:
    response = _call_tool("generate_audio_overview", {"notebook_id": ""})
    assert "error" in response


# ---------------------------------------------------------------------------
# follow_citation
# ---------------------------------------------------------------------------


def test_follow_citation_returns_source() -> None:
    response = _call_tool(
        "follow_citation",
        {"notebook_id": "nb-1", "citation_id": "cit-42"},
    )
    payload = _result_payload(response)
    assert payload["source_id"]
    assert payload["snippet"]
    assert payload["source_url"]


def test_follow_citation_requires_citation_id() -> None:
    response = _call_tool("follow_citation", {"notebook_id": "nb-1", "citation_id": ""})
    assert "error" in response


# ---------------------------------------------------------------------------
# attach_to_chat
# ---------------------------------------------------------------------------


def test_attach_to_chat_returns_context_size() -> None:
    response = _call_tool("attach_to_chat", {"notebook_id": "nb-1"})
    payload = _result_payload(response)
    assert isinstance(payload["context_size_tokens"], int)
    assert payload["context_size_tokens"] >= 0


def test_attach_to_chat_requires_notebook_id() -> None:
    response = _call_tool("attach_to_chat", {"notebook_id": ""})
    assert "error" in response


# ---------------------------------------------------------------------------
# Resources — extended metadata and per-source reads
# ---------------------------------------------------------------------------


def test_notebook_metadata_includes_sources() -> None:
    backend = StubNotebookBackend()
    response = handle_request(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "resources/read",
            "params": {"uri": "notebook://nb-1"},
        },
        backend,
    )
    assert "result" in response, response
    contents = response["result"]["contents"]
    assert contents
    data = json.loads(contents[0]["text"])
    assert data["id"] == "nb-1"
    assert "sources" in data
    assert isinstance(data["sources"], list)
    if data["sources"]:
        source = data["sources"][0]
        assert {"id", "title", "type", "url"} <= set(source.keys())


def test_notebook_source_resource_read() -> None:
    backend = StubNotebookBackend()
    response = handle_request(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "resources/read",
            "params": {"uri": "notebook://nb-1/source/src-1"},
        },
        backend,
    )
    assert "result" in response, response
    contents = response["result"]["contents"]
    assert contents
    # The source body is text content for the model.
    assert contents[0]["text"]


def test_notebook_resource_unknown_uri_errors() -> None:
    backend = StubNotebookBackend()
    response = handle_request(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "resources/read",
            "params": {"uri": "bogus://nope"},
        },
        backend,
    )
    assert "error" in response


# ---------------------------------------------------------------------------
# Unknown tool & JSON-RPC error envelope
# ---------------------------------------------------------------------------


def test_unknown_tool_returns_minus32000() -> None:
    response = _call_tool("not_a_real_tool", {})
    assert "error" in response
    assert response["error"]["code"] == -32000


@pytest.mark.parametrize(
    "tool_name,args",
    [
        ("list_notebooks", {}),
        ("create_notebook", {"title": "t", "sources": []}),
        ("add_source_to_notebook", {"notebook_id": "n", "source_url_or_path": "u"}),
        ("generate_audio_overview", {"notebook_id": "n"}),
        ("follow_citation", {"notebook_id": "n", "citation_id": "c"}),
        ("attach_to_chat", {"notebook_id": "n"}),
    ],
)
def test_phase2_tool_dispatch_happy_paths(tool_name: str, args: dict[str, Any]) -> None:
    response = _call_tool(tool_name, args)
    assert "result" in response, f"{tool_name} failed: {response}"


# ---------------------------------------------------------------------------
# Stub backend determinism
# ---------------------------------------------------------------------------


def test_stub_backend_deterministic_create_notebook() -> None:
    backend1 = StubNotebookBackend()
    backend2 = StubNotebookBackend()
    nb1 = backend1.create_notebook("Same", ["src://x"])
    nb2 = backend2.create_notebook("Same", ["src://x"])
    assert nb1["id"] == nb2["id"]
    assert nb1["url"] == nb2["url"]


def test_stub_backend_deterministic_follow_citation() -> None:
    backend = StubNotebookBackend()
    a = backend.follow_citation("nb-1", "cit-42")
    b = backend.follow_citation("nb-1", "cit-42")
    assert a == b
