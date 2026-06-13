"""Tests for the Notion REST API client (Phase 2 of #2759).

All tests are unit-level and mock httpx to avoid network calls.
"""

from __future__ import annotations

import json
import logging
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: add repo root to sys.path and stub heavy transitive deps.
# This mirrors the pattern used in test_integrations_phase_1.py.
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_PACKAGE_STUBS: list[tuple[str, str | None]] = [
    ("src", "src"),
    ("src.shared", "src/shared"),
    ("src.shared.python", "src/shared/python"),
    ("src.shared.python.ai", "src/shared/python/ai"),
    ("src.shared.python.ai.integrations", "src/shared/python/ai/integrations"),
    ("src.shared.python.logging_pkg", None),
    ("src.shared.python.logging_pkg.logging_config", None),
]
for _mod_name, _rel_path in _PACKAGE_STUBS:
    if _mod_name not in sys.modules:
        _stub = types.ModuleType(_mod_name)
        if _rel_path is not None:
            _stub.__path__ = [str(ROOT / _rel_path)]  # type: ignore[attr-defined]
        sys.modules[_mod_name] = _stub

_logging_config_stub = sys.modules["src.shared.python.logging_pkg.logging_config"]
_logging_config_stub.get_logger = logging.getLogger  # type: ignore[attr-defined]
_logging_config_stub.setup_logging = lambda *a, **kw: None  # type: ignore[attr-defined]


def _make_stub(name: str) -> types.ModuleType:
    stub = types.ModuleType(name)
    sys.modules[name] = stub
    return stub


_exc_stub = sys.modules["src.shared.python.ai.exceptions"]
if not hasattr(_exc_stub, "ToolExecutionError"):
    _exc_stub.ToolExecutionError = Exception  # type: ignore[attr-defined]

_types_stub = sys.modules["src.shared.python.ai.types"]
if not hasattr(_types_stub, "ToolResult"):
    _types_stub.ToolResult = dict  # type: ignore[attr-defined]

# Import tool_registry and patch get_global_registry before notion module loads.
from src.shared.python.ai.tool_registry import ToolRegistry  # noqa: E402

_fresh_registry = ToolRegistry()


def _get_global_registry_stub() -> ToolRegistry:
    return _fresh_registry


import src.shared.python.ai.tool_registry as _tr_mod  # noqa: E402

_tr_mod.get_global_registry = _get_global_registry_stub  # type: ignore[attr-defined]

# Remove the cached notion module so our patched registry is used.
sys.modules.pop("src.shared.python.ai.integrations.notion", None)

import src.shared.python.ai.integrations.notion as _notion_mod  # noqa: E402
from src.shared.python.ai.integrations.notion import (  # noqa: E402
    _markdown_to_notion_blocks,
    notion_push_report,
    notion_read_knowledge_base,
    set_notion_api_token,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_httpx_response(status_code: int, body: dict) -> MagicMock:
    """Return a mock that behaves like an httpx.Response."""
    mock = MagicMock()
    mock.status_code = status_code
    mock.text = json.dumps(body)
    mock.json.return_value = body
    if status_code >= 400:
        import httpx

        mock.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}",
            request=MagicMock(),
            response=mock,
        )
    else:
        mock.raise_for_status.return_value = None
    return mock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_token():
    """Reset the global token before and after every test."""
    _notion_mod._NOTION_API_TOKEN = None
    yield
    _notion_mod._NOTION_API_TOKEN = None


@pytest.fixture()
def with_token():
    """Set a dummy token for tests that need one."""
    set_notion_api_token("test-secret-token")


# ---------------------------------------------------------------------------
# Token / configuration tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_missing_token_raises_value_error_for_read(monkeypatch):
    """notion_read_knowledge_base raises ValueError when no token is set."""
    monkeypatch.delenv("NOTION_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Notion API token not configured"):
        notion_read_knowledge_base("onboarding")


@pytest.mark.unit
def test_missing_token_raises_value_error_for_push(monkeypatch):
    """notion_push_report raises ValueError when no token is set."""
    monkeypatch.delenv("NOTION_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Notion API token not configured"):
        notion_push_report("Report", "# Hello", parent_page_id="abc123")


# ---------------------------------------------------------------------------
# DbC / input validation tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_empty_query_raises_value_error(with_token):
    """notion_read_knowledge_base rejects empty query."""
    with pytest.raises(ValueError, match="query must not be empty"):
        notion_read_knowledge_base("")


@pytest.mark.unit
def test_empty_title_raises_value_error(with_token):
    """notion_push_report rejects empty title."""
    with pytest.raises(ValueError, match="title must not be empty"):
        notion_push_report("", "content", parent_page_id="abc123")


@pytest.mark.unit
def test_missing_parent_page_id_raises_value_error(with_token):
    """notion_push_report raises ValueError when parent_page_id is omitted."""
    with pytest.raises(ValueError, match="parent_page_id is required"):
        notion_push_report("Report", "# Hello")


# ---------------------------------------------------------------------------
# HTTP 401 raises RuntimeError
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_http_401_raises_runtime_error_for_read(with_token):
    """notion_read_knowledge_base raises RuntimeError on HTTP 401."""
    mock_resp = _make_httpx_response(401, {"message": "Unauthorized"})
    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.post.return_value = mock_resp

    with patch.object(_notion_mod.httpx, "Client", return_value=mock_client):
        with pytest.raises(RuntimeError, match="Notion API error 401"):
            notion_read_knowledge_base("onboarding")


@pytest.mark.unit
def test_http_401_raises_runtime_error_for_push(with_token):
    """notion_push_report raises RuntimeError on HTTP 401."""
    mock_resp = _make_httpx_response(401, {"message": "Unauthorized"})
    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.post.return_value = mock_resp

    with patch.object(_notion_mod.httpx, "Client", return_value=mock_client):
        with pytest.raises(RuntimeError, match="Notion API error 401"):
            notion_push_report("Report", "# Hello", parent_page_id="abc123")


# ---------------------------------------------------------------------------
# notion_read_knowledge_base happy path
# ---------------------------------------------------------------------------


_SEARCH_RESPONSE = {
    "object": "list",
    "results": [
        {
            "object": "page",
            "id": "page-id-001",
            "url": "https://www.notion.so/page-id-001",
            "properties": {"title": {"title": [{"plain_text": "Onboarding Guide"}]}},
        },
        {
            "object": "page",
            "id": "page-id-002",
            "url": "https://www.notion.so/page-id-002",
            "properties": {"Name": {"title": [{"plain_text": "Employee Handbook"}]}},
        },
    ],
    "has_more": False,
}


@pytest.mark.unit
def test_notion_read_knowledge_base_returns_results(with_token):
    """notion_read_knowledge_base parses results into the expected shape."""
    mock_resp = _make_httpx_response(200, _SEARCH_RESPONSE)
    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.post.return_value = mock_resp

    with patch.object(_notion_mod.httpx, "Client", return_value=mock_client):
        result = notion_read_knowledge_base("onboarding")

    assert "results" in result
    assert isinstance(result["results"], list)
    assert len(result["results"]) == 2

    first = result["results"][0]
    assert first["id"] == "page-id-001"
    assert first["title"] == "Onboarding Guide"
    assert first["url"] == "https://www.notion.so/page-id-001"

    # Second result uses "Name" property key
    second = result["results"][1]
    assert second["title"] == "Employee Handbook"

    assert result["has_more"] is False
    assert "next_cursor" not in result


@pytest.mark.unit
def test_notion_read_knowledge_base_pagination(with_token):
    """notion_read_knowledge_base includes next_cursor when has_more is True."""
    paginated_response = dict(_SEARCH_RESPONSE)
    paginated_response["has_more"] = True
    paginated_response["next_cursor"] = "cursor-xyz"

    mock_resp = _make_httpx_response(200, paginated_response)
    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.post.return_value = mock_resp

    with patch.object(_notion_mod.httpx, "Client", return_value=mock_client):
        result = notion_read_knowledge_base("onboarding")

    assert result["has_more"] is True
    assert result["next_cursor"] == "cursor-xyz"


# ---------------------------------------------------------------------------
# notion_push_report happy path
# ---------------------------------------------------------------------------


_PAGE_RESPONSE = {
    "object": "page",
    "id": "new-page-id-123",
    "url": "https://www.notion.so/new-page-id-123",
}


@pytest.mark.unit
def test_notion_push_report_returns_success(with_token):
    """notion_push_report returns success dict with page_id and url."""
    mock_resp = _make_httpx_response(200, _PAGE_RESPONSE)
    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.post.return_value = mock_resp

    with patch.object(_notion_mod.httpx, "Client", return_value=mock_client):
        result = notion_push_report(
            "Q1 Report", "# Hello\nContent here.", parent_page_id="parent-abc"
        )

    assert result["success"] is True
    assert result["page_id"] == "new-page-id-123"
    assert result["url"] == "https://www.notion.so/new-page-id-123"


# ---------------------------------------------------------------------------
# _markdown_to_notion_blocks unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_markdown_heading_1():
    """# Heading converts to heading_1 block."""
    blocks = _markdown_to_notion_blocks("# My Heading")
    assert len(blocks) == 1
    assert blocks[0]["type"] == "heading_1"
    assert blocks[0]["heading_1"]["rich_text"][0]["text"]["content"] == "My Heading"


@pytest.mark.unit
def test_markdown_heading_2():
    """## Heading converts to heading_2 block."""
    blocks = _markdown_to_notion_blocks("## Sub Heading")
    assert len(blocks) == 1
    assert blocks[0]["type"] == "heading_2"
    assert blocks[0]["heading_2"]["rich_text"][0]["text"]["content"] == "Sub Heading"


@pytest.mark.unit
def test_markdown_heading_3():
    """### Heading converts to heading_3 block."""
    blocks = _markdown_to_notion_blocks("### Deep Heading")
    assert len(blocks) == 1
    assert blocks[0]["type"] == "heading_3"


@pytest.mark.unit
def test_markdown_paragraph():
    """Plain text converts to paragraph block."""
    blocks = _markdown_to_notion_blocks("This is a paragraph.")
    assert len(blocks) == 1
    assert blocks[0]["type"] == "paragraph"
    assert (
        blocks[0]["paragraph"]["rich_text"][0]["text"]["content"]
        == "This is a paragraph."
    )


@pytest.mark.unit
@pytest.mark.parametrize("prefix", ["- ", "* "])
def test_markdown_bullet_list(prefix):
    """Bullet list items convert to bulleted_list_item blocks."""
    blocks = _markdown_to_notion_blocks(f"{prefix}First item")
    assert len(blocks) == 1
    assert blocks[0]["type"] == "bulleted_list_item"
    assert (
        blocks[0]["bulleted_list_item"]["rich_text"][0]["text"]["content"]
        == "First item"
    )


@pytest.mark.unit
def test_markdown_numbered_list():
    """Numbered list items convert to numbered_list_item blocks."""
    blocks = _markdown_to_notion_blocks("1. Step one\n2. Step two")
    assert len(blocks) == 2
    for block in blocks:
        assert block["type"] == "numbered_list_item"
    assert (
        blocks[0]["numbered_list_item"]["rich_text"][0]["text"]["content"] == "Step one"
    )
    assert (
        blocks[1]["numbered_list_item"]["rich_text"][0]["text"]["content"] == "Step two"
    )


@pytest.mark.unit
def test_markdown_code_block():
    """Fenced code blocks convert to code blocks."""
    md = "```python\nprint('hello')\n```"
    blocks = _markdown_to_notion_blocks(md)
    assert len(blocks) == 1
    assert blocks[0]["type"] == "code"
    assert blocks[0]["code"]["language"] == "python"
    assert blocks[0]["code"]["rich_text"][0]["text"]["content"] == "print('hello')"


@pytest.mark.unit
def test_markdown_mixed_content():
    """Mixed markdown converts to the correct sequence of block types."""
    md = "# Title\n\nSome paragraph.\n\n- item one\n- item two\n\n1. step"
    blocks = _markdown_to_notion_blocks(md)
    types = [b["type"] for b in blocks]
    assert types == [
        "heading_1",
        "paragraph",
        "bulleted_list_item",
        "bulleted_list_item",
        "numbered_list_item",
    ]
