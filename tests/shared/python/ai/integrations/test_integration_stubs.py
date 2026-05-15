"""Tests for AI integration stubs.

Verifies that NotImplementedError is raised when a token is configured,
preventing fake-success responses from misleading users.
"""

from __future__ import annotations

import enum
import logging
import sys
import types
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: stub broken imports so integration modules load in pytest run.
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

# Stub tool_registry used by all integration modules.
_tool_registry_stub = types.ModuleType("src.shared.python.ai.tool_registry")
sys.modules["src.shared.python.ai.tool_registry"] = _tool_registry_stub

_ToolCategory = enum.Enum("ToolCategory", ["ANALYSIS", "DATA_LOADING"])
_tool_registry_stub.ToolCategory = _ToolCategory  # type: ignore[attr-defined]


class _FakeRegistry:
    """Minimal registry that passes through decorated functions as-is."""

    def register(self, name, description, category=None, requires_confirmation=False):
        def decorator(func):
            return func

        return decorator


_tool_registry_stub.get_global_registry = lambda: _FakeRegistry()  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------

import pytest  # noqa: E402

from src.shared.python.ai.integrations import affine, linear, notion  # noqa: E402
from src.shared.python.ai.integrations.affine import (  # noqa: E402
    affine_sync_notes,
    set_affine_api_token,
)
from src.shared.python.ai.integrations.linear import (  # noqa: E402
    linear_create_issue,
    linear_query_issues,
    set_linear_api_token,
)
from src.shared.python.ai.integrations.notion import (  # noqa: E402
    notion_push_report,
    notion_read_knowledge_base,
    set_notion_api_token,
)


@pytest.fixture(autouse=True)
def reset_tokens():
    """Reset integration tokens before and after each test."""
    linear._LINEAR_API_TOKEN = None
    notion._NOTION_API_TOKEN = None
    affine._AFFINE_API_TOKEN = None
    yield
    linear._LINEAR_API_TOKEN = None
    notion._NOTION_API_TOKEN = None
    affine._AFFINE_API_TOKEN = None


# ---------------------------------------------------------------------------
# Linear
# ---------------------------------------------------------------------------


def test_linear_query_issues_no_token_returns_error():
    """Without a token, return a not-configured error dict (safe/honest)."""
    result = linear_query_issues("some query")
    assert "error" in result


def test_linear_query_issues_with_token_raises_not_implemented():
    """With a token, raise NotImplementedError instead of fake data."""
    set_linear_api_token("real-looking-token-123")
    with pytest.raises(
        NotImplementedError, match="Linear integration is not yet implemented"
    ):
        linear_query_issues("auth migration")


def test_linear_create_issue_no_token_returns_error():
    """Without a token, return a not-configured error dict."""
    result = linear_create_issue("Title", "Description")
    assert "error" in result


def test_linear_create_issue_with_token_raises_not_implemented():
    """With a token, raise NotImplementedError instead of fake success."""
    set_linear_api_token("real-looking-token-123")
    with pytest.raises(
        NotImplementedError, match="Linear integration is not yet implemented"
    ):
        linear_create_issue("My Issue", "Some description", team_id="TEAM-1")


# ---------------------------------------------------------------------------
# Notion
# ---------------------------------------------------------------------------


def test_notion_push_report_no_token_returns_error():
    """Without a token, return a not-configured error dict."""
    result = notion_push_report("Report", "# Content")
    assert "error" in result


def test_notion_push_report_with_token_raises_not_implemented():
    """With a token, raise NotImplementedError instead of fake success."""
    set_notion_api_token("secret_notion_token_abc123")
    with pytest.raises(
        NotImplementedError, match="Notion integration is not yet implemented"
    ):
        notion_push_report("My Report", "# Content", parent_page_id="PAGE-123")


def test_notion_read_knowledge_base_no_token_returns_error():
    """Without a token, return a not-configured error dict."""
    result = notion_read_knowledge_base("search term")
    assert "error" in result


def test_notion_read_knowledge_base_with_token_raises_not_implemented():
    """With a token, raise NotImplementedError instead of fake articles."""
    set_notion_api_token("secret_notion_token_abc123")
    with pytest.raises(
        NotImplementedError, match="Notion integration is not yet implemented"
    ):
        notion_read_knowledge_base("auth migration")


# ---------------------------------------------------------------------------
# Affine
# ---------------------------------------------------------------------------


def test_affine_sync_notes_no_token_returns_error():
    """Without a token, return a not-configured error dict."""
    result = affine_sync_notes("Title", "# Content")
    assert "error" in result


def test_affine_sync_notes_with_token_raises_not_implemented():
    """With a token, raise NotImplementedError instead of fake success."""
    set_affine_api_token("affine_token_xyz789")
    with pytest.raises(
        NotImplementedError, match="Affine integration is not yet implemented"
    ):
        affine_sync_notes("My Note", "# Content", workspace_id="WS-1")
