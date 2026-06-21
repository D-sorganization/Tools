"""Tests for the Linear GraphQL API client (Phase 2 of #2759).

All tests are unit-level and mock the httpx HTTP layer so no real network
calls are made.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ._bootstrap import bootstrap_integration_client_test

# ---------------------------------------------------------------------------
# Bootstrap: add the repo root to sys.path so that ``src.*`` imports resolve,
# and stub out heavy transitive dependencies.
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[5]
bootstrap_integration_client_test(ROOT)

from src.shared.python.ai.tool_registry import ToolRegistry  # noqa: E402

_fresh_registry = ToolRegistry()


def _get_global_registry_stub() -> ToolRegistry:
    return _fresh_registry


import src.shared.python.ai.tool_registry as _tr_mod  # noqa: E402

_saved_get_global_registry = _tr_mod.get_global_registry
_tr_mod.get_global_registry = _get_global_registry_stub  # type: ignore[attr-defined]
try:
    # Import the module under test AFTER patching the registry.
    import src.shared.python.ai.integrations.linear as linear_mod  # noqa: E402
    from src.shared.python.ai.integrations.linear import (  # noqa: E402
        linear_create_issue,
        linear_query_issues,
        set_linear_api_token,
    )
finally:
    _tr_mod.get_global_registry = _saved_get_global_registry  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_ISSUES_RESPONSE: dict[str, Any] = {
    "data": {
        "issues": {
            "nodes": [
                {
                    "id": "ISS-1",
                    "title": "Auth bug",
                    "description": "Login fails on SSO",
                    "state": {"name": "In Progress"},
                    "assignee": {"name": "Alice"},
                    "priority": 1,
                    "url": "https://linear.app/team/issue/ISS-1",
                }
            ]
        }
    }
}

_SAMPLE_CREATE_RESPONSE: dict[str, Any] = {
    "data": {
        "issueCreate": {
            "success": True,
            "issue": {
                "id": "ISS-42",
                "title": "New feature request",
                "url": "https://linear.app/team/issue/ISS-42",
            },
        }
    }
}


def _make_mock_response(body: dict[str, Any], status_code: int = 200) -> MagicMock:
    """Return a mock httpx.Response-like object."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = json.dumps(body)
    resp.json.return_value = body
    return resp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_token(monkeypatch):
    """Reset default-credentials token and env var before each test."""
    linear_mod.get_default_credentials().token = None
    monkeypatch.delenv("LINEAR_API_KEY", raising=False)
    yield
    linear_mod.get_default_credentials().token = None


@pytest.fixture()
def with_token():
    """Set a dummy token for tests that need one."""
    set_linear_api_token("test-token-abc")
    yield
    set_linear_api_token.__wrapped__ if hasattr(
        set_linear_api_token, "__wrapped__"
    ) else None


# ---------------------------------------------------------------------------
# Tests: linear_query_issues
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_query_issues_returns_shaped_dict(with_token):
    """linear_query_issues returns a dict with 'issues' list of expected keys."""
    mock_resp = _make_mock_response(_SAMPLE_ISSUES_RESPONSE)

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_resp

        result = linear_query_issues("auth bug")

    assert "issues" in result
    assert len(result["issues"]) == 1
    issue = result["issues"][0]
    assert issue["id"] == "ISS-1"
    assert issue["title"] == "Auth bug"
    assert issue["state"] == "In Progress"
    assert issue["assignee"] == "Alice"
    assert issue["priority"] == 1
    assert issue["url"] == "https://linear.app/team/issue/ISS-1"


@pytest.mark.unit
def test_query_issues_empty_query_raises_value_error(with_token):
    """linear_query_issues raises ValueError for an empty query string."""
    with pytest.raises(ValueError, match="non-empty string"):
        linear_query_issues("")


@pytest.mark.unit
def test_query_issues_whitespace_query_raises_value_error(with_token):
    """linear_query_issues raises ValueError for a whitespace-only query."""
    with pytest.raises(ValueError, match="non-empty string"):
        linear_query_issues("   ")


# ---------------------------------------------------------------------------
# Tests: linear_create_issue
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_create_issue_returns_success_dict(with_token):
    """linear_create_issue returns a dict with success and issue fields."""
    mock_resp = _make_mock_response(_SAMPLE_CREATE_RESPONSE)

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_resp

        result = linear_create_issue(
            title="New feature request",
            description="Detailed description",
            team_id="TEAM-1",
        )

    assert result["success"] is True
    assert result["issue"]["id"] == "ISS-42"
    assert result["issue"]["title"] == "New feature request"
    assert "linear.app" in result["issue"]["url"]


@pytest.mark.unit
def test_create_issue_empty_title_raises_value_error(with_token):
    """linear_create_issue raises ValueError for an empty title."""
    with pytest.raises(ValueError, match="non-empty string"):
        linear_create_issue(title="", description="Some desc")


@pytest.mark.unit
def test_create_issue_whitespace_title_raises_value_error(with_token):
    """linear_create_issue raises ValueError for a whitespace-only title."""
    with pytest.raises(ValueError, match="non-empty string"):
        linear_create_issue(title="  ", description="Some desc")


# ---------------------------------------------------------------------------
# Tests: missing token
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_missing_token_raises_value_error():
    """Both tools raise ValueError when no API token is configured."""
    with pytest.raises(ValueError, match="Linear API token not configured"):
        linear_query_issues("something")


@pytest.mark.unit
def test_env_var_token_used_as_fallback(monkeypatch):
    """LINEAR_API_KEY env var is used when module-level token is not set."""
    monkeypatch.setenv("LINEAR_API_KEY", "env-token-xyz")
    mock_resp = _make_mock_response(_SAMPLE_ISSUES_RESPONSE)

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_resp

        result = linear_query_issues("auth bug")

    assert "issues" in result
    # Confirm the env-var token was sent in the Authorization header.
    call_kwargs = mock_client.post.call_args
    headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers", {})
    assert headers.get("Authorization") == "Bearer env-token-xyz"


# ---------------------------------------------------------------------------
# Tests: HTTP errors
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_http_401_raises_runtime_error(with_token):
    """A 401 response raises RuntimeError with status code in the message."""
    mock_resp = _make_mock_response({"error": "Unauthorized"}, status_code=401)

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_resp

        with pytest.raises(RuntimeError, match="401"):
            linear_query_issues("auth bug")


@pytest.mark.unit
def test_http_500_raises_runtime_error(with_token):
    """A 500 response raises RuntimeError with status code in the message."""
    mock_resp = _make_mock_response({"error": "Internal Server Error"}, status_code=500)

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_resp

        with pytest.raises(RuntimeError, match="500"):
            linear_create_issue(title="Bug", description="desc")


# ---------------------------------------------------------------------------
# Tests: network errors
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_network_error_raises_runtime_error(with_token):
    """A network-level httpx error raises RuntimeError with 'connection failed'."""
    import httpx as httpx_lib

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.post.side_effect = httpx_lib.ConnectError("connection refused")

        with pytest.raises(RuntimeError, match="Linear API connection failed"):
            linear_query_issues("auth bug")


@pytest.mark.unit
def test_timeout_error_raises_runtime_error(with_token):
    """A timeout raises RuntimeError with 'connection failed'."""
    import httpx as httpx_lib

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.post.side_effect = httpx_lib.TimeoutException("timed out")

        with pytest.raises(RuntimeError, match="Linear API connection failed"):
            linear_create_issue(title="Bug", description="desc")
