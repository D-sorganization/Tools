"""Unit tests for the Phase 2 Affine GraphQL client.

All tests are isolated: they mock ``httpx.Client.post`` so no real network
requests are made.
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: add repo root to sys.path and stub heavy transitive deps.
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


_exc_stub = _make_stub("src.shared.python.ai.exceptions")
_exc_stub.ToolExecutionError = Exception  # type: ignore[attr-defined]

_types_stub = _make_stub("src.shared.python.ai.types")
_types_stub.ToolResult = dict  # type: ignore[attr-defined]

# Patch get_global_registry before importing the module under test.
from src.shared.python.ai.tool_registry import ToolRegistry  # noqa: E402

_fresh_registry = ToolRegistry()
import src.shared.python.ai.tool_registry as _tr_mod  # noqa: E402

_tr_mod.get_global_registry = lambda: _fresh_registry  # type: ignore[attr-defined]

import src.shared.python.ai.integrations.affine as affine_mod  # noqa: E402
from src.shared.python.ai.integrations.affine import (  # noqa: E402
    affine_list_workspaces,
    affine_sync_notes,
    set_affine_api_token,
    set_affine_base_url,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_TOKEN = "test-session-token"
_DEFAULT_WS_ID = "ws-abc123"
_DEFAULT_DOC_ID = "doc-xyz789"

_WORKSPACES_RESPONSE = {
    "data": {"workspaces": [{"id": _DEFAULT_WS_ID, "name": "My Workspace"}]}
}
_CREATE_DOC_RESPONSE = {"data": {"createDoc": {"id": _DEFAULT_DOC_ID}}}


def _make_mock_response(json_body: dict, status_code: int = 200) -> MagicMock:
    """Build a mock httpx Response."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_body
    mock_resp.text = str(json_body)
    if status_code >= 400:
        import httpx

        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            message=f"HTTP {status_code}",
            request=MagicMock(),
            response=mock_resp,
        )
    else:
        mock_resp.raise_for_status.return_value = None
    return mock_resp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_token_and_url(monkeypatch):
    """Reset module-level token and base URL between tests."""
    monkeypatch.setattr(affine_mod, "_AFFINE_API_TOKEN", None)
    monkeypatch.setattr(
        affine_mod, "_AFFINE_BASE_URL", "https://app.affine.pro/graphql"
    )
    monkeypatch.delenv("AFFINE_API_KEY", raising=False)
    monkeypatch.delenv("AFFINE_BASE_URL", raising=False)
    yield


# ---------------------------------------------------------------------------
# Tests: affine_sync_notes — with workspace_id
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_affine_sync_notes_with_workspace_id_returns_success():
    """affine_sync_notes with workspace_id calls createDoc and returns success dict.

    Precondition: token set, workspace_id provided.
    Postcondition: returns dict with success=True, doc_id, workspace_id, note.
    """
    set_affine_api_token(_DEFAULT_TOKEN)

    create_resp = _make_mock_response(_CREATE_DOC_RESPONSE)

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.post.return_value = create_resp

        result = affine_sync_notes(
            title="My Note",
            markdown_content="# Hello",
            workspace_id=_DEFAULT_WS_ID,
        )

    assert result["success"] is True
    assert result["doc_id"] == _DEFAULT_DOC_ID
    assert result["workspace_id"] == _DEFAULT_WS_ID
    assert "note" in result
    # Only one POST call (createDoc) — no workspace listing needed.
    assert mock_client.post.call_count == 1


# ---------------------------------------------------------------------------
# Tests: affine_sync_notes — without workspace_id (auto-fetch)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_affine_sync_notes_without_workspace_id_fetches_workspaces():
    """affine_sync_notes without workspace_id auto-fetches workspaces first.

    Precondition: token set, workspace_id omitted.
    Postcondition: two POST calls (list workspaces, createDoc); uses first ws.
    """
    set_affine_api_token(_DEFAULT_TOKEN)

    ws_resp = _make_mock_response(_WORKSPACES_RESPONSE)
    create_resp = _make_mock_response(_CREATE_DOC_RESPONSE)

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.post.side_effect = [ws_resp, create_resp]

        result = affine_sync_notes(
            title="Auto WS Note",
            markdown_content="# Body",
        )

    assert result["success"] is True
    assert result["doc_id"] == _DEFAULT_DOC_ID
    assert result["workspace_id"] == _DEFAULT_WS_ID
    assert mock_client.post.call_count == 2


# ---------------------------------------------------------------------------
# Tests: affine_list_workspaces
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_affine_list_workspaces_returns_shaped_dict():
    """affine_list_workspaces returns {"workspaces": [{"id": ..., "name": ...}]}.

    Precondition: token set, API returns workspace list.
    Postcondition: result has "workspaces" key with list of dicts.
    """
    set_affine_api_token(_DEFAULT_TOKEN)

    ws_resp = _make_mock_response(_WORKSPACES_RESPONSE)

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.post.return_value = ws_resp

        result = affine_list_workspaces()

    assert "workspaces" in result
    workspaces = result["workspaces"]
    assert isinstance(workspaces, list)
    assert len(workspaces) == 1
    assert workspaces[0]["id"] == _DEFAULT_WS_ID
    assert workspaces[0]["name"] == "My Workspace"


# ---------------------------------------------------------------------------
# Tests: missing token
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_affine_sync_notes_missing_token_raises_value_error():
    """affine_sync_notes raises ValueError when no API token is configured.

    Precondition: no token set, no AFFINE_API_KEY env var.
    Postcondition: ValueError with descriptive message.
    """
    with pytest.raises(ValueError, match="Affine API token not configured"):
        affine_sync_notes(
            title="No Token Note",
            markdown_content="content",
            workspace_id=_DEFAULT_WS_ID,
        )


@pytest.mark.unit
def test_affine_list_workspaces_missing_token_raises_value_error():
    """affine_list_workspaces raises ValueError when no API token is configured."""
    with pytest.raises(ValueError, match="Affine API token not configured"):
        affine_list_workspaces()


@pytest.mark.unit
def test_affine_token_from_env_var(monkeypatch):
    """AFFINE_API_KEY env var is used when no in-memory token is set.

    Precondition: env var set, workspace_id provided.
    Postcondition: request succeeds (no ValueError raised).
    """
    monkeypatch.setenv("AFFINE_API_KEY", _DEFAULT_TOKEN)

    create_resp = _make_mock_response(_CREATE_DOC_RESPONSE)

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.post.return_value = create_resp

        result = affine_sync_notes(
            title="Env Token Note",
            markdown_content="body",
            workspace_id=_DEFAULT_WS_ID,
        )

    assert result["success"] is True


# ---------------------------------------------------------------------------
# Tests: HTTP errors
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_affine_sync_notes_http_error_raises_runtime_error():
    """HTTP 401 from Affine API raises RuntimeError with status code.

    Precondition: token set, server returns 401.
    Postcondition: RuntimeError message contains '401'.
    """

    set_affine_api_token(_DEFAULT_TOKEN)

    error_resp = _make_mock_response({}, status_code=401)

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.post.return_value = error_resp

        with pytest.raises(RuntimeError, match="Affine API error 401"):
            affine_sync_notes(
                title="Error Note",
                markdown_content="body",
                workspace_id=_DEFAULT_WS_ID,
            )


@pytest.mark.unit
def test_affine_sync_notes_network_error_raises_runtime_error():
    """Network failure raises RuntimeError with connection message.

    Precondition: token set, httpx raises RequestError.
    Postcondition: RuntimeError message contains 'connection failed'.
    """
    import httpx

    set_affine_api_token(_DEFAULT_TOKEN)

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.post.side_effect = httpx.ConnectError("connection refused")

        with pytest.raises(RuntimeError, match="Affine API connection failed"):
            affine_sync_notes(
                title="Network Error Note",
                markdown_content="body",
                workspace_id=_DEFAULT_WS_ID,
            )


# ---------------------------------------------------------------------------
# Tests: GraphQL errors field
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_affine_sync_notes_graphql_errors_raises_runtime_error():
    """GraphQL-level errors field in response raises RuntimeError.

    Precondition: token set, server returns HTTP 200 with errors field.
    Postcondition: RuntimeError message contains 'GraphQL error'.
    """
    set_affine_api_token(_DEFAULT_TOKEN)

    gql_error_body = {
        "data": None,
        "errors": [{"message": "Workspace not found"}],
    }
    error_resp = _make_mock_response(gql_error_body, status_code=200)

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.post.return_value = error_resp

        with pytest.raises(RuntimeError, match="Affine GraphQL error"):
            affine_sync_notes(
                title="GraphQL Error Note",
                markdown_content="body",
                workspace_id=_DEFAULT_WS_ID,
            )


# ---------------------------------------------------------------------------
# Tests: DbC — empty title
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_affine_sync_notes_empty_title_raises_value_error():
    """affine_sync_notes raises ValueError for empty title.

    Precondition: title is empty string.
    Postcondition: ValueError raised before any network call.
    """
    set_affine_api_token(_DEFAULT_TOKEN)

    with pytest.raises(ValueError, match="title must be a non-empty string"):
        affine_sync_notes(
            title="",
            markdown_content="body",
            workspace_id=_DEFAULT_WS_ID,
        )


# ---------------------------------------------------------------------------
# Tests: set_affine_base_url
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_set_affine_base_url_used_in_requests(monkeypatch):
    """set_affine_base_url causes _run_affine_query to POST to the custom URL.

    Precondition: custom base URL set.
    Postcondition: POST is called with the custom endpoint.
    """
    set_affine_api_token(_DEFAULT_TOKEN)
    custom_url = "http://localhost:3010/graphql"
    set_affine_base_url(custom_url)

    create_resp = _make_mock_response(_CREATE_DOC_RESPONSE)

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.post.return_value = create_resp

        affine_sync_notes(
            title="Self-hosted Note",
            markdown_content="body",
            workspace_id=_DEFAULT_WS_ID,
        )

    called_url = mock_client.post.call_args[0][0]
    assert called_url == custom_url
