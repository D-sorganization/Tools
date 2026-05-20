"""VCR cassette tests for the AFFiNE GraphQL integration client.

Replays HTTP traffic from YAML cassettes so we can assert the real
GraphQL request/response shape used by
``src.shared.python.ai.integrations.affine`` without a live account.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from src.shared.python.ai.integrations import affine


@pytest.fixture
def affine_token_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Provide a fake Affine token and reset module-level state."""
    monkeypatch.setenv("AFFINE_API_KEY", "affine_fake_token_for_replay")
    monkeypatch.delenv("AFFINE_BASE_URL", raising=False)
    monkeypatch.setattr(affine, "_AFFINE_API_TOKEN", None)
    monkeypatch.setattr(affine, "_AFFINE_BASE_URL", "https://app.affine.pro/graphql")
    yield


@pytest.mark.xfail(reason="VCR broken in CI")
@pytest.mark.vcr
@pytest.mark.unit
def test_affine_list_workspaces_parses_response(affine_token_env: None) -> None:
    """``affine_list_workspaces`` returns ``{"workspaces": [...]}``."""
    result = affine.affine_list_workspaces()
    assert "workspaces" in result
    workspaces = result["workspaces"]
    assert isinstance(workspaces, list)
    assert len(workspaces) >= 1
    first = workspaces[0]
    assert "id" in first
    assert "name" in first


@pytest.mark.xfail(reason="VCR broken in CI")
@pytest.mark.vcr
@pytest.mark.unit
def test_affine_sync_notes_with_explicit_workspace(
    affine_token_env: None,
) -> None:
    """``affine_sync_notes`` creates a doc when given an explicit workspace."""
    result = affine.affine_sync_notes(
        title="Note from VCR replay",
        markdown_content="Body content.",
        workspace_id="ws-synthetic-1",
    )
    assert result["success"] is True
    assert result["doc_id"]
    assert result["workspace_id"] == "ws-synthetic-1"
    assert "note" in result


@pytest.mark.xfail(reason="VCR broken in CI")
@pytest.mark.vcr
@pytest.mark.unit
def test_affine_sync_notes_autofills_workspace(affine_token_env: None) -> None:
    """When ``workspace_id`` is omitted the first workspace is used."""
    result = affine.affine_sync_notes(
        title="Auto-workspace note",
        markdown_content="Body.",
    )
    assert result["success"] is True
    assert result["doc_id"]
    assert result["workspace_id"]  # populated from list_workspaces response


@pytest.mark.unit
def test_affine_sync_notes_rejects_empty_title() -> None:
    """Precondition: empty title raises ValueError before any HTTP."""
    with pytest.raises(ValueError, match="non-empty"):
        affine.affine_sync_notes(title="", markdown_content="x")


@pytest.mark.unit
def test_set_affine_api_token_rejects_empty() -> None:
    """Precondition: ``set_affine_api_token('')`` raises ValueError."""
    with pytest.raises(ValueError, match="non-empty"):
        affine.set_affine_api_token("")


@pytest.mark.unit
def test_set_affine_base_url_rejects_empty() -> None:
    """Precondition: ``set_affine_base_url('')`` raises ValueError."""
    with pytest.raises(ValueError, match="non-empty"):
        affine.set_affine_base_url("")
