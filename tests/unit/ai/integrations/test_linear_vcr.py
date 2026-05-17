"""VCR cassette tests for the Linear integration client.

These tests verify that the real API call shape produced by
``src.shared.python.ai.integrations.linear`` matches what the Linear
GraphQL API expects, by replaying pre-recorded HTTP traffic stored in
cassettes alongside this file.

The tests run fully offline (``record_mode="none"`` in
``conftest.py``); no live Linear credentials are required.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from src.shared.python.ai.integrations import linear


@pytest.fixture
def linear_token_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Provide a fake Linear API token so the client can authenticate.

    The real token is never used because VCR replays the cassette and
    redacts the ``authorization`` header.
    """
    monkeypatch.setenv("LINEAR_API_KEY", "lin_api_fake_token_for_replay")
    # Reset module-level cache between tests.
    monkeypatch.setattr(linear, "_LINEAR_API_TOKEN", None)
    yield


@pytest.mark.vcr
@pytest.mark.unit
def test_linear_query_issues_returns_parsed_list(linear_token_env: None) -> None:
    """``linear_query_issues`` parses GraphQL nodes into flat issue dicts."""
    result = linear.linear_query_issues(query="bug")
    assert "issues" in result
    issues = result["issues"]
    assert isinstance(issues, list)
    assert len(issues) >= 1
    first = issues[0]
    # Verify the parser produced the documented flat shape.
    for key in ("id", "title", "description", "state", "assignee", "priority", "url"):
        assert key in first


@pytest.mark.vcr
@pytest.mark.unit
def test_linear_query_issues_empty_results(linear_token_env: None) -> None:
    """Empty result set still returns a well-formed dict."""
    result = linear.linear_query_issues(query="no-matches-xyz")
    assert result == {"issues": []}


@pytest.mark.vcr
@pytest.mark.unit
def test_linear_create_issue_returns_success(linear_token_env: None) -> None:
    """``linear_create_issue`` returns ``success`` and an issue payload."""
    result = linear.linear_create_issue(
        title="Sample issue from VCR replay",
        description="Body content used in cassette.",
        team_id="team_synthetic_id",
    )
    assert result["success"] is True
    issue = result["issue"]
    assert issue["id"]
    assert issue["title"] == "Sample issue from VCR replay"
    assert issue["url"].startswith("https://")


@pytest.mark.unit
def test_linear_query_issues_rejects_empty_query() -> None:
    """Precondition: empty query raises ``ValueError`` (no HTTP attempted)."""
    with pytest.raises(ValueError, match="non-empty"):
        linear.linear_query_issues(query="")


@pytest.mark.unit
def test_linear_create_issue_rejects_empty_title() -> None:
    """Precondition: empty title raises ``ValueError`` (no HTTP attempted)."""
    with pytest.raises(ValueError, match="non-empty"):
        linear.linear_create_issue(title="", description="x")
