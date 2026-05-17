"""VCR cassette tests for the Notion integration client.

Replays HTTP traffic from YAML cassettes so we can verify that the
real Notion REST API call shape matches what the integration produces
without needing a live token.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from src.shared.python.ai.integrations import notion


@pytest.fixture
def notion_token_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Provide a fake Notion token so requests can be constructed."""
    monkeypatch.setenv("NOTION_API_KEY", "secret_fake_notion_token_for_replay")
    monkeypatch.setattr(notion, "_NOTION_API_TOKEN", None)
    yield


@pytest.mark.vcr
@pytest.mark.unit
def test_notion_push_report_returns_page_metadata(notion_token_env: None) -> None:
    """``notion_push_report`` returns ``success``, ``page_id``, ``url``."""
    result = notion.notion_push_report(
        title="Sample VCR report",
        markdown_content="# Heading\n\nParagraph body.\n\n- bullet 1\n- bullet 2",
        parent_page_id="00000000-0000-0000-0000-000000000001",
    )
    assert result["success"] is True
    assert result["page_id"]
    assert result["url"].startswith("https://")


@pytest.mark.vcr
@pytest.mark.unit
def test_notion_read_knowledge_base_parses_results(notion_token_env: None) -> None:
    """``notion_read_knowledge_base`` flattens search hits to id/title/url."""
    result = notion.notion_read_knowledge_base(query="engineering")
    assert "results" in result
    assert "has_more" in result
    assert isinstance(result["results"], list)
    assert len(result["results"]) >= 1
    first = result["results"][0]
    assert set(first.keys()) == {"id", "title", "url"}


@pytest.mark.vcr
@pytest.mark.unit
def test_notion_read_knowledge_base_pagination(notion_token_env: None) -> None:
    """When ``has_more`` is true the response includes a ``next_cursor``."""
    result = notion.notion_read_knowledge_base(query="paged")
    assert result["has_more"] is True
    assert result["next_cursor"]


@pytest.mark.unit
def test_notion_push_report_requires_parent_page_id(notion_token_env: None) -> None:
    """Precondition: empty parent_page_id raises ValueError."""
    with pytest.raises(ValueError, match="parent_page_id"):
        notion.notion_push_report(title="x", markdown_content="y", parent_page_id="")


@pytest.mark.unit
def test_notion_push_report_requires_title(notion_token_env: None) -> None:
    """Precondition: empty title raises ValueError before any HTTP."""
    with pytest.raises(ValueError, match="title"):
        notion.notion_push_report(title="", markdown_content="y", parent_page_id="abc")


@pytest.mark.unit
def test_notion_read_knowledge_base_requires_query() -> None:
    """Precondition: empty query raises ValueError before any HTTP."""
    with pytest.raises(ValueError, match="query"):
        notion.notion_read_knowledge_base(query="")
