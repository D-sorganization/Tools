"""Notion integration tools for Sidekick."""

from __future__ import annotations

import logging
from typing import Any

from src.shared.python.ai.tool_registry import ToolCategory, get_global_registry

logger = logging.getLogger(__name__)

# Placeholder API token management
_NOTION_API_TOKEN: str | None = None


def set_notion_api_token(token: str) -> None:
    """Store the Notion API token in memory for session use."""
    global _NOTION_API_TOKEN
    _NOTION_API_TOKEN = token


registry = get_global_registry()


@registry.register(
    "notion_push_report",
    "Push a generated report to a Notion workspace.",
    category=ToolCategory.ANALYSIS,
    requires_confirmation=True,
)
def notion_push_report(
    title: str, markdown_content: str, parent_page_id: str = ""
) -> dict[str, Any]:
    """Push markdown content as a new page in Notion.

    Args:
        title: The title of the report page.
        markdown_content: The markdown body of the report.
        parent_page_id: The ID of the parent Notion page or database.
    """
    if not _NOTION_API_TOKEN:
        return {"error": "Notion API token is not configured."}

    raise NotImplementedError(
        "Notion integration is not yet implemented. Real API calls are not made."
    )


@registry.register(
    "notion_read_knowledge_base",
    "Read an article from the Notion knowledge base.",
    category=ToolCategory.DATA_LOADING,
)
def notion_read_knowledge_base(query: str) -> dict[str, Any]:
    """Search and read knowledge base articles from Notion.

    Args:
        query: Search term for the knowledge base.
    """
    if not _NOTION_API_TOKEN:
        return {"error": "Notion API token is not configured."}

    raise NotImplementedError(
        "Notion integration is not yet implemented. Real API calls are not made."
    )
