"""Notion integration tools for Sidekick."""

from __future__ import annotations

import logging
from typing import Any

from src.shared.python.ai.tool_registry import ToolCategory, get_global_registry

logger = logging.getLogger(__name__)

# Reserved for Phase 2: store the token when set_notion_api_token() is called.
# The tool functions below refuse unconditionally until Phase 2 implements the
# real REST client — token presence is irrelevant until then.
_NOTION_API_TOKEN: str | None = None


def set_notion_api_token(token: str) -> None:
    """Store the Notion API token in memory for session use (Phase 2)."""
    global _NOTION_API_TOKEN  # noqa: PLW0603
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
    raise NotImplementedError(
        "Notion integration is not yet implemented (Phase 2 of #2759). "
        "Configure your Notion API token via settings, then implement the REST "
        "client in src/shared/python/ai/integrations/notion.py."
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
    raise NotImplementedError(
        "Notion integration is not yet implemented (Phase 2 of #2759). "
        "Configure your Notion API token via settings, then implement the REST "
        "client in src/shared/python/ai/integrations/notion.py."
    )
