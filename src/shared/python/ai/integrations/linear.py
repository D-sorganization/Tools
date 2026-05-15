"""Linear integration tools for Sidekick."""

from __future__ import annotations

import logging
from typing import Any

from src.shared.python.ai.tool_registry import ToolCategory, get_global_registry

logger = logging.getLogger(__name__)

# Placeholder API token management
_LINEAR_API_TOKEN: str | None = None


def set_linear_api_token(token: str) -> None:
    """Store the Linear API token in memory for session use."""
    global _LINEAR_API_TOKEN
    _LINEAR_API_TOKEN = token


registry = get_global_registry()


@registry.register(
    "linear_query_issues",
    "Query Linear issues based on a search term or status.",
    category=ToolCategory.ANALYSIS,
)
def linear_query_issues(query: str, status: str = "open") -> dict[str, Any]:
    """Query Linear for issues.

    Args:
        query: The search term to find relevant issues.
        status: The status of issues to return (e.g. 'open', 'done').
    """
    if not _LINEAR_API_TOKEN:
        return {
            "error": "Linear API token is not configured. Please provide it in settings."  # noqa: E501
        }

    raise NotImplementedError(
        "Linear integration is not yet implemented. Real API calls are not made."
    )


@registry.register(
    "linear_create_issue",
    "Create a new issue in Linear.",
    category=ToolCategory.ANALYSIS,
    requires_confirmation=True,
)
def linear_create_issue(
    title: str, description: str, team_id: str = ""
) -> dict[str, Any]:
    """Create a new Linear issue.

    Args:
        title: The title of the new issue.
        description: Detailed description in Markdown.
        team_id: The team ID to create the issue under.
    """
    if not _LINEAR_API_TOKEN:
        return {
            "error": "Linear API token is not configured. Please provide it in settings."  # noqa: E501
        }

    raise NotImplementedError(
        "Linear integration is not yet implemented. Real API calls are not made."
    )
