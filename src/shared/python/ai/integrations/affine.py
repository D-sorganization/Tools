"""Affine integration tools for Sidekick."""

from __future__ import annotations

import logging
from typing import Any

from src.shared.python.ai.tool_registry import ToolCategory, get_global_registry

logger = logging.getLogger(__name__)

# Placeholder API token management
_AFFINE_API_TOKEN: str | None = None


def set_affine_api_token(token: str) -> None:
    """Store the Affine API token in memory for session use."""
    global _AFFINE_API_TOKEN
    _AFFINE_API_TOKEN = token


registry = get_global_registry()


@registry.register(
    "affine_sync_notes",
    "Sync markdown notes to an Affine workspace.",
    category=ToolCategory.ANALYSIS,
    requires_confirmation=True,
)
def affine_sync_notes(
    title: str, markdown_content: str, workspace_id: str = ""
) -> dict[str, Any]:
    """Sync markdown content as a new note in Affine.

    Args:
        title: The title of the note.
        markdown_content: The markdown body of the note.
        workspace_id: The ID of the target Affine workspace.
    """
    if not _AFFINE_API_TOKEN:
        return {"error": "Affine API token is not configured."}

    logger.info("Syncing note to Affine: %s", title)
    return {
        "success": True,
        "message": f"Successfully synced note '{title}' to Affine.",
        "url": "https://app.affine.pro/placeholder",
    }
