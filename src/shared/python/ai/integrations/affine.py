"""Affine integration tools for Sidekick."""

from __future__ import annotations

import logging
from typing import Any

from src.shared.python.ai.tool_registry import ToolCategory, get_global_registry

logger = logging.getLogger(__name__)

# Reserved for Phase 2: store the token when set_affine_api_token() is called.
# The tool functions below refuse unconditionally until Phase 2 implements the
# real REST client — token presence is irrelevant until then.
_AFFINE_API_TOKEN: str | None = None


def set_affine_api_token(token: str) -> None:
    """Store the Affine API token in memory for session use (Phase 2)."""
    global _AFFINE_API_TOKEN  # noqa: PLW0603
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
    raise NotImplementedError(
        "Affine integration is not yet implemented (Phase 2 of #2759). "
        "Configure your Affine API token via settings, then implement the REST "
        "client in src/shared/python/ai/integrations/affine.py."
    )
