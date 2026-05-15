"""Obsidian integration tools for Sidekick."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.shared.python.ai.tool_registry import ToolCategory, get_global_registry

logger = logging.getLogger(__name__)

# Reserved for Phase 2: store the vault path when set_obsidian_vault_path() is called.
# The tool functions below refuse unconditionally until Phase 2 implements real
# local-filesystem Vault I/O — path configuration is irrelevant until then.
_OBSIDIAN_VAULT_PATH: Path | None = None


def set_obsidian_vault_path(path: str | Path) -> None:
    """Set the local Obsidian Vault path (Phase 2)."""
    global _OBSIDIAN_VAULT_PATH  # noqa: PLW0603
    _OBSIDIAN_VAULT_PATH = Path(path).resolve()


registry = get_global_registry()


@registry.register(
    "obsidian_read_note",
    "Read a markdown note from the configured Obsidian Vault.",
    category=ToolCategory.DATA_LOADING,
)
def obsidian_read_note(note_name: str) -> dict[str, Any]:
    """Read a markdown note from the Obsidian Vault.

    Args:
        note_name: The name of the note (with or without .md extension).
    """
    raise NotImplementedError(
        "Obsidian integration is not yet implemented (Phase 2 of #2759). "
        "Configure your Obsidian Vault path via settings, then implement the "
        "filesystem client in src/shared/python/ai/integrations/obsidian.py."
    )


@registry.register(
    "obsidian_write_note",
    "Create or update a markdown note in the configured Obsidian Vault.",
    category=ToolCategory.ANALYSIS,
    requires_confirmation=True,
)
def obsidian_write_note(
    note_name: str, markdown_content: str, overwrite: bool = False
) -> dict[str, Any]:
    """Write markdown content to a note in the Obsidian Vault.

    Args:
        note_name: The name of the note.
        markdown_content: The content to write.
        overwrite: Whether to overwrite if the note already exists.
    """
    raise NotImplementedError(
        "Obsidian integration is not yet implemented (Phase 2 of #2759). "
        "Configure your Obsidian Vault path via settings, then implement the "
        "filesystem client in src/shared/python/ai/integrations/obsidian.py."
    )
