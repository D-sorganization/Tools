"""Obsidian integration tools for Sidekick."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.shared.python.ai.tool_registry import ToolCategory, get_global_registry

logger = logging.getLogger(__name__)

# Configured Obsidian Vault path
_OBSIDIAN_VAULT_PATH: Path | None = None


def set_obsidian_vault_path(path: str | Path) -> None:
    """Set the local Obsidian Vault path."""
    global _OBSIDIAN_VAULT_PATH
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
    if not _OBSIDIAN_VAULT_PATH:
        return {"error": "Obsidian Vault path is not configured."}

    if not note_name.endswith(".md"):
        note_name += ".md"

    note_path = _OBSIDIAN_VAULT_PATH / note_name

    if not note_path.exists():
        return {"error": f"Note '{note_name}' not found in vault."}

    try:
        content = note_path.read_text(encoding="utf-8")
        return {"success": True, "content": content, "path": str(note_path)}
    except Exception as e:
        logger.error("Failed to read Obsidian note %s: %s", note_name, e)
        return {"error": str(e)}


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
    if not _OBSIDIAN_VAULT_PATH:
        return {"error": "Obsidian Vault path is not configured."}

    if not note_name.endswith(".md"):
        note_name += ".md"

    note_path = _OBSIDIAN_VAULT_PATH / note_name

    if note_path.exists() and not overwrite:
        return {
            "error": f"Note '{note_name}' already exists. Set overwrite=True to overwrite."  # noqa: E501
        }

    try:
        # Ensure vault path exists if we are writing nested notes
        note_path.parent.mkdir(parents=True, exist_ok=True)
        note_path.write_text(markdown_content, encoding="utf-8")
        logger.info("Wrote Obsidian note: %s", note_path)
        return {
            "success": True,
            "message": f"Successfully wrote note '{note_name}'.",
            "path": str(note_path),
        }
    except Exception as e:
        logger.error("Failed to write Obsidian note %s: %s", note_name, e)
        return {"error": str(e)}
