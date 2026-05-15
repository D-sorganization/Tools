"""Obsidian integration tools for Sidekick."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from urllib.parse import quote

from src.shared.python.ai.tool_registry import ToolCategory, get_global_registry

logger = logging.getLogger(__name__)

_OBSIDIAN_VAULT_PATH: Path | None = None
_OBSIDIAN_REST_API_URL: str | None = None
_OBSIDIAN_REST_API_KEY: str | None = None


def set_obsidian_vault_path(path: str | Path) -> None:
    """Set the local Obsidian Vault path."""
    global _OBSIDIAN_VAULT_PATH  # noqa: PLW0603
    _OBSIDIAN_VAULT_PATH = Path(path).resolve()


def set_obsidian_rest_api(url: str, api_key: str = "") -> None:
    """Configure the obsidian-local-rest-api plugin connection."""
    global _OBSIDIAN_REST_API_URL, _OBSIDIAN_REST_API_KEY  # noqa: PLW0603
    _OBSIDIAN_REST_API_URL = url
    _OBSIDIAN_REST_API_KEY = api_key


def _require_vault() -> Path:
    """Return the configured vault path or raise ValueError."""
    if _OBSIDIAN_VAULT_PATH is None:
        raise ValueError(
            "Obsidian vault path not configured. Call set_obsidian_vault_path()."
        )
    return _OBSIDIAN_VAULT_PATH


def _normalize_note_name(note_name: str) -> str:
    """Ensure the note name ends with .md."""
    return note_name if note_name.endswith(".md") else f"{note_name}.md"


def _resolve_note_path(vault: Path, note_name: str) -> Path:
    """Resolve and validate the note path within the vault.

    Precondition: note_name is a non-empty string.
    Raises ValueError for path traversal attempts.
    """
    if not note_name or not isinstance(note_name, str):
        raise TypeError("note_name must be a non-empty string")

    # Strip leading slashes / backslashes
    cleaned = note_name.lstrip("/\\")

    # Reject if .. appears in any component
    parts = Path(cleaned).parts
    if any(part == ".." for part in parts):
        raise ValueError(f"Path traversal detected in note_name: {note_name!r}")

    normalized = _normalize_note_name(cleaned)
    candidate = (vault / normalized).resolve()

    # Verify the resolved path is still inside the vault
    try:
        candidate.relative_to(vault)
    except ValueError:
        raise ValueError(
            f"Path traversal detected in note_name: {note_name!r}"
        ) from None

    return candidate


def _try_rest_read(note_name: str) -> dict[str, Any] | None:
    """Attempt to read a note via the local REST API.

    Returns None on any failure so the caller can fall back to filesystem.
    """
    if not _OBSIDIAN_REST_API_URL:
        return None
    try:
        import httpx  # noqa: PLC0415

        encoded = quote(_normalize_note_name(note_name))
        url = f"{_OBSIDIAN_REST_API_URL.rstrip('/')}/vault/{encoded}"
        headers = {}
        if _OBSIDIAN_REST_API_KEY:
            headers["Authorization"] = f"Bearer {_OBSIDIAN_REST_API_KEY}"
        response = httpx.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            content = response.text
            return {
                "note_name": note_name,
                "path": url,
                "content": content,
                "size_bytes": len(content.encode("utf-8")),
            }
        logger.debug(
            "REST API returned %d for %s; falling back to filesystem",
            response.status_code,
            note_name,
        )
    except Exception:  # noqa: BLE001
        logger.debug(
            "REST API unavailable for read of %s; falling back to filesystem",
            note_name,
        )
    return None


def _try_rest_write(note_name: str, markdown_content: str) -> bool:
    """Attempt to write a note via the local REST API.

    Returns True on success, False on any failure.
    """
    if not _OBSIDIAN_REST_API_URL:
        return False
    try:
        import httpx  # noqa: PLC0415

        encoded = quote(_normalize_note_name(note_name))
        url = f"{_OBSIDIAN_REST_API_URL.rstrip('/')}/vault/{encoded}"
        headers = {"Content-Type": "text/markdown"}
        if _OBSIDIAN_REST_API_KEY:
            headers["Authorization"] = f"Bearer {_OBSIDIAN_REST_API_KEY}"
        response = httpx.put(
            url,
            content=markdown_content.encode("utf-8"),
            headers=headers,
            timeout=5,
        )
        if response.status_code in (200, 201, 204):
            return True
        logger.debug(
            "REST API returned %d for write of %s; falling back to filesystem",
            response.status_code,
            note_name,
        )
    except Exception:  # noqa: BLE001
        logger.debug(
            "REST API unavailable for write of %s; falling back to filesystem",
            note_name,
        )
    return False


registry = get_global_registry()


@registry.register(
    "obsidian_read_note",
    "Read a markdown note from the configured Obsidian Vault.",
    category=ToolCategory.DATA_LOADING,
)
def obsidian_read_note(note_name: str) -> dict[str, Any]:
    """Read a markdown note from the Obsidian Vault.

    Precondition: note_name is a non-empty string.
    Precondition: Vault path must be configured via set_obsidian_vault_path().

    Args:
        note_name: The name of the note (with or without .md extension).

    Returns:
        dict with keys: note_name, path, content, size_bytes.

    Raises:
        TypeError: If note_name is not a non-empty string.
        ValueError: If vault is not configured or path traversal is detected.
        FileNotFoundError: If the note does not exist in the vault.
    """
    if not note_name or not isinstance(note_name, str):
        raise TypeError("note_name must be a non-empty string")

    vault = _require_vault()

    # Try REST API first if configured
    rest_result = _try_rest_read(note_name)
    if rest_result is not None:
        return rest_result

    # Filesystem path
    path = _resolve_note_path(vault, note_name)

    if not path.exists():
        # Case-insensitive recursive fallback search
        base_name = Path(_normalize_note_name(note_name.lstrip("/\\"))).name.lower()
        matches = [p for p in vault.rglob("*.md") if p.name.lower() == base_name]
        if matches:
            path = matches[0]
            logger.debug("Note found via glob fallback: %s", path)
        else:
            raise FileNotFoundError(f"Note '{note_name}' not found in vault at {vault}")

    content = path.read_text(encoding="utf-8")
    return {
        "note_name": note_name,
        "path": str(path),
        "content": content,
        "size_bytes": len(content.encode("utf-8")),
    }


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

    Precondition: note_name is a non-empty string.
    Precondition: markdown_content is a string (may be empty).
    Precondition: Vault path must be configured via set_obsidian_vault_path().

    Args:
        note_name: The name of the note.
        markdown_content: The markdown content to write.
        overwrite: Whether to overwrite the note if it already exists.

    Returns:
        dict with keys: success, note_name, path, size_bytes.

    Raises:
        TypeError: If note_name or markdown_content are wrong types.
        ValueError: If vault is not configured or path traversal is detected.
        FileExistsError: If the note exists and overwrite=False.
    """
    if not note_name or not isinstance(note_name, str):
        raise TypeError("note_name must be a non-empty string")
    if not isinstance(markdown_content, str):
        raise TypeError("markdown_content must be a string")

    vault = _require_vault()
    path = _resolve_note_path(vault, note_name)

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Note '{note_name}' already exists. Pass overwrite=True to replace."
        )

    # Try REST API first if configured (REST handles its own existence semantics)
    if _try_rest_write(note_name, markdown_content):
        size = len(markdown_content.encode("utf-8"))
        logger.info("Note '%s' written via REST API", note_name)
        return {
            "success": True,
            "note_name": note_name,
            "path": str(path),
            "size_bytes": size,
        }

    # Filesystem write
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_content, encoding="utf-8")
    size = len(markdown_content.encode("utf-8"))
    logger.info("Note '%s' written to %s", note_name, path)
    return {
        "success": True,
        "note_name": note_name,
        "path": str(path),
        "size_bytes": size,
    }


def obsidian_list_notes(folder: str = "") -> dict[str, Any]:
    """List all .md notes in the vault (optionally filtered to a subfolder).

    Precondition: Vault path must be configured via set_obsidian_vault_path().

    Args:
        folder: Optional subfolder path within the vault to restrict listing.

    Returns:
        dict with keys: notes (list of dicts with name/path/size_bytes), total.

    Raises:
        ValueError: If vault is not configured.
    """
    vault = _require_vault()
    base = vault / folder if folder else vault

    notes = []
    for md_file in sorted(base.rglob("*.md")):
        notes.append(
            {
                "name": md_file.stem,
                "path": str(md_file),
                "size_bytes": md_file.stat().st_size,
            }
        )

    return {"notes": notes, "total": len(notes)}
