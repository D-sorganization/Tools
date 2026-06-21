"""Unit tests for the Obsidian vault filesystem client (Phase 2, #2759)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ._bootstrap import bootstrap_integration_client_test

# ---------------------------------------------------------------------------
# Bootstrap: add repo root to sys.path and stub heavy transitive dependencies.
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[5]
bootstrap_integration_client_test(ROOT)

from src.shared.python.ai.tool_registry import ToolRegistry  # noqa: E402

_fresh_registry = ToolRegistry()


def _get_global_registry_stub() -> ToolRegistry:
    return _fresh_registry


import src.shared.python.ai.tool_registry as _tr_mod  # noqa: E402

_saved_get_global_registry = _tr_mod.get_global_registry
_tr_mod.get_global_registry = _get_global_registry_stub  # type: ignore[attr-defined]
try:
    import src.shared.python.ai.integrations.obsidian as obsidian_module  # noqa: E402
    from src.shared.python.ai.integrations.obsidian import (  # noqa: E402
        obsidian_list_notes,
        obsidian_read_note,
        obsidian_write_note,
        set_obsidian_vault_path,
    )
finally:
    _tr_mod.get_global_registry = _saved_get_global_registry  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _configure_vault(tmp_path: Path) -> None:
    """Point the module-level vault at tmp_path."""
    set_obsidian_vault_path(tmp_path)


def _reset_vault() -> None:
    """Clear vault configuration so tests don't bleed state."""
    obsidian_module.get_default_config().vault_path = None
    os.environ.pop("OBSIDIAN_VAULT_PATH", None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestObsidianWriteNote:
    def test_write_creates_file_with_correct_content(self, tmp_path: Path) -> None:
        """obsidian_write_note creates a .md file with the supplied content."""
        _configure_vault(tmp_path)
        try:
            result = obsidian_write_note("my_note", "# Hello\nWorld")
            note_path = tmp_path / "my_note.md"
            assert note_path.exists()
            assert note_path.read_text(encoding="utf-8") == "# Hello\nWorld"
            assert result["success"] is True
            assert Path(result["path"]).name == "my_note.md"
            assert result["bytes_written"] == len(b"# Hello\nWorld")
        finally:
            _reset_vault()

    def test_write_raises_file_exists_error_when_overwrite_false(
        self, tmp_path: Path
    ) -> None:
        """Raises FileExistsError when note exists and overwrite=False."""
        _configure_vault(tmp_path)
        try:
            (tmp_path / "existing.md").write_text("old", encoding="utf-8")
            with pytest.raises(FileExistsError, match="existing"):
                obsidian_write_note("existing", "new content", overwrite=False)
        finally:
            _reset_vault()

    def test_write_succeeds_when_overwrite_true(self, tmp_path: Path) -> None:
        """obsidian_write_note overwrites an existing note when overwrite=True."""
        _configure_vault(tmp_path)
        try:
            (tmp_path / "note.md").write_text("old", encoding="utf-8")
            result = obsidian_write_note("note", "new content", overwrite=True)
            assert result["success"] is True
            assert (tmp_path / "note.md").read_text(encoding="utf-8") == "new content"
        finally:
            _reset_vault()

    def test_write_creates_parent_directories(self, tmp_path: Path) -> None:
        """obsidian_write_note creates missing parent directories."""
        _configure_vault(tmp_path)
        try:
            result = obsidian_write_note("subdir/deep/note", "content")
            assert (tmp_path / "subdir" / "deep" / "note.md").exists()
            assert result["success"] is True
        finally:
            _reset_vault()

    def test_write_accepts_md_extension_in_note_name(self, tmp_path: Path) -> None:
        """obsidian_write_note accepts note names that already include .md."""
        _configure_vault(tmp_path)
        try:
            obsidian_write_note("explicit.md", "content")
            assert (tmp_path / "explicit.md").exists()
        finally:
            _reset_vault()


@pytest.mark.unit
class TestObsidianReadNote:
    def test_read_returns_correct_content(self, tmp_path: Path) -> None:
        """obsidian_read_note reads file content correctly."""
        _configure_vault(tmp_path)
        try:
            (tmp_path / "greeting.md").write_text("# Hi", encoding="utf-8")
            result = obsidian_read_note("greeting")
            assert result["content"] == "# Hi"
            assert Path(result["path"]).name == "greeting.md"
            assert "modified_at" in result
        finally:
            _reset_vault()

    def test_read_with_md_extension(self, tmp_path: Path) -> None:
        """obsidian_read_note works when caller includes .md extension."""
        _configure_vault(tmp_path)
        try:
            (tmp_path / "greeting.md").write_text("# Hi", encoding="utf-8")
            result = obsidian_read_note("greeting.md")
            assert result["content"] == "# Hi"
        finally:
            _reset_vault()

    def test_read_raises_file_not_found_for_nonexistent_note(
        self, tmp_path: Path
    ) -> None:
        """obsidian_read_note raises FileNotFoundError for missing notes."""
        _configure_vault(tmp_path)
        try:
            with pytest.raises(FileNotFoundError, match="ghost_note"):
                obsidian_read_note("ghost_note")
        finally:
            _reset_vault()

    def test_read_fallback_glob_finds_note_in_subdirectory(
        self, tmp_path: Path
    ) -> None:
        """obsidian_read_note falls back to glob search and finds notes in subdirs."""
        _configure_vault(tmp_path)
        try:
            subdir = tmp_path / "projects"
            subdir.mkdir()
            (subdir / "deep_note.md").write_text("found it", encoding="utf-8")
            result = obsidian_read_note("projects/deep_note")
            assert result["content"] == "found it"
        finally:
            _reset_vault()


@pytest.mark.unit
class TestObsidianListNotes:
    def test_list_returns_all_md_files(self, tmp_path: Path) -> None:
        """obsidian_list_notes returns all .md files in the vault."""
        _configure_vault(tmp_path)
        try:
            (tmp_path / "a.md").write_text("a", encoding="utf-8")
            (tmp_path / "b.md").write_text("b", encoding="utf-8")
            sub = tmp_path / "sub"
            sub.mkdir()
            (sub / "c.md").write_text("c", encoding="utf-8")
            result = obsidian_list_notes()
            assert result["count"] == 3
            assert set(result["notes"]) == {"a.md", "b.md", "sub/c.md"}
        finally:
            _reset_vault()

    def test_list_filtered_to_subfolder(self, tmp_path: Path) -> None:
        """obsidian_list_notes restricts listing to a given subfolder."""
        _configure_vault(tmp_path)
        try:
            (tmp_path / "root_note.md").write_text("root", encoding="utf-8")
            sub = tmp_path / "archive"
            sub.mkdir()
            (sub / "old.md").write_text("old", encoding="utf-8")
            result = obsidian_list_notes(folder="archive")
            assert result["count"] == 1
            assert result["notes"] == ["archive/old.md"]
        finally:
            _reset_vault()


@pytest.mark.unit
class TestObsidianValidation:
    def test_vault_not_configured_raises_value_error(self) -> None:
        """obsidian_read_note raises RuntimeError when vault path is not configured."""
        _reset_vault()
        with pytest.raises(RuntimeError, match="not configured"):
            obsidian_read_note("any_note")

    def test_vault_not_configured_write_raises_value_error(self) -> None:
        """obsidian_write_note raises RuntimeError when vault path is not configured."""
        _reset_vault()
        with pytest.raises(RuntimeError, match="not configured"):
            obsidian_write_note("any_note", "content")

    def test_path_traversal_rejected_in_read(self, tmp_path: Path) -> None:
        """obsidian_read_note rejects path traversal attempts."""
        _configure_vault(tmp_path)
        try:
            with pytest.raises(ValueError, match="traversal"):
                obsidian_read_note("../../../etc/passwd")
        finally:
            _reset_vault()

    def test_path_traversal_rejected_in_write(self, tmp_path: Path) -> None:
        """obsidian_write_note rejects path traversal attempts."""
        _configure_vault(tmp_path)
        try:
            with pytest.raises(ValueError, match="traversal"):
                obsidian_write_note("../../secret", "evil")
        finally:
            _reset_vault()

    def test_empty_note_name_raises_value_error_on_read(self, tmp_path: Path) -> None:
        """obsidian_read_note raises ValueError for empty note_name."""
        _configure_vault(tmp_path)
        try:
            with pytest.raises(ValueError, match="non-empty"):
                obsidian_read_note("")
        finally:
            _reset_vault()

    def test_non_string_markdown_raises_type_error_on_write(
        self, tmp_path: Path
    ) -> None:
        """Raises TypeError when markdown_content is not a string."""
        _configure_vault(tmp_path)
        try:
            with pytest.raises(TypeError, match="string"):
                obsidian_write_note("note", 42)  # type: ignore[arg-type]
        finally:
            _reset_vault()
