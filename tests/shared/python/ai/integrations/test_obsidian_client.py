"""Unit tests for the Obsidian vault filesystem client (Phase 2, #2759)."""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: add repo root to sys.path and stub heavy transitive dependencies.
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_PACKAGE_STUBS: list[tuple[str, str | None]] = [
    ("src", "src"),
    ("src.shared", "src/shared"),
    ("src.shared.python", "src/shared/python"),
    ("src.shared.python.ai", "src/shared/python/ai"),
    ("src.shared.python.ai.integrations", "src/shared/python/ai/integrations"),
    ("src.shared.python.logging_pkg", None),
    ("src.shared.python.logging_pkg.logging_config", None),
]
for _mod_name, _rel_path in _PACKAGE_STUBS:
    if _mod_name not in sys.modules:
        _stub = types.ModuleType(_mod_name)
        if _rel_path is not None:
            _stub.__path__ = [str(ROOT / _rel_path)]  # type: ignore[attr-defined]
        sys.modules[_mod_name] = _stub

_logging_config_stub = sys.modules["src.shared.python.logging_pkg.logging_config"]
_logging_config_stub.get_logger = logging.getLogger  # type: ignore[attr-defined]
_logging_config_stub.setup_logging = lambda *a, **kw: None  # type: ignore[attr-defined]


def _make_stub(name: str) -> types.ModuleType:
    stub = types.ModuleType(name)
    sys.modules[name] = stub
    return stub


_exc_stub = _make_stub("src.shared.python.ai.exceptions")
_exc_stub.ToolExecutionError = Exception  # type: ignore[attr-defined]

_types_stub = _make_stub("src.shared.python.ai.types")
_types_stub.ToolResult = dict  # type: ignore[attr-defined]

from src.shared.python.ai.tool_registry import ToolRegistry  # noqa: E402

_fresh_registry = ToolRegistry()


def _get_global_registry_stub() -> ToolRegistry:
    return _fresh_registry


import src.shared.python.ai.tool_registry as _tr_mod  # noqa: E402

_tr_mod.get_global_registry = _get_global_registry_stub  # type: ignore[attr-defined]

import src.shared.python.ai.integrations.obsidian as obsidian_module  # noqa: E402
from src.shared.python.ai.integrations.obsidian import (  # noqa: E402
    obsidian_list_notes,
    obsidian_read_note,
    obsidian_write_note,
    set_obsidian_vault_path,
)

sys.modules.pop("src.shared.python.ai.exceptions", None)
sys.modules.pop("src.shared.python.ai.types", None)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _configure_vault(tmp_path: Path) -> None:
    """Point the module-level vault at tmp_path."""
    set_obsidian_vault_path(tmp_path)


def _reset_vault() -> None:
    """Clear the vault path so tests don't bleed state."""
    obsidian_module._OBSIDIAN_VAULT_PATH = None
    obsidian_module._OBSIDIAN_REST_API_URL = None
    obsidian_module._OBSIDIAN_REST_API_KEY = None


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
            assert result["note_name"] == "my_note"
            assert result["size_bytes"] == len(b"# Hello\nWorld")
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
            assert result["note_name"] == "greeting"
            assert result["size_bytes"] == len(b"# Hi")
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
            # Request by name only (no subdir prefix) — should find via glob
            result = obsidian_read_note("deep_note")
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
            assert result["total"] == 3
            names = {n["name"] for n in result["notes"]}
            assert names == {"a", "b", "c"}
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
            assert result["total"] == 1
            assert result["notes"][0]["name"] == "old"
        finally:
            _reset_vault()


@pytest.mark.unit
class TestObsidianValidation:
    def test_vault_not_configured_raises_value_error(self) -> None:
        """All tool functions raise ValueError when vault path is not configured."""
        _reset_vault()
        with pytest.raises(ValueError, match="not configured"):
            obsidian_read_note("any_note")

    def test_vault_not_configured_write_raises_value_error(self) -> None:
        """obsidian_write_note raises ValueError when vault path is not configured."""
        _reset_vault()
        with pytest.raises(ValueError, match="not configured"):
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

    def test_empty_note_name_raises_type_error_on_read(self, tmp_path: Path) -> None:
        """obsidian_read_note raises TypeError for empty note_name."""
        _configure_vault(tmp_path)
        try:
            with pytest.raises(TypeError, match="non-empty"):
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
