"""Unit tests for the local-vault Obsidian integration (Tools #2896).

These tests cover the real local-filesystem client that replaces the
``NotImplementedError`` stubs left in place when Tools #2759 was closed
prematurely as ``completed``.
"""

from __future__ import annotations

import logging
import os
import sys
import types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: follow the pattern used in tests/shared/python/ai/
# test_adapter_contract.py to make ``src.shared.python.*`` imports resolve
# without requiring real ``__init__.py`` files on each ancestor.
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_PACKAGE_STUBS: list[tuple[str, str | None]] = [
    ("src", "src"),
    ("src.shared", "src/shared"),
    ("src.shared.python", "src/shared/python"),
    ("src.shared.python.ai", "src/shared/python/ai"),
    ("src.shared.python.ai.integrations", "src/shared/python/ai/integrations"),
]
for _mod_name, _rel_path in _PACKAGE_STUBS:
    if _mod_name not in sys.modules:
        _stub = types.ModuleType(_mod_name)
        if _rel_path is not None:
            _stub.__path__ = [str(ROOT / _rel_path)]
        sys.modules[_mod_name] = _stub

_logging_config_stub = sys.modules.setdefault(
    "src.shared.python.logging_pkg",
    types.ModuleType("src.shared.python.logging_pkg"),
)
_logging_config_stub = sys.modules.setdefault(
    "src.shared.python.logging_pkg.logging_config",
    types.ModuleType("src.shared.python.logging_pkg.logging_config"),
)
_logging_config_stub.get_logger = logging.getLogger  # type: ignore[attr-defined]
_logging_config_stub.setup_logging = lambda *a, **kw: None  # type: ignore[attr-defined]

from src.shared.python.ai.integrations import obsidian  # noqa: E402
from src.shared.python.ai.integrations.obsidian import (  # noqa: E402
    ObsidianPathError,
    obsidian_list_notes,
    obsidian_read_note,
    obsidian_search,
    obsidian_write_note,
    set_obsidian_vault_path,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    """Provide an isolated, configured vault for a single test."""
    (tmp_path / "welcome.md").write_text("# Welcome\n\nHello vault.", encoding="utf-8")
    sub = tmp_path / "projects"
    sub.mkdir()
    (sub / "alpha.md").write_text("# Alpha\n\nProject alpha notes.", encoding="utf-8")
    (sub / "beta.md").write_text("# Beta\n\nProject beta notes.", encoding="utf-8")
    (tmp_path / "not-a-note.txt").write_text("plain text", encoding="utf-8")
    set_obsidian_vault_path(tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def _reset_vault():
    """Ensure module state does not leak between tests."""
    obsidian._OBSIDIAN_VAULT_PATH = None  # type: ignore[attr-defined]
    saved = os.environ.pop("OBSIDIAN_VAULT_PATH", None)
    yield
    obsidian._OBSIDIAN_VAULT_PATH = None  # type: ignore[attr-defined]
    if saved is not None:
        os.environ["OBSIDIAN_VAULT_PATH"] = saved


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def test_obsidian_read_note_returns_content(vault: Path) -> None:
    result = obsidian_read_note("welcome")
    assert result["content"] == "# Welcome\n\nHello vault."
    assert result["path"].endswith("welcome.md")
    assert "modified_at" in result


def test_obsidian_read_note_accepts_md_extension(vault: Path) -> None:
    result = obsidian_read_note("welcome.md")
    assert "Hello vault." in result["content"]


def test_obsidian_read_note_reads_subfolder_note(vault: Path) -> None:
    result = obsidian_read_note("projects/alpha")
    assert "Project alpha notes." in result["content"]


def test_obsidian_read_note_missing_raises(vault: Path) -> None:
    with pytest.raises(FileNotFoundError):
        obsidian_read_note("does-not-exist")


def test_obsidian_read_note_rejects_empty_name(vault: Path) -> None:
    with pytest.raises(ValueError):
        obsidian_read_note("")


def test_obsidian_read_note_rejects_non_string(vault: Path) -> None:
    with pytest.raises(TypeError):
        obsidian_read_note(123)  # type: ignore[arg-type]


def test_obsidian_read_note_unicode_roundtrip(vault: Path) -> None:
    (vault / "unicode.md").write_text("# 你好 🌟\n\némojis", encoding="utf-8")
    result = obsidian_read_note("unicode")
    assert "你好" in result["content"]
    assert "🌟" in result["content"]
    assert "émojis" in result["content"]


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def test_obsidian_write_note_creates_file(vault: Path) -> None:
    result = obsidian_write_note("new-note", "# New\n\nbody")
    assert result["success"] is True
    assert result["bytes_written"] > 0
    assert (vault / "new-note.md").read_text(encoding="utf-8") == "# New\n\nbody"


def test_obsidian_write_note_creates_parent_directories(vault: Path) -> None:
    obsidian_write_note("inbox/today/log", "captured")
    assert (vault / "inbox" / "today" / "log.md").read_text(encoding="utf-8") == (
        "captured"
    )


def test_obsidian_write_note_refuses_overwrite_by_default(vault: Path) -> None:
    with pytest.raises(FileExistsError):
        obsidian_write_note("welcome", "REPLACED")


def test_obsidian_write_note_overwrite_when_flag_set(vault: Path) -> None:
    obsidian_write_note("welcome", "REPLACED", overwrite=True)
    assert (vault / "welcome.md").read_text(encoding="utf-8") == "REPLACED"


def test_obsidian_write_note_rejects_empty_name(vault: Path) -> None:
    with pytest.raises(ValueError):
        obsidian_write_note("", "body")


def test_obsidian_write_note_rejects_non_string_content(vault: Path) -> None:
    with pytest.raises(TypeError):
        obsidian_write_note("ok", 123)  # type: ignore[arg-type]


def test_obsidian_write_note_unicode_content(vault: Path) -> None:
    obsidian_write_note("uni", "日本語 — café ☕")
    assert (vault / "uni.md").read_text(encoding="utf-8") == "日本語 — café ☕"


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------


def test_obsidian_list_notes_returns_only_md(vault: Path) -> None:
    result = obsidian_list_notes()
    names = result["notes"]
    assert "welcome.md" in names
    assert "projects/alpha.md" in names or "projects\\alpha.md" in names
    assert "projects/beta.md" in names or "projects\\beta.md" in names
    assert all(n.endswith(".md") for n in names)
    assert "not-a-note.txt" not in names


def test_obsidian_list_notes_filters_by_folder(vault: Path) -> None:
    result = obsidian_list_notes(folder="projects")
    names = result["notes"]
    assert len(names) == 2
    assert all("alpha" in n or "beta" in n for n in names)


def test_obsidian_list_notes_empty_folder(vault: Path) -> None:
    (vault / "empty").mkdir()
    result = obsidian_list_notes(folder="empty")
    assert result["notes"] == []


def test_obsidian_list_notes_rejects_nonexistent_folder(vault: Path) -> None:
    with pytest.raises(FileNotFoundError):
        obsidian_list_notes(folder="does-not-exist")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def test_obsidian_search_finds_substring(vault: Path) -> None:
    result = obsidian_search("Project alpha")
    matches = result["matches"]
    assert len(matches) == 1
    assert matches[0]["path"].endswith("alpha.md")
    assert "Project alpha" in matches[0]["snippet"]


def test_obsidian_search_finds_across_multiple_notes(vault: Path) -> None:
    result = obsidian_search("Project")
    assert len(result["matches"]) == 2


def test_obsidian_search_empty_query_rejected(vault: Path) -> None:
    with pytest.raises(ValueError):
        obsidian_search("")


def test_obsidian_search_returns_no_matches(vault: Path) -> None:
    result = obsidian_search("nonexistent-magic-string-xyz")
    assert result["matches"] == []


def test_obsidian_search_empty_vault_returns_no_matches(tmp_path: Path) -> None:
    set_obsidian_vault_path(tmp_path)
    result = obsidian_search("anything")
    assert result["matches"] == []


def test_obsidian_search_is_case_insensitive(vault: Path) -> None:
    result = obsidian_search("WELCOME")
    assert len(result["matches"]) == 1


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------


def test_obsidian_read_blocks_parent_traversal(vault: Path) -> None:
    with pytest.raises(ObsidianPathError):
        obsidian_read_note("../secret")


def test_obsidian_read_blocks_nested_parent_traversal(vault: Path) -> None:
    with pytest.raises(ObsidianPathError):
        obsidian_read_note("projects/../../secret")


def test_obsidian_write_blocks_parent_traversal(vault: Path) -> None:
    with pytest.raises(ObsidianPathError):
        obsidian_write_note("../escape", "x")


def test_obsidian_read_blocks_absolute_posix_path(vault: Path) -> None:
    with pytest.raises(ObsidianPathError):
        obsidian_read_note("/etc/passwd")


@pytest.mark.skipif(os.name != "nt", reason="Windows-specific drive-letter check")
def test_obsidian_read_blocks_drive_letter_on_windows(vault: Path) -> None:
    with pytest.raises(ObsidianPathError):
        obsidian_read_note("C:\\Windows\\System32\\drivers\\etc\\hosts")


def test_obsidian_list_blocks_parent_traversal(vault: Path) -> None:
    with pytest.raises(ObsidianPathError):
        obsidian_list_notes(folder="../")


# ---------------------------------------------------------------------------
# Configuration (set_obsidian_vault_path / env var / unset)
# ---------------------------------------------------------------------------


def test_obsidian_read_without_vault_configured_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="vault"):
        obsidian_read_note("anything")


def test_obsidian_write_without_vault_configured_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="vault"):
        obsidian_write_note("anything", "x")


def test_obsidian_list_without_vault_configured_raises() -> None:
    with pytest.raises(RuntimeError, match="vault"):
        obsidian_list_notes()


def test_obsidian_search_without_vault_configured_raises() -> None:
    with pytest.raises(RuntimeError, match="vault"):
        obsidian_search("x")


def test_set_obsidian_vault_path_rejects_nonexistent(tmp_path: Path) -> None:
    bogus = tmp_path / "no-such-dir"
    with pytest.raises(FileNotFoundError):
        set_obsidian_vault_path(bogus)


def test_set_obsidian_vault_path_rejects_file(tmp_path: Path) -> None:
    a_file = tmp_path / "f.txt"
    a_file.write_text("x", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        set_obsidian_vault_path(a_file)


def test_obsidian_vault_path_from_env(tmp_path: Path) -> None:
    (tmp_path / "from-env.md").write_text("env note", encoding="utf-8")
    os.environ["OBSIDIAN_VAULT_PATH"] = str(tmp_path)
    # No explicit set_obsidian_vault_path call — env var must be honored.
    result = obsidian_read_note("from-env")
    assert "env note" in result["content"]


# ---------------------------------------------------------------------------
# Registry integration — tools must be registered and callable
# ---------------------------------------------------------------------------


def test_obsidian_tools_registered() -> None:
    from src.shared.python.ai.tool_registry import get_global_registry

    reg = get_global_registry()
    names = {t.name for t in reg.list_tools()}
    assert "obsidian_read_note" in names
    assert "obsidian_write_note" in names
    assert "obsidian_list_notes" in names
    assert "obsidian_search" in names


def test_obsidian_read_no_longer_raises_notimplementederror(vault: Path) -> None:
    # The phantom-completion regression check: the tool must not raise
    # NotImplementedError under any circumstance now.
    try:
        obsidian_read_note("welcome")
    except NotImplementedError as exc:  # pragma: no cover
        pytest.fail(f"obsidian_read_note still raises NotImplementedError: {exc}")
