"""Integration tests for the Obsidian vault client (Tools #2938).

Verifies that the vault client operates against a real local filesystem
using a temporary directory vault fixture.  No mocked data is used —
every assertion reads from or writes to actual files on disk.

Acceptance criteria from #2938:
- obsidian_vault_client reads + writes against a real local vault
- A temp vault fixture creates notes, reads them back, lists vault contents
- The #2759 safety concern (fake data with real tokens) is not reachable

Cross-references: #2834, #2896, #2759, #2938
"""

from __future__ import annotations

import logging
import os
import sys
import types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: add the repo root to sys.path so that ``src.*`` imports resolve,
# and stub out heavy transitive dependencies that the integration modules pull
# in via tool_registry (logging_pkg, exceptions, types).
# Mirrors the pattern used in tests/shared/python/ai/test_integrations_phase_1.py
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_PACKAGE_STUBS: list[tuple[str, str | None]] = [
    ("src", "src"),
    ("src.shared", "src/shared"),
    ("src.shared.python", "src/shared/python"),
    ("src.shared.python.config", "src/shared/python/config"),
    ("src.shared.python.ai", "src/shared/python/ai"),
    ("src.shared.python.ai.adapters", "src/shared/python/ai/adapters"),
]
for _mod_name, _rel_path in _PACKAGE_STUBS:
    if _mod_name not in sys.modules:
        _stub = types.ModuleType(_mod_name)
        if _rel_path is not None:
            _stub.__path__ = [str(ROOT / _rel_path)]
        sys.modules[_mod_name] = _stub

# Stub logging_pkg so tool_registry can import get_logger without extra deps.
_logging_config_stub = sys.modules.setdefault(
    "src.shared.python.logging_pkg.logging_config",
    types.ModuleType("src.shared.python.logging_pkg.logging_config"),
)
_logging_config_stub.get_logger = logging.getLogger  # type: ignore[attr-defined]
_logging_config_stub.setup_logging = lambda *a, **kw: None  # type: ignore[attr-defined]

# Stub logging_pkg parent so submodule lookup succeeds.
_logging_pkg_stub = sys.modules.setdefault(
    "src.shared.python.logging_pkg",
    types.ModuleType("src.shared.python.logging_pkg"),
)

# Stub httpx so integration modules that import it do not fail on missing dep.
if "httpx" not in sys.modules:
    _httpx_stub = types.ModuleType("httpx")
    _httpx_stub.AsyncClient = object  # type: ignore[attr-defined]
    sys.modules["httpx"] = _httpx_stub

# ---------------------------------------------------------------------------
# Module-level import (must succeed for all tests in this file to run)
# ---------------------------------------------------------------------------

from src.shared.python.ai.integrations import obsidian as _obsidian_mod  # noqa: E402

obsidian_read_note = _obsidian_mod.obsidian_read_note
obsidian_write_note = _obsidian_mod.obsidian_write_note
obsidian_list_notes = _obsidian_mod.obsidian_list_notes
obsidian_search = _obsidian_mod.obsidian_search
set_obsidian_vault_path = _obsidian_mod.set_obsidian_vault_path
ObsidianPathError = _obsidian_mod.ObsidianPathError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def temp_vault(tmp_path: Path) -> Path:
    """Create a temporary directory that acts as an Obsidian vault.

    Populates the vault with a few starter notes so every test has a
    non-empty vault without needing to write their own seed data.

    Postcondition: ``set_obsidian_vault_path`` is called with the new
    directory, so the module-level singleton points at this temp vault.
    After the test the vault path is reset to avoid cross-test pollution.
    """
    vault = tmp_path / "test_vault"
    vault.mkdir()

    # Seed data
    (vault / "welcome.md").write_text(
        "# Welcome\nThis is the welcome note.", encoding="utf-8"
    )
    (vault / "daily").mkdir()
    (vault / "daily" / "2026-05-17.md").write_text(
        "# Daily Note\nToday I worked on [[welcome]].", encoding="utf-8"
    )

    # Point the module at this vault
    set_obsidian_vault_path(vault)

    yield vault

    # Reset so other tests (or the module state) are not polluted.
    _obsidian_mod._OBSIDIAN_VAULT_PATH = None  # noqa: SLF001


# ---------------------------------------------------------------------------
# Tests — Real filesystem, no mocks
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_write_and_read_back_note(temp_vault: Path) -> None:
    """Create a note via the API and verify it can be read back verbatim."""
    content = "# My Note\n\nThis is [[wikilink]] content."
    result = obsidian_write_note("my_note", content)

    assert result["success"] is True
    assert result["created"] is True
    assert result["bytes_written"] > 0

    read_result = obsidian_read_note("my_note")
    assert read_result["content"] == content
    assert read_result["path"].endswith("my_note.md")
    assert read_result["modified_at"]  # non-empty ISO-8601 timestamp


@pytest.mark.integration
def test_list_vault_contents(temp_vault: Path) -> None:
    """List all notes — result must include the seeded notes."""
    result = obsidian_list_notes()
    notes = result["notes"]

    assert result["count"] == len(notes)
    assert result["count"] >= 2  # welcome.md + daily/2026-05-17.md
    assert "welcome.md" in notes
    assert "daily/2026-05-17.md" in notes


@pytest.mark.integration
def test_list_subfolder(temp_vault: Path) -> None:
    """Listing a subfolder returns only notes inside that folder."""
    result = obsidian_list_notes("daily")
    notes = result["notes"]
    assert "daily/2026-05-17.md" in notes
    # welcome.md must NOT appear because it is in the root
    assert "welcome.md" not in notes


@pytest.mark.integration
def test_search_finds_hit(temp_vault: Path) -> None:
    """Full-text search must find content written via the API."""
    # First confirm the unique keyword is absent from seeded notes
    result = obsidian_search("xyzunique999")
    assert result["count"] == 0

    # Write a note with the term then search again
    obsidian_write_note("search_target", "This contains the xyzunique999 keyword.")
    result = obsidian_search("xyzunique999")
    assert result["count"] >= 1
    assert any("xyzunique999" in m["snippet"].lower() for m in result["matches"])


@pytest.mark.integration
def test_search_daily_note_body(temp_vault: Path) -> None:
    """Search discovers text inside sub-folder notes."""
    result = obsidian_search("worked on")
    assert result["count"] >= 1
    paths = [m["path"] for m in result["matches"]]
    assert any("2026-05-17" in p for p in paths)


@pytest.mark.integration
def test_overwrite_existing_note(temp_vault: Path) -> None:
    """Overwriting a note with ``overwrite=True`` replaces the content."""
    obsidian_write_note("editable", "original content")
    result = obsidian_write_note("editable", "updated content", overwrite=True)
    assert result["created"] is False

    read_back = obsidian_read_note("editable")
    assert read_back["content"] == "updated content"


@pytest.mark.integration
def test_cannot_overwrite_without_flag(temp_vault: Path) -> None:
    """Writing to an existing note without overwrite=True must raise FileExistsError."""
    obsidian_write_note("protected", "original")
    with pytest.raises(FileExistsError):
        obsidian_write_note("protected", "replacement")


@pytest.mark.integration
def test_path_traversal_rejected(temp_vault: Path) -> None:
    """Paths containing ``..`` must raise ObsidianPathError, not traverse the vault."""
    with pytest.raises(ObsidianPathError):
        obsidian_read_note("../outside")


@pytest.mark.integration
def test_absolute_path_rejected(temp_vault: Path) -> None:
    """Absolute note paths must be rejected regardless of OS."""
    with pytest.raises(ObsidianPathError):
        obsidian_read_note("/etc/passwd")


@pytest.mark.integration
def test_write_nested_note_auto_mkdir(temp_vault: Path) -> None:
    """Writing a deeply nested note creates parent directories automatically."""
    result = obsidian_write_note("inbox/today/log", "# Log\n- entry 1")
    assert result["success"] is True
    note_path = temp_vault / "inbox" / "today" / "log.md"
    assert note_path.exists()
    assert note_path.read_text(encoding="utf-8") == "# Log\n- entry 1"


@pytest.mark.integration
def test_read_missing_note_raises(temp_vault: Path) -> None:
    """Reading a non-existent note must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        obsidian_read_note("does_not_exist")


@pytest.mark.integration
def test_no_token_safety_concern(temp_vault: Path) -> None:
    """Regression for #2759: real vault ops never return hardcoded fake data.

    Specifically, reads must return the actual file content — not a canned
    mock response — which would be dangerous if real tokens were configured.
    """
    unique_body = "# Real Data Test\n\nContent-ID: bfac3e2a-unique"
    obsidian_write_note("real_data_check", unique_body)
    result = obsidian_read_note("real_data_check")
    # The exact content written must come back — no interposition of mock data.
    assert result["content"] == unique_body
    # The path must be a real filesystem path (absolute, resolvable).
    resolved = Path(result["path"])
    assert resolved.is_absolute()
    assert resolved.exists()


@pytest.mark.integration
def test_no_vault_configured_raises() -> None:
    """Without a vault path configured, all operations must raise RuntimeError."""
    # Ensure nothing is configured for this test (reset module state).
    original = _obsidian_mod._OBSIDIAN_VAULT_PATH  # noqa: SLF001
    _obsidian_mod._OBSIDIAN_VAULT_PATH = None  # noqa: SLF001
    original_env = os.environ.pop("OBSIDIAN_VAULT_PATH", None)
    try:
        with pytest.raises(RuntimeError, match="not configured"):
            obsidian_list_notes()
    finally:
        _obsidian_mod._OBSIDIAN_VAULT_PATH = original  # noqa: SLF001
        if original_env is not None:
            os.environ["OBSIDIAN_VAULT_PATH"] = original_env


@pytest.mark.integration
def test_implementation_is_real_not_stub(temp_vault: Path) -> None:
    """Confirm the obsidian module contains a real implementation.

    The vault_client must not raise NotImplementedError — the issue #2938
    acceptance criterion is that Phase 2 replaced all NotImplementedError
    stubs with real local-filesystem operations.
    """
    # If any of these raise NotImplementedError, the implementation is still a stub.
    obsidian_write_note("impl_check", "test content")
    obsidian_read_note("impl_check")
    obsidian_list_notes()
    obsidian_search("test")
    # Reaching here means: no stubs, real implementation confirmed.
