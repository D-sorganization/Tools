"""Tests for sidekick.notes_tab and sidekick.notes_store (issue #2931).

TDD: these tests drove the implementation of notes_tab.py and notes_store.py.

Tests cover:
- SidekickNotesStore CRUD and color persistence
- NotesTab widget creation, card display, and persistence across simulated
  session restart (store re-construction)
- Markdown rendering (basic CommonMark)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SHARED = Path(__file__).resolve().parents[4] / "src" / "shared" / "python"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# SidekickNotesStore — pure-Python tests (no Qt required)
# ---------------------------------------------------------------------------


def test_notes_store_create_and_list(tmp_path: Path) -> None:
    """create_note persists a card; list_notes returns it."""
    from sidekick.notes_store import SidekickNotesStore

    store = SidekickNotesStore(tmp_path)
    card = store.create_note("Hello", "# Hello world")
    assert card.title == "Hello"
    assert card.markdown_body == "# Hello world"
    notes = store.list_notes()
    assert len(notes) == 1
    assert notes[0].note_id == card.note_id


def test_notes_store_default_color(tmp_path: Path) -> None:
    """Notes get a default color when none is specified."""
    from sidekick.notes_store import SidekickNotesStore

    store = SidekickNotesStore(tmp_path)
    card = store.create_note("Default color")
    assert card.color.startswith("#")


def test_notes_store_custom_color(tmp_path: Path) -> None:
    """Notes support a custom background color."""
    from sidekick.notes_store import SidekickNotesStore

    store = SidekickNotesStore(tmp_path)
    card = store.create_note("Colored", color="#ffaacc")
    assert card.color == "#ffaacc"


def test_notes_store_color_persists(tmp_path: Path) -> None:
    """Color is preserved when reloading the note from disk."""
    from sidekick.notes_store import SidekickNotesStore

    store = SidekickNotesStore(tmp_path)
    original = store.create_note("Persist color", color="#aabbcc")
    reloaded = store.load_note(original.note_id)
    assert reloaded is not None
    assert reloaded.color == "#aabbcc"


def test_notes_store_update(tmp_path: Path) -> None:
    """update_note changes title, body, and color."""
    from sidekick.notes_store import SidekickNotesStore

    store = SidekickNotesStore(tmp_path)
    card = store.create_note("Original", "original body")
    updated = store.update_note(
        card.note_id, title="Updated", body="updated body", color="#112233"
    )
    assert updated.title == "Updated"
    assert updated.markdown_body == "updated body"
    assert updated.color == "#112233"


def test_notes_store_delete(tmp_path: Path) -> None:
    """delete_note removes the card from list_notes."""
    from sidekick.notes_store import SidekickNotesStore

    store = SidekickNotesStore(tmp_path)
    card = store.create_note("To delete")
    assert len(store.list_notes()) == 1
    result = store.delete_note(card.note_id)
    assert result is True
    assert len(store.list_notes()) == 0


def test_notes_store_delete_missing_returns_false(tmp_path: Path) -> None:
    """Deleting a non-existent note returns False without raising."""
    from sidekick.notes_store import SidekickNotesStore

    store = SidekickNotesStore(tmp_path)
    result = store.delete_note("nonexistent-note-id1")
    assert result is False


def test_notes_store_precondition_empty_title(tmp_path: Path) -> None:
    """create_note with an empty title raises ValueError."""
    from sidekick.notes_store import SidekickNotesStore

    store = SidekickNotesStore(tmp_path)
    with pytest.raises(ValueError, match="title must not be empty"):
        store.create_note("")


def test_notes_store_precondition_bad_root() -> None:
    """Constructing store with a non-existent root raises ValueError."""
    from sidekick.notes_store import SidekickNotesStore

    with pytest.raises(ValueError, match="project_root must exist"):
        SidekickNotesStore("/this/path/does/not/exist/ever")


def test_notes_store_persistence_across_restart(tmp_path: Path) -> None:
    """Notes created in one store instance are visible in a new instance."""
    from sidekick.notes_store import SidekickNotesStore

    store1 = SidekickNotesStore(tmp_path)
    card = store1.create_note("Persisted", "# Persistent body")

    # Simulate session restart: create a new store from the same path
    store2 = SidekickNotesStore(tmp_path)
    notes = store2.list_notes()
    assert len(notes) == 1
    assert notes[0].note_id == card.note_id
    assert notes[0].markdown_body == "# Persistent body"


# ---------------------------------------------------------------------------
# Markdown rendering helper — no Qt required
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("md_input", "expected_fragment"),
    [
        ("# Hello", "<h1>"),
        ("**bold**", "bold"),
        ("- item one", "item one"),
    ],
)
def test_render_markdown_html_output(md_input: str, expected_fragment: str) -> None:
    """_render_markdown produces HTML containing expected fragments."""
    pytest.importorskip("mistune")
    from sidekick.notes_tab import _render_markdown

    html = _render_markdown(md_input)
    assert expected_fragment in html


def test_render_markdown_fallback_without_mistune(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_render_markdown falls back to <pre> when mistune is absent."""
    import sidekick.notes_tab as nt_module

    monkeypatch.setattr(nt_module, "_MISTUNE_AVAILABLE", False)
    result = nt_module._render_markdown("hello")
    assert "<pre>" in result


# ---------------------------------------------------------------------------
# NotesTab Qt widget tests (require PyQt6)
# ---------------------------------------------------------------------------


@pytest.mark.gui
def test_notes_tab_creates_without_error(tmp_path: Path, qtbot) -> None:  # type: ignore[no-untyped-def]
    """NotesTab can be instantiated and added to qtbot."""
    pytest.importorskip("PyQt6")
    from sidekick.notes_tab import NotesTab

    tab = NotesTab(project_root=tmp_path)
    qtbot.addWidget(tab)
    assert tab.objectName() == "SidekickNotesTab"


@pytest.mark.gui
def test_notes_tab_shows_empty_label_when_no_notes(tmp_path: Path, qtbot) -> None:  # type: ignore[no-untyped-def]
    """Empty notes tab shows a helpful empty-state label."""
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QLabel
    from sidekick.notes_tab import NotesTab

    tab = NotesTab(project_root=tmp_path)
    qtbot.addWidget(tab)
    label = tab.findChild(QLabel, "NotesTabEmptyLabel")
    assert label is not None


@pytest.mark.gui
def test_notes_tab_new_button_exists(tmp_path: Path, qtbot) -> None:  # type: ignore[no-untyped-def]
    """NotesTab has a '+ New Note' button."""
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QPushButton
    from sidekick.notes_tab import NotesTab

    tab = NotesTab(project_root=tmp_path)
    qtbot.addWidget(tab)
    btn = tab.findChild(QPushButton, "NotesTabNewButton")
    assert btn is not None
    assert "New Note" in btn.text()


@pytest.mark.gui
def test_notes_tab_displays_card_after_store_insert(tmp_path: Path, qtbot) -> None:  # type: ignore[no-untyped-def]
    """NotesTab shows a card widget for each stored note."""
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QWidget
    from sidekick.notes_store import SidekickNotesStore
    from sidekick.notes_tab import NotesTab

    store = SidekickNotesStore(tmp_path)
    card = store.create_note("My Note", "## Body")

    tab = NotesTab(project_root=tmp_path)
    qtbot.addWidget(tab)

    card_widget = tab.findChild(QWidget, f"NoteCard_{card.note_id}")
    assert card_widget is not None


@pytest.mark.gui
def test_notes_tab_card_has_color_button(tmp_path: Path, qtbot) -> None:  # type: ignore[no-untyped-def]
    """Each card widget contains a color-picker button."""
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QPushButton
    from sidekick.notes_store import SidekickNotesStore
    from sidekick.notes_tab import NotesTab

    store = SidekickNotesStore(tmp_path)
    card = store.create_note("Colored Note", color="#aabbcc")

    tab = NotesTab(project_root=tmp_path)
    qtbot.addWidget(tab)

    color_btn = tab.findChild(QPushButton, f"NoteCardColor_{card.note_id}")
    assert color_btn is not None


@pytest.mark.gui
def test_notes_tab_persistence_across_session_restart(tmp_path: Path, qtbot) -> None:  # type: ignore[no-untyped-def]
    """Notes created via store persist when NotesTab is reconstructed."""
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QWidget
    from sidekick.notes_store import SidekickNotesStore
    from sidekick.notes_tab import NotesTab

    # Write a note
    store = SidekickNotesStore(tmp_path)
    card = store.create_note("Restart note", "persisted body")

    # Create tab in new "session" (new object, same directory)
    tab2 = NotesTab(project_root=tmp_path)
    qtbot.addWidget(tab2)

    card_widget = tab2.findChild(QWidget, f"NoteCard_{card.note_id}")
    assert card_widget is not None
