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
from typing import Any

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


@pytest.mark.gui
def test_note_edit_dialog_new(qtbot) -> None:  # type: ignore[no-untyped-def]
    """_NoteEditDialog has default values when created without a card."""
    pytest.importorskip("PyQt6")
    from sidekick.notes_tab import _NoteEditDialog

    dlg = _NoteEditDialog()
    qtbot.addWidget(dlg)
    assert dlg.windowTitle() == "New Note"
    assert dlg.title_text == ""
    assert dlg.body_text == ""
    assert dlg.color.startswith("#")


@pytest.mark.gui
def test_note_edit_dialog_edit(qtbot) -> None:  # type: ignore[no-untyped-def]
    """_NoteEditDialog populates edit widgets when initialized with a card."""
    pytest.importorskip("PyQt6")
    from notes.models import NoteCard
    from sidekick.notes_tab import _NoteEditDialog

    card = NoteCard(
        note_id="123", title="Old Title", markdown_body="Old Body", color="#123456"
    )
    dlg = _NoteEditDialog(card=card)
    qtbot.addWidget(dlg)
    assert dlg.windowTitle() == "Edit Note"
    assert dlg.title_text == "Old Title"
    assert dlg.body_text == "Old Body"
    assert dlg.color == "#123456"


@pytest.mark.gui
def test_note_edit_dialog_pick_color(qtbot, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    """_pick_color updates the selected color when a valid color is chosen."""
    pytest.importorskip("PyQt6")
    from PyQt6.QtGui import QColor
    from PyQt6.QtWidgets import QColorDialog
    from sidekick.notes_tab import _NoteEditDialog

    dlg = _NoteEditDialog()
    qtbot.addWidget(dlg)

    mock_color = QColor("#ff0000")
    monkeypatch.setattr(QColorDialog, "getColor", lambda *args, **kwargs: mock_color)

    dlg._pick_color()
    assert dlg.color == "#ff0000"


@pytest.mark.gui
def test_note_edit_dialog_pick_color_invalid(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_pick_color retains the previous color if color selection is cancelled."""
    pytest.importorskip("PyQt6")
    from PyQt6.QtGui import QColor
    from PyQt6.QtWidgets import QColorDialog
    from sidekick.notes_tab import _NoteEditDialog

    dlg = _NoteEditDialog()
    qtbot.addWidget(dlg)
    initial_color = dlg.color

    mock_color = QColor()
    monkeypatch.setattr(QColorDialog, "getColor", lambda *args, **kwargs: mock_color)

    dlg._pick_color()
    assert dlg.color == initial_color


@pytest.mark.gui
def test_notes_tab_on_new_note_accepted(
    tmp_path: Path, qtbot: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_on_new_note creates a note and refreshes when dialog is accepted."""
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QDialog
    from sidekick.notes_tab import NotesTab

    tab = NotesTab(project_root=tmp_path)
    qtbot.addWidget(tab)

    class MockDialog:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def exec(self) -> int:
            return QDialog.DialogCode.Accepted

        @property
        def title_text(self) -> str:
            return "New Title"

        @property
        def body_text(self) -> str:
            return "New Body"

        @property
        def color(self) -> str:
            return "#ffffff"

    monkeypatch.setattr("sidekick.notes_tab._NoteEditDialog", MockDialog)

    signal_received = False

    def handle_changed() -> None:
        nonlocal signal_received
        signal_received = True

    tab.notes_changed.connect(handle_changed)

    tab._on_new_note()

    assert signal_received
    assert len(tab.note_ids()) == 1


@pytest.mark.gui
def test_notes_tab_on_new_note_rejected(
    tmp_path: Path, qtbot: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_on_new_note does not create a note when dialog is rejected."""
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QDialog
    from sidekick.notes_tab import NotesTab

    tab = NotesTab(project_root=tmp_path)
    qtbot.addWidget(tab)

    class MockDialog:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def exec(self) -> int:
            return QDialog.DialogCode.Rejected

        @property
        def title_text(self) -> str:
            return "New Title"

        @property
        def body_text(self) -> str:
            return "New Body"

        @property
        def color(self) -> str:
            return "#ffffff"

    monkeypatch.setattr("sidekick.notes_tab._NoteEditDialog", MockDialog)

    tab._on_new_note()
    assert len(tab.note_ids()) == 0


@pytest.mark.gui
def test_notes_tab_on_new_note_empty_title(
    tmp_path: Path, qtbot: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_on_new_note does not create a note when dialog has empty title."""
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QDialog
    from sidekick.notes_tab import NotesTab

    tab = NotesTab(project_root=tmp_path)
    qtbot.addWidget(tab)

    class MockDialog:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def exec(self) -> int:
            return QDialog.DialogCode.Accepted

        @property
        def title_text(self) -> str:
            return ""

        @property
        def body_text(self) -> str:
            return "New Body"

        @property
        def color(self) -> str:
            return "#ffffff"

    monkeypatch.setattr("sidekick.notes_tab._NoteEditDialog", MockDialog)

    tab._on_new_note()
    assert len(tab.note_ids()) == 0


@pytest.mark.gui
def test_notes_tab_on_edit_note_accepted(
    tmp_path: Path, qtbot: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_on_edit_note updates the note and refreshes when edit dialog is accepted."""
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QDialog
    from sidekick.notes_store import SidekickNotesStore
    from sidekick.notes_tab import NotesTab

    store = SidekickNotesStore(tmp_path)
    card = store.create_note("Original Title", "Original Body", color="#111111")

    tab = NotesTab(project_root=tmp_path)
    qtbot.addWidget(tab)

    class MockDialog:
        def __init__(self, card: Any = None, parent: Any = None) -> None:
            assert card is not None
            assert card.note_id == note_id_expected

        def exec(self) -> int:
            return QDialog.DialogCode.Accepted

        @property
        def title_text(self) -> str:
            return "Edited Title"

        @property
        def body_text(self) -> str:
            return "Edited Body"

        @property
        def color(self) -> str:
            return "#222222"

    note_id_expected = card.note_id
    monkeypatch.setattr("sidekick.notes_tab._NoteEditDialog", MockDialog)

    signal_received = False

    def handle_changed() -> None:
        nonlocal signal_received
        signal_received = True

    tab.notes_changed.connect(handle_changed)

    tab._on_edit_note(card.note_id)

    assert signal_received
    notes = store.list_notes()
    assert len(notes) == 1
    assert notes[0].title == "Edited Title"
    assert notes[0].markdown_body == "Edited Body"
    assert notes[0].color == "#222222"


@pytest.mark.gui
def test_notes_tab_on_edit_note_not_found(
    tmp_path: Path, qtbot: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """_on_edit_note logs a warning when the target note does not exist."""
    pytest.importorskip("PyQt6")
    from sidekick.notes_tab import NotesTab

    tab = NotesTab(project_root=tmp_path)
    qtbot.addWidget(tab)

    with caplog.at_level("WARNING"):
        tab._on_edit_note("missing-note-id")

    assert "Cannot edit: note" in caplog.text


@pytest.mark.gui
def test_notes_tab_on_delete_note(tmp_path: Path, qtbot: Any) -> None:
    """_on_delete_note deletes note, refreshes, and emits signal."""
    pytest.importorskip("PyQt6")
    from sidekick.notes_store import SidekickNotesStore
    from sidekick.notes_tab import NotesTab

    store = SidekickNotesStore(tmp_path)
    card = store.create_note("To Delete", "Body")

    tab = NotesTab(project_root=tmp_path)
    qtbot.addWidget(tab)
    assert len(tab.note_ids()) == 1

    signal_received = False

    def handle_changed() -> None:
        nonlocal signal_received
        signal_received = True

    tab.notes_changed.connect(handle_changed)

    tab._on_delete_note(card.note_id)

    assert signal_received
    assert len(tab.note_ids()) == 0


@pytest.mark.gui
def test_notes_tab_on_delete_note_not_found(
    tmp_path: Path, qtbot: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """_on_delete_note logs a warning when the note to delete does not exist."""
    pytest.importorskip("PyQt6")
    from sidekick.notes_tab import NotesTab

    tab = NotesTab(project_root=tmp_path)
    qtbot.addWidget(tab)

    with caplog.at_level("WARNING"):
        tab._on_delete_note("missing-note-id")

    assert "Cannot delete: note" in caplog.text


@pytest.mark.gui
def test_notes_tab_on_change_color_accepted(
    tmp_path: Path, qtbot: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_on_change_color updates note's color when valid color is picked."""
    pytest.importorskip("PyQt6")
    from PyQt6.QtGui import QColor
    from PyQt6.QtWidgets import QColorDialog
    from sidekick.notes_store import SidekickNotesStore
    from sidekick.notes_tab import NotesTab

    store = SidekickNotesStore(tmp_path)
    card = store.create_note("Title", "Body", color="#111111")

    tab = NotesTab(project_root=tmp_path)
    qtbot.addWidget(tab)

    mock_color = QColor("#333333")
    monkeypatch.setattr(QColorDialog, "getColor", lambda *args, **kwargs: mock_color)

    signal_received = False

    def handle_changed() -> None:
        nonlocal signal_received
        signal_received = True

    tab.notes_changed.connect(handle_changed)

    tab._on_change_color(card.note_id)

    assert signal_received
    notes = store.list_notes()
    assert notes[0].color == "#333333"


@pytest.mark.gui
def test_notes_tab_on_change_color_invalid(
    tmp_path: Path, qtbot: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_on_change_color ignores color updates when color is invalid."""
    pytest.importorskip("PyQt6")
    from PyQt6.QtGui import QColor
    from PyQt6.QtWidgets import QColorDialog
    from sidekick.notes_store import SidekickNotesStore
    from sidekick.notes_tab import NotesTab

    store = SidekickNotesStore(tmp_path)
    card = store.create_note("Title", "Body", color="#111111")

    tab = NotesTab(project_root=tmp_path)
    qtbot.addWidget(tab)

    mock_color = QColor()
    monkeypatch.setattr(QColorDialog, "getColor", lambda *args, **kwargs: mock_color)

    tab._on_change_color(card.note_id)
    notes = store.list_notes()
    assert notes[0].color == "#111111"


@pytest.mark.gui
def test_notes_tab_on_change_color_not_found(
    tmp_path: Path, qtbot: Any
) -> None:
    """_on_change_color returns early if the note does not exist."""
    pytest.importorskip("PyQt6")
    from sidekick.notes_tab import NotesTab

    tab = NotesTab(project_root=tmp_path)
    qtbot.addWidget(tab)

    tab._on_change_color("missing-note-id")


def test_render_markdown_with_mistune_mocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_render_markdown calls mistune.create_markdown when mistune is available."""
    import sidekick.notes_tab as nt_module

    monkeypatch.setattr(nt_module, "_MISTUNE_AVAILABLE", True)

    class MockMarkdown:
        def __init__(self, plugins: Any = None) -> None:
            pass

        def __call__(self, text: str) -> str:
            return f"MOCKED HTML: {text}"

    class MockMistune:
        @staticmethod
        def create_markdown(plugins: Any = None) -> MockMarkdown:
            return MockMarkdown(plugins)

    monkeypatch.setattr(nt_module, "mistune", MockMistune, raising=False)

    result = nt_module._render_markdown("hello")
    assert "MOCKED HTML: hello" in result
