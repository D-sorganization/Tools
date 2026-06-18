from pathlib import Path

import pytest
from PyQt6.QtCore import Qt

from notes import integration


class FakeMainWindow:
    def __init__(self) -> None:
        self.added_docks: list[tuple[Qt.DockWidgetArea, object]] = []

    def addDockWidget(self, area: Qt.DockWidgetArea, dock: object) -> None:
        self.added_docks.append((area, dock))


class FakeNotesDockWidget:
    instances: list["FakeNotesDockWidget"] = []

    def __init__(
        self,
        *,
        project_dir: Path | str,
        title: str,
        parent: object,
    ) -> None:
        self.project_dir = project_dir
        self.title = title
        self.parent = parent
        self.instances.append(self)


def test_attach_notes_dock_constructs_and_adds_dock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeNotesDockWidget.instances.clear()
    monkeypatch.setattr(integration, "NotesDockWidget", FakeNotesDockWidget)
    main_window = FakeMainWindow()
    project_dir = Path("project")

    dock = integration.attach_notes_dock(
        main_window,
        project_dir,
        area=Qt.DockWidgetArea.LeftDockWidgetArea,
        title="Project Notes",
    )

    assert dock is FakeNotesDockWidget.instances[0]
    assert dock.project_dir == project_dir
    assert dock.title == "Project Notes"
    assert dock.parent is main_window
    assert main_window.added_docks == [(Qt.DockWidgetArea.LeftDockWidgetArea, dock)]


def test_attach_notes_dock_uses_right_area_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeNotesDockWidget.instances.clear()
    monkeypatch.setattr(integration, "NotesDockWidget", FakeNotesDockWidget)
    main_window = FakeMainWindow()

    dock = integration.attach_notes_dock(main_window, "project")

    assert main_window.added_docks == [(Qt.DockWidgetArea.RightDockWidgetArea, dock)]


def test_attach_notes_dock_requires_add_dock_widget() -> None:
    with pytest.raises(ValueError, match="main_window must support addDockWidget"):
        integration.attach_notes_dock(object(), "project")
