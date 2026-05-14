"""Tests for Sidekick project file explorer file-opening actions."""

from __future__ import annotations

from pathlib import Path

import pytest

_QT_APP = None


class RecordingLauncher:
    def __init__(self, failure: OSError | None = None) -> None:
        self.failure = failure
        self.opened: list[Path] = []

    def open_file(self, path: Path) -> None:
        self.opened.append(path)
        if self.failure is not None:
            raise self.failure


def _qt_widgets():
    global _QT_APP
    try:
        from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets
    except ImportError:
        pytest.skip("Qt widgets unavailable")

    _QT_APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return QtWidgets


def _explorer(tmp_path: Path, launcher: RecordingLauncher | None = None):
    _qt_widgets()
    from upstream_drift_tools.ui.tools_sidebar.project_file_explorer import (
        ProjectFileExplorer,
    )

    return ProjectFileExplorer(
        project_root=tmp_path,
        default_program_launcher=launcher or RecordingLauncher(),
    )


def _index_for_path(explorer, path: Path):
    return explorer._model.index(str(path))


def _action_texts(menu) -> list[str]:
    return [action.text() for action in menu.actions()]


def test_default_open_action_is_offered_only_for_files(tmp_path: Path) -> None:
    explorer = _explorer(tmp_path)
    file_path = tmp_path / "case.py"
    file_path.write_text("print('case')\n", encoding="utf-8")
    directory = tmp_path / "folder"
    directory.mkdir()

    file_menu = explorer._context_menu_for_index(_index_for_path(explorer, file_path))
    directory_menu = explorer._context_menu_for_index(
        _index_for_path(explorer, directory)
    )

    assert file_menu is not None
    assert _action_texts(file_menu) == ["Open with Default Program"]
    assert directory_menu is None


def test_default_open_rejects_paths_outside_project_root(tmp_path: Path) -> None:
    launcher = RecordingLauncher()
    explorer = _explorer(tmp_path, launcher)
    outside_file = tmp_path.parent / "outside.txt"
    outside_file.write_text("outside\n", encoding="utf-8")

    explorer._open_with_default_program(_index_for_path(explorer, outside_file))

    assert launcher.opened == []


def test_default_open_uses_injected_launcher_for_project_files(tmp_path: Path) -> None:
    launcher = RecordingLauncher()
    explorer = _explorer(tmp_path, launcher)
    file_path = tmp_path / "case.txt"
    file_path.write_text("case\n", encoding="utf-8")

    explorer._open_with_default_program(_index_for_path(explorer, file_path))

    assert launcher.opened == [file_path.resolve()]


def test_default_open_failure_is_reported_without_crashing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    QtWidgets = _qt_widgets()
    launcher = RecordingLauncher(OSError("cannot launch"))
    explorer = _explorer(tmp_path, launcher)
    file_path = tmp_path / "case.txt"
    file_path.write_text("case\n", encoding="utf-8")
    failures: list[tuple[str, str]] = []
    warnings: list[tuple[str, str]] = []
    explorer.default_open_failed.connect(
        lambda path, error: failures.append((path, error))
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda _parent, title, message: warnings.append((title, message)),
    )

    explorer._open_with_default_program(_index_for_path(explorer, file_path))

    assert failures == [(str(file_path.resolve()), "cannot launch")]
    assert warnings == [("Open with Default Program Failed", "cannot launch")]


def test_double_click_file_open_signal_is_unchanged(tmp_path: Path) -> None:
    explorer = _explorer(tmp_path)
    file_path = tmp_path / "case.txt"
    file_path.write_text("case\n", encoding="utf-8")
    emitted: list[str] = []
    explorer.file_open_requested.connect(emitted.append)

    explorer._open_index(_index_for_path(explorer, file_path))

    assert emitted == [str(file_path.resolve())]
