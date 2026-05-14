"""Project-scoped file explorer widget for the unified tools sidebar."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Protocol

from .design_tokens import (
    SIDEKICK_PROJECT_EXPLORER_OBJECT_NAME,
    SIDEKICK_PROJECT_TREE_OBJECT_NAME,
)
from .qt_compat import FileSystemModel, QtCore, QtWidgets, Signal


class DefaultProgramLauncher(Protocol):
    """Launch files through the operating system default application."""

    def open_file(self, path: Path) -> None:
        """Open ``path`` with the operating system default application."""


class WindowsDefaultProgramLauncher:
    """Default launcher for Windows file associations."""

    def open_file(self, path: Path) -> None:
        """Open ``path`` with the Windows default application."""
        if sys.platform != "win32":
            raise RuntimeError(
                "Opening with the default program is unavailable on this platform."
            )
        os.startfile(str(path))  # type: ignore[attr-defined]


class ProjectFileExplorer(QtWidgets.QWidget):
    """Read-only project file explorer with an open-file signal."""

    file_open_requested = Signal(str)
    default_open_failed = Signal(str, str)

    def __init__(
        self,
        project_root: str | Path | None = None,
        default_program_launcher: DefaultProgramLauncher | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName(SIDEKICK_PROJECT_EXPLORER_OBJECT_NAME)
        self._project_root = Path.cwd()
        self._default_program_launcher = (
            default_program_launcher or WindowsDefaultProgramLauncher()
        )
        self._model = FileSystemModel(self)
        self._model.setReadOnly(True)
        self._tree = QtWidgets.QTreeView(self)
        self._tree.setObjectName(SIDEKICK_PROJECT_TREE_OBJECT_NAME)
        self._tree.setModel(self._model)
        self._tree.doubleClicked.connect(self._open_index)
        self._tree.setContextMenuPolicy(_custom_context_menu_policy())
        self._tree.customContextMenuRequested.connect(self._show_context_menu)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._tree)

        self.set_project_root(project_root or Path.cwd())

    @property
    def project_root(self) -> Path:
        """Return the scoped project root."""
        return self._project_root

    def set_project_root(self, project_root: str | Path) -> None:
        """Scope the explorer to ``project_root``."""
        root = Path(project_root).expanduser().resolve()
        if not root.exists() or not root.is_dir():
            raise ValueError(f"Project root is not a directory: {root}")
        self._project_root = root
        root_index = self._model.setRootPath(str(root))
        self._tree.setRootIndex(root_index)
        self._tree.setColumnWidth(0, 240)
        for column in range(1, self._model.columnCount()):
            self._tree.hideColumn(column)

    def _open_index(self, index: QtCore.QModelIndex) -> None:
        path = self._path_for_index(index)
        if path is not None and self._can_open_file(path):
            self.file_open_requested.emit(str(path))

    def _show_context_menu(self, pos: QtCore.QPoint) -> None:
        menu = self._context_menu_for_index(self._tree.indexAt(pos))
        if menu is not None:
            menu.exec(self._tree.viewport().mapToGlobal(pos))

    def _context_menu_for_index(
        self, index: QtCore.QModelIndex
    ) -> QtWidgets.QMenu | None:
        path = self._path_for_index(index)
        if path is None or not self._can_open_file(path):
            return None

        menu = QtWidgets.QMenu(self)
        menu.addAction("Open with Default Program").triggered.connect(
            lambda: self._open_with_default_program(index)
        )
        return menu

    def _open_with_default_program(self, index: QtCore.QModelIndex) -> None:
        path = self._path_for_index(index)
        if path is None or not self._can_open_file(path):
            return

        try:
            self._default_program_launcher.open_file(path)
        except (OSError, RuntimeError) as exc:
            message = str(exc)
            self.default_open_failed.emit(str(path), message)
            QtWidgets.QMessageBox.warning(
                self,
                "Open with Default Program Failed",
                message,
            )

    def _path_for_index(self, index: QtCore.QModelIndex) -> Path | None:
        if not index.isValid():
            return None
        return Path(self._model.filePath(index)).resolve()

    def _can_open_file(self, path: Path) -> bool:
        return path.is_file() and self._is_within_project(path)

    def _is_within_project(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self._project_root)
        except ValueError:
            return False
        return True


def _custom_context_menu_policy() -> QtCore.Qt.ContextMenuPolicy:
    policy_type = getattr(QtCore.Qt, "ContextMenuPolicy", None)
    if policy_type is not None:
        return policy_type.CustomContextMenu
    return QtCore.Qt.CustomContextMenu
