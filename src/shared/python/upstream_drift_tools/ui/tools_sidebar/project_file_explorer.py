"""Project-scoped file explorer widget for the unified tools sidebar."""

from __future__ import annotations

from pathlib import Path

from .design_tokens import (
    SIDEKICK_PROJECT_EXPLORER_OBJECT_NAME,
    SIDEKICK_PROJECT_TREE_OBJECT_NAME,
)
from .qt_compat import FileSystemModel, QtCore, QtWidgets, Signal


class ProjectFileExplorer(QtWidgets.QWidget):
    """Read-only project file explorer with an open-file signal."""

    file_open_requested = Signal(str)

    def __init__(
        self,
        project_root: str | Path | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName(SIDEKICK_PROJECT_EXPLORER_OBJECT_NAME)
        self._project_root = Path.cwd()
        self._model = FileSystemModel(self)
        self._model.setReadOnly(True)
        self._tree = QtWidgets.QTreeView(self)
        self._tree.setObjectName(SIDEKICK_PROJECT_TREE_OBJECT_NAME)
        self._tree.setModel(self._model)
        self._tree.doubleClicked.connect(self._open_index)

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
        path = Path(self._model.filePath(index)).resolve()
        if path.is_file() and self._is_within_project(path):
            self.file_open_requested.emit(str(path))

    def _is_within_project(self, path: Path) -> bool:
        try:
            path.relative_to(self._project_root)
        except ValueError:
            return False
        return True
