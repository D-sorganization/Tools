"""PyQt6 user interface for the tile-based launcher."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import webbrowser
from collections.abc import Iterable
from pathlib import Path

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QIcon, QPalette, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from tile_launcher.manager import AppManager
from tile_launcher.models import AppDefinition, LaunchType

logger = logging.getLogger(__name__)


class SelectionDialog(QDialog):
    """Dialog that allows users to pick an app from a provided list."""

    def __init__(self, title: str, apps: Iterable[AppDefinition]) -> None:
        """Initialize the dialog with a title and list of apps."""
        super().__init__()
        self.setWindowTitle(title)
        self.setModal(True)
        self.selected_id: str | None = None

        layout = QVBoxLayout()
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        for app in apps:
            item = QListWidgetItem(f"{app.name} — {app.relative_path}")
            item.setData(Qt.ItemDataRole.UserRole, app.id)
            self.list_widget.addItem(item)
        layout.addWidget(self.list_widget)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def _on_accept(self) -> None:
        """Handle OK button click."""
        selected_items = self.list_widget.selectedItems()
        if not selected_items:
            QMessageBox.information(
                self, "Select an app", "Choose an app before continuing."
            )
            return

        self.selected_id = selected_items[0].data(Qt.ItemDataRole.UserRole)
        self.accept()


class LauncherWindow(QMainWindow):
    """Main window hosting the tile-based launcher experience."""

    def __init__(self, manager: AppManager) -> None:
        """Initialize the main window."""
        super().__init__()
        self.manager = manager
        self.edit_mode = False
        self.setWindowTitle("Tools Tile Launcher")
        self.resize(1100, 750)

        central_widget = QWidget()
        outer_layout = QVBoxLayout()
        central_widget.setLayout(outer_layout)
        self.setCentralWidget(central_widget)

        header = self._build_header()
        outer_layout.addLayout(header)

        self.tiles: QListWidget = QListWidget()
        self.tiles.setViewMode(QListWidget.ViewMode.IconMode)
        self.tiles.setMovement(QListView.Movement.Static)
        self.tiles.setIconSize(QSize(96, 96))
        self.tiles.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.tiles.setUniformItemSizes(True)
        self.tiles.setSpacing(18)
        self.tiles.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tiles.setDragEnabled(False)
        self.tiles.setAcceptDrops(False)
        self.tiles.setDropIndicatorShown(True)
        self.tiles.itemDoubleClicked.connect(self._launch_selected)
        if model := self.tiles.model():
            model.rowsMoved.connect(self._sync_layout_from_view)

        outer_layout.addWidget(self.tiles)
        self._refresh_tiles()
        self._apply_dark_theme()

    def _build_header(self) -> QHBoxLayout:
        """Create the header row with action buttons."""
        button_row = QHBoxLayout()

        add_button = QPushButton("Add Tile")
        add_button.clicked.connect(self._add_tile)
        button_row.addWidget(add_button)

        remove_button = QPushButton("Remove Tile")
        remove_button.clicked.connect(self._remove_tile)
        button_row.addWidget(remove_button)

        self.modify_button = QPushButton("Modify Layout")
        self.modify_button.setCheckable(True)
        self.modify_button.clicked.connect(self._toggle_edit_mode)
        button_row.addWidget(self.modify_button)

        reset_button = QPushButton("Reset Layout")
        reset_button.clicked.connect(self._reset_layout)
        button_row.addWidget(reset_button)

        button_row.addStretch(1)
        return button_row

    def _apply_dark_theme(self) -> None:
        """Apply a dark color scheme to the window."""
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, Qt.GlobalColor.black)
        palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Base, Qt.GlobalColor.black)
        palette.setColor(QPalette.ColorRole.AlternateBase, Qt.GlobalColor.darkGray)
        palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Button, Qt.GlobalColor.darkGray)
        palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Highlight, Qt.GlobalColor.gray)
        palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
        self.setPalette(palette)
        self.setStyleSheet(
            """
            QWidget { background-color: #121212; color: #f0f0f0; }
            QListWidget { background-color: #181818; border: 1px solid #2c2c2c; }
            QListWidget::item { border-radius: 12px; margin: 8px; padding: 12px; }
            QListWidget::item:selected { background-color: #2f4f6a; }
            QPushButton {
                background-color: #2a2a2a;
                color: #f0f0f0;
                padding: 8px 14px;
                border-radius: 6px;
            }
            QPushButton:hover { background-color: #3a3a3a; }
            QPushButton:checked { background-color: #2f4f6a; }
            """
        )

    def _refresh_tiles(self) -> None:
        """Reload the list of tiles from the manager."""
        self.tiles.clear()
        for app in self.manager.apps_in_layout():
            item = QListWidgetItem()
            item.setText(app.name)
            item.setData(Qt.ItemDataRole.UserRole, app.id)
            item.setIcon(self._icon_for_app(app))
            item.setSizeHint(self.tiles.iconSize())
            item.setToolTip(app.description or app.relative_path)
            self.tiles.addItem(item)

    def _icon_for_app(self, app: AppDefinition) -> QIcon:
        """Load the icon for the given app, or a fallback."""
        icon_path = self._logo_path(app)
        if icon_path and icon_path.exists():
            pixmap = QPixmap(str(icon_path)).scaled(
                96, 96, Qt.AspectRatioMode.KeepAspectRatio
            )
            return QIcon(pixmap)

        fallback = QPixmap(96, 96)
        fallback.fill(Qt.GlobalColor.darkCyan)
        return QIcon(fallback)

    def _logo_path(self, app: AppDefinition) -> Path | None:
        """Resolve the full path to the app's logo."""
        if not app.logo:
            return None
        return self.manager.repository_root / app.logo  # type: ignore[no-any-return]

    def _add_tile(self) -> None:
        """Show the dialog to add a new tile."""
        dialog = SelectionDialog("Add Tile", self.manager.available_to_add())
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.selected_id:
            self.manager.add_app(dialog.selected_id)
            self._refresh_tiles()

    def _remove_tile(self) -> None:
        """Show the dialog to remove an existing tile."""
        dialog = SelectionDialog("Remove Tile", self.manager.apps_in_layout())
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.selected_id:
            self.manager.remove_app(dialog.selected_id)
            self._refresh_tiles()

    def _toggle_edit_mode(self) -> None:
        """Toggle the layout modification mode (drag and drop)."""
        self.edit_mode = not self.edit_mode
        self.modify_button.setChecked(self.edit_mode)
        if self.edit_mode:
            self.tiles.setDragEnabled(True)
            self.tiles.setAcceptDrops(True)
            self.tiles.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
            self.tiles.setMovement(QListWidget.Movement.Free)
        else:
            self.tiles.setDragEnabled(False)
            self.tiles.setAcceptDrops(False)
            self.tiles.setDragDropMode(QAbstractItemView.DragDropMode.NoDragDrop)
            self.tiles.setMovement(QListWidget.Movement.Static)
            self._sync_layout_from_view()

    def _reset_layout(self) -> None:
        """Reset the layout to the default state."""
        self.manager.reset_layout()
        self._refresh_tiles()

    def _sync_layout_from_view(self) -> None:
        """Update the manager's layout order based on the current view."""
        ordered_ids: list[str] = []
        for index in range(self.tiles.count()):
            item = self.tiles.item(index)
            if item is not None:
                ordered_ids.append(item.data(Qt.ItemDataRole.UserRole))
        if ordered_ids:
            self.manager.reorder(ordered_ids)

    def _launch_selected(self, item: QListWidgetItem) -> None:
        """Launch the app corresponding to the clicked tile."""
        if self.edit_mode:
            return

        app_id = item.data(Qt.ItemDataRole.UserRole)
        app = self.manager.get_app(app_id)
        self._launch_app(app)

    def _launch_app(self, app: AppDefinition) -> None:
        """Execute the launch logic for the given app."""
        target_path = app.resolved_path(self.manager.repository_root)
        if not target_path.exists():
            QMessageBox.warning(
                self,
                "Missing Target",
                (
                    f"{app.name} could not be launched because the path does not "
                    f"exist:\n{target_path}"
                ),
            )
            return

        try:
            if app.launch_type == LaunchType.PYTHON:
                subprocess.Popen([sys.executable, str(target_path)])
            elif app.launch_type == LaunchType.BAT:
                self._launch_batch(target_path, app.name)
            elif app.launch_type == LaunchType.HTML:
                webbrowser.open(target_path.as_uri())
            elif app.launch_type == LaunchType.FILE:
                self._open_file(target_path)
            else:
                QMessageBox.warning(
                    self,
                    "Unsupported",
                    f"Unsupported launch type: {app.launch_type}",
                )
        except OSError as exc:  # pragma: no cover - OS-specific
            logger.exception("Failed to launch %s", app.name)
            QMessageBox.critical(
                self,
                "Launch Failed",
                f"Unable to launch {app.name}: {exc}",
            )

    @staticmethod
    def _open_file(target_path: Path) -> None:
        """Open a file using the system default handler."""
        if sys.platform.startswith("win"):
            os.startfile(target_path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(target_path)])
        else:
            subprocess.Popen(["xdg-open", str(target_path)])

    def _launch_batch(self, target_path: Path, app_name: str) -> None:
        """Launch a Windows batch file."""
        if sys.platform.startswith("win"):
            subprocess.Popen(["cmd", "/c", str(target_path)])
            return

        QMessageBox.information(
            self,
            "Windows Script",
            f"{app_name} is configured as a Windows batch file and can only run on Windows.",
        )


def run() -> None:
    """Entry point for launching the PyQt6 application."""

    app = QApplication(sys.argv)
    manager = AppManager.from_default_paths()
    window = LauncherWindow(manager)
    window.show()
    sys.exit(app.exec())
