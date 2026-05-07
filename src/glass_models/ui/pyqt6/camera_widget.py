"""PyQt6 widget for advanced camera controls and saved viewpoints.

Provides UI for:
- Standard view buttons (6x1 grid layout)
- Saved viewpoints list with Save/Load/Delete controls
- Orthographic/Perspective projection toggle
- Zoom to fit button

Signals:
- standard_view_selected(str): Emitted when a standard view button is clicked
- saved_viewpoint_selected(str): Emitted when a saved viewpoint is selected
- viewpoint_saved(str): Emitted when user saves current viewpoint
- viewpoint_deleted(str): Emitted when user deletes a viewpoint
- projection_changed(str): Emitted when orthographic/perspective toggled
- zoom_to_fit_requested(): Emitted when zoom to fit is clicked
"""

import logging

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

# Standard view names
STANDARD_VIEWS = ["Top", "Bottom", "Front", "Back", "Left", "Right", "Isometric"]


class CameraControlWidget(QWidget):
    """Control widget for camera and viewpoint management.

    Provides buttons for standard views, saved viewpoints list, and
    projection mode selection.

    Signals:
        standard_view_selected: str - name of standard view clicked
        saved_viewpoint_selected: str - name of saved viewpoint selected
        viewpoint_saved: str - name of viewpoint user wants to save
        viewpoint_deleted: str - name of viewpoint to delete
        projection_changed: str - "orthographic" or "perspective"
        zoom_to_fit_requested: no args - zoom to fit button clicked
    """

    standard_view_selected = pyqtSignal(str)
    saved_viewpoint_selected = pyqtSignal(str)
    viewpoint_saved = pyqtSignal(str)
    viewpoint_deleted = pyqtSignal(str)
    projection_changed = pyqtSignal(str)
    zoom_to_fit_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize camera control widget.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self._setup_ui()
        self._connect_signals()
        logger.debug("CameraControlWidget initialized")

    def _setup_ui(self) -> None:
        """Set up the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Standard views group
        standard_group = QGroupBox("Standard Views", self)
        standard_layout = QVBoxLayout()

        # Top 3 views
        top_row_layout = QHBoxLayout()
        self.top_button = QPushButton("Top")
        self.bottom_button = QPushButton("Bottom")
        self.front_button = QPushButton("Front")
        top_row_layout.addWidget(self.top_button)
        top_row_layout.addWidget(self.bottom_button)
        top_row_layout.addWidget(self.front_button)
        standard_layout.addLayout(top_row_layout)

        # Bottom 3 views
        bottom_row_layout = QHBoxLayout()
        self.back_button = QPushButton("Back")
        self.left_button = QPushButton("Left")
        self.right_button = QPushButton("Right")
        bottom_row_layout.addWidget(self.back_button)
        bottom_row_layout.addWidget(self.left_button)
        bottom_row_layout.addWidget(self.right_button)
        standard_layout.addLayout(bottom_row_layout)

        # Isometric
        iso_layout = QHBoxLayout()
        self.isometric_button = QPushButton("Isometric")
        iso_layout.addWidget(self.isometric_button)
        iso_layout.addStretch()
        standard_layout.addLayout(iso_layout)

        standard_group.setLayout(standard_layout)
        layout.addWidget(standard_group)

        # Saved viewpoints group
        saved_group = QGroupBox("Saved Viewpoints", self)
        saved_layout = QVBoxLayout()

        # List widget
        self.viewpoint_list = QListWidget()
        self.viewpoint_list.setMinimumHeight(120)
        saved_layout.addWidget(QLabel("Viewpoints:"))
        saved_layout.addWidget(self.viewpoint_list)

        # Control buttons
        button_layout = QHBoxLayout()
        self.save_button = QPushButton("Save...")
        self.load_button = QPushButton("Load")
        self.delete_button = QPushButton("Delete")
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.delete_button)
        saved_layout.addLayout(button_layout)

        saved_group.setLayout(saved_layout)
        layout.addWidget(saved_group)

        # Projection and misc group
        projection_group = QGroupBox("View Options", self)
        projection_layout = QVBoxLayout()

        # Projection mode
        proj_mode_layout = QHBoxLayout()
        proj_mode_layout.addWidget(QLabel("Projection:"))
        self.orthographic_check = QCheckBox("Orthographic")
        self.perspective_check = QCheckBox("Perspective")
        self.perspective_check.setChecked(True)
        proj_mode_layout.addWidget(self.orthographic_check)
        proj_mode_layout.addWidget(self.perspective_check)
        proj_mode_layout.addStretch()
        projection_layout.addLayout(proj_mode_layout)

        # Zoom to fit
        zoom_layout = QHBoxLayout()
        self.zoom_fit_button = QPushButton("Zoom to Fit")
        zoom_layout.addWidget(self.zoom_fit_button)
        zoom_layout.addStretch()
        projection_layout.addLayout(zoom_layout)

        projection_group.setLayout(projection_layout)
        layout.addWidget(projection_group)

        layout.addStretch()

    def _connect_signals(self) -> None:
        """Connect widget signals to slots."""
        # Standard view buttons
        self.top_button.clicked.connect(self._on_top_clicked)
        self.bottom_button.clicked.connect(self._on_bottom_clicked)
        self.front_button.clicked.connect(self._on_front_clicked)
        self.back_button.clicked.connect(self._on_back_clicked)
        self.left_button.clicked.connect(self._on_left_clicked)
        self.right_button.clicked.connect(self._on_right_clicked)
        self.isometric_button.clicked.connect(self._on_isometric_clicked)

        # Saved viewpoints
        self.save_button.clicked.connect(self._on_save_viewpoint)
        self.load_button.clicked.connect(self._on_load_viewpoint)
        self.delete_button.clicked.connect(self._on_delete_viewpoint)
        self.viewpoint_list.itemDoubleClicked.connect(self._on_viewpoint_double_clicked)

        # Projection
        self.orthographic_check.stateChanged.connect(self._on_projection_changed)
        self.perspective_check.stateChanged.connect(self._on_projection_changed)

        # Zoom
        self.zoom_fit_button.clicked.connect(self.zoom_to_fit_requested.emit)

    def _on_top_clicked(self) -> None:
        """Handle Top button click."""
        logger.debug("Top view selected")
        self.standard_view_selected.emit("Top")

    def _on_bottom_clicked(self) -> None:
        """Handle Bottom button click."""
        logger.debug("Bottom view selected")
        self.standard_view_selected.emit("Bottom")

    def _on_front_clicked(self) -> None:
        """Handle Front button click."""
        logger.debug("Front view selected")
        self.standard_view_selected.emit("Front")

    def _on_back_clicked(self) -> None:
        """Handle Back button click."""
        logger.debug("Back view selected")
        self.standard_view_selected.emit("Back")

    def _on_left_clicked(self) -> None:
        """Handle Left button click."""
        logger.debug("Left view selected")
        self.standard_view_selected.emit("Left")

    def _on_right_clicked(self) -> None:
        """Handle Right button click."""
        logger.debug("Right view selected")
        self.standard_view_selected.emit("Right")

    def _on_isometric_clicked(self) -> None:
        """Handle Isometric button click."""
        logger.debug("Isometric view selected")
        self.standard_view_selected.emit("Isometric")

    def _on_save_viewpoint(self) -> None:
        """Handle Save button click."""
        text, ok = QInputDialog.getText(
            self,
            "Save Viewpoint",
            "Viewpoint name:",
            text="My View",
        )
        if ok and text:
            logger.debug("Saving viewpoint: %s", text)
            self.viewpoint_saved.emit(text)

    def _on_load_viewpoint(self) -> None:
        """Handle Load button click."""
        current_item = self.viewpoint_list.currentItem()
        if current_item:
            name = current_item.text()
            logger.debug("Loading viewpoint: %s", name)
            self.saved_viewpoint_selected.emit(name)
        else:
            logger.debug("No viewpoint selected for loading")

    def _on_delete_viewpoint(self) -> None:
        """Handle Delete button click."""
        current_item = self.viewpoint_list.currentItem()
        if current_item:
            name = current_item.text()
            logger.debug("Deleting viewpoint: %s", name)
            self.viewpoint_deleted.emit(name)
            # Remove from list (caller should update via update_viewpoint_list)
            self.viewpoint_list.takeItem(self.viewpoint_list.row(current_item))
        else:
            logger.debug("No viewpoint selected for deletion")

    def _on_viewpoint_double_clicked(self, item: QListWidgetItem) -> None:
        """Handle viewpoint list item double-click."""
        name = item.text()
        logger.debug("Double-clicked viewpoint: %s", name)
        self.saved_viewpoint_selected.emit(name)

    def _on_projection_changed(self) -> None:
        """Handle projection mode change."""
        if self.orthographic_check.isChecked():
            logger.debug("Switched to orthographic projection")
            self.perspective_check.blockSignals(True)
            self.perspective_check.setChecked(False)
            self.perspective_check.blockSignals(False)
            self.projection_changed.emit("orthographic")
        elif self.perspective_check.isChecked():
            logger.debug("Switched to perspective projection")
            self.orthographic_check.blockSignals(True)
            self.orthographic_check.setChecked(False)
            self.orthographic_check.blockSignals(False)
            self.projection_changed.emit("perspective")

    def update_viewpoint_list(self, viewpoints: list[str]) -> None:
        """Update the saved viewpoints list.

        Args:
            viewpoints: List of viewpoint names to display
        """
        self.viewpoint_list.clear()
        for name in sorted(viewpoints):
            self.viewpoint_list.addItem(name)
        logger.debug("Updated viewpoint list with %d items", len(viewpoints))

    def get_projection_mode(self) -> str:
        """Get current projection mode.

        Returns:
            "orthographic" or "perspective"
        """
        if self.orthographic_check.isChecked():
            return "orthographic"
        return "perspective"

    def set_projection_mode(self, mode: str) -> None:
        """Set projection mode.

        Args:
            mode: "orthographic" or "perspective"
        """
        if mode == "orthographic":
            self.orthographic_check.blockSignals(True)
            self.perspective_check.blockSignals(True)
            self.orthographic_check.setChecked(True)
            self.perspective_check.setChecked(False)
            self.orthographic_check.blockSignals(False)
            self.perspective_check.blockSignals(False)
            logger.debug("Set projection to orthographic")
        elif mode == "perspective":
            self.orthographic_check.blockSignals(True)
            self.perspective_check.blockSignals(True)
            self.orthographic_check.setChecked(False)
            self.perspective_check.setChecked(True)
            self.orthographic_check.blockSignals(False)
            self.perspective_check.blockSignals(False)
            logger.debug("Set projection to perspective")
