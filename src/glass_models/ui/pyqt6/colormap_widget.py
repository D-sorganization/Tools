"""PyQt6 widget for colormap selection and editing (GitHub issue #545).

This module provides a complete UI for managing colormaps:
- ColormapSelectorWidget: Dropdown selection of 20+ colormaps
- Preview canvas with matplotlib integration
- Custom colormap editor with color stop management
- Save/load custom colormaps

Features:
- Real-time colormap preview
- Add/remove color stops
- Drag-and-drop color selection
- Custom colormap persistence
- B&W preview for print-friendly validation
- Colorblind-friendly indicators
"""

from __future__ import annotations

import logging
from collections.abc import Callable

try:
    from PyQt6.QtCore import Qt, pyqtSignal
    from PyQt6.QtWidgets import (
        QColorDialog,
        QComboBox,
        QDialog,
        QFrame,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QListWidgetItem,
        QMessageBox,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    HAS_MATPLOTLIB_QT = True
except ImportError:
    HAS_MATPLOTLIB_QT = False

from glass_models.viz.colormaps import ColormapManager

logger = logging.getLogger(__name__)


class ColormapPreviewCanvas(QFrame if HAS_PYQT6 else object):
    """Canvas for displaying colormap preview with matplotlib.

    Displays a horizontal bar showing the colormap gradient.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize preview canvas.

        Args:
            parent: Parent widget.

        Raises:
            ImportError: If matplotlib integration not available.
        """
        if not HAS_PYQT6 or not HAS_MATPLOTLIB_QT:
            raise ImportError(
                "PyQt6 and matplotlib with Qt5 backend required. "
                "Install with: pip install PyQt6 matplotlib"
            )

        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Sunken)
        self.setMinimumHeight(80)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.figure = Figure(figsize=(6, 0.5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self._cmap: Callable[[float], tuple[float, ...]] | None = None

    def set_colormap(self, cmap: Callable[[float], tuple[float, ...]]) -> None:
        """Set and display a colormap.

        Args:
            cmap: Callable colormap mapping [0, 1] -> RGBA.
        """
        self._cmap = cmap

        # Clear previous plot
        self.figure.clear()

        # Create gradient preview
        import numpy as np

        ax = self.figure.add_axes([0.1, 0.3, 0.8, 0.4])
        gradient = np.linspace(0, 1, 256).reshape(1, -1)

        # Convert colormap to matplotlib-compatible format
        def cmap_to_matplotlib(x: np.ndarray) -> np.ndarray:  # type: ignore[name-defined]
            """Convert colormap to matplotlib array."""
            result = np.zeros((*x.shape, 4))
            for i in range(x.shape[1]):
                val = float(x[0, i])
                rgba = cmap(val)
                result[0, i] = rgba
            return result

        ax.imshow(cmap_to_matplotlib(gradient), aspect="auto", extent=[0, 1, 0, 1])
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_yticks([])
        ax.set_xlabel("Value")

        self.canvas.draw()


class ColormapSelectorWidget(QWidget if HAS_PYQT6 else object):
    """Widget for selecting and previewing colormaps.

    Features:
    - Dropdown selection of 20+ built-in colormaps
    - Live preview canvas
    - Categories filter (sequential, diverging, categorical)
    - Colorblind-friendly indicators
    - B&W preview option
    - Custom colormap editor button

    Signals:
        colormap_changed: Emitted when user selects a different colormap.
                         Argument: (colormap_name: str)
    """

    colormap_changed = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:  # type: ignore[name-defined]
        """Initialize colormap selector widget.

        Args:
            parent: Parent widget.

        Raises:
            ImportError: If PyQt6 not available.
        """
        if not HAS_PYQT6:
            raise ImportError(
                "PyQt6 required for ColormapSelectorWidget. "
                "Install with: pip install PyQt6"
            )

        super().__init__(parent)

        self.manager = ColormapManager()
        self._current_colormap_name = "viridis"
        self._show_bw = False

        self._setup_ui()
        self._populate_colormaps()

    def _setup_ui(self) -> None:
        """Set up the UI layout."""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("Colormap Selector")
        title.setStyleSheet("font-weight: bold; font-size: 12pt;")
        layout.addWidget(title)

        # Colormap selection
        selection_layout = QHBoxLayout()
        selection_layout.addWidget(QLabel("Colormap:"))

        self.colormap_combo = QComboBox()
        self.colormap_combo.currentTextChanged.connect(self._on_colormap_selected)
        selection_layout.addWidget(self.colormap_combo)

        self.colorblind_indicator = QLabel()
        selection_layout.addWidget(self.colorblind_indicator)

        layout.addLayout(selection_layout)

        # Category filter
        category_layout = QHBoxLayout()
        category_layout.addWidget(QLabel("Category:"))

        self.category_combo = QComboBox()
        self.category_combo.addItem("All")
        for category in self.manager.list_colormap_categories():
            self.category_combo.addItem(category.title())
        self.category_combo.currentTextChanged.connect(self._on_category_changed)
        category_layout.addWidget(self.category_combo)

        layout.addLayout(category_layout)

        # Preview canvas
        preview_label = QLabel("Preview:")
        layout.addWidget(preview_label)

        self.preview_canvas = ColormapPreviewCanvas()
        layout.addWidget(self.preview_canvas)

        # B&W preview checkbox
        bw_layout = QHBoxLayout()
        self.bw_button = QPushButton("Show B&W Preview")
        self.bw_button.setCheckable(True)
        self.bw_button.toggled.connect(self._on_bw_toggled)
        bw_layout.addWidget(self.bw_button)
        bw_layout.addStretch()
        layout.addLayout(bw_layout)

        # Custom colormap editor
        editor_layout = QHBoxLayout()
        self.custom_editor_button = QPushButton("Edit Custom Colormap")
        self.custom_editor_button.clicked.connect(self._open_custom_editor)
        editor_layout.addWidget(self.custom_editor_button)
        editor_layout.addStretch()
        layout.addLayout(editor_layout)

        layout.addStretch()

    def _populate_colormaps(self) -> None:
        """Populate colormap dropdown with all available colormaps."""
        colormaps = self.manager.list_colormaps()
        for name in sorted(colormaps):
            self.colormap_combo.addItem(name)

        # Set default
        if "viridis" in colormaps:
            self.colormap_combo.setCurrentText("viridis")

    def _on_colormap_selected(self, name: str) -> None:
        """Handle colormap selection change.

        Args:
            name: Selected colormap name.
        """
        if not name:
            return

        try:
            self._current_colormap_name = name
            cmap = self.manager.get_colormap(name)

            # Update preview
            if self._show_bw:
                bw_cmap = self.manager.to_bw(cmap)
                self.preview_canvas.set_colormap(bw_cmap)
            else:
                self.preview_canvas.set_colormap(cmap)

            # Update colorblind indicator
            self._update_colorblind_indicator(name)

            # Emit signal
            self.colormap_changed.emit(name)

        except Exception as e:
            logger.error(f"Failed to load colormap '{name}': {e}")

    def _update_colorblind_indicator(self, name: str) -> None:
        """Update colorblind-friendly indicator.

        Args:
            name: Colormap name.
        """
        try:
            colorblind_cmaps = self.manager.list_colorblind_friendly_colormaps()
            if name in colorblind_cmaps:
                self.colorblind_indicator.setText("✓ Colorblind-friendly")
                self.colorblind_indicator.setStyleSheet("color: green;")
            else:
                self.colorblind_indicator.setText("✗ Not colorblind-friendly")
                self.colorblind_indicator.setStyleSheet("color: red;")
        except Exception as e:
            logger.debug(f"Failed to update colorblind indicator: {e}")

    def _on_category_changed(self, category: str) -> None:
        """Filter colormaps by category.

        Args:
            category: Selected category name.
        """
        self.colormap_combo.blockSignals(True)
        self.colormap_combo.clear()

        if category == "All":
            colormaps = self.manager.list_colormaps()
        else:
            category_key = category.lower()
            colormaps = self.manager.colormaps_by_category(category_key)

        for name in sorted(colormaps):
            self.colormap_combo.addItem(name)

        # Restore previous selection if possible
        if self.colormap_combo.findText(self._current_colormap_name) >= 0:
            self.colormap_combo.setCurrentText(self._current_colormap_name)
        else:
            self.colormap_combo.setCurrentIndex(0)

        self.colormap_combo.blockSignals(False)

    def _on_bw_toggled(self, checked: bool) -> None:
        """Handle B&W preview toggle.

        Args:
            checked: Whether B&W preview is enabled.
        """
        self._show_bw = checked
        self._on_colormap_selected(self._current_colormap_name)

    def _open_custom_editor(self) -> None:
        """Open custom colormap editor dialog."""
        dialog = CustomColormapEditorDialog(self.manager, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Refresh colormap list and select new custom colormap
            custom_name = dialog.get_colormap_name()
            self._populate_colormaps()
            if custom_name and self.colormap_combo.findText(custom_name) >= 0:
                self.colormap_combo.setCurrentText(custom_name)

    def get_current_colormap(
        self,
    ) -> Callable[[float], tuple[float, ...]]:  # type: ignore[name-defined]
        """Get currently selected colormap.

        Returns:
            Callable colormap.
        """
        return self.manager.get_colormap(self._current_colormap_name)


class CustomColormapEditorDialog(QDialog if HAS_PYQT6 else object):
    """Dialog for creating and editing custom colormaps.

    Features:
    - Add/remove color stops
    - Drag-and-drop reordering
    - Visual color picker
    - Live preview
    - Save custom colormap
    """

    def __init__(
        self,
        manager: ColormapManager,
        parent: QWidget | None = None,
    ) -> None:  # type: ignore[name-defined]
        """Initialize custom colormap editor.

        Args:
            manager: ColormapManager instance.
            parent: Parent widget.
        """
        if not HAS_PYQT6:
            raise ImportError("PyQt6 required for CustomColormapEditorDialog")

        super().__init__(parent)
        self.setWindowTitle("Custom Colormap Editor")
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)

        self.manager = manager
        self._colormap_name: str | None = None
        self._color_stops: list[tuple[float, str]] = [
            (0.0, "#FF0000"),
            (1.0, "#0000FF"),
        ]

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the UI layout."""
        layout = QVBoxLayout(self)

        # Name input
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Colormap Name:"))
        from PyQt6.QtWidgets import QLineEdit

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g., my_custom_colormap")
        name_layout.addWidget(self.name_input)
        layout.addLayout(name_layout)

        # Color stops list
        layout.addWidget(QLabel("Color Stops:"))
        self.stops_list = QListWidget()
        self.stops_list.setMaximumHeight(150)
        layout.addWidget(self.stops_list)

        # Add/remove buttons
        button_layout = QHBoxLayout()

        self.add_button = QPushButton("Add Color Stop")
        self.add_button.clicked.connect(self._add_color_stop)
        button_layout.addWidget(self.add_button)

        self.remove_button = QPushButton("Remove Color Stop")
        self.remove_button.clicked.connect(self._remove_color_stop)
        button_layout.addWidget(self.remove_button)

        layout.addLayout(button_layout)

        # Preview
        layout.addWidget(QLabel("Preview:"))
        self.preview_canvas = ColormapPreviewCanvas()
        layout.addWidget(self.preview_canvas)

        # Dialog buttons
        dialog_layout = QHBoxLayout()
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self._save_colormap)
        dialog_layout.addWidget(self.save_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        dialog_layout.addWidget(self.cancel_button)

        layout.addLayout(dialog_layout)

        self._populate_color_stops()
        self._update_preview()

    def _populate_color_stops(self) -> None:
        """Populate color stops list widget."""
        self.stops_list.clear()
        for position, color in sorted(self._color_stops):
            item = QListWidgetItem()
            item.setText(f"Position {position:.2f}: {color}")
            # Store data for later retrieval
            item.setData(Qt.ItemDataRole.UserRole, (position, color))
            self.stops_list.addItem(item)

    def _add_color_stop(self) -> None:
        """Add a new color stop."""
        color = QColorDialog.getColor()
        if color.isValid():
            hex_color = color.name()
            position = 0.5  # Default middle position
            self._color_stops.append((position, hex_color))
            self._color_stops.sort(key=lambda x: x[0])
            self._populate_color_stops()
            self._update_preview()

    def _remove_color_stop(self) -> None:
        """Remove selected color stop."""
        current = self.stops_list.currentRow()
        if current >= 0:
            item = self.stops_list.item(current)
            position, color = item.data(Qt.ItemDataRole.UserRole)
            self._color_stops = [
                (p, c)
                for p, c in self._color_stops
                if not (p == position and c == color)
            ]
            self._populate_color_stops()
            self._update_preview()

    def _update_preview(self) -> None:
        """Update preview canvas."""
        try:
            if len(self._color_stops) < 2:
                return

            colors = [color for _, color in sorted(self._color_stops)]
            positions = [position for position, _ in sorted(self._color_stops)]

            cmap = self.manager.create_custom_colormap(colors, positions)
            self.preview_canvas.set_colormap(cmap)
        except Exception as e:
            logger.error(f"Failed to update preview: {e}")

    def _save_colormap(self) -> None:
        """Save custom colormap."""
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Invalid Input", "Please enter a colormap name")
            return

        try:
            colors = [color for _, color in sorted(self._color_stops)]
            positions = [position for position, _ in sorted(self._color_stops)]

            self.manager.create_custom_colormap(colors, positions, name=name)
            self._colormap_name = name
            QMessageBox.information(
                self, "Success", f"Colormap '{name}' saved successfully"
            )
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save colormap: {e}")

    def get_colormap_name(self) -> str | None:
        """Get the name of the saved colormap.

        Returns:
            Colormap name, or None if dialog was cancelled.
        """
        return self._colormap_name
