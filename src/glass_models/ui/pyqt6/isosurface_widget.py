"""PyQt6 widget for iso-surface control and visualization.

Provides UI controls for:
- Single iso-value slider with real-time preview
- Multi-level mode (parse comma-separated values)
- Transparency control (alpha slider 0-100%)
- Secondary field colormapping selector
- Enable/disable individual iso-surfaces
"""

import logging

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class IsoSurfaceControlWidget(QWidget):
    """Control widget for iso-surface extraction and visualization.

    Signals:
        iso_value_changed: Emitted when iso-value changes (float)
        iso_values_changed: Emitted when multi-level values change (List[float])
        transparency_changed: Emitted when transparency changes (int 0-100)
        colormap_changed: Emitted when colormap field changes (str)
        surface_enabled_changed: Emitted when surface enable state changes
    """

    iso_value_changed = pyqtSignal(float)
    iso_values_changed = pyqtSignal(list)
    transparency_changed = pyqtSignal(int)
    colormap_changed = pyqtSignal(str)
    surface_enabled_changed = pyqtSignal(bool)

    def __init__(
        self,
        parent: QWidget | None = None,
        field_range: tuple = (0.0, 1.0),
    ) -> None:
        """Initialize iso-surface control widget.

        Args:
            parent: Parent widget
            field_range: Tuple of (min, max) for iso-value range
        """
        super().__init__(parent)
        self.field_range = field_range
        self._iso_values: list[float] = []
        self._multi_level_mode = False
        self._surfaces: list[dict] = []

        logger.debug("IsoSurfaceControlWidget initialized with range %s", field_range)

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the UI components."""
        layout = QVBoxLayout(self)

        # Mode selection
        mode_layout = QHBoxLayout()
        self.mode_label = QLabel("Mode:")
        self.single_mode_button = QPushButton("Single Level")
        self.multi_mode_button = QPushButton("Multi Level")
        self.single_mode_button.setCheckable(True)
        self.multi_mode_button.setCheckable(True)
        self.single_mode_button.setChecked(True)

        mode_layout.addWidget(self.mode_label)
        mode_layout.addWidget(self.single_mode_button)
        mode_layout.addWidget(self.multi_mode_button)
        layout.addLayout(mode_layout)

        # Single iso-value control
        single_group = QGroupBox("Single Iso-Surface", self)
        single_layout = QVBoxLayout()

        # Iso-value slider
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Iso-Value:"))
        self.iso_slider = QSlider(Qt.Orientation.Horizontal)
        self.iso_slider.setMinimum(0)
        self.iso_slider.setMaximum(1000)
        self.iso_slider.setValue(500)
        slider_layout.addWidget(self.iso_slider)

        self.iso_value_label = QLabel("0.500")
        slider_layout.addWidget(self.iso_value_label)
        single_layout.addLayout(slider_layout)

        # Transparency control
        alpha_layout = QHBoxLayout()
        alpha_layout.addWidget(QLabel("Transparency:"))
        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setMinimum(0)
        self.alpha_slider.setMaximum(100)
        self.alpha_slider.setValue(100)
        alpha_layout.addWidget(self.alpha_slider)

        self.alpha_value_label = QLabel("100%")
        alpha_layout.addWidget(self.alpha_value_label)
        single_layout.addLayout(alpha_layout)

        single_group.setLayout(single_layout)
        layout.addWidget(single_group)

        # Multi-level control
        multi_group = QGroupBox("Multi-Level Iso-Surfaces", self)
        multi_group.setEnabled(False)
        multi_layout = QVBoxLayout()

        # Values input
        values_layout = QHBoxLayout()
        values_layout.addWidget(QLabel("Iso-Values (comma-separated):"))
        self.values_input = QLineEdit()
        self.values_input.setPlaceholderText("e.g., 0.3, 0.5, 0.7")
        values_layout.addWidget(self.values_input)
        self.extract_button = QPushButton("Extract")
        values_layout.addWidget(self.extract_button)
        multi_layout.addLayout(values_layout)

        # Surface list
        self.surface_list = QListWidget()
        self.surface_list.setMaximumHeight(150)
        multi_layout.addWidget(QLabel("Extracted Surfaces:"))
        multi_layout.addWidget(self.surface_list)

        multi_group.setLayout(multi_layout)
        self.multi_group = multi_group
        layout.addWidget(multi_group)

        # Colormap selection
        colormap_layout = QHBoxLayout()
        colormap_layout.addWidget(QLabel("Secondary Field:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(
            ["None", "Temperature", "Velocity", "Pressure", "Strain"]
        )
        colormap_layout.addWidget(self.colormap_combo)
        colormap_layout.addStretch()
        layout.addLayout(colormap_layout)

        # Enable/disable
        self.enable_checkbox = QCheckBox("Enable Iso-Surface Rendering")
        self.enable_checkbox.setChecked(True)
        layout.addWidget(self.enable_checkbox)

        layout.addStretch()
        self.setLayout(layout)

    def _connect_signals(self) -> None:
        """Connect UI signals to slots."""
        self.iso_slider.sliderMoved.connect(self._on_slider_moved)
        self.iso_slider.valueChanged.connect(self._on_slider_moved)
        self.alpha_slider.sliderMoved.connect(self._on_alpha_changed)
        self.alpha_slider.valueChanged.connect(self._on_alpha_changed)
        self.colormap_combo.currentTextChanged.connect(self.colormap_changed.emit)
        self.enable_checkbox.toggled.connect(self.surface_enabled_changed.emit)

        self.single_mode_button.clicked.connect(self._on_single_mode)
        self.multi_mode_button.clicked.connect(self._on_multi_mode)
        self.extract_button.clicked.connect(self._on_extract_multiple)

    def _on_slider_moved(self) -> None:
        """Handle iso-value slider change."""
        iso_value = self._slider_to_iso_value()
        self.iso_value_label.setText(f"{iso_value:.3f}")
        self.iso_value_changed.emit(iso_value)

    def _on_alpha_changed(self) -> None:
        """Handle transparency slider change."""
        alpha = self.alpha_slider.value()
        self.alpha_value_label.setText(f"{alpha}%")
        self.transparency_changed.emit(alpha)

    def _on_single_mode(self) -> None:
        """Switch to single iso-surface mode."""
        self.single_mode_button.setChecked(True)
        self.multi_mode_button.setChecked(False)
        self.multi_group.setEnabled(False)
        self._multi_level_mode = False
        logger.debug("Switched to single-level mode")

    def _on_multi_mode(self) -> None:
        """Switch to multi-level iso-surface mode."""
        self.single_mode_button.setChecked(False)
        self.multi_mode_button.setChecked(True)
        self.multi_group.setEnabled(True)
        self._multi_level_mode = True
        logger.debug("Switched to multi-level mode")

    def _on_extract_multiple(self) -> None:
        """Extract multiple iso-surfaces from input."""
        text = self.values_input.text().strip()
        if not text:
            logger.warning("No iso-values provided")
            return

        try:
            iso_values = [float(v.strip()) for v in text.split(",")]
            self._iso_values = iso_values

            # Update list widget
            self.surface_list.clear()
            for i, iso_val in enumerate(iso_values):
                item = QListWidgetItem(f"Surface {i + 1}: {iso_val:.3f}")
                self.surface_list.addItem(item)

            logger.debug("Extracted %d iso-values: %s", len(iso_values), iso_values)
            self.iso_values_changed.emit(iso_values)

        except ValueError as e:
            logger.error("Failed to parse iso-values: %s", str(e))

    def _slider_to_iso_value(self) -> float:
        """Convert slider position to iso-value."""
        min_val, max_val = self.field_range
        normalized = self.iso_slider.value() / self.iso_slider.maximum()
        return min_val + normalized * (max_val - min_val)

    def _iso_value_to_slider(self, iso_value: float) -> None:
        """Convert iso-value to slider position."""
        min_val, max_val = self.field_range
        if max_val == min_val:
            normalized = 0.0
        else:
            normalized = (iso_value - min_val) / (max_val - min_val)
        self.iso_slider.setValue(int(normalized * self.iso_slider.maximum()))

    def set_field_range(self, min_val: float, max_val: float) -> None:
        """Update field range for slider."""
        self.field_range = (min_val, max_val)
        logger.debug("Field range updated to [%.3f, %.3f]", min_val, max_val)

    def get_iso_value(self) -> float:
        """Get current iso-value."""
        return self._slider_to_iso_value()

    def get_iso_values(self) -> list[float]:
        """Get current iso-values (multi-level mode)."""
        return self._iso_values

    def get_transparency(self) -> int:
        """Get current transparency (0-100)."""
        return self.alpha_slider.value()

    def get_colormap(self) -> str:
        """Get selected secondary field."""
        return self.colormap_combo.currentText()

    def is_enabled(self) -> bool:
        """Check if iso-surface rendering is enabled."""
        return self.enable_checkbox.isChecked()

    def is_multi_level(self) -> bool:
        """Check if in multi-level mode."""
        return self._multi_level_mode

    def set_iso_value(self, iso_value: float) -> None:
        """Set iso-value programmatically."""
        self._iso_value_to_slider(iso_value)

    def set_transparency(self, alpha: int) -> None:
        """Set transparency programmatically."""
        self.alpha_slider.setValue(max(0, min(100, alpha)))


__all__ = ["IsoSurfaceControlWidget"]
