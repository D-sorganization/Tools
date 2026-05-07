"""PyQt6 widget for contour line control and visualization.

Provides UI controls for:
- Level count slider (1-50 levels)
- Custom levels input (comma-separated values)
- Spacing selector (uniform/logarithmic)
- Line style dropdown (solid/dashed/dotted)
- Color selector per level
- Enable/disable toggle
- Real-time preview signals
"""

import logging
from typing import Any

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class ContourControlWidget(QWidget):
    """Control widget for contour line extraction and visualization.

    Provides controls for:
    - Number of contour levels
    - Custom level values
    - Level spacing (uniform/log)
    - Line styles
    - Colors per level
    - Enable/disable toggle

    Signals:
        levels_changed: Emitted when number of levels changes (int)
        custom_levels_changed: Emitted when custom levels change (list[float])
        spacing_changed: Emitted when spacing type changes (str)
        line_style_changed: Emitted when line style changes (str)
        color_changed: Emitted when color changes (QColor, int)
        contours_enabled_changed: Emitted when enabled state changes (bool)
    """

    levels_changed = pyqtSignal(int)
    custom_levels_changed = pyqtSignal(list)
    spacing_changed = pyqtSignal(str)
    line_style_changed = pyqtSignal(str)
    color_changed = pyqtSignal(QColor, int)  # color, level_index
    contours_enabled_changed = pyqtSignal(bool)

    def __init__(
        self,
        parent: QWidget | None = None,
        field_range: tuple = (0.0, 1.0),
    ) -> None:
        """Initialize contour control widget.

        Args:
            parent: Parent widget
            field_range: Tuple of (min, max) for level range
        """
        super().__init__(parent)
        self.field_range = field_range
        self._colors: list[QColor] = []
        self._custom_mode = False

        logger.debug("ContourControlWidget initialized with range %s", field_range)

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the UI components."""
        layout = QVBoxLayout(self)

        # Mode selection
        mode_layout = QHBoxLayout()
        self.mode_label = QLabel("Mode:")
        self.auto_mode_button = self._create_button("Auto Levels")
        self.custom_mode_button = self._create_button("Custom Levels")
        self.auto_mode_button.setCheckable(True)
        self.custom_mode_button.setCheckable(True)
        self.auto_mode_button.setChecked(True)

        mode_layout.addWidget(self.mode_label)
        mode_layout.addWidget(self.auto_mode_button)
        mode_layout.addWidget(self.custom_mode_button)
        mode_layout.addStretch()
        layout.addLayout(mode_layout)

        # Auto levels control
        auto_group = QGroupBox("Auto-Generated Levels", self)
        auto_layout = QVBoxLayout()

        # Level count spinner
        level_layout = QHBoxLayout()
        level_layout.addWidget(QLabel("Number of Levels:"))
        self.level_spinbox = QSpinBox()
        self.level_spinbox.setMinimum(1)
        self.level_spinbox.setMaximum(50)
        self.level_spinbox.setValue(10)
        level_layout.addWidget(self.level_spinbox)
        level_layout.addStretch()
        auto_layout.addLayout(level_layout)

        # Spacing selector
        spacing_layout = QHBoxLayout()
        spacing_layout.addWidget(QLabel("Spacing:"))
        self.spacing_combo = QComboBox()
        self.spacing_combo.addItems(["Uniform", "Logarithmic"])
        spacing_layout.addWidget(self.spacing_combo)
        spacing_layout.addStretch()
        auto_layout.addLayout(spacing_layout)

        auto_group.setLayout(auto_layout)
        layout.addWidget(auto_group)

        # Custom levels control
        custom_group = QGroupBox("Custom Levels", self)
        custom_group.setEnabled(False)
        custom_layout = QVBoxLayout()

        # Custom values input
        values_layout = QHBoxLayout()
        values_layout.addWidget(QLabel("Values (comma-separated):"))
        self.custom_input = QLineEdit()
        self.custom_input.setPlaceholderText("e.g., 0.1, 0.3, 0.5, 0.7, 0.9")
        values_layout.addWidget(self.custom_input)
        custom_layout.addLayout(values_layout)

        # Level list
        custom_layout.addWidget(QLabel("Defined Levels:"))
        self.level_list = QListWidget()
        self.level_list.setMaximumHeight(120)
        custom_layout.addWidget(self.level_list)

        custom_group.setLayout(custom_layout)
        self.custom_group = custom_group
        layout.addWidget(custom_group)

        # Line style and appearance
        style_group = QGroupBox("Line Appearance", self)
        style_layout = QVBoxLayout()

        # Line style selector
        line_layout = QHBoxLayout()
        line_layout.addWidget(QLabel("Line Style:"))
        self.style_combo = QComboBox()
        self.style_combo.addItems(["Solid", "Dashed", "Dotted"])
        line_layout.addWidget(self.style_combo)
        line_layout.addStretch()
        style_layout.addLayout(line_layout)

        # Color selection per level
        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("Level Color:"))
        self.color_select_label = QLabel("Level 0")
        color_layout.addWidget(self.color_select_label)
        self.color_button = self._create_button("Select Color")
        color_layout.addWidget(self.color_button)
        color_layout.addStretch()
        style_layout.addLayout(color_layout)

        style_group.setLayout(style_layout)
        layout.addWidget(style_group)

        # Enable/disable
        self.enable_checkbox = QCheckBox("Enable Contour Lines")
        self.enable_checkbox.setChecked(True)
        layout.addWidget(self.enable_checkbox)

        layout.addStretch()
        self.setLayout(layout)

    def _create_button(self, text: str) -> Any:
        """Create a styled button.

        Args:
            text: Button text

        Returns:
            QPushButton instance
        """
        from PyQt6.QtWidgets import QPushButton

        return QPushButton(text)

    def _connect_signals(self) -> None:
        """Connect UI signals to slots."""
        self.level_spinbox.valueChanged.connect(self._on_levels_changed)
        self.spacing_combo.currentTextChanged.connect(self._on_spacing_changed)
        self.style_combo.currentTextChanged.connect(self._on_style_changed)
        self.custom_input.textChanged.connect(self._on_custom_levels_changed)

        self.auto_mode_button.clicked.connect(self._on_auto_mode)
        self.custom_mode_button.clicked.connect(self._on_custom_mode)
        self.color_button.clicked.connect(self._on_color_selected)

        self.enable_checkbox.toggled.connect(self.contours_enabled_changed.emit)

    def _on_levels_changed(self) -> None:
        """Handle level count change."""
        count = self.level_spinbox.value()
        self.levels_changed.emit(count)
        logger.debug("Level count changed to %d", count)

    def _on_spacing_changed(self, text: str) -> None:
        """Handle spacing type change."""
        spacing = text.lower()
        self.spacing_changed.emit(spacing)
        logger.debug("Spacing changed to %s", spacing)

    def _on_style_changed(self, text: str) -> None:
        """Handle line style change."""
        self.line_style_changed.emit(text.lower())
        logger.debug("Line style changed to %s", text)

    def _on_auto_mode(self) -> None:
        """Switch to auto-generated levels mode."""
        self.auto_mode_button.setChecked(True)
        self.custom_mode_button.setChecked(False)
        self.custom_group.setEnabled(False)
        self._custom_mode = False
        logger.debug("Switched to auto-level mode")

    def _on_custom_mode(self) -> None:
        """Switch to custom levels mode."""
        self.auto_mode_button.setChecked(False)
        self.custom_mode_button.setChecked(True)
        self.custom_group.setEnabled(True)
        self._custom_mode = True
        logger.debug("Switched to custom-level mode")

    def _on_custom_levels_changed(self) -> None:
        """Handle custom levels input change."""
        text = self.custom_input.text().strip()
        if not text:
            self.level_list.clear()
            return

        try:
            levels = [float(v.strip()) for v in text.split(",")]
            levels.sort()

            # Update list widget
            self.level_list.clear()
            for i, level_val in enumerate(levels):
                item = QListWidgetItem(f"Level {i + 1}: {level_val:.3f}")
                self.level_list.addItem(item)

            logger.debug("Custom levels parsed: %s", levels)
            self.custom_levels_changed.emit(levels)

        except ValueError as e:
            logger.error("Failed to parse custom levels: %s", str(e))

    def _on_color_selected(self) -> None:
        """Handle color selection for a level."""
        initial_color = QColor("steelblue") if not self._colors else self._colors[0]
        color = QColorDialog.getColor(initial_color, self, "Select Contour Line Color")

        if color.isValid():
            # Store color
            if not self._colors:
                self._colors = [color]
            else:
                self._colors[0] = color

            # Update button appearance
            self.color_button.setStyleSheet(
                f"background-color: {color.name()}; color: white;"
            )

            logger.debug("Color selected: %s", color.name())
            self.color_changed.emit(color, 0)

    def set_field_range(self, min_val: float, max_val: float) -> None:
        """Update field range for level controls.

        Args:
            min_val: Minimum field value
            max_val: Maximum field value
        """
        self.field_range = (min_val, max_val)
        logger.debug("Field range updated to [%.3f, %.3f]", min_val, max_val)

    def get_level_count(self) -> int:
        """Get current number of levels.

        Returns:
            Number of contour levels
        """
        return self.level_spinbox.value()

    def get_custom_levels(self) -> list[float]:
        """Get custom level values.

        Returns:
            List of custom level values, or empty if in auto mode
        """
        if not self._custom_mode:
            return []

        text = self.custom_input.text().strip()
        if not text:
            return []

        try:
            return sorted([float(v.strip()) for v in text.split(",")])
        except ValueError:
            return []

    def get_spacing(self) -> str:
        """Get current spacing type.

        Returns:
            'uniform' or 'logarithmic'
        """
        return self.spacing_combo.currentText().lower()

    def get_line_style(self) -> str:
        """Get current line style.

        Returns:
            'solid', 'dashed', or 'dotted'
        """
        return self.style_combo.currentText().lower()

    def is_custom_mode(self) -> bool:
        """Check if in custom levels mode.

        Returns:
            True if custom mode is active
        """
        return self._custom_mode

    def is_enabled(self) -> bool:
        """Check if contour rendering is enabled.

        Returns:
            True if enabled
        """
        return self.enable_checkbox.isChecked()

    def set_level_count(self, count: int) -> None:
        """Set level count programmatically.

        Args:
            count: Number of levels
        """
        self.level_spinbox.setValue(max(1, min(50, count)))

    def set_spacing(self, spacing: str) -> None:
        """Set spacing type programmatically.

        Args:
            spacing: 'uniform' or 'logarithmic'
        """
        if spacing.lower() == "logarithmic":
            self.spacing_combo.setCurrentText("Logarithmic")
        else:
            self.spacing_combo.setCurrentText("Uniform")

    def set_line_style(self, style: str) -> None:
        """Set line style programmatically.

        Args:
            style: 'solid', 'dashed', or 'dotted'
        """
        style_map = {
            "solid": "Solid",
            "dashed": "Dashed",
            "dotted": "Dotted",
        }
        text = style_map.get(style.lower(), "Solid")
        self.style_combo.setCurrentText(text)


__all__ = ["ContourControlWidget"]
