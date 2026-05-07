"""High-resolution export dialog for PyQt6 applications.

This module provides a comprehensive export dialog for configuring
high-resolution rendering with anti-aliasing, DPI, and format settings.

Example:
    >>> from PyQt6.QtWidgets import QApplication
    >>> app = QApplication([])
    >>> dialog = HighResolutionExportDialog()
    >>> if dialog.exec():
    ...     config = dialog.get_export_config()
    ...     print(config)
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    """Configuration for high-resolution export.

    Attributes:
        resolution: Resolution name ("1080p", "2K", "4K", "8K", or "custom")
        width: Custom width in pixels (for custom resolution)
        height: Custom height in pixels (for custom resolution)
        aa_level: Anti-aliasing level (1, 2, 4, or 8)
        dpi: Output DPI (72-600)
        format: Output format ("PNG", "JPG", or "Both")
        output_dir: Output directory path
    """

    resolution: str
    width: int | None = None
    height: int | None = None
    aa_level: int = 1
    dpi: int = 72
    format: str = "PNG"
    output_dir: str | None = None


class HighResolutionExportDialog(QDialog):
    """Dialog for configuring high-resolution export settings.

    Provides preset buttons for common resolutions, custom resolution input,
    anti-aliasing level selector, DPI input, and format selection.
    """

    RESOLUTIONS = {
        "1080p": (1920, 1080),
        "2K": (2560, 1440),
        "4K": (3840, 2160),
        "8K": (7680, 4320),
    }
    AA_LEVELS = [1, 2, 4, 8]
    MIN_DPI = 72
    MAX_DPI = 600
    FORMATS = ["PNG", "JPG", "Both"]

    def __init__(
        self,
        parent: QWidget | None = None,
        initial_dir: str | None = None,
    ) -> None:
        """Initialize the export dialog.

        Args:
            parent: Parent widget
            initial_dir: Initial directory for file selection
        """
        super().__init__(parent)
        self.initial_dir = initial_dir or str(Path.home())
        self._setup_ui()
        self._connect_signals()
        logger.debug("HighResolutionExportDialog initialized")

    def _setup_ui(self) -> None:
        """Set up the dialog UI components."""
        self.setWindowTitle("High-Resolution Export")
        self.setMinimumWidth(500)

        layout = QVBoxLayout()

        # Resolution preset buttons
        layout.addWidget(self._create_resolution_group())

        # Custom resolution section
        layout.addWidget(self._create_custom_resolution_group())

        # Anti-aliasing section
        layout.addWidget(self._create_aa_group())

        # DPI and format section
        layout.addWidget(self._create_output_group())

        # Output directory selection
        layout.addWidget(self._create_directory_group())

        # Buttons
        button_layout = QHBoxLayout()
        self.export_button = QPushButton("Export")
        self.export_button.setDefault(True)
        self.cancel_button = QPushButton("Cancel")

        button_layout.addStretch()
        button_layout.addWidget(self.export_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def _create_resolution_group(self) -> QGroupBox:
        """Create resolution preset buttons."""
        group = QGroupBox("Resolution")
        layout = QHBoxLayout()

        self.resolution_buttons = {}
        for res_name in self.RESOLUTIONS:
            btn = QPushButton(res_name)
            btn.setCheckable(True)
            btn.clicked.connect(
                lambda checked, r=res_name: self._on_resolution_selected(r)
            )
            layout.addWidget(btn)
            self.resolution_buttons[res_name] = btn

        # Select 1080p by default
        self.resolution_buttons["1080p"].setChecked(True)
        self._current_resolution = "1080p"

        layout.addStretch()
        group.setLayout(layout)
        return group

    def _create_custom_resolution_group(self) -> QGroupBox:
        """Create custom resolution input controls."""
        group = QGroupBox("Custom Resolution")
        layout = QFormLayout()

        self.custom_width = QSpinBox()
        self.custom_width.setMinimum(1)
        self.custom_width.setMaximum(20000)
        self.custom_width.setValue(1920)

        self.custom_height = QSpinBox()
        self.custom_height.setMinimum(1)
        self.custom_height.setMaximum(20000)
        self.custom_height.setValue(1080)

        layout.addRow("Width (px):", self.custom_width)
        layout.addRow("Height (px):", self.custom_height)

        group.setLayout(layout)
        return group

    def _create_aa_group(self) -> QGroupBox:
        """Create anti-aliasing level selector."""
        group = QGroupBox("Anti-Aliasing")
        layout = QFormLayout()

        self.aa_combo = QComboBox()
        self.aa_combo.addItems([str(level) for level in self.AA_LEVELS])
        self.aa_combo.setCurrentText("1")

        layout.addRow("AA Level:", self.aa_combo)
        group.setLayout(layout)
        return group

    def _create_output_group(self) -> QGroupBox:
        """Create DPI and format selection controls."""
        group = QGroupBox("Output Settings")
        layout = QFormLayout()

        self.dpi_spin = QSpinBox()
        self.dpi_spin.setMinimum(self.MIN_DPI)
        self.dpi_spin.setMaximum(self.MAX_DPI)
        self.dpi_spin.setValue(72)
        self.dpi_spin.setSuffix(" DPI")

        self.format_combo = QComboBox()
        self.format_combo.addItems(self.FORMATS)

        layout.addRow("DPI:", self.dpi_spin)
        layout.addRow("Format:", self.format_combo)
        group.setLayout(layout)
        return group

    def _create_directory_group(self) -> QGroupBox:
        """Create directory selection controls."""
        group = QGroupBox("Output Directory")
        layout = QHBoxLayout()

        self.dir_label = QLineEdit()
        self.dir_label.setText(self.initial_dir)
        self.dir_label.setReadOnly(True)

        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self._on_browse_directory)

        layout.addWidget(QLabel("Directory:"))
        layout.addWidget(self.dir_label)
        layout.addWidget(self.browse_button)

        group.setLayout(layout)
        return group

    def _connect_signals(self) -> None:
        """Connect button signals to slots."""
        self.export_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

    def _on_resolution_selected(self, resolution: str) -> None:
        """Handle resolution preset selection."""
        # Uncheck all other buttons
        for name, btn in self.resolution_buttons.items():
            btn.setChecked(name == resolution)

        # Update custom resolution inputs if preset selected
        if resolution in self.RESOLUTIONS:
            width, height = self.RESOLUTIONS[resolution]
            self.custom_width.setValue(width)
            self.custom_height.setValue(height)
            self._current_resolution = resolution
        else:
            self._current_resolution = "custom"

        logger.debug("Resolution selected: %s", resolution)

    def _on_browse_directory(self) -> None:
        """Handle directory browse button click."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            self.initial_dir,
        )
        if directory:
            self.dir_label.setText(directory)
            self.initial_dir = directory
            logger.debug("Output directory selected: %s", directory)

    def get_export_config(self) -> ExportConfig:
        """Get the current export configuration.

        Returns:
            ExportConfig with current settings
        """
        # Determine resolution
        resolution = self._current_resolution
        width = None
        height = None

        if resolution not in self.RESOLUTIONS:
            resolution = "custom"
            width = self.custom_width.value()
            height = self.custom_height.value()
        else:
            # Still get custom width/height for consistency
            width = self.custom_width.value()
            height = self.custom_height.value()

        config = ExportConfig(
            resolution=resolution,
            width=width,
            height=height,
            aa_level=int(self.aa_combo.currentText()),
            dpi=self.dpi_spin.value(),
            format=self.format_combo.currentText(),
            output_dir=self.dir_label.text(),
        )

        logger.debug("Export config: %s", config)
        return config

    def set_export_config(self, config: ExportConfig) -> None:
        """Set dialog values from an ExportConfig.

        Args:
            config: ExportConfig to apply
        """
        # Set resolution
        if config.resolution in self.RESOLUTIONS:
            self._on_resolution_selected(config.resolution)
        else:
            self._current_resolution = "custom"

        # Set custom dimensions
        if config.width is not None:
            self.custom_width.setValue(config.width)
        if config.height is not None:
            self.custom_height.setValue(config.height)

        # Set AA level
        self.aa_combo.setCurrentText(str(config.aa_level))

        # Set DPI
        self.dpi_spin.setValue(config.dpi)

        # Set format
        self.format_combo.setCurrentText(config.format)

        # Set directory
        if config.output_dir:
            self.dir_label.setText(config.output_dir)

        logger.debug("Export config applied")


class ExportProgressDialog(QDialog):
    """Simple progress dialog for rendering operations.

    Displays current operation and progress information.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize progress dialog.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Rendering...")
        self.setMinimumWidth(400)
        self.setModal(True)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowCloseButtonHint)

        layout = QVBoxLayout()

        self.status_label = QLabel("Initializing rendering...")
        layout.addWidget(self.status_label)

        self.setLayout(layout)

    def set_status(self, message: str) -> None:
        """Update status message.

        Args:
            message: Status message to display
        """
        self.status_label.setText(message)

    def set_progress(self, current: int, total: int, view: str = "") -> None:
        """Update progress information.

        Args:
            current: Current item number
            total: Total items
            view: Current view name (optional)
        """
        if view:
            message = f"Rendering {view}... ({current}/{total})"
        else:
            message = f"Progress: {current}/{total}"

        self.set_status(message)
