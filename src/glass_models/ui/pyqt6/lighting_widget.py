"""PyQt6 widget for lighting control in 3D visualization.

Provides UI controls for:
- Preset selector dropdown (headlight, studio_3light, ambient_only)
- Azimuth slider (0-360°)
- Elevation slider (0-90°)
- Ambient component slider (0-1)
- Specular component slider (0-1)
- Light intensity slider (0-1)
- Real-time feedback of current settings

GitHub issue #541: Lighting & Shading Control for 3D Visualization.
"""

import logging

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class LightingControlWidget(QWidget):
    """Control widget for lighting configuration and adjustment.

    Signals:
        preset_changed: Emitted when preset selection changes (str preset_name)
        azimuth_changed: Emitted when azimuth angle changes (float degrees)
        elevation_changed: Emitted when elevation angle changes (float degrees)
        ambient_changed: Emitted when ambient component changes (float 0-1)
        specular_changed: Emitted when specular component changes (float 0-1)
        intensity_changed: Emitted when light intensity changes (float 0-1)
    """

    preset_changed = pyqtSignal(str)
    azimuth_changed = pyqtSignal(float)
    elevation_changed = pyqtSignal(float)
    ambient_changed = pyqtSignal(float)
    specular_changed = pyqtSignal(float)
    intensity_changed = pyqtSignal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize lighting control widget.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)

        self._suppress_signals = False

        logger.debug("LightingControlWidget initialized")

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the UI components."""
        layout = QVBoxLayout(self)

        # ===================================================================
        # Preset Selection Group
        # ===================================================================
        preset_group = QGroupBox("Lighting Preset", self)
        preset_layout = QHBoxLayout()

        preset_layout.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(
            [
                "headlight",
                "studio_3light",
                "ambient_only",
            ]
        )
        self.preset_combo.setCurrentText("headlight")
        preset_layout.addWidget(self.preset_combo)

        preset_group.setLayout(preset_layout)
        layout.addWidget(preset_group)

        # ===================================================================
        # Light Direction Group (Spherical Coordinates)
        # ===================================================================
        direction_group = QGroupBox("Light Direction", self)
        direction_layout = QVBoxLayout()

        # Azimuth slider (0-360°)
        azimuth_h_layout = QHBoxLayout()
        azimuth_h_layout.addWidget(QLabel("Azimuth (°):"))
        self.azimuth_slider = QSlider(Qt.Orientation.Horizontal)
        self.azimuth_slider.setMinimum(0)
        self.azimuth_slider.setMaximum(360)
        self.azimuth_slider.setValue(0)
        self.azimuth_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.azimuth_slider.setTickInterval(45)
        azimuth_h_layout.addWidget(self.azimuth_slider)

        self.azimuth_spinbox = QSpinBox()
        self.azimuth_spinbox.setMinimum(0)
        self.azimuth_spinbox.setMaximum(360)
        self.azimuth_spinbox.setValue(0)
        self.azimuth_spinbox.setSuffix("°")
        azimuth_h_layout.addWidget(self.azimuth_spinbox)
        direction_layout.addLayout(azimuth_h_layout)

        # Elevation slider (0-90°)
        elevation_h_layout = QHBoxLayout()
        elevation_h_layout.addWidget(QLabel("Elevation (°):"))
        self.elevation_slider = QSlider(Qt.Orientation.Horizontal)
        self.elevation_slider.setMinimum(0)
        self.elevation_slider.setMaximum(90)
        self.elevation_slider.setValue(45)
        self.elevation_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.elevation_slider.setTickInterval(15)
        elevation_h_layout.addWidget(self.elevation_slider)

        self.elevation_spinbox = QSpinBox()
        self.elevation_spinbox.setMinimum(0)
        self.elevation_spinbox.setMaximum(90)
        self.elevation_spinbox.setValue(45)
        self.elevation_spinbox.setSuffix("°")
        elevation_h_layout.addWidget(self.elevation_spinbox)
        direction_layout.addLayout(elevation_h_layout)

        direction_group.setLayout(direction_layout)
        layout.addWidget(direction_group)

        # ===================================================================
        # Material Properties Group
        # ===================================================================
        material_group = QGroupBox("Material Properties", self)
        material_layout = QVBoxLayout()

        # Ambient component (0-1)
        ambient_h_layout = QHBoxLayout()
        ambient_h_layout.addWidget(QLabel("Ambient:"))
        self.ambient_slider = QSlider(Qt.Orientation.Horizontal)
        self.ambient_slider.setMinimum(0)
        self.ambient_slider.setMaximum(100)
        self.ambient_slider.setValue(20)
        ambient_h_layout.addWidget(self.ambient_slider)

        self.ambient_label = QLabel("0.20")
        ambient_h_layout.addWidget(self.ambient_label)
        material_layout.addLayout(ambient_h_layout)

        # Specular component (0-1)
        specular_h_layout = QHBoxLayout()
        specular_h_layout.addWidget(QLabel("Specular:"))
        self.specular_slider = QSlider(Qt.Orientation.Horizontal)
        self.specular_slider.setMinimum(0)
        self.specular_slider.setMaximum(100)
        self.specular_slider.setValue(50)
        specular_h_layout.addWidget(self.specular_slider)

        self.specular_label = QLabel("0.50")
        specular_h_layout.addWidget(self.specular_label)
        material_layout.addLayout(specular_h_layout)

        material_group.setLayout(material_layout)
        layout.addWidget(material_group)

        # ===================================================================
        # Light Intensity Group
        # ===================================================================
        intensity_group = QGroupBox("Light Intensity", self)
        intensity_layout = QHBoxLayout()

        intensity_layout.addWidget(QLabel("Intensity:"))
        self.intensity_slider = QSlider(Qt.Orientation.Horizontal)
        self.intensity_slider.setMinimum(0)
        self.intensity_slider.setMaximum(100)
        self.intensity_slider.setValue(100)
        intensity_layout.addWidget(self.intensity_slider)

        self.intensity_label = QLabel("1.00")
        intensity_layout.addWidget(self.intensity_label)

        intensity_group.setLayout(intensity_layout)
        layout.addWidget(intensity_group)

        layout.addStretch()

    def _connect_signals(self) -> None:
        """Connect UI signals to slots."""
        # Preset combo
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)

        # Azimuth
        self.azimuth_slider.sliderMoved.connect(self._on_azimuth_slider_moved)
        self.azimuth_spinbox.valueChanged.connect(self._on_azimuth_spinbox_changed)

        # Elevation
        self.elevation_slider.sliderMoved.connect(self._on_elevation_slider_moved)
        self.elevation_spinbox.valueChanged.connect(self._on_elevation_spinbox_changed)

        # Ambient
        self.ambient_slider.sliderMoved.connect(self._on_ambient_slider_moved)

        # Specular
        self.specular_slider.sliderMoved.connect(self._on_specular_slider_moved)

        # Intensity
        self.intensity_slider.sliderMoved.connect(self._on_intensity_slider_moved)

    def _on_preset_changed(self, preset_name: str) -> None:
        """Handle preset selection change."""
        if self._suppress_signals:
            return

        logger.debug("Preset changed to: %s", preset_name)
        self.preset_changed.emit(preset_name)

    def _on_azimuth_slider_moved(self, value: int) -> None:
        """Handle azimuth slider movement."""
        if self._suppress_signals:
            return

        self._suppress_signals = True
        self.azimuth_spinbox.setValue(value)
        self._suppress_signals = False

        azimuth_deg = float(value)
        logger.debug("Azimuth changed to: %.1f°", azimuth_deg)
        self.azimuth_changed.emit(azimuth_deg)

    def _on_azimuth_spinbox_changed(self, value: int) -> None:
        """Handle azimuth spinbox change."""
        if self._suppress_signals:
            return

        self._suppress_signals = True
        self.azimuth_slider.setValue(value)
        self._suppress_signals = False

        azimuth_deg = float(value)
        logger.debug("Azimuth changed to: %.1f°", azimuth_deg)
        self.azimuth_changed.emit(azimuth_deg)

    def _on_elevation_slider_moved(self, value: int) -> None:
        """Handle elevation slider movement."""
        if self._suppress_signals:
            return

        self._suppress_signals = True
        self.elevation_spinbox.setValue(value)
        self._suppress_signals = False

        elevation_deg = float(value)
        logger.debug("Elevation changed to: %.1f°", elevation_deg)
        self.elevation_changed.emit(elevation_deg)

    def _on_elevation_spinbox_changed(self, value: int) -> None:
        """Handle elevation spinbox change."""
        if self._suppress_signals:
            return

        self._suppress_signals = True
        self.elevation_slider.setValue(value)
        self._suppress_signals = False

        elevation_deg = float(value)
        logger.debug("Elevation changed to: %.1f°", elevation_deg)
        self.elevation_changed.emit(elevation_deg)

    def _on_ambient_slider_moved(self, value: int) -> None:
        """Handle ambient slider movement."""
        if self._suppress_signals:
            return

        ambient = value / 100.0
        self.ambient_label.setText(f"{ambient:.2f}")

        logger.debug("Ambient changed to: %.2f", ambient)
        self.ambient_changed.emit(ambient)

    def _on_specular_slider_moved(self, value: int) -> None:
        """Handle specular slider movement."""
        if self._suppress_signals:
            return

        specular = value / 100.0
        self.specular_label.setText(f"{specular:.2f}")

        logger.debug("Specular changed to: %.2f", specular)
        self.specular_changed.emit(specular)

    def _on_intensity_slider_moved(self, value: int) -> None:
        """Handle intensity slider movement."""
        if self._suppress_signals:
            return

        intensity = value / 100.0
        self.intensity_label.setText(f"{intensity:.2f}")

        logger.debug("Intensity changed to: %.2f", intensity)
        self.intensity_changed.emit(intensity)

    # =========================================================================
    # Public API: Query and Set Widget State
    # =========================================================================

    def get_preset(self) -> str:
        """Get currently selected preset name."""
        return self.preset_combo.currentText()

    def set_preset(self, preset_name: str) -> None:
        """Set the preset selection.

        Args:
            preset_name: Name of preset to select
        """
        idx = self.preset_combo.findText(preset_name)
        if idx >= 0:
            self.preset_combo.setCurrentIndex(idx)

    def get_azimuth(self) -> float:
        """Get current azimuth angle in degrees."""
        return float(self.azimuth_slider.value())

    def set_azimuth(self, degrees: float) -> None:
        """Set azimuth angle in degrees.

        Args:
            degrees: Azimuth angle (0-360)
        """
        value = int(max(0, min(360, degrees)))
        self.azimuth_slider.setValue(value)

    def get_elevation(self) -> float:
        """Get current elevation angle in degrees."""
        return float(self.elevation_slider.value())

    def set_elevation(self, degrees: float) -> None:
        """Set elevation angle in degrees.

        Args:
            degrees: Elevation angle (0-90)
        """
        value = int(max(0, min(90, degrees)))
        self.elevation_slider.setValue(value)

    def get_ambient(self) -> float:
        """Get current ambient component (0-1)."""
        return self.ambient_slider.value() / 100.0

    def set_ambient(self, value: float) -> None:
        """Set ambient component.

        Args:
            value: Ambient component (0-1)
        """
        slider_value = int(max(0, min(100, value * 100)))
        self.ambient_slider.setValue(slider_value)

    def get_specular(self) -> float:
        """Get current specular component (0-1)."""
        return self.specular_slider.value() / 100.0

    def set_specular(self, value: float) -> None:
        """Set specular component.

        Args:
            value: Specular component (0-1)
        """
        slider_value = int(max(0, min(100, value * 100)))
        self.specular_slider.setValue(slider_value)

    def get_intensity(self) -> float:
        """Get current light intensity (0-1)."""
        return self.intensity_slider.value() / 100.0

    def set_intensity(self, value: float) -> None:
        """Set light intensity.

        Args:
            value: Light intensity (0-1)
        """
        slider_value = int(max(0, min(100, value * 100)))
        self.intensity_slider.setValue(slider_value)
