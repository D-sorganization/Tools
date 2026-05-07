"""PyQt6 widget for vector field glyph control and visualization.

Provides UI controls for:
- Density slider (1-100%)
- Scale slider for glyph size
- Style selector (Arrows, Cones, Spheres)
- Colormap selector
- Opacity slider
- Real-time density updates with performance monitoring
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Colormap options
COLORMAPS = [
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "twilight",
    "jet",
    "coolwarm",
    "hot",
    "gray",
]

# Glyph style options
GLYPH_STYLES = ["Arrows", "Cones", "Spheres"]


class GlyphControlWidget(QWidget):
    """Control widget for vector field glyph visualization.

    Provides sliders and selectors for controlling glyph density, size,
    style, colormapping, and opacity with real-time updates.

    Signals:
        density_changed: Emitted when density changes (float 0-1)
        scale_changed: Emitted when scale changes (float)
        style_changed: Emitted when style changes (str)
        colormap_changed: Emitted when colormap changes (str)
        opacity_changed: Emitted when opacity changes (float 0-1)
    """

    density_changed = pyqtSignal(float)
    scale_changed = pyqtSignal(float)
    style_changed = pyqtSignal(str)
    colormap_changed = pyqtSignal(str)
    opacity_changed = pyqtSignal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize glyph control widget.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)

        self._density = 1.0
        self._scale = 1.0
        self._style = "Arrows"
        self._colormap = "viridis"
        self._opacity = 1.0

        logger.debug("GlyphControlWidget initialized")

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the UI components."""
        layout = QVBoxLayout(self)

        # Density control group
        density_group = self._create_density_group()
        layout.addWidget(density_group)

        # Scale control group
        scale_group = self._create_scale_group()
        layout.addWidget(scale_group)

        # Style selection group
        style_group = self._create_style_group()
        layout.addWidget(style_group)

        # Colormap selection group
        colormap_group = self._create_colormap_group()
        layout.addWidget(colormap_group)

        # Opacity control group
        opacity_group = self._create_opacity_group()
        layout.addWidget(opacity_group)

        layout.addStretch()

    def _create_density_group(self) -> QGroupBox:
        """Create density control group."""
        group = QGroupBox("Glyph Density")
        layout = QVBoxLayout()

        # Slider
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Density:"))

        self.density_slider = QSlider(Qt.Orientation.Horizontal)
        self.density_slider.setMinimum(1)
        self.density_slider.setMaximum(100)
        self.density_slider.setValue(100)
        self.density_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.density_slider.setTickInterval(10)
        slider_layout.addWidget(self.density_slider)

        # Percentage display
        self.density_label = QLabel("100%")
        self.density_label.setMinimumWidth(40)
        slider_layout.addWidget(self.density_label)

        layout.addLayout(slider_layout)

        # Info
        info_label = QLabel("Lower density = faster rendering")
        info_label.setStyleSheet("font-size: 11px; color: gray;")
        layout.addWidget(info_label)

        group.setLayout(layout)
        return group

    def _create_scale_group(self) -> QGroupBox:
        """Create scale control group."""
        group = QGroupBox("Glyph Scale")
        layout = QVBoxLayout()

        # Slider
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Scale:"))

        self.scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.scale_slider.setMinimum(10)
        self.scale_slider.setMaximum(500)
        self.scale_slider.setValue(100)
        self.scale_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.scale_slider.setTickInterval(50)
        slider_layout.addWidget(self.scale_slider)

        # Multiplier display
        self.scale_label = QLabel("1.0x")
        self.scale_label.setMinimumWidth(40)
        slider_layout.addWidget(self.scale_label)

        layout.addLayout(slider_layout)

        # Info
        info_label = QLabel("Controls glyph size relative to field magnitude")
        info_label.setStyleSheet("font-size: 11px; color: gray;")
        layout.addWidget(info_label)

        group.setLayout(layout)
        return group

    def _create_style_group(self) -> QGroupBox:
        """Create style selection group."""
        group = QGroupBox("Glyph Style")
        layout = QVBoxLayout()

        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Style:"))

        self.style_combo = QComboBox()
        self.style_combo.addItems(GLYPH_STYLES)
        self.style_combo.setCurrentText("Arrows")
        selector_layout.addWidget(self.style_combo)
        selector_layout.addStretch()

        layout.addLayout(selector_layout)

        # Descriptions
        descriptions = {
            "Arrows": "3D arrows (best for directional flow)",
            "Cones": "Cone glyphs (cleaner appearance)",
            "Spheres": "Spheres colored by magnitude",
        }

        self.style_info = QLabel(descriptions["Arrows"])
        self.style_info.setStyleSheet("font-size: 11px; color: gray;")
        layout.addWidget(self.style_info)

        group.setLayout(layout)
        return group

    def _create_colormap_group(self) -> QGroupBox:
        """Create colormap selection group."""
        group = QGroupBox("Colormap")
        layout = QVBoxLayout()

        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Colormap:"))

        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(COLORMAPS)
        self.colormap_combo.setCurrentText("viridis")
        selector_layout.addWidget(self.colormap_combo)
        selector_layout.addStretch()

        layout.addLayout(selector_layout)

        # Info
        info_label = QLabel("Color glyphs by secondary field (if provided)")
        info_label.setStyleSheet("font-size: 11px; color: gray;")
        layout.addWidget(info_label)

        group.setLayout(layout)
        return group

    def _create_opacity_group(self) -> QGroupBox:
        """Create opacity control group."""
        group = QGroupBox("Opacity")
        layout = QVBoxLayout()

        # Slider
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Opacity:"))

        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.opacity_slider.setTickInterval(10)
        slider_layout.addWidget(self.opacity_slider)

        # Percentage display
        self.opacity_label = QLabel("100%")
        self.opacity_label.setMinimumWidth(40)
        slider_layout.addWidget(self.opacity_label)

        layout.addLayout(slider_layout)

        group.setLayout(layout)
        return group

    def _connect_signals(self) -> None:
        """Connect UI signals to slots."""
        self.density_slider.valueChanged.connect(self._on_density_changed)
        self.scale_slider.valueChanged.connect(self._on_scale_changed)
        self.style_combo.currentTextChanged.connect(self._on_style_changed)
        self.colormap_combo.currentTextChanged.connect(self._on_colormap_changed)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)

    def _on_density_changed(self, value: int) -> None:
        """Handle density slider change."""
        self._density = value / 100.0
        self.density_label.setText(f"{value}%")
        self.density_changed.emit(self._density)

    def _on_scale_changed(self, value: int) -> None:
        """Handle scale slider change."""
        self._scale = value / 100.0
        self.scale_label.setText(f"{self._scale:.1f}x")
        self.scale_changed.emit(self._scale)

    def _on_style_changed(self, style: str) -> None:
        """Handle style combo change."""
        self._style = style
        descriptions = {
            "Arrows": "3D arrows (best for directional flow)",
            "Cones": "Cone glyphs (cleaner appearance)",
            "Spheres": "Spheres colored by magnitude",
        }
        self.style_info.setText(descriptions.get(style, ""))
        self.style_changed.emit(style)

    def _on_colormap_changed(self, colormap: str) -> None:
        """Handle colormap combo change."""
        self._colormap = colormap
        self.colormap_changed.emit(colormap)

    def _on_opacity_changed(self, value: int) -> None:
        """Handle opacity slider change."""
        self._opacity = value / 100.0
        self.opacity_label.setText(f"{value}%")
        self.opacity_changed.emit(self._opacity)

    # Property accessors
    def get_density(self) -> float:
        """Get current density factor (0-1)."""
        return self._density

    def set_density(self, density: float) -> None:
        """Set density factor (0-1), updating UI."""
        if not 0 <= density <= 1:
            raise ValueError(f"density must be in [0, 1], got {density}")
        self._density = density
        self.density_slider.blockSignals(True)
        self.density_slider.setValue(int(density * 100))
        self.density_label.setText(f"{int(density * 100)}%")
        self.density_slider.blockSignals(False)

    def get_scale(self) -> float:
        """Get current scale factor."""
        return self._scale

    def set_scale(self, scale: float) -> None:
        """Set scale factor, updating UI."""
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        self._scale = scale
        self.scale_slider.blockSignals(True)
        self.scale_slider.setValue(int(scale * 100))
        self.scale_label.setText(f"{scale:.1f}x")
        self.scale_slider.blockSignals(False)

    def get_style(self) -> str:
        """Get current glyph style."""
        return self._style

    def set_style(self, style: str) -> None:
        """Set glyph style, updating UI."""
        if style not in GLYPH_STYLES:
            raise ValueError(f"Invalid style: {style}")
        self._style = style
        self.style_combo.blockSignals(True)
        self.style_combo.setCurrentText(style)
        self.style_combo.blockSignals(False)

    def get_colormap(self) -> str:
        """Get current colormap."""
        return self._colormap

    def set_colormap(self, colormap: str) -> None:
        """Set colormap, updating UI."""
        if colormap not in COLORMAPS:
            raise ValueError(f"Invalid colormap: {colormap}")
        self._colormap = colormap
        self.colormap_combo.blockSignals(True)
        self.colormap_combo.setCurrentText(colormap)
        self.colormap_combo.blockSignals(False)

    def get_opacity(self) -> float:
        """Get current opacity (0-1)."""
        return self._opacity

    def set_opacity(self, opacity: float) -> None:
        """Set opacity (0-1), updating UI."""
        if not 0 <= opacity <= 1:
            raise ValueError(f"opacity must be in [0, 1], got {opacity}")
        self._opacity = opacity
        self.opacity_slider.blockSignals(True)
        self.opacity_slider.setValue(int(opacity * 100))
        self.opacity_label.setText(f"{int(opacity * 100)}%")
        self.opacity_slider.blockSignals(False)
