"""FEA results viewer with iso-surface visualization.

Integrates iso-surface rendering into a 3D visualization viewer.
Supports:
- Multiple overlaid iso-surfaces
- Color surfaces by value or secondary field
- Real-time iso-value adjustment
- Actor management (add/remove)
"""

import logging
from typing import Any

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QLabel,
    QMessageBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from glass_models.viz.isosurface import IsoSurfaceExtractor

from .isosurface_widget import IsoSurfaceControlWidget

logger = logging.getLogger(__name__)


class FEAResultsViewer(QWidget):
    """FEA results viewer with iso-surface visualization.

    This widget combines a 3D viewer (placeholder) with iso-surface
    extraction and rendering controls.

    Signals:
        visualization_updated: Emitted when visualization is updated
    """

    visualization_updated = pyqtSignal(dict)

    def __init__(
        self,
        parent: QWidget | None = None,
        field_data: np.ndarray | None = None,
    ) -> None:
        """Initialize FEA results viewer.

        Args:
            parent: Parent widget
            field_data: 3D scalar field data (optional)
        """
        super().__init__(parent)
        self.field_data = field_data
        self.extractor = IsoSurfaceExtractor()
        self._iso_surfaces: dict[float, dict[str, Any]] = {}
        self._actors: list[Any] = []

        logger.debug("FEAResultsViewer initialized")

        self._setup_ui()
        self._connect_signals()

        if field_data is not None:
            self._update_field_range()

    def _setup_ui(self) -> None:
        """Set up the UI."""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("FEA Results Viewer with Iso-Surface Rendering")
        title.setStyleSheet("font-weight: bold; font-size: 12pt;")
        layout.addWidget(title)

        # Main content splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Control panel
        self.control_widget = IsoSurfaceControlWidget(
            field_range=(0.0, 1.0)
            if self.field_data is None
            else (
                float(np.nanmin(self.field_data)),
                float(np.nanmax(self.field_data)),
            )
        )
        splitter.addWidget(self.control_widget)

        # Right: 3D viewer placeholder
        viewer_placeholder = QWidget()
        viewer_layout = QVBoxLayout(viewer_placeholder)
        viewer_label = QLabel(
            "3D Viewer\n(VTK/PyVista integration would be added here)"
        )
        viewer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        viewer_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        viewer_layout.addWidget(viewer_label)
        splitter.addWidget(viewer_placeholder)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        layout.addWidget(splitter)

        self.setLayout(layout)

    def _connect_signals(self) -> None:
        """Connect control signals."""
        self.control_widget.iso_value_changed.connect(self._on_iso_value_changed)
        self.control_widget.iso_values_changed.connect(self._on_iso_values_changed)
        self.control_widget.transparency_changed.connect(self._on_transparency_changed)
        self.control_widget.colormap_changed.connect(self._on_colormap_changed)
        self.control_widget.surface_enabled_changed.connect(self._on_surface_enabled)

    def _update_field_range(self) -> None:
        """Update control widget with field data range."""
        if self.field_data is None:
            return

        field_min = float(np.nanmin(self.field_data))
        field_max = float(np.nanmax(self.field_data))

        self.control_widget.set_field_range(field_min, field_max)
        logger.debug("Field range: [%.3f, %.3f]", field_min, field_max)

    def _on_iso_value_changed(self, iso_value: float) -> None:
        """Handle single iso-value change."""
        if self.field_data is None:
            logger.warning("No field data loaded")
            return

        if not self.control_widget.is_enabled():
            return

        logger.debug("Extracting iso-surface at value %.3f", iso_value)

        try:
            surface = self.extractor.extract(self.field_data, iso_value)

            if surface is None or len(surface.get("vertices", [])) == 0:
                logger.warning("Iso-surface extraction failed for %.3f", iso_value)
                return

            self._iso_surfaces[iso_value] = surface
            self._update_visualization()

        except Exception as e:
            logger.error("Error extracting iso-surface: %s", str(e))
            QMessageBox.warning(
                self,
                "Extraction Error",
                f"Failed to extract iso-surface: {str(e)}",
            )

    def _on_iso_values_changed(self, iso_values: list[float]) -> None:
        """Handle multi-level iso-value change."""
        if self.field_data is None:
            logger.warning("No field data loaded")
            return

        if not self.control_widget.is_enabled():
            return

        logger.debug("Extracting %d iso-surfaces", len(iso_values))

        try:
            surfaces = self.extractor.extract_multiple(self.field_data, iso_values)

            self._iso_surfaces.clear()
            for surface in surfaces:
                iso_val = surface["field_value"]
                self._iso_surfaces[iso_val] = surface

            logger.debug("Extracted %d surfaces", len(self._iso_surfaces))
            self._update_visualization()

        except Exception as e:
            logger.error("Error extracting iso-surfaces: %s", str(e))
            QMessageBox.warning(
                self,
                "Extraction Error",
                f"Failed to extract iso-surfaces: {str(e)}",
            )

    def _on_transparency_changed(self, alpha: int) -> None:
        """Handle transparency change."""
        logger.debug("Transparency changed to %d%%", alpha)
        self._update_visualization()

    def _on_colormap_changed(self, colormap: str) -> None:
        """Handle colormap/secondary field change."""
        logger.debug("Colormap changed to %s", colormap)
        self._update_visualization()

    def _on_surface_enabled(self, enabled: bool) -> None:
        """Handle surface enable/disable."""
        logger.debug("Surface rendering %s", "enabled" if enabled else "disabled")
        if not enabled:
            self._clear_actors()
        else:
            self._update_visualization()

    def _update_visualization(self) -> None:
        """Update the 3D visualization."""
        self._clear_actors()

        if not self.control_widget.is_enabled():
            return

        alpha = self.control_widget.get_transparency() / 100.0
        colormap = self.control_widget.get_colormap()

        # Create visualization info
        viz_info = {
            "surfaces": self._iso_surfaces,
            "transparency": alpha,
            "colormap": colormap,
            "num_surfaces": len(self._iso_surfaces),
        }

        logger.debug(
            "Updated visualization: %d surfaces, alpha=%.2f, colormap=%s",
            len(self._iso_surfaces),
            alpha,
            colormap,
        )

        self.visualization_updated.emit(viz_info)

    def _clear_actors(self) -> None:
        """Clear visualization actors."""
        self._actors.clear()
        logger.debug("Cleared visualization actors")

    def load_field_data(self, field_data: np.ndarray) -> None:
        """Load field data for visualization.

        Args:
            field_data: 3D scalar field array
        """
        if field_data.ndim != 3:
            raise ValueError(f"Expected 3D field, got shape {field_data.shape}")

        self.field_data = field_data
        self.extractor.clear_cache()
        self._iso_surfaces.clear()

        self._update_field_range()
        logger.info(
            "Loaded field data: shape %s, range [%.3f, %.3f]",
            field_data.shape,
            float(np.nanmin(field_data)),
            float(np.nanmax(field_data)),
        )

    def get_surfaces(self) -> dict[float, dict[str, Any]]:
        """Get extracted iso-surfaces.

        Returns:
            Dictionary mapping iso-value to surface data
        """
        return self._iso_surfaces.copy()

    def clear(self) -> None:
        """Clear all data and visualization."""
        self.field_data = None
        self._iso_surfaces.clear()
        self._clear_actors()
        self.extractor.clear_cache()
        logger.debug("FEAResultsViewer cleared")


__all__ = ["FEAResultsViewer"]
