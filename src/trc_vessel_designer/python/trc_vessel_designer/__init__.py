"""TRC Vessel Designer - Thermal Reaction Chamber Design Tool.

Provides UI components for TRC vessel design and analysis.
Uses the shared TRC geometry engine from upstream_drift_tools.
"""

from __future__ import annotations

# Re-export the shared engine components
from shared.python.upstream_drift_tools.calculators.mechanical.trc_geometry import (
    LayerConfig,
    LayerResult,
    TRCGeometryEngine,
    VesselDimensions,
    VesselGeometryResult,
)

# Import UI components when available
try:
    from .ui.pyqt6.main_window import TRCVesselDesignerWidget
except ImportError:
    TRCVesselDesignerWidget = None  # type: ignore[assignment, misc]

__all__ = [
    "TRCVesselDesignerWidget",
    "TRCGeometryEngine",
    "VesselDimensions",
    "LayerConfig",
    "LayerResult",
    "VesselGeometryResult",
]
