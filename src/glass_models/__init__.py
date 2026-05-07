"""Glass bath FEA models and visualization tools.

This package provides FEA (Finite Element Analysis) models and visualization
tools for glass furnace simulations, including iso-surface rendering,
field visualization, and high-resolution publication-quality export.
"""

__version__ = "0.1.0"

# Lazy imports to avoid requiring all dependencies
try:
    from glass_models.ui.pyqt6.export_dialog import (
        ExportConfig,
        ExportProgressDialog,
        HighResolutionExportDialog,
    )
except ImportError:
    # PyQt6 not available
    pass

try:
    from glass_models.viz.high_res_renderer import HighResolutionRenderer
except ImportError:
    # PyVista not available
    pass

__all__ = [
    "HighResolutionRenderer",
    "HighResolutionExportDialog",
    "ExportConfig",
    "ExportProgressDialog",
]
