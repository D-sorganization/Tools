"""
Multi-Parameter Analysis - GUI Registration
============================================

Metadata for GUI framework integration.
"""

GUI_METADATA = {
    "name": "Multi-Parameter Analysis",
    "description": "Run multi-parameter sensitivity analysis with grid evaluation",
    "category": "analysis",
    "version": "1.0.0",
    "entry_point": "multi_param_analysis.ui.pyqt6.main_window:MultiParamAnalysisWindow",
    "icon": "grid",
    "keywords": [
        "multi-parameter",
        "sensitivity",
        "analysis",
        "grid",
        "heatmap",
        "surface plot",
        "parallel",
    ],
    "dependencies": {
        "required": ["PyQt6", "numpy"],
        "optional": [],
    },
    "features": [
        "2D parameter grid evaluation",
        "Parallel processing support",
        "Variance-based sensitivity analysis",
        "Multiple demo functions (Rosenbrock, Rastrigin, etc.)",
        "Statistics summary",
        "Data preview",
    ],
}


def get_metadata() -> dict:
    """Return GUI metadata for framework registration."""
    return GUI_METADATA
