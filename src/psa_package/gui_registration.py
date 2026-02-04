"""
PSA Package - GUI Registration
==============================

Metadata for GUI framework integration.
"""

GUI_METADATA = {
    "name": "PSA System Analysis",
    "description": "Two-stage Pressure Swing Adsorption system analysis with sensitivity analysis and O2 safety calculations",
    "category": "process_simulation",
    "version": "1.0.0",
    "entry_point": "upstream_drift_tools.process_calculators.psa_package.psa_gui:PSAMainWindow",
    "icon": "filter",
    "keywords": [
        "psa",
        "pressure swing adsorption",
        "hydrogen",
        "separation",
        "gas processing",
        "sensitivity analysis",
    ],
    "dependencies": {
        "required": ["PyQt6", "numpy", "matplotlib"],
        "optional": ["streamlit", "jupyter"],
    },
    "features": [
        "Two-stage PSA mass balance",
        "Component-wise flow calculations",
        "H2 recovery and purity metrics",
        "O2 safety analysis",
        "Interactive sensitivity plots",
        "3D surface and contour visualizations",
        "Process flow diagram view",
    ],
}


def get_metadata() -> dict:
    """Return GUI metadata for framework registration."""
    return GUI_METADATA
