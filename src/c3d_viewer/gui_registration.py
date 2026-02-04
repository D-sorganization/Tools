"""
C3D Motion Capture Viewer - GUI Registration
=============================================

Metadata for GUI framework integration.
"""

GUI_METADATA = {
    "name": "C3D Motion Capture Viewer",
    "description": "View and analyze C3D motion capture files",
    "category": "biomechanics",
    "version": "1.0.0",
    "entry_point": "c3d_viewer.ui.pyqt6.main_window:C3DViewerWindow",
    "icon": "body",
    "keywords": [
        "C3D",
        "motion capture",
        "biomechanics",
        "markers",
        "force plate",
        "trajectory",
        "gait analysis",
    ],
    "dependencies": {
        "required": ["PyQt6"],
        "optional": ["ezc3d", "pandas", "numpy"],
    },
    "features": [
        "C3D file parsing and loading",
        "Marker label and trajectory display",
        "Analog channel visualization",
        "Force plate analysis",
        "Event marker display",
        "Export to CSV, JSON, NPZ",
        "Unit conversion support",
    ],
}


def get_metadata() -> dict:
    """Return GUI metadata for framework registration."""
    return GUI_METADATA
