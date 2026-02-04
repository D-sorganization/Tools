"""
Parametric URDF Builder - GUI Registration
==========================================

Metadata for GUI framework integration.
"""

GUI_METADATA = {
    "name": "Parametric URDF Builder",
    "description": "Generate parametric URDF models for robotics applications",
    "category": "robotics",
    "version": "1.0.0",
    "entry_point": "urdf_builder_gui.ui.pyqt6.main_window:URDFBuilderWindow",
    "icon": "robot",
    "keywords": [
        "URDF",
        "parametric",
        "robotics",
        "humanoid",
        "model generation",
        "simulation",
        "ROS",
    ],
    "dependencies": {
        "required": ["PyQt6"],
        "optional": ["model_generation"],
    },
    "features": [
        "Height/weight-based parametric generation",
        "Gender factor adjustment",
        "Body proportion customization",
        "Geometry type selection",
        "Joint parameter configuration",
        "URDF file export",
        "Structure preview",
    ],
}


def get_metadata() -> dict:
    """Return GUI metadata for framework registration."""
    return GUI_METADATA
