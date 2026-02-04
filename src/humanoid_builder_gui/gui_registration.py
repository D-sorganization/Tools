"""
Humanoid Character Builder - GUI Registration
==============================================

Metadata for GUI framework integration.
"""

GUI_METADATA = {
    "name": "Humanoid Character Builder",
    "description": "Build parametric humanoid characters with anthropometric calculations",
    "category": "robotics",
    "version": "1.0.0",
    "entry_point": "humanoid_builder_gui.ui.pyqt6.main_window:HumanoidBuilderWindow",
    "icon": "person",
    "keywords": [
        "humanoid",
        "character",
        "anthropometry",
        "URDF",
        "biomechanics",
        "body",
        "mesh",
        "inertia",
    ],
    "dependencies": {
        "required": ["PyQt6"],
        "optional": ["numpy", "trimesh"],
    },
    "features": [
        "Height-based anthropometry calculations",
        "Mass distribution by body segment",
        "Configurable body proportions",
        "Build type presets (ectomorph, mesomorph, endomorph)",
        "Gender-specific anthropometric data",
        "URDF export",
        "Segment details table",
        "BMI calculation",
    ],
}


def get_metadata() -> dict:
    """Return GUI metadata for framework registration."""
    return GUI_METADATA
