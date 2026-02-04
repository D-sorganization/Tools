"""
Inertia Calculator - GUI Registration
=====================================

Metadata for GUI framework integration.
"""

GUI_METADATA = {
    "name": "Inertia Calculator",
    "description": "Calculate and validate inertia tensors for rigid bodies",
    "category": "robotics",
    "version": "1.0.0",
    "entry_point": "inertia_calculator.ui.pyqt6.main_window:InertiaCalculatorWindow",
    "icon": "cube",
    "keywords": [
        "inertia",
        "tensor",
        "moment of inertia",
        "rigid body",
        "URDF",
        "robotics",
        "dynamics",
    ],
    "dependencies": {
        "required": ["PyQt6", "numpy"],
        "optional": ["trimesh"],
    },
    "features": [
        "Manual inertia input",
        "Primitive shape calculations (box, cylinder, sphere)",
        "Inertia tensor validation",
        "URDF format output",
        "Matrix visualization",
    ],
}


def get_metadata() -> dict:
    """Return GUI metadata for framework registration."""
    return GUI_METADATA
