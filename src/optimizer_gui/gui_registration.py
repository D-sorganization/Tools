"""
Adam Optimizer - GUI Registration
=================================

Metadata for GUI framework integration.
"""

GUI_METADATA = {
    "name": "Adam Optimizer",
    "description": "Configure and run Adam-based optimization",
    "category": "optimization",
    "version": "1.0.0",
    "entry_point": "optimizer_gui.ui.pyqt6.main_window:OptimizerWindow",
    "icon": "chart-line",
    "keywords": [
        "Adam",
        "optimizer",
        "gradient descent",
        "machine learning",
        "parameter tuning",
        "convergence",
    ],
    "dependencies": {
        "required": ["PyQt6", "numpy"],
        "optional": ["scipy"],
    },
    "features": [
        "Adam optimizer hyperparameter configuration",
        "Parameter bounds setup",
        "Convergence tolerance settings",
        "History tracking and visualization",
        "Multiple optimization methods (Adam, Grid Search, L-BFGS-B)",
        "Demo mode with Rosenbrock function",
    ],
}


def get_metadata() -> dict:
    """Return GUI metadata for framework registration."""
    return GUI_METADATA
