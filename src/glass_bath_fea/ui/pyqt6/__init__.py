"""PyQt6 user interface for Glass Bath FEA."""

try:
    from .main_window import GlassBathFEAWidget

    __all__ = ["GlassBathFEAWidget"]
except ImportError:
    # PyQt6 not available
    __all__ = []
