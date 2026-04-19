# URDF Builder GUI
"""Parametric URDF Builder GUI for standalone use.

This __init__.py ensures the python/ subdirectory is on sys.path
so that all package submodules are importable.
"""

import sys
from pathlib import Path

# Ensure the python/ subdirectory is on sys.path so that the real package
# modules (contracts, anthropometric_model, urdf_generator) are findable.
_PYTHON_DIR = str(Path(__file__).resolve().parent / "python")
if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)

__version__ = "1.0.0"
