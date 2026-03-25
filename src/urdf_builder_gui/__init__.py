# URDF Builder GUI
"""Parametric URDF Builder GUI for standalone use.

This __init__.py bridges the tool root to the canonical package location
at python/urdf_builder_gui/ by extending the package's __path__.  When
Python imports ``urdf_builder_gui`` from this directory it will find
submodules (contracts, anthropometric_model, etc.) inside the python/
sub-package tree rather than expecting duplicate files here.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Canonical package directory — the single source of truth for all modules.
_CANONICAL_PKG = Path(__file__).resolve().parent / "python" / "urdf_builder_gui"

# Extend __path__ so that sub-module lookups (e.g. urdf_builder_gui.contracts)
# resolve into the canonical directory even when Python found this __init__.py
# first via the tool-root sys.path entry.
_canonical_str = str(_CANONICAL_PKG)
if _canonical_str not in __path__:  # type: ignore[name-defined]
    __path__.append(_canonical_str)  # type: ignore[name-defined]

# Also ensure the python/ directory itself is on sys.path for standalone
# launcher scripts that do bare ``import urdf_builder_gui``.
_PYTHON_DIR = str(_CANONICAL_PKG.parent)
if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)

__version__ = "1.0.0"
