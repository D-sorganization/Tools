# URDF Builder GUI
"""Parametric URDF Builder GUI for standalone use.

This __init__.py bridges the tool root to the canonical package location
at python/urdf_builder_gui/ by extending the package's __path__.  When
Python imports ``urdf_builder_gui`` from this directory it will find
submodules (contracts, anthropometric_model, etc.) inside the python/
sub-package tree rather than expecting duplicate files here.
"""

from __future__ import annotations

from pathlib import Path

# Canonical package directory — the single source of truth for all modules.
_CANONICAL_PKG = Path(__file__).resolve().parent / "python" / "urdf_builder_gui"

# Extend __path__ so that sub-module lookups (e.g. urdf_builder_gui.contracts)
# resolve into the canonical directory.  Use insert(0, …) so the canonical
# tree always wins even if a flat copy is accidentally re-introduced here
# (issue #3346 / GH1693).
_canonical_str = str(_CANONICAL_PKG)
if _canonical_str not in __path__:
    __path__.insert(0, _canonical_str)

__version__ = "1.0.0"
