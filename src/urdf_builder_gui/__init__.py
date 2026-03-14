# URDF Builder GUI
"""Parametric URDF Builder GUI for standalone use.

This __init__.py re-exports the core modules from the python/
subdirectory for seamless imports.
"""

import importlib
import sys
from pathlib import Path

# Add the python/ directory so the real package modules are importable.
_PYTHON_DIR = str(Path(__file__).resolve().parent / "python")
if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)

# Import and expose the real package's submodules.
# This allows `from urdf_builder_gui.contracts import require` to work
# whether Python resolves this directory or python/urdf_builder_gui/.
_real_pkg_dir = Path(__file__).resolve().parent / "python" / "urdf_builder_gui"

for _mod_path in _real_pkg_dir.glob("*.py"):
    _mod_name = _mod_path.stem
    if _mod_name.startswith("_"):
        continue
    try:
        _mod = importlib.import_module(f"urdf_builder_gui.{_mod_name}")
    except ImportError:
        # Try from the python/ path
        try:
            _spec = importlib.util.spec_from_file_location(
                f"urdf_builder_gui.{_mod_name}", str(_mod_path)
            )
            if _spec and _spec.loader:
                _mod = importlib.util.module_from_spec(_spec)
                sys.modules[f"urdf_builder_gui.{_mod_name}"] = _mod
                _spec.loader.exec_module(_mod)
        except Exception:
            pass

__version__ = "1.0.0"
