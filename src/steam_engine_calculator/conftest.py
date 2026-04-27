"""Root conftest for steam_engine_calculator — sets up PYTHONPATH."""

import sys
from pathlib import Path

# Ensure the python/ subdir is first on sys.path for package discovery.
_python_dir = str(Path(__file__).resolve().parent / "python")
if _python_dir not in sys.path:
    sys.path.insert(0, _python_dir)
