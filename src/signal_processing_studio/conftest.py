"""Root conftest for signal_processing_studio — sets up PYTHONPATH."""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent

# Ensure the inner python/ subdir is first on sys.path for package discovery.
_python_dir = str(_root / "python")
if _python_dir not in sys.path:
    sys.path.insert(0, _python_dir)

# Ensure shared python (for signal_toolkit, etc.) is also on path.
_shared_python_dir = str(_root.parent / "shared" / "python")
if _shared_python_dir not in sys.path:
    sys.path.insert(1, _shared_python_dir)
