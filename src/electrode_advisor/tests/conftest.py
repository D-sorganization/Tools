"""Path setup for electrode_advisor tests."""

import sys
from pathlib import Path

# Add the python/ subdirectory so `electrode_advisor` is importable
_PYTHON_DIR = str(Path(__file__).resolve().parent.parent / "python")
if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)

# Add shared python for upstream_drift_tools etc.
_SHARED_DIR = str(Path(__file__).resolve().parent.parent.parent / "shared" / "python")
if _SHARED_DIR not in sys.path:
    sys.path.insert(0, _SHARED_DIR)
