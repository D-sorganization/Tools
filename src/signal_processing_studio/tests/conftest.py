"""Configure test path for signal_processing_studio tests."""

import sys
from pathlib import Path

# Prepend the python/ directory so tests import the real package
# instead of the top-level __init__.py placeholder.
_python_dir = str(Path(__file__).resolve().parent.parent / "python")
if _python_dir not in sys.path:
    sys.path.insert(0, _python_dir)

# Also ensure shared python (for signal_toolkit, etc.) is on path.
_shared_python_dir = str(Path(__file__).resolve().parents[3] / "shared" / "python")
if _shared_python_dir not in sys.path:
    sys.path.insert(1, _shared_python_dir)
