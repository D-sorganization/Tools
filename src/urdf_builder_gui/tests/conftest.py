"""Configure test path for urdf_builder_gui tests."""

import sys
from pathlib import Path

# Prepend the python/ directory so tests import the real package
# instead of the top-level __init__.py placeholder.
_python_dir = str(Path(__file__).resolve().parent.parent / "python")
if _python_dir not in sys.path:
    sys.path.insert(0, _python_dir)
