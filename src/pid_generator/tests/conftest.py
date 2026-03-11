"""Pytest configuration for P&ID Generator tests.

Inserts the tool's Python source tree at the front of sys.path so that
'import pid_generator' resolves to the actual package (with __version__
and the ui subpackage), rather than the namespace package created by
pytest's default path discovery of the src/pid_generator/ tool directory.
"""

import sys
from pathlib import Path

_python_src = Path(__file__).resolve().parent.parent / "python"
if str(_python_src) not in sys.path:
    sys.path.insert(0, str(_python_src))
