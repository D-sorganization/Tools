"""Pytest configuration for tests.

This file sets up the test environment, including PYTHONPATH configuration
to ensure imports work correctly in both local and CI environments.
"""

import sys
from pathlib import Path

try:
    from utils.path_helpers import ensure_utils_in_path
except ImportError:

    def ensure_utils_in_path() -> None:
        pass


# Add python/src to PYTHONPATH for test imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    ensure_utils_in_path()
