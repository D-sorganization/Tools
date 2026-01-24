"""Pytest configuration for data_processor tests."""

from pathlib import Path

try:
    from utils.path_helpers import ensure_utils_in_path
except ImportError:

    def ensure_utils_in_path():
        pass


# Add parent directory to Python path for imports
test_dir = Path(__file__).parent
project_root = test_dir.parent
ensure_utils_in_path()
