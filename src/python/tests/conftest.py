"""Pytest configuration for python/src tests.

Uses shared path setup from utils.path_helpers.
"""

from utils.path_helpers import ensure_utils_in_path

# Ensure utils is available for test imports
ensure_utils_in_path()
