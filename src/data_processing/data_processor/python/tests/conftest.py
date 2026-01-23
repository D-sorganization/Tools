"""Pytest configuration for data_processor tests."""

import sys
from pathlib import Path

# Add parent directory to Python path for imports
test_dir = Path(__file__).parent
project_root = test_dir.parent
sys.path.insert(0, str(project_root))
