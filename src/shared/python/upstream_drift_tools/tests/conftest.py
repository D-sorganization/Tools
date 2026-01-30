
import sys
from pathlib import Path

# Add the source directory to sys.path to allow imports of upstream_drift_tools
# This assumes the tests are located at src/shared/python/upstream_drift_tools/tests
# and the package root is at src/shared/python
TEST_DIR = Path(__file__).resolve().parent
SHARED_PYTHON_DIR = TEST_DIR.parents[1]
sys.path.insert(0, str(SHARED_PYTHON_DIR))
