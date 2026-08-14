"""Pytest configuration for pdf_renamer tests.

Uses shared path setup from utils.path_helpers, and puts this self-contained
sub-app's own ``src`` root on ``sys.path`` so ``import pdf_renamer`` resolves.

The repository-level ``[tool.pytest.ini_options] pythonpath`` list does not
carry this sub-app, so before this hook existed every module in this directory
failed at collection with ``ModuleNotFoundError: No module named 'pdf_renamer'``.
That went unnoticed because CI only selects this directory when a pdf_renamer
source file changes (``scripts/select_tests_for_changes.py``). Keeping the entry
local to this conftest avoids adding a second top-level ``pdf_renamer`` import
root for the whole monorepo.
"""

import sys
from pathlib import Path

from utils.path_helpers import ensure_utils_in_path

# Ensure utils is available for test imports
ensure_utils_in_path()

_SUBAPP_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SUBAPP_SRC) not in sys.path:
    sys.path.insert(0, str(_SUBAPP_SRC))
