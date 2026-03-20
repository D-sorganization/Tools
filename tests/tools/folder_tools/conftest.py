"""Conftest for folder_tools tests — make folder_tool package importable."""

from __future__ import annotations

import sys
from pathlib import Path

# Add folder_tool directory to sys.path for bare imports like
# ``from folder_tool_constants import ...``
_this_file = Path(__file__).resolve()
_repo_root = _this_file.parents[3]
_folder_tool_dir = str(_repo_root / "src" / "tools" / "folder_tools" / "folder_tool")
if _folder_tool_dir not in sys.path:
    sys.path.insert(0, _folder_tool_dir)
