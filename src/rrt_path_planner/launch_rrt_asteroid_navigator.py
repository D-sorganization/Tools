#!/usr/bin/env python3
"""Standalone launcher for the RRT asteroid navigator."""

# ruff: noqa: E402, I001

from __future__ import annotations

import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parent
for _ in range(10):
    if (REPO_ROOT / "pyproject.toml").exists():
        break
    REPO_ROOT = REPO_ROOT.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from _bootstrap import bootstrap

REPO_ROOT = bootstrap(__file__)
PYTHON_SRC = REPO_ROOT / "src" / "scientific_modeling" / "rrt_path_planner" / "python" / "src"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

from star_wars_rrt import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
