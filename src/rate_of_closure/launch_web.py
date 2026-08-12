#!/usr/bin/env python3
"""Launch the Rate of Closure Impact Explorer React web application."""

from __future__ import annotations

import sys
from pathlib import Path

# Add repository root to sys.path to allow importing _bootstrap
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from _bootstrap import bootstrap  # noqa: E402

_REPO_ROOT = bootstrap(__file__)

from shared.python.gui_launcher import launch_web_from_gui_info  # noqa: E402

from .application.morris.runtime import MorrisAuthorityRuntime  # noqa: E402
from .gui_registration import GUI_INFO  # noqa: E402


def main() -> int:
    """Own the private Morris authority for the complete Vite lifecycle."""
    with MorrisAuthorityRuntime.start() as authority:
        result: int = launch_web_from_gui_info(
            GUI_INFO, __file__, env_vars=authority.vite_env
        )
        return result


if __name__ == "__main__":
    sys.exit(main())
