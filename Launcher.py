# ruff: noqa: T201
"""DEPRECATED: Legacy launcher entry point.

This file is deprecated. Use the unified launcher instead:

    python launch.py --tool <tool_name>

Or use the gui_launcher module directly:

    from gui_launcher.registry import auto_discover_guis
    from gui_launcher.launcher import launch_from_gui_info

See launch.py for the new unified entry point.
"""

import logging
import warnings

logger = logging.getLogger(__name__)


def main() -> None:
    """Legacy launcher main — deprecated."""
    warnings.warn(
        "Launcher.py is deprecated. Use 'python launch.py --tool <name>' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    logger.warning("Launcher.py is deprecated. Use launch.py instead.")
    print(
        "WARNING: Launcher.py is deprecated. "
        "Use 'python launch.py --tool <tool_name>' instead."
    )


if __name__ == "__main__":
    main()
