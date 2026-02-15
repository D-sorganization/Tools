# ruff: noqa: T201
"""Unified tool launcher for the Tools platform.

Usage:
    python launch.py --list          # List available tools
    python launch.py --tool <name>   # Launch a specific tool

This is the single canonical entry point for launching all GUI tools
in the repository. It replaces the legacy Launcher.py and individual
launch_pyqt6.py scripts.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

SRC_DIR = Path(__file__).resolve().parent / "src"


def _setup_logging() -> None:
    """Configure logging for the launcher."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified tool launcher for the Tools platform.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available tools.",
    )
    parser.add_argument(
        "--tool",
        type=str,
        default=None,
        help="Name of the tool to launch.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the unified launcher.

    Args:
        argv: Optional command-line arguments for testing.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    args = _parse_args(argv)
    _setup_logging()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        from gui_launcher.registry import GUIRegistry, auto_discover_guis
    except ImportError:
        logger.error("gui_launcher module not found. Install it first.")
        return 1

    # Discover available tools
    count = auto_discover_guis([SRC_DIR])
    registry = GUIRegistry.instance()
    tools = registry.list_tools()

    if args.list:
        print(f"\nAvailable tools ({count} discovered):\n")
        for tool in sorted(tools, key=lambda t: t.name):
            desc = getattr(tool, "description", "")
            print(f"  {tool.tool_name:<30} {desc}")
        print()
        GUIRegistry._instance = None
        return 0

    if args.tool:
        from gui_launcher.launcher import launch_from_gui_info

        # Find the tool by name
        matching = [t for t in tools if t.tool_name == args.tool]
        if not matching:
            logger.error(
                "Tool '%s' not found. Use --list to see available tools.", args.tool
            )
            GUIRegistry._instance = None
            return 1

        tool = matching[0]
        gui_info = {
            "name": tool.name,
            "tool_name": tool.tool_name,
            "description": getattr(tool, "description", ""),
        }
        # Add pyqt6 config if available
        if hasattr(tool, "gui_configs") and tool.gui_configs:
            from gui_launcher.launcher import GUIType

            pyqt6_config = tool.gui_configs.get(GUIType.PYQT6)
            if pyqt6_config:
                gui_info["pyqt6"] = {
                    "module": pyqt6_config.module,
                    "class": pyqt6_config.class_name,
                }

        GUIRegistry._instance = None
        return launch_from_gui_info(gui_info)

    # No --list or --tool: show help
    _parse_args(["--help"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
