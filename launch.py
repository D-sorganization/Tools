#!/usr/bin/env python3
"""Unified tool launcher for the Tools repository.

This is the single entry point for launching any registered PyQt6 tool.
It auto-discovers all tools via their gui_registration.py files and can
launch them by name.

Usage:
    # List all available tools
    python launch.py --list

    # Launch a specific tool by name
    python launch.py --tool "Pressure Drop Calculator"

    # Launch by tool_name identifier
    python launch.py --tool pressure_drop_calculator

    # Launch the unified tools launcher (default)
    python launch.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Bootstrap imports for development mode
_REPO_ROOT = Path(__file__).resolve().parent
_SHARED_PYTHON = _REPO_ROOT / "src" / "shared" / "python"
sys.path.insert(0, str(_SHARED_PYTHON))
from upstream_drift_tools.bootstrap import ensure_paths  # noqa: E402

ensure_paths(_REPO_ROOT)

from gui_launcher import launch_pyqt6_app  # noqa: E402
from gui_launcher.registry import auto_discover_guis, get_registry  # noqa: E402


def discover_all_tools() -> int:
    """Discover all tools from gui_registration.py files.

    Returns:
        Number of tools discovered.
    """
    src_dir = _REPO_ROOT / "src"
    return int(auto_discover_guis([src_dir]))


def list_tools() -> None:
    """Print all available tools grouped by category."""
    count = discover_all_tools()
    registry = get_registry()

    if count == 0:
        print("No tools found.")  # noqa: T201
        return

    print(f"Discovered {count} tool registrations.\n")  # noqa: T201

    categories = registry.list_categories()
    for category in categories:
        tools = registry.list_tools(category=category)
        if tools:
            print(f"  [{category}]")  # noqa: T201
            for tool in tools:
                print(f"    {tool.tool_name:40s} {tool.display_name}")  # noqa: T201
            print()  # noqa: T201


def launch_tool(tool_identifier: str) -> int:
    """Launch a tool by name or tool_name.

    Args:
        tool_identifier: Either the display name or the tool_name.

    Returns:
        Exit code.

    Raises:
        TypeError: If tool_identifier is not a string.
        ValueError: If tool_identifier is an empty string.
    """
    if not isinstance(tool_identifier, str):
        raise TypeError(f"tool_identifier must be a str, got {type(tool_identifier)}")
    if not tool_identifier:
        raise ValueError("tool_identifier must not be an empty string")

    discover_all_tools()
    registry = get_registry()

    # Try exact tool_name match first
    registration = registry.get(tool_identifier)

    # If not found, try matching by display name (case-insensitive)
    if registration is None:
        needle_lower = tool_identifier.lower()
        for tool in registry.list_tools():
            if tool.display_name.lower() == needle_lower:
                registration = tool
                break

    # If still not found, try partial match
    if registration is None:
        matches = []
        needle = tool_identifier.lower()
        for tool in registry.list_tools():
            tool_name_lower = tool.tool_name.lower()
            display_name_lower = tool.display_name.lower()
            if needle in tool_name_lower or needle in display_name_lower:
                matches.append(tool)
        if len(matches) == 1:
            registration = matches[0]
        elif len(matches) > 1:
            print(f"Ambiguous tool name '{tool_identifier}'. Matches:")  # noqa: T201
            for m in matches:
                print(f"  - {m.tool_name} ({m.display_name})")  # noqa: T201
            return 1

    if registration is None:
        print(f"Tool '{tool_identifier}' not found.")  # noqa: T201
        print("\nUse --list to see all available tools.")  # noqa: T201
        return 1

    from gui_launcher.launcher import GUIType

    gui_configs = registration.gui_configs
    config = gui_configs.get(GUIType.PYQT6)
    if config is None:
        display_name = registration.display_name
        print(f"Tool '{display_name}' has no PyQt6 configuration.")  # noqa: T201
        return 1

    display_name = registration.display_name
    print(f"Launching: {display_name}")  # noqa: T201
    return int(launch_pyqt6_app(config))


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Unified tool launcher for the Tools repository.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python launch.py --list
  python launch.py --tool "Pressure Drop Calculator"
  python launch.py --tool baghouse_calculator
  python launch.py                                    # Default: list tools
""",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available tools",
    )
    parser.add_argument(
        "--tool",
        type=str,
        help="Tool name or identifier to launch",
    )

    args = parser.parse_args()

    if args.list or (not args.tool):
        list_tools()
        return 0

    return launch_tool(args.tool)


if __name__ == "__main__":
    sys.exit(main())
