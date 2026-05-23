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
from importlib import import_module
from pathlib import Path
from typing import Any

# Bootstrap imports for development mode
_REPO_ROOT = Path(__file__).resolve().parent
_SHARED_PYTHON = _REPO_ROOT / "src" / "shared" / "python"
sys.path.insert(0, str(_SHARED_PYTHON))


def _ensure_bootstrap_paths() -> None:
    """Load the bootstrap helper lazily to avoid static type-check import churn."""
    bootstrap = import_module("upstream_drift_tools.bootstrap")
    bootstrap.ensure_paths(_REPO_ROOT)


_ensure_bootstrap_paths()


def _emit_stdout(message: str = "") -> None:
    """Write a single line to stdout for the CLI contract."""
    sys.stdout.write(f"{message}\n")


def get_registry() -> Any:
    """Return the GUI registry module singleton."""
    return import_module("gui_launcher.registry").get_registry()


def discover_all_tools() -> int:
    """Discover all tools from gui_registration.py files.

    Returns:
        Number of tools discovered.
    """
    auto_discover_guis = import_module("gui_launcher.registry").auto_discover_guis
    src_dir = _REPO_ROOT / "src"
    return int(auto_discover_guis([src_dir]))


def list_tools() -> None:
    """Print all available tools grouped by category."""
    count = discover_all_tools()
    registry = get_registry()

    if count == 0:
        _emit_stdout("No tools found.")
        return

    _emit_stdout(f"Discovered {count} tool registrations.")
    _emit_stdout()

    categories = registry.list_categories()
    for category in categories:
        tools = registry.list_tools(category=category)
        if tools:
            _emit_stdout(f"  [{category}]")
            for tool in tools:
                _emit_stdout(f"    {tool.tool_name:40s} {tool.display_name}")
            _emit_stdout()


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
            _emit_stdout(f"Ambiguous tool name '{tool_identifier}'. Matches:")
            for m in matches:
                _emit_stdout(f"  - {m.tool_name} ({m.display_name})")
            return 1

    if registration is None:
        _emit_stdout(f"Tool '{tool_identifier}' not found.")
        _emit_stdout()
        _emit_stdout("Use --list to see all available tools.")
        return 1

    GUIType = import_module("gui_launcher.launcher").GUIType
    launch_pyqt6_app = import_module("gui_launcher").launch_pyqt6_app

    gui_configs = registration.gui_configs
    config = gui_configs.get(GUIType.PYQT6)
    if config is None:
        print(
            f"Tool '{registration.display_name}' has no PyQt6 configuration."
        )  # noqa: T201
        return 1

    display_name = registration.display_name
    _emit_stdout(f"Launching: {display_name}")
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
