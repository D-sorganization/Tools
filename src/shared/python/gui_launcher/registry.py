"""GUI Registry for centralized tool registration and discovery."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .launcher import GUIType, LaunchConfig

logger = logging.getLogger(__name__)


@dataclass
class GUIRegistration:
    """Registration entry for a GUI component."""

    tool_name: str
    display_name: str
    description: str
    gui_configs: Dict[GUIType, LaunchConfig]
    category: str = "General"
    icon: Optional[str] = None
    repository: Optional[str] = None


class GUIRegistry:
    """Central registry for all available GUI components.

    This allows tools from different repositories to register their GUIs
    and be discovered by launcher applications.
    """

    _instance: Optional["GUIRegistry"] = None

    def __init__(self) -> None:
        """Initialize the registry."""
        self._registrations: Dict[str, GUIRegistration] = {}

    @classmethod
    def instance(cls) -> "GUIRegistry":
        """Get the singleton registry instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(
        self,
        tool_name: str,
        display_name: str,
        description: str,
        gui_configs: Dict[GUIType, LaunchConfig],
        category: str = "General",
        icon: Optional[str] = None,
        repository: Optional[str] = None,
    ) -> None:
        """Register a GUI component.

        Args:
            tool_name: Unique identifier for the tool
            display_name: Human-readable name
            description: Brief description of the tool
            gui_configs: Dictionary mapping GUIType to LaunchConfig
            category: Category for grouping tools
            icon: Optional path to icon file
            repository: Optional repository identifier
        """
        registration = GUIRegistration(
            tool_name=tool_name,
            display_name=display_name,
            description=description,
            gui_configs=gui_configs,
            category=category,
            icon=icon,
            repository=repository,
        )
        self._registrations[tool_name] = registration
        logger.debug(f"Registered GUI: {tool_name} ({len(gui_configs)} variants)")

    def unregister(self, tool_name: str) -> bool:
        """Unregister a GUI component.

        Args:
            tool_name: Identifier of the tool to unregister

        Returns:
            True if the tool was found and removed
        """
        if tool_name in self._registrations:
            del self._registrations[tool_name]
            return True
        return False

    def get(self, tool_name: str) -> Optional[GUIRegistration]:
        """Get a registration by tool name.

        Args:
            tool_name: Identifier of the tool

        Returns:
            GUIRegistration or None if not found
        """
        return self._registrations.get(tool_name)

    def get_config(
        self,
        tool_name: str,
        gui_type: GUIType,
    ) -> Optional[LaunchConfig]:
        """Get the launch config for a specific tool and GUI type.

        Args:
            tool_name: Identifier of the tool
            gui_type: Type of GUI to get config for

        Returns:
            LaunchConfig or None if not found
        """
        registration = self._registrations.get(tool_name)
        if registration:
            return registration.gui_configs.get(gui_type)
        return None

    def list_tools(self, category: Optional[str] = None) -> List[GUIRegistration]:
        """List all registered tools.

        Args:
            category: Optional category filter

        Returns:
            List of GUIRegistration objects
        """
        tools = list(self._registrations.values())
        if category:
            tools = [t for t in tools if t.category == category]
        return sorted(tools, key=lambda t: t.display_name)

    def list_categories(self) -> List[str]:
        """Get all unique categories.

        Returns:
            Sorted list of category names
        """
        categories = set(reg.category for reg in self._registrations.values())
        return sorted(categories)

    def get_available_gui_types(self, tool_name: str) -> List[GUIType]:
        """Get available GUI types for a tool.

        Args:
            tool_name: Identifier of the tool

        Returns:
            List of available GUIType values
        """
        registration = self._registrations.get(tool_name)
        if registration:
            return list(registration.gui_configs.keys())
        return []

    def clear(self) -> None:
        """Clear all registrations."""
        self._registrations.clear()


def get_registry() -> GUIRegistry:
    """Get the global GUI registry.

    Returns:
        The singleton GUIRegistry instance
    """
    return GUIRegistry.instance()


def register_gui(
    tool_name: str,
    display_name: str,
    description: str,
    gui_configs: Dict[GUIType, LaunchConfig],
    **kwargs,
) -> None:
    """Convenience function to register a GUI with the global registry.

    Args:
        tool_name: Unique identifier for the tool
        display_name: Human-readable name
        description: Brief description of the tool
        gui_configs: Dictionary mapping GUIType to LaunchConfig
        **kwargs: Additional registration options
    """
    registry = get_registry()
    registry.register(
        tool_name=tool_name,
        display_name=display_name,
        description=description,
        gui_configs=gui_configs,
        **kwargs,
    )


def auto_discover_guis(search_paths: List[Path]) -> int:
    """Automatically discover and register GUI components.

    Searches for gui_registration.py files in the given paths
    and executes them to register GUIs.

    Args:
        search_paths: List of paths to search for GUI registrations

    Returns:
        Number of GUIs registered
    """
    import importlib.util

    count = 0

    for search_path in search_paths:
        if not search_path.exists():
            continue

        for reg_file in search_path.rglob("gui_registration.py"):
            try:
                spec = importlib.util.spec_from_file_location(
                    f"gui_reg_{count}",
                    reg_file,
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    count += 1
                    logger.debug(f"Loaded GUI registration from: {reg_file}")
            except Exception as e:
                logger.warning(f"Failed to load GUI registration from {reg_file}: {e}")

    return count
