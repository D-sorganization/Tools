"""
Shared configuration loading utility for consistent configuration management.

This module provides reusable functions for loading and managing configuration
across the repository, following DRY principles.

Includes XDG Base Directory Specification support via ``get_xdg_config_dir``.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any

from .file_utils import safe_read_json, safe_write_json

logger = logging.getLogger(__name__)


def get_xdg_config_dir(app_name: str) -> Path:
    """Return the XDG-compliant configuration directory for *app_name*.

    Follows the XDG Base Directory Specification:
    - Linux/macOS: ``$XDG_CONFIG_HOME/<app_name>`` (defaults to
      ``~/.config/<app_name>`` when ``XDG_CONFIG_HOME`` is not set).
    - Windows: ``%APPDATA%/<app_name>`` (defaults to
      ``~/.config/<app_name>`` when ``APPDATA`` is not set).

    The directory is **not** created by this function — the caller is
    responsible for creating it when needed.

    Args:
        app_name: Application name used as the subdirectory.  Must be a
            non-empty string and must not contain path separators.

    Returns:
        Absolute ``Path`` to the application configuration directory.

    Raises:
        TypeError: If *app_name* is not a string.
        ValueError: If *app_name* is empty or contains path separators.
    """
    if not isinstance(app_name, str):
        raise TypeError(f"app_name must be a str, got {type(app_name).__name__}")
    if not app_name:
        raise ValueError("app_name must not be empty")
    if "/" in app_name or "\\" in app_name:
        raise ValueError(f"app_name must not contain path separators, got {app_name!r}")

    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / ".config")
    else:
        xdg_home = os.environ.get("XDG_CONFIG_HOME", "")
        base = Path(xdg_home) if xdg_home else Path.home() / ".config"

    return base / app_name


class ConfigLoader:
    """Load and manage configuration from JSON files."""

    def __init__(
        self, config_path: Path | str, default_config: dict[str, Any] | None = None
    ):
        """Initialize configuration loader.

        Args:
            config_path: Path to configuration file
            default_config: Default configuration to use if file doesn't exist
        """
        assert config_path is not None, "config_path must be provided"
        self.config_path = Path(config_path)
        self.default_config = default_config or {}
        self._config: dict[str, Any] | None = None

    def load(self, reload: bool = False) -> dict[str, Any]:
        """Load configuration from file.

        Args:
            reload: Force reload even if already loaded

        Returns:
            Configuration dictionary
        """
        assert reload is not None, "reload must be provided"
        if self._config is not None and not reload:
            return self._config

        loaded = safe_read_json(self.config_path, self.default_config.copy())
        self._config = (
            loaded if isinstance(loaded, dict) else self.default_config.copy()
        )
        return self._config

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.

        Args:
            key: Configuration key (supports dot notation, e.g., "section.key")
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        assert key is not None, "key must be provided"
        if self._config is None:
            self.load()

        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        assert key is not None, "key must be provided"
        if self._config is None:
            self.load()
        if self._config is None:
            self._config = {}

        keys = key.split(".")
        config: dict[str, Any] = self._config

        # Navigate to the parent dict
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # Set the value
        config[keys[-1]] = value

    def save(self) -> bool:
        """Save configuration to file.

        Returns:
            True if save succeeded, False otherwise
        """
        if self._config is None:
            logger.warning("No configuration loaded to save")
            return False

        return safe_write_json(self.config_path, self._config)

    def reload(self) -> dict[str, Any]:
        """Reload configuration from file.

        Returns:
            Reloaded configuration dictionary
        """
        return self.load(reload=True)


def load_config(
    config_path: Path | str,
    default_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convenience function to load configuration.

    Args:
        config_path: Path to configuration file
        default_config: Default configuration to use if file doesn't exist

    Returns:
        Configuration dictionary
    """
    assert config_path is not None, "config_path must be provided"
    loader = ConfigLoader(config_path, default_config)
    return loader.load()
