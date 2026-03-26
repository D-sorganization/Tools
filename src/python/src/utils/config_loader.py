"""
Shared configuration loading utility for consistent configuration management.

This module provides reusable functions for loading and managing configuration
across the repository, following DRY principles.
"""

import logging
from pathlib import Path
from typing import Any

from .file_utils import safe_read_json, safe_write_json

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and manage configuration from JSON files."""

    def __init__(self, config_path: Path | str, default_config: dict[str, Any] | None = None):
        """Initialize configuration loader.

        Args:
            config_path: Path to configuration file
            default_config: Default configuration to use if file doesn't exist
        """
        if not (config_path is not None):
            raise ValueError("config_path must be provided")
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
        if not (reload is not None):
            raise ValueError("reload must be provided")
        if self._config is not None and not reload:
            return self._config

        self._config = safe_read_json(self.config_path, self.default_config.copy())
        return self._config

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.

        Args:
            key: Configuration key (supports dot notation, e.g., "section.key")
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        if not (key is not None):
            raise ValueError("key must be provided")
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
        if not (key is not None):
            raise ValueError("key must be provided")
        if self._config is None:
            self.load()

        keys = key.split(".")
        config = self._config

        # Navigate to the parent dict
        if self._config is None:
            self._config = {}
        config = self._config

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
    if not (config_path is not None):
        raise ValueError("config_path must be provided")
    loader = ConfigLoader(config_path, default_config)
    return loader.load()
