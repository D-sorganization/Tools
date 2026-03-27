"""Configuration management for saving and loading app settings.

Provides a unified interface for persisting configuration across sessions.
Works with all GUI implementations (TKinter, PyQt6, React).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from data_processor.contracts import require


class ConfigManager:
    """Manages application configuration persistence."""

    def __init__(
        self,
        config_dir: Path | str | None = None,
        config_filename: str = "data_processor_configs.json",
    ):
        """Initialize the configuration manager.

        Args:
            config_dir: Directory to store configurations. Defaults to user's home.
            config_filename: Name of the config file.
        """
        if not (config_filename is not None):
            raise ValueError("config_filename must be provided")
        if config_dir is None:
            self.config_dir = Path.home() / ".data_processor"
        else:
            self.config_dir = Path(config_dir)

        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.config_dir / config_filename

    def save_config(self, name: str, settings: dict[str, Any]) -> None:
        """Save a named configuration.

        **Pre-conditions** (DbC):
          - ``name`` must be a non-empty string.
          - ``settings`` must be a dict.
        """
        if not (name is not None):
            raise ValueError("name must be provided")
        require(
            isinstance(name, str) and bool(name.strip()), "name must be non-empty", name
        )
        require(isinstance(settings, dict), "settings must be a dict", type(settings))
        configs = self._load_all_configs()

        configs[name] = {
            "settings": settings,
            "created": configs.get(name, {}).get("created", datetime.now().isoformat()),
            "modified": datetime.now().isoformat(),
        }

        self._save_all_configs(configs)

    def load_config(self, name: str) -> dict[str, Any]:
        """Load a named configuration.

        Args:
            name: Name of the configuration to load

        Returns:
            Dictionary of settings

        Raises:
            KeyError: If configuration doesn't exist
        """
        configs = self._load_all_configs()

        if name not in configs:
            raise KeyError(f"Configuration not found: {name}")

        return configs[name]["settings"]

    def delete_config(self, name: str) -> None:
        """Delete a named configuration.

        Args:
            name: Name of the configuration to delete
        """
        if not (name is not None):
            raise ValueError("name must be provided")
        configs = self._load_all_configs()

        if name in configs:
            del configs[name]
            self._save_all_configs(configs)

    def list_configs(self) -> list[str]:
        """List all saved configuration names.

        Returns:
            List of configuration names
        """
        configs = self._load_all_configs()
        return list(configs.keys())

    def get_config_info(self, name: str) -> dict[str, Any]:
        """Get metadata about a configuration.

        Args:
            name: Name of the configuration

        Returns:
            Dictionary with created, modified timestamps
        """
        configs = self._load_all_configs()

        if name not in configs:
            raise KeyError(f"Configuration not found: {name}")

        return {
            "name": name,
            "created": configs[name].get("created"),
            "modified": configs[name].get("modified"),
        }

    def export_config(self, name: str, export_path: Path | str) -> None:
        """Export a configuration to a file.

        Args:
            name: Name of the configuration
            export_path: Path to export to
        """
        if not (name is not None):
            raise ValueError("name must be provided")
        settings = self.load_config(name)
        export_path = Path(export_path)

        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2, default=str)

    def import_config(self, name: str, import_path: Path | str) -> None:
        """Import a configuration from a file.

        Args:
            name: Name to save the configuration as
            import_path: Path to import from
        """
        if not (name is not None):
            raise ValueError("name must be provided")
        import_path = Path(import_path)

        with open(import_path, encoding="utf-8") as f:
            settings = json.load(f)

        self.save_config(name, settings)

    def _load_all_configs(self) -> dict[str, Any]:
        """Load all configurations from file."""
        if not self.config_file.exists():
            return {}

        try:
            with open(self.config_file, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_all_configs(self, configs: dict[str, Any]) -> None:
        """Save all configurations to file."""
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(configs, f, indent=2, default=str)


__all__ = ["ConfigManager"]
