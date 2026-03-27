"""Plot configuration management for saving and loading plot settings.

Provides persistence for plot configurations including:
- Signal selections
- Axis settings
- Color schemes
- Trendline settings
- Time range settings
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class PlotConfigManager:
    """Manages plot configuration persistence."""

    def __init__(
        self,
        config_dir: Path | str | None = None,
        filename: str = "plot_configs.json",
    ):
        """Initialize the plot configuration manager.

        Args:
            config_dir: Directory to store plot configs. Defaults to user's home.
            filename: Name of the config file.
        """
        if not (filename is not None):
            raise ValueError("filename must be provided")
        if config_dir is None:
            self.config_dir = Path.home() / ".data_processor"
        else:
            self.config_dir = Path(config_dir)

        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.config_dir / filename

    def save_plot_config(
        self,
        name: str,
        config: dict[str, Any],
    ) -> None:
        """Save a named plot configuration.

        Args:
            name: Unique name for this plot configuration
            config: Dictionary containing plot settings

        Expected config structure:
        {
            "name": "Plot Title",
            "description": "Optional description",
            "signals": ["signal1", "signal2"],
            "x_axis": "time",
            "chart_type": "line",  # line, scatter, bar
            "color_scheme": "default",
            "custom_colors": ["#ff0000", "#00ff00"],
            "legend": {
                "visible": True,
                "position": "right",
                "labels": {"signal1": "Custom Label"}
            },
            "trendline": {
                "enabled": True,
                "type": "linear",  # linear, polynomial, exponential, power
                "degree": 2,  # for polynomial
                "signals": ["signal1"],
                "time_window": {"start": None, "end": None}
            },
            "time_range": {
                "start": "10:00:00",
                "end": "14:00:00"
            },
            "axis_settings": {
                "x_label": "Time",
                "y_label": "Value",
                "y_min": None,
                "y_max": None
            },
            "filter_preview": {
                "enabled": True,
                "filter_type": "moving_average",
                "params": {"window_size": 10}
            }
        }
        """
        if not (name is not None):
            raise ValueError("name must be provided")
        configs = self._load_all_configs()

        configs[name] = {
            "config": config,
            "created": configs.get(name, {}).get("created", datetime.now().isoformat()),
            "modified": datetime.now().isoformat(),
        }

        self._save_all_configs(configs)

    def load_plot_config(self, name: str) -> dict[str, Any]:
        """Load a named plot configuration.

        Args:
            name: Name of the plot configuration to load

        Returns:
            Dictionary of plot settings

        Raises:
            KeyError: If configuration doesn't exist
        """
        configs = self._load_all_configs()

        if name not in configs:
            raise KeyError(f"Plot configuration not found: {name}")

        return configs[name]["config"]

    def delete_plot_config(self, name: str) -> None:
        """Delete a named plot configuration.

        Args:
            name: Name of the plot configuration to delete
        """
        if not (name is not None):
            raise ValueError("name must be provided")
        configs = self._load_all_configs()

        if name in configs:
            del configs[name]
            self._save_all_configs(configs)

    def list_plot_configs(self) -> list[str]:
        """List all saved plot configuration names.

        Returns:
            List of plot configuration names
        """
        configs = self._load_all_configs()
        return list(configs.keys())

    def get_plot_config_info(self, name: str) -> dict[str, Any]:
        """Get metadata about a plot configuration.

        Args:
            name: Name of the plot configuration

        Returns:
            Dictionary with metadata
        """
        configs = self._load_all_configs()

        if name not in configs:
            raise KeyError(f"Plot configuration not found: {name}")

        config = configs[name]["config"]
        return {
            "name": name,
            "title": config.get("name", name),
            "description": config.get("description", ""),
            "signal_count": len(config.get("signals", [])),
            "created": configs[name].get("created"),
            "modified": configs[name].get("modified"),
        }

    def duplicate_plot_config(self, source_name: str, new_name: str) -> None:
        """Create a copy of an existing plot configuration.

        Args:
            source_name: Name of the configuration to copy
            new_name: Name for the new configuration
        """
        if not (source_name is not None):
            raise ValueError("source_name must be provided")
        config = self.load_plot_config(source_name)
        config["name"] = new_name
        self.save_plot_config(new_name, config)

    def update_plot_config(self, name: str, updates: dict[str, Any]) -> None:
        """Update specific fields in a plot configuration.

        Args:
            name: Name of the configuration to update
            updates: Dictionary of fields to update
        """
        if not (name is not None):
            raise ValueError("name must be provided")
        config = self.load_plot_config(name)
        config.update(updates)
        self.save_plot_config(name, config)

    def export_plot_config(self, name: str, export_path: Path | str) -> None:
        """Export a plot configuration to a file.

        Args:
            name: Name of the plot configuration
            export_path: Path to export to
        """
        if not (name is not None):
            raise ValueError("name must be provided")
        config = self.load_plot_config(name)
        export_path = Path(export_path)

        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    def import_plot_config(
        self, import_path: Path | str, name: str | None = None
    ) -> str:
        """Import a plot configuration from a file.

        Args:
            import_path: Path to import from
            name: Optional name override

        Returns:
            Name of the imported configuration
        """
        if not (import_path is not None):
            raise ValueError("import_path must be provided")
        import_path = Path(import_path)

        with open(import_path, encoding="utf-8") as f:
            config = json.load(f)

        config_name = name or config.get("name", import_path.stem)
        self.save_plot_config(config_name, config)
        return config_name

    def export_all_plots(self, export_dir: Path | str) -> list[str]:
        """Export all plot configurations to a directory.

        Args:
            export_dir: Directory to export to

        Returns:
            List of exported file paths
        """
        if not (export_dir is not None):
            raise ValueError("export_dir must be provided")
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

        exported = []
        for name in self.list_plot_configs():
            file_path = export_dir / f"{name}.json"
            self.export_plot_config(name, file_path)
            exported.append(str(file_path))

        return exported

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
            json.dump(configs, f, indent=2)


__all__ = ["PlotConfigManager"]
