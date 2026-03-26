"""Signal list management for saving and loading signal selections.

Provides persistence for user's signal selections across sessions.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class SignalListManager:
    """Manages signal list persistence."""

    def __init__(
        self,
        config_dir: Path | str | None = None,
        filename: str = "signal_lists.json",
    ):
        """Initialize the signal list manager.

        Args:
            config_dir: Directory to store signal lists. Defaults to user's home.
            filename: Name of the signal lists file.
        """
        if not (filename is not None):
            raise ValueError("filename must be provided")
        if config_dir is None:
            self.config_dir = Path.home() / ".data_processor"
        else:
            self.config_dir = Path(config_dir)

        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.lists_file = self.config_dir / filename

    def save_signal_list(
        self,
        name: str,
        signals: list[str],
        description: str = "",
    ) -> None:
        """Save a named signal list.

        Args:
            name: Unique name for this signal list
            signals: List of signal names
            description: Optional description
        """
        if not (name is not None):
            raise ValueError("name must be provided")
        lists = self._load_all_lists()

        lists[name] = {
            "signals": signals,
            "description": description,
            "count": len(signals),
            "created": lists.get(name, {}).get("created", datetime.now().isoformat()),
            "modified": datetime.now().isoformat(),
        }

        self._save_all_lists(lists)

    def load_signal_list(self, name: str) -> list[str]:
        """Load a named signal list.

        Args:
            name: Name of the signal list to load

        Returns:
            List of signal names

        Raises:
            KeyError: If signal list doesn't exist
        """
        lists = self._load_all_lists()

        if name not in lists:
            raise KeyError(f"Signal list not found: {name}")

        return lists[name]["signals"]

    def delete_signal_list(self, name: str) -> None:
        """Delete a named signal list.

        Args:
            name: Name of the signal list to delete
        """
        if not (name is not None):
            raise ValueError("name must be provided")
        lists = self._load_all_lists()

        if name in lists:
            del lists[name]
            self._save_all_lists(lists)

    def list_signal_sets(self) -> list[str]:
        """List all saved signal set names.

        Returns:
            List of signal set names
        """
        lists = self._load_all_lists()
        return list(lists.keys())

    def get_signal_list_info(self, name: str) -> dict[str, Any]:
        """Get metadata about a signal list.

        Args:
            name: Name of the signal list

        Returns:
            Dictionary with metadata (count, created, modified, description)
        """
        lists = self._load_all_lists()

        if name not in lists:
            raise KeyError(f"Signal list not found: {name}")

        return {
            "name": name,
            "count": lists[name]["count"],
            "description": lists[name].get("description", ""),
            "created": lists[name].get("created"),
            "modified": lists[name].get("modified"),
        }

    def export_signal_list(self, name: str, export_path: Path | str) -> None:
        """Export a signal list to a file.

        Args:
            name: Name of the signal list
            export_path: Path to export to
        """
        if not (name is not None):
            raise ValueError("name must be provided")
        signals = self.load_signal_list(name)
        export_path = Path(export_path)

        with open(export_path, "w", encoding="utf-8") as f:
            json.dump({"name": name, "signals": signals}, f, indent=2)

    def import_signal_list(self, import_path: Path | str, name: str | None = None) -> str:
        """Import a signal list from a file.

        Args:
            import_path: Path to import from
            name: Optional name override (uses file name if not provided)

        Returns:
            Name of the imported signal list
        """
        if not (import_path is not None):
            raise ValueError("import_path must be provided")
        import_path = Path(import_path)

        with open(import_path, encoding="utf-8") as f:
            data = json.load(f)

        list_name = name or data.get("name", import_path.stem)
        signals = data.get("signals", data if isinstance(data, list) else [])

        self.save_signal_list(list_name, signals)
        return list_name

    def _load_all_lists(self) -> dict[str, Any]:
        """Load all signal lists from file."""
        if not self.lists_file.exists():
            return {}

        try:
            with open(self.lists_file, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_all_lists(self, lists: dict[str, Any]) -> None:
        """Save all signal lists to file."""
        with open(self.lists_file, "w", encoding="utf-8") as f:
            json.dump(lists, f, indent=2)


__all__ = ["SignalListManager"]
