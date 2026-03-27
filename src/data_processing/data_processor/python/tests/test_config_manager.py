"""Tests for ConfigManager — configuration persistence.

Covers save/load round-trip, delete, list, export/import, and
missing-config error handling.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from data_processor.core.config_manager import ConfigManager


@pytest.fixture()
def config_mgr(tmp_path: Path) -> ConfigManager:
    """Create a ConfigManager using a temp directory."""
    return ConfigManager(config_dir=tmp_path, config_filename="test_config.json")


class TestConfigManagerSaveLoad:
    """Test save and load round-trip."""

    def test_save_and_load_round_trip(self, config_mgr: ConfigManager) -> None:
        """Saved settings must be retrievable."""
        settings = {"filter_type": "butterworth", "cutoff": 0.1}
        config_mgr.save_config("my_filter", settings)
        loaded = config_mgr.load_config("my_filter")
        assert loaded == settings

    def test_overwrite_preserves_created_timestamp(
        self, config_mgr: ConfigManager
    ) -> None:
        """Overwriting a config preserves the original created timestamp."""
        config_mgr.save_config("cfg", {"v": 1})
        info1 = config_mgr.get_config_info("cfg")
        config_mgr.save_config("cfg", {"v": 2})
        info2 = config_mgr.get_config_info("cfg")
        assert info1["created"] == info2["created"]
        assert info2["modified"] >= info1["modified"]

    def test_load_nonexistent_raises_key_error(self, config_mgr: ConfigManager) -> None:
        """Loading a missing config must raise KeyError."""
        with pytest.raises(KeyError, match="Configuration not found"):
            config_mgr.load_config("nonexistent")


class TestConfigManagerList:
    """Test list and delete operations."""

    def test_list_empty(self, config_mgr: ConfigManager) -> None:
        """Fresh manager should return empty list."""
        assert config_mgr.list_configs() == []

    def test_list_after_save(self, config_mgr: ConfigManager) -> None:
        """list_configs should return names of saved configs."""
        config_mgr.save_config("a", {"x": 1})
        config_mgr.save_config("b", {"y": 2})
        names = config_mgr.list_configs()
        assert set(names) == {"a", "b"}

    def test_delete_removes_config(self, config_mgr: ConfigManager) -> None:
        """Deleting a config removes it from persistent storage."""
        config_mgr.save_config("del_me", {"z": 3})
        config_mgr.delete_config("del_me")
        assert "del_me" not in config_mgr.list_configs()

    def test_delete_nonexistent_is_noop(self, config_mgr: ConfigManager) -> None:
        """Deleting a nonexistent config should not raise."""
        config_mgr.delete_config("ghost")  # should not raise


class TestConfigManagerExportImport:
    """Test export and import operations."""

    def test_export_import_round_trip(
        self, config_mgr: ConfigManager, tmp_path: Path
    ) -> None:
        """Export then import should recover the original settings."""
        settings = {"rate": 1000, "method": "fft"}
        config_mgr.save_config("exportable", settings)

        export_file = tmp_path / "exported.json"
        config_mgr.export_config("exportable", export_file)

        # Verify file contents
        with open(export_file) as f:
            exported = json.load(f)
        assert exported == settings

        # Import under a new name
        config_mgr.import_config("imported", export_file)
        loaded = config_mgr.load_config("imported")
        assert loaded == settings
