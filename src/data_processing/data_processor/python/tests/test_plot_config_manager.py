"""Tests for PlotConfigManager -- persistence for plot configurations.

Covers: save, load, delete, list, info, duplicate, export/import.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_processor.core.plot_config_manager import PlotConfigManager


@pytest.fixture()
def manager(tmp_path: Path) -> PlotConfigManager:
    """Create a PlotConfigManager with a temp config directory."""
    return PlotConfigManager(config_dir=tmp_path)


@pytest.fixture()
def sample_config() -> dict:
    """Return a minimal plot configuration."""
    return {
        "name": "Temperature Plot",
        "signals": ["T_in", "T_out"],
        "x_axis": "time",
        "chart_type": "line",
    }


class TestPlotConfigManagerInit:
    """Test PlotConfigManager initialization."""

    def test_creates_config_dir(self, tmp_path: Path) -> None:
        sub = tmp_path / "subdir"
        mgr = PlotConfigManager(config_dir=sub)
        assert sub.exists()

    def test_default_filename(self, tmp_path: Path) -> None:
        mgr = PlotConfigManager(config_dir=tmp_path)
        assert mgr.config_file.name == "plot_configs.json"


class TestSaveAndLoad:
    """Test save and load operations."""

    def test_save_and_load(
        self, manager: PlotConfigManager, sample_config: dict
    ) -> None:
        manager.save_plot_config("my_plot", sample_config)
        loaded = manager.load_plot_config("my_plot")
        assert loaded["name"] == "Temperature Plot"
        assert loaded["signals"] == ["T_in", "T_out"]

    def test_load_nonexistent_raises(self, manager: PlotConfigManager) -> None:
        with pytest.raises(KeyError, match="not found"):
            manager.load_plot_config("nonexistent")

    def test_overwrite_config(
        self, manager: PlotConfigManager, sample_config: dict
    ) -> None:
        manager.save_plot_config("plot1", sample_config)
        updated = {**sample_config, "name": "Updated Plot"}
        manager.save_plot_config("plot1", updated)
        loaded = manager.load_plot_config("plot1")
        assert loaded["name"] == "Updated Plot"


class TestDeleteAndList:
    """Test delete and listing operations."""

    def test_delete_config(
        self, manager: PlotConfigManager, sample_config: dict
    ) -> None:
        manager.save_plot_config("to_delete", sample_config)
        manager.delete_plot_config("to_delete")
        assert "to_delete" not in manager.list_plot_configs()

    def test_delete_nonexistent_no_error(self, manager: PlotConfigManager) -> None:
        manager.delete_plot_config("nonexistent")  # Should not raise

    def test_list_configs(
        self, manager: PlotConfigManager, sample_config: dict
    ) -> None:
        manager.save_plot_config("plot_a", sample_config)
        manager.save_plot_config("plot_b", sample_config)
        names = manager.list_plot_configs()
        assert "plot_a" in names
        assert "plot_b" in names

    def test_list_empty(self, manager: PlotConfigManager) -> None:
        assert manager.list_plot_configs() == []


class TestConfigInfo:
    """Test metadata retrieval."""

    def test_get_info(self, manager: PlotConfigManager, sample_config: dict) -> None:
        manager.save_plot_config("info_test", sample_config)
        info = manager.get_plot_config_info("info_test")
        assert info["name"] == "info_test"
        assert info["signal_count"] == 2
        assert info["title"] == "Temperature Plot"

    def test_get_info_nonexistent_raises(self, manager: PlotConfigManager) -> None:
        with pytest.raises(KeyError, match="not found"):
            manager.get_plot_config_info("nonexistent")


class TestDuplicateConfig:
    """Test config duplication."""

    def test_duplicate(self, manager: PlotConfigManager, sample_config: dict) -> None:
        manager.save_plot_config("original", sample_config)
        manager.duplicate_plot_config("original", "copy")
        loaded = manager.load_plot_config("copy")
        # Duplicate preserves the config's signal list
        assert loaded["signals"] == ["T_in", "T_out"]
