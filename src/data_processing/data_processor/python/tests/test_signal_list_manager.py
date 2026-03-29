"""Tests for SignalListManager -- persistence for signal selections.

Covers: save, load, delete, list, info, export/import.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_processor.core.signal_list_manager import SignalListManager


@pytest.fixture()
def manager(tmp_path: Path) -> SignalListManager:
    """Create a SignalListManager with a temp config directory."""
    return SignalListManager(config_dir=tmp_path)


class TestSignalListManagerInit:
    """Test initialization."""

    def test_creates_config_dir(self, tmp_path: Path) -> None:
        sub = tmp_path / "signals"
        SignalListManager(config_dir=sub)
        assert sub.exists()

    def test_default_filename(self, tmp_path: Path) -> None:
        mgr = SignalListManager(config_dir=tmp_path)
        assert mgr.lists_file.name == "signal_lists.json"


class TestSaveAndLoad:
    """Test save and load."""

    def test_save_and_load(self, manager: SignalListManager) -> None:
        manager.save_signal_list("temps", ["T_in", "T_out", "T_amb"])
        loaded = manager.load_signal_list("temps")
        assert loaded == ["T_in", "T_out", "T_amb"]

    def test_load_nonexistent_raises(self, manager: SignalListManager) -> None:
        with pytest.raises(KeyError, match="not found"):
            manager.load_signal_list("nonexistent")

    def test_save_with_description(self, manager: SignalListManager) -> None:
        manager.save_signal_list("pressures", ["P1", "P2"], description="Pressure signals")
        info = manager.get_signal_list_info("pressures")
        assert info["description"] == "Pressure signals"

    def test_overwrite_list(self, manager: SignalListManager) -> None:
        manager.save_signal_list("list1", ["a", "b"])
        manager.save_signal_list("list1", ["c", "d", "e"])
        loaded = manager.load_signal_list("list1")
        assert loaded == ["c", "d", "e"]


class TestDeleteAndList:
    """Test delete and listing."""

    def test_delete_list(self, manager: SignalListManager) -> None:
        manager.save_signal_list("delete_me", ["x"])
        manager.delete_signal_list("delete_me")
        assert "delete_me" not in manager.list_signal_sets()

    def test_delete_nonexistent_no_error(self, manager: SignalListManager) -> None:
        manager.delete_signal_list("nonexistent")  # Should not raise

    def test_list_sets(self, manager: SignalListManager) -> None:
        manager.save_signal_list("set_a", ["a"])
        manager.save_signal_list("set_b", ["b"])
        names = manager.list_signal_sets()
        assert "set_a" in names
        assert "set_b" in names

    def test_list_empty(self, manager: SignalListManager) -> None:
        assert manager.list_signal_sets() == []


class TestSignalListInfo:
    """Test metadata retrieval."""

    def test_get_info(self, manager: SignalListManager) -> None:
        manager.save_signal_list("info_test", ["sig1", "sig2", "sig3"])
        info = manager.get_signal_list_info("info_test")
        assert info["name"] == "info_test"
        assert info["count"] == 3

    def test_get_info_nonexistent_raises(self, manager: SignalListManager) -> None:
        with pytest.raises(KeyError, match="not found"):
            manager.get_signal_list_info("nonexistent")


class TestExportImport:
    """Test export and import."""

    def test_export_and_import(self, manager: SignalListManager, tmp_path: Path) -> None:
        manager.save_signal_list("export_me", ["x", "y", "z"])
        export_path = tmp_path / "exported.json"
        manager.export_signal_list("export_me", export_path)

        assert export_path.exists()
        data = json.loads(export_path.read_text())
        assert data["name"] == "export_me"
        assert data["signals"] == ["x", "y", "z"]

        # Import into same manager under new name
        imported_name = manager.import_signal_list(export_path, name="imported_list")
        assert imported_name == "imported_list"
        assert manager.load_signal_list("imported_list") == ["x", "y", "z"]

    def test_import_without_name_uses_file_name(
        self, manager: SignalListManager, tmp_path: Path
    ) -> None:
        export_path = tmp_path / "my_signals.json"
        export_path.write_text(json.dumps({"signals": ["a", "b"]}))
        imported = manager.import_signal_list(export_path)
        assert imported == "my_signals"
