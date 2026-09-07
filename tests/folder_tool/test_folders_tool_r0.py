"""Unit tests for folder_tool/Folders_Tool_r0.py."""

import sys
import tkinter as tk
import types
from unittest.mock import patch

import pytest


# Mock out mixins to prevent circular dependency resolution
class DummyUi:
    def create_scrollable_interface(self):
        pass

    def _setup_application_icon(self):
        pass


class DummyFile:
    pass


class DummyProc:
    pass


mock_ui = types.ModuleType("folder_tool_ui")
mock_ui.UICreationMixin = DummyUi
sys.modules["folder_tool_ui"] = mock_ui

mock_file = types.ModuleType("folder_tool_file_ops")
mock_file.FileOperationsMixin = DummyFile
sys.modules["folder_tool_file_ops"] = mock_file

mock_proc = types.ModuleType("folder_tool_processing")
mock_proc.ProcessingMixin = DummyProc
sys.modules["folder_tool_processing"] = mock_proc

from folder_tool.Folders_Tool_r0 import (
    FolderProcessorApp,
    _get_log_path,
    safe_write_text,
)


class TestFolderProcessorApp:
    def test_init_success(self):
        root = tk.Tk()
        # Mock create_scrollable_interface and _setup_application_icon to avoid GUI rendering blocking
        with patch.object(FolderProcessorApp, "create_scrollable_interface"):
            with patch.object(FolderProcessorApp, "_setup_application_icon"):
                app = FolderProcessorApp(root)
                assert app.root == root
                assert app.source_folders == []
                assert app.dest_folder == ""
        root.destroy()

    def test_init_none_root(self):
        with pytest.raises((AssertionError, ValueError)):
            FolderProcessorApp(None)

    def test_validate_constants(self):
        root = tk.Tk()
        with patch.object(FolderProcessorApp, "create_scrollable_interface"):
            with patch.object(FolderProcessorApp, "_setup_application_icon"):
                app = FolderProcessorApp(root)
                with patch(
                    "folder_tool.Folders_Tool_r0.validate_constants"
                ) as mock_val:
                    app._validate_constants()
                    mock_val.assert_called_once()
        root.destroy()

    def test_get_constants_info(self):
        root = tk.Tk()
        with patch.object(FolderProcessorApp, "create_scrollable_interface"):
            with patch.object(FolderProcessorApp, "_setup_application_icon"):
                app = FolderProcessorApp(root)
                with patch(
                    "folder_tool.Folders_Tool_r0.get_constants_info",
                    return_value={"mock": "data"},
                ):
                    res = app.get_constants_info()
                    assert res == {"mock": "data"}
        root.destroy()

    def test_export_constants_documentation(self):
        root = tk.Tk()
        with patch.object(FolderProcessorApp, "create_scrollable_interface"):
            with patch.object(FolderProcessorApp, "_setup_application_icon"):
                app = FolderProcessorApp(root)
                with patch(
                    "folder_tool.Folders_Tool_r0.export_constants_documentation",
                    return_value=True,
                ):
                    res = app.export_constants_documentation("path")
                    assert res is True
        root.destroy()


class TestSafeWriteText:
    def test_safe_write_text_success(self, tmp_path):
        f = tmp_path / "subdir" / "file.txt"
        safe_write_text(str(f), "content")
        assert f.read_text() == "content"

    def test_safe_write_text_no_parents(self, tmp_path):
        f = tmp_path / "file.txt"
        safe_write_text(str(f), "content", create_parents=False)
        assert f.read_text() == "content"

    def test_safe_write_text_none_path(self):
        with pytest.raises((AssertionError, ValueError)):
            safe_write_text(None, "content")


class TestGetLogPath:
    """Tests for _get_log_path (XDG-aware log file resolution)."""

    def test_returns_path_object(self, tmp_path, monkeypatch):
        """_get_log_path returns a Path ending in folder_processor.log."""
        import sys

        if sys.platform != "win32":
            monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

        result = _get_log_path()
        assert result.name == "folder_processor.log"

    def test_log_in_xdg_config_home(self, tmp_path, monkeypatch):
        """Log file placed under $XDG_CONFIG_HOME/folder_tool/ on POSIX."""
        import sys

        if sys.platform == "win32":
            pytest.skip("XDG_CONFIG_HOME is a POSIX convention")

        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        result = _get_log_path()
        assert result == tmp_path / "folder_tool" / "folder_processor.log"

    def test_creates_config_directory(self, tmp_path, monkeypatch):
        """_get_log_path creates the config directory if absent."""
        import sys

        if sys.platform == "win32":
            pytest.skip("XDG_CONFIG_HOME is a POSIX convention")

        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        expected_dir = tmp_path / "folder_tool"
        assert not expected_dir.exists()

        _get_log_path()
        assert expected_dir.exists()
