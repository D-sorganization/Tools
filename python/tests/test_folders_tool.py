"""Tests for Folders_Tool_r0.py."""

import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
import tkinter

# Add folder_tool directory to path
sys.path.append(str(Path(__file__).parent.parent / "folder_tool"))

from Folders_Tool_r0 import FolderProcessorApp, MAX_FILE_SIZE_MB

class TestFolderProcessorApp:
    """Test cases for FolderProcessorApp."""

    @pytest.fixture
    def mock_root(self):
        """Mock Tkinter root."""
        root = Mock()
        return root

    @pytest.fixture
    def mock_tk_vars(self):
        """Mock Tkinter variables."""
        with patch("tkinter.BooleanVar") as mock_bool, \
             patch("tkinter.StringVar") as mock_string, \
             patch("tkinter.DoubleVar") as mock_double, \
             patch("tkinter.IntVar") as mock_int:

            mock_bool.return_value.get.return_value = False
            mock_string.return_value.get.return_value = ""
            mock_double.return_value.get.return_value = 0.0
            mock_int.return_value.get.return_value = 0

            yield {
                "bool": mock_bool,
                "string": mock_string,
                "double": mock_double,
                "int": mock_int
            }

    def test_init(self, mock_root, mock_tk_vars):
        """Test initialization of FolderProcessorApp."""
        with patch("tkinter.ttk.Style"), \
             patch("tkinter.Canvas"), \
             patch("tkinter.ttk.Scrollbar"), \
             patch("tkinter.ttk.Frame"), \
             patch("tkinter.ttk.LabelFrame"), \
             patch("tkinter.Listbox"), \
             patch("tkinter.ttk.Button"), \
             patch("tkinter.ttk.Label"), \
             patch("tkinter.ttk.Entry"), \
             patch("tkinter.ttk.Checkbutton"), \
             patch("tkinter.ttk.Radiobutton"), \
             patch("tkinter.ttk.Progressbar"), \
             patch("Folders_Tool_r0.ctypes"):  # Mock ctypes used in _setup_application_icon

            app = FolderProcessorApp(mock_root)

            assert app.root == mock_root
            assert app.source_folders == []
            assert app.dest_folder == ""

    def test_validate_constants(self, mock_root, mock_tk_vars):
        """Test constant validation."""
        with patch("tkinter.ttk.Style"), \
             patch("tkinter.Canvas"), \
             patch("tkinter.ttk.Scrollbar"), \
             patch("tkinter.ttk.Frame"), \
             patch("tkinter.ttk.LabelFrame"), \
             patch("tkinter.Listbox"), \
             patch("tkinter.ttk.Button"), \
             patch("tkinter.ttk.Label"), \
             patch("tkinter.ttk.Entry"), \
             patch("tkinter.ttk.Checkbutton"), \
             patch("tkinter.ttk.Radiobutton"), \
             patch("tkinter.ttk.Progressbar"), \
             patch("Folders_Tool_r0.ctypes"):

            app = FolderProcessorApp(mock_root)
            # If no exception, it passed
            assert True

    def test_get_unique_path(self, mock_root, mock_tk_vars, tmp_path):
        """Test _get_unique_path."""
        with patch("tkinter.ttk.Style"), \
             patch("tkinter.Canvas"), \
             patch("tkinter.ttk.Scrollbar"), \
             patch("tkinter.ttk.Frame"), \
             patch("tkinter.ttk.LabelFrame"), \
             patch("tkinter.Listbox"), \
             patch("tkinter.ttk.Button"), \
             patch("tkinter.ttk.Label"), \
             patch("tkinter.ttk.Entry"), \
             patch("tkinter.ttk.Checkbutton"), \
             patch("tkinter.ttk.Radiobutton"), \
             patch("tkinter.ttk.Progressbar"), \
             patch("Folders_Tool_r0.ctypes"):

            app = FolderProcessorApp(mock_root)

            p = tmp_path / "test.txt"
            p.write_text("content")

            # Test first conflict
            unique_path = app._get_unique_path(str(p))
            assert unique_path == str(tmp_path / "test (1).txt")

            # Create the first conflict file and test second conflict
            Path(unique_path).write_text("content 2")
            unique_path_2 = app._get_unique_path(str(p))
            assert unique_path_2 == str(tmp_path / "test (2).txt")

    def test_validate_inputs_no_source(self, mock_root, mock_tk_vars):
        """Test validate_inputs with no source."""
        with patch("tkinter.ttk.Style"), \
             patch("tkinter.Canvas"), \
             patch("tkinter.ttk.Scrollbar"), \
             patch("tkinter.ttk.Frame"), \
             patch("tkinter.ttk.LabelFrame"), \
             patch("tkinter.Listbox"), \
             patch("tkinter.ttk.Button"), \
             patch("tkinter.ttk.Label"), \
             patch("tkinter.ttk.Entry"), \
             patch("tkinter.ttk.Checkbutton"), \
             patch("tkinter.ttk.Radiobutton"), \
             patch("tkinter.ttk.Progressbar"), \
             patch("Folders_Tool_r0.ctypes"), \
             patch("tkinter.messagebox.showerror") as mock_error:

            app = FolderProcessorApp(mock_root)
            app.source_folders = []

            assert app.validate_inputs() is False
            mock_error.assert_called()

    def test_validate_inputs_no_dest(self, mock_root, mock_tk_vars):
        """Test validate_inputs with no dest."""
        with patch("tkinter.ttk.Style"), \
             patch("tkinter.Canvas"), \
             patch("tkinter.ttk.Scrollbar"), \
             patch("tkinter.ttk.Frame"), \
             patch("tkinter.ttk.LabelFrame"), \
             patch("tkinter.Listbox"), \
             patch("tkinter.ttk.Button"), \
             patch("tkinter.ttk.Label"), \
             patch("tkinter.ttk.Entry"), \
             patch("tkinter.ttk.Checkbutton"), \
             patch("tkinter.ttk.Radiobutton"), \
             patch("tkinter.ttk.Progressbar"), \
             patch("Folders_Tool_r0.ctypes"), \
             patch("tkinter.messagebox.showerror") as mock_error:

            app = FolderProcessorApp(mock_root)
            app.source_folders = ["/some/path"]
            app.dest_folder = ""

            assert app.validate_inputs() is False
            mock_error.assert_called()
