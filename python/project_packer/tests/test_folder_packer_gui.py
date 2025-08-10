"""Tests for folder_packer_gui.py module."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Import the module to test
sys.path.insert(0, str(Path(__file__).parent.parent))
from folder_packer_gui import (
    EXCLUDE_PATTERNS,
    INCLUDE_EXTENSIONS,
    FolderPackerGUI,
    main,
)


class TestFolderPackerGUI:
    """Test cases for folder_packer_gui.py module."""

    @pytest.fixture()
    def mock_root(self) -> Mock:
        """Create a mock Tkinter root window."""
        mock_root = Mock()
        mock_root.title = Mock()
        mock_root.geometry = Mock()
        mock_root.resizable = Mock()
        mock_root.columnconfigure = Mock()
        mock_root.rowconfigure = Mock()
        mock_root.update_idletasks = Mock()
        return mock_root

    @pytest.fixture()
    def gui_instance(self, mock_root: Mock) -> FolderPackerGUI:
        """Create a FolderPackerGUI instance with mocked dependencies."""
        with (
            patch("folder_packer_gui.ttk.Frame"),
            patch("folder_packer_gui.ttk.Label"),
            patch("folder_packer_gui.ttk.LabelFrame"),
            patch("folder_packer_gui.tk.Listbox"),
            patch("folder_packer_gui.ttk.Button"),
            patch("folder_packer_gui.ttk.ttk.Entry"),
            patch("folder_packer_gui.tk.Text"),
            patch("folder_packer_gui.ttk.Scrollbar"),
            patch("folder_packer_gui.ttk.Style"),
        ):
            gui = FolderPackerGUI(mock_root)
            return gui

    def test_init(self, mock_root: Mock) -> None:
        """Test GUI initialization."""
        with (
            patch("folder_packer_gui.ttk.Frame"),
            patch("folder_packer_gui.ttk.Label"),
            patch("folder_packer_gui.ttk.LabelFrame"),
            patch("folder_packer_gui.tk.Listbox"),
            patch("folder_packer_gui.ttk.Button"),
            patch("folder_packer_gui.ttk.Entry"),
            patch("folder_packer_gui.tk.Text"),
            patch("folder_packer_gui.ttk.Scrollbar"),
            patch("folder_packer_gui.ttk.Style"),
        ):
            gui = FolderPackerGUI(mock_root)

            assert gui.root == mock_root
            assert gui.source_folders == []
            assert gui.output_directory == ""

    def test_add_folder_new(self, gui_instance: FolderPackerGUI) -> None:
        """Test adding a new folder."""
        test_folder = "/test/path"
        with patch("folder_packer_gui.filedialog.askdirectory") as mock_ask:
            mock_ask.return_value = test_folder
            gui_instance.add_folder()

            assert test_folder in gui_instance.source_folders

    def test_add_folder_existing(self, gui_instance: FolderPackerGUI) -> None:
        """Test adding an existing folder (should not duplicate)."""
        test_folder = "/test/path"
        gui_instance.source_folders = [test_folder]

        with patch("folder_packer_gui.filedialog.askdirectory") as mock_ask:
            mock_ask.return_value = test_folder
            gui_instance.add_folder()

            # Should not duplicate
            assert gui_instance.source_folders.count(test_folder) == 1

    def test_add_folder_cancelled(self, gui_instance: FolderPackerGUI) -> None:
        """Test adding folder when dialog is cancelled."""
        with patch("folder_packer_gui.filedialog.askdirectory") as mock_ask:
            mock_ask.return_value = ""
            gui_instance.add_folder()

            # Should not add empty path
            assert len(gui_instance.source_folders) == 0

    def test_remove_selected_folders(self, gui_instance: FolderPackerGUI) -> None:
        """Test removing selected folders."""
        # Setup test data
        test_folders = ["/test1", "/test2", "/test3"]
        gui_instance.source_folders = test_folders.copy()

        # Mock listbox selection
        gui_instance.folders_listbox = Mock()
        gui_instance.folders_listbox.curselection.return_value = [
            0,
            2,
        ]  # Select first and third
        gui_instance.folders_listbox.get.side_effect = lambda x: test_folders[x]
        gui_instance.folders_listbox.delete = Mock()

        gui_instance.remove_selected_folders()

        # Should remove selected folders
        assert "/test1" not in gui_instance.source_folders
        assert "/test2" in gui_instance.source_folders  # Not selected
        assert "/test3" not in gui_instance.source_folders

    def test_browse_output(self, gui_instance: FolderPackerGUI) -> None:
        """Test browsing for output directory."""
        test_directory = "/output/path"
        gui_instance.output_entry = Mock()

        with patch("folder_packer_gui.filedialog.askdirectory") as mock_ask:
            mock_ask.return_value = test_directory
            gui_instance.browse_output()

            assert gui_instance.output_directory == test_directory
            gui_instance.output_entry.delete.assert_called_once_with(0, "end")
            gui_instance.output_entry.insert.assert_called_once_with(0, test_directory)

    def test_browse_output_cancelled(self, gui_instance: FolderPackerGUI) -> None:
        """Test browsing for output directory when cancelled."""
        gui_instance.output_entry = Mock()

        with patch("folder_packer_gui.filedialog.askdirectory") as mock_ask:
            mock_ask.return_value = ""
            gui_instance.browse_output()

            # Should not change output directory
            assert gui_instance.output_directory == ""

    def test_pack_folders_no_source_folders(
        self, gui_instance: FolderPackerGUI,
    ) -> None:
        """Test packing when no source folders are selected."""
        gui_instance.source_folders = []

        with patch("folder_packer_gui.messagebox.showwarning") as mock_warning:
            gui_instance.pack_folders()
            mock_warning.assert_called_once()

    def test_pack_folders_no_output_directory(
        self, gui_instance: FolderPackerGUI,
    ) -> None:
        """Test packing when no output directory is selected."""
        gui_instance.source_folders = ["/test/folder"]
        gui_instance.output_directory = ""

        with patch("folder_packer_gui.messagebox.showwarning") as mock_warning:
            gui_instance.pack_folders()
            mock_warning.assert_called_once()

    def test_pack_folders_success(
        self, gui_instance: FolderPackerGUI, tmp_path: Path,
    ) -> None:
        """Test successful folder packing."""
        # Setup test data
        source_folder = tmp_path / "source"
        source_folder.mkdir()
        (source_folder / "test.txt").write_text("test content")

        gui_instance.source_folders = [str(source_folder)]
        gui_instance.output_directory = str(tmp_path / "output")

        with patch.object(gui_instance, "pack_single_folder") as mock_pack:
            mock_pack.return_value = True
            with patch.object(gui_instance, "update_status") as mock_update:
                gui_instance.pack_folders()

                mock_pack.assert_called_once_with(str(source_folder))
                assert (
                    mock_update.call_count >= 2
                )  # At least packing and success messages

    def test_pack_single_folder_success(
        self, gui_instance: FolderPackerGUI, tmp_path: Path,
    ) -> None:
        """Test successful single folder packing."""
        source_folder = tmp_path / "source"
        source_folder.mkdir()
        (source_folder / "test.txt").write_text("test content")

        output_dir = tmp_path / "output"
        gui_instance.output_directory = str(output_dir)

        result = gui_instance.pack_single_folder(str(source_folder))

        assert result is True
        assert (output_dir / "source").exists()
        assert (output_dir / "source" / "test.txt").exists()

    def test_pack_single_folder_not_exists(self, gui_instance: FolderPackerGUI) -> None:
        """Test packing non-existent folder."""
        gui_instance.output_directory = "/output/path"

        result = gui_instance.pack_single_folder("/non/existent/path")

        assert result is False

    def test_should_include_file_config_file(
        self, gui_instance: FolderPackerGUI,
    ) -> None:
        """Test file inclusion for configuration files."""
        config_files = [".env", ".config", ".conf", ".cfg", ".ini", ".toml"]

        for ext in config_files:
            file_path = Path(f"test{ext}")
            assert gui_instance.should_include_file(file_path) is True

    def test_should_include_file_regular_file(
        self, gui_instance: FolderPackerGUI,
    ) -> None:
        """Test file inclusion for regular files."""
        # These files should be included because they're in INCLUDE_EXTENSIONS
        regular_files = [".txt", ".py", ".md", ".json"]

        for ext in regular_files:
            file_path = Path(f"test{ext}")
            assert gui_instance.should_include_file(file_path) is True

    def test_should_include_directory_excluded_patterns(
        self, gui_instance: FolderPackerGUI,
    ) -> None:
        """Test directory inclusion for excluded patterns."""
        excluded_dirs = ["__pycache__", ".git", "node_modules", ".venv"]

        for dir_name in excluded_dirs:
            dir_path = Path(dir_name)
            assert gui_instance.should_include_directory(dir_path) is False

    def test_should_include_directory_valid(
        self, gui_instance: FolderPackerGUI, tmp_path: Path,
    ) -> None:
        """Test directory inclusion for valid directories."""
        valid_dir = tmp_path / "valid_folder"
        valid_dir.mkdir()
        (valid_dir / "test.txt").write_text("test content")

        assert gui_instance.should_include_directory(valid_dir) is True

    def test_update_status(self, gui_instance: FolderPackerGUI) -> None:
        """Test status update functionality."""
        gui_instance.status_text = Mock()
        gui_instance.root = Mock()

        test_message = "Test status message"
        gui_instance.update_status(test_message)

        # Should insert message and scroll to end
        gui_instance.status_text.insert.assert_called()
        gui_instance.status_text.see.assert_called_with("end")
        gui_instance.root.update_idletasks.assert_called_once()

    def test_constants_defined(self) -> None:
        """Test that constants are properly defined."""
        assert isinstance(INCLUDE_EXTENSIONS, set)
        assert len(INCLUDE_EXTENSIONS) > 0
        assert all(isinstance(ext, str) for ext in INCLUDE_EXTENSIONS)

        assert isinstance(EXCLUDE_PATTERNS, set)
        assert len(EXCLUDE_PATTERNS) > 0
        assert all(isinstance(pattern, str) for pattern in EXCLUDE_PATTERNS)

    def test_main_function(self) -> None:
        """Test main function."""
        with patch("folder_packer_gui.tk.Tk") as mock_tk_class:
            mock_root = Mock()
            mock_tk_class.return_value = mock_root

            with patch("folder_packer_gui.FolderPackerGUI") as mock_gui_class:
                mock_gui = Mock()
                mock_gui_class.return_value = mock_gui

                main()

                mock_tk_class.assert_called_once()
                mock_gui_class.assert_called_once_with(mock_root)
                mock_root.mainloop.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
