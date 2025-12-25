"""Tests for Folders_Tool_r0.py."""

import os
import sys
from unittest.mock import MagicMock

# Mock tkinter before importing modules that use it
sys.modules["tkinter"] = MagicMock()
sys.modules["tkinter.ttk"] = MagicMock()
sys.modules["tkinter.filedialog"] = MagicMock()
sys.modules["tkinter.messagebox"] = MagicMock()
sys.modules["tkinter.scrolledtext"] = MagicMock()
sys.modules["tkinter.simpledialog"] = MagicMock()

# Link submodules to parent module
sys.modules["tkinter"].ttk = sys.modules["tkinter.ttk"]  # type: ignore[attr-defined]
sys.modules["tkinter"].filedialog = sys.modules["tkinter.filedialog"]  # type: ignore[attr-defined]
sys.modules["tkinter"].messagebox = sys.modules["tkinter.messagebox"]  # type: ignore[attr-defined]
sys.modules["tkinter"].scrolledtext = sys.modules["tkinter.scrolledtext"]  # type: ignore[attr-defined]
sys.modules["tkinter"].simpledialog = sys.modules["tkinter.simpledialog"]  # type: ignore[attr-defined]

import tkinter as tk
from collections.abc import Generator
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add folder_tool directory to path
sys.path.append(str(Path(__file__).parent.parent / "folder_tool"))

from Folders_Tool_r0 import FolderProcessorApp


class TestFolderProcessorApp:
    """Test cases for FolderProcessorApp."""

    @pytest.fixture()
    def mock_root(self) -> Mock:
        """Mock Tkinter root."""
        return Mock()

    @pytest.fixture()
    def mock_tk_vars(self) -> Generator[dict[str, Mock], None, None]:
        """Mock Tkinter variables."""
        with (
            patch("tkinter.BooleanVar") as mock_bool,
            patch("tkinter.StringVar") as mock_string,
            patch("tkinter.DoubleVar") as mock_double,
            patch("tkinter.IntVar") as mock_int,
        ):
            mock_bool.return_value.get.return_value = False
            mock_string.return_value.get.return_value = ""
            mock_double.return_value.get.return_value = 0.0
            mock_int.return_value.get.return_value = 0

            yield {
                "bool": mock_bool,
                "string": mock_string,
                "double": mock_double,
                "int": mock_int,
            }

    def test_init(self, mock_root: Mock, mock_tk_vars: dict[str, Mock]) -> None:
        """Test initialization of FolderProcessorApp."""
        with (
            patch("tkinter.ttk.Style"),
            patch("tkinter.Canvas"),
            patch("tkinter.ttk.Scrollbar"),
            patch("tkinter.ttk.Frame"),
            patch("tkinter.ttk.LabelFrame"),
            patch("tkinter.Listbox"),
            patch("tkinter.ttk.Button"),
            patch("tkinter.ttk.Label"),
            patch("tkinter.ttk.Entry"),
            patch("tkinter.ttk.Checkbutton"),
            patch("tkinter.ttk.Radiobutton"),
            patch("tkinter.ttk.Progressbar"),
            patch("Folders_Tool_r0.ctypes"),
        ):  # Mock ctypes used in _setup_application_icon
            app = FolderProcessorApp(mock_root)

            assert app.root == mock_root
            assert app.source_folders == []
            assert app.dest_folder == ""

    def test_validate_constants(
        self, mock_root: Mock, mock_tk_vars: dict[str, Mock]
    ) -> None:
        """Test constant validation."""
        with (
            patch("tkinter.ttk.Style"),
            patch("tkinter.Canvas"),
            patch("tkinter.ttk.Scrollbar"),
            patch("tkinter.ttk.Frame"),
            patch("tkinter.ttk.LabelFrame"),
            patch("tkinter.Listbox"),
            patch("tkinter.ttk.Button"),
            patch("tkinter.ttk.Label"),
            patch("tkinter.ttk.Entry"),
            patch("tkinter.ttk.Checkbutton"),
            patch("tkinter.ttk.Radiobutton"),
            patch("tkinter.ttk.Progressbar"),
            patch("Folders_Tool_r0.ctypes"),
        ):
            FolderProcessorApp(mock_root)
            # If no exception, it passed
            assert True

    def test_get_unique_path(
        self, mock_root: Mock, mock_tk_vars: dict[str, Mock], tmp_path: Path
    ) -> None:
        """Test _get_unique_path."""
        with (
            patch("tkinter.ttk.Style"),
            patch("tkinter.Canvas"),
            patch("tkinter.ttk.Scrollbar"),
            patch("tkinter.ttk.Frame"),
            patch("tkinter.ttk.LabelFrame"),
            patch("tkinter.Listbox"),
            patch("tkinter.ttk.Button"),
            patch("tkinter.ttk.Label"),
            patch("tkinter.ttk.Entry"),
            patch("tkinter.ttk.Checkbutton"),
            patch("tkinter.ttk.Radiobutton"),
            patch("tkinter.ttk.Progressbar"),
            patch("Folders_Tool_r0.ctypes"),
        ):
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

    def test_validate_inputs_no_source(
        self, mock_root: Mock, mock_tk_vars: dict[str, Mock]
    ) -> None:
        """Test validate_inputs with no source."""
        with (
            patch("tkinter.ttk.Style"),
            patch("tkinter.Canvas"),
            patch("tkinter.ttk.Scrollbar"),
            patch("tkinter.ttk.Frame"),
            patch("tkinter.ttk.LabelFrame"),
            patch("tkinter.Listbox"),
            patch("tkinter.ttk.Button"),
            patch("tkinter.ttk.Label"),
            patch("tkinter.ttk.Entry"),
            patch("tkinter.ttk.Checkbutton"),
            patch("tkinter.ttk.Radiobutton"),
            patch("tkinter.ttk.Progressbar"),
            patch("Folders_Tool_r0.ctypes"),
            patch("Folders_Tool_r0.messagebox.showerror") as mock_error,
        ):
            app = FolderProcessorApp(mock_root)
            app.source_folders = []

            assert app.validate_inputs() is False
            mock_error.assert_called()

    def test_validate_inputs_no_dest(
        self, mock_root: Mock, mock_tk_vars: dict[str, Mock]
    ) -> None:
        """Test validate_inputs with no dest."""
        with (
            patch("tkinter.ttk.Style"),
            patch("tkinter.Canvas"),
            patch("tkinter.ttk.Scrollbar"),
            patch("tkinter.ttk.Frame"),
            patch("tkinter.ttk.LabelFrame"),
            patch("tkinter.Listbox"),
            patch("tkinter.ttk.Button"),
            patch("tkinter.ttk.Label"),
            patch("tkinter.ttk.Entry"),
            patch("tkinter.ttk.Checkbutton"),
            patch("tkinter.ttk.Radiobutton"),
            patch("tkinter.ttk.Progressbar"),
            patch("Folders_Tool_r0.ctypes"),
            patch("Folders_Tool_r0.messagebox.showerror") as mock_error,
        ):
            app = FolderProcessorApp(mock_root)
            app.source_folders = ["/some/path"]
            app.dest_folder = ""

            assert app.validate_inputs() is False
            mock_error.assert_called()

    def test_safe_copy_file_success(
        self, mock_root: Mock, mock_tk_vars: dict[str, Mock], tmp_path: Path
    ) -> None:
        """Test _safe_copy_file success case."""
        with (
            patch("tkinter.ttk.Style"),
            patch("tkinter.Canvas"),
            patch("tkinter.ttk.Scrollbar"),
            patch("tkinter.ttk.Frame"),
            patch("tkinter.ttk.LabelFrame"),
            patch("tkinter.Listbox"),
            patch("tkinter.ttk.Button"),
            patch("tkinter.ttk.Label"),
            patch("tkinter.ttk.Entry"),
            patch("tkinter.ttk.Checkbutton"),
            patch("tkinter.ttk.Radiobutton"),
            patch("tkinter.ttk.Progressbar"),
            patch("Folders_Tool_r0.ctypes"),
        ):
            app = FolderProcessorApp(mock_root)

            # Setup source and dest
            source_file = tmp_path / "source.txt"
            source_file.write_text("content")

            dest_file = tmp_path / "dest.txt"

            # Perform copy
            result = app._safe_copy_file(str(source_file), str(dest_file))

            assert result is True
            assert dest_file.exists()
            assert dest_file.read_text() == "content"

    def test_safe_copy_file_fail_source_not_found(
        self, mock_root: Mock, mock_tk_vars: dict[str, Mock], tmp_path: Path
    ) -> None:
        """Test _safe_copy_file source not found."""
        with (
            patch("tkinter.ttk.Style"),
            patch("tkinter.Canvas"),
            patch("tkinter.ttk.Scrollbar"),
            patch("tkinter.ttk.Frame"),
            patch("tkinter.ttk.LabelFrame"),
            patch("tkinter.Listbox"),
            patch("tkinter.ttk.Button"),
            patch("tkinter.ttk.Label"),
            patch("tkinter.ttk.Entry"),
            patch("tkinter.ttk.Checkbutton"),
            patch("tkinter.ttk.Radiobutton"),
            patch("tkinter.ttk.Progressbar"),
            patch("Folders_Tool_r0.ctypes"),
        ):
            app = FolderProcessorApp(mock_root)

            source_file = tmp_path / "non_existent.txt"
            dest_file = tmp_path / "dest.txt"

            with pytest.raises(FileNotFoundError):
                app._safe_copy_file(str(source_file), str(dest_file))

    def test_create_output_zip_success(
        self, mock_root: Mock, mock_tk_vars: dict[str, Mock], tmp_path: Path
    ) -> None:
        """Test create_output_zip success."""
        with (
            patch("tkinter.ttk.Style"),
            patch("tkinter.Canvas"),
            patch("tkinter.ttk.Scrollbar"),
            patch("tkinter.ttk.Frame"),
            patch("tkinter.ttk.LabelFrame"),
            patch("tkinter.Listbox"),
            patch("tkinter.ttk.Button"),
            patch("tkinter.ttk.Label"),
            patch("tkinter.ttk.Entry"),
            patch("tkinter.ttk.Checkbutton"),
            patch("tkinter.ttk.Radiobutton"),
            patch("tkinter.ttk.Progressbar"),
            patch("Folders_Tool_r0.ctypes"),
        ):
            app = FolderProcessorApp(mock_root)

            # Setup destination folder with files
            dest_folder = tmp_path / "output_folder"
            dest_folder.mkdir()
            (dest_folder / "file1.txt").write_text("content1")

            app.dest_folder = str(dest_folder)

            # Mock update_progress to avoid errors if it tries to update UI
            app.update_progress = Mock()

            zip_path = app.create_output_zip()

            assert os.path.exists(zip_path)
            assert zip_path.endswith(".zip")

            # Verify zip content
            import zipfile

            with zipfile.ZipFile(zip_path, "r") as zf:
                assert "file1.txt" in zf.namelist()

    def test_create_output_zip_empty_dest(
        self, mock_root: Mock, mock_tk_vars: dict[str, Mock], tmp_path: Path
    ) -> None:
        """Test create_output_zip with empty destination."""
        with (
            patch("tkinter.ttk.Style"),
            patch("tkinter.Canvas"),
            patch("tkinter.ttk.Scrollbar"),
            patch("tkinter.ttk.Frame"),
            patch("tkinter.ttk.LabelFrame"),
            patch("tkinter.Listbox"),
            patch("tkinter.ttk.Button"),
            patch("tkinter.ttk.Label"),
            patch("tkinter.ttk.Entry"),
            patch("tkinter.ttk.Checkbutton"),
            patch("tkinter.ttk.Radiobutton"),
            patch("tkinter.ttk.Progressbar"),
            patch("Folders_Tool_r0.ctypes"),
        ):
            app = FolderProcessorApp(mock_root)

            dest_folder = tmp_path / "empty_folder"
            dest_folder.mkdir()
            app.dest_folder = str(dest_folder)

            # Should raise ValueError because folder is empty
            with pytest.raises(ValueError, match="Destination folder is empty"):
                app.create_output_zip()

    def test_validate_size_inputs(
        self, mock_root: Mock, mock_tk_vars: dict[str, Mock]
    ) -> None:
        """Test validate_size_inputs."""
        with (
            patch("tkinter.ttk.Style"),
            patch("tkinter.Canvas"),
            patch("tkinter.ttk.Scrollbar"),
            patch("tkinter.ttk.Frame"),
            patch("tkinter.ttk.LabelFrame"),
            patch("tkinter.Listbox"),
            patch("tkinter.ttk.Button"),
            patch("tkinter.ttk.Label"),
            patch("tkinter.ttk.Entry"),
            patch("tkinter.ttk.Checkbutton"),
            patch("tkinter.ttk.Radiobutton"),
            patch("tkinter.ttk.Progressbar"),
            patch("Folders_Tool_r0.ctypes"),
            patch("Folders_Tool_r0.messagebox.showwarning") as mock_warning,
            patch("Folders_Tool_r0.messagebox.showerror") as mock_error,
        ):
            app = FolderProcessorApp(mock_root)

            # Use distinct mocks for min and max to avoid shared state if StringVar
            # returns singleton mock
            app.min_file_size = Mock()
            app.max_file_size = Mock()

            # Valid inputs
            app.min_file_size.get.return_value = "0"
            app.max_file_size.get.return_value = "10"
            assert app.validate_size_inputs() is True

            # Invalid min size (negative)
            app.min_file_size.get.return_value = "-1"
            assert app.validate_size_inputs() is False
            mock_warning.assert_called()

            # Invalid max size (negative)
            app.min_file_size.get.return_value = "0"
            app.max_file_size.get.return_value = "-1"
            assert app.validate_size_inputs() is False

            # Min > Max
            app.min_file_size.get.return_value = "20"
            app.max_file_size.get.return_value = "10"
            assert app.validate_size_inputs() is False

            # Value error (non-numeric) - Should return False and show error
            app.min_file_size.get.side_effect = ValueError("Invalid float")
            assert app.validate_size_inputs() is False
            mock_error.assert_called()

            # Reset side effect
            app.min_file_size.get.side_effect = None

            # Also mock min/max file size for validate_file_filters test requirement

    def test_validate_file_filters(
        self, mock_root: Mock, mock_tk_vars: dict[str, Mock]
    ) -> None:
        """Test validate_file_filters."""
        with (
            patch("tkinter.ttk.Style"),
            patch("tkinter.Canvas"),
            patch("tkinter.ttk.Scrollbar"),
            patch("tkinter.ttk.Frame"),
            patch("tkinter.ttk.LabelFrame"),
            patch("tkinter.Listbox"),
            patch("tkinter.ttk.Button"),
            patch("tkinter.ttk.Label"),
            patch("tkinter.ttk.Entry"),
            patch("tkinter.ttk.Checkbutton"),
            patch("tkinter.ttk.Radiobutton"),
            patch("tkinter.ttk.Progressbar"),
            patch("Folders_Tool_r0.ctypes"),
            patch("os.path.getsize") as mock_getsize,
            patch("os.path.exists"),
        ):
            app = FolderProcessorApp(mock_root)

            # Use distinct mocks
            app.min_file_size = Mock()
            app.max_file_size = Mock()

            # Setup filter extensions
            app.filter_extensions.get.return_value = ".txt,.log"

            # Mock size inputs
            app.min_file_size.get.return_value = "0"
            app.max_file_size.get.return_value = "100"

            # Mock file size to 1 byte
            mock_getsize.return_value = 1

            assert app.validate_file_filters("test.txt") is True
            assert app.validate_file_filters("test.log") is True
            assert app.validate_file_filters("test.py") is False

            # Empty filter
            app.filter_extensions.get.return_value = ""
            assert app.validate_file_filters("test.py") is True

    def test_validate_application_state(
        self, mock_root: Mock, mock_tk_vars: dict[str, Mock]
    ) -> None:
        """Test validate_application_state."""
        with (
            patch("tkinter.ttk.Style"),
            patch("tkinter.Canvas"),
            patch("tkinter.ttk.Scrollbar"),
            patch("tkinter.ttk.Frame"),
            patch("tkinter.ttk.LabelFrame"),
            patch("tkinter.Listbox"),
            patch("tkinter.ttk.Button"),
            patch("tkinter.ttk.Label"),
            patch("tkinter.ttk.Entry"),
            patch("tkinter.ttk.Checkbutton"),
            patch("tkinter.ttk.Radiobutton"),
            patch("tkinter.ttk.Progressbar"),
            patch("Folders_Tool_r0.ctypes"),
            patch("os.path.exists") as mock_exists,
            patch("os.access") as mock_access,
        ):
            app = FolderProcessorApp(mock_root)
            app.source_folders = ["/src"]
            app.dest_folder = "/dest"

            mock_exists.return_value = True
            mock_access.return_value = True

            state = app.validate_application_state()
            # Keys based on implementation
            assert state["source_folders_exist"] is True
            assert state["destination_exists"] is True
            assert state["destination_writable"] is True

            app.source_folders = []
            state = app.validate_application_state()
            assert state["source_folders_exist"] is True

    def test_safe_extract_archive(
        self, mock_root: Mock, mock_tk_vars: dict[str, Mock], tmp_path: Path
    ) -> None:
        """Test safe_extract_archive."""
        # Mock UI creation methods to avoid patching all widgets
        with (
            patch("tkinter.ttk.Style"),
            patch.object(FolderProcessorApp, "create_scrollable_interface"),
            patch.object(FolderProcessorApp, "_setup_application_icon"),
            patch("Folders_Tool_r0.ctypes"),
            patch("shutil.unpack_archive"),
            patch("Folders_Tool_r0.logger"),
            patch("shutil.rmtree") as mock_rmtree,
            patch("os.path.getsize") as mock_getsize,
        ):
            app = FolderProcessorApp(mock_root)
            # Init manually creates vars, but we need to ensure safe_extract_var is
            # working
            app.safe_extract_var.get.return_value = True

            # 1. Archive file not found
            with pytest.raises(FileNotFoundError):
                app.safe_extract_archive(str(tmp_path / "non_existent.zip"))

            # 2. Archive file is empty
            empty_zip = tmp_path / "empty.zip"
            empty_zip.touch()
            success, msg = app.safe_extract_archive(str(empty_zip))
            assert success is False
            assert "empty" in msg

            # 3. Successful extraction (mocked)
            valid_zip = tmp_path / "valid.zip"
            valid_zip.write_text("content")

            # Combined mocks for success case
            with (
                patch("os.access", return_value=True),
                patch("pathlib.Path.unlink") as mock_unlink,
                patch(
                    "pathlib.Path.iterdir", return_value=[Path("extracted_file.txt")]
                ),
                patch("os.walk") as mock_walk,
                patch("pathlib.Path.exists", return_value=True),
            ):
                # Mock result for getsize: Archive size > 0, then file sizes
                mock_getsize.side_effect = lambda p: 100

                app._get_unique_path = Mock(return_value=str(tmp_path / "valid"))

                # Setup os.walk to return one file
                mock_walk.return_value = [
                    (str(tmp_path / "valid"), [], ["extracted_file.txt"])
                ]

                success, msg = app.safe_extract_archive(str(valid_zip))

                assert success is True
                assert "Successfully extracted" in msg
                mock_unlink.assert_called_once()

            # 4. Extraction failure (exception during unpack)
            with (
                patch("os.access", return_value=True),
                patch("shutil.unpack_archive", side_effect=Exception("Unpack error")),
            ):
                mock_getsize.side_effect = None
                mock_getsize.return_value = 100

                success, msg = app.safe_extract_archive(str(valid_zip))
                assert success is False
                assert "Failed to extract" in msg
                mock_rmtree.assert_called()

    def test_cancel_processing(
        self, mock_root: Mock, mock_tk_vars: dict[str, Mock]
    ) -> None:
        """Test cancel_processing."""
        with (
            patch("tkinter.ttk.Style"),
            patch("tkinter.Canvas"),
            patch("tkinter.ttk.Scrollbar"),
            patch("tkinter.ttk.Frame"),
            patch("tkinter.ttk.LabelFrame"),
            patch("tkinter.Listbox"),
            patch("tkinter.ttk.Button"),
            patch("tkinter.ttk.Label"),
            patch("tkinter.ttk.Entry"),
            patch("tkinter.ttk.Checkbutton"),
            patch("tkinter.ttk.Radiobutton"),
            patch("tkinter.ttk.Progressbar"),
            patch("Folders_Tool_r0.ctypes"),
        ):
            app = FolderProcessorApp(mock_root)
            app.status_var = Mock()

            app.cancel_processing()

            assert app.cancel_operation is True
            app.status_var.set.assert_called_with("Cancelling operation...")

    def test_processing_complete(
        self, mock_root: Mock, mock_tk_vars: dict[str, Mock]
    ) -> None:
        """Test processing_complete."""
        with (
            patch("tkinter.ttk.Style"),
            patch("tkinter.Canvas"),
            patch("tkinter.ttk.Scrollbar"),
            patch("tkinter.ttk.Frame"),
            patch("tkinter.ttk.LabelFrame"),
            patch("tkinter.Listbox"),
            patch("tkinter.ttk.Button"),
            patch("tkinter.ttk.Label"),
            patch("tkinter.ttk.Entry"),
            patch("tkinter.ttk.Checkbutton"),
            patch("tkinter.ttk.Radiobutton"),
            patch("tkinter.ttk.Progressbar"),
            patch("Folders_Tool_r0.ctypes"),
        ):
            app = FolderProcessorApp(mock_root)
            app.run_button = Mock()
            app.cancel_button = Mock()
            app.progress_var = Mock()
            app.status_var = Mock()

            app.processing_complete()

            app.run_button.config.assert_called_with(state=tk.NORMAL)
            app.cancel_button.config.assert_called_with(state=tk.DISABLED)
            app.progress_var.set.assert_called_with(0)
            app.status_var.set.assert_called_with("Ready")

    def test_update_progress(
        self, mock_root: Mock, mock_tk_vars: dict[str, Mock]
    ) -> None:
        """Test update_progress."""
        with (
            patch("tkinter.ttk.Style"),
            patch("tkinter.Canvas"),
            patch("tkinter.ttk.Scrollbar"),
            patch("tkinter.ttk.Frame"),
            patch("tkinter.ttk.LabelFrame"),
            patch("tkinter.Listbox"),
            patch("tkinter.ttk.Button"),
            patch("tkinter.ttk.Label"),
            patch("tkinter.ttk.Entry"),
            patch("tkinter.ttk.Checkbutton"),
            patch("tkinter.ttk.Radiobutton"),
            patch("tkinter.ttk.Progressbar"),
            patch("Folders_Tool_r0.ctypes"),
        ):
            app = FolderProcessorApp(mock_root)
            app.progress_var = Mock()
            app.status_var = Mock()

            # Normal update
            app.update_progress(50, "Halfway")
            app.progress_var.set.assert_called_with(50.0)
            app.status_var.set.assert_called_with("Halfway")

            # Clamp high
            app.update_progress(150)
            app.progress_var.set.assert_called_with(100.0)

            # Clamp low
            app.update_progress(-50)
            app.progress_var.set.assert_called_with(0.0)

            # Invalid type (should log warning but not crash)
            app.update_progress("invalid")

    def test_run_processing_threaded(
        self, mock_root: Mock, mock_tk_vars: dict[str, Mock]
    ) -> None:
        """Test run_processing_threaded."""
        with (
            patch("tkinter.ttk.Style"),
            patch("tkinter.Canvas"),
            patch("tkinter.ttk.Scrollbar"),
            patch("tkinter.ttk.Frame"),
            patch("tkinter.ttk.LabelFrame"),
            patch("tkinter.Listbox"),
            patch("tkinter.ttk.Button"),
            patch("tkinter.ttk.Label"),
            patch("tkinter.ttk.Entry"),
            patch("tkinter.ttk.Checkbutton"),
            patch("tkinter.ttk.Radiobutton"),
            patch("tkinter.ttk.Progressbar"),
            patch("Folders_Tool_r0.ctypes"),
            patch("threading.Thread") as mock_thread,
        ):
            app = FolderProcessorApp(mock_root)
            app.run_button = Mock()
            app.cancel_button = Mock()

            app.run_processing_threaded()

            assert app.cancel_operation is False
            app.run_button.config.assert_called_with(state=tk.DISABLED)
            app.cancel_button.config.assert_called_with(state=tk.NORMAL)

            mock_thread.assert_called_once()
            mock_thread.return_value.start.assert_called_once()
