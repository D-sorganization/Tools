"""Tests for folder_fix_pro.py."""

import hashlib
import os
import sys
from collections.abc import Generator
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add folder_tool_pro directory to path
sys.path.append(str(Path(__file__).parent.parent / "folder_tool_pro"))

from folder_fix_pro import (
    FileHasher,
    FolderFixPro,
    OperationReport,
)


class TestFileHasher:
    """Test cases for FileHasher."""

    def test_hash_file(self, tmp_path: Path) -> None:
        """Test hashing a file."""
        p = tmp_path / "test.txt"
        p.write_text("test content")

        expected_hash = hashlib.sha256(b"test content").hexdigest()
        assert FileHasher.hash_file(p) == expected_hash

    def test_hash_file_fast(self, tmp_path: Path) -> None:
        """Test fast hashing."""
        p = tmp_path / "test.txt"
        p.write_text("test content")

        # Calculate expected hash for fast method
        size = p.stat().st_size
        hasher = hashlib.sha256()
        hasher.update(str(size).encode())
        with open(p, "rb") as f:
            chunk = f.read(65536)  # DEFAULT_CHUNK_SIZE
            hasher.update(chunk)
            # File is small, so last chunk logic won't trigger if it's smaller than 2*CHUNK_SIZE

        expected_hash = hasher.hexdigest()
        assert FileHasher.hash_file_fast(p) == expected_hash

    def test_hash_file_fast_large_file(self, tmp_path: Path) -> None:
        """Test fast hashing with large file to trigger last chunk logic."""
        p = tmp_path / "large_test.bin"
        # Create file larger than 2 * DEFAULT_CHUNK_SIZE (65536 * 2 = 131072)
        size = 150000
        content = os.urandom(size)
        p.write_bytes(content)

        # Calculate expected hash manually
        hasher = hashlib.sha256()
        hasher.update(str(size).encode())

        # First chunk
        hasher.update(content[:65536])

        # Last chunk
        hasher.update(content[-65536:])

        assert FileHasher.hash_file_fast(p) == hasher.hexdigest()


class TestOperationReport:
    """Test cases for OperationReport."""

    def test_add_operation(self) -> None:
        """Test adding operations."""
        report = OperationReport()
        report.add_operation("copy", {"source": "a", "dest": "b"})

        assert len(report.operations) == 1
        assert report.stats["copy"] == 1

    def test_add_error(self) -> None:
        """Test adding errors."""
        report = OperationReport()
        report.add_error("test error")

        assert len(report.errors) == 1

    def test_finalize(self) -> None:
        """Test finalizing report."""
        report = OperationReport()
        assert report.end_time is None
        report.finalize()
        assert report.end_time is not None


class TestFolderFixPro:
    """Test cases for FolderFixPro GUI class."""

    @pytest.fixture  # type: ignore  # noqa: PGH003
    def mock_root(self) -> Mock:
        """Mock Tkinter root."""
        return Mock()

    @pytest.fixture  # type: ignore  # noqa: PGH003
    def mock_tk_vars(self) -> Generator[dict[str, Mock], None, None]:
        """Mock Tkinter variables."""
        with patch("tkinter.BooleanVar") as mock_bool, patch(
            "tkinter.StringVar"
        ) as mock_string, patch("tkinter.DoubleVar") as mock_double, patch(
            "tkinter.IntVar"
        ) as mock_int:

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
        """Test initialization of FolderFixPro."""
        with patch("tkinter.Menu"), patch("tkinter.ttk.Notebook"), patch(
            "tkinter.ttk.Frame"
        ), patch("tkinter.ttk.Label"), patch("tkinter.ttk.LabelFrame"), patch(
            "tkinter.ttk.Entry"
        ), patch(
            "tkinter.ttk.Button"
        ), patch(
            "tkinter.Text"
        ), patch(
            "tkinter.ttk.Progressbar"
        ), patch(
            "tkinter.ttk.Radiobutton"
        ), patch(
            "tkinter.ttk.Checkbutton"
        ), patch(
            "tkinter.ttk.Treeview"
        ), patch(
            "tkinter.ttk.Scrollbar"
        ), patch(
            "tkinter.Listbox"
        ), patch(
            "tkinter.ttk.Style"
        ), patch(
            "folder_fix_pro.ctypes"
        ):

            app = FolderFixPro(mock_root)

            assert app.root == mock_root
            assert app.current_theme == "dark"
            assert isinstance(app.operation_report, OperationReport)

    def test_should_include_file(
        self, mock_root: Mock, mock_tk_vars: dict[str, Mock], tmp_path: Path
    ) -> None:
        """Test _should_include_file logic."""
        with patch("tkinter.Menu"), patch("tkinter.ttk.Notebook"), patch(
            "tkinter.ttk.Frame"
        ), patch("tkinter.ttk.Label"), patch("tkinter.ttk.LabelFrame"), patch(
            "tkinter.ttk.Entry"
        ), patch(
            "tkinter.ttk.Button"
        ), patch(
            "tkinter.Text"
        ), patch(
            "tkinter.ttk.Progressbar"
        ), patch(
            "tkinter.ttk.Radiobutton"
        ), patch(
            "tkinter.ttk.Checkbutton"
        ), patch(
            "tkinter.ttk.Treeview"
        ), patch(
            "tkinter.ttk.Scrollbar"
        ), patch(
            "tkinter.Listbox"
        ), patch(
            "tkinter.ttk.Style"
        ), patch(
            "folder_fix_pro.ctypes"
        ):

            app = FolderFixPro(mock_root)

            # Setup defaults
            app.skip_hidden_var = Mock()
            app.skip_hidden_var.get.return_value = False

            app.min_size_entry = Mock()
            app.min_size_entry.get.return_value = "0"

            app.max_size_entry = Mock()
            # Set a large max size
            app.max_size_entry.get.return_value = "10"  # 10 MB

            app.ext_filter_entry = Mock()
            app.ext_filter_entry.get.return_value = ""

            app.regex_filter_entry = Mock()
            app.regex_filter_entry.get.return_value = ""

            # Create test files
            file1 = tmp_path / "normal.txt"
            file1.write_text("content")

            file2 = tmp_path / ".hidden"
            file2.write_text("content")

            # Test normal inclusion
            assert app._should_include_file(file1) is True

            # Test hidden exclusion
            app.skip_hidden_var.get.return_value = True
            assert app._should_include_file(file2) is False
            app.skip_hidden_var.get.return_value = False

            # Test extension filter
            app.ext_filter_entry.get.return_value = ".jpg,.png"
            assert app._should_include_file(file1) is False

            jpg_file = tmp_path / "test.jpg"
            jpg_file.write_text("content")
            assert app._should_include_file(jpg_file) is True

            # Test regex filter
            app.ext_filter_entry.get.return_value = ""
            app.regex_filter_entry.get.return_value = r"^test.*"

            assert app._should_include_file(jpg_file) is True
            assert app._should_include_file(file1) is False

    def test_count_files(
        self, mock_root: Mock, mock_tk_vars: dict[str, Mock], tmp_path: Path
    ) -> None:
        """Test _count_files logic."""
        with patch("tkinter.Menu"), patch("tkinter.ttk.Notebook"), patch(
            "tkinter.ttk.Frame"
        ), patch("tkinter.ttk.Label"), patch("tkinter.ttk.LabelFrame"), patch(
            "tkinter.ttk.Entry"
        ), patch(
            "tkinter.ttk.Button"
        ), patch(
            "tkinter.Text"
        ), patch(
            "tkinter.ttk.Progressbar"
        ), patch(
            "tkinter.ttk.Radiobutton"
        ), patch(
            "tkinter.ttk.Checkbutton"
        ), patch(
            "tkinter.ttk.Treeview"
        ), patch(
            "tkinter.ttk.Scrollbar"
        ), patch(
            "tkinter.Listbox"
        ), patch(
            "tkinter.ttk.Style"
        ), patch(
            "folder_fix_pro.ctypes"
        ):

            app = FolderFixPro(mock_root)

            # Create structure
            # root/
            #   src1/
            #     f1.txt
            #     f2.txt
            #   src2/
            #     f3.txt

            src1 = tmp_path / "src1"
            src1.mkdir()
            (src1 / "f1.txt").write_text("c")
            (src1 / "f2.txt").write_text("c")

            src2 = tmp_path / "src2"
            src2.mkdir()
            (src2 / "f3.txt").write_text("c")

            app.source_folders = [str(src1), str(src2)]

            assert app._count_files() == 3

    def test_operation_analyze(
        self, mock_root: Mock, mock_tk_vars: dict[str, Mock], tmp_path: Path
    ) -> None:
        """Test _operation_analyze."""
        with patch("tkinter.Menu"), patch("tkinter.ttk.Notebook"), patch(
            "tkinter.ttk.Frame"
        ), patch("tkinter.ttk.Label"), patch("tkinter.ttk.LabelFrame"), patch(
            "tkinter.ttk.Entry"
        ), patch(
            "tkinter.ttk.Button"
        ), patch(
            "tkinter.Text"
        ), patch(
            "tkinter.ttk.Progressbar"
        ), patch(
            "tkinter.ttk.Radiobutton"
        ), patch(
            "tkinter.ttk.Checkbutton"
        ), patch(
            "tkinter.ttk.Treeview"
        ), patch(
            "tkinter.ttk.Scrollbar"
        ), patch(
            "tkinter.Listbox"
        ), patch(
            "tkinter.ttk.Style"
        ), patch(
            "folder_fix_pro.ctypes"
        ):

            app = FolderFixPro(mock_root)

            src = tmp_path / "src"
            src.mkdir()
            (src / "f1.txt").write_text("a")
            (src / "f2.log").write_text("b" * 100)

            app.source_folders = [str(src)]
            app._should_include_file = Mock(return_value=True)  # type: ignore  # noqa: PGH003
            app._update_progress = Mock()  # type: ignore  # noqa: PGH003
            app._show_analysis_results = Mock()  # type: ignore  # noqa: PGH003
            app._log_message = Mock()  # type: ignore  # noqa: PGH003

            app._operation_analyze()

            # Verify results
            assert app._show_analysis_results.called
            stats = app._show_analysis_results.call_args[0][0]
            assert stats["total_files"] == 2
            assert stats["total_size"] == 101
            assert stats["file_types"][".txt"] == 1
            assert stats["file_types"][".log"] == 1

    def test_operation_combine(
        self, mock_root: Mock, mock_tk_vars: dict[str, Mock], tmp_path: Path
    ) -> None:
        """Test _operation_combine."""
        with patch("tkinter.Menu"), patch("tkinter.ttk.Notebook"), patch(
            "tkinter.ttk.Frame"
        ), patch("tkinter.ttk.Label"), patch("tkinter.ttk.LabelFrame"), patch(
            "tkinter.ttk.Entry"
        ), patch(
            "tkinter.ttk.Button"
        ), patch(
            "tkinter.Text"
        ), patch(
            "tkinter.ttk.Progressbar"
        ), patch(
            "tkinter.ttk.Radiobutton"
        ), patch(
            "tkinter.ttk.Checkbutton"
        ), patch(
            "tkinter.ttk.Treeview"
        ), patch(
            "tkinter.ttk.Scrollbar"
        ), patch(
            "tkinter.Listbox"
        ), patch(
            "tkinter.ttk.Style"
        ), patch(
            "folder_fix_pro.ctypes"
        ):

            app = FolderFixPro(mock_root)

            src = tmp_path / "src"
            src.mkdir()
            (src / "f1.txt").write_text("content")

            dest = tmp_path / "dest"

            app.source_folders = [str(src)]
            app.dest_folder = str(dest)
            app._should_include_file = Mock(return_value=True)  # type: ignore  # noqa: PGH003
            app.preview_var.get.return_value = False  # type: ignore  # noqa: PGH003
            app._update_progress = Mock()  # type: ignore  # noqa: PGH003

            app._operation_combine()

            assert (dest / "f1.txt").exists()
            assert (dest / "f1.txt").read_text() == "content"

    def test_operation_flatten(
        self, mock_root: Mock, mock_tk_vars: dict[str, Mock], tmp_path: Path
    ) -> None:
        """Test _operation_flatten."""
        with patch("tkinter.Menu"), patch("tkinter.ttk.Notebook"), patch(
            "tkinter.ttk.Frame"
        ), patch("tkinter.ttk.Label"), patch("tkinter.ttk.LabelFrame"), patch(
            "tkinter.ttk.Entry"
        ), patch(
            "tkinter.ttk.Button"
        ), patch(
            "tkinter.Text"
        ), patch(
            "tkinter.ttk.Progressbar"
        ), patch(
            "tkinter.ttk.Radiobutton"
        ), patch(
            "tkinter.ttk.Checkbutton"
        ), patch(
            "tkinter.ttk.Treeview"
        ), patch(
            "tkinter.ttk.Scrollbar"
        ), patch(
            "tkinter.Listbox"
        ), patch(
            "tkinter.ttk.Style"
        ), patch(
            "folder_fix_pro.ctypes"
        ):

            app = FolderFixPro(mock_root)

            src = tmp_path / "src"
            src.mkdir()
            sub = src / "sub"
            sub.mkdir()
            (sub / "f1.txt").write_text("content")

            dest = tmp_path / "dest"

            app.source_folders = [str(src)]
            app.dest_folder = str(dest)
            app.organize_type_var.get.return_value = False  # type: ignore  # noqa: PGH003
            app.organize_date_var.get.return_value = False  # type: ignore  # noqa: PGH003
            app._should_include_file = Mock(return_value=True)  # type: ignore  # noqa: PGH003
            app.preview_var.get.return_value = False  # type: ignore  # noqa: PGH003
            app._update_progress = Mock()  # type: ignore  # noqa: PGH003

            app._operation_flatten()

            # File should be at root of dest, not in subfolder
            assert (dest / "f1.txt").exists()
