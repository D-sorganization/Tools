"""Tests for folder_packer_pro.py."""

import os
import sys
from collections.abc import Callable, Generator
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add folder_packer_pro directory to path
sys.path.append(str(Path(__file__).parent.parent / "folder_packer_pro"))

from folder_packer_pro import (
    EncryptionManager,
    FolderPackerPro,
    PackageManifest,
)


class TestEncryptionManager:
    """Test cases for EncryptionManager."""

    def test_encryption_decryption(self) -> None:
        """Test encrypting and decrypting data."""
        data = b"test data"
        password = "test_password"  # noqa: S105

        encrypted = EncryptionManager.encrypt_data(data, password)
        assert encrypted != data

        decrypted = EncryptionManager.decrypt_data(encrypted, password)
        assert decrypted == data

    def test_derive_key(self) -> None:
        """Test key derivation."""
        password = "test_password"  # noqa: S105
        salt = os.urandom(16)

        key1 = EncryptionManager.derive_key(password, salt)
        key2 = EncryptionManager.derive_key(password, salt)

        assert key1 == key2
        assert len(key1) > 0


class TestPackageManifest:
    """Test cases for PackageManifest."""

    def test_init(self) -> None:
        """Test manifest initialization."""
        manifest = PackageManifest()
        assert manifest.files == []
        assert manifest.metadata == {}
        assert manifest.stats["total_files"] == 0

    def test_add_file(self) -> None:
        """Test adding file to manifest."""
        manifest = PackageManifest()
        manifest.add_file("test.txt", 100, "checksum")

        assert len(manifest.files) == 1
        assert manifest.files[0]["path"] == "test.txt"
        assert manifest.files[0]["size"] == 100
        assert manifest.files[0]["checksum"] == "checksum"
        assert manifest.stats["total_files"] == 1
        assert manifest.stats["total_size"] == 100

    def test_serialization(self) -> None:
        """Test to_json and from_json."""
        manifest = PackageManifest()
        manifest.add_file("test.txt", 100, "checksum")
        manifest.set_metadata("version", "1.0")

        json_str = manifest.to_json()
        manifest2 = PackageManifest.from_json(json_str)

        assert len(manifest2.files) == 1
        assert manifest2.files[0]["path"] == "test.txt"
        assert manifest2.metadata["version"] == "1.0"
        assert manifest2.stats["total_files"] == 1


class TestFolderPackerPro:
    """Test cases for FolderPackerPro GUI class."""

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
        """Test initialization of FolderPackerPro."""
        with patch("tkinter.Menu"), patch("tkinter.ttk.Notebook"), patch(
            "tkinter.ttk.Frame"
        ), patch("tkinter.ttk.Label"), patch("tkinter.ttk.LabelFrame"), patch(
            "tkinter.ttk.Entry"
        ), patch(
            "tkinter.ttk.Button"
        ), patch(
            "tkinter.scrolledtext.ScrolledText"
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
            "tkinter.Text"
        ), patch(
            "tkinter.ttk.Style"
        ):

            app = FolderPackerPro(mock_root)

            assert app.root == mock_root
            assert app.current_theme == "dark"
            assert app.manifest is not None

    def test_should_exclude(
        self, mock_root: Mock, mock_tk_vars: dict[str, Mock]
    ) -> None:
        """Test _should_exclude logic."""
        with patch("tkinter.Menu"), patch("tkinter.ttk.Notebook"), patch(
            "tkinter.ttk.Frame"
        ), patch("tkinter.ttk.Label"), patch("tkinter.ttk.LabelFrame"), patch(
            "tkinter.ttk.Entry"
        ), patch(
            "tkinter.ttk.Button"
        ), patch(
            "tkinter.scrolledtext.ScrolledText"
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
            "tkinter.Text"
        ), patch(
            "tkinter.ttk.Style"
        ):

            app = FolderPackerPro(mock_root)

            # Setup vars
            app.include_git_var.get.return_value = False  # type: ignore  # noqa: PGH003
            app.exclude_patterns = ["*.pyc", "dist"]  # type: ignore  # noqa: PGH003

            # Test git exclusion
            assert app._should_exclude(Path("path/to/.git/config")) is True
            # Test pattern exclusion
            assert app._should_exclude(Path("path/to/file.pyc")) is True
            # For directories, it checks the name.
            assert app._should_exclude(Path("path/to/dist")) is True
            # For files inside dist, the walker would skip dist, but should_exclude on the file itself follows name rules
            # so file.txt is NOT excluded unless "dist" is in its name.
            assert app._should_exclude(Path("path/to/dist/file.txt")) is False

            # Test inclusion
            assert app._should_exclude(Path("path/to/source.py")) is False

            # Test git inclusion
            app.include_git_var.get.return_value = True  # type: ignore  # noqa: PGH003
            assert app._should_exclude(Path("path/to/.git/config")) is False

    def test_collect_folder_stats(
        self, mock_root: Mock, mock_tk_vars: dict[str, Mock], tmp_path: Path
    ) -> None:
        """Test _collect_folder_stats."""
        with patch("tkinter.Menu"), patch("tkinter.ttk.Notebook"), patch(
            "tkinter.ttk.Frame"
        ), patch("tkinter.ttk.Label"), patch("tkinter.ttk.LabelFrame"), patch(
            "tkinter.ttk.Entry"
        ), patch(
            "tkinter.ttk.Button"
        ), patch(
            "tkinter.scrolledtext.ScrolledText"
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
            "tkinter.Text"
        ), patch(
            "tkinter.ttk.Style"
        ):

            app = FolderPackerPro(mock_root)
            app.include_git_var.get.return_value = False  # type: ignore  # noqa: PGH003
            app.exclude_patterns = []  # type: ignore  # noqa: PGH003

            # Create dummy structure

            file1 = tmp_path / "file1.txt"
            file1.write_text("a" * 10)

            src_dir = tmp_path / "src"
            src_dir.mkdir()
            file2 = src_dir / "file2.py"
            file2.write_text("a" * 20)

            git_dir = tmp_path / ".git"
            git_dir.mkdir()
            (git_dir / "config").write_text("content")

            stats = app._collect_folder_stats(tmp_path)

            assert stats["total_files"] == 2
            assert stats["total_size"] == 30
            assert stats["file_types"][".txt"] == 1
            assert stats["file_types"][".py"] == 1

    def test_scan_folder(
        self, mock_root: Mock, mock_tk_vars: dict[str, Mock], tmp_path: Path
    ) -> None:
        """Test _scan_folder."""
        with patch("tkinter.Menu"), patch("tkinter.ttk.Notebook"), patch(
            "tkinter.ttk.Frame"
        ), patch("tkinter.ttk.Label"), patch("tkinter.ttk.LabelFrame"), patch(
            "tkinter.ttk.Entry"
        ), patch(
            "tkinter.ttk.Button"
        ), patch(
            "tkinter.scrolledtext.ScrolledText"
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
            "tkinter.Text"
        ), patch(
            "tkinter.ttk.Style"
        ), patch(
            "tkinter.messagebox.showerror"
        ) as mock_error:

            app = FolderPackerPro(mock_root)
            app.pack_source_entry = Mock()
            app._collect_folder_stats = Mock(return_value={})  # type: ignore  # noqa: PGH003
            app._display_stats = Mock()  # type: ignore  # noqa: PGH003

            # Empty source
            app.pack_source_entry.get.return_value = ""
            app._scan_folder()
            app._collect_folder_stats.assert_not_called()

            # Invalid source
            app.pack_source_entry.get.return_value = "/non/existent"
            app._scan_folder()
            mock_error.assert_called()

            # Valid source
            app.pack_source_entry.get.return_value = str(tmp_path)

            # Mock root.after to execute immediately
            def immediate_after(delay: object, callback: Callable[[], None]) -> None:
                callback()

            app.root.after.side_effect = immediate_after  # type: ignore  # noqa: PGH003

            app._scan_folder()
            app._collect_folder_stats.assert_called_with(Path(str(tmp_path)))
            app._display_stats.assert_called()

    def test_browse_handlers(
        self, mock_root: Mock, mock_tk_vars: dict[str, Mock], tmp_path: Path
    ) -> None:
        """Test browse handlers."""
        with patch("tkinter.Menu"), patch("tkinter.ttk.Notebook"), patch(
            "tkinter.ttk.Frame"
        ), patch("tkinter.ttk.Label"), patch("tkinter.ttk.LabelFrame"), patch(
            "tkinter.ttk.Entry"
        ), patch(
            "tkinter.ttk.Button"
        ), patch(
            "tkinter.scrolledtext.ScrolledText"
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
            "tkinter.Text"
        ), patch(
            "tkinter.ttk.Style"
        ), patch(
            "tkinter.filedialog.askdirectory"
        ) as mock_askdir, patch(
            "tkinter.filedialog.asksaveasfilename"
        ) as mock_save, patch(
            "tkinter.filedialog.askopenfilename"
        ) as mock_open:

            app = FolderPackerPro(mock_root)
            app.pack_source_entry = Mock()
            app.pack_output_entry = Mock()
            app.unpack_source_entry = Mock()
            app.unpack_dest_entry = Mock()
            app._scan_folder = Mock()  # type: ignore  # noqa: PGH003
            app._log_message = Mock()  # type: ignore  # noqa: PGH003

            # _browse_pack_source
            mock_askdir.return_value = str(tmp_path)
            app._browse_pack_source()
            app.pack_source_entry.delete.assert_called()
            app.pack_source_entry.insert.assert_called_with(0, str(tmp_path))
            app._scan_folder.assert_called()

            # _browse_pack_output
            mock_save.return_value = str(tmp_path / "pkg.fpp")
            app._browse_pack_output()
            app.pack_output_entry.delete.assert_called()
            app.pack_output_entry.insert.assert_called_with(
                0, str(tmp_path / "pkg.fpp")
            )

            # _browse_unpack_source
            mock_open.return_value = str(tmp_path / "pkg.fpp")
            app._browse_unpack_source()
            app.unpack_source_entry.delete.assert_called()
            app.unpack_source_entry.insert.assert_called_with(
                0, str(tmp_path / "pkg.fpp")
            )

            # _browse_unpack_dest
            mock_askdir.return_value = str(tmp_path / "dest")
            app._browse_unpack_dest()
            app.unpack_dest_entry.delete.assert_called()
            app.unpack_dest_entry.insert.assert_called_with(0, str(tmp_path / "dest"))
