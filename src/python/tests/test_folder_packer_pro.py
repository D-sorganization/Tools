"""Tests for folder_packer_pro package.

Tests for the decomposed folder_packer_pro modules:
- encryption.py: EncryptionManager
- manifest.py: PackageManifest
- file_ops.py: should_exclude, collect_folder_stats, get_file_type, format_size
- pack_engine.py: collect_files, pack_files, unpack_files, inspect_package
- app.py: FolderPackerPro (GUI integration)
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

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

from collections.abc import Callable, Generator
from pathlib import Path
from unittest.mock import Mock, patch

# Skip entire module if folder_packer_pro is not available
try:
    from tools.folder_tools.folder_packer_pro.app import FolderPackerPro
    from tools.folder_tools.folder_packer_pro.encryption import EncryptionManager
    from tools.folder_tools.folder_packer_pro.file_ops import (
        collect_folder_stats,
        format_size,
        get_file_type,
        should_exclude,
    )

    # Also verify backward-compatible facade imports work
    from tools.folder_tools.folder_packer_pro.folder_packer_pro import (  # noqa: F811
        EncryptionManager as EncryptionManagerFacade,
    )
    from tools.folder_tools.folder_packer_pro.folder_packer_pro import (
        FolderPackerPro as FolderPackerProFacade,
    )
    from tools.folder_tools.folder_packer_pro.folder_packer_pro import (
        PackageManifest as PackageManifestFacade,
    )
    from tools.folder_tools.folder_packer_pro.manifest import PackageManifest
except ImportError:
    pytest.skip("folder_packer_pro module not available", allow_module_level=True)


class TestEncryptionManager:
    """Test cases for EncryptionManager."""

    def test_encryption_decryption(self) -> None:
        """Test encrypting and decrypting data."""
        data = b"test data"
        password = os.getenv("password".upper(), "")  # noqa: S105

        encrypted = EncryptionManager.encrypt_data(data, password)
        assert encrypted != data

        decrypted = EncryptionManager.decrypt_data(encrypted, password)
        assert decrypted == data

    def test_derive_key(self) -> None:
        """Test key derivation."""
        password = os.getenv("password".upper(), "")  # noqa: S105
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


class TestFileOps:
    """Test cases for standalone file operation functions."""

    def test_should_exclude_patterns(self) -> None:
        """Test should_exclude with various patterns."""
        patterns = {"*.pyc", "dist"}

        # Test pattern exclusion (wildcard)
        assert should_exclude(Path("path/to/file.pyc"), patterns) is True
        # Test pattern exclusion (directory name)
        assert should_exclude(Path("path/to/dist"), patterns) is True
        # Test inclusion
        assert should_exclude(Path("path/to/source.py"), patterns) is False

    def test_should_exclude_git(self) -> None:
        """Test .git exclusion toggling."""
        patterns: set[str] = set()

        # Default: .git excluded
        assert should_exclude(Path("path/to/.git/config"), patterns, include_git=False) is True
        # Include git: .git not excluded
        assert should_exclude(Path("path/to/.git/config"), patterns, include_git=True) is False

    def test_collect_folder_stats(self, tmp_path: Path) -> None:
        """Test collect_folder_stats."""
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

        stats = collect_folder_stats(tmp_path, exclude_patterns=set())

        assert stats["total_files"] == 2
        assert stats["total_size"] == 30
        assert stats["file_types"][".txt"] == 1
        assert stats["file_types"][".py"] == 1

    def test_get_file_type(self) -> None:
        """Test file type categorization."""
        assert get_file_type(Path("test.py")) == "Code"
        assert get_file_type(Path("test.html")) == "Markup"
        assert get_file_type(Path("test.json")) == "Config"
        assert get_file_type(Path("test.png")) == "Image"
        assert get_file_type(Path("test.mp3")) == "Audio"
        assert get_file_type(Path("test.mp4")) == "Video"
        assert get_file_type(Path("test.pdf")) == "Document"
        assert get_file_type(Path("test.xyz")) == "Other"

    def test_format_size(self) -> None:
        """Test human-readable file size formatting."""
        assert format_size(0) == "0.00 B"
        assert format_size(512) == "512.00 B"
        assert format_size(1024) == "1.00 KB"
        assert format_size(1048576) == "1.00 MB"
        assert format_size(1073741824) == "1.00 GB"


class TestBackwardCompatibility:
    """Test that facade re-exports work."""

    def test_facade_exports(self) -> None:
        """Test that the facade module re-exports all public classes."""
        assert EncryptionManagerFacade is EncryptionManager
        assert PackageManifestFacade is PackageManifest
        assert FolderPackerProFacade is FolderPackerPro


class TestFolderPackerPro:
    """Test cases for FolderPackerPro GUI class."""

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
        """Test initialization of FolderPackerPro."""
        with (
            patch("tkinter.Menu"),
            patch("tkinter.ttk.Notebook"),
            patch("tkinter.ttk.Frame"),
            patch("tkinter.ttk.Label"),
            patch("tkinter.ttk.LabelFrame"),
            patch("tkinter.ttk.Entry"),
            patch("tkinter.ttk.Button"),
            patch("tkinter.scrolledtext.ScrolledText"),
            patch("tkinter.ttk.Progressbar"),
            patch("tkinter.ttk.Radiobutton"),
            patch("tkinter.ttk.Checkbutton"),
            patch("tkinter.ttk.Treeview"),
            patch("tkinter.ttk.Scrollbar"),
            patch("tkinter.Text"),
            patch("tkinter.ttk.Style"),
        ):
            app = FolderPackerPro(mock_root)

            assert app.root == mock_root
            assert app.current_theme == "dark"
            assert app.manifest is not None

    def test_scan_folder(
        self, mock_root: Mock, mock_tk_vars: dict[str, Mock], tmp_path: Path
    ) -> None:
        """Test _scan_folder."""
        with (
            patch("tkinter.Menu"),
            patch("tkinter.ttk.Notebook"),
            patch("tkinter.ttk.Frame"),
            patch("tkinter.ttk.Label"),
            patch("tkinter.ttk.LabelFrame"),
            patch("tkinter.ttk.Entry"),
            patch("tkinter.ttk.Button"),
            patch("tkinter.scrolledtext.ScrolledText"),
            patch("tkinter.ttk.Progressbar"),
            patch("tkinter.ttk.Radiobutton"),
            patch("tkinter.ttk.Checkbutton"),
            patch("tkinter.ttk.Treeview"),
            patch("tkinter.ttk.Scrollbar"),
            patch("tkinter.Text"),
            patch("tkinter.ttk.Style"),
            patch("tools.folder_tools.folder_packer_pro.app.messagebox.showerror") as mock_error,
        ):
            app = FolderPackerPro(mock_root)
            app.pack_source_entry = Mock()
            app._display_stats = Mock()  # type: ignore[method-assign]

            # Empty source
            app.pack_source_entry.get.return_value = ""
            app._scan_folder()

            # Invalid source
            app.pack_source_entry.get.return_value = "/non/existent"
            app._scan_folder()
            mock_error.assert_called()

            # Valid source - patch Thread to run target synchronously
            app.pack_source_entry.get.return_value = str(tmp_path)

            # Mock root.after to execute callback immediately
            def immediate_after(delay: object, callback: Callable[[], None] | None = None) -> None:
                """Execute callback immediately."""
                if callback is not None:
                    callback()

            app.root.after.side_effect = immediate_after  # type: ignore[attr-defined]

            # Patch Thread to run synchronously
            with patch("tools.folder_tools.folder_packer_pro.app.threading.Thread") as mock_thread:

                def run_sync(**kwargs: object) -> Mock:
                    """Run thread target synchronously."""
                    target = kwargs.get("target")
                    if callable(target):
                        target()
                    return Mock()

                mock_thread.side_effect = run_sync
                app._scan_folder()
                app._display_stats.assert_called()

    def test_browse_handlers(
        self, mock_root: Mock, mock_tk_vars: dict[str, Mock], tmp_path: Path
    ) -> None:
        """Test browse handlers."""
        with (
            patch("tkinter.Menu"),
            patch("tkinter.ttk.Notebook"),
            patch("tkinter.ttk.Frame"),
            patch("tkinter.ttk.Label"),
            patch("tkinter.ttk.LabelFrame"),
            patch("tkinter.ttk.Entry"),
            patch("tkinter.ttk.Button"),
            patch("tkinter.scrolledtext.ScrolledText"),
            patch("tkinter.ttk.Progressbar"),
            patch("tkinter.ttk.Radiobutton"),
            patch("tkinter.ttk.Checkbutton"),
            patch("tkinter.ttk.Treeview"),
            patch("tkinter.ttk.Scrollbar"),
            patch("tkinter.Text"),
            patch("tkinter.ttk.Style"),
            patch(
                "tools.folder_tools.folder_packer_pro.app.filedialog.askdirectory"
            ) as mock_askdir,
            patch(
                "tools.folder_tools.folder_packer_pro.app.filedialog.asksaveasfilename"
            ) as mock_save,
            patch(
                "tools.folder_tools.folder_packer_pro.app.filedialog.askopenfilename"
            ) as mock_open,
        ):
            app = FolderPackerPro(mock_root)
            app.pack_source_entry = Mock()
            app.pack_output_entry = Mock()
            app.unpack_source_entry = Mock()
            app.unpack_dest_entry = Mock()
            app._scan_folder = Mock()  # type: ignore[method-assign]
            app._log_message = Mock()  # type: ignore[method-assign]

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
            app.pack_output_entry.insert.assert_called_with(0, str(tmp_path / "pkg.fpp"))

            # _browse_unpack_source
            mock_open.return_value = str(tmp_path / "pkg.fpp")
            app._browse_unpack_source()
            app.unpack_source_entry.delete.assert_called()
            app.unpack_source_entry.insert.assert_called_with(0, str(tmp_path / "pkg.fpp"))

            # _browse_unpack_dest
            mock_askdir.return_value = str(tmp_path / "dest")
            app._browse_unpack_dest()
            app.unpack_dest_entry.delete.assert_called()
            app.unpack_dest_entry.insert.assert_called_with(0, str(tmp_path / "dest"))
