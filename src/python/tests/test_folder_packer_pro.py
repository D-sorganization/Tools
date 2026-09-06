# ruff: noqa: E501
"""Tests for the folder_packer_pro package.

Covers the parts of ``src/folder_packer_pro`` that the suites under
``tests/folder_packer_pro/`` do not: the EncryptionManager crypto API, the
PackageManifest serialization contract, the backward-compatible facade
re-exports, and the FolderPackerPro app bootstrap/browse handlers.
"""

import os
from collections.abc import Generator
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from folder_packer_pro.app import FolderPackerPro
from folder_packer_pro.encryption import EncryptionManager
from folder_packer_pro.folder_packer_pro import (
    EncryptionManager as EncryptionManagerFacade,
)
from folder_packer_pro.folder_packer_pro import FolderPackerPro as FolderPackerProFacade
from folder_packer_pro.folder_packer_pro import PackageManifest as PackageManifestFacade
from folder_packer_pro.manifest import PackageManifest


class TestEncryptionManager:
    """Test cases for EncryptionManager."""

    def test_encryption_decryption(self) -> None:
        """Test encrypting and decrypting data."""
        data = b"test data"
        password = "test_password"

        encrypted = EncryptionManager.encrypt_data(data, password)
        assert encrypted != data

        decrypted = EncryptionManager.decrypt_data(encrypted, password)
        assert decrypted == data

    def test_derive_key(self) -> None:
        """Test key derivation."""
        password = "test_password"
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
            patch("folder_packer_pro.app.filedialog.askdirectory") as mock_askdir,
            patch("folder_packer_pro.app.filedialog.asksaveasfilename") as mock_save,
            patch("folder_packer_pro.app.filedialog.askopenfilename") as mock_open,
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


if __name__ == "__main__":
    pytest.main([__file__])
