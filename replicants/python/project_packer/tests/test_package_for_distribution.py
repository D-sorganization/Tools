"""Tests for package_for_distribution.py module."""

import sys
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Import the module to test
sys.path.insert(0, str(Path(__file__).parent.parent))
from package_for_distribution import (
    copy_required_files,
    create_deployment_package,
    create_zip_archive,
    main,
)


class TestPackageForDistribution:
    """Test cases for package_for_distribution.py module."""

    def test_create_deployment_package_success(self, tmp_path: Path) -> None:
        """Test successful deployment package creation."""
        with patch("package_for_distribution.Path") as mock_path_class:
            # Mock the Path class to return our tmp_path
            mock_path_class.return_value.parent = tmp_path

            # Mock the packages directory creation
            packages_dir = tmp_path / "packages"
            packages_dir.mkdir()

            with patch("package_for_distribution.copy_required_files") as mock_copy:
                mock_copy.return_value = True
                with patch("package_for_distribution.create_zip_archive") as mock_zip:
                    result = create_deployment_package()
                    assert result is True
                    mock_copy.assert_called_once()
                    mock_zip.assert_called_once()

    def test_create_deployment_package_copy_failure(self, tmp_path: Path) -> None:
        """Test deployment package creation when file copying fails."""
        with patch("package_for_distribution.Path") as mock_path_class:
            # Mock the Path class to return our tmp_path
            mock_path_class.return_value.parent = tmp_path

            packages_dir = tmp_path / "packages"
            packages_dir.mkdir()

            with patch("package_for_distribution.copy_required_files") as mock_copy:
                mock_copy.return_value = False
                result = create_deployment_package()
                assert result is False

    def test_copy_required_files_all_exist(self, tmp_path: Path) -> None:
        """Test copying required files when all exist."""
        source_dir = tmp_path / "source"
        package_dir = tmp_path / "package"
        source_dir.mkdir()
        package_dir.mkdir()

        # Create test files
        test_files = ["folder_packer_gui.py", "build_exe.py", "README.md"]
        for file in test_files:
            (source_dir / file).write_text("test content")

        with patch("package_for_distribution.REQUIRED_FILES", test_files):
            result = copy_required_files(source_dir, package_dir)
            assert result is True

            # Check that files were copied
            for file in test_files:
                assert (package_dir / file).exists()

    def test_copy_required_files_some_missing(self, tmp_path: Path) -> None:
        """Test copying required files when some are missing."""
        source_dir = tmp_path / "source"
        package_dir = tmp_path / "package"
        source_dir.mkdir()
        package_dir.mkdir()

        # Create only some test files
        test_files = ["folder_packer_gui.py", "build_exe.py", "README.md"]
        existing_files = ["folder_packer_gui.py", "README.md"]
        for file in existing_files:
            (source_dir / file).write_text("test content")

        with patch("package_for_distribution.REQUIRED_FILES", test_files):
            result = copy_required_files(source_dir, package_dir)
            assert result is True

            # Check that existing files were copied
            for file in existing_files:
                assert (package_dir / file).exists()

    def test_copy_required_files_copy_error(self, tmp_path: Path) -> None:
        """Test copying required files when copy operation fails."""
        source_dir = tmp_path / "source"
        package_dir = tmp_path / "package"
        source_dir.mkdir()
        package_dir.mkdir()

        # Create test file
        test_file = "folder_packer_gui.py"
        (source_dir / test_file).write_text("test content")

        with patch("package_for_distribution.REQUIRED_FILES", [test_file]):
            with patch("shutil.copy2") as mock_copy:
                mock_copy.side_effect = OSError("Permission denied")
                result = copy_required_files(source_dir, package_dir)
                assert result is False

    def test_create_zip_archive_success(self, tmp_path: Path) -> None:
        """Test successful zip archive creation."""
        source_dir = tmp_path / "source"
        zip_path = tmp_path / "test.zip"
        source_dir.mkdir()

        # Create test files
        (source_dir / "file1.txt").write_text("content 1")
        (source_dir / "file2.txt").write_text("content 2")
        (source_dir / "subdir").mkdir()
        (source_dir / "subdir" / "file3.txt").write_text("content 3")

        create_zip_archive(source_dir, zip_path)

        # Verify zip was created
        assert zip_path.exists()

        # Verify contents
        with zipfile.ZipFile(zip_path, "r") as zipf:
            file_list = zipf.namelist()
            assert "file1.txt" in file_list
            assert "file2.txt" in file_list
            assert "subdir/file3.txt" in file_list

    def test_create_zip_archive_empty_directory(self, tmp_path: Path) -> None:
        """Test zip archive creation from empty directory."""
        source_dir = tmp_path / "source"
        zip_path = tmp_path / "test.zip"
        source_dir.mkdir()

        create_zip_archive(source_dir, zip_path)

        # Verify zip was created (even if empty)
        assert zip_path.exists()

    def test_main_success(self) -> None:
        """Test main function when package creation succeeds."""
        with patch("package_for_distribution.create_deployment_package") as mock_create:
            mock_create.return_value = True
            with patch("sys.exit") as mock_exit:
                main()
                mock_exit.assert_not_called()

    def test_main_failure(self) -> None:
        """Test main function when package creation fails."""
        with patch("package_for_distribution.create_deployment_package") as mock_create:
            mock_create.return_value = False
            with patch("sys.exit") as mock_exit:
                main()
                mock_exit.assert_called_once_with(1)

    def test_package_version_constant(self) -> None:
        """Test that package version constant is properly defined."""
        from package_for_distribution import PACKAGE_VERSION

        assert isinstance(PACKAGE_VERSION, str)
        assert len(PACKAGE_VERSION) > 0

    def test_package_base_name_constant(self) -> None:
        """Test that package base name constant is properly defined."""
        from package_for_distribution import PACKAGE_BASE_NAME

        assert isinstance(PACKAGE_BASE_NAME, str)
        assert len(PACKAGE_BASE_NAME) > 0

    def test_required_files_constant(self) -> None:
        """Test that required files constant is properly defined."""
        from package_for_distribution import REQUIRED_FILES

        assert isinstance(REQUIRED_FILES, list)
        assert len(REQUIRED_FILES) > 0
        assert all(isinstance(f, str) for f in REQUIRED_FILES)


if __name__ == "__main__":
    pytest.main([__file__])
