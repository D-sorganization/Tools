"""Tests for os_utils - OS path operation utilities.

These tests verify the OS utility functions using
Design by Contract principles.
"""

import os
from pathlib import Path

import pytest


class TestSafeJoinPathContract:
    """Design by Contract tests for safe_join_path function."""

    def test_returns_path(self, tmp_path):
        """Postcondition: Returns a Path object."""
        from utils.os_utils import safe_join_path

        result = safe_join_path(tmp_path, "subdir")
        assert isinstance(result, Path)

    def test_rejects_traversal(self, tmp_path):
        """Precondition: Rejects path traversal attempts."""
        from utils.os_utils import safe_join_path

        with pytest.raises(ValueError):
            safe_join_path(tmp_path, "..", "etc", "passwd")


class TestSafeJoinPath:
    """Functional tests for safe_join_path."""

    def test_joins_simple_paths(self, tmp_path):
        """Test joining simple path parts."""
        from utils.os_utils import safe_join_path

        result = safe_join_path(tmp_path, "subdir", "file.txt")
        assert result == tmp_path / "subdir" / "file.txt"

    def test_accepts_string_base(self):
        """Test accepting string as base path."""
        from utils.os_utils import safe_join_path

        result = safe_join_path("/tmp", "test")
        assert isinstance(result, Path)


class TestGetCurrentDirContract:
    """Design by Contract tests for get_current_dir function."""

    def test_returns_path(self):
        """Postcondition: Returns a Path object."""
        from utils.os_utils import get_current_dir

        result = get_current_dir()
        assert isinstance(result, Path)

    def test_returns_existing_path(self):
        """Postcondition: Returns an existing path."""
        from utils.os_utils import get_current_dir

        result = get_current_dir()
        assert result.exists()


class TestGetCurrentDir:
    """Functional tests for get_current_dir."""

    def test_matches_os_getcwd(self):
        """Test matching os.getcwd()."""
        from utils.os_utils import get_current_dir

        result = get_current_dir()
        assert str(result) == os.getcwd()


class TestChangeDirectoryContract:
    """Design by Contract tests for change_directory context manager."""

    def test_raises_for_nonexistent_dir(self, tmp_path):
        """Precondition: Raises FileNotFoundError for nonexistent directory."""
        from utils.os_utils import change_directory

        with pytest.raises(FileNotFoundError):
            with change_directory(tmp_path / "nonexistent"):
                pass

    def test_raises_for_file_not_dir(self, tmp_path):
        """Precondition: Raises NotADirectoryError for file."""
        from utils.os_utils import change_directory

        test_file = tmp_path / "file.txt"
        test_file.write_text("content")

        with pytest.raises(NotADirectoryError):
            with change_directory(test_file):
                pass


class TestChangeDirectory:
    """Functional tests for change_directory."""

    def test_changes_and_restores_directory(self, tmp_path):
        """Test changing and restoring directory."""
        from utils.os_utils import change_directory

        original = Path.cwd()

        with change_directory(tmp_path) as new_dir:
            assert Path.cwd() == tmp_path
            assert new_dir == tmp_path

        assert Path.cwd() == original

    def test_restores_on_exception(self, tmp_path):
        """Test restoring directory even on exception."""
        from utils.os_utils import change_directory

        original = Path.cwd()

        with pytest.raises(ValueError):
            with change_directory(tmp_path):
                raise ValueError("Test error")

        assert Path.cwd() == original

    def test_yields_path_object(self, tmp_path):
        """Test yielding Path object."""
        from utils.os_utils import change_directory

        with change_directory(tmp_path) as result:
            assert isinstance(result, Path)


class TestPathExistsContract:
    """Design by Contract tests for path_exists function."""

    def test_returns_bool(self, tmp_path):
        """Postcondition: Returns a boolean."""
        from utils.os_utils import path_exists

        result = path_exists(tmp_path)
        assert isinstance(result, bool)


class TestPathExists:
    """Functional tests for path_exists."""

    def test_returns_true_for_existing_path(self, tmp_path):
        """Test returning True for existing path."""
        from utils.os_utils import path_exists

        assert path_exists(tmp_path) is True

    def test_returns_false_for_nonexistent_path(self, tmp_path):
        """Test returning False for nonexistent path."""
        from utils.os_utils import path_exists

        assert path_exists(tmp_path / "nonexistent") is False

    def test_accepts_string_path(self, tmp_path):
        """Test accepting string path."""
        from utils.os_utils import path_exists

        result = path_exists(str(tmp_path))
        assert result is True


class TestEnsureDirContract:
    """Design by Contract tests for ensure_dir function."""

    def test_returns_path(self, tmp_path):
        """Postcondition: Returns a Path object."""
        from utils.os_utils import ensure_dir

        result = ensure_dir(tmp_path)
        assert isinstance(result, Path)

    def test_raises_for_missing_without_create(self, tmp_path):
        """Precondition: Raises FileNotFoundError when create=False."""
        from utils.os_utils import ensure_dir

        with pytest.raises(FileNotFoundError):
            ensure_dir(tmp_path / "nonexistent", create=False)


class TestEnsureDir:
    """Functional tests for ensure_dir."""

    def test_returns_existing_dir(self, tmp_path):
        """Test returning existing directory."""
        from utils.os_utils import ensure_dir

        result = ensure_dir(tmp_path)
        assert result == tmp_path

    def test_creates_new_dir(self, tmp_path):
        """Test creating new directory."""
        from utils.os_utils import ensure_dir

        new_dir = tmp_path / "new_directory"
        result = ensure_dir(new_dir, create=True)

        assert result == new_dir
        assert new_dir.exists()

    def test_raises_for_file_as_dir(self, tmp_path):
        """Test raising for file path when directory expected."""
        from utils.os_utils import ensure_dir

        test_file = tmp_path / "file.txt"
        test_file.write_text("content")

        with pytest.raises(NotADirectoryError):
            ensure_dir(test_file)
