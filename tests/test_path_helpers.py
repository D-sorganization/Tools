"""Tests for path_helpers - Path helper utilities.

These tests verify the path helper functions using
Design by Contract principles.
"""

from pathlib import Path

import pytest


class TestGetFileDirContract:
    """Design by Contract tests for get_file_dir function."""

    def test_returns_path(self, tmp_path):
        """Postcondition: Returns a Path object."""
        from utils.path_helpers import get_file_dir

        test_file = tmp_path / "test.py"
        test_file.write_text("")

        result = get_file_dir(test_file)
        assert isinstance(result, Path)

    def test_returns_resolved_path(self, tmp_path):
        """Postcondition: Returns an absolute path."""
        from utils.path_helpers import get_file_dir

        test_file = tmp_path / "test.py"
        test_file.write_text("")

        result = get_file_dir(test_file)
        assert result.is_absolute()


class TestGetFileDir:
    """Functional tests for get_file_dir."""

    def test_returns_parent_directory(self, tmp_path):
        """Test returning parent directory of file."""
        from utils.path_helpers import get_file_dir

        subdir = tmp_path / "src" / "module"
        subdir.mkdir(parents=True)
        test_file = subdir / "script.py"
        test_file.write_text("")

        result = get_file_dir(test_file)
        assert result == subdir

    def test_accepts_string_path(self, tmp_path):
        """Test accepting string path."""
        from utils.path_helpers import get_file_dir

        test_file = tmp_path / "test.py"
        test_file.write_text("")

        result = get_file_dir(str(test_file))
        assert result == tmp_path


class TestGetRelativePathContract:
    """Design by Contract tests for get_relative_path function."""

    def test_returns_path(self, tmp_path):
        """Postcondition: Returns a Path object."""
        from utils.path_helpers import get_relative_path

        result = get_relative_path(tmp_path, tmp_path / "subdir")
        assert isinstance(result, Path)


class TestGetRelativePath:
    """Functional tests for get_relative_path."""

    def test_returns_relative_path_for_child(self, tmp_path):
        """Test returning relative path for child directory."""
        from utils.path_helpers import get_relative_path

        child = tmp_path / "src" / "module"
        child.mkdir(parents=True)

        result = get_relative_path(tmp_path, child)
        assert result == Path("src/module") or result == Path("src\\module")

    def test_returns_absolute_for_unrelated_paths(self, tmp_path):
        """Test returning absolute path for unrelated paths."""
        from utils.path_helpers import get_relative_path

        other_path = Path("/some/other/path")
        result = get_relative_path(tmp_path, other_path)

        # Should return the resolved to_path since they're not related
        assert result.is_absolute()


class TestFindNearestFileContract:
    """Design by Contract tests for find_nearest_file function."""

    def test_returns_path_or_none(self, tmp_path):
        """Postcondition: Returns Path or None."""
        from utils.path_helpers import find_nearest_file

        result = find_nearest_file("nonexistent.txt", tmp_path)
        assert result is None or isinstance(result, Path)


class TestFindNearestFile:
    """Functional tests for find_nearest_file."""

    def test_finds_file_in_current_directory(self, tmp_path):
        """Test finding file in current directory."""
        from utils.path_helpers import find_nearest_file

        test_file = tmp_path / "setup.py"
        test_file.write_text("")

        result = find_nearest_file("setup.py", tmp_path)
        assert result == test_file

    def test_finds_file_in_parent_directory(self, tmp_path):
        """Test finding file in parent directory."""
        from utils.path_helpers import find_nearest_file

        # Create marker file in root
        marker = tmp_path / ".project-root"
        marker.write_text("")

        # Search from child
        child = tmp_path / "src" / "pkg"
        child.mkdir(parents=True)

        result = find_nearest_file(".project-root", child)
        assert result == marker

    def test_returns_none_when_not_found(self, tmp_path):
        """Test returning None when file not found."""
        from utils.path_helpers import find_nearest_file

        result = find_nearest_file("does_not_exist.xyz", tmp_path)
        assert result is None


class TestFindNearestDirContract:
    """Design by Contract tests for find_nearest_dir function."""

    def test_returns_path_or_none(self, tmp_path):
        """Postcondition: Returns Path or None."""
        from utils.path_helpers import find_nearest_dir

        result = find_nearest_dir("nonexistent_dir", tmp_path)
        assert result is None or isinstance(result, Path)


class TestFindNearestDir:
    """Functional tests for find_nearest_dir."""

    def test_finds_directory_in_current(self, tmp_path):
        """Test finding directory in current location."""
        from utils.path_helpers import find_nearest_dir

        venv_dir = tmp_path / ".venv"
        venv_dir.mkdir()

        result = find_nearest_dir(".venv", tmp_path)
        assert result == venv_dir

    def test_finds_directory_in_parent(self, tmp_path):
        """Test finding directory in parent."""
        from utils.path_helpers import find_nearest_dir

        # Create .git in root
        git_dir = tmp_path / ".git"
        git_dir.mkdir()

        # Search from child
        child = tmp_path / "src" / "submodule"
        child.mkdir(parents=True)

        result = find_nearest_dir(".git", child)
        assert result == git_dir

    def test_returns_none_when_not_found(self, tmp_path):
        """Test returning None when not found."""
        from utils.path_helpers import find_nearest_dir

        result = find_nearest_dir("nonexistent_directory", tmp_path)
        assert result is None


class TestNormalizePathContract:
    """Design by Contract tests for normalize_path function."""

    def test_returns_path(self, tmp_path):
        """Postcondition: Returns a Path object."""
        from utils.path_helpers import normalize_path

        result = normalize_path(tmp_path)
        assert isinstance(result, Path)

    def test_returns_absolute_path(self, tmp_path):
        """Postcondition: Returns an absolute path."""
        from utils.path_helpers import normalize_path

        result = normalize_path("relative/path")
        assert result.is_absolute()


class TestNormalizePath:
    """Functional tests for normalize_path."""

    def test_resolves_relative_path(self):
        """Test resolving relative path."""
        from utils.path_helpers import normalize_path

        result = normalize_path(".")
        assert result.is_absolute()
        assert result.exists()

    def test_accepts_path_object(self, tmp_path):
        """Test accepting Path object."""
        from utils.path_helpers import normalize_path

        result = normalize_path(tmp_path)
        assert result == tmp_path.resolve()

    def test_accepts_string(self, tmp_path):
        """Test accepting string."""
        from utils.path_helpers import normalize_path

        result = normalize_path(str(tmp_path))
        assert result == tmp_path.resolve()


class TestSafeJoinPathContract:
    """Design by Contract tests for safe_join_path function."""

    def test_returns_path(self, tmp_path):
        """Postcondition: Returns a Path object."""
        from utils.path_helpers import safe_join_path

        result = safe_join_path(tmp_path, "subdir", "file.txt")
        assert isinstance(result, Path)

    def test_raises_on_traversal(self, tmp_path):
        """Precondition: Raises ValueError on path traversal."""
        from utils.path_helpers import safe_join_path

        with pytest.raises(ValueError, match="[Uu]nsafe|traversal"):
            safe_join_path(tmp_path, "..", "etc", "passwd")


class TestSafeJoinPath:
    """Functional tests for safe_join_path."""

    def test_joins_simple_path(self, tmp_path):
        """Test joining simple path parts."""
        from utils.path_helpers import safe_join_path

        result = safe_join_path(tmp_path, "src", "module", "file.py")
        expected = tmp_path / "src" / "module" / "file.py"
        assert result == expected.resolve()

    def test_rejects_double_dot(self, tmp_path):
        """Test rejecting double dot traversal."""
        from utils.path_helpers import safe_join_path

        with pytest.raises(ValueError):
            safe_join_path(tmp_path, "subdir", "..", "..", "escape")

    def test_rejects_absolute_part(self, tmp_path):
        """Test rejecting absolute path part."""
        from utils.path_helpers import safe_join_path

        with pytest.raises(ValueError):
            safe_join_path(tmp_path, "/absolute/path")

    def test_result_within_base(self, tmp_path):
        """Test that result stays within base."""
        from utils.path_helpers import safe_join_path

        result = safe_join_path(tmp_path, "nested", "path")
        # Result should be relative to base
        assert str(result).startswith(str(tmp_path.resolve()))
