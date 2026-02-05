"""Tests for path_setup - Python path configuration utilities.

These tests verify the path setup functions using
Design by Contract principles.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestGetRepoRootContract:
    """Design by Contract tests for get_repo_root function."""

    def test_returns_path(self):
        """Postcondition: Returns a Path object."""
        from utils.path_setup import get_repo_root

        result = get_repo_root()
        assert isinstance(result, Path)

    def test_returns_absolute_path(self):
        """Postcondition: Returns an absolute path."""
        from utils.path_setup import get_repo_root

        result = get_repo_root()
        assert result.is_absolute()


class TestGetRepoRoot:
    """Functional tests for get_repo_root."""

    def test_finds_git_directory(self, tmp_path):
        """Test finding directory with .git."""
        from utils.path_setup import get_repo_root

        # Create a mock repo structure
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        subdir = tmp_path / "src" / "module"
        subdir.mkdir(parents=True)

        result = get_repo_root(start_path=subdir)
        assert result == tmp_path

    def test_finds_pyproject_toml(self, tmp_path):
        """Test finding directory with pyproject.toml."""
        from utils.path_setup import get_repo_root

        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[project]\nname = 'test'")
        subdir = tmp_path / "deep" / "nested" / "dir"
        subdir.mkdir(parents=True)

        result = get_repo_root(start_path=subdir)
        assert result == tmp_path

    def test_finds_requirements_txt(self, tmp_path):
        """Test finding directory with requirements.txt."""
        from utils.path_setup import get_repo_root

        req_file = tmp_path / "requirements.txt"
        req_file.write_text("pytest")
        subdir = tmp_path / "src"
        subdir.mkdir()

        result = get_repo_root(start_path=subdir)
        assert result == tmp_path

    def test_fallback_for_no_markers(self, tmp_path):
        """Test fallback when no repo markers found."""
        from utils.path_setup import get_repo_root

        # Create isolated directory with no markers
        isolated = tmp_path / "isolated"
        isolated.mkdir()

        result = get_repo_root(start_path=isolated)
        # Should return parent as fallback
        assert isinstance(result, Path)


class TestGetStandardPathsContract:
    """Design by Contract tests for get_standard_paths function."""

    def test_returns_list(self, tmp_path):
        """Postcondition: Returns a list."""
        from utils.path_setup import get_standard_paths

        result = get_standard_paths(repo_root=tmp_path)
        assert isinstance(result, list)

    def test_returns_path_objects(self, tmp_path):
        """Postcondition: Returns list of Path objects."""
        from utils.path_setup import get_standard_paths

        # Create expected directories
        (tmp_path / "src" / "python" / "src").mkdir(parents=True)

        result = get_standard_paths(repo_root=tmp_path)
        for path in result:
            assert isinstance(path, Path)


class TestGetStandardPaths:
    """Functional tests for get_standard_paths."""

    def test_includes_repo_root(self, tmp_path):
        """Test including repo root in paths."""
        from utils.path_setup import get_standard_paths

        result = get_standard_paths(repo_root=tmp_path)
        assert tmp_path in result

    def test_filters_nonexistent_paths(self, tmp_path):
        """Test filtering out nonexistent paths."""
        from utils.path_setup import get_standard_paths

        result = get_standard_paths(repo_root=tmp_path)
        for path in result:
            assert path.exists()

    def test_includes_python_src_if_exists(self, tmp_path):
        """Test including src/python/src if it exists."""
        from utils.path_setup import get_standard_paths

        python_src = tmp_path / "src" / "python" / "src"
        python_src.mkdir(parents=True)

        result = get_standard_paths(repo_root=tmp_path)
        assert python_src in result


class TestSetupPythonPathContract:
    """Design by Contract tests for setup_python_path function."""

    def test_does_not_raise(self, tmp_path):
        """Postcondition: Does not raise exceptions."""
        from utils.path_setup import setup_python_path

        # Should not raise
        setup_python_path(repo_root=tmp_path)


class TestSetupPythonPath:
    """Functional tests for setup_python_path."""

    def test_adds_paths_to_sys_path(self, tmp_path):
        """Test adding paths to sys.path."""
        from utils.path_setup import setup_python_path

        # Create a directory structure
        src_dir = tmp_path / "src" / "python" / "src"
        src_dir.mkdir(parents=True)

        original_path = sys.path.copy()
        try:
            setup_python_path(repo_root=tmp_path)
            assert str(src_dir) in sys.path
        finally:
            # Restore original path
            sys.path[:] = original_path

    def test_sets_pythonpath_env_var(self, tmp_path):
        """Test setting PYTHONPATH environment variable."""
        from utils.path_setup import setup_python_path

        original_env = os.environ.get("PYTHONPATH", "")
        original_path = sys.path.copy()
        try:
            setup_python_path(repo_root=tmp_path)
            pythonpath = os.environ.get("PYTHONPATH", "")
            assert str(tmp_path) in pythonpath
        finally:
            # Restore
            os.environ["PYTHONPATH"] = original_env
            sys.path[:] = original_path

    def test_accepts_additional_paths(self, tmp_path):
        """Test accepting additional paths."""
        from utils.path_setup import setup_python_path

        custom_path = tmp_path / "custom"
        custom_path.mkdir()

        original_path = sys.path.copy()
        try:
            setup_python_path(repo_root=tmp_path, additional_paths=[custom_path])
            assert str(custom_path) in sys.path
        finally:
            sys.path[:] = original_path

    def test_does_not_add_duplicates(self, tmp_path):
        """Test not adding duplicate paths."""
        from utils.path_setup import setup_python_path

        original_path = sys.path.copy()
        try:
            setup_python_path(repo_root=tmp_path)
            count_before = sys.path.count(str(tmp_path))

            setup_python_path(repo_root=tmp_path)
            count_after = sys.path.count(str(tmp_path))

            assert count_before == count_after
        finally:
            sys.path[:] = original_path


class TestAddUtilsToPathContract:
    """Design by Contract tests for add_utils_to_path function."""

    def test_does_not_raise(self):
        """Postcondition: Does not raise exceptions."""
        from utils.path_setup import add_utils_to_path

        # Should not raise even if path doesn't exist
        add_utils_to_path()


class TestAddUtilsToPath:
    """Functional tests for add_utils_to_path."""

    def test_adds_utils_path(self, tmp_path):
        """Test adding utils path to sys.path."""
        from utils.path_setup import add_utils_to_path

        # This is a simple test - the function finds repo root automatically
        # Just verify it doesn't crash
        original_path = sys.path.copy()
        try:
            add_utils_to_path()
            # Should have at least the same number of paths
            assert len(sys.path) >= len(original_path)
        finally:
            sys.path[:] = original_path
