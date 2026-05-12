"""Tests for LoD fixes in build_exe.py.

Verifies that the LoD violation fix (importlib.util.find_spec -> find_spec direct import)
works correctly and does not break existing behavior.
"""

import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


@pytest.fixture()
def build_exe_module():
    """Import build_exe module from project_packer."""
    import importlib
    import sys

    # Add project_packer to path temporarily
    packer_path = str(Path(__file__).parent.parent.parent / "src" / "project_packer")
    sys.path.insert(0, packer_path)
    try:
        if "build_exe" in sys.modules:
            del sys.modules["build_exe"]
        mod = importlib.import_module("build_exe")
        yield mod
    finally:
        sys.path.remove(packer_path)
        if "build_exe" in sys.modules:
            del sys.modules["build_exe"]


class TestBuildExeLoDFix:
    """Tests verifying the LoD fix: importlib.util.find_spec replaced with find_spec."""

    def test_check_pyinstaller_uses_find_spec_directly(self, build_exe_module) -> None:
        """Verify check_pyinstaller uses find_spec directly (not importlib.util.find_spec)."""
        import inspect

        source = inspect.getsource(build_exe_module.check_pyinstaller)
        assert "importlib.util.find_spec" not in source, (
            "LoD violation: should use find_spec directly, not importlib.util.find_spec"
        )
        assert "find_spec" in source, "check_pyinstaller should call find_spec"

    def test_check_pyinstaller_available(self, build_exe_module) -> None:
        """Test PyInstaller availability check when available."""
        with patch("build_exe.find_spec") as mock_find_spec:
            mock_find_spec.return_value = Mock()
            assert build_exe_module.check_pyinstaller() is True
            mock_find_spec.assert_called_once_with("PyInstaller")

    def test_check_pyinstaller_not_available(self, build_exe_module) -> None:
        """Test PyInstaller availability check when not available."""
        with patch("build_exe.find_spec") as mock_find_spec:
            mock_find_spec.return_value = None
            assert build_exe_module.check_pyinstaller() is False
            mock_find_spec.assert_called_once_with("PyInstaller")

    def test_find_spec_import_at_module_level(self, build_exe_module) -> None:
        """Verify find_spec is imported at module level (not accessed via importlib.util)."""
        assert hasattr(build_exe_module, "find_spec"), (
            "find_spec must be imported at module level in build_exe"
        )

    def test_install_pyinstaller_success(self, build_exe_module) -> None:
        """Test successful PyInstaller installation."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)
            assert build_exe_module.install_pyinstaller() is True

    def test_install_pyinstaller_failure(self, build_exe_module) -> None:
        """Test failed PyInstaller installation."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "pip")
            assert build_exe_module.install_pyinstaller() is False

    def test_build_exe_no_print_calls(self, build_exe_module) -> None:
        """Verify no print() calls exist in build_exe module source."""
        import inspect

        source = inspect.getsource(build_exe_module)
        lines = source.splitlines()
        print_lines = [
            f"line {i + 1}: {line}"
            for i, line in enumerate(lines)
            if "print(" in line and not line.strip().startswith("#")
        ]
        assert not print_lines, f"Found print() calls in build_exe.py: {print_lines}"
