"""Tests for dependency_utils."""

import sys
from unittest.mock import MagicMock, patch

from src.tools.dependency_utils import check_dependencies, install_packages


def test_check_dependencies_all_present():
    """All packages already importable → no missing packages reported."""
    missing = check_dependencies(["sys", "os"])
    assert missing == []


def test_check_dependencies_missing():
    """Packages absent from sys.modules raise ImportError → reported as missing."""
    fake_pkg = "_totally_fake_pkg_xyz"
    sys.modules.pop(fake_pkg, None)

    missing = check_dependencies([fake_pkg])
    assert missing == [fake_pkg]


def test_check_dependencies_mixed():
    """Mix of present and absent packages returns only the missing ones."""
    fake_pkg = "_totally_fake_pkg_xyz"
    sys.modules.pop(fake_pkg, None)

    missing = check_dependencies(["sys", fake_pkg, "os"])
    assert missing == [fake_pkg]


@patch("subprocess.run")
def test_install_packages_success(mock_run):
    """All packages install successfully returns True."""
    mock_run.return_value = MagicMock(returncode=0)
    success = install_packages(["pandas", "numpy"])
    assert success is True
    assert mock_run.call_count == 2
    args, _ = mock_run.call_args_list[0]
    assert "pandas" in args[0]


@patch("subprocess.run")
def test_install_packages_failure(mock_run):
    """A failed install returns False."""
    mock_run.return_value = MagicMock(returncode=1, stderr="Failed")
    success = install_packages(["customtkinter"])
    assert success is False
    assert mock_run.call_count == 1


def test_install_packages_empty():
    """Empty list always returns True without calling pip."""
    success = install_packages([])
    assert success is True
