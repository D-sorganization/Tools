"""Tests for dependency_utils."""

from unittest.mock import MagicMock, patch

from src.tools.dependency_utils import check_dependencies, install_packages


@patch("builtins.__import__")
def test_check_dependencies_all_present(mock_import):
    missing = check_dependencies(["sys", "os"])
    assert missing == []
    assert mock_import.call_count == 2


@patch("builtins.__import__", side_effect=ImportError("No module named 'fake_module'"))
def test_check_dependencies_missing(mock_import):
    missing = check_dependencies(["fake_module"])
    assert missing == ["fake_module"]
    assert mock_import.call_count == 1


@patch("subprocess.run")
def test_install_packages_success(mock_run):
    mock_run.return_value = MagicMock(returncode=0)
    success = install_packages(["pandas", "numpy"])
    assert success is True
    assert mock_run.call_count == 2
    # Verify mapping is used (pandas mapped to pandas)
    args, kwargs = mock_run.call_args_list[0]
    assert "pandas" in args[0]


@patch("subprocess.run")
def test_install_packages_failure(mock_run):
    mock_run.return_value = MagicMock(returncode=1, stderr="Failed")
    success = install_packages(["customtkinter"])
    assert success is False
    assert mock_run.call_count == 1


def test_install_packages_empty():
    success = install_packages([])
    assert success is True
