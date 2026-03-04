"""Tests for ui_utils."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.tools.ui_utils import find_icon, set_qt_icon, set_tk_icon


@patch("src.tools.ui_utils.get_repo_root")
@patch("pathlib.Path.exists")
def test_find_icon_success(mock_exists, mock_root, tmp_path):
    mock_root.return_value = tmp_path
    mock_exists.return_value = True

    icon_path = find_icon("test.ico")
    assert icon_path is not None
    assert str(icon_path) == str(tmp_path / "test.ico")


@patch("src.tools.ui_utils.get_repo_root")
@patch("pathlib.Path.exists")
def test_find_icon_failure(mock_exists, mock_root, tmp_path):
    mock_root.return_value = tmp_path
    mock_exists.return_value = False

    icon_path = find_icon("test.ico")
    assert icon_path is None


@patch("src.tools.ui_utils.find_icon")
def test_set_tk_icon_success(mock_find):
    mock_find.return_value = Path("fake.ico")
    mock_root = MagicMock()
    success = set_tk_icon(mock_root, "fake.ico")
    assert success is True
    mock_root.iconbitmap.assert_called_once_with("fake.ico")


@patch("src.tools.ui_utils.find_icon")
def test_set_tk_icon_no_icon(mock_find):
    mock_find.return_value = None
    mock_root = MagicMock()
    success = set_tk_icon(mock_root, "fake.ico")
    assert success is False
    mock_root.iconbitmap.assert_not_called()


@patch("builtins.__import__")
@patch("src.tools.ui_utils.find_icon")
def test_set_qt_icon_import_error(mock_find, mock_import):
    mock_find.return_value = Path("fake.ico")

    # Simulate an import error when trying to import PyQt6
    def fake_import(name, *args, **kwargs):
        if name == "PyQt6.QtGui":
            raise ImportError("PyQt6 not installed")
        return MagicMock()

    mock_import.side_effect = fake_import
    mock_window = MagicMock()
    success = set_qt_icon(mock_window, "fake.ico")
    assert success is False
