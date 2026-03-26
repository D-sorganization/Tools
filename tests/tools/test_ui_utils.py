"""Tests for ui_utils — full coverage including DbC and all branches."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from contracts import PreconditionError

from tools.ui_utils import find_icon, set_qt_icon, set_tk_icon

# ─── find_icon ─────────────────────────────────────────────────


@patch("tools.ui_utils.get_repo_root")
@patch("pathlib.Path.exists")
def test_find_icon_success(mock_exists, mock_root, tmp_path):
    mock_root.return_value = tmp_path
    mock_exists.return_value = True

    icon_path = find_icon("test.ico")
    assert icon_path is not None
    assert str(icon_path) == str(tmp_path / "test.ico")


@patch("tools.ui_utils.get_repo_root")
@patch("pathlib.Path.exists")
def test_find_icon_failure(mock_exists, mock_root, tmp_path):
    mock_root.return_value = tmp_path
    mock_exists.return_value = False

    icon_path = find_icon("test.ico")
    assert icon_path is None


def test_find_icon_real_missing():
    """Without mock: should return None for non-existent icon."""
    icon_path = find_icon("__definitely_not_here_xyz.ico")
    assert icon_path is None


def test_find_icon_dbc_empty_name():
    with pytest.raises(PreconditionError):
        find_icon("")


def test_find_icon_dbc_non_string():
    with pytest.raises(PreconditionError):
        find_icon(None)  # type: ignore[arg-type]


# ─── set_tk_icon ───────────────────────────────────────────────


@patch("tools.ui_utils.find_icon")
def test_set_tk_icon_success(mock_find):
    mock_find.return_value = Path("fake.ico")
    mock_root = MagicMock()
    success = set_tk_icon(mock_root, "fake.ico")
    assert success is True
    mock_root.iconbitmap.assert_called_once_with("fake.ico")


@patch("tools.ui_utils.find_icon")
def test_set_tk_icon_no_icon(mock_find):
    mock_find.return_value = None
    mock_root = MagicMock()
    success = set_tk_icon(mock_root, "fake.ico")
    assert success is False
    mock_root.iconbitmap.assert_not_called()


@patch("tools.ui_utils.find_icon")
def test_set_tk_icon_os_error(mock_find):
    """iconbitmap raising OSError returns False gracefully."""
    mock_find.return_value = Path("fake.ico")
    mock_root = MagicMock()
    mock_root.iconbitmap.side_effect = OSError("cannot set icon")
    success = set_tk_icon(mock_root, "fake.ico")
    assert success is False


@patch("tools.ui_utils.find_icon")
def test_set_tk_icon_runtime_error(mock_find):
    """iconbitmap raising RuntimeError returns False gracefully."""
    mock_find.return_value = Path("fake.ico")
    mock_root = MagicMock()
    mock_root.iconbitmap.side_effect = RuntimeError("bad state")
    success = set_tk_icon(mock_root, "fake.ico")
    assert success is False


# ─── set_qt_icon ───────────────────────────────────────────────


@patch("tools.ui_utils.find_icon")
def test_set_qt_icon_no_icon(mock_find):
    mock_find.return_value = None
    success = set_qt_icon(MagicMock(), "fake.ico")
    assert success is False


@patch("builtins.__import__")
@patch("tools.ui_utils.find_icon")
def test_set_qt_icon_import_error(mock_find, mock_import):
    mock_find.return_value = Path("fake.ico")

    def fake_import(name, *args, **kwargs):
        if name == "PyQt6.QtGui":
            raise ImportError("PyQt6 not installed")
        return MagicMock()

    mock_import.side_effect = fake_import
    mock_window = MagicMock()
    success = set_qt_icon(mock_window, "fake.ico")
    assert success is False


@patch("tools.ui_utils.find_icon")
def test_set_qt_icon_success_with_mock(mock_find):
    """If PyQt6 is available, icon should be set successfully."""
    mock_find.return_value = Path("fake.ico")
    mock_window = MagicMock()

    mock_qicon = MagicMock()
    with patch.dict(
        "sys.modules",
        {"PyQt6": MagicMock(), "PyQt6.QtGui": MagicMock(QIcon=mock_qicon)},
    ):
        success = set_qt_icon(mock_window, "fake.ico")

    assert success is True
    mock_window.setWindowIcon.assert_called_once()


def test_get_repo_root_fallback():
    import builtins
    import sys

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("upstream_drift_tools.utils.paths", "tools.launch_utils"):
            raise ImportError(f"Mock missing {name}")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        if "tools.ui_utils" in sys.modules:
            del sys.modules["tools.ui_utils"]

        import tools.ui_utils

        assert callable(tools.ui_utils.get_repo_root)
        root = tools.ui_utils.get_repo_root()
        assert isinstance(root, Path)
