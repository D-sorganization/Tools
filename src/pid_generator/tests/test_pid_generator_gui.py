from typing import Any

"""Tests for P&ID Generator GUI registration and launcher plumbing."""

from __future__ import annotations  # noqa: F404

import importlib

import pytest


def test_gui_info_structure() -> Any:
    """gui_registration.GUI_INFO has all required canonical keys."""
    from pid_generator.gui_registration import GUI_INFO

    assert GUI_INFO["name"] == "P&ID Generator"
    assert GUI_INFO["tool_name"] == "pid_generator"
    assert "description" in GUI_INFO
    assert "category" in GUI_INFO
    assert "icon" in GUI_INFO

    pyqt6 = GUI_INFO["pyqt6"]
    assert pyqt6["module"] == "pid_generator.ui.pyqt6.main_window"
    assert pyqt6["class"] == "PIDGeneratorMainWindow"
    assert "dependencies" in pyqt6
    assert "settings_app" in pyqt6


def test_get_gui_info_returns_gui_info() -> Any:
    """get_gui_info() helper returns the same GUI_INFO dict."""
    from pid_generator.gui_registration import GUI_INFO, get_gui_info

    assert get_gui_info() is GUI_INFO


@pytest.mark.skipif(
    importlib.util.find_spec("PyQt6") is None or importlib.util.find_spec("ezdxf") is None,
    reason="PyQt6 and ezdxf required",
)
def test_main_window_class_importable() -> Any:
    """The PIDGeneratorMainWindow class can be imported."""
    from pid_generator.ui.pyqt6.main_window import PIDGeneratorMainWindow

    assert callable(PIDGeneratorMainWindow)


def test_pid_generator_package_version() -> Any:
    """pid_generator package exposes __version__."""
    import pid_generator

    assert hasattr(pid_generator, "__version__")
    assert isinstance(pid_generator.__version__, str)
