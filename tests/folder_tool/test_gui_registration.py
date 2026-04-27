"""Unit tests for folder_tool.gui_registration module."""

from __future__ import annotations

import pytest


@pytest.fixture
def gui_info():
    """Load GUI registration info from the module."""
    from folder_tool.gui_registration import get_gui_info

    return get_gui_info()


class TestGuiRegistration:
    """Tests for folder_tool GUI registration metadata."""

    def test_get_gui_info_returns_dict(self, gui_info):
        assert isinstance(gui_info, dict)

    def test_gui_info_has_required_keys(self, gui_info):
        required_keys = {"name", "tool_name", "description", "category", "icon"}
        assert required_keys.issubset(gui_info.keys())

    def test_tool_name_is_folder_tool(self, gui_info):
        assert gui_info["tool_name"] == "folder_tool"

    def test_name_is_nonempty_string(self, gui_info):
        assert isinstance(gui_info["name"], str)
        assert len(gui_info["name"]) > 0

    def test_description_is_nonempty_string(self, gui_info):
        assert isinstance(gui_info["description"], str)
        assert len(gui_info["description"]) > 0

    def test_category_is_nonempty_string(self, gui_info):
        assert isinstance(gui_info["category"], str)
        assert len(gui_info["category"]) > 0

    def test_icon_is_nonempty_string(self, gui_info):
        assert isinstance(gui_info["icon"], str)
        assert len(gui_info["icon"]) > 0

    def test_pyqt6_section_present(self, gui_info):
        assert "pyqt6" in gui_info
        pyqt6 = gui_info["pyqt6"]
        assert isinstance(pyqt6, dict)
        assert "module" in pyqt6
        assert "class" in pyqt6

    def test_gui_info_module_constant_matches(self):
        """Ensure GUI_INFO constant and get_gui_info() return the same object."""
        from folder_tool.gui_registration import GUI_INFO, get_gui_info

        assert get_gui_info() is GUI_INFO
