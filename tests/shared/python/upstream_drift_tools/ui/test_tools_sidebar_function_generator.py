"""Tests for the optional Function Generator Sidekick tab."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _qt_widgets() -> object:
    try:
        from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets
    except ImportError:
        pytest.skip("Qt widgets unavailable")
    return QtWidgets


def test_function_generator_help_metadata_imports_without_ui_dependencies() -> None:
    env = os.environ.copy()
    pythonpath = [
        str(Path("src").resolve()),
        str(Path("src/shared/python").resolve()),
    ]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    script = """
import sys
from upstream_drift_tools.ui.tools_sidebar import DEFAULT_SIDEBAR_TAB_HELP

metadata = DEFAULT_SIDEBAR_TAB_HELP["function_generator"]
assert metadata["title"] == "Function Generator"
assert "waveform" in metadata["summary"]
loaded = [
    name for name in sys.modules
    if name.startswith("function_generator.python")
    or name.startswith("matplotlib")
    or name.startswith("numpy")
    or name.startswith("signal_toolkit")
]
assert loaded == [], loaded
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_sidekick_function_generator_default_tab_contract(tmp_path: Path) -> None:
    from upstream_drift_tools.ui.tools_sidebar import UnifiedToolsSidebar

    QtWidgets = _qt_widgets()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app

    sidebar = UnifiedToolsSidebar(project_root=tmp_path)
    definition = sidebar._tab_definitions["function_generator"]

    assert "function_generator" in sidebar.available_tab_ids()
    assert "function_generator" in sidebar.hidden_tab_ids()
    assert "function_generator" not in sidebar.visible_tab_ids()
    assert definition.title == "Function Generator"
    assert definition.visible is False
    assert definition.duplicate_enabled is True
    assert definition.help_metadata["title"] == "Function Generator"


def test_sidekick_function_generator_visibility_persists(tmp_path: Path) -> None:
    from upstream_drift_tools.ui.tools_sidebar import SidebarState, UnifiedToolsSidebar

    QtWidgets = _qt_widgets()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app

    sidebar = UnifiedToolsSidebar(project_root=tmp_path)

    assert sidebar.set_tab_visible("function_generator", True) is True
    assert "function_generator" in sidebar.visible_tab_ids()
    assert sidebar.set_tab_visible("function_generator", False) is True
    assert "function_generator" in sidebar.hidden_tab_ids()

    state_path = tmp_path / "sidekick-state.json"
    assert sidebar.set_tab_visible("function_generator", True) is True
    sidebar.save_state(state_path)
    restored = UnifiedToolsSidebar(
        project_root=tmp_path,
        state=SidebarState.load_json(state_path),
    )

    assert "function_generator" in restored.visible_tab_ids()


def test_sidekick_function_generator_unavailable_placeholder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from upstream_drift_tools.ui.tools_sidebar import (
        SIDEKICK_PLACEHOLDER_OBJECT_NAME,
        UnifiedToolsSidebar,
        default_tabs,
    )

    QtWidgets = _qt_widgets()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app

    def fail_function_generator_import(name: str) -> object:
        if name == "function_generator.gui_registration":
            raise ImportError("missing optional function generator UI")
        return original_import_module(name)

    original_import_module = default_tabs.importlib.import_module
    monkeypatch.setattr(
        default_tabs.importlib,
        "import_module",
        fail_function_generator_import,
    )

    sidebar = UnifiedToolsSidebar(project_root=tmp_path)

    assert sidebar.set_tab_visible("function_generator", True) is True
    assert sidebar.set_active_tab("function_generator") is True
    tab = sidebar.tabs.currentWidget()

    assert tab is not None
    assert tab.objectName() == SIDEKICK_PLACEHOLDER_OBJECT_NAME
    assert "Function Generator" in tab.findChild(QtWidgets.QLabel).text()


def test_sidekick_function_generator_uses_registered_widget_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from upstream_drift_tools.ui.tools_sidebar import (
        SIDEKICK_FUNCTION_GENERATOR_OBJECT_NAME,
        UnifiedToolsSidebar,
        default_tabs,
    )

    QtWidgets = _qt_widgets()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app

    class FakeFunctionGeneratorWidget(QtWidgets.QWidget):
        def __init__(
            self,
            parent: object | None = None,
            *,
            use_builtin_theme: bool = True,
        ) -> None:
            super().__init__(parent)
            self.use_builtin_theme = use_builtin_theme

    fake_registration = SimpleNamespace(
        get_gui_info=lambda: {
            "pyqt6": {
                "module": "fake_function_generator_ui",
                "class": "FakeFunctionGeneratorWidget",
            }
        }
    )
    fake_widget_module = SimpleNamespace(
        FakeFunctionGeneratorWidget=FakeFunctionGeneratorWidget
    )

    def import_fake(name: str) -> object:
        if name == "function_generator.gui_registration":
            return fake_registration
        if name == "fake_function_generator_ui":
            return fake_widget_module
        return original_import_module(name)

    original_import_module = default_tabs.importlib.import_module
    monkeypatch.setattr(default_tabs.importlib, "import_module", import_fake)

    sidebar = UnifiedToolsSidebar(project_root=tmp_path)

    assert sidebar.set_tab_visible("function_generator", True) is True
    assert sidebar.set_active_tab("function_generator") is True
    tab = sidebar.tabs.currentWidget()

    assert isinstance(tab, FakeFunctionGeneratorWidget)
    assert tab.objectName() == SIDEKICK_FUNCTION_GENERATOR_OBJECT_NAME
    assert tab.use_builtin_theme is False
