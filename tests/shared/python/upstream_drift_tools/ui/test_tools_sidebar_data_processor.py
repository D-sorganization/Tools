from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


def _qt_widgets() -> object:
    try:
        from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets
    except ImportError:
        pytest.skip("Qt widgets unavailable")
    return QtWidgets


def test_sidekick_data_processor_visibility_persists_when_enabled(
    tmp_path: Path,
) -> None:
    from upstream_drift_tools.ui.tools_sidebar import SidebarState, UnifiedToolsSidebar

    QtWidgets = _qt_widgets()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app

    sidebar = UnifiedToolsSidebar(project_root=tmp_path)

    assert "data_processor" in sidebar.hidden_tab_ids()
    assert sidebar.set_tab_visible("data_processor", True) is True
    assert "data_processor" in sidebar.visible_tab_ids()

    state_path = tmp_path / "sidekick-state.json"
    sidebar.save_state(state_path)
    restored = UnifiedToolsSidebar(
        project_root=tmp_path,
        state=SidebarState.load_json(state_path),
    )

    assert "data_processor" in restored.visible_tab_ids()
    assert restored.set_active_tab("data_processor") is True


def test_sidekick_data_processor_unavailable_placeholder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from upstream_drift_tools.ui.tools_sidebar import (
        SIDEKICK_PLACEHOLDER_OBJECT_NAME,
        UnifiedToolsSidebar,
        data_processor_tab,
    )

    QtWidgets = _qt_widgets()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app

    def fail_import(name: str) -> object:
        if name == "upstream_drift_tools.ui.widgets.data_processor_widget":
            raise ImportError("missing optional data processor UI")
        return original_import(name)

    original_import = data_processor_tab.importlib.import_module
    monkeypatch.setattr(data_processor_tab.importlib, "import_module", fail_import)

    sidebar = UnifiedToolsSidebar(project_root=tmp_path)

    assert sidebar.set_tab_visible("data_processor", True) is True
    assert sidebar.set_active_tab("data_processor") is True
    tab = sidebar.tabs.currentWidget()

    assert tab is not None
    assert tab.objectName() == SIDEKICK_PLACEHOLDER_OBJECT_NAME
    assert "Data Processor" in tab.findChild(QtWidgets.QLabel).text()


def test_sidekick_data_processor_exports_results_to_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from upstream_drift_tools.ui.tools_sidebar import (
        UnifiedToolsSidebar,
        data_processor_tab,
    )

    QtWidgets = _qt_widgets()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app

    class FakeDataProcessorWidget(QtWidgets.QWidget):
        def __init__(self, parent: object | None = None) -> None:
            super().__init__(parent)
            self.engine = SimpleNamespace(
                data=pd.DataFrame(
                    {
                        "temperature": [293.15, 294.0],
                        "status": ["ok", "warn"],
                    }
                )
            )

    fake_module = SimpleNamespace(DataProcessorWidget=FakeDataProcessorWidget)

    def import_fake(name: str) -> object:
        if name == "upstream_drift_tools.ui.widgets.data_processor_widget":
            return fake_module
        return original_import(name)

    original_import = data_processor_tab.importlib.import_module
    monkeypatch.setattr(data_processor_tab.importlib, "import_module", import_fake)

    sidebar = UnifiedToolsSidebar(project_root=tmp_path)

    assert sidebar.set_tab_visible("data_processor", True) is True
    assert sidebar.set_active_tab("data_processor") is True
    tab = sidebar.tabs.currentWidget()

    columns = tab.findChild(QtWidgets.QLineEdit, "SidekickDataProcessorColumns")
    variable = tab.findChild(QtWidgets.QLineEdit, "SidekickDataProcessorVariable")
    export_button = tab.findChild(
        QtWidgets.QPushButton,
        "SidekickDataProcessorExportWorkspace",
    )

    assert columns is not None
    assert variable is not None
    assert export_button is not None

    columns.setText("temperature")
    variable.setText("temperature_processed")
    export_button.click()

    assert sidebar.registry.get("temperature_processed") == [293.15, 294.0]
