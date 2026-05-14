"""UI-facing Sidekick calculator startup import diagnostics."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_sidekick_calculator_reports_missing_startup_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    try:
        from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets
    except ImportError:
        pytest.skip("Qt widgets unavailable")

    from upstream_drift_tools.ui.tools_sidebar import SidebarState, UnifiedToolsSidebar
    from upstream_drift_tools.ui.tools_sidebar.calculator_startup import (
        CalculatorStartupConfig,
        CalculatorStartupImport,
    )

    def fake_import_module(module_name: str) -> object:
        raise ImportError(f"{module_name} is not installed")

    monkeypatch.setattr(
        "upstream_drift_tools.ui.tools_sidebar.calculator_startup."
        "importlib.import_module",
        fake_import_module,
    )
    config = CalculatorStartupConfig(
        (CalculatorStartupImport("missing_calculator_dep", "mcd"),)
    )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app
    sidebar = UnifiedToolsSidebar(
        project_root=tmp_path,
        state=SidebarState(calculator_startup_imports=config.to_list()),
    )

    assert sidebar.set_active_tab("calculator") is True
    calculator = sidebar.tabs.currentWidget()
    status = calculator.findChild(
        QtWidgets.QLabel,
        "SidekickCalculatorStartupStatus",
    )

    assert status is not None
    assert "Optional dependency unavailable" in status.text()
    assert "missing_calculator_dep" in status.text()
    assert calculator.loaded_startup_dependencies() == ()
