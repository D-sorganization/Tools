"""Tests for Terminal/Workspace/Python-REPL tab settings descriptors + panels."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.serial

if sys.platform == "win32" and os.environ.get("PYTEST_XDIST_WORKER"):
    pytest.skip(
        "Qt runtime-tab-settings tests run serially on Windows.",
        allow_module_level=True,
    )

_SHARED = Path(__file__).resolve().parents[3] / "src" / "shared" / "python"
_TEST_PKG = Path(__file__).resolve().parent


def _fix_sidekick_import() -> None:
    shared_str = str(_SHARED)
    if shared_str in sys.path:
        sys.path.remove(shared_str)
    sys.path.insert(0, shared_str)
    top_mod = sys.modules.get("sidekick")
    if top_mod is not None and getattr(top_mod, "__file__", None) is not None:
        if str(_TEST_PKG) in str(Path(top_mod.__file__).resolve().parent):
            del sys.modules["sidekick"]


def _import(name: str) -> Any:
    _fix_sidekick_import()
    import importlib

    return importlib.import_module(f"sidekick.ui.tools_sidebar.{name}")


# ─── Fakes ───────────────────────────────────────────────────────


class _FakeAppearanceWidget:
    def __init__(self) -> None:
        self.applied: list[Any] = []

    def apply_appearance(self, appearance: Any) -> None:
        self.applied.append(appearance)


class _FakeResult:
    def __init__(self, warnings: tuple[Any, ...] = ()) -> None:
        self.warnings = warnings


class _FakeReplWidget(_FakeAppearanceWidget):
    def __init__(self, result: _FakeResult | None = None) -> None:
        super().__init__()
        self.configs: list[Any] = []
        self._result = result or _FakeResult()

    def apply_startup_config(self, config: Any) -> _FakeResult:
        self.configs.append(config)
        return self._result


class _FakeSidebar:
    def __init__(
        self, values: dict[str, Any] | None = None, widget: Any = None
    ) -> None:
        self._values = dict(values or {})
        self.updated: list[tuple[str, dict[str, Any]]] = []
        self._widget = widget

    def tab_settings(self, tab_id: str) -> dict[str, Any]:
        return {"schema_version": 1, "values": dict(self._values)}

    def update_tab_settings(
        self, tab_id: str, values: dict[str, Any]
    ) -> dict[str, Any]:
        self.updated.append((tab_id, dict(values)))
        self._values = dict(values)
        return values

    def tab_widget(self, tab_id: str) -> Any:
        return self._widget


# ─── Pure: parse_startup_rows ────────────────────────────────────


def test_parse_startup_rows_valid() -> None:
    r = _import("runtime_tab_settings")
    config = r.parse_startup_rows([("numpy", "np", True), ("sympy", "", True)])
    aliases = {imp.alias for imp in config.imports}
    assert aliases == {"np", "sympy"}  # blank alias defaults to module tail


def test_parse_startup_rows_skips_blank() -> None:
    r = _import("runtime_tab_settings")
    config = r.parse_startup_rows([("", "", True), ("numpy", "np", True)])
    assert len(config.imports) == 1


def test_parse_startup_rows_dotted_module_alias_default() -> None:
    r = _import("runtime_tab_settings")
    config = r.parse_startup_rows([("matplotlib.pyplot", "", True)])
    assert config.imports[0].alias == "pyplot"


def test_parse_startup_rows_invalid_module_raises() -> None:
    r = _import("runtime_tab_settings")
    with pytest.raises(ValueError):
        r.parse_startup_rows([("not a module!", "bad", True)])


def test_parse_startup_rows_duplicate_alias_raises() -> None:
    r = _import("runtime_tab_settings")
    with pytest.raises(ValueError):
        r.parse_startup_rows([("numpy", "x", True), ("scipy", "x", True)])


# ─── Pure: apply_appearance_to_tab ───────────────────────────────


def test_apply_appearance_to_tab_no_accessor() -> None:
    r = _import("runtime_tab_settings")
    ap = _import("appearance")
    assert (
        r.apply_appearance_to_tab(
            object(), "terminal", ap.DEFAULT_DARK_PANEL_APPEARANCE
        )
        is False
    )


def test_apply_appearance_to_tab_widget_without_method() -> None:
    r = _import("runtime_tab_settings")
    ap = _import("appearance")
    sidebar = _FakeSidebar(widget=object())
    assert (
        r.apply_appearance_to_tab(sidebar, "terminal", ap.DEFAULT_DARK_PANEL_APPEARANCE)
        is False
    )


def test_apply_appearance_to_tab_success() -> None:
    r = _import("runtime_tab_settings")
    ap = _import("appearance")
    widget = _FakeAppearanceWidget()
    sidebar = _FakeSidebar(widget=widget)
    ok = r.apply_appearance_to_tab(
        sidebar, "terminal", ap.DEFAULT_DARK_PANEL_APPEARANCE
    )
    assert ok is True
    assert widget.applied == [ap.DEFAULT_DARK_PANEL_APPEARANCE]


# ─── Descriptor round trip ───────────────────────────────────────


def test_repl_descriptor_round_trips_startup_imports() -> None:
    r = _import("runtime_tab_settings")
    settings_mod = _import("settings")

    class _Def:
        tab_id = r.PYTHON_REPL_TAB_ID
        settings = r.PYTHON_REPL_TAB_SETTINGS

    class _State:
        tab_settings: dict[str, Any] = {}

    store = settings_mod.SidebarTabSettingsStore([_Def()], _State())
    materialized = store.settings_for(r.PYTHON_REPL_TAB_ID)["values"]
    assert "startup_imports" in materialized
    assert len(materialized["startup_imports"]) == 5  # full bundle default


# ─── Qt: AppearanceSettingsPanel ─────────────────────────────────


def test_appearance_panel_loads_and_collects(qapp: Any) -> None:
    r = _import("runtime_tab_settings")
    sidebar = _FakeSidebar(values={"background": "#123456", "border_width": 5})
    panel = r.build_terminal_settings_panel(sidebar, "terminal")
    collected = panel.collect()
    assert collected.background == "#123456"
    assert collected.border_width == 5


def test_appearance_panel_save_persists_and_applies(qapp: Any) -> None:
    r = _import("runtime_tab_settings")
    widget = _FakeAppearanceWidget()
    sidebar = _FakeSidebar(values={"background": "#222222"}, widget=widget)
    panel = r.build_terminal_settings_panel(sidebar, "terminal")
    panel._on_save()
    assert sidebar.updated, "appearance not persisted"
    tab_id, values = sidebar.updated[-1]
    assert tab_id == "terminal"
    assert values["background"] == "#222222"
    assert len(widget.applied) == 1


def test_appearance_panel_reset(qapp: Any) -> None:
    r = _import("runtime_tab_settings")
    ap = _import("appearance")
    sidebar = _FakeSidebar(values={"background": "#010101"})
    panel = r.build_workspace_settings_panel(sidebar, "workspace")
    panel._on_reset()
    assert panel.collect().background == ap.DEFAULT_LIGHT_PANEL_APPEARANCE.background


# ─── Qt: PythonReplSettingsPanel ─────────────────────────────────


def test_repl_panel_loads_default_packages(qapp: Any) -> None:
    r = _import("runtime_tab_settings")
    panel = r.build_python_repl_settings_panel(_FakeSidebar(), "python_repl")
    rows = panel.package_rows()
    modules = {module for module, _alias, _enabled in rows}
    assert {"numpy", "scipy", "pandas", "matplotlib.pyplot", "sympy"} <= modules


def test_repl_panel_save_persists_packages_and_applies(qapp: Any) -> None:
    r = _import("runtime_tab_settings")
    widget = _FakeReplWidget()
    sidebar = _FakeSidebar(widget=widget)
    panel = r.build_python_repl_settings_panel(sidebar, "python_repl")
    panel._on_save()
    assert sidebar.updated
    _tab, values = sidebar.updated[-1]
    assert "startup_imports" in values
    assert len(values["startup_imports"]) == 5
    assert widget.configs, "startup config not applied to live REPL"
    assert widget.applied, "appearance not applied to live REPL"


def test_repl_panel_invalid_package_blocks_save(qapp: Any) -> None:
    r = _import("runtime_tab_settings")
    widget = _FakeReplWidget()
    sidebar = _FakeSidebar(widget=widget)
    panel = r.build_python_repl_settings_panel(sidebar, "python_repl")
    panel._add_package_row("not a module!", "bad", enabled=True)
    panel._on_save()
    assert not sidebar.updated  # invalid row prevents persistence
    assert "invalid" in panel._status.text().lower()


def test_repl_panel_add_and_remove_rows(qapp: Any) -> None:
    r = _import("runtime_tab_settings")
    panel = r.build_python_repl_settings_panel(_FakeSidebar(), "python_repl")
    before = len(panel.package_rows())
    panel._add_package_row("networkx", "nx", enabled=True)
    assert len(panel.package_rows()) == before + 1
    panel._packages_table.setCurrentCell(panel._packages_table.rowCount() - 1, 0)
    panel._remove_selected_row()
    assert len(panel.package_rows()) == before


# ─── Live sidebar integration ────────────────────────────────────


def _make_sidebar(tmp_path: Path, qtbot: Any) -> Any:
    _fix_sidekick_import()
    from PyQt6 import QtWidgets
    from sidekick.ui.tools_sidebar.sidebar import UnifiedToolsSidebar

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = QtWidgets.QMainWindow()
    qtbot.addWidget(win)
    return UnifiedToolsSidebar(project_root=tmp_path, parent=win)


@pytest.mark.parametrize("tab_id", ["terminal", "python_repl", "workspace"])
def test_runtime_tabs_declare_settings(tab_id: str, tmp_path: Path, qtbot: Any) -> None:
    sidebar = _make_sidebar(tmp_path, qtbot)
    definition = sidebar.get_tab_definition(tab_id)
    assert definition is not None
    assert definition.settings is not None


def test_tab_widget_accessor(tmp_path: Path, qtbot: Any) -> None:
    sidebar = _make_sidebar(tmp_path, qtbot)
    assert sidebar.tab_widget("workspace") is sidebar._tab_widgets.get("workspace")


def test_workspace_shows_empty_state(tmp_path: Path, qtbot: Any) -> None:
    sidebar = _make_sidebar(tmp_path, qtbot)
    workspace = sidebar.tab_widget("workspace")
    # No variables registered yet -> empty-state label shown, table hidden.
    # Use isHidden() (explicit flag) since isVisible() is False while the
    # parent window is unshown in the headless test.
    assert workspace._empty_label.isHidden() is False
    assert workspace._table.isHidden() is True
