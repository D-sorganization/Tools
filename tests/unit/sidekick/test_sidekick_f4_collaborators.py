"""Unit tests for the F4 sidebar-decomposition collaborators.

Tests cover:
- TabCollection: id/widget bookkeeping, add/remove/replace/sync
- DockChromeController: collapse/expand toggle, dock-area set
- VisibilityPersistence: scoped key, save/load round-trip, project isolation
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# TabCollection
# ---------------------------------------------------------------------------


class TestTabCollection:
    """TabCollection manages the id↔widget↔order mapping (F4)."""

    def _make_collection(self, qtbot: Any) -> Any:
        try:
            from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets
            from upstream_drift_tools.ui.tools_sidebar.tab_collection import (
                TabCollection,
            )
        except ImportError:
            pytest.skip("Qt/sidekick unavailable")

        _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        tabs = QtWidgets.QTabWidget()
        qtbot.addWidget(tabs)
        return TabCollection(tabs)

    def test_raises_on_none_qt_tabs(self) -> None:
        """TabCollection must raise TypeError when constructed with None."""
        try:
            from upstream_drift_tools.ui.tools_sidebar.tab_collection import (
                TabCollection,
            )
        except ImportError:
            pytest.skip("sidekick unavailable")

        with pytest.raises(TypeError, match="qt_tabs must not be None"):
            TabCollection(None)  # type: ignore[arg-type]

    def test_add_and_visible_ids(self, qtbot: Any) -> None:
        """add() must append to visible_ids() in order."""
        col = self._make_collection(qtbot)
        try:
            from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets
        except ImportError:
            pytest.skip("Qt unavailable")

        w1 = QtWidgets.QWidget()
        w2 = QtWidgets.QWidget()
        col.add("a", "Tab A", w1)
        col.add("b", "Tab B", w2)

        assert col.visible_ids() == [
            "a",
            "b",
        ], "visible_ids() order must match add() order"

    def test_set_definitions_mutates_in_place(self, qtbot: Any) -> None:
        """set_definitions() must mutate the backing dict in place (#3138).

        UnifiedToolsSidebar aliases ``_tab_definitions`` to the collection's
        private dict.  If set_definitions() *reassigned* the dict, the sidebar
        alias would observe stale (empty) state, breaking settings/pop-out.
        """
        col = self._make_collection(qtbot)
        try:
            from upstream_drift_tools.ui.tools_sidebar.tab_definition import (
                SidebarTabDefinition,
            )
        except ImportError:
            pytest.skip("Qt unavailable")

        alias = col._tab_definitions  # noqa: SLF001 - simulate sidebar alias
        col.set_definitions(
            [SidebarTabDefinition(tab_id="chat", title="Chat", factory=lambda *_: None)]
        )

        assert alias is col._tab_definitions, (  # noqa: SLF001
            "set_definitions() must not rebind the backing dict"
        )
        assert "chat" in alias, "alias must observe the new definition in place"
        assert col.definition_for("chat") is not None

    def test_sync_order_mutates_ids_in_place(self, qtbot: Any) -> None:
        """sync_order_from_widget() must mutate _tab_ids in place (#3138)."""
        col = self._make_collection(qtbot)
        try:
            from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets
        except ImportError:
            pytest.skip("Qt unavailable")

        col.add("a", "A", QtWidgets.QWidget())
        col.add("b", "B", QtWidgets.QWidget())
        alias = col._tab_ids  # noqa: SLF001 - simulate sidebar alias

        col.sync_order_from_widget()

        assert alias is col._tab_ids, (  # noqa: SLF001
            "sync_order_from_widget() must not rebind the backing list"
        )
        assert alias == ["a", "b"], "alias must observe current visual order"

    def test_add_duplicate_raises(self, qtbot: Any) -> None:
        """add() must raise ValueError for a duplicate tab_id."""
        col = self._make_collection(qtbot)
        try:
            from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets
        except ImportError:
            pytest.skip("Qt unavailable")

        col.add("x", "X", QtWidgets.QWidget())
        with pytest.raises(ValueError, match="Duplicate"):
            col.add("x", "X2", QtWidgets.QWidget())

    def test_remove_shrinks_ids(self, qtbot: Any) -> None:
        """remove() must delete the id from visible_ids()."""
        col = self._make_collection(qtbot)
        try:
            from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets
        except ImportError:
            pytest.skip("Qt unavailable")

        col.add("a", "A", QtWidgets.QWidget())
        col.add("b", "B", QtWidgets.QWidget())
        result = col.remove("a")

        assert result is True, "remove() must return True on success"
        assert "a" not in col.visible_ids(), "removed id must not be in visible_ids()"
        assert "b" in col.visible_ids(), "remaining id must still be visible"

    def test_replace_swaps_widget(self, qtbot: Any) -> None:
        """replace() must swap the widget reference and keep id stable."""
        col = self._make_collection(qtbot)
        try:
            from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets
        except ImportError:
            pytest.skip("Qt unavailable")

        old_w = QtWidgets.QWidget()
        new_w = QtWidgets.QWidget()
        col.add("chat", "Chat", old_w)

        result = col.replace(old_w, new_w)

        assert result is True, "replace() must return True"
        assert col.widget_for("chat") is new_w, (
            "widget_for() must return the new widget after replace()"
        )
        assert "chat" in col.visible_ids(), "id must still be in visible_ids()"

    def test_clear_resets_state(self, qtbot: Any) -> None:
        """clear() must empty both visible_ids and the widget map."""
        col = self._make_collection(qtbot)
        try:
            from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets
        except ImportError:
            pytest.skip("Qt unavailable")

        col.add("t", "T", QtWidgets.QWidget())
        col.clear()

        assert col.visible_ids() == [], "visible_ids() must be empty after clear()"
        assert col.widget_for("t") is None, (
            "widget_for() must return None after clear()"
        )

    def test_contains_and_index_of(self, qtbot: Any) -> None:
        """contains() and index_of() must reflect actual id list."""
        col = self._make_collection(qtbot)
        try:
            from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets
        except ImportError:
            pytest.skip("Qt unavailable")

        col.add("first", "First", QtWidgets.QWidget())
        col.add("second", "Second", QtWidgets.QWidget())

        assert col.contains("first")
        assert not col.contains("missing")
        assert col.index_of("first") == 0
        assert col.index_of("second") == 1
        assert col.index_of("nope") == -1


# ---------------------------------------------------------------------------
# DockChromeController
# ---------------------------------------------------------------------------


class TestDockChromeController:
    """DockChromeController manages collapse/expand and dock setup (F4)."""

    def _make_controller(self, qtbot: Any) -> Any:
        try:
            from upstream_drift_tools.ui.tools_sidebar.dock_chrome import (
                DockChromeController,
            )
            from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets
            from upstream_drift_tools.ui.tools_sidebar.state import SidebarState
        except ImportError:
            pytest.skip("Qt/sidekick unavailable")

        _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        sidebar_w = QtWidgets.QWidget()
        tabs_w = QtWidgets.QTabWidget()
        qtbot.addWidget(sidebar_w)
        qtbot.addWidget(tabs_w)
        state = SidebarState(width=400)
        return DockChromeController(
            sidebar_widget=sidebar_w,
            tabs_widget=tabs_w,
            initial_state=state,
        )

    def test_raises_on_none_sidebar(self, qtbot: Any) -> None:
        """DockChromeController must raise TypeError when sidebar_widget is None."""
        try:
            from upstream_drift_tools.ui.tools_sidebar.dock_chrome import (
                DockChromeController,
            )
            from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets
            from upstream_drift_tools.ui.tools_sidebar.state import SidebarState
        except ImportError:
            pytest.skip("sidekick unavailable")

        _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        tabs_w = QtWidgets.QTabWidget()
        qtbot.addWidget(tabs_w)
        with pytest.raises(TypeError, match="sidebar_widget"):
            DockChromeController(
                sidebar_widget=None,  # type: ignore[arg-type]
                tabs_widget=tabs_w,
                initial_state=SidebarState(),
            )

    def test_toggle_collapsed_hides_tabs(self, qtbot: Any) -> None:
        """toggle_collapsed() must hide the tabs widget when collapsing."""
        ctrl = self._make_controller(qtbot)

        assert not ctrl.is_collapsed, "starts expanded"
        ctrl.toggle_collapsed()
        assert ctrl.is_collapsed, "must be collapsed after toggle"
        assert ctrl._tabs.isVisible() is False, "tabs must be hidden when collapsed"  # noqa: SLF001

    def test_toggle_collapsed_shows_tabs_on_expand(self, qtbot: Any) -> None:
        """A second toggle_collapsed() must restore the tabs to visible."""
        ctrl = self._make_controller(qtbot)
        ctrl.toggle_collapsed()  # collapse
        ctrl.toggle_collapsed()  # expand

        assert not ctrl.is_collapsed, "must be expanded after double toggle"
        assert ctrl._tabs.isVisible() is True, "tabs must be visible after expanding"  # noqa: SLF001

    def test_dock_widget_is_none_before_install(self, qtbot: Any) -> None:
        """dock_widget must be None before install_as_dock() is called."""
        ctrl = self._make_controller(qtbot)
        assert ctrl.dock_widget is None, "dock_widget must start as None"

    def test_set_dock_area_rejects_unknown(self, qtbot: Any) -> None:
        """set_dock_area() must return False for unknown area strings."""
        ctrl = self._make_controller(qtbot)
        try:
            from upstream_drift_tools.ui.tools_sidebar.state import SidebarState
        except ImportError:
            pytest.skip("sidekick unavailable")

        result = ctrl.set_dock_area("middle", SidebarState())
        assert result is False, "unknown area must return False"

    def test_set_dock_area_accepts_left_right(self, qtbot: Any) -> None:
        """set_dock_area() must return True for 'left' and 'right'."""
        ctrl = self._make_controller(qtbot)
        try:
            from upstream_drift_tools.ui.tools_sidebar.state import SidebarState
        except ImportError:
            pytest.skip("sidekick unavailable")

        state = SidebarState()
        assert ctrl.set_dock_area("left", state) is True
        assert state.dock_area == "left"
        assert ctrl.set_dock_area("right", state) is True
        assert state.dock_area == "right"


# ---------------------------------------------------------------------------
# VisibilityPersistence
# ---------------------------------------------------------------------------


class TestVisibilityPersistence:
    """VisibilityPersistence reads/writes visible tabs to QSettings (F4/F5)."""

    def _make_vp(self, tmp_path: Any, project_root: Any = None) -> Any:
        try:
            from upstream_drift_tools.ui.tools_sidebar.visibility_persistence import (
                VisibilityPersistence,
            )
        except ImportError:
            pytest.skip("sidekick unavailable")

        root = project_root or tmp_path / "proj"
        return VisibilityPersistence(project_root=root)

    def test_raises_on_bad_type(self) -> None:
        """VisibilityPersistence must raise TypeError for invalid project_root types."""
        try:
            from upstream_drift_tools.ui.tools_sidebar.visibility_persistence import (
                VisibilityPersistence,
            )
        except ImportError:
            pytest.skip("sidekick unavailable")

        with pytest.raises(TypeError, match="project_root"):
            VisibilityPersistence(project_root=42)  # type: ignore[arg-type]

    def test_load_returns_none_before_save(self, tmp_path: Any) -> None:
        """load() must return None when nothing has been saved yet."""
        vp = self._make_vp(tmp_path)
        patch_path = (
            "upstream_drift_tools.ui.tools_sidebar.visibility_persistence.QtCore"
        )
        with patch(patch_path) as mock_qt:
            mock_settings = MagicMock()
            mock_settings.value.return_value = None
            mock_qt.QSettings.return_value = mock_settings

            result = vp.load(known_ids={"a", "b"})

        assert result is None, "load() must return None when QSettings has no value"

    def test_save_and_load_round_trip(self, tmp_path: Any) -> None:
        """save() + load() must return the same id list (filtered to known)."""
        vp = self._make_vp(tmp_path)
        saved_data: list[str] = []

        with patch(
            "upstream_drift_tools.ui.tools_sidebar.visibility_persistence.QtCore"
        ) as mock_qt:
            mock_settings = MagicMock()

            def fake_set_value(key: str, val: object) -> None:
                saved_data.extend(val)  # type: ignore[arg-type]

            mock_settings.setValue.side_effect = fake_set_value
            mock_settings.value.side_effect = lambda k, default: (
                saved_data if saved_data else default
            )
            mock_qt.QSettings.return_value = mock_settings

            vp.save(["chat", "files", "terminal"])
            result = vp.load(known_ids={"chat", "files", "terminal", "workspace"})

        assert result is not None
        assert set(result) == {
            "chat",
            "files",
            "terminal",
        }, "load() must return exactly what was saved (filtered to known_ids)"

    def test_two_projects_use_different_keys(self, tmp_path: Any) -> None:
        """Two VPs for different roots must use different keys."""
        try:
            from upstream_drift_tools.ui.tools_sidebar.visibility_persistence import (
                VisibilityPersistence,
            )
        except ImportError:
            pytest.skip("sidekick unavailable")

        root_a = tmp_path / "proj_a"
        root_b = tmp_path / "proj_b"
        vp_a = VisibilityPersistence(project_root=root_a)
        vp_b = VisibilityPersistence(project_root=root_b)

        # Access the private key to assert isolation (white-box)
        assert vp_a._key != vp_b._key, (  # noqa: SLF001
            "Different roots must produce different QSettings keys (F5 isolation)"
        )
