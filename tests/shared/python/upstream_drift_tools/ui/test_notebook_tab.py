"""Tests for NotebookTab — Sidekick Jupyter integration Phase 1.

Issue #2875: [Jupyter Sidekick Phase 1] Notebook UI Tab and Dependency Management.

Test order follows TDD red-green: tests were written before the implementation
so each test drove a specific design decision in notebook_tab.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_qt_widgets() -> Any:
    """Return QtWidgets or skip if Qt is not available."""
    try:
        from upstream_drift_tools.ui.tools_sidebar.qt_compat import QtWidgets
    except ImportError:
        pytest.skip("Qt widgets unavailable")
    return QtWidgets


_QT_APP_REF: list[Any] = []


def _get_app(QtWidgets: Any) -> Any:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    if not _QT_APP_REF:
        _QT_APP_REF.append(app)
    return app


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------


class TestNotebookTabRegistration:
    """NotebookTab must be registered in the default Sidekick tab definitions."""

    def test_notebook_tab_id_constant_is_defined(self) -> None:
        """NOTEBOOK_TAB_ID must be exported from the notebook_tab module."""
        from upstream_drift_tools.ui.tools_sidebar import notebook_tab

        assert hasattr(notebook_tab, "NOTEBOOK_TAB_ID")
        assert isinstance(notebook_tab.NOTEBOOK_TAB_ID, str)
        assert notebook_tab.NOTEBOOK_TAB_ID == "notebook"

    def test_build_notebook_tab_factory_is_callable(self) -> None:
        """build_notebook_tab must be a callable factory."""
        from upstream_drift_tools.ui.tools_sidebar import notebook_tab

        assert callable(notebook_tab.build_notebook_tab)

    def test_notebook_tab_appears_in_default_tab_definitions(
        self, tmp_path: Path
    ) -> None:
        """'notebook' tab_id must appear in build_default_tab_definitions output."""
        QtWidgets = _ensure_qt_widgets()
        _get_app(QtWidgets)
        from upstream_drift_tools.ui.tools_sidebar.default_tabs import (
            build_default_tab_definitions,
        )
        from upstream_drift_tools.ui.tools_sidebar.tab_definition import (
            SidebarTabDefinition,
        )

        tabs = build_default_tab_definitions(None, SidebarTabDefinition)
        tab_ids = [t.tab_id for t in tabs]
        assert "notebook" in tab_ids

    def test_notebook_tab_definition_has_correct_title(self, tmp_path: Path) -> None:
        """The registered notebook tab must have a human-readable title."""
        QtWidgets = _ensure_qt_widgets()
        _get_app(QtWidgets)
        from upstream_drift_tools.ui.tools_sidebar.default_tabs import (
            build_default_tab_definitions,
        )
        from upstream_drift_tools.ui.tools_sidebar.tab_definition import (
            SidebarTabDefinition,
        )

        tabs = build_default_tab_definitions(None, SidebarTabDefinition)
        notebook = next((t for t in tabs if t.tab_id == "notebook"), None)
        assert notebook is not None
        title_lower = notebook.title.lower()
        assert "notebook" in title_lower or "jupyter" in title_lower

    def test_notebook_tab_help_metadata_exists(self) -> None:
        """help_content must include a 'notebook' entry."""
        from upstream_drift_tools.ui.tools_sidebar.help_content import (
            DEFAULT_SIDEBAR_TAB_HELP,
        )

        assert "notebook" in DEFAULT_SIDEBAR_TAB_HELP
        assert "summary" in DEFAULT_SIDEBAR_TAB_HELP["notebook"]


# ---------------------------------------------------------------------------
# Graceful degradation tests
# ---------------------------------------------------------------------------


class TestNotebookTabGracefulDegradation:
    """When Jupyter deps are absent the tab must degrade, not crash."""

    def test_no_crash_when_jupyter_client_absent(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """build_notebook_tab must not raise ImportError when jupyter_client is None."""
        QtWidgets = _ensure_qt_widgets()
        _get_app(QtWidgets)

        # Block jupyter_client at the sys.modules level.
        monkeypatch.setitem(sys.modules, "jupyter_client", None)
        monkeypatch.setitem(sys.modules, "nbformat", None)

        from upstream_drift_tools.ui.tools_sidebar import notebook_tab

        fake_sidebar = QtWidgets.QWidget()
        fake_sidebar.project_root = tmp_path  # type: ignore[attr-defined]
        try:
            widget = notebook_tab.build_notebook_tab(fake_sidebar)
            assert widget is not None
        finally:
            fake_sidebar.deleteLater()

    def test_shows_install_message_when_jupyter_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """When jupyter_client is absent the widget must mention 'pip install'."""
        QtWidgets = _ensure_qt_widgets()
        _get_app(QtWidgets)

        monkeypatch.setitem(sys.modules, "jupyter_client", None)
        monkeypatch.setitem(sys.modules, "nbformat", None)

        # Force the module to be re-evaluated with the patched sys.modules.
        import importlib

        import upstream_drift_tools.ui.tools_sidebar.notebook_tab as nt_mod

        importlib.reload(nt_mod)

        fake_sidebar = QtWidgets.QWidget()
        fake_sidebar.project_root = tmp_path  # type: ignore[attr-defined]
        try:
            widget = nt_mod.build_notebook_tab(fake_sidebar)
            # Collect all QLabel text in the widget hierarchy.
            labels = widget.findChildren(QtWidgets.QLabel)
            combined = " ".join(lbl.text() for lbl in labels).lower()
            assert "pip install" in combined or "install jupyter" in combined
        finally:
            fake_sidebar.deleteLater()

    def test_widget_is_qwidget_when_jupyter_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Even without Jupyter the factory must return a QWidget."""
        QtWidgets = _ensure_qt_widgets()
        _get_app(QtWidgets)

        monkeypatch.setitem(sys.modules, "jupyter_client", None)

        from upstream_drift_tools.ui.tools_sidebar import notebook_tab

        fake_sidebar = QtWidgets.QWidget()
        fake_sidebar.project_root = tmp_path  # type: ignore[attr-defined]
        try:
            widget = notebook_tab.build_notebook_tab(fake_sidebar)
            assert isinstance(widget, QtWidgets.QWidget)
        finally:
            fake_sidebar.deleteLater()

    def test_no_crash_when_jupyter_installed(
        self,
        tmp_path: Path,
    ) -> None:
        """Normal (Jupyter present) path must also produce a QWidget."""
        QtWidgets = _ensure_qt_widgets()
        _get_app(QtWidgets)

        from upstream_drift_tools.ui.tools_sidebar import notebook_tab

        fake_sidebar = QtWidgets.QWidget()
        fake_sidebar.project_root = tmp_path  # type: ignore[attr-defined]
        try:
            widget = notebook_tab.build_notebook_tab(fake_sidebar)
            assert isinstance(widget, QtWidgets.QWidget)
        finally:
            fake_sidebar.deleteLater()


# ---------------------------------------------------------------------------
# Session metadata tests
# ---------------------------------------------------------------------------


class TestNotebookSessionMetadata:
    """NotebookTab must track per-instance notebook path and kernel env."""

    def _make_widget(self, tmp_path: Path) -> Any:
        QtWidgets = _ensure_qt_widgets()
        _get_app(QtWidgets)
        from upstream_drift_tools.ui.tools_sidebar.notebook_tab import (
            SidekickNotebookWidget,
        )

        return SidekickNotebookWidget(project_root=tmp_path, parent=None)

    def test_initial_session_metadata_has_none_path(self, tmp_path: Path) -> None:
        """Freshly constructed widget must have notebook_path == None."""
        widget = self._make_widget(tmp_path)
        try:
            assert widget.session_metadata["notebook_path"] is None
        finally:
            widget.deleteLater()

    def test_initial_session_metadata_has_none_kernel_env(self, tmp_path: Path) -> None:
        """Freshly constructed widget must have kernel_env == None."""
        widget = self._make_widget(tmp_path)
        try:
            assert widget.session_metadata["kernel_env"] is None
        finally:
            widget.deleteLater()

    def test_tab_tracks_notebook_path(self, tmp_path: Path) -> None:
        """open_notebook must update session_metadata['notebook_path']."""
        widget = self._make_widget(tmp_path)
        try:
            widget.open_notebook("/path/to/notebook.ipynb")
            assert widget.session_metadata["notebook_path"] == "/path/to/notebook.ipynb"
        finally:
            widget.deleteLater()

    def test_tab_tracks_kernel_environment(self, tmp_path: Path) -> None:
        """set_kernel_environment must update session_metadata['kernel_env']."""
        widget = self._make_widget(tmp_path)
        try:
            widget.set_kernel_environment("my-venv")
            assert widget.session_metadata["kernel_env"] == "my-venv"
        finally:
            widget.deleteLater()

    def test_metadata_isolated_per_tab_instance(self, tmp_path: Path) -> None:
        """Two widget instances must have independent session metadata dicts."""
        widget1 = self._make_widget(tmp_path)
        widget2 = self._make_widget(tmp_path)
        try:
            widget1.open_notebook("/a.ipynb")
            widget2.open_notebook("/b.ipynb")
            assert widget1.session_metadata["notebook_path"] == "/a.ipynb"
            assert widget2.session_metadata["notebook_path"] == "/b.ipynb"
        finally:
            widget1.deleteLater()
            widget2.deleteLater()

    def test_open_notebook_requires_string_path(self, tmp_path: Path) -> None:
        """open_notebook must raise TypeError for non-string input (DbC)."""
        widget = self._make_widget(tmp_path)
        try:
            with pytest.raises(TypeError):
                widget.open_notebook(123)  # type: ignore[arg-type]
        finally:
            widget.deleteLater()

    def test_set_kernel_environment_requires_string(self, tmp_path: Path) -> None:
        """set_kernel_environment must raise TypeError for non-string input (DbC)."""
        widget = self._make_widget(tmp_path)
        try:
            with pytest.raises(TypeError):
                widget.set_kernel_environment(None)  # type: ignore[arg-type]
        finally:
            widget.deleteLater()

    def test_session_metadata_keys_are_stable(self, tmp_path: Path) -> None:
        """session_metadata must always expose exactly the two documented keys."""
        widget = self._make_widget(tmp_path)
        try:
            keys = set(widget.session_metadata.keys())
            assert keys == {"notebook_path", "kernel_env"}
        finally:
            widget.deleteLater()
