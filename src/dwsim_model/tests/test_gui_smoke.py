"""
test_gui_smoke.py
=================
Widget construction smoke tests for the 5 DWSIM GUI modules.

Uses a headless Tkinter root (Tk can run without a display when mocked or
on CI with xvfb-run / Tk's built-in test mode).  Each test verifies that
the widget can be instantiated and key attributes exist without requiring
a real DWSIM runtime.
"""

from __future__ import annotations

import pytest

tk = pytest.importorskip(
    "tkinter", reason="tkinter not available — skipping GUI smoke tests"
)
ttk = pytest.importorskip(
    "tkinter.ttk", reason="tkinter.ttk not available — skipping GUI smoke tests"
)
from unittest.mock import MagicMock  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_root():
    """Create a minimal headless Tk root, skip test if display unavailable."""
    try:
        root = tk.Tk()
        root.withdraw()  # hide the window
        return root
    except tk.TclError as exc:
        pytest.skip(f"No display available for Tkinter: {exc}")


# ---------------------------------------------------------------------------
# widgets.py smoke test
# ---------------------------------------------------------------------------


class TestWidgets:
    def test_apply_styles_does_not_raise(self):
        root = _make_root()
        try:
            from dwsim_model.gui.widgets import apply_styles

            apply_styles(root)
        finally:
            root.destroy()

    def test_validated_entry_construction(self):
        root = _make_root()
        try:
            from dwsim_model.gui.widgets import ValidatedEntry

            frame = ttk.Frame(root)
            entry = ValidatedEntry(frame, validator=lambda v: float(v) > 0)
            assert entry is not None
        finally:
            root.destroy()

    def test_section_frame_construction(self):
        root = _make_root()
        try:
            from dwsim_model.gui.widgets import SectionFrame

            frame = SectionFrame(root, title="Test Section")
            assert frame is not None
        finally:
            root.destroy()


# ---------------------------------------------------------------------------
# feeds_tab.py smoke test
# ---------------------------------------------------------------------------


class TestFeedsTab:
    def test_feeds_tab_construction(self):
        root = _make_root()
        try:
            from dwsim_model.gui.tabs.feeds_tab import FeedsTab

            notebook = ttk.Notebook(root)
            controller = MagicMock()
            tab = FeedsTab(notebook, controller)
            assert isinstance(tab, ttk.Frame)
        finally:
            root.destroy()

    def test_feeds_tab_get_config_returns_dict(self):
        root = _make_root()
        try:
            from dwsim_model.gui.tabs.feeds_tab import FeedsTab

            notebook = ttk.Notebook(root)
            controller = MagicMock()
            tab = FeedsTab(notebook, controller)
            cfg = tab.get_config()
            assert isinstance(cfg, dict)
        finally:
            root.destroy()


# ---------------------------------------------------------------------------
# reactors_tab.py smoke test
# ---------------------------------------------------------------------------


class TestReactorsTab:
    def test_reactors_tab_construction(self):
        root = _make_root()
        try:
            from dwsim_model.gui.tabs.reactors_tab import ReactorsTab

            notebook = ttk.Notebook(root)
            controller = MagicMock()
            tab = ReactorsTab(notebook, controller)
            assert isinstance(tab, ttk.Frame)
        finally:
            root.destroy()

    def test_reactors_tab_get_config_returns_dict(self):
        root = _make_root()
        try:
            from dwsim_model.gui.tabs.reactors_tab import ReactorsTab

            notebook = ttk.Notebook(root)
            controller = MagicMock()
            tab = ReactorsTab(notebook, controller)
            cfg = tab.get_config()
            assert isinstance(cfg, dict)
        finally:
            root.destroy()


# ---------------------------------------------------------------------------
# results_tab.py smoke test
# ---------------------------------------------------------------------------


class TestResultsTab:
    def test_results_tab_construction(self):
        root = _make_root()
        try:
            from dwsim_model.gui.tabs.results_tab import ResultsTab

            notebook = ttk.Notebook(root)
            controller = MagicMock()
            tab = ResultsTab(notebook, controller)
            assert isinstance(tab, ttk.Frame)
        finally:
            root.destroy()

    def test_results_tab_log_does_not_raise(self):
        root = _make_root()
        try:
            from dwsim_model.gui.tabs.results_tab import ResultsTab

            notebook = ttk.Notebook(root)
            controller = MagicMock()
            tab = ResultsTab(notebook, controller)
            tab.log("Test message", "INFO")
        finally:
            root.destroy()
