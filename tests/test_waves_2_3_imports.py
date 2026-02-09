"""Tests for Waves 2-3: utils namespace cleanup and import standardization.

Wave 2 verifies:
- utils/__init__.py no longer eagerly imports all submodule symbols
- Direct submodule imports still work (from utils.logging_utils import ...)
- upstream_drift_tools.utils has documented overlap table

Wave 3 verifies:
- Direct package imports work (from upstream_drift_tools.X import ...)
- Direct package imports work (from signal_toolkit import ...)
- Direct package imports work (from gui_launcher import ...)
- Direct package imports work (from plot_engine.specs import ...)
- Direct package imports work (from plot_theme import ...)
- Backward compat: shared.python.X still importable for transition
"""

from __future__ import annotations

import importlib
import sys

import pytest

# ── Wave 2: utils namespace is thin ─────────────────────────────────────────


class TestUtilsNamespaceThin:
    """utils/__init__.py should NOT eagerly import submodule symbols."""

    def test_no_eager_get_logger(self) -> None:
        """get_logger should not be importable from bare 'utils'."""
        # Reload to ensure clean state
        if "utils" in sys.modules:
            importlib.reload(sys.modules["utils"])
        import utils

        assert not hasattr(utils, "get_logger"), (
            "utils.__init__.py should not eagerly import get_logger; "
            "import from utils.logging_utils instead"
        )

    def test_no_eager_safe_execute(self) -> None:
        import utils

        assert not hasattr(utils, "safe_execute")

    def test_no_eager_BaseTestCase(self) -> None:
        import utils

        assert not hasattr(utils, "BaseTestCase")

    def test_no_eager_debug_log(self) -> None:
        import utils

        assert not hasattr(utils, "debug_log")

    def test_submodule_import_still_works(self) -> None:
        """Direct submodule imports must continue to work."""
        from utils.logging_utils import get_logger

        assert callable(get_logger)

    def test_error_handling_submodule(self) -> None:
        from utils.error_handling import safe_execute

        assert callable(safe_execute)

    def test_path_setup_submodule(self) -> None:
        from utils.path_setup import get_repo_root

        root = get_repo_root()
        assert root.exists()


class TestUpstreamDriftToolsUtilsDocumented:
    """upstream_drift_tools.utils docstring documents overlaps."""

    def test_docstring_mentions_overlap(self) -> None:
        import upstream_drift_tools.utils

        doc = upstream_drift_tools.utils.__doc__ or ""
        assert "Overlap" in doc or "overlap" in doc
        assert "get_repo_root" in doc
        assert "get_logger" in doc


# ── Wave 3: direct package imports ──────────────────────────────────────────


class TestDirectUpstreamDriftToolsImport:
    """from upstream_drift_tools.X should work without shared.python prefix."""

    def test_utils_paths(self) -> None:
        from upstream_drift_tools.utils.paths import get_repo_root

        root = get_repo_root()
        assert root.exists()

    def test_utils_state_manager(self) -> None:
        from upstream_drift_tools.utils.state_manager import StateManager

        assert StateManager is not None

    def test_utils_unit_constants(self) -> None:
        from upstream_drift_tools.utils.unit_constants import R_UNIVERSAL

        assert abs(R_UNIVERSAL - 8.314462618) < 1e-6

    def test_bootstrap(self) -> None:
        from upstream_drift_tools.bootstrap import ensure_paths

        assert callable(ensure_paths)


class TestDirectSignalToolkitImport:
    """from signal_toolkit should work without shared.python prefix."""

    def test_signal_generator(self) -> None:
        try:
            from signal_toolkit import SignalGenerator

            assert SignalGenerator is not None
        except ImportError:
            pytest.skip("signal_toolkit not available (optional dep)")


class TestDirectGuiLauncherImport:
    """from gui_launcher should work without shared.python prefix."""

    def test_gui_type(self) -> None:
        try:
            from gui_launcher import GUIType

            assert GUIType is not None
        except ImportError:
            pytest.skip("gui_launcher not available")


class TestDirectPlotEngineImport:
    """from plot_engine.specs should work without shared.python prefix."""

    def test_plot_spec(self) -> None:
        try:
            from plot_engine.specs import PlotSpec

            assert PlotSpec is not None
        except ImportError:
            pytest.skip("plot_engine not available (needs pydantic)")


class TestDirectPlotThemeImport:
    """from plot_theme should work without shared.python prefix."""

    def test_plot_theme_manager(self) -> None:
        try:
            from plot_theme import PlotThemeManager

            assert PlotThemeManager is not None
        except ImportError:
            pytest.skip("plot_theme not available")


class TestDirectChatImport:
    """from chat should work without shared.python prefix."""

    def test_chat_models(self) -> None:
        try:
            from chat.models import ChatMessageRequest

            assert ChatMessageRequest is not None
        except ImportError:
            pytest.skip("chat models not available (needs pydantic)")


# ── Backward compatibility: shared.python.X still works ─────────────────────


class TestBackwardCompatSharedPython:
    """The old shared.python.X import style should still resolve."""

    def test_shared_python_theme_still_works(self) -> None:
        """Theme imports use shared.python prefix intentionally."""
        try:
            from shared.python.theme import ThemeManager  # noqa: F401

            assert ThemeManager is not None
        except ImportError:
            pytest.skip("theme not available (needs PyQt6)")

    def test_shared_python_upstream_drift_tools_still_resolves(self) -> None:
        """Even though we prefer direct, the old path still works."""
        from shared.python.upstream_drift_tools.utils.paths import (
            get_repo_root,
        )

        root = get_repo_root()
        assert root.exists()

    def test_shared_python_signal_toolkit_still_resolves(self) -> None:
        try:
            from shared.python.signal_toolkit import SignalGenerator  # noqa: F401

            assert SignalGenerator is not None
        except ImportError:
            pytest.skip("signal_toolkit not available")
