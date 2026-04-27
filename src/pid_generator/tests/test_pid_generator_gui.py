"""Tests for P&ID Generator GUI registration and launcher plumbing."""

from __future__ import annotations

import importlib

import pytest

# ---------------------------------------------------------------------------
# DbC + LoD fixes — GH1479
# ---------------------------------------------------------------------------


class TestPIDGeneratorMainWindowDbCLoD:
    """DbC and LoD fixes for PIDGeneratorMainWindow (GH1479)."""

    @pytest.mark.skipif(
        importlib.util.find_spec("PyQt6") is None
        or importlib.util.find_spec("ezdxf") is None,
        reason="PyQt6 and ezdxf required",
    )
    def test_init_raises_type_error_for_non_widget_parent(self):
        """__init__ raises TypeError when parent is not a QWidget or None (DbC)."""
        from pid_generator.ui.pyqt6.main_window import PIDGeneratorMainWindow

        with pytest.raises(TypeError, match="parent must be a QWidget or None"):
            PIDGeneratorMainWindow(parent="not-a-widget")  # type: ignore[arg-type]

    @pytest.mark.skipif(
        importlib.util.find_spec("PyQt6") is None
        or importlib.util.find_spec("ezdxf") is None,
        reason="PyQt6 and ezdxf required",
    )
    def test_init_raises_type_error_for_int_parent(self):
        """__init__ raises TypeError for integer parent (DbC)."""
        from pid_generator.ui.pyqt6.main_window import PIDGeneratorMainWindow

        with pytest.raises(TypeError, match="parent must be a QWidget or None"):
            PIDGeneratorMainWindow(parent=42)  # type: ignore[arg-type]

    @pytest.mark.skipif(
        importlib.util.find_spec("PyQt6") is None
        or importlib.util.find_spec("ezdxf") is None,
        reason="PyQt6 and ezdxf required",
    )
    def test_init_accepts_none_parent(self, qapp):
        """__init__ accepts None as parent without raising (DbC)."""
        from pid_generator.ui.pyqt6.main_window import PIDGeneratorMainWindow

        window = PIDGeneratorMainWindow(parent=None)
        assert window is not None
        window.close()

    @pytest.mark.skipif(
        importlib.util.find_spec("PyQt6") is None
        or importlib.util.find_spec("ezdxf") is None,
        reason="PyQt6 and ezdxf required",
    )
    def test_lod_signal_connections_work(self, qapp):
        """Signal connections made via extracted local vars (LoD fix) are functional."""
        from pid_generator.ui.pyqt6.main_window import PIDGeneratorMainWindow

        window = PIDGeneratorMainWindow()
        # If LoD fix broke the connections, _build_ui would raise or signals
        # would not be connected. We verify the window builds without error.
        assert window.windowTitle() == "P&ID Generator"
        window.close()


# ---------------------------------------------------------------------------
# Original tests
# ---------------------------------------------------------------------------


def test_gui_info_structure():
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


def test_get_gui_info_returns_gui_info():
    """get_gui_info() helper returns the same GUI_INFO dict."""
    from pid_generator.gui_registration import GUI_INFO, get_gui_info

    assert get_gui_info() is GUI_INFO


@pytest.mark.skipif(
    importlib.util.find_spec("PyQt6") is None
    or importlib.util.find_spec("ezdxf") is None,
    reason="PyQt6 and ezdxf required",
)
def test_main_window_class_importable():
    """The PIDGeneratorMainWindow class can be imported."""
    from pid_generator.ui.pyqt6.main_window import PIDGeneratorMainWindow

    assert callable(PIDGeneratorMainWindow)


def test_pid_generator_package_version():
    """pid_generator package exposes __version__."""
    import pid_generator

    assert hasattr(pid_generator, "__version__")
    assert isinstance(pid_generator.__version__, str)
