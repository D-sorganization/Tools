"""
C3D Motion Capture Viewer GUI Tests
===================================

TDD tests for the C3D Motion Capture Viewer GUI components.
Tests cover PyQt6 main window, metadata display, and export options.
"""

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


class TestC3DViewerMainWindow:
    """Tests for the PyQt6 C3D Viewer main window."""

    @pytest.fixture
    def mock_qt_app(self) -> Any:
        """Create mock Qt application for headless testing."""
        with patch.dict(
            sys.modules,
            {
                "PyQt6": MagicMock(),
                "PyQt6.QtWidgets": MagicMock(),
                "PyQt6.QtCore": MagicMock(),
                "PyQt6.QtGui": MagicMock(),
            },
        ):
            yield

    def test_main_window_imports(self, mock_qt_app) -> Any:
        """Test that main window module can be imported."""
        try:
            from c3d_viewer.ui.pyqt6 import main_window

            assert hasattr(main_window, "C3DViewerWindow")
        except ImportError:
            pytest.skip("PyQt6 main window not yet implemented")

    def test_main_window_class_exists(self, mock_qt_app) -> Any:
        """Test that window class is defined and callable."""
        try:
            from c3d_viewer.ui.pyqt6.main_window import C3DViewerWindow

            assert callable(C3DViewerWindow)
        except ImportError:
            pytest.skip("Main window not yet implemented")


class TestC3DDataProcessing:
    """Tests for C3D data processing logic."""

    def test_frame_to_time_conversion(self) -> Any:
        """Test frame index to time conversion."""
        frame_rate = 100.0  # Hz
        frame_count = 1000

        duration = frame_count / frame_rate
        assert duration == pytest.approx(10.0)

        # Frame 500 should be at 5.0 seconds
        frame = 500
        time = frame / frame_rate
        assert time == pytest.approx(5.0)

    def test_unit_conversion_mm_to_m(self) -> Any:
        """Test unit conversion from mm to m."""
        # Conversion factors
        mm_to_m = 0.001
        m_to_mm = 1000.0

        value_mm = 1500.0  # mm
        value_m = value_mm * mm_to_m
        assert value_m == pytest.approx(1.5)

        value_m2 = 2.5  # m
        value_mm2 = value_m2 * m_to_mm
        assert value_mm2 == pytest.approx(2500.0)

    def test_biomechanical_range_validation(self) -> Any:
        """Test biomechanical marker position validation."""
        # Valid range for biomechanical markers: 1mm to 10m
        min_valid = 0.001  # 1mm in meters
        max_valid = 10.0  # 10m

        # Test valid positions
        valid_positions = [0.5, 1.0, 1.8, 2.0]  # Typical human positions
        for pos in valid_positions:
            assert min_valid <= pos <= max_valid

        # Test invalid (too small - likely mm/m confusion)
        too_small = 0.0001  # 0.1mm - suspiciously small
        assert too_small < min_valid

        # Test invalid (too large - likely error)
        too_large = 15.0  # 15m - unrealistic
        assert too_large > max_valid

    def test_analog_rate_relation(self) -> Any:
        """Test analog rate is typically multiple of frame rate."""
        frame_rate = 100.0  # Hz
        analog_rate = 1000.0  # Hz

        # Analog rate should be integer multiple of frame rate
        subframes = analog_rate / frame_rate
        assert subframes == pytest.approx(10.0)
        assert subframes == int(subframes)


class TestForcePlateAnalysis:
    """Tests for force plate data analysis."""

    def test_cop_calculation_valid(self) -> Any:
        """Test center of pressure calculation with valid force."""
        import numpy as np

        # Force plate data
        fz = np.array([500.0, 600.0, 700.0])  # N
        mx = np.array([50.0, 60.0, 70.0])  # N·m
        my = np.array([-25.0, -30.0, -35.0])  # N·m

        # COP calculation: cop_x = -my/fz, cop_y = mx/fz
        cop_x = -my / fz
        cop_y = mx / fz

        assert cop_x[0] == pytest.approx(0.05, rel=1e-3)  # 50mm
        assert cop_y[0] == pytest.approx(0.1, rel=1e-3)  # 100mm

    def test_cop_invalid_when_no_contact(self) -> Any:
        """Test COP is NaN when force is too small."""
        import numpy as np

        fz = np.array([5.0])  # N - too small for valid COP
        min_force_threshold = 10.0  # N

        # COP should be NaN when Fz < threshold
        valid_contact = np.abs(fz) > min_force_threshold
        assert not valid_contact[0]

    def test_force_plate_channel_naming(self) -> Any:
        """Test force plate channel naming conventions."""
        # Standard naming patterns
        standard_channels = ["Fx1", "Fy1", "Fz1", "Mx1", "My1", "Mz1"]
        for ch in standard_channels:
            # Should match pattern: [FM][xyz][0-9]+
            import re

            assert re.match(r"^[FM][xyz]\d+$", ch)


class TestC3DViewerGUIRegistration:
    """Tests for GUI framework registration."""

    def test_gui_registration_exists(self) -> Any:
        """Test that gui_registration.py exists and has required metadata."""
        try:
            from c3d_viewer import gui_registration

            assert hasattr(gui_registration, "GUI_METADATA")
            metadata = gui_registration.GUI_METADATA

            assert "name" in metadata
            assert "description" in metadata
            assert "category" in metadata
            assert "entry_point" in metadata
        except ImportError:
            pytest.skip("GUI registration not yet implemented")

    def test_gui_registration_category(self) -> Any:
        """Test that viewer is in biomechanics category."""
        try:
            from c3d_viewer import gui_registration

            assert gui_registration.GUI_METADATA["category"] == "biomechanics"
        except ImportError:
            pytest.skip("GUI registration not yet implemented")

    def test_launcher_exists(self) -> Any:
        """Test that launcher script exists."""
        try:
            from c3d_viewer import launch_pyqt6

            assert hasattr(launch_pyqt6, "main")
        except ImportError:
            pytest.skip("Launcher not yet implemented")


# --------------------------------------------------------------------------- #
# File-load UI states (#3978): success, reader errors, demo fallback.          #
#                                                                             #
# These tests drive the REAL PyQt6 window through ``_load_file``.              #
# They must paint the "loaded" label only after a successful load, surface     #
# reader failures (ValueError/RuntimeError/OSError) as a visible error state,  #
# and annotate demo data so fabricated numbers are never shown under a real    #
# file name.                                                                   #
# --------------------------------------------------------------------------- #

_WINDOW_MODULE_NAME = "_c3d_viewer_main_window_under_test"


def _select_file(window: Any, module: Any, path: Path) -> None:
    """Drive ``_load_file`` as if the user had chosen *path* in the dialog."""
    with patch.object(
        module.QFileDialog,
        "getOpenFileName",
        return_value=(str(path), "C3D Files (*.c3d)"),
    ):
        window._load_file()


def _write_valid_c3d(directory: Path) -> Path | None:
    """Write a minimal valid C3D file; return None when ezc3d is unavailable."""
    try:
        import ezc3d
        import numpy as np
    except ImportError:
        return None

    c3d = ezc3d.c3d()
    c3d.add_parameter("POINT", "LABELS", ["LASI", "RASI"])
    c3d.add_parameter("POINT", "DESCRIPTIONS", ["Left hip", "Right hip"])
    c3d.add_parameter("POINT", "UNITS", ["mm"])
    c3d.add_parameter("POINT", "RATE", [100.0])
    c3d.add_parameter("POINT", "FRAMES", [10])
    c3d.add_parameter("ANALOG", "LABELS", ["Fx1"])
    c3d.add_parameter("ANALOG", "DESCRIPTIONS", ["Force x"])
    c3d.add_parameter("ANALOG", "UNITS", ["N"])
    c3d.add_parameter("ANALOG", "SCALE", [1.0])
    c3d.add_parameter("ANALOG", "OFFSET", [0.0])
    c3d.add_parameter("ANALOG", "RATE", [100.0])
    c3d["data"]["points"] = np.zeros((4, 2, 10))
    c3d["data"]["analogs"] = np.zeros((1, 1, 10))
    out = directory / "valid_capture.c3d"
    c3d.write(str(out))
    return out


class TestC3DViewerFileLoadStates:
    """File-load UI states for the PyQt6 C3D Viewer window (#3978)."""

    @pytest.fixture(scope="session")
    def qt_display(self) -> None:
        """Skip the class when no Qt platform backend can be initialised."""
        offscreen = os.environ.get("QT_QPA_PLATFORM") == "offscreen"
        has_display = bool(
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        )
        native = sys.platform in ("win32", "darwin")
        if not (offscreen or has_display or native):
            pytest.skip("No Qt platform available (set QT_QPA_PLATFORM=offscreen)")
        from PyQt6.QtWidgets import QApplication

        if QApplication.instance() is None:
            QApplication([])

    @pytest.fixture(scope="class")
    def main_window_module(self) -> Any:
        """Load the real PyQt6 main-window module by explicit file path.

        The repository ships two packages named ``c3d_viewer`` - the GUI
        registration metadata package (``src/c3d_viewer``) and the GUI
        package (``src/c3d_viewer/python/c3d_viewer``) - so a plain
        ``import c3d_viewer`` resolves to whichever wins on ``sys.path``.
        Loading the window module by file path is deterministic and leaves
        the registration-package tests in this module untouched.
        """
        window_file = (
            Path(__file__).resolve().parents[1]
            / "python"
            / "c3d_viewer"
            / "ui"
            / "pyqt6"
            / "main_window.py"
        )
        spec = importlib.util.spec_from_file_location(_WINDOW_MODULE_NAME, window_file)
        if spec is None or spec.loader is None:
            pytest.skip(f"PyQt6 main window not available: no spec for {window_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[_WINDOW_MODULE_NAME] = module
        try:
            spec.loader.exec_module(module)
        except ImportError as exc:
            pytest.skip(f"PyQt6 main window not available: {exc}")
        yield module
        sys.modules.pop(_WINDOW_MODULE_NAME, None)

    @pytest.fixture
    def viewer_window(self, qt_display, qtbot, main_window_module) -> Any:
        """Create a real C3DViewerWindow registered for teardown."""
        window = main_window_module.C3DViewerWindow()
        qtbot.addWidget(window)
        return window

    def test_good_file_shows_green_label_with_filename(
        self, viewer_window, main_window_module, tmp_path
    ) -> Any:
        """A successful load paints the label green with the bare file name."""
        good = _write_valid_c3d(tmp_path)
        if good is None:
            pytest.skip("ezc3d not available")

        _select_file(viewer_window, main_window_module, good)

        palette = main_window_module.CATPPUCCIN_MOCHA
        assert viewer_window.file_label.text() == good.name
        assert palette["green"] in viewer_window.file_label.styleSheet()
        assert viewer_window.info_labels["marker_count"].text() == "2"
        assert viewer_window.marker_list.count() == 2
        assert viewer_window.analog_table.rowCount() == 1
        assert viewer_window.export_status.toPlainText() == ""

    def test_corrupt_file_shows_error_state_and_not_green(
        self, viewer_window, main_window_module, tmp_path
    ) -> Any:
        """A corrupt file shows a visible error state instead of raising."""
        corrupt = tmp_path / "corrupt.c3d"
        corrupt.write_text("this is definitely not a c3d file")
        # The reader validates the header itself before touching ezc3d, so a
        # sentinel keeps the ValueError path deterministic in environments
        # where ezc3d is missing (which would otherwise raise ImportError).
        with patch("shared.python.sidekick.lab.bio.c3d_reader.ezc3d", MagicMock()):
            _select_file(viewer_window, main_window_module, corrupt)

        palette = main_window_module.CATPPUCCIN_MOCHA
        label = viewer_window.file_label
        assert "corrupt.c3d" in label.text()
        assert "load failed" in label.text()
        assert palette["green"] not in label.styleSheet()
        assert palette["red"] in label.styleSheet()
        assert viewer_window.info_labels["marker_count"].text() == "-"
        assert viewer_window.events_list.count() == 0
        assert viewer_window.marker_list.count() == 0
        assert viewer_window.analog_table.rowCount() == 0
        status = viewer_window.export_status.toPlainText()
        assert "corrupt.c3d" in status
        assert "Not a valid C3D file" in status
        assert "Choose a valid C3D file" in status

    def test_stale_panels_cleared_after_good_load_then_corrupt(
        self, viewer_window, main_window_module, tmp_path
    ) -> Any:
        """A failed re-load clears panels from the previously loaded file."""
        good = _write_valid_c3d(tmp_path)
        if good is None:
            pytest.skip("ezc3d not available")
        corrupt = tmp_path / "corrupt.c3d"
        corrupt.write_text("garbage")

        _select_file(viewer_window, main_window_module, good)
        assert viewer_window.marker_list.count() == 2
        with patch("shared.python.sidekick.lab.bio.c3d_reader.ezc3d", MagicMock()):
            _select_file(viewer_window, main_window_module, corrupt)

        palette = main_window_module.CATPPUCCIN_MOCHA
        assert palette["red"] in viewer_window.file_label.styleSheet()
        assert palette["green"] not in viewer_window.file_label.styleSheet()
        assert viewer_window.info_labels["frame_count"].text() == "-"
        assert viewer_window.events_list.count() == 0
        assert viewer_window.marker_list.count() == 0
        assert viewer_window.analog_table.rowCount() == 0

    def test_truncated_file_shows_error_state(
        self, viewer_window, main_window_module, tmp_path
    ) -> Any:
        """A truncated file (RuntimeError from ezc3d) shows the error state."""
        good = _write_valid_c3d(tmp_path)
        if good is None:
            pytest.skip("ezc3d not available")
        truncated = tmp_path / "truncated.c3d"
        truncated.write_bytes(good.read_bytes()[:128])

        _select_file(viewer_window, main_window_module, truncated)

        palette = main_window_module.CATPPUCCIN_MOCHA
        assert "truncated.c3d" in viewer_window.file_label.text()
        assert "load failed" in viewer_window.file_label.text()
        assert palette["green"] not in viewer_window.file_label.styleSheet()
        assert palette["red"] in viewer_window.file_label.styleSheet()
        assert viewer_window.marker_list.count() == 0

    def test_missing_file_shows_error_state(
        self, viewer_window, main_window_module, tmp_path
    ) -> Any:
        """A vanished file (OSError from the reader) shows the error state."""
        missing = tmp_path / "vanished.c3d"

        _select_file(viewer_window, main_window_module, missing)

        palette = main_window_module.CATPPUCCIN_MOCHA
        assert "vanished.c3d" in viewer_window.file_label.text()
        assert "load failed" in viewer_window.file_label.text()
        assert palette["green"] not in viewer_window.file_label.styleSheet()
        assert palette["red"] in viewer_window.file_label.styleSheet()
        assert viewer_window.marker_list.count() == 0
        assert "vanished.c3d" in viewer_window.export_status.toPlainText()

    def test_missing_library_shows_annotated_demo_data(
        self, viewer_window, main_window_module, tmp_path
    ) -> Any:
        """Without ezc3d, demo data is shown under an annotated label."""
        chosen = tmp_path / "real_capture.c3d"
        chosen.write_text("placeholder - reader raises ImportError before reading")

        with patch("shared.python.sidekick.lab.bio.c3d_reader.ezc3d", None):
            _select_file(viewer_window, main_window_module, chosen)

        palette = main_window_module.CATPPUCCIN_MOCHA
        label = viewer_window.file_label
        assert "real_capture.c3d" in label.text()
        assert "demo data" in label.text()
        assert "library unavailable" in label.text()
        assert palette["green"] not in label.styleSheet()
        assert palette["yellow"] in label.styleSheet()
        assert viewer_window.marker_list.count() == 12
        assert "demo" in viewer_window.info_labels["marker_count"].text().lower()
        status = viewer_window.export_status.toPlainText()
        assert "ezc3d" in status
        assert "pip install ezc3d" in status

    def test_demo_annotation_replaces_green_after_good_load(
        self, viewer_window, main_window_module, tmp_path
    ) -> Any:
        """Demo fallback after a good load never leaves a green real name."""
        good = _write_valid_c3d(tmp_path)
        if good is None:
            pytest.skip("ezc3d not available")
        second = tmp_path / "second.c3d"
        second.write_text("placeholder")

        _select_file(viewer_window, main_window_module, good)
        palette = main_window_module.CATPPUCCIN_MOCHA
        assert palette["green"] in viewer_window.file_label.styleSheet()
        assert viewer_window.export_status.toPlainText() == ""
        with patch("shared.python.sidekick.lab.bio.c3d_reader.ezc3d", None):
            _select_file(viewer_window, main_window_module, second)

        label = viewer_window.file_label
        assert "second.c3d" in label.text()
        assert "demo data" in label.text()
        assert palette["green"] not in label.styleSheet()
        assert palette["yellow"] in label.styleSheet()
        assert viewer_window.marker_list.count() == 12
        assert "pip install ezc3d" in viewer_window.export_status.toPlainText()
