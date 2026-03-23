"""Tests for Signal Processing Studio.

Tests cross-widget signal routing, polynomial resample logic,
and the unified studio launcher.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_INNER_PYTHON_DIR = str(Path(__file__).resolve().parents[1] / "python")

# Force the inner signal_processing_studio package to take precedence.
# pytest may have already resolved the outer __init__.py from src/.
# Ensure the inner python/ dir is at position 0 so the real package is found
# before the outer namespace package in src/.
if _INNER_PYTHON_DIR in sys.path:
    sys.path.remove(_INNER_PYTHON_DIR)
sys.path.insert(0, _INNER_PYTHON_DIR)

for _mod in list(sys.modules.keys()):
    # Only clear the core package and its modules — not the tests subpackage
    # (which pytest has already registered under signal_processing_studio.tests.*).
    if _mod in (
        "signal_processing_studio",
        "signal_processing_studio.signal_bus",
        "signal_processing_studio.main_window",
    ):
        del sys.modules[_mod]

# Invalidate Python's path finder caches so the repositioned sys.path entry
# takes effect. importlib is accessible via the `import importlib.util` above.
importlib.invalidate_caches()

from signal_toolkit.core import Signal

from signal_processing_studio.signal_bus import SignalBus

# =============================================================================
# Resample Drawn Points Tests (polynomial_generator._resample_drawn_points)
# =============================================================================


class TestResampleDrawnPoints:
    """Tests for PolynomialGeneratorWidget._resample_drawn_points."""

    @pytest.fixture(autouse=True)
    def _import_resample(self):
        """Import the static method under test."""
        from signal_toolkit.polynomial_generator import PolynomialGeneratorWidget

        self.resample = PolynomialGeneratorWidget._resample_drawn_points

    def test_uniform_x_spacing(self) -> None:
        """Output points should have uniformly spaced x values."""
        points: list[tuple[float, float]] = [(0.0, 0.0)]
        for i in range(1, 50):
            points.append((i * 0.1, float(i)))
        for i in range(5):
            points.append((5.0 + i * 2.0, 50.0 + float(i)))

        result = self.resample(points, n=20)

        xs = [p[0] for p in result]
        assert len(result) == 20
        diffs = np.diff(xs)
        assert np.allclose(diffs, diffs[0], rtol=1e-10)

    def test_single_point_passthrough(self) -> None:
        """A single point should be returned as-is."""
        result = self.resample([(3.0, 7.0)], n=10)
        assert result == [(3.0, 7.0)]

    def test_empty_input(self) -> None:
        """Empty input should return empty list."""
        result = self.resample([], n=10)
        assert result == []

    def test_two_points(self) -> None:
        """Two points should interpolate to n evenly-spaced points."""
        result = self.resample([(0.0, 0.0), (10.0, 10.0)], n=5)

        assert len(result) == 5
        assert result[0][0] == pytest.approx(0.0)
        assert result[-1][0] == pytest.approx(10.0)
        for x, y in result:
            assert y == pytest.approx(x, abs=1e-10)

    def test_unsorted_x_values(self) -> None:
        """Points with unsorted x should be handled correctly."""
        points = [(5.0, 5.0), (0.0, 0.0), (10.0, 10.0), (2.5, 2.5)]
        result = self.resample(points, n=5)

        xs = [p[0] for p in result]
        assert xs == sorted(xs)
        assert np.allclose(np.diff(xs), np.diff(xs)[0])

    def test_duplicate_x_averaged(self) -> None:
        """Duplicate x values should have their y values averaged."""
        points = [(0.0, 0.0), (5.0, 4.0), (5.0, 6.0), (10.0, 10.0)]
        result = self.resample(points, n=3)

        xs = [p[0] for p in result]
        ys = [p[1] for p in result]
        assert xs[1] == pytest.approx(5.0)
        assert ys[1] == pytest.approx(5.0)

    def test_all_same_x(self) -> None:
        """All points at the same x should return original points."""
        points = [(5.0, 1.0), (5.0, 2.0), (5.0, 3.0)]
        result = self.resample(points, n=10)
        assert len(result) == 3

    def test_preserves_endpoint_range(self) -> None:
        """Output x range should match input x range."""
        points = [(2.0, 10.0), (8.0, 20.0), (5.0, 15.0)]
        result = self.resample(points, n=15)

        xs = [p[0] for p in result]
        assert xs[0] == pytest.approx(2.0)
        assert xs[-1] == pytest.approx(8.0)

    def test_default_n(self) -> None:
        """Default n=30 should produce 30 points."""
        points = [(float(i), float(i)) for i in range(100)]
        result = self.resample(points)
        assert len(result) == 30


# =============================================================================
# SignalBus Routing Tests
# =============================================================================


def _make_bus(
    toolkit_signal: Signal | None = None,
    status_callback=None,
) -> tuple:
    """Create a SignalBus with mocked widgets (no Qt event loop needed)."""
    func_gen = MagicMock()
    func_gen.signal_generated = MagicMock()
    func_gen.signal_generated.connect = MagicMock()

    toolkit = MagicMock()
    toolkit.current_signal = toolkit_signal

    poly_gen = MagicMock()
    poly_gen.polynomial_generated = MagicMock()
    poly_gen.polynomial_generated.connect = MagicMock()

    bus = SignalBus.__new__(SignalBus)
    bus.signal_routed = MagicMock()
    bus.func_gen = func_gen
    bus.toolkit = toolkit
    bus.poly_gen = poly_gen
    bus._status_callback = status_callback

    return bus, func_gen, toolkit, poly_gen


class TestSignalBus:
    """Tests for SignalBus cross-widget routing."""

    def test_func_gen_signal_routed_to_toolkit(self) -> None:
        """Function Generator signal should be routed to toolkit."""
        bus, func_gen, toolkit, _ = _make_bus()

        mock_signal = MagicMock()
        mock_signal.name = "test_sine"
        mock_signal.n_samples = 1000
        mock_signal.fs = 100.0

        bus._on_func_gen_signal(mock_signal)

        toolkit.load_external_signal.assert_called_once_with(mock_signal)
        bus.signal_routed.emit.assert_called_once()

    def test_poly_coefficients_reversed(self) -> None:
        """Polynomial coefficients should be reversed before creating Signal."""
        t = np.linspace(0, 10, 100)
        toolkit_signal = Signal(t, np.zeros(100))
        bus, _, toolkit, _ = _make_bus(toolkit_signal=toolkit_signal)

        # np.polyfit returns [highest, ..., lowest]
        # y = 3*x^2 + 2*x + 1 -> polyfit returns [3, 2, 1]
        bus._on_poly_generated("knee", [3.0, 2.0, 1.0])

        toolkit.load_external_signal.assert_called_once()
        routed_signal = toolkit.load_external_signal.call_args[0][0]
        assert routed_signal.name == "Polynomial (knee)"

        # Reversed coeffs: 1 + 2t + 3t^2
        assert routed_signal.values[0] == pytest.approx(1.0, abs=0.01)
        assert routed_signal.values[-1] == pytest.approx(321.0, abs=1.0)

    def test_poly_default_time_range(self) -> None:
        """When toolkit has no signal, default time range should be used."""
        bus, _, toolkit, _ = _make_bus(toolkit_signal=None)

        bus._on_poly_generated("hip", [1.0, 0.0])

        routed_signal = toolkit.load_external_signal.call_args[0][0]
        assert len(routed_signal.time) == 1000
        assert routed_signal.time[0] == pytest.approx(0.0)
        assert routed_signal.time[-1] == pytest.approx(10.0)

    def test_send_current_to_toolkit_none(self) -> None:
        """send_current_to_toolkit should do nothing when no current signal."""
        bus, func_gen, toolkit, _ = _make_bus()
        func_gen.current_signal = None

        bus.send_current_to_toolkit()
        toolkit.load_external_signal.assert_not_called()

    def test_send_current_to_toolkit(self) -> None:
        """send_current_to_toolkit should route the func_gen current signal."""
        bus, func_gen, toolkit, _ = _make_bus()

        mock_signal = MagicMock()
        mock_signal.name = "test"
        mock_signal.n_samples = 500
        mock_signal.fs = 50.0
        func_gen.current_signal = mock_signal

        bus.send_current_to_toolkit()
        toolkit.load_external_signal.assert_called_once_with(mock_signal)

    def test_status_callback(self) -> None:
        """Status callback should be called when provided."""
        messages: list[str] = []
        bus, _, toolkit, _ = _make_bus(status_callback=messages.append)

        mock_signal = MagicMock()
        mock_signal.name = "wave"
        mock_signal.n_samples = 100
        mock_signal.fs = 10.0
        bus._on_func_gen_signal(mock_signal)

        assert len(messages) == 1
        assert "wave" in messages[0]

    def test_status_no_callback(self) -> None:
        """Status with no callback should not raise."""
        bus, _, _, _ = _make_bus(status_callback=None)

        mock_signal = MagicMock()
        mock_signal.name = "safe"
        mock_signal.n_samples = 10
        mock_signal.fs = 1.0
        bus._on_func_gen_signal(mock_signal)


# =============================================================================
# load_external_signal Tests
# =============================================================================


class TestLoadExternalSignal:
    """Tests for SignalToolkitWidget.load_external_signal."""

    def test_load_external_signal_sets_current(self) -> None:
        """load_external_signal should set current_signal and original_signal."""
        with patch("signal_toolkit.widget.QWidget.__init__"):
            from signal_toolkit.widget import SignalToolkitWidget

            widget = SignalToolkitWidget.__new__(SignalToolkitWidget)
            widget.current_signal = None
            widget.original_signal = None
            widget._update_plot = MagicMock()
            widget._log = MagicMock()
            widget.signal_updated = MagicMock()

            t = np.linspace(0, 5, 500)
            signal = Signal(t, np.sin(t), name="ext_sine")

            widget.load_external_signal(signal)

            assert widget.current_signal is signal
            assert widget.original_signal is not None
            assert widget.original_signal is not signal
            widget._update_plot.assert_called_once()
            widget.signal_updated.emit.assert_called_once_with(signal)
            widget._log.assert_called_once()


# =============================================================================
# use_builtin_theme Tests
# =============================================================================


class TestBuiltinTheme:
    """Tests for use_builtin_theme parameter on widgets."""

    def test_polynomial_generator_theme_flag_exists(self) -> None:
        """PolynomialGeneratorWidget.__init__ should accept use_builtin_theme."""
        import inspect

        from signal_toolkit.polynomial_generator import PolynomialGeneratorWidget

        sig = inspect.signature(PolynomialGeneratorWidget.__init__)
        assert "use_builtin_theme" in sig.parameters
        assert sig.parameters["use_builtin_theme"].default is True

    def test_signal_toolkit_theme_flag_exists(self) -> None:
        """SignalToolkitWidget.__init__ should accept use_builtin_theme."""
        import inspect

        from signal_toolkit.widget import SignalToolkitWidget

        sig = inspect.signature(SignalToolkitWidget.__init__)
        assert "use_builtin_theme" in sig.parameters
        assert sig.parameters["use_builtin_theme"].default is True


# =============================================================================
# Launcher / Registration Tests
# =============================================================================


class TestLauncher:
    """Tests for Signal Processing Studio launcher."""

    def test_launcher_module_loads(self) -> None:
        """Launcher module should load without errors."""
        # Import by file path to avoid polluting sys.modules
        spec = importlib.util.spec_from_file_location(
            "studio_launch_pyqt6",
            _REPO_ROOT / "src" / "signal_processing_studio" / "launch_pyqt6.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod is not None

    def test_signal_bus_class(self) -> None:
        """SignalBus should have expected methods."""
        assert hasattr(SignalBus, "_on_func_gen_signal")
        assert hasattr(SignalBus, "_on_poly_generated")
        assert hasattr(SignalBus, "send_current_to_toolkit")

    def test_main_window_class(self) -> None:
        """SignalProcessingStudio should be importable."""
        from signal_processing_studio.main_window import SignalProcessingStudio

        assert SignalProcessingStudio is not None


# =============================================================================
# DbC Precondition Tests — GH1487
# =============================================================================


class TestSignalBusDbC:
    """Tests for DbC preconditions added to SignalBus (GH1487)."""

    def test_on_func_gen_signal_raises_for_none_signal(self) -> None:
        """_on_func_gen_signal should raise ValueError when signal is None."""
        bus, _, _, _ = _make_bus()
        with pytest.raises(ValueError, match="signal must not be None"):
            bus._on_func_gen_signal(None)

    def test_on_poly_generated_raises_for_none_joint_name(self) -> None:
        """_on_poly_generated should raise ValueError when joint_name is None."""
        bus, _, _, _ = _make_bus()
        with pytest.raises(ValueError, match="joint_name must not be None"):
            bus._on_poly_generated(None, [1.0, 0.0])

    def test_on_poly_generated_raises_type_error_for_non_list_coeffs(self) -> None:
        """_on_poly_generated should raise TypeError when coeffs is not a list."""
        bus, _, _, _ = _make_bus()
        with pytest.raises(TypeError, match="coeffs must be a list"):
            bus._on_poly_generated("knee", (1.0, 0.0))  # tuple instead of list

    def test_on_poly_generated_accepts_empty_list_coeffs(self) -> None:
        """_on_poly_generated should accept an empty list for coeffs."""
        bus, _, toolkit, _ = _make_bus(toolkit_signal=None)
        # Empty list reversed is still empty; SignalGenerator handles it
        try:
            bus._on_poly_generated("hip", [])
        except TypeError:
            pytest.fail("Empty list for coeffs should not raise TypeError")

    def test_status_raises_for_none_message(self) -> None:
        """_status should raise ValueError when message is None."""
        bus, _, _, _ = _make_bus()
        with pytest.raises(ValueError, match="message must not be None"):
            bus._status(None)


class TestMainWindowDbCLoD:
    """Tests for DbC and LoD fixes in main_window.py (GH1487)."""

    def test_on_poly_fallback_raises_for_none_joint_name(self) -> None:
        """_on_poly_fallback should raise ValueError when joint_name is None."""
        from signal_processing_studio.main_window import SignalProcessingStudio

        studio = SignalProcessingStudio.__new__(SignalProcessingStudio)
        studio.toolkit = MagicMock()
        studio.toolkit.current_signal = None

        with pytest.raises(ValueError, match="joint_name must not be None"):
            studio._on_poly_fallback(None, [1.0, 0.0])

    def test_on_poly_fallback_raises_type_error_for_non_list_coeffs(self) -> None:
        """_on_poly_fallback should raise TypeError when coeffs is not a list."""
        from signal_processing_studio.main_window import SignalProcessingStudio

        studio = SignalProcessingStudio.__new__(SignalProcessingStudio)
        studio.toolkit = MagicMock()
        studio.toolkit.current_signal = None

        with pytest.raises(TypeError, match="coeffs must be a list"):
            studio._on_poly_fallback("knee", (1.0, 0.0))

    def test_connect_action_helper_exists(self) -> None:
        """_connect_action module-level helper should be importable."""
        from signal_processing_studio.main_window import _connect_action

        assert callable(_connect_action)

    def test_connect_action_raises_for_none_action(self) -> None:
        """_connect_action should raise ValueError when action is None."""
        from signal_processing_studio.main_window import _connect_action

        with pytest.raises(ValueError, match="action must not be None"):
            _connect_action(None, lambda: None)

    def test_connect_action_raises_for_none_slot(self) -> None:
        """_connect_action should raise ValueError when slot is None."""
        from signal_processing_studio.main_window import _connect_action

        with pytest.raises(ValueError, match="slot must not be None"):
            _connect_action(MagicMock(), None)
