"""Tests for Function Generator GUI.

Tests the PyQt6 GUI launcher and its integration with the SignalGenerator engine.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# Bootstrap for test discovery
_REPO_ROOT = Path(__file__).resolve().parents[3]
import sys

sys.path.insert(0, str(_REPO_ROOT / "src" / "shared" / "python"))
from upstream_drift_tools.bootstrap import ensure_paths  # noqa: E402

ensure_paths(_REPO_ROOT)

os.environ["HEADLESS"] = "true"


class TestSignalGeneratorEngine:
    """Test suite for SignalGenerator engine integration."""

    def test_engine_imports(self) -> None:
        """Test that all engine imports work correctly."""
        from shared.python.signal_toolkit import Signal, SignalGenerator

        assert Signal is not None
        assert SignalGenerator is not None

    def test_sinusoid_generation(self) -> None:
        """Test sinusoid signal generation."""
        from shared.python.signal_toolkit import SignalGenerator

        t = np.linspace(0, 1, 1000)
        signal = SignalGenerator.sinusoid(t, amplitude=2.0, frequency=5.0)

        assert signal is not None
        assert len(signal.values) == 1000
        assert np.max(signal.values) <= 2.0
        assert np.min(signal.values) >= -2.0

    def test_square_wave_generation(self) -> None:
        """Test square wave generation."""
        from shared.python.signal_toolkit import SignalGenerator

        t = np.linspace(0, 1, 1000)
        signal = SignalGenerator.square(t, frequency=5.0, amplitude=1.0, duty_cycle=0.5)

        assert signal is not None
        assert len(signal.values) == 1000

    def test_triangle_wave_generation(self) -> None:
        """Test triangle wave generation."""
        from shared.python.signal_toolkit import SignalGenerator

        t = np.linspace(0, 1, 1000)
        signal = SignalGenerator.triangle(t, frequency=5.0, amplitude=1.0)

        assert signal is not None
        assert len(signal.values) == 1000

    def test_exponential_generation(self) -> None:
        """Test exponential signal generation."""
        from shared.python.signal_toolkit import SignalGenerator

        t = np.linspace(0, 1, 1000)
        signal = SignalGenerator.exponential(t, amplitude=1.0, decay_rate=2.0)

        assert signal is not None
        assert signal.values[0] > signal.values[-1]  # Decaying

    def test_chirp_generation(self) -> None:
        """Test chirp signal generation."""
        from shared.python.signal_toolkit import SignalGenerator

        t = np.linspace(0, 1, 1000)
        signal = SignalGenerator.chirp(t, f0=1.0, f1=10.0)

        assert signal is not None
        assert len(signal.values) == 1000

    def test_polynomial_generation(self) -> None:
        """Test polynomial signal generation."""
        from shared.python.signal_toolkit import SignalGenerator

        t = np.linspace(0, 1, 100)
        coeffs = [1, 2, 3]  # 1 + 2t + 3t^2
        signal = SignalGenerator.polynomial(t, coeffs)

        assert signal is not None
        # Check first value (t=0): should be 1
        assert abs(signal.values[0] - 1.0) < 0.001

    def test_superposition(self) -> None:
        """Test signal superposition."""
        from shared.python.signal_toolkit import SignalGenerator

        t = np.linspace(0, 1, 1000)
        s1 = SignalGenerator.sinusoid(t, amplitude=1.0, frequency=5.0)
        s2 = SignalGenerator.sinusoid(t, amplitude=0.5, frequency=10.0)

        combined = SignalGenerator.superposition([s1, s2])
        assert combined is not None
        assert len(combined.values) == 1000


class TestFunctionGeneratorGUI:
    """Test suite for Function Generator GUI."""

    @pytest.fixture
    def mock_qt_app(self):
        """Create a mock Qt application for testing."""
        with patch("PyQt6.QtWidgets.QApplication"):
            yield

    def test_launcher_dependencies(self) -> None:
        """Test that the launcher can check dependencies."""
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from launch_pyqt6 import check_dependencies

        missing = check_dependencies()
        assert isinstance(missing, list)

    def test_module_imports(self) -> None:
        """Test that module imports work correctly."""
        from function_generator import Signal, SignalGenerator

        assert Signal is not None
        assert SignalGenerator is not None

    @pytest.mark.skipif(
        os.environ.get("DISPLAY") is None and sys.platform != "win32",
        reason="No display available",
    )
    def test_widget_creation(self, mock_qt_app) -> None:
        """Test that the widget can be created."""
        try:
            from function_generator.python.function_generator.ui.pyqt6.main_window import (
                FunctionGeneratorWidget,
            )

            with patch.object(FunctionGeneratorWidget, "_init_ui", return_value=None):
                with patch.object(
                    FunctionGeneratorWidget, "_apply_styling", return_value=None
                ):
                    with patch.object(
                        FunctionGeneratorWidget, "_connect_signals", return_value=None
                    ):
                        with patch.object(
                            FunctionGeneratorWidget,
                            "_generate_signal",
                            return_value=None,
                        ):
                            widget = FunctionGeneratorWidget.__new__(
                                FunctionGeneratorWidget
                            )
                            assert widget is not None
        except ImportError as e:
            pytest.skip(f"Qt not available: {e}")


class TestGUIRegistration:
    """Test suite for GUI registration."""

    def test_registration_imports(self) -> None:
        """Test that registration module can be imported."""
        try:
            from shared.python.gui_launcher import GUIType, LaunchConfig, register_gui

            assert GUIType is not None
            assert LaunchConfig is not None
            assert register_gui is not None
        except ImportError as e:
            pytest.skip(f"GUI launcher not available: {e}")
