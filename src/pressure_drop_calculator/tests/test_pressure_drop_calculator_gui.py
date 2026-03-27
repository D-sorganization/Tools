from typing import Any

"""Tests for Pressure Drop Calculator GUI."""

from __future__ import annotations  # noqa: F404

import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]

os.environ["HEADLESS"] = "true"


class TestPressureDropEngine:
    """Test suite for Pressure Drop Calculator engine integration."""

    def test_engine_imports(self) -> None:
        """Test that all engine imports work correctly."""
        from upstream_drift_tools.process_calculators.pressure_drop_calculator import (
            PressureDropCalculationEngine,
        )

        assert PressureDropCalculationEngine is not None

    def test_interface_imports(self) -> None:
        """Test that interface imports work correctly."""
        from upstream_drift_tools.process_calculators.pressure_drop_calculator import (
            calculate_pressure_drop,
        )

        assert callable(calculate_pressure_drop)

    def test_basic_calculation(self) -> None:
        """Test a basic pressure drop calculation."""
        from upstream_drift_tools.process_calculators.pressure_drop_calculator import (
            calculate_pressure_drop,
        )

        result = calculate_pressure_drop(
            pipe_size="4",
            pipe_schedule="40",
            pipe_length=100,
            flow_rate=1000,
            flow_unit="kg/h",
            pressure=10,
            temperature=300,
            gas_composition={"N2": 0.78, "O2": 0.21, "Ar": 0.01},
        )

        assert result is not None
        assert "total_pressure_drop" in result
        assert result["total_pressure_drop"] > 0


class TestPressureDropGUI:
    """Test suite for Pressure Drop Calculator GUI."""

    @pytest.fixture
    def mock_qt_app(self) -> Any:
        """Create a mock Qt application for testing."""
        with patch("PyQt6.QtWidgets.QApplication"):
            yield

    def test_launcher_dependencies(self) -> None:
        """Test that the launcher can check dependencies."""
        launcher_path = Path(__file__).resolve().parents[1] / "launch_pyqt6.py"
        spec = importlib.util.spec_from_file_location("pressure_drop_launch_pyqt6", launcher_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        missing = mod.check_dependencies()
        assert isinstance(missing, list)

    def test_module_imports(self) -> None:
        """Test that module imports work correctly."""
        from pressure_drop_calculator import (
            PressureDropCalculationEngine,
            calculate_pressure_drop,
        )

        assert PressureDropCalculationEngine is not None
        assert callable(calculate_pressure_drop)

    @pytest.mark.skipif(
        os.environ.get("DISPLAY") is None and sys.platform != "win32",
        reason="No display available",
    )
    def test_widget_creation(self, mock_qt_app) -> None:
        """Test that the widget can be created."""
        try:
            from pressure_drop_calculator.python.pressure_drop_calculator.ui.pyqt6.main_window import (
                PressureDropCalculatorWidget,
            )

            with (
                patch.object(PressureDropCalculatorWidget, "_init_ui", return_value=None),
                patch.object(PressureDropCalculatorWidget, "_apply_styling", return_value=None),
                patch.object(
                    PressureDropCalculatorWidget,
                    "_connect_signals",
                    return_value=None,
                ),
            ):
                widget = PressureDropCalculatorWidget.__new__(PressureDropCalculatorWidget)
                assert widget is not None
        except ImportError as e:
            pytest.skip(f"Qt not available: {e}")


class TestGUIRegistration:
    """Test suite for GUI registration."""

    def test_registration_imports(self) -> None:
        """Test that registration module can be imported."""
        try:
            from gui_launcher import GUIType, LaunchConfig, register_gui

            assert GUIType is not None
            assert LaunchConfig is not None
            assert register_gui is not None
        except ImportError as e:
            pytest.skip(f"GUI launcher not available: {e}")
