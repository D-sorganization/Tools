"""Tests for Acid Gas Dewpoint Calculator GUI."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Bootstrap for test discovery
_REPO_ROOT = Path(__file__).resolve().parents[3]
import sys

from upstream_drift_tools.bootstrap import ensure_paths  # noqa: E402

ensure_paths(_REPO_ROOT)


class TestAcidGasDewpointEngine:
    """Tests for the acid gas dewpoint calculation engine."""

    def test_calculator_import(self) -> None:
        """Test that the calculator can be imported."""
        from upstream_drift_tools.process_calculators.acid_gas_dewpoint_calculator import (
            AcidGasDewpointCalculator,
        )

        calculator = AcidGasDewpointCalculator()
        assert calculator is not None

    def test_composition_dataclass(self) -> None:
        """Test the AcidGasComposition dataclass."""
        from upstream_drift_tools.process_calculators.acid_gas_dewpoint_calculator import (
            AcidGasComposition,
        )

        comp = AcidGasComposition(h2o=0.15, hf=0.001, hcl=0.002, h2s=0.005)
        assert comp.h2o == 0.15
        assert comp.hf == 0.001
        assert comp.hcl == 0.002
        assert comp.h2s == 0.005
        assert comp.total == pytest.approx(0.158, abs=0.001)

    def test_dewpoint_calculation(self) -> None:
        """Test dewpoint calculation for a mixture."""
        from upstream_drift_tools.process_calculators.acid_gas_dewpoint_calculator import (
            AcidGasComposition,
            AcidGasDewpointCalculator,
        )

        calculator = AcidGasDewpointCalculator()
        composition = AcidGasComposition(h2o=0.15, hf=0.001, hcl=0.002, h2s=0.005)

        result = calculator.calculate_dewpoint_mixture(
            temperature_c=150, pressure_bar=30, composition=composition
        )

        # Check result structure
        assert result.temperature_c == 150
        assert result.pressure_bar == 30
        assert result.overall_dewpoint_c is not None
        assert result.limiting_component in ["H2O", "HF", "HCl", "H2S", "Unknown"]
        assert result.condensation_risk is not None

    def test_vapor_pressure_calculation(self) -> None:
        """Test vapor pressure calculation for water."""
        from upstream_drift_tools.process_calculators.acid_gas_dewpoint_calculator import (
            AcidGasDewpointCalculator,
        )

        calculator = AcidGasDewpointCalculator()
        vp = calculator.calculate_vapor_pressure(100, "H2O")

        # At 100°C, water vapor pressure should be close to 101325 Pa (1 atm)
        assert 95000 < vp < 105000

    def test_quick_calculation_function(self) -> None:
        """Test the quick calculation convenience function."""
        from upstream_drift_tools.process_calculators.acid_gas_dewpoint_calculator import (
            quick_dewpoint_calculation,
        )

        result = quick_dewpoint_calculation(
            temperature_c=150,
            pressure_bar=30,
            h2o_fraction=0.15,
            hf_fraction=0.001,
            hcl_fraction=0.002,
            h2s_fraction=0.005,
        )

        assert "overall_dewpoint_c" in result
        assert "limiting_component" in result
        assert "condensation_risk" in result
        assert "dewpoint_margin_c" in result


class TestAcidGasDewpointGUI:
    """Tests for the PyQt6 GUI components."""

    @pytest.fixture
    def widget(self) -> MagicMock:
        """Create a mock widget for testing."""
        os.environ["HEADLESS"] = "true"

        with patch.dict("sys.modules", {"PyQt6.QtWidgets": MagicMock()}):
            mock_widget = MagicMock()
            mock_widget.composition_spins = {
                "H2O": MagicMock(value=MagicMock(return_value=15.0)),
                "HF": MagicMock(value=MagicMock(return_value=0.01)),
                "HCl": MagicMock(value=MagicMock(return_value=0.02)),
                "H2S": MagicMock(value=MagicMock(return_value=0.1)),
            }
            mock_widget.temp_spin = MagicMock(value=MagicMock(return_value=150.0))
            mock_widget.pressure_spin = MagicMock(value=MagicMock(return_value=30.0))
            return mock_widget

    def test_preset_compositions_defined(self) -> None:
        """Test that preset compositions are available."""
        os.environ["HEADLESS"] = "true"

        try:
            from acid_gas_dewpoint.python.acid_gas_dewpoint.ui.pyqt6.main_window import (
                AcidGasDewpointCalculatorWidget,
            )

            presets = AcidGasDewpointCalculatorWidget.PRESET_COMPOSITIONS
            assert "Typical Syngas" in presets
            assert "High Acid Content" in presets
            assert "Coal Gasification" in presets
            assert "Biomass Gasification" in presets
        except ImportError:
            pytest.skip("PyQt6 not available")

    def test_methods_defined(self) -> None:
        """Test that calculation methods are defined."""
        os.environ["HEADLESS"] = "true"

        try:
            from acid_gas_dewpoint.python.acid_gas_dewpoint.ui.pyqt6.main_window import (
                AcidGasDewpointCalculatorWidget,
            )

            methods = AcidGasDewpointCalculatorWidget.METHODS
            assert "antoine" in methods
            assert "extended_antoine" in methods
        except ImportError:
            pytest.skip("PyQt6 not available")


class TestResultDisplay:
    """Tests for result display formatting."""

    def test_result_to_dict(self) -> None:
        """Test that results can be converted to dictionary."""
        from upstream_drift_tools.process_calculators.acid_gas_dewpoint_calculator import (
            AcidGasComposition,
            AcidGasDewpointCalculator,
        )

        calculator = AcidGasDewpointCalculator()
        composition = AcidGasComposition(h2o=0.15, hf=0.001, hcl=0.002, h2s=0.005)

        result = calculator.calculate_dewpoint_mixture(
            temperature_c=150, pressure_bar=30, composition=composition
        )

        result_dict = result.to_dict()

        assert "timestamp" in result_dict
        assert "input" in result_dict
        assert "dewpoints" in result_dict
        assert "vapor_pressures_pa" in result_dict
        assert "safety" in result_dict
        assert "method" in result_dict
        assert "sources" in result_dict
        assert "warnings" in result_dict

    def test_safety_assessment(self) -> None:
        """Test safety assessment in results."""
        from upstream_drift_tools.process_calculators.acid_gas_dewpoint_calculator import (
            AcidGasComposition,
            AcidGasDewpointCalculator,
        )

        calculator = AcidGasDewpointCalculator()

        # High temperature should give low risk
        composition = AcidGasComposition(h2o=0.15, hf=0.001, hcl=0.002, h2s=0.005)
        result = calculator.calculate_dewpoint_mixture(
            temperature_c=200, pressure_bar=30, composition=composition
        )

        assert (
            "LOW" in result.condensation_risk or "VERY LOW" in result.condensation_risk
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
