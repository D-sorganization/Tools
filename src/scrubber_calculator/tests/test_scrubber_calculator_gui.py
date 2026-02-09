"""Tests for Scrubber Calculator GUI module."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Bootstrap for test discovery
_REPO_ROOT = Path(__file__).resolve().parents[3]
import sys

sys.path.insert(0, str(_REPO_ROOT / "src" / "shared" / "python"))
from upstream_drift_tools.bootstrap import ensure_paths  # noqa: E402

ensure_paths(_REPO_ROOT)

MODULE_DIR = Path(__file__).parent.parent

# Directly load the scrubber_calculator module to avoid import issues
# in the shared package's __init__.py
SCRUBBER_MODULE_PATH = (
    _REPO_ROOT
    / "src"
    / "shared"
    / "python"
    / "upstream_drift_tools"
    / "process_calculators"
    / "scrubber_calculator.py"
)


def load_scrubber_module():
    """Load scrubber calculator module directly."""
    spec = importlib.util.spec_from_file_location(
        "scrubber_calculator", SCRUBBER_MODULE_PATH
    )
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return module
    except ImportError:
        return None


# Try to load the module
scrubber_module = load_scrubber_module()
ENGINE_AVAILABLE = scrubber_module is not None


@pytest.mark.skipif(not ENGINE_AVAILABLE, reason="Scrubber engine not available")
class TestScrubberCalculatorEngine:
    """Tests for the scrubber calculator engine functions."""

    def test_packing_database_exists(self) -> None:
        """Test that packing database is available."""
        assert scrubber_module is not None
        assert len(scrubber_module.PACKING_DATABASE) >= 4
        assert "Metal Pall Rings" in scrubber_module.PACKING_DATABASE
        assert "Ceramic Raschig Rings" in scrubber_module.PACKING_DATABASE

    def test_packing_properties(self) -> None:
        """Test packing properties are valid."""
        assert scrubber_module is not None
        packing = scrubber_module.PACKING_DATABASE["Metal Pall Rings"]
        assert packing.specific_surface_area > 0
        assert 0 < packing.void_fraction < 1
        assert packing.packing_factor > 0
        assert packing.c_flood > 0

    def test_calculate_gas_density(self) -> None:
        """Test gas density calculation."""
        assert scrubber_module is not None
        # At 473.15 K (200°C), 1.5 bar, MW=22
        density = scrubber_module.calculate_gas_density(473.15, 150000.0, 22.0)
        assert density > 0
        assert 0.5 < density < 2.0  # Reasonable range for syngas

    def test_calculate_flooding_velocity(self) -> None:
        """Test flooding velocity calculation."""
        assert scrubber_module is not None
        packing = scrubber_module.PACKING_DATABASE["Metal Pall Rings"]
        u_flood = scrubber_module.calculate_flooding_velocity(
            liquid_mass_flux=5.0,  # kg/(m²·s)
            gas_density=1.0,  # kg/m³
            liquid_density=1100.0,  # kg/m³
            packing=packing,
        )
        assert u_flood > 0
        assert 1.0 < u_flood < 10.0  # Typical range

    def test_calculate_ntu_removal(self) -> None:
        """Test NTU calculation for removal."""
        assert scrubber_module is not None
        # 99% removal: inlet = 1000 ppmv, outlet = 10 ppmv
        ntu = scrubber_module.calculate_ntu_removal(0.001, 0.00001)
        assert ntu > 0
        assert 4.0 < ntu < 5.0  # ln(100) ≈ 4.6

    def test_calculate_ntu_removal_invalid(self) -> None:
        """Test NTU calculation with invalid inputs."""
        assert scrubber_module is not None
        assert scrubber_module.calculate_ntu_removal(0, 0.001) == pytest.approx(0.0)
        assert scrubber_module.calculate_ntu_removal(0.001, 0) == pytest.approx(0.0)
        assert scrubber_module.calculate_ntu_removal(0.001, 0.001) == pytest.approx(0.0)

    def test_calculate_htu(self) -> None:
        """Test HTU calculation."""
        assert scrubber_module is not None
        packing = scrubber_module.PACKING_DATABASE["Metal Pall Rings"]
        htu = scrubber_module.calculate_htu(
            gas_mass_flux=2.0,
            liquid_mass_flux=6.0,
            gas_density=1.0,
            packing=packing,
            kla=200.0,
        )
        assert 0.1 <= htu <= 3.0  # Clamped range

    def test_calculate_required_packed_height(self) -> None:
        """Test packed height calculation."""
        assert scrubber_module is not None
        height = scrubber_module.calculate_required_packed_height(
            ntu=4.6, htu=0.5, safety_factor=1.2
        )
        assert height == pytest.approx(4.6 * 0.5 * 1.2, rel=0.01)

    def test_calculate_pressure_drop(self) -> None:
        """Test pressure drop calculation."""
        assert scrubber_module is not None
        packing = scrubber_module.PACKING_DATABASE["Metal Pall Rings"]
        dp = scrubber_module.calculate_pressure_drop(
            gas_velocity=2.0,
            gas_density=1.0,
            liquid_mass_flux=5.0,
            liquid_density=1100.0,
            packing=packing,
            packed_height=3.0,
        )
        assert dp > 0
        assert dp < 10000  # Less than 10 kPa for 3m height

    def test_calculate_caustic_requirement(self) -> None:
        """Test caustic requirement calculation."""
        assert scrubber_module is not None
        result = scrubber_module.calculate_caustic_requirement(
            acid_gas_removed={"HCl": 1.0},  # 1 kg/hr HCl
            caustic_concentration=20.0,
        )
        assert "naoh_pure_kg_hr" in result
        assert result["naoh_pure_kg_hr"] > 0
        assert "naoh_solution_L_hr" in result

    def test_calculate_heat_transfer_duty(self) -> None:
        """Test heat transfer duty calculation."""
        assert scrubber_module is not None
        result = scrubber_module.calculate_heat_transfer_duty(
            gas_flow_kg_hr=10000.0,
            inlet_temp_c=200.0,
            outlet_temp_c=38.0,
            water_condensed_kg_hr=100.0,
        )
        assert "total_heat_kw" in result
        assert result["total_heat_kw"] > 0
        assert result["sensible_heat_kw"] > 0

    def test_calculate_cooling_water_requirement(self) -> None:
        """Test cooling water requirement calculation."""
        assert scrubber_module is not None
        result = scrubber_module.calculate_cooling_water_requirement(
            heat_duty_kw=500.0,
            water_inlet_temp_c=25.0,
            outlet_gas_temp_c=38.0,
        )
        assert "water_flow_L_min" in result
        assert result["water_flow_L_min"] > 0

    def test_calculate_column_diameter(self) -> None:
        """Test column diameter calculation."""
        assert scrubber_module is not None
        result = scrubber_module.calculate_column_diameter(
            gas_flow_kg_hr=10000.0,
            gas_density=1.0,
            flooding_velocity=3.0,
            percent_of_flood=70.0,
        )
        assert "diameter_m" in result
        assert result["diameter_m"] > 0
        assert "cross_section_m2" in result


class TestScrubberCalculatorPyQt6:
    """Tests for Scrubber Calculator PyQt6 GUI."""

    def test_main_window_import(self) -> None:
        """Test that main window can be imported."""
        from scrubber_calculator.ui.pyqt6.main_window import ScrubberCalculatorWindow

        assert ScrubberCalculatorWindow is not None

    def test_main_window_creation(self) -> None:
        """Test main window creation with mocked Qt."""
        with patch.dict(sys.modules, {"PyQt6.QtWidgets": MagicMock()}):
            from scrubber_calculator.ui.pyqt6.main_window import (
                get_stylesheet,
            )

            stylesheet = get_stylesheet()
            assert "QMainWindow" in stylesheet
            assert "#1e1e2e" in stylesheet  # Catppuccin base color

    def test_result_card_class(self) -> None:
        """Test ResultCard class exists."""
        from scrubber_calculator.ui.pyqt6.main_window import ResultCard

        assert ResultCard is not None


class TestScrubberCalculatorLaunchers:
    """Tests for launcher scripts."""

    def test_launch_pyqt6_exists(self) -> None:
        """Test PyQt6 launcher exists."""
        launcher = MODULE_DIR / "launch_pyqt6.py"
        assert launcher.exists()

    def test_launch_web_exists(self) -> None:
        """Test web launcher exists."""
        launcher = MODULE_DIR / "launch_web.py"
        assert launcher.exists()

    def test_gui_registration_exists(self) -> None:
        """Test GUI registration exists."""
        registration = MODULE_DIR / "gui_registration.py"
        assert registration.exists()


class TestScrubberCalculatorWebApp:
    """Tests for web application files."""

    def test_web_directory_structure(self) -> None:
        """Test web directory structure exists."""
        web_dir = MODULE_DIR / "web"
        assert web_dir.exists()
        assert (web_dir / "package.json").exists()
        assert (web_dir / "src" / "App.tsx").exists()
        assert (web_dir / "src" / "components" / "ScrubberCalculator.tsx").exists()

    def test_package_json_valid(self) -> None:
        """Test package.json is valid JSON."""
        import json

        web_dir = MODULE_DIR / "web"
        package_json = web_dir / "package.json"
        with open(package_json) as f:
            data = json.load(f)
        assert "name" in data
        assert data["name"] == "scrubber-calculator"
        assert "dependencies" in data
        assert "react" in data["dependencies"]
