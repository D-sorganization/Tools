"""Tests for WGS Reactor Calculator GUI module."""

from __future__ import annotations

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


class TestWGSReactorEngine:
    """Tests for the WGS reactor engine functions."""

    def test_equilibrium_constant_hts(self) -> None:
        """Test equilibrium constant at HTS temperature."""
        from wgs_reactor.ui.pyqt6.main_window import WGSReactorEngine

        engine = WGSReactorEngine()
        # At 400°C (673.15 K), K should be relatively low (< 100)
        K_eq = engine.calculate_equilibrium_constant(673.15)
        assert K_eq > 0
        assert K_eq < 100

    def test_equilibrium_constant_lts(self) -> None:
        """Test equilibrium constant at LTS temperature."""
        from wgs_reactor.ui.pyqt6.main_window import WGSReactorEngine

        engine = WGSReactorEngine()
        # At 220°C (493.15 K), K should be higher
        K_eq = engine.calculate_equilibrium_constant(493.15)
        assert K_eq > 0
        # LTS has higher K than HTS (more favorable for CO conversion)
        K_hts = engine.calculate_equilibrium_constant(673.15)
        assert K_eq > K_hts

    def test_equilibrium_composition_typical_syngas(self) -> None:
        """Test equilibrium composition with typical syngas."""
        from wgs_reactor.ui.pyqt6.main_window import WGSReactorEngine

        engine = WGSReactorEngine()
        inlet = {"CO": 25.0, "H2": 20.0, "CO2": 10.0, "H2O": 5.0}

        result = engine.calculate_equilibrium_composition(
            inlet, temperature=673.15, pressure=25.0, steam_ratio=2.0
        )

        assert "conversion" in result
        assert "composition" in result
        assert "h2_co_ratio" in result
        assert "equilibrium_constant" in result
        assert "heat_released" in result

        # Conversion should be positive
        assert result["conversion"] >= 0
        assert result["conversion"] <= 100

        # Product composition should sum to ~100%
        total = sum(result["composition"].values())
        assert total == pytest.approx(100.0, rel=0.01)

    def test_equilibrium_composition_zero_input(self) -> None:
        """Test equilibrium composition with zero input."""
        from wgs_reactor.ui.pyqt6.main_window import WGSReactorEngine

        engine = WGSReactorEngine()
        inlet = {"CO": 0.0, "H2": 0.0, "CO2": 0.0, "H2O": 0.0}

        result = engine.calculate_equilibrium_composition(
            inlet, temperature=673.15, pressure=25.0, steam_ratio=2.0
        )

        assert result["conversion"] == pytest.approx(0.0)

    def test_size_reactor(self) -> None:
        """Test reactor sizing."""
        from wgs_reactor.ui.pyqt6.main_window import WGSReactorEngine

        engine = WGSReactorEngine()
        sizing = engine.size_reactor(
            feed_rate=100.0, conversion=75.0, temperature=673.15
        )

        assert "reactor_volume" in sizing
        assert "catalyst_volume" in sizing
        assert "diameter" in sizing
        assert "length" in sizing
        assert "heat_duty" in sizing
        assert "ghsv" in sizing

        # Reactor volume should be positive
        assert sizing["reactor_volume"] > 0
        # Catalyst volume < reactor volume
        assert sizing["catalyst_volume"] < sizing["reactor_volume"]
        # L/D ratio should be ~3
        assert sizing["length"] == pytest.approx(sizing["diameter"] * 3, rel=0.1)


class TestWGSReactorPyQt6:
    """Tests for WGS Reactor Calculator PyQt6 GUI."""

    def test_main_window_import(self) -> None:
        """Test that main window can be imported."""
        from wgs_reactor.ui.pyqt6.main_window import WGSReactorWindow

        assert WGSReactorWindow is not None

    def test_main_window_creation(self) -> None:
        """Test main window creation with mocked Qt."""
        with patch.dict(sys.modules, {"PyQt6.QtWidgets": MagicMock()}):
            from wgs_reactor.ui.pyqt6.main_window import get_stylesheet

            stylesheet = get_stylesheet()
            assert "QMainWindow" in stylesheet
            assert "#1e1e2e" in stylesheet  # Catppuccin base color

    def test_result_card_class(self) -> None:
        """Test ResultCard class exists."""
        from wgs_reactor.ui.pyqt6.main_window import ResultCard

        assert ResultCard is not None

    def test_engine_class(self) -> None:
        """Test WGSReactorEngine class exists."""
        from wgs_reactor.ui.pyqt6.main_window import WGSReactorEngine

        assert WGSReactorEngine is not None


class TestWGSReactorLaunchers:
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


class TestWGSReactorWebApp:
    """Tests for web application files."""

    def test_web_directory_structure(self) -> None:
        """Test web directory structure exists."""
        web_dir = MODULE_DIR / "web"
        assert web_dir.exists()
        assert (web_dir / "package.json").exists()
        assert (web_dir / "src" / "App.tsx").exists()
        assert (web_dir / "src" / "components" / "WGSReactorCalculator.tsx").exists()

    def test_package_json_valid(self) -> None:
        """Test package.json is valid JSON."""
        import json

        web_dir = MODULE_DIR / "web"
        package_json = web_dir / "package.json"
        with open(package_json) as f:
            data = json.load(f)
        assert "name" in data
        assert data["name"] == "wgs-reactor-calculator"
        assert "dependencies" in data
        assert "react" in data["dependencies"]
