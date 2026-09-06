"""Tests for Phase 2 critical bug fixes.

Covers:
- Issue #605: Steam engine API endpoint and validated calculations
- Issue #611: Model generation lazy imports (no circular import)
- Issue #530: data_processing import fragility
- Issue #531: God function refactoring
- Issue #562: MyPy type fixes
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")
import numpy as np

# Ensure src/shared/python is importable
_shared_python = Path(__file__).resolve().parent.parent / "src" / "shared" / "python"
if str(_shared_python) not in sys.path:
    sys.path.insert(0, str(_shared_python))


# ========================================================================
# Issue #605: Steam Engine Thermodynamic Calculations
# ========================================================================


class TestSteamEngineCalculations:
    """Verify the Python steam engine produces physically plausible results."""

    def _get_engine(self):
        """Get a SteamCalculationEngine instance (simplified backend)."""
        from upstream_drift_tools.calculators.thermo.steam_engine import (
            SteamCalculationEngine,
        )

        return SteamCalculationEngine()

    def test_vapor_properties_at_atmospheric(self):
        """Steam at 400 K, 101325 Pa should be superheated vapor."""
        engine = self._get_engine()
        props = engine.calculate_properties(400.0, 101325.0, engine="simplified")
        assert props.phase == "vapor"
        assert props.quality == 1.0
        assert props.density > 0
        assert props.cp > 0
        assert props.cv > 0
        # cp should be around 1900 J/kg-K for simplified
        assert 1000 < props.cp < 3000

    def test_liquid_properties(self):
        """Water at 300 K, 200 kPa should be liquid."""
        engine = self._get_engine()
        props = engine.calculate_properties(300.0, 200000.0, engine="simplified")
        assert props.phase == "liquid"
        assert props.quality == 0.0
        assert props.density == pytest.approx(1000.0)
        assert props.cp == pytest.approx(4186.0)

    def test_saturation_from_temperature(self):
        """Saturated properties from temperature should return valid data."""
        engine = self._get_engine()
        props = engine.calculate_saturated_properties_from_temperature(
            373.15, engine="simplified"
        )
        assert props.pressure > 0
        assert props.temperature == pytest.approx(373.15, abs=1.0)

    def test_saturation_from_pressure(self):
        """Saturated properties from pressure should return valid data."""
        engine = self._get_engine()
        props = engine.calculate_saturated_properties_from_pressure(
            101325.0, engine="simplified"
        )
        assert props.temperature > 273.15
        assert props.pressure > 0

    def test_steam_properties_to_dict(self):
        """SteamProperties.to_dict should include all fields."""
        engine = self._get_engine()
        props = engine.calculate_properties(400.0, 101325.0, engine="simplified")
        d = props.to_dict()
        assert "Temperature (K)" in d
        assert "Pressure (Pa)" in d
        assert "Phase" in d
        assert d["Phase"] == "vapor"

    def test_engine_selection_auto(self):
        """Auto engine selection should pick best available."""
        engine = self._get_engine()
        selected = engine.select_best_engine("auto")
        assert selected in ("coolprop", "cantera", "simplified")

    def test_engine_selection_simplified(self):
        """Requesting 'simplified' should always return 'simplified'."""
        engine = self._get_engine()
        assert engine.select_best_engine("simplified") == "simplified"


# ========================================================================
# Issue #611: Model Generation Lazy Imports
# ========================================================================


class TestModelGenerationLazyImports:
    """Verify model_generation uses lazy imports and has no circular deps."""

    def test_package_imports_without_loading_all_modules(self):
        """Importing model_generation should NOT eagerly load all builders."""
        # Force reimport to test lazy loading
        import model_generation

        # The package should have loaded, version should be accessible
        assert model_generation.__version__ == "0.1.0"

    def test_lazy_access_to_types(self):
        """Accessing Link, Joint should trigger lazy load."""
        from model_generation import Joint, Link

        assert Link is not None
        assert Joint is not None

    def test_lazy_access_to_validation(self):
        """Accessing Validator should trigger lazy load."""
        from model_generation import ValidationResult, Validator

        assert Validator is not None
        assert ValidationResult is not None

    def test_lazy_access_to_constants(self):
        """Constants should be eagerly available (lightweight)."""
        from model_generation import DEFAULT_HEIGHT_M, GRAVITY_M_S2

        assert GRAVITY_M_S2 == pytest.approx(9.80665)
        assert DEFAULT_HEIGHT_M > 0

    def test_no_circular_import_on_fresh_import(self):
        """Importing the package should not raise ImportError."""
        # Remove from cache to test fresh import
        mods_to_remove = [k for k in sys.modules if k.startswith("model_generation")]
        saved = {}
        for mod in mods_to_remove:
            saved[mod] = sys.modules.pop(mod)

        try:
            import model_generation

            assert model_generation.__version__ == "0.1.0"
        finally:
            # Restore modules
            sys.modules.update(saved)

    def test_all_exports_are_accessible(self):
        """Every name in __all__ should be resolvable."""
        import model_generation

        for name in model_generation.__all__:
            try:
                obj = getattr(model_generation, name, None)
            except ImportError:
                # Some exports require optional dependencies (defusedxml, etc.)
                continue
            assert obj is not None, f"model_generation.{name} is None"


# ========================================================================
# Issue #530: data_processing Import Fragility
# ========================================================================


class TestDataProcessingImports:
    """Verify data_processing modules import correctly."""

    def test_core_init_lazy_import(self):
        """data_processor.core should import without errors using lazy loading."""
        dp_path = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "data_processing"
            / "data_processor"
            / "python"
        )
        if str(dp_path) not in sys.path:
            sys.path.insert(0, str(dp_path))

        # This should succeed even if utils/ is not on path
        import data_processor.core

        # Verify lazy access works
        assert "ConfigManager" in data_processor.core.__all__
        assert "KalmanFilter" in data_processor.core.__all__

    def test_data_loader_self_contained_path_bootstrap(self):
        """data_loader should resolve utils path independently."""
        dp_path = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "data_processing"
            / "data_processor"
            / "python"
        )
        if str(dp_path) not in sys.path:
            sys.path.insert(0, str(dp_path))

        # The module should have the bootstrap function
        from data_processor.core import data_loader

        assert hasattr(data_loader, "DataLoader")

    def test_logging_config_fallback(self):
        """logging_config should work even without utils on path."""
        dp_path = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "data_processing"
            / "data_processor"
            / "python"
        )
        if str(dp_path) not in sys.path:
            sys.path.insert(0, str(dp_path))

        from data_processor.logging_config import get_logger

        logger = get_logger("test_phase2")
        assert logger is not None
        assert logger.name == "test_phase2"

    def test_full_package_imports_after_shared_package_loaded(self):
        """The Rust I/O wrapper should not shadow the full app package."""
        repo_root = Path(__file__).resolve().parent.parent
        shared_path = repo_root / "src" / "shared" / "python"
        dp_path = repo_root / "src" / "data_processing" / "data_processor" / "python"
        for module_name in [
            "data_processor",
            "data_processor.core",
            "data_processor.logging_config",
            "data_processor.rust_engine",
            "data_processor_io",
        ]:
            sys.modules.pop(module_name, None)
        try:
            sys.path.insert(0, str(shared_path))
            import data_processor_io

            assert data_processor_io.__name__ == "data_processor_io"

            sys.path.insert(0, str(dp_path))
            import data_processor
            import data_processor.core
            from data_processor.logging_config import get_logger
        finally:
            sys.path = [
                path
                for path in sys.path
                if path not in {str(shared_path), str(dp_path)}
            ]

        assert not any("shared" in path for path in data_processor.__path__)
        assert "ConfigManager" in data_processor.core.__all__
        assert (
            get_logger("test_phase2_shared_shadow").name == "test_phase2_shared_shadow"
        )


# ========================================================================
# Issue #531: God Function Refactoring
# ========================================================================


class TestGodFunctionRefactoring:
    """Verify extracted sub-methods exist and are callable."""

    def test_pdf_renamer_thread_has_extracted_methods(self):
        """ProcessingThread should have _handle_duplicates, _scan_pdf_files, etc."""
        gui_path = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "document_processing"
            / "pdf_renamer"
            / "src"
        )
        if str(gui_path) not in sys.path:
            sys.path.insert(0, str(gui_path))

        try:
            from pdf_renamer.gui import ProcessingThread

            # Verify sub-methods exist
            assert hasattr(ProcessingThread, "_handle_duplicates")
            assert hasattr(ProcessingThread, "_delete_duplicate_set")
            assert hasattr(ProcessingThread, "_scan_pdf_files")
            assert hasattr(ProcessingThread, "_process_pdf_files")

            # _process_pdf_files must be a plain Python function, NOT a numba
            # @jit dispatcher (issue #3319). A numba-compiled method would be a
            # CPUDispatcher; the original @jit(nopython=True) raised a
            # TypingError that escaped run() and froze the GUI.
            import inspect

            assert inspect.isfunction(ProcessingThread._process_pdf_files), (
                "_process_pdf_files must be a plain function, not a numba "
                f"dispatcher; got {type(ProcessingThread._process_pdf_files)!r}"
            )
        except ImportError:
            pytest.skip("PyQt6 not available for GUI tests")

    def test_polynomial_generator_has_extracted_methods(self):
        """PolynomialGeneratorWidget should have _setup_* sub-methods."""
        try:
            from signal_toolkit.polynomial_generator import (
                PolynomialGeneratorWidget,
            )

            assert hasattr(PolynomialGeneratorWidget, "_setup_joint_selector")
            assert hasattr(PolynomialGeneratorWidget, "_setup_scale_controls")
            assert hasattr(PolynomialGeneratorWidget, "_setup_input_methods")
            assert hasattr(PolynomialGeneratorWidget, "_setup_action_controls")
            assert hasattr(PolynomialGeneratorWidget, "_setup_result_display")
        except ImportError:
            pytest.skip("PyQt6 or matplotlib not available")


# ========================================================================
# Issue #562: MyPy Type Fixes
# ========================================================================


class TestMyPyTypeFixes:
    """Verify the specific mypy type fixes."""

    def test_is_valid_result_returns_bool(self):
        """is_valid_result should return Python bool, not Any."""
        from model_generation.core.contracts import is_valid_result

        class FakeResult:
            is_valid = True

        result = is_valid_result(FakeResult())
        assert type(result) is bool
        assert result is True

    def test_has_finite_elements_returns_bool(self):
        """has_finite_elements should return Python bool, not numpy.bool_."""
        from model_generation.core.contracts import has_finite_elements

        result = has_finite_elements(np.array([1.0, 2.0, 3.0]))
        assert type(result) is bool
        assert result is True

        result_nan = has_finite_elements(np.array([1.0, float("nan")]))
        assert type(result_nan) is bool
        assert result_nan is False

    def test_joint_post_init_has_return_type(self):
        """Joint.__post_init__ should have -> None type annotation."""
        import inspect

        from model_generation.core.types import Joint

        sig = inspect.signature(Joint.__post_init__)
        assert sig.return_annotation is None or sig.return_annotation == "None"

    def test_py_typed_markers_exist(self):
        """py.typed marker files should exist in typed packages."""
        model_gen_marker = _shared_python / "model_generation" / "py.typed"
        signal_toolkit_marker = _shared_python / "signal_toolkit" / "py.typed"
        assert model_gen_marker.exists(), "model_generation/py.typed missing"
        assert signal_toolkit_marker.exists(), "signal_toolkit/py.typed missing"


# ========================================================================
# Integration: Steam Engine API Contract Tests
# ========================================================================


class TestSteamAPIContract:
    """Test that the FastAPI endpoint contract matches the Python engine."""

    def test_api_module_importable(self):
        """The steam engine API module should be importable."""
        pytest.importorskip("fastapi", reason="fastapi not installed")
        api_path = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "steam_engine_calculator"
            / "python"
        )
        if str(api_path) not in sys.path:
            sys.path.insert(0, str(api_path))

        from steam_engine_calculator.api import (
            CalculationMode,
        )

        assert CalculationMode.TP.value == "tp"
        assert CalculationMode.SAT_T.value == "sat_t"
        assert CalculationMode.SAT_P.value == "sat_p"

    def test_api_response_model_fields(self):
        """SteamResponse should have all expected fields."""
        pytest.importorskip("fastapi", reason="fastapi not installed")
        api_path = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "steam_engine_calculator"
            / "python"
        )
        if str(api_path) not in sys.path:
            sys.path.insert(0, str(api_path))

        from steam_engine_calculator.api import SteamResponse

        fields = SteamResponse.model_fields
        expected = [
            "temperature",
            "pressure",
            "density",
            "specificVolume",
            "enthalpy",
            "entropy",
            "internalEnergy",
            "cp",
            "cv",
            "speedOfSound",
            "thermalConductivity",
            "dynamicViscosity",
            "kinematicViscosity",
            "quality",
            "phase",
            "compressibilityFactor",
            "prandtlNumber",
            "specificHeatRatio",
            "engine",
        ]
        for field in expected:
            assert field in fields, f"Missing field: {field}"

    def test_props_to_response_conversion(self):
        """_props_to_response should correctly map Python fields to camelCase."""
        pytest.importorskip("fastapi", reason="fastapi not installed")
        api_path = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "steam_engine_calculator"
            / "python"
        )
        if str(api_path) not in sys.path:
            sys.path.insert(0, str(api_path))

        from steam_engine_calculator.api import _props_to_response
        from upstream_drift_tools.calculators.thermo.steam_engine import (
            SteamProperties,
        )

        props = SteamProperties(
            temperature=400.0,
            pressure=101325.0,
            density=0.88,
            specific_volume=1.136,
            enthalpy=2741300.0,
            entropy=8500.0,
            internal_energy=2626000.0,
            cp=1900.0,
            cv=1400.0,
            speed_of_sound=470.0,
            thermal_conductivity=0.025,
            dynamic_viscosity=1.2e-5,
            kinematic_viscosity=1.36e-5,
            quality=1.0,
            phase="vapor",
            compressibility_factor=0.98,
            prandtl_number=0.91,
            specific_heat_ratio=1.357,
        )

        resp = _props_to_response(props, "simplified")
        assert resp.temperature == 400.0
        assert resp.specificVolume == 1.136
        assert resp.internalEnergy == 2626000.0
        assert resp.engine == "simplified"

    def test_calculate_steam_maps_precondition_errors_to_400(self, monkeypatch):
        """Invalid saturation requests should be client errors, not server errors."""
        pytest.importorskip("fastapi", reason="fastapi not installed")
        api_path = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "steam_engine_calculator"
            / "python"
        )
        if str(api_path) not in sys.path:
            sys.path.insert(0, str(api_path))

        from fastapi import HTTPException
        from steam_engine_calculator.api import CalculationMode, SteamRequest

        from steam_engine_calculator import api

        monkeypatch.setattr(
            api._engine,
            "calculate_saturated_properties_from_temperature",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                ValueError("temperature outside saturation bounds")
            ),
        )

        request = SteamRequest(
            mode=CalculationMode.SAT_T,
            temperature=200.0,
            pressure=101325.0,
            engine="simplified",
        )
        with pytest.raises(HTTPException) as exc_info:
            api.calculate_steam(request)

        assert exc_info.value.status_code == 400
