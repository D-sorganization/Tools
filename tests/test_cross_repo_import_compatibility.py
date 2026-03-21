"""Cross-repo import compatibility tests for UpstreamDrift and Gasification_Model.

These tests verify that all public API symbols from the ``upstream_drift_tools``
package are importable and structurally sound, simulating what UpstreamDrift and
Gasification_Model do at import time. A failure here means a downstream repo
would break on ``pip install ud-tools && import upstream_drift_tools``.

Marker: integration — run with ``pytest -m integration``
"""

from __future__ import annotations

import pytest


@pytest.mark.integration
class TestUpstreamDriftToolsTopLevel:
    """Contract: upstream_drift_tools top-level namespace is importable and
    exports all documented symbols."""

    def test_package_imports_without_error(self) -> None:
        """Downstream repos must be able to import the package without error."""
        import upstream_drift_tools  # noqa: F401

    def test_package_version_is_present(self) -> None:
        """__version__ must be a non-empty string for packaging tooling."""
        import upstream_drift_tools

        assert hasattr(upstream_drift_tools, "__version__")
        assert isinstance(upstream_drift_tools.__version__, str)
        assert len(upstream_drift_tools.__version__) > 0

    def test_all_direct_exports_are_importable(self) -> None:
        """Every explicitly imported symbol in __all__ must be accessible on the module.

        Subpackage names (e.g. 'calculators', 'data_processing') are excluded because
        they are lazy — only bound after an explicit subpackage import. This matches
        how downstream repos use the package: they import symbols directly, not via
        hasattr() on the parent module.
        """
        import importlib

        import upstream_drift_tools

        # Subpackage names listed in __all__ for documentation only; they are not
        # bound as module-level attributes until explicitly imported.
        _subpackage_names = {
            "calculators",
            "data_processing",
            "lab",
            "process_calculators",
            "theme",
            "ui",
            "utils",
        }

        for symbol in upstream_drift_tools.__all__:
            if symbol in _subpackage_names:
                # Verify subpackages are importable as ``upstream_drift_tools.<name>``
                mod_name = f"upstream_drift_tools.{symbol}"
                try:
                    importlib.import_module(mod_name)
                except ImportError as exc:
                    pytest.fail(
                        f"Subpackage '{mod_name}' listed in __all__ is not importable: {exc}"
                    )
            else:
                assert hasattr(upstream_drift_tools, symbol), (
                    f"Symbol '{symbol}' listed in __all__ is not accessible "
                    f"on the upstream_drift_tools module"
                )


@pytest.mark.integration
class TestProtocolsContract:
    """Contract: Protocol interfaces are importable and runtime-checkable.

    UpstreamDrift and Gasification_Model use isinstance() checks against
    these protocols — they must remain @runtime_checkable.
    """

    def test_calculator_protocol_importable(self) -> None:
        """Calculator protocol must be importable from top-level."""
        from upstream_drift_tools import Calculator

        assert Calculator is not None

    def test_process_calculator_protocol_importable(self) -> None:
        """ProcessCalculator protocol must be importable from top-level."""
        from upstream_drift_tools import ProcessCalculator

        assert ProcessCalculator is not None

    def test_data_transformer_protocol_importable(self) -> None:
        """DataTransformer protocol must be importable from top-level."""
        from upstream_drift_tools import DataTransformer

        assert DataTransformer is not None

    def test_state_serializable_protocol_importable(self) -> None:
        """StateSerializable protocol must be importable from top-level."""
        from upstream_drift_tools import StateSerializable

        assert StateSerializable is not None

    def test_unit_converter_protocol_importable(self) -> None:
        """UnitConverter protocol must be importable from top-level."""
        from upstream_drift_tools import UnitConverter

        assert UnitConverter is not None

    def test_protocols_are_runtime_checkable(self) -> None:
        """Protocols must support isinstance() checks (runtime_checkable).

        This is required for downstream code that does:
            if isinstance(obj, Calculator): ...
        """
        from upstream_drift_tools import (
            Calculator,
            DataTransformer,
            ProcessCalculator,
            StateSerializable,
            UnitConverter,
        )

        # An object with no matching methods is NOT an instance
        plain = object()
        for proto in (
            Calculator,
            ProcessCalculator,
            DataTransformer,
            StateSerializable,
            UnitConverter,
        ):
            # Should not raise TypeError — protocols must be runtime-checkable
            result = isinstance(plain, proto)
            assert isinstance(result, bool)


@pytest.mark.integration
class TestCalculationDataClasses:
    """Contract: CalculationResult and ValidationResult data classes have correct
    fields and defaults.

    Downstream repos unpack these objects by field name — field names and
    defaults are part of the stable API surface.
    """

    def test_calculation_result_importable(self) -> None:
        """CalculationResult must be importable from top-level."""
        from upstream_drift_tools import CalculationResult

        assert CalculationResult is not None

    def test_calculation_result_default_fields(self) -> None:
        """CalculationResult default instance must have correct field structure."""
        from upstream_drift_tools import CalculationResult

        result = CalculationResult()
        assert isinstance(result.values, dict)
        assert isinstance(result.units, dict)
        assert isinstance(result.warnings, list)
        assert isinstance(result.metadata, dict)
        assert result.values == {}
        assert result.units == {}
        assert result.warnings == []
        assert result.metadata == {}

    def test_calculation_result_accepts_values(self) -> None:
        """CalculationResult must accept field values at construction."""
        from upstream_drift_tools import CalculationResult

        result = CalculationResult(
            values={"pressure": 101325.0},
            units={"pressure": "Pa"},
            warnings=["high temperature"],
            metadata={"solver": "analytical"},
        )
        assert result.values["pressure"] == pytest.approx(101325.0)
        assert result.units["pressure"] == "Pa"
        assert result.warnings == ["high temperature"]
        assert result.metadata["solver"] == "analytical"

    def test_validation_result_importable(self) -> None:
        """ValidationResult must be importable from top-level."""
        from upstream_drift_tools import ValidationResult

        assert ValidationResult is not None

    def test_validation_result_default_fields(self) -> None:
        """ValidationResult default instance must have correct field structure."""
        from upstream_drift_tools import ValidationResult

        result = ValidationResult()
        assert result.valid is True
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)
        assert result.errors == []
        assert result.warnings == []

    def test_validation_result_invalid(self) -> None:
        """ValidationResult must accept failure state."""
        from upstream_drift_tools import ValidationResult

        result = ValidationResult(
            valid=False,
            errors=["flow_rate must be positive"],
            warnings=["temperature approaching limit"],
        )
        assert result.valid is False
        assert len(result.errors) == 1
        assert len(result.warnings) == 1


@pytest.mark.integration
class TestCalculatorsSubpackage:
    """Contract: upstream_drift_tools.calculators subpackage is importable
    and exports BaseCalculationEngine."""

    def test_calculators_subpackage_importable(self) -> None:
        """Calculators subpackage must import without error."""
        from upstream_drift_tools import calculators  # noqa: F401

    def test_base_calculation_engine_importable(self) -> None:
        """BaseCalculationEngine must be importable from calculators subpackage."""
        from upstream_drift_tools.calculators import BaseCalculationEngine

        assert BaseCalculationEngine is not None

    def test_base_calculation_engine_is_class(self) -> None:
        """BaseCalculationEngine must be a class (not a function or constant)."""
        from upstream_drift_tools.calculators import BaseCalculationEngine

        assert isinstance(BaseCalculationEngine, type)


@pytest.mark.integration
class TestDataProcessingSubpackage:
    """Contract: upstream_drift_tools.data_processing subpackage is importable
    and exports its documented symbols."""

    def test_data_processing_subpackage_importable(self) -> None:
        """data_processing subpackage must import without error."""
        from upstream_drift_tools import data_processing  # noqa: F401

    def test_data_processor_engine_importable(self) -> None:
        """DataProcessorEngine must be importable from data_processing subpackage."""
        from upstream_drift_tools.data_processing import DataProcessorEngine

        assert DataProcessorEngine is not None

    def test_processing_result_importable(self) -> None:
        """ProcessingResult must be importable from data_processing subpackage."""
        from upstream_drift_tools.data_processing import ProcessingResult

        assert ProcessingResult is not None

    def test_data_reader_writer_importable(self) -> None:
        """DataReader and DataWriter must be importable from data_processing."""
        from upstream_drift_tools.data_processing import DataReader, DataWriter

        assert DataReader is not None
        assert DataWriter is not None

    def test_exception_hierarchy_importable(self) -> None:
        """All custom exceptions must be importable — downstream code catches them."""
        from upstream_drift_tools.data_processing import (
            ColumnNotFoundError,
            DataNotLoadedError,
            DataProcessingError,
            FilterError,
            FitError,
            TransformationError,
            UnsupportedOperationError,
        )

        for exc_cls in (
            DataProcessingError,
            DataNotLoadedError,
            ColumnNotFoundError,
            TransformationError,
            FilterError,
            FitError,
            UnsupportedOperationError,
        ):
            assert issubclass(exc_cls, Exception)


@pytest.mark.integration
class TestInputValidatorContract:
    """Contract: InputValidator static methods match the expected API surface.

    Downstream calculators call these methods to validate user input — the
    method signatures and exception types are part of the stable API.
    """

    def test_input_validator_importable(self) -> None:
        """InputValidator must be importable from top-level."""
        from upstream_drift_tools import InputValidator

        assert InputValidator is not None

    def test_require_positive_accepts_valid_value(self) -> None:
        """require_positive must not raise for a positive value."""
        from upstream_drift_tools import InputValidator

        InputValidator.require_positive("flow_rate", 1.5)  # must not raise

    def test_require_positive_rejects_zero(self) -> None:
        """require_positive must raise ValueError for zero."""
        from upstream_drift_tools import InputValidator

        with pytest.raises(ValueError, match="flow_rate"):
            InputValidator.require_positive("flow_rate", 0.0)

    def test_require_positive_rejects_negative(self) -> None:
        """require_positive must raise ValueError for negative values."""
        from upstream_drift_tools import InputValidator

        with pytest.raises(ValueError):
            InputValidator.require_positive("pressure", -1.0)

    def test_require_in_range_accepts_valid(self) -> None:
        """require_in_range must not raise when value is within bounds."""
        from upstream_drift_tools import InputValidator

        InputValidator.require_in_range("efficiency", 0.85, 0.0, 1.0)

    def test_require_in_range_rejects_out_of_bounds(self) -> None:
        """require_in_range must raise ValueError when value is outside [low, high]."""
        from upstream_drift_tools import InputValidator

        with pytest.raises(ValueError):
            InputValidator.require_in_range("efficiency", 1.5, 0.0, 1.0)

    def test_require_keys_accepts_complete_dict(self) -> None:
        """require_keys must not raise when all required keys are present."""
        from upstream_drift_tools import InputValidator

        InputValidator.require_keys(
            {"temperature": 300.0, "pressure": 101325.0},
            {"temperature", "pressure"},
        )

    def test_require_keys_rejects_missing_keys(self) -> None:
        """require_keys must raise ValueError when required keys are absent."""
        from upstream_drift_tools import InputValidator

        with pytest.raises(ValueError, match="Missing"):
            InputValidator.require_keys(
                {"temperature": 300.0}, {"temperature", "pressure"}
            )

    def test_validate_temperature_accepts_positive(self) -> None:
        """validate_temperature must accept temperatures above 0 K."""
        from upstream_drift_tools import InputValidator

        InputValidator.validate_temperature(298.15)  # must not raise

    def test_validate_temperature_rejects_zero(self) -> None:
        """validate_temperature must raise ValueError for 0 K (absolute zero)."""
        from upstream_drift_tools import InputValidator

        with pytest.raises(ValueError):
            InputValidator.validate_temperature(0.0)

    def test_validate_pressure_accepts_positive(self) -> None:
        """validate_pressure must accept pressures above 0 Pa."""
        from upstream_drift_tools import InputValidator

        InputValidator.validate_pressure(101325.0)  # must not raise

    def test_validate_pressure_rejects_zero(self) -> None:
        """validate_pressure must raise ValueError for 0 Pa."""
        from upstream_drift_tools import InputValidator

        with pytest.raises(ValueError):
            InputValidator.validate_pressure(0.0)

    def test_validate_composition_accepts_valid(self) -> None:
        """validate_composition must accept fractions that sum to 1.0."""
        from upstream_drift_tools import InputValidator

        InputValidator.validate_composition({"CH4": 0.6, "CO2": 0.3, "N2": 0.1})

    def test_validate_composition_rejects_invalid_sum(self) -> None:
        """validate_composition must raise ValueError when fractions do not sum to 1."""
        from upstream_drift_tools import InputValidator

        with pytest.raises(ValueError):
            InputValidator.validate_composition({"CH4": 0.5, "CO2": 0.3})

    def test_validate_composition_rejects_negative_fraction(self) -> None:
        """validate_composition must raise ValueError for negative fractions."""
        from upstream_drift_tools import InputValidator

        with pytest.raises(ValueError):
            InputValidator.validate_composition({"CH4": -0.1, "CO2": 1.1})
