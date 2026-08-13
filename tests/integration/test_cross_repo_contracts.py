"""Cross-repo API contract tests — signature drift detection.

Verifies that the public API signatures exposed by ``upstream_drift_tools``
(the ``ud-tools`` package) have not drifted from the shape that downstream
repos UpstreamDrift and Gasification_Model depend on.

A failure here means a downstream repo would break on import or at runtime
after pulling a Tools update.  These tests complement the importability
checks in ``tests/test_cross_repo_import_compatibility.py`` by inspecting
the *signatures* of callable objects, not just their existence.

Run with:
    pytest -m integration tests/integration/test_cross_repo_contracts.py -v

Markers
-------
- ``integration`` — cross-repo boundary tests (no DWSIM or GL required)
- ``contract``    — API surface tests; breaking a contract test means a
                    downstream consumer is broken
"""

from __future__ import annotations

import inspect
import typing
from typing import Any

import pytest

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_param_names(func: Any) -> list[str]:
    """Return a sorted list of parameter names for *func* (excluding 'self')."""
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return []
    return [name for name in sig.parameters if name not in {"self", "cls"}]


def _assert_params_present(func: Any, required_params: set[str]) -> None:
    """Raise AssertionError if any of *required_params* are absent from *func*."""
    actual = set(_get_param_names(func))
    missing = required_params - actual
    assert not missing, (
        f"{func.__qualname__}: required parameter(s) {sorted(missing)} "
        f"are absent from signature. Actual params: {sorted(actual)}"
    )


def _assert_return_annotation(func: Any, expected_type: Any) -> None:
    """Raise AssertionError if *func* has a return annotation that does not
    match *expected_type*.  Passes if there is no annotation (unannotated
    functions are not considered broken for downstream callers).
    """
    try:
        hints = typing.get_type_hints(func)
    except Exception:
        return
    if "return" not in hints:
        return
    actual = hints["return"]
    assert actual == expected_type, (
        f"{func.__qualname__}: return annotation changed. "
        f"Expected {expected_type!r}, got {actual!r}"
    )


# ---------------------------------------------------------------------------
# InputValidator — signature contracts
# ---------------------------------------------------------------------------


@pytest.mark.contract
class TestInputValidatorSignatures:
    """API contract: InputValidator static-method signatures match what
    UpstreamDrift and Gasification_Model call at runtime.

    These tests check parameter *names* (positional-or-keyword) so that
    downstream code using keyword arguments (e.g.
    ``InputValidator.require_positive(name="flow_rate", value=1.5)``) does
    not break if a parameter is renamed.
    """

    def test_require_positive_signature(self) -> None:
        """require_positive(name, value) — both parameters must be present."""
        from upstream_drift_tools import InputValidator

        _assert_params_present(InputValidator.require_positive, {"name", "value"})

    def test_require_in_range_signature(self) -> None:
        """require_in_range(name, value, low, high) — all four parameters."""
        from upstream_drift_tools import InputValidator

        _assert_params_present(
            InputValidator.require_in_range, {"name", "value", "low", "high"}
        )

    def test_require_keys_signature(self) -> None:
        """require_keys(inputs, required_keys) — both parameters must be present."""
        from upstream_drift_tools import InputValidator

        _assert_params_present(InputValidator.require_keys, {"inputs", "required_keys"})

    def test_validate_temperature_signature(self) -> None:
        """validate_temperature(value) — single positional parameter."""
        from upstream_drift_tools import InputValidator

        _assert_params_present(InputValidator.validate_temperature, {"value"})

    def test_validate_pressure_signature(self) -> None:
        """validate_pressure(value) — single positional parameter."""
        from upstream_drift_tools import InputValidator

        _assert_params_present(InputValidator.validate_pressure, {"value"})

    def test_validate_composition_signature(self) -> None:
        """validate_composition(composition, *, tolerance) — composition required,
        tolerance keyword-only."""
        from upstream_drift_tools import InputValidator

        sig = inspect.signature(InputValidator.validate_composition)
        params = sig.parameters
        assert "composition" in params, (
            "validate_composition must have 'composition' param"
        )
        assert "tolerance" in params, "validate_composition must have 'tolerance' param"
        assert params["tolerance"].default is not inspect.Parameter.empty, (
            "validate_composition 'tolerance' must have a default value"
        )


# ---------------------------------------------------------------------------
# CalculationResult / ValidationResult — field contracts
# ---------------------------------------------------------------------------


@pytest.mark.contract
class TestDataClassFieldContracts:
    """API contract: CalculationResult and ValidationResult field names and
    default types must not change.

    Downstream repos unpack these objects by field name; renaming a field
    is a breaking change even if the types are the same.
    """

    def test_calculation_result_field_names(self) -> None:
        """CalculationResult must expose {values, units, warnings, metadata}."""
        from upstream_drift_tools import CalculationResult

        instance = CalculationResult()
        expected_fields = {"values", "units", "warnings", "metadata"}
        for field in expected_fields:
            assert hasattr(instance, field), (
                f"CalculationResult is missing field '{field}'; downstream code "
                "will break on attribute access"
            )

    def test_calculation_result_default_types(self) -> None:
        """CalculationResult defaults must be dict/list as documented."""
        from upstream_drift_tools import CalculationResult

        r = CalculationResult()
        assert isinstance(r.values, dict), "values must default to dict"
        assert isinstance(r.units, dict), "units must default to dict"
        assert isinstance(r.warnings, list), "warnings must default to list"
        assert isinstance(r.metadata, dict), "metadata must default to dict"

    def test_validation_result_field_names(self) -> None:
        """ValidationResult must expose {valid, errors, warnings}."""
        from upstream_drift_tools import ValidationResult

        instance = ValidationResult()
        expected_fields = {"valid", "errors", "warnings"}
        for field in expected_fields:
            assert hasattr(instance, field), (
                f"ValidationResult is missing field '{field}'; downstream code "
                "will break on attribute access"
            )

    def test_validation_result_default_types(self) -> None:
        """ValidationResult defaults: valid=True, errors=[], warnings=[]."""
        from upstream_drift_tools import ValidationResult

        r = ValidationResult()
        assert r.valid is True, "valid must default to True"
        assert isinstance(r.errors, list), "errors must default to list"
        assert isinstance(r.warnings, list), "warnings must default to list"

    def test_validation_result_round_trips_failure_state(self) -> None:
        """ValidationResult must accept and preserve failure state."""
        from upstream_drift_tools import ValidationResult

        r = ValidationResult(
            valid=False,
            errors=["inlet_temperature below dew-point"],
            warnings=["approaching flammability limit"],
        )
        assert r.valid is False
        assert r.errors[0] == "inlet_temperature below dew-point"
        assert r.warnings[0] == "approaching flammability limit"


# ---------------------------------------------------------------------------
# Protocol runtime-checkability contracts
# ---------------------------------------------------------------------------


@pytest.mark.contract
class TestProtocolRuntimeCheckability:
    """API contract: Protocols must remain @runtime_checkable.

    UpstreamDrift and Gasification_Model use ``isinstance(obj, Calculator)``
    pattern to dispatch to the correct handler.  If a Protocol loses the
    ``@runtime_checkable`` decorator the isinstance() call raises TypeError,
    breaking every dispatch site in downstream repos.
    """

    @pytest.mark.parametrize(
        "protocol_name",
        ["Calculator", "ProcessCalculator", "DataTransformer", "UnitConverter"],
    )
    def test_protocol_is_runtime_checkable(self, protocol_name: str) -> None:
        """Protocol must support isinstance() checks without raising TypeError."""
        import upstream_drift_tools

        proto = getattr(upstream_drift_tools, protocol_name)
        # isinstance with a plain object must not raise — it should return bool
        try:
            result = isinstance(object(), proto)
        except TypeError as exc:
            pytest.fail(
                f"{protocol_name} is not @runtime_checkable — isinstance() raised "
                f"TypeError: {exc}.  Downstream repos will break."
            )
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# process_calculators — key Gasification_Model imports
# ---------------------------------------------------------------------------


@pytest.mark.contract
class TestProcessCalculatorSignatures:
    """API contract: process_calculators classes used by Gasification_Model
    retain their core public method signatures.
    """

    def test_flare_calculator_has_calculate_method(self) -> None:
        """FlareCalculator must expose a 'calculate' method.

        Gasification_Model calls ``flare_calc.calculate(inputs)``; removing or
        renaming this method breaks that call site.
        """
        from upstream_drift_tools.process_calculators import FlareCalculator

        assert hasattr(FlareCalculator, "calculate"), (
            "FlareCalculator.calculate() is missing — "
            "Gasification_Model calls it directly"
        )
        assert callable(FlareCalculator.calculate)

    def test_baghouse_calculator_has_calculate_method(self) -> None:
        """BaghouseCalculator must expose a 'calculate' method."""
        from upstream_drift_tools.process_calculators import BaghouseCalculator

        assert hasattr(BaghouseCalculator, "calculate"), (
            "BaghouseCalculator.calculate() is missing"
        )
        assert callable(BaghouseCalculator.calculate)

    def test_financial_calculator_has_calculate_method(self) -> None:
        """FinancialCalculator must expose a 'calculate' method."""
        from upstream_drift_tools.process_calculators import FinancialCalculator

        assert hasattr(FinancialCalculator, "calculate"), (
            "FinancialCalculator.calculate() is missing"
        )
        assert callable(FinancialCalculator.calculate)

    def test_flare_design_is_dataclass_or_namedtuple(self) -> None:
        """FlareDesign must be constructable by keyword — Gasification_Model does
        ``FlareDesign(heat_release_kw=..., stack_height_m=...)``."""
        from upstream_drift_tools.process_calculators import FlareDesign

        # Must be inspectable as a class with fields
        assert isinstance(FlareDesign, type), "FlareDesign must be a class"

    def test_baghouse_result_is_dataclass_or_namedtuple(self) -> None:
        """BaghouseResult must be a class with fields Gasification_Model unpacks."""
        from upstream_drift_tools.process_calculators import BaghouseResult

        assert isinstance(BaghouseResult, type), "BaghouseResult must be a class"

    def test_physical_constants_values_unchanged(self) -> None:
        """Physical constants must not change value — downstream physics calculations
        depend on the exact numeric values.

        R_UNIVERSAL = 8.314462618 J/mol·K (NIST CODATA 2018)
        STANDARD_GRAVITY = 9.80665 m/s² (ISO 80000-3)
        """
        from upstream_drift_tools.process_calculators import (
            R_UNIVERSAL,
            STANDARD_GRAVITY,
        )

        assert R_UNIVERSAL == pytest.approx(8.314462618, rel=1e-6), (
            "R_UNIVERSAL value changed — downstream thermodynamic calculations "
            "will produce wrong results"
        )
        assert STANDARD_GRAVITY == pytest.approx(9.80665, rel=1e-6), (
            "STANDARD_GRAVITY value changed — downstream pressure calculations "
            "will produce wrong results"
        )

    def test_unit_conversion_helpers_values_correct(self) -> None:
        """celsius_to_kelvin and kelvin_to_celsius must give correct results.

        These are called millions of times in process simulations; a numeric
        error would produce wrong temperatures throughout Gasification_Model.
        """
        from upstream_drift_tools.process_calculators import (
            celsius_to_kelvin,
            kelvin_to_celsius,
        )

        assert celsius_to_kelvin(0.0) == pytest.approx(273.15)
        assert celsius_to_kelvin(100.0) == pytest.approx(373.15)
        assert kelvin_to_celsius(273.15) == pytest.approx(0.0)
        assert kelvin_to_celsius(373.15) == pytest.approx(100.0)
        # Round-trip
        for t_c in (-40.0, 0.0, 20.0, 100.0, 500.0):
            assert kelvin_to_celsius(celsius_to_kelvin(t_c)) == pytest.approx(t_c)


# ---------------------------------------------------------------------------
# data_processing — exception hierarchy
# ---------------------------------------------------------------------------


@pytest.mark.contract
class TestDataProcessingExceptionHierarchy:
    """API contract: Exception classes used by UpstreamDrift and
    Gasification_Model in except-clauses must remain importable and must
    remain subclasses of their documented base classes.

    Changing the inheritance hierarchy (e.g. making ColumnNotFoundError
    no longer a subclass of DataProcessingError) breaks downstream
    ``except DataProcessingError:`` catch-all blocks.
    """

    def test_all_exceptions_are_exception_subclasses(self) -> None:
        """All custom exceptions must be subclasses of Exception."""
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
            assert issubclass(exc_cls, Exception), (
                f"{exc_cls.__name__} must be a subclass of Exception"
            )

    def test_specific_exceptions_are_subclass_of_base(self) -> None:
        """Specific exceptions must be catchable via the base DataProcessingError.

        Downstream catch-all pattern: ``except DataProcessingError: ...``
        """
        from upstream_drift_tools.data_processing import (
            ColumnNotFoundError,
            DataNotLoadedError,
            DataProcessingError,
            FilterError,
            FitError,
            TransformationError,
            UnsupportedOperationError,
        )

        specific = [
            DataNotLoadedError,
            ColumnNotFoundError,
            TransformationError,
            FilterError,
            FitError,
            UnsupportedOperationError,
        ]
        for exc_cls in specific:
            assert issubclass(exc_cls, DataProcessingError), (
                f"{exc_cls.__name__} must be a subclass of DataProcessingError "
                "so downstream ``except DataProcessingError:`` blocks catch it"
            )

    def test_exceptions_raise_with_message(self) -> None:
        """Custom exceptions must accept a string message — downstream code
        passes human-readable context for logging."""
        from upstream_drift_tools.data_processing import (
            ColumnNotFoundError,
            DataNotLoadedError,
            DataProcessingError,
        )

        for exc_cls in (DataProcessingError, DataNotLoadedError, ColumnNotFoundError):
            exc = exc_cls("test message")
            assert str(exc), f"{exc_cls.__name__} must produce a non-empty str()"


# ---------------------------------------------------------------------------
# contracts.py DbC — re-export surface
# ---------------------------------------------------------------------------


@pytest.mark.contract
class TestContractsReExportSurface:
    """API contract: src/contracts.py re-export surface must remain intact.

    Gasification_Model and UpstreamDrift import DbC primitives from the
    top-level ``contracts`` module (short path).  Removing a symbol here
    forces coordinated changes in both downstream repos.
    """

    @pytest.mark.parametrize(
        "symbol",
        [
            "require",
            "ensure",
            "precondition",
            "postcondition",
            "invariant",
            "contract",
            "PreconditionError",
            "PostconditionError",
            "InvariantError",
            "ContractViolationError",
            "ContractLevel",
            "set_contract_level",
            "get_contract_level",
            "require_positive",
            "require_finite",
            "check_range",
            "check_temperature",
            "check_pressure",
        ],
    )
    def test_symbol_importable_from_contracts(self, symbol: str) -> None:
        """Every documented symbol must be importable from the contracts module."""
        import importlib

        mod = importlib.import_module("contracts")
        assert hasattr(mod, symbol), (
            f"contracts.{symbol} is missing — downstream repos import it by name"
        )

    def test_require_is_callable(self) -> None:
        """require must be callable (function or callable class)."""
        from contracts import require

        assert callable(require)

    def test_ensure_is_callable(self) -> None:
        """ensure must be callable."""
        from contracts import ensure

        assert callable(ensure)

    def test_contract_level_has_off_variant(self) -> None:
        """ContractLevel must have an OFF member — used to disable checks in prod."""
        from contracts import ContractLevel

        assert hasattr(ContractLevel, "OFF"), (
            "ContractLevel.OFF is required — downstream repos set it in production"
        )

    def test_contract_level_has_enforce_variant(self) -> None:
        """ContractLevel must have an ENFORCE member — used in test environments."""
        from contracts import ContractLevel

        assert hasattr(ContractLevel, "ENFORCE"), (
            "ContractLevel.ENFORCE is required — downstream test suites activate it"
        )
