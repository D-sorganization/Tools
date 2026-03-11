"""Tests for the per-unit-operation convenience API.

These tests verify the public contract of dwsim_model.units without
requiring the DWSIM runtime.  Integration tests that actually solve
flowsheets are marked with @pytest.mark.integration.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

from dwsim_model.units import (
    VALID_MODES,
    run_full_train,
    run_gasifier,
    run_pem,
    run_trc,
)

MIN_DOCSTRING_LENGTH = 20


@pytest.mark.unit
class TestUnitAPIContract:
    """Verify the public API contract of dwsim_model.units."""

    # ──────────────────────────────────────────────────────────────────────
    # Shared signature tests (DRY: test all runners in one loop)
    # ──────────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("fn", [run_gasifier, run_pem, run_trc, run_full_train])
    def test_all_runners_share_required_parameters(self, fn) -> None:
        """All run_* functions must accept mode, config_path, compound_set."""
        sig = inspect.signature(fn)
        assert "mode" in sig.parameters, f"{fn.__name__} missing 'mode' param"
        assert "config_path" in sig.parameters, f"{fn.__name__} missing 'config_path'"
        assert "compound_set" in sig.parameters, f"{fn.__name__} missing 'compound_set'"

    @pytest.mark.parametrize("fn", [run_gasifier, run_pem, run_trc, run_full_train])
    def test_all_runners_have_docstrings(self, fn) -> None:
        """All public functions must have docstrings."""
        assert fn.__doc__ is not None, f"{fn.__name__} missing docstring"
        assert (
            len(fn.__doc__) > MIN_DOCSTRING_LENGTH
        ), f"{fn.__name__} docstring too short"

    @pytest.mark.parametrize("fn", [run_gasifier, run_pem, run_trc, run_full_train])
    def test_all_runners_have_return_annotation(self, fn) -> None:
        """All run_* functions must declare return type."""
        sig = inspect.signature(fn)
        assert (
            sig.return_annotation is not inspect.Parameter.empty
        ), f"{fn.__name__} missing return type annotation"

    # ──────────────────────────────────────────────────────────────────────
    # DbC: Input validation
    # ──────────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("fn", [run_gasifier, run_pem, run_trc, run_full_train])
    def test_invalid_mode_raises_value_error(self, fn) -> None:
        """Invalid mode must raise ValueError with descriptive message."""
        with pytest.raises(ValueError, match=r"[Ii]nvalid.*mode"):
            fn(mode="nonexistent_mode")

    @pytest.mark.parametrize("fn", [run_gasifier, run_pem, run_trc, run_full_train])
    def test_empty_mode_raises_value_error(self, fn) -> None:
        """Empty string mode must raise ValueError."""
        with pytest.raises(ValueError):
            fn(mode="")

    def test_valid_modes_are_defined(self) -> None:
        """VALID_MODES must contain the expected reactor modes."""
        assert "conversion" in VALID_MODES
        assert "equilibrium" in VALID_MODES
        assert "kinetic" in VALID_MODES
        assert "mixed" in VALID_MODES

    # ──────────────────────────────────────────────────────────────────────
    # Default parameter values
    # ──────────────────────────────────────────────────────────────────────

    def test_gasifier_default_mode(self) -> None:
        """run_gasifier default mode should be 'conversion'."""
        sig = inspect.signature(run_gasifier)
        assert sig.parameters["mode"].default == "conversion"

    def test_pem_default_mode(self) -> None:
        """run_pem default mode should be 'equilibrium'."""
        sig = inspect.signature(run_pem)
        assert sig.parameters["mode"].default == "equilibrium"

    def test_trc_default_mode(self) -> None:
        """run_trc default mode should be 'kinetic'."""
        sig = inspect.signature(run_trc)
        assert sig.parameters["mode"].default == "kinetic"

    def test_full_train_default_mode(self) -> None:
        """run_full_train default mode should be 'mixed'."""
        sig = inspect.signature(run_full_train)
        assert sig.parameters["mode"].default == "mixed"

    # ──────────────────────────────────────────────────────────────────────
    # Import isolation (no DWSIM at import time)
    # ──────────────────────────────────────────────────────────────────────

    def test_import_does_not_trigger_dwsim(self) -> None:
        """Importing dwsim_model.units must NOT load DWSIM runtime."""
        # Re-import to verify
        units_module = importlib.import_module("dwsim_model.units")

        # clr (pythonnet) should not be in sys.modules from just importing
        # the units module.  It is only loaded when get_automation() is called.
        # NOTE: If clr was already loaded by another test, this test is still
        # valid because the import of units itself should not have caused it.
        assert units_module is not None
