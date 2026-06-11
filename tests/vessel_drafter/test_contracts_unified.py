"""Tests for unified contracts API in vessel_drafter.

Verifies that vessel_drafter.contracts re-exports the shared contracts API
and that the legacy (name, value) parameter-order helpers are preserved for
backward compatibility with existing callers.

Closes #1862.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Shared contracts API re-export
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVesselDrafterContractsReExport:
    """vessel_drafter.contracts must re-export the shared contracts primitives."""

    def test_require_positive_raises_on_non_positive(self) -> None:
        from vessel_drafter.contracts import require_positive

        with pytest.raises(ValueError, match="positive"):
            require_positive("diameter", 0.0)

    def test_require_positive_passes_on_positive(self) -> None:
        from vessel_drafter.contracts import require_positive

        # Must not raise
        require_positive("diameter", 1.0)

    def test_require_nonnegative_raises_on_negative(self) -> None:
        from vessel_drafter.contracts import require_nonnegative

        with pytest.raises(ValueError, match="nonneg"):
            require_nonnegative("thickness", -0.1)

    def test_require_nonnegative_passes_on_zero(self) -> None:
        from vessel_drafter.contracts import require_nonnegative

        require_nonnegative("thickness", 0.0)

    def test_require_fraction_raises_below_zero(self) -> None:
        from vessel_drafter.contracts import require_fraction

        with pytest.raises(ValueError):
            require_fraction("ratio", -0.1)

    def test_require_fraction_raises_above_one(self) -> None:
        from vessel_drafter.contracts import require_fraction

        with pytest.raises(ValueError):
            require_fraction("ratio", 1.1)

    def test_require_fraction_passes_at_bounds(self) -> None:
        from vessel_drafter.contracts import require_fraction

        require_fraction("ratio", 0.0)
        require_fraction("ratio", 1.0)

    def test_require_integer_at_least_raises_below_minimum(self) -> None:
        from vessel_drafter.contracts import require_integer_at_least

        with pytest.raises(ValueError, match="electrode_count"):
            require_integer_at_least("electrode_count", 0, 1)

    def test_require_integer_at_least_passes_at_minimum(self) -> None:
        from vessel_drafter.contracts import require_integer_at_least

        require_integer_at_least("electrode_count", 1, 1)

    def test_require_less_or_equal_raises_above_maximum(self) -> None:
        from vessel_drafter.contracts import require_less_or_equal

        with pytest.raises(ValueError, match="height"):
            require_less_or_equal("height", 10.0, 9.0)

    def test_require_less_or_equal_passes_at_maximum(self) -> None:
        from vessel_drafter.contracts import require_less_or_equal

        require_less_or_equal("height", 9.0, 9.0)

    def test_require_finite_raises_on_inf(self) -> None:
        import math

        from vessel_drafter.contracts import require_finite

        with pytest.raises(ValueError, match="finite"):
            require_finite("wall_temp", math.inf)

    def test_require_finite_raises_on_nan(self) -> None:
        import math

        from vessel_drafter.contracts import require_finite

        with pytest.raises(ValueError, match="finite"):
            require_finite("wall_temp", math.nan)

    def test_require_finite_passes_on_real(self) -> None:
        from vessel_drafter.contracts import require_finite

        require_finite("wall_temp", 1200.0)


# ---------------------------------------------------------------------------
# Shared primitives are accessible
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSharedPrimitivesAccessible:
    """vessel_drafter.contracts must expose the shared require/ensure primitives."""

    def test_precondition_error_importable(self) -> None:
        from vessel_drafter.contracts import PreconditionError  # noqa: F401

        assert PreconditionError is not None

    def test_require_importable(self) -> None:
        from vessel_drafter.contracts import require  # noqa: F401

        assert require is not None

    def test_require_bool_style_works(self) -> None:
        """Shared require(bool, msg, value) API must work from vessel_drafter."""
        from vessel_drafter.contracts import PreconditionError, require

        with pytest.raises(PreconditionError):
            require(False, "must be positive", -1.0)
