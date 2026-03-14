"""Tests for gui/controls_utils.py — pure-logic input parsing helpers.

No Qt dependency needed. All functions use duck-typed .value attributes,
which we supply via simple mock objects.

Covers:
- parse_float: success, ValueError on bad input
- parse_coeffs: success, empty, ValueError on bad input
- parse_coeffs_lenient: tolerant parsing, empty/bad tokens
- clamp_dt: boundary values, clamping logic
- require_positive: zero, negative, positive
- require_non_negative: negative, zero, positive
- Module-level font constants and their constraints
"""

from __future__ import annotations

import pytest

from double_pendulum_golf.gui.controls_utils import (
    FONT_BODY,
    FONT_BTN,
    FONT_EDIT,
    FONT_GROUP,
    FONT_STATUS,
    MIN_FONT_PX,
    clamp_dt,
    parse_coeffs,
    parse_coeffs_lenient,
    parse_float,
    require_non_negative,
    require_positive,
)

# ---------------------------------------------------------------------------
# Minimal mock widget
# ---------------------------------------------------------------------------


class _Widget:
    """Minimal duck-typing stand-in for LabeledInput."""

    def __init__(self, value: str) -> None:
        self.value = value


# ---------------------------------------------------------------------------
# Tests for parse_float
# ---------------------------------------------------------------------------


class TestParseFloat:
    def test_valid_integer_string(self) -> None:
        w = _Widget("42")
        assert parse_float(w, "gravity") == pytest.approx(42.0)

    def test_valid_float_string(self) -> None:
        w = _Widget("3.14159")
        assert parse_float(w, "pi") == pytest.approx(3.14159)

    def test_negative_float(self) -> None:
        w = _Widget("-9.81")
        assert parse_float(w, "g") == pytest.approx(-9.81)

    def test_zero(self) -> None:
        w = _Widget("0.0")
        assert parse_float(w, "zero") == pytest.approx(0.0)

    def test_scientific_notation(self) -> None:
        w = _Widget("1e-3")
        assert parse_float(w, "eps") == pytest.approx(1e-3)

    def test_empty_string_raises(self) -> None:
        w = _Widget("")
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_float(w, "length")

    def test_non_numeric_raises(self) -> None:
        w = _Widget("abc")
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_float(w, "length")

    def test_error_includes_name(self) -> None:
        w = _Widget("NaN_string")
        with pytest.raises(ValueError, match="my_field"):
            parse_float(w, "my_field")


# ---------------------------------------------------------------------------
# Tests for parse_coeffs
# ---------------------------------------------------------------------------


class TestParseCoeffs:
    def test_single_coefficient(self) -> None:
        w = _Widget("5.0")
        assert parse_coeffs(w, "torque") == [5.0]

    def test_multiple_coefficients(self) -> None:
        w = _Widget("1.0, 2.0, 3.0")
        assert parse_coeffs(w, "coeffs") == [1.0, 2.0, 3.0]

    def test_strips_whitespace(self) -> None:
        w = _Widget("  1.0 ,  2.0  ")
        assert parse_coeffs(w, "c") == [1.0, 2.0]

    def test_empty_string_returns_empty(self) -> None:
        w = _Widget("")
        assert parse_coeffs(w, "c") == []

    def test_trailing_comma_ignored(self) -> None:
        """Trailing comma produces empty token which is ignored."""
        w = _Widget("1.0, 2.0,")
        result = parse_coeffs(w, "c")
        assert result == [1.0, 2.0]

    def test_non_numeric_raises(self) -> None:
        w = _Widget("1.0, abc, 3.0")
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_coeffs(w, "torque_coeffs")

    def test_error_includes_name(self) -> None:
        w = _Widget("bad_input")
        with pytest.raises(ValueError, match="my_name"):
            parse_coeffs(w, "my_name")

    def test_negative_coefficients(self) -> None:
        w = _Widget("-1.0, -2.5")
        assert parse_coeffs(w, "c") == [-1.0, -2.5]

    def test_zero_coefficient(self) -> None:
        w = _Widget("0.0, 1.0")
        assert parse_coeffs(w, "c") == [0.0, 1.0]


# ---------------------------------------------------------------------------
# Tests for parse_coeffs_lenient
# ---------------------------------------------------------------------------


class TestParseCoeffsLenient:
    def test_valid_coefficients(self) -> None:
        w = _Widget("1.0, 2.0, 3.0")
        assert parse_coeffs_lenient(w) == [1.0, 2.0, 3.0]

    def test_empty_returns_zero_list(self) -> None:
        w = _Widget("")
        assert parse_coeffs_lenient(w) == [0.0]

    def test_invalid_token_returns_empty_list(self) -> None:
        """Non-numeric token should cause immediate return of []."""
        w = _Widget("1.0, bad, 3.0")
        result = parse_coeffs_lenient(w)
        assert result == []

    def test_whitespace_only_returns_zero_list(self) -> None:
        w = _Widget("   ")
        assert parse_coeffs_lenient(w) == [0.0]

    def test_trailing_comma_ignored(self) -> None:
        w = _Widget("1.0, 2.0,")
        assert parse_coeffs_lenient(w) == [1.0, 2.0]

    def test_negative_coefficients(self) -> None:
        w = _Widget("-5.5, 0.0")
        assert parse_coeffs_lenient(w) == [-5.5, 0.0]

    def test_single_valid_value(self) -> None:
        w = _Widget("42")
        assert parse_coeffs_lenient(w) == [42.0]

    def test_all_empty_tokens_returns_zero(self) -> None:
        w = _Widget(", , ,")
        assert parse_coeffs_lenient(w) == [0.0]


# ---------------------------------------------------------------------------
# Tests for clamp_dt
# ---------------------------------------------------------------------------


class TestClampDt:
    def test_within_range_unchanged(self) -> None:
        assert clamp_dt(0.01) == pytest.approx(0.01)

    def test_above_max_clamped_to_01(self) -> None:
        assert clamp_dt(1.0) == pytest.approx(0.1)

    def test_below_min_clamped_to_1e5(self) -> None:
        assert clamp_dt(0.0) == pytest.approx(1e-5)

    def test_exact_min_boundary(self) -> None:
        assert clamp_dt(1e-5) == pytest.approx(1e-5)

    def test_exact_max_boundary(self) -> None:
        assert clamp_dt(0.1) == pytest.approx(0.1)

    def test_negative_input_clamped_to_min(self) -> None:
        assert clamp_dt(-0.5) == pytest.approx(1e-5)

    def test_non_float_raises(self) -> None:
        with pytest.raises(AssertionError):
            clamp_dt(5)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tests for require_positive
# ---------------------------------------------------------------------------


class TestRequirePositive:
    def test_positive_value_returned(self) -> None:
        assert require_positive(5.0, "length") == pytest.approx(5.0)

    def test_small_positive_returned(self) -> None:
        assert require_positive(1e-10, "eps") == pytest.approx(1e-10)

    def test_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            require_positive(0.0, "mass")

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            require_positive(-1.0, "mass")

    def test_error_message_includes_name(self) -> None:
        with pytest.raises(ValueError, match="my_param"):
            require_positive(-5.0, "my_param")


# ---------------------------------------------------------------------------
# Tests for require_non_negative
# ---------------------------------------------------------------------------


class TestRequireNonNegative:
    def test_positive_value_returned(self) -> None:
        assert require_non_negative(3.0, "friction") == pytest.approx(3.0)

    def test_zero_returned(self) -> None:
        assert require_non_negative(0.0, "damping") == pytest.approx(0.0)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="must be non-negative"):
            require_non_negative(-0.1, "friction")

    def test_error_message_includes_name(self) -> None:
        with pytest.raises(ValueError, match="stiffness"):
            require_non_negative(-1.0, "stiffness")


# ---------------------------------------------------------------------------
# Tests for module-level font constants
# ---------------------------------------------------------------------------


class TestFontConstants:
    """Font constants must meet minimum readability threshold."""

    def test_min_font_px_positive(self) -> None:
        assert MIN_FONT_PX > 0

    def test_font_body_meets_minimum(self) -> None:
        assert FONT_BODY >= MIN_FONT_PX

    def test_font_group_meets_minimum(self) -> None:
        assert FONT_GROUP >= MIN_FONT_PX

    def test_font_edit_meets_minimum(self) -> None:
        assert FONT_EDIT >= MIN_FONT_PX

    def test_font_btn_meets_minimum(self) -> None:
        assert FONT_BTN >= MIN_FONT_PX

    def test_font_status_meets_minimum(self) -> None:
        assert FONT_STATUS >= MIN_FONT_PX

    def test_all_are_integers(self) -> None:
        for name, val in [
            ("MIN_FONT_PX", MIN_FONT_PX),
            ("FONT_BODY", FONT_BODY),
            ("FONT_GROUP", FONT_GROUP),
            ("FONT_EDIT", FONT_EDIT),
            ("FONT_BTN", FONT_BTN),
            ("FONT_STATUS", FONT_STATUS),
        ]:
            assert isinstance(val, int), f"{name} should be an int"
