"""Tests for FieldOrigin enum and OriginValue container (Tools #5353 item 3)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from shared.python.golf_club.field_origin import FieldOrigin, OriginValue

pytestmark = [pytest.mark.unit]


def test_field_origin_members() -> None:
    """FieldOrigin must have the five required members with exact lowercase values."""
    assert FieldOrigin.MEASURED == "measured"
    assert FieldOrigin.IDENTIFIED == "identified"
    assert FieldOrigin.PRESCRIBED == "prescribed"
    assert FieldOrigin.SYNTHETIC == "synthetic"
    assert FieldOrigin.ABSENT == "absent"


@pytest.mark.parametrize(
    "member",
    [
        FieldOrigin.MEASURED,
        FieldOrigin.IDENTIFIED,
        FieldOrigin.PRESCRIBED,
        FieldOrigin.SYNTHETIC,
        FieldOrigin.ABSENT,
    ],
)
def test_field_origin_parse_roundtrip(member: FieldOrigin) -> None:
    """FieldOrigin.parse must round-trip enum instances and lowercase strings."""
    assert FieldOrigin.parse(member) is member
    assert FieldOrigin.parse(member.value) is member


@pytest.mark.parametrize(
    "invalid_value",
    [
        "Measured",
        "MEASURED",
        " measured",
        "measured ",
        "  absent  ",
        "Absent",
        "unknown",
        "",
        1,
        0,
        None,
        True,
        False,
        [],
        {},
    ],
)
def test_field_origin_parse_invalid_raises_value_error(invalid_value: object) -> None:
    """FieldOrigin.parse must raise ValueError naming allowed values."""
    with pytest.raises(ValueError) as exc_info:
        FieldOrigin.parse(invalid_value)
    message = str(exc_info.value)
    for expected in ("measured", "identified", "prescribed", "synthetic", "absent"):
        assert expected in message


def test_origin_value_absent_requires_none() -> None:
    """ABSENT origin requires value to be None."""
    ov_default = OriginValue(FieldOrigin.ABSENT)
    assert ov_default.origin is FieldOrigin.ABSENT
    assert ov_default.value is None

    ov_explicit = OriginValue(FieldOrigin.ABSENT, None)
    assert ov_explicit.origin is FieldOrigin.ABSENT
    assert ov_explicit.value is None

    with pytest.raises(ValueError, match="ABSENT"):
        OriginValue(FieldOrigin.ABSENT, 0.0)

    with pytest.raises(ValueError, match="ABSENT"):
        OriginValue(FieldOrigin.ABSENT, 1.0)


@pytest.mark.parametrize(
    "origin",
    [
        FieldOrigin.MEASURED,
        FieldOrigin.IDENTIFIED,
        FieldOrigin.PRESCRIBED,
        FieldOrigin.SYNTHETIC,
    ],
)
def test_origin_value_non_absent_stores_float(origin: FieldOrigin) -> None:
    """Non-absent origins require finite floats; integers are coerced to float."""
    ov_int = OriginValue(origin, 2)
    assert ov_int.value == 2.0
    assert isinstance(ov_int.value, float)
    assert ov_int.number == 2.0

    ov_float = OriginValue(origin, 3.14)
    assert ov_float.value == 3.14
    assert ov_float.number == 3.14

    ov_zero = OriginValue(origin, 0.0)
    assert ov_zero.value == 0.0
    assert ov_zero.number == 0.0


@pytest.mark.parametrize(
    "invalid_val",
    [
        None,
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
        False,
        "string",
    ],
)
def test_origin_value_non_absent_rejects_non_finite_or_bool(
    invalid_val: object,
) -> None:
    """Non-absent origins reject None, NaN, inf, bool, and non-numeric types."""
    with pytest.raises(ValueError):
        OriginValue(FieldOrigin.MEASURED, invalid_val)  # type: ignore[arg-type]


def test_origin_value_number_property_absent_raises() -> None:
    """Accessing .number on an ABSENT OriginValue must raise ValueError."""
    ov = OriginValue(FieldOrigin.ABSENT)
    with pytest.raises(ValueError, match="is absent"):
        _ = ov.number


def test_origin_value_immutable() -> None:
    """OriginValue is frozen and must reject attribute assignment."""
    ov = OriginValue(FieldOrigin.MEASURED, 10.0)
    with pytest.raises(FrozenInstanceError):
        ov.value = 20.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        ov.origin = FieldOrigin.ABSENT  # type: ignore[misc]
