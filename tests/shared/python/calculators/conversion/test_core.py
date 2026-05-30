"""Tests for the stateless conversion-core helpers (issue #3101 F6/F9).

Was a 0-byte stub. Covers the documented-error contract on unknown units,
the Rankine/temperature finite guard, and the SCFM standard-state correction.
"""

from __future__ import annotations

import math

import pytest
from sidekick.calculators.conversion import core
from sidekick.calculators.conversion.tables import StandardCondition


@pytest.mark.unit
def test_convert_via_table_unknown_unit_raises_value_error() -> None:
    """Unknown units raise the documented ``ValueError`` (not bare KeyError) (F6)."""
    table = {"kg/s": 1.0, "kg/h": 1.0 / 3600.0}
    with pytest.raises(ValueError, match="Unknown source unit"):
        core.convert_via_table(1.0, "bogus", "kg/s", table)
    with pytest.raises(ValueError, match="Unknown target unit"):
        core.convert_via_table(1.0, "kg/s", "bogus", table)


@pytest.mark.unit
def test_convert_via_table_round_trip() -> None:
    table = {"kg/s": 1.0, "kg/h": 1.0 / 3600.0}
    out = core.convert_via_table(3600.0, "kg/h", "kg/s", table)
    assert out == pytest.approx(1.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("from_u", "to_u"),
    [(a, b) for a in "KCFR" for b in "KCFR"],
)
def test_temperature_round_trip_all_pairs(from_u: str, to_u: str) -> None:
    """All 4x4 temperature pairs round-trip (F9)."""
    value = 300.0 if from_u in {"K", "R"} else 25.0
    converted = core.convert_temperature(value, from_u, to_u)
    back = core.convert_temperature(converted, to_u, from_u)
    assert back == pytest.approx(value, rel=1e-9, abs=1e-9)


@pytest.mark.unit
def test_temperature_known_values() -> None:
    assert core.convert_temperature(0.0, "C", "K") == pytest.approx(273.15)
    assert core.convert_temperature(212.0, "F", "C") == pytest.approx(100.0)
    assert core.convert_temperature(491.67, "R", "K") == pytest.approx(273.15, abs=1e-2)


@pytest.mark.unit
def test_temperature_non_finite_raises() -> None:
    """Non-finite temperature is rejected (F9)."""
    with pytest.raises(ValueError, match="finite"):
        core.convert_temperature(math.inf, "K", "C")
    with pytest.raises(ValueError, match="finite"):
        core.convert_temperature(math.nan, "R", "K")


@pytest.mark.unit
def test_scfm_correction_to_stp_changes_value() -> None:
    """SCFM->STP m3/hr applies the standard-state correction (F1)."""
    raw = core.scfm_to_standard_m3_per_hour(
        1000.0, StandardCondition.SCFM_60F, StandardCondition.SCFM_60F
    )
    corrected = core.scfm_to_standard_m3_per_hour(
        1000.0, StandardCondition.SCFM_60F, StandardCondition.STP
    )
    # SCFM (288.7 K, 101325 Pa) -> STP (273.15 K, 100000 Pa): ~4% difference.
    assert corrected != pytest.approx(raw)
    assert corrected / raw == pytest.approx(
        (273.15 / 288.706) * (101325.0 / 100000.0), rel=1e-6
    )


@pytest.mark.unit
def test_scfm_round_trip_through_standard() -> None:
    m3hr = core.scfm_to_standard_m3_per_hour(
        500.0, StandardCondition.SCFM_60F, StandardCondition.STP
    )
    back = core.standard_m3_per_hour_to_scfm(
        m3hr, StandardCondition.STP, StandardCondition.SCFM_60F
    )
    assert back == pytest.approx(500.0, rel=1e-9)
