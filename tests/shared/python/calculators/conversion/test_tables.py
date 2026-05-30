"""Tests for the canonical conversion tables (issue #3101 F2/F4).

Was a 0-byte stub. Confirms the single mass-flow table treats ``ton`` as a
short ton and that the legacy converter now agrees with it.
"""

from __future__ import annotations

import pytest
from sidekick.calculators.conversion import flow_rate_converter as frc
from sidekick.calculators.conversion.tables import (
    MASS_FLOW_FACTORS,
    StandardCondition,
)
from sidekick.utils import unit_constants as uc


@pytest.mark.unit
def test_canonical_ton_is_short_ton() -> None:
    assert MASS_FLOW_FACTORS["ton/hr"] == pytest.approx(
        uc.SHORT_TON_TO_KILOGRAM / 3600.0
    )
    assert MASS_FLOW_FACTORS["tonne/hr"] == pytest.approx(1000.0 / 3600.0)


@pytest.mark.unit
def test_ton_semantics_agree_across_modules() -> None:
    """The two mass-flow tables agree on ``ton`` after the F2 fix."""
    assert MASS_FLOW_FACTORS["ton/hr"] == pytest.approx(
        frc.MASS_FLOW_CONVERSIONS["ton/hr"]
    )
    assert MASS_FLOW_FACTORS["tonne/hr"] == pytest.approx(
        frc.MASS_FLOW_CONVERSIONS["tonne/hr"]
    )


@pytest.mark.unit
def test_stp_standard_condition_is_one_bar() -> None:
    """Canonical STP is the IUPAC 0°C / 1 bar definition (F4)."""
    t_stp, p_stp, _ = StandardCondition.STP.value
    assert t_stp == pytest.approx(uc.STP_TEMPERATURE_K)
    assert p_stp == pytest.approx(uc.STP_PRESSURE_PA)
    assert p_stp == pytest.approx(100000.0)


@pytest.mark.unit
def test_stp_matches_legacy_converter_definition() -> None:
    """Both STP paths now yield the same density (F4)."""
    t_canon, p_canon, _ = StandardCondition.STP.value
    t_legacy, p_legacy, _ = frc.STANDARD_CONDITIONS["STP"]
    assert (t_canon, p_canon) == (t_legacy, p_legacy)
