"""Tests for upstream_drift_tools.calculators.conversion.tables module.

Covers:
- GasProperties dataclass
- StandardCondition enum
- Gas property data integrity
- Conversion table completeness
"""

from __future__ import annotations

import pytest
from upstream_drift_tools.calculators.conversion.tables import (
    GasProperties,
    StandardCondition,
)

# ── GasProperties ───────────────────────────────────────────────────────


class TestGasProperties:
    """Test GasProperties dataclass."""

    def test_construction(self) -> None:
        gas = GasProperties(
            name="Nitrogen",
            molecular_weight=28.014,
            density_stp=1.2506,
            specific_heat_ratio=1.4,
            critical_temp=126.2,
            critical_pressure=3394000.0,
            source="NIST",
        )
        assert gas.name == "Nitrogen"
        assert gas.molecular_weight == pytest.approx(28.014)
        assert gas.specific_heat_ratio == pytest.approx(1.4)

    def test_fields_accessible(self) -> None:
        gas = GasProperties(
            name="Oxygen",
            molecular_weight=31.998,
            density_stp=1.429,
            specific_heat_ratio=1.395,
            critical_temp=154.6,
            critical_pressure=5043000.0,
            source="NIST",
        )
        assert gas.density_stp > 0
        assert gas.critical_temp > 0
        assert gas.critical_pressure > 0

    def test_molecular_weight_positive(self) -> None:
        gas = GasProperties(
            name="H2",
            molecular_weight=2.016,
            density_stp=0.0899,
            specific_heat_ratio=1.41,
            critical_temp=33.2,
            critical_pressure=1296000.0,
            source="NIST",
        )
        assert gas.molecular_weight > 0


# ── StandardCondition ───────────────────────────────────────────────────


class TestStandardCondition:
    """Test StandardCondition enum."""

    def test_stp_values(self) -> None:
        """STP: 0°C (273.15 K), 100 kPa."""
        temp_k, pressure_pa, label = StandardCondition.STP.value
        assert temp_k == pytest.approx(273.15)
        assert pressure_pa == pytest.approx(100000.0)
        assert isinstance(label, str)

    def test_stp_old_values(self) -> None:
        """Old STP: 0°C, 101.325 kPa."""
        temp_k, pressure_pa, label = StandardCondition.STP_OLD.value
        assert temp_k == pytest.approx(273.15)
        assert pressure_pa == pytest.approx(101325.0)

    def test_ntp_values(self) -> None:
        """NTP: 20°C (293.15 K), 101.325 kPa."""
        temp_k, pressure_pa, _ = StandardCondition.NTP.value
        assert temp_k == pytest.approx(293.15)

    def test_satp_values(self) -> None:
        """SATP: 25°C (298.15 K), 1 bar."""
        temp_k, pressure_pa, _ = StandardCondition.SATP.value
        assert temp_k == pytest.approx(298.15)
        assert pressure_pa == pytest.approx(100000.0)

    def test_all_members_have_three_elements(self) -> None:
        for member in StandardCondition:
            val = member.value
            assert len(val) == 3, f"{member.name} should have (T, P, label)"
            assert isinstance(val[0], (int, float)), f"{member.name} T must be numeric"
            assert isinstance(val[1], (int, float)), f"{member.name} P must be numeric"
            assert isinstance(val[2], str), f"{member.name} label must be str"

    def test_temperatures_reasonable(self) -> None:
        """All temperatures should be between 200 K and 400 K."""
        for member in StandardCondition:
            temp_k = member.value[0]
            assert 200 <= temp_k <= 400, f"{member.name}: T={temp_k} K out of range"

    def test_pressures_reasonable(self) -> None:
        """All pressures should be between 90 kPa and 110 kPa."""
        for member in StandardCondition:
            pressure_pa = member.value[1]
            assert 90_000 <= pressure_pa <= 110_000, (
                f"{member.name}: P={pressure_pa} Pa out of range"
            )
