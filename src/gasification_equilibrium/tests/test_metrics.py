"""Tests for the post-solve metrics module.

Tests verify:
1. Gas mole fraction computation
2. H2/CO ratio
3. Carbon conversion
4. Cold gas efficiency
5. Composition dict and dry mole fractions
"""

import numpy as np
import pytest

from gasification_equilibrium.python.metrics import (
    carbon_conversion,
    cold_gas_efficiency,
    composition_dict,
    dry_mole_fractions,
    gas_mole_fractions,
    h2_co_ratio,
)
from gasification_equilibrium.python.thermo_data import SPECIES_DB

BASIC_KEYS = ["H2", "CO", "CO2", "H2O", "CH4", "N2", "C_solid"]


class TestGasMoleFractions:
    """Test gas_mole_fractions computation."""

    def test_pure_h2(self) -> None:
        moles = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        fracs, total = gas_mole_fractions(moles, BASIC_KEYS)
        assert abs(fracs[0] - 1.0) < 1e-10  # H2 = 100%
        assert abs(total - 1.0) < 1e-10

    def test_excludes_solid(self) -> None:
        """C_solid should not contribute to gas mole fractions."""
        moles = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # Only C_solid
        fracs, total = gas_mole_fractions(moles, BASIC_KEYS)
        assert total < 1e-10  # No gas phase

    def test_fractions_sum_to_one(self) -> None:
        moles = np.array([0.3, 0.2, 0.15, 0.25, 0.05, 0.05, 0.0])
        fracs, total = gas_mole_fractions(moles, BASIC_KEYS)
        gas_mask = np.array([SPECIES_DB[k]["phase"] == "gas" for k in BASIC_KEYS])
        gas_fracs_sum = np.sum(fracs * gas_mask)
        assert abs(gas_fracs_sum - 1.0) < 1e-10

    def test_nonnegative(self) -> None:
        moles = np.array([0.3, 0.2, 0.15, 0.25, 0.05, 0.05, 0.1])
        fracs, total = gas_mole_fractions(moles, BASIC_KEYS)
        assert np.all(fracs >= 0)


class TestH2CoRatio:
    """Test H2/CO molar ratio."""

    def test_equal_h2_co(self) -> None:
        moles = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ratio = h2_co_ratio(moles, BASIC_KEYS)
        assert abs(ratio - 1.0) < 1e-10

    def test_2_to_1_ratio(self) -> None:
        moles = np.array([2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ratio = h2_co_ratio(moles, BASIC_KEYS)
        assert abs(ratio - 2.0) < 1e-10

    def test_zero_co_returns_zero(self) -> None:
        moles = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ratio = h2_co_ratio(moles, BASIC_KEYS)
        assert ratio == 0.0

    def test_negligible_co_returns_zero(self) -> None:
        moles = np.array([1.0, 1e-15, 0.0, 0.0, 0.0, 0.0, 0.0])
        ratio = h2_co_ratio(moles, BASIC_KEYS)
        assert ratio == 0.0


class TestCarbonConversion:
    """Test carbon conversion calculation."""

    def test_full_conversion(self) -> None:
        """No solid carbon remaining."""
        moles = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # All C in CO
        cc = carbon_conversion(moles, BASIC_KEYS, {"C": 1.0})
        assert abs(cc - 1.0) < 1e-10

    def test_zero_conversion(self) -> None:
        """All carbon remains as solid."""
        moles = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # All C_solid
        cc = carbon_conversion(moles, BASIC_KEYS, {"C": 1.0})
        assert abs(cc - 0.0) < 1e-10

    def test_partial_conversion(self) -> None:
        """Half of carbon gasified."""
        moles = np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5])
        cc = carbon_conversion(moles, BASIC_KEYS, {"C": 1.0})
        assert abs(cc - 0.5) < 1e-10

    def test_no_feed_carbon_returns_one(self) -> None:
        moles = np.zeros(7)
        cc = carbon_conversion(moles, BASIC_KEYS, {"H": 2.0, "O": 1.0})
        assert cc == 1.0

    def test_clipped_to_zero_one(self) -> None:
        moles = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        cc = carbon_conversion(moles, BASIC_KEYS, {"C": 1.0})
        assert 0.0 <= cc <= 1.0


class TestColdGasEfficiency:
    """Test cold gas efficiency (HHV basis)."""

    def test_zero_when_no_combustibles(self) -> None:
        fracs = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])  # Only CO2
        cge = cold_gas_efficiency(fracs, 1.0, BASIC_KEYS, {"C": 1.0, "H": 1.0})
        assert cge == pytest.approx(0.0, abs=0.01)

    def test_positive_with_h2_and_co(self) -> None:
        fracs = np.array([0.4, 0.3, 0.1, 0.1, 0.05, 0.05, 0.0])
        cge = cold_gas_efficiency(fracs, 1.0, BASIC_KEYS, {"C": 1.0, "H": 1.0})
        assert cge > 0

    def test_zero_feed_energy(self) -> None:
        fracs = np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        cge = cold_gas_efficiency(fracs, 1.0, BASIC_KEYS, {})
        assert cge == 0.0

    def test_capped_at_2(self) -> None:
        """CGE should not exceed 2.0 even with extreme inputs."""
        fracs = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        cge = cold_gas_efficiency(fracs, 100.0, BASIC_KEYS, {"C": 0.001, "H": 0.001})
        assert cge <= 2.0


class TestCompositionDict:
    """Test composition_dict helper."""

    def test_basic(self) -> None:
        fracs = np.array([0.5, 0.3, 0.2])
        keys = ["A", "B", "C"]
        d = composition_dict(fracs, keys)
        assert d == {"A": 0.5, "B": 0.3, "C": pytest.approx(0.2)}

    def test_length(self) -> None:
        fracs = np.array([0.1, 0.2, 0.3, 0.4])
        keys = ["a", "b", "c", "d"]
        d = composition_dict(fracs, keys)
        assert len(d) == 4


class TestDryMoleFractions:
    """Test dry mole fraction calculation."""

    def test_removes_h2o(self) -> None:
        comp = {"H2": 0.4, "CO": 0.3, "CO2": 0.1, "H2O": 0.2}
        dry = dry_mole_fractions(comp)
        assert "H2O" not in dry

    def test_renormalizes(self) -> None:
        comp = {"H2": 0.4, "CO": 0.3, "CO2": 0.1, "H2O": 0.2}
        dry = dry_mole_fractions(comp)
        total = sum(dry.values())
        assert abs(total - 1.0) < 0.02

    def test_no_h2o_unchanged(self) -> None:
        comp = {"H2": 0.5, "CO": 0.3, "CO2": 0.2}
        dry = dry_mole_fractions(comp)
        assert abs(dry["H2"] - 0.5) < 1e-10

    def test_all_h2o(self) -> None:
        """If H2O >= 1.0, returns original dict."""
        comp = {"H2O": 1.0}
        dry = dry_mole_fractions(comp)
        assert "H2O" in dry
