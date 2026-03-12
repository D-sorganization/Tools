"""Tests for thermodynamic data and NASA polynomial functions.

Tests verify:
1. NASA polynomial evaluations against known reference values
2. Thermodynamic consistency (G = H - TS)
3. Species database integrity (including C3H8, Ar)
4. Element data completeness
"""

import numpy as np

from gasification_equilibrium.python.thermo_data import (
    ATOMIC_WEIGHTS,
    HEATING_VALUES_HHV,
    HEATING_VALUES_LHV,
    P_REF,
    R_GAS,
    SPECIES_DB,
    T_REF,
    cp_j_per_mol_k,
    cp_over_r,
    enthalpy_j_per_mol,
    entropy_j_per_mol_k,
    g_over_rt,
    get_all_species,
    get_coeffs,
    get_elements_in_system,
    get_gas_species,
    gibbs_dimensionless,
    h_over_rt,
    s_over_r,
)


class TestSpeciesDatabase:
    """Verify integrity of species database."""

    def test_all_species_have_required_fields(self) -> None:
        required = [
            "name",
            "formula",
            "mw",
            "elements",
            "phase",
            "T_low",
            "T_mid",
            "T_high",
            "coeff_low",
            "coeff_high",
        ]
        for key, sp in SPECIES_DB.items():
            for field in required:
                assert field in sp, f"{key} missing field '{field}'"

    def test_all_coefficients_have_7_elements(self) -> None:
        for key, sp in SPECIES_DB.items():
            assert (
                len(sp["coeff_low"]) == 7
            ), f"{key} coeff_low has {len(sp['coeff_low'])} elements"
            assert (
                len(sp["coeff_high"]) == 7
            ), f"{key} coeff_high has {len(sp['coeff_high'])} elements"

    def test_temperature_ranges_valid(self) -> None:
        for key, sp in SPECIES_DB.items():
            assert (
                sp["T_low"] < sp["T_mid"] < sp["T_high"]
            ), f"{key} has invalid T range"
            assert sp["T_low"] > 0, f"{key} T_low must be positive"

    def test_molecular_weights_positive(self) -> None:
        for key, sp in SPECIES_DB.items():
            assert sp["mw"] > 0, f"{key} has non-positive molecular weight"

    def test_phases_are_valid(self) -> None:
        for key, sp in SPECIES_DB.items():
            assert sp["phase"] in ("gas", "solid", "liquid"), f"{key} has invalid phase"

    def test_elements_are_known(self) -> None:
        known = set(ATOMIC_WEIGHTS.keys())
        for key, sp in SPECIES_DB.items():
            for elem in sp["elements"]:
                assert elem in known, f"{key} has unknown element '{elem}'"

    def test_minimum_species_count(self) -> None:
        assert (
            len(SPECIES_DB) >= 14
        ), "Should have at least 14 species (including C3H8, Ar)"

    def test_essential_species_present(self) -> None:
        essential = ["H2", "CO", "CO2", "H2O", "CH4", "N2", "C_solid"]
        for sp in essential:
            assert sp in SPECIES_DB, f"Essential species '{sp}' missing"

    def test_new_species_present(self) -> None:
        """C3H8 and Ar were added in Phase 2."""
        assert "C3H8" in SPECIES_DB, "Propane (C3H8) missing"
        assert "Ar" in SPECIES_DB, "Argon (Ar) missing"

    def test_c3h8_elements(self) -> None:
        assert SPECIES_DB["C3H8"]["elements"] == {"C": 3, "H": 8}

    def test_ar_is_noble_gas(self) -> None:
        assert SPECIES_DB["Ar"]["elements"] == {}
        assert SPECIES_DB["Ar"]["phase"] == "gas"

    def test_graphite_is_solid(self) -> None:
        assert SPECIES_DB["C_solid"]["phase"] == "solid"

    def test_molecular_weight_h2(self) -> None:
        assert abs(SPECIES_DB["H2"]["mw"] - 2.016) < 0.01

    def test_molecular_weight_co2(self) -> None:
        assert abs(SPECIES_DB["CO2"]["mw"] - 44.009) < 0.1

    def test_molecular_weight_c3h8(self) -> None:
        assert abs(SPECIES_DB["C3H8"]["mw"] - 44.096) < 0.1

    def test_molecular_weight_ar(self) -> None:
        assert abs(SPECIES_DB["Ar"]["mw"] - 39.948) < 0.1


class TestNASAPolynomials:
    """Test NASA polynomial evaluation functions."""

    def test_cp_over_r_positive_at_298(self) -> None:
        """Cp must be positive at standard conditions."""
        for key in SPECIES_DB:
            coeffs = get_coeffs(key, T_REF)
            cp = cp_over_r(T_REF, coeffs)
            assert cp > 0, f"{key} has negative Cp/R at 298 K: {cp}"

    def test_cp_over_r_h2_at_298(self) -> None:
        """H2 Cp/R ~ 3.47 at 298 K (Cp ~ 28.8 J/mol/K)."""
        coeffs = get_coeffs("H2", T_REF)
        cp = cp_over_r(T_REF, coeffs)
        assert 3.0 < cp < 4.0, f"H2 Cp/R at 298K = {cp}, expected ~3.47"

    def test_cp_over_r_n2_at_298(self) -> None:
        """N2 Cp/R ~ 3.50 at 298 K (diatomic ideal)."""
        coeffs = get_coeffs("N2", T_REF)
        cp = cp_over_r(T_REF, coeffs)
        assert 3.0 < cp < 4.0, f"N2 Cp/R at 298K = {cp}, expected ~3.50"

    def test_cp_over_r_ar_is_monatomic(self) -> None:
        """Argon Cp/R = 2.5 (ideal monatomic gas)."""
        coeffs = get_coeffs("Ar", T_REF)
        cp = cp_over_r(T_REF, coeffs)
        assert abs(cp - 2.5) < 0.01, f"Ar Cp/R at 298K = {cp}, expected 2.5"

    def test_enthalpy_h2o_at_298(self) -> None:
        """H2O enthalpy should be computable."""
        h = enthalpy_j_per_mol("H2O", T_REF)
        assert isinstance(h, float)

    def test_gibbs_consistency(self) -> None:
        """G/RT = H/RT - S/R for all species at multiple temperatures."""
        for T in [300, 500, 800, 1000, 1200, 1500, 2000]:
            for key in SPECIES_DB:
                coeffs = get_coeffs(key, T)
                g = g_over_rt(T, coeffs)
                h = h_over_rt(T, coeffs)
                s = s_over_r(T, coeffs)
                assert (
                    abs(g - (h - s)) < 1e-10
                ), f"{key} at {T}K: G/RT={g} != H/RT-S/R={h - s}"

    def test_coefficient_selection_low_range(self) -> None:
        """Uses low-T coefficients for T <= T_mid."""
        sp = SPECIES_DB["H2"]
        coeffs = get_coeffs("H2", sp["T_mid"] - 1)
        assert coeffs == sp["coeff_low"]

    def test_coefficient_selection_high_range(self) -> None:
        """Uses high-T coefficients for T > T_mid."""
        sp = SPECIES_DB["H2"]
        coeffs = get_coeffs("H2", sp["T_mid"] + 1)
        assert coeffs == sp["coeff_high"]

    def test_coefficient_selection_at_boundary(self) -> None:
        """At T_mid, uses low-T coefficients."""
        sp = SPECIES_DB["CO2"]
        coeffs = get_coeffs("CO2", sp["T_mid"])
        assert coeffs == sp["coeff_low"]

    def test_cp_increases_with_temperature_for_co2(self) -> None:
        """CO2 Cp should generally increase with temperature."""
        cp_300 = cp_j_per_mol_k("CO2", 300)
        cp_1000 = cp_j_per_mol_k("CO2", 1000)
        assert cp_1000 > cp_300, "CO2 Cp should increase with T"

    def test_entropy_increases_with_temperature(self) -> None:
        """Entropy should increase with temperature for all species."""
        for key in SPECIES_DB:
            s_400 = entropy_j_per_mol_k(key, 400)
            s_1200 = entropy_j_per_mol_k(key, 1200)
            assert s_1200 > s_400, f"{key} entropy should increase with T"

    def test_gibbs_dimensionless_returns_float(self) -> None:
        for key in SPECIES_DB:
            g = gibbs_dimensionless(key, 1000.0)
            assert isinstance(g, (float, np.floating)), f"{key} gibbs should be float"

    def test_c3h8_gibbs_evaluates(self) -> None:
        """Propane should have valid Gibbs energy at various temperatures."""
        for T in [300, 500, 800, 1200]:
            g = gibbs_dimensionless("C3H8", T)
            assert np.isfinite(g), f"C3H8 Gibbs at {T}K is not finite"


class TestHelperFunctions:
    """Test utility functions."""

    def test_get_gas_species_excludes_solids(self) -> None:
        gases = get_gas_species()
        for g in gases:
            assert SPECIES_DB[g]["phase"] == "gas"

    def test_get_gas_species_includes_c3h8_and_ar(self) -> None:
        gases = get_gas_species()
        assert "C3H8" in gases
        assert "Ar" in gases

    def test_get_all_species_includes_solids(self) -> None:
        all_sp = get_all_species()
        phases = {SPECIES_DB[k]["phase"] for k in all_sp}
        assert "solid" in phases
        assert "gas" in phases

    def test_get_elements_in_system(self) -> None:
        elements = get_elements_in_system(["H2", "CO", "CO2"])
        assert "H" in elements
        assert "C" in elements
        assert "O" in elements
        assert "N" not in elements

    def test_get_elements_in_system_with_sulfur(self) -> None:
        elements = get_elements_in_system(["H2S", "SO2"])
        assert "S" in elements
        assert "H" in elements
        assert "O" in elements

    def test_get_elements_in_system_ar_has_no_elements(self) -> None:
        """Argon has no elements in its composition dict."""
        elements = get_elements_in_system(["Ar"])
        assert len(elements) == 0

    def test_get_elements_sorted(self) -> None:
        elements = get_elements_in_system(get_all_species())
        assert elements == sorted(elements)


class TestConstants:
    """Test physical constants."""

    def test_r_gas_value(self) -> None:
        assert abs(R_GAS - 8.314) < 0.01

    def test_p_ref_value(self) -> None:
        assert abs(P_REF - 101325.0) < 1.0

    def test_t_ref_value(self) -> None:
        assert abs(T_REF - 298.15) < 0.1

    def test_heating_values_positive(self) -> None:
        for sp, hv in HEATING_VALUES_HHV.items():
            assert hv > 0, f"{sp} HHV must be positive"
        for sp, hv in HEATING_VALUES_LHV.items():
            assert hv > 0, f"{sp} LHV must be positive"

    def test_hhv_greater_than_lhv(self) -> None:
        for sp in HEATING_VALUES_HHV:
            if sp in HEATING_VALUES_LHV:
                assert (
                    HEATING_VALUES_HHV[sp] >= HEATING_VALUES_LHV[sp]
                ), f"{sp}: HHV should be >= LHV"

    def test_atomic_weights_has_all_elements(self) -> None:
        assert "C" in ATOMIC_WEIGHTS
        assert "H" in ATOMIC_WEIGHTS
        assert "O" in ATOMIC_WEIGHTS
        assert "N" in ATOMIC_WEIGHTS
        assert "S" in ATOMIC_WEIGHTS
        assert "Ar" in ATOMIC_WEIGHTS
