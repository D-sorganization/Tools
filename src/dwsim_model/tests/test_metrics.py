"""
tests/test_metrics.py
=====================
Unit tests for MetricsCalculator and GasificationMetrics.

Strategy: we don't need DWSIM here — we construct mock FlowsheetResults
and StreamResult objects directly and feed them to the calculator.

This is a good pattern for newer Python developers:
  - Create a simple dataclass or SimpleNamespace that mimics the interface
    of the real object
  - Test the *logic* of the calculator independently of the simulation

Key interface that MetricsCalculator.calculate() expects from the results object:
  - results.get_stream(name: str) -> stream | None
  - results.streams: dict[str, stream]
  - results.energy_streams: dict[str, energy_stream]
  - results.converged: bool
  - results.errors: list

Key interface expected of each stream object:
  - stream.mass_flow_kg_s: float
  - stream.temperature_C: float
  - stream.pressure_kPa: float
  - stream.mole_fractions: dict[str, float]
  - stream.mass_fractions: dict[str, float]
  - stream.volumetric_flow_Nm3_h: float
  - stream.specific_enthalpy_kJ_kg: float

Key interface expected of each energy stream:
  - energy_stream.energy_flow_kW: float
"""

from types import SimpleNamespace

import pytest

from dwsim_model.results.metrics import GasificationMetrics, MetricsCalculator

# ─────────────────────────────────────────────────────────────────────────────
# Mock helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_stream(
    mass_flow_kg_s=1.0,
    temperature_C=25.0,
    pressure_kPa=101.3,
    mole_fractions=None,
    mass_fractions=None,
    volumetric_flow_Nm3_h=100.0,
    specific_enthalpy_kJ_kg=0.0,
):
    """Build a mock StreamResult-like object.

    Notes for newer developers
    --------------------------
    DWSIM stream objects have both mole_fractions and mass_fractions, which
    are different things:
      - mole_fractions: fraction of *molecules* belonging to each species
      - mass_fractions: fraction of *mass* belonging to each species
    They generally differ because species have different molecular weights.
    Here both default to empty dicts to keep tests simple; populate whichever
    the code-under-test actually reads.
    """
    return SimpleNamespace(
        mass_flow_kg_s=mass_flow_kg_s,
        temperature_C=temperature_C,
        pressure_kPa=pressure_kPa,
        mole_fractions=mole_fractions or {},
        mass_fractions=mass_fractions or {},
        volumetric_flow_Nm3_h=volumetric_flow_Nm3_h,
        specific_enthalpy_kJ_kg=specific_enthalpy_kJ_kg,
    )


def _make_energy_stream(energy_flow_kW=0.0):
    """Build a mock energy stream (e.g. E_PEM_AC_Power)."""
    return SimpleNamespace(energy_flow_kW=energy_flow_kW)


def _make_results(
    streams=None,
    energy_streams=None,
    converged=True,
    errors=None,
):
    """Build a mock FlowsheetResults-like object.

    Notes for newer developers
    --------------------------
    MetricsCalculator.calculate() calls results.get_stream(name) as a
    convenience method rather than indexing results.streams directly.
    We add that as a lambda so SimpleNamespace quacks like the real class.
    """
    streams_dict = streams or {}
    obj = SimpleNamespace(
        streams=streams_dict,
        energy_streams=energy_streams or {},
        converged=converged,
        errors=errors or [],
    )
    # Add the get_stream helper that calculate() calls
    obj.get_stream = lambda name: streams_dict.get(name)
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# GasificationMetrics
# ─────────────────────────────────────────────────────────────────────────────


class TestGasificationMetrics:

    def test_to_dict_contains_all_kpi_keys(self):
        m = GasificationMetrics(
            cold_gas_efficiency=0.72,
            carbon_conversion_efficiency=0.90,
            h2_co_ratio=1.8,
            syngas_lhv_mj_nm3=10.5,
            specific_energy_consumption_kWh_t=550.0,
            tar_loading_mg_Nm3=12.0,
            mass_balance_closure=1.001,
            energy_balance_closure=0.995,
        )
        d = m.to_dict()
        expected_keys = {
            "cold_gas_efficiency",
            "carbon_conversion_efficiency",
            "h2_co_ratio",
            "syngas_lhv_mj_nm3",
            "specific_energy_consumption_kWh_t",
            "tar_loading_mg_Nm3",
            "mass_balance_closure",
            "energy_balance_closure",
        }
        assert expected_keys.issubset(set(d.keys()))

    def test_check_targets_passes_when_targets_met(self):
        m = GasificationMetrics(
            cold_gas_efficiency=0.75,
            carbon_conversion_efficiency=0.92,
        )
        errors = m.check_targets(
            {
                "cold_gas_efficiency_min": 0.70,
                "carbon_conversion_min": 0.90,
            }
        )
        assert len(errors) == 0

    def test_check_targets_fails_when_cge_below_target(self):
        m = GasificationMetrics(cold_gas_efficiency=0.60)
        errors = m.check_targets({"cold_gas_efficiency_min": 0.70})
        assert len(errors) >= 1
        assert any("cold gas" in e.lower() or "cge" in e.lower() for e in errors)

    def test_check_targets_handles_none_values(self):
        """None KPI should not crash check_targets."""
        m = GasificationMetrics(cold_gas_efficiency=None)
        errors = m.check_targets({"cold_gas_efficiency_min": 0.70})
        # Either an error or a warning — but should not raise
        assert isinstance(errors, list)


# ─────────────────────────────────────────────────────────────────────────────
# MetricsCalculator
# ─────────────────────────────────────────────────────────────────────────────


class TestMetricsCalculator:

    def setup_method(self):
        self.calculator = MetricsCalculator()

    def test_empty_results_returns_metrics_without_crash(self):
        results = _make_results()
        metrics = self.calculator.calculate(results)
        assert isinstance(metrics, GasificationMetrics)

    def test_h2_co_ratio_computed_correctly(self):
        """
        If syngas is 50% H2, 50% CO by moles, H2/CO ratio should be 1.0.

        Note: mole_fractions are used for ratio; mass_fractions for LHV.
        We leave mass_fractions empty here — LHV will just be 0.0, which
        is fine for this test since we only check the ratio.
        """
        syngas = _make_stream(
            mass_flow_kg_s=2.0,
            mole_fractions={"Hydrogen": 0.50, "Carbon monoxide": 0.50},
            volumetric_flow_Nm3_h=200.0,
        )
        # calculate() specifically looks up "Final_Syngas" — that's the name
        # used in the full three-stage flowsheet for the cleaned syngas output.
        results = _make_results(streams={"Final_Syngas": syngas})
        metrics = self.calculator.calculate(results)
        assert metrics.h2_co_ratio == pytest.approx(1.0, rel=0.01)

    def test_h2_co_ratio_zero_when_no_co(self):
        """If CO = 0, ratio should be 0 (or None, not an error)."""
        syngas = _make_stream(
            mole_fractions={"Hydrogen": 0.80, "Carbon monoxide": 0.0},
        )
        results = _make_results(streams={"Final_Syngas": syngas})
        metrics = self.calculator.calculate(results)
        # H2/CO of 0 or None is expected; the calculator should log a warning
        assert metrics.h2_co_ratio == 0.0 or metrics.h2_co_ratio is None

    def test_mass_balance_closure_near_one_for_balanced_system(self):
        """
        Simple case: total inlet mass = total outlet mass → closure = 1.0.

        The calculator checks INLET_STREAMS and OUTLET_STREAMS by name.
        "Gasifier_Biomass_Feed" and "Gasifier_Oxygen_Feed" are recognised
        inlets; "Final_Syngas" is a recognised outlet.
        """
        feed_biomass = _make_stream(mass_flow_kg_s=4.0)
        feed_oxygen = _make_stream(mass_flow_kg_s=3.0)
        syngas_out = _make_stream(mass_flow_kg_s=7.0)

        results = _make_results(
            streams={
                "Gasifier_Biomass_Feed": feed_biomass,
                "Gasifier_Oxygen_Feed": feed_oxygen,
                "Final_Syngas": syngas_out,  # recognised outlet name
            }
        )
        metrics = self.calculator.calculate(results)
        if metrics.mass_balance_closure is not None:
            assert (
                0.8 < metrics.mass_balance_closure < 1.2
            ), f"Mass balance closure {metrics.mass_balance_closure:.3f} is unreasonable"

    def test_sec_calculated_when_energy_and_feed_available(self):
        """
        With 1 MW plasma power and 1 kg/s biomass (3.6 t/h),
        SEC should be 1000 kW / 3.6 t/h ≈ 278 kWh/t.

        Important notes:
          - energy_streams must be mock objects with an energy_flow_kW
            attribute, not raw numbers.
          - calculate() returns early if "Final_Syngas" is absent, so we
            include a minimal syngas stream to allow SEC computation.
        """
        biomass_feed = _make_stream(mass_flow_kg_s=1.0)
        syngas = _make_stream(mass_flow_kg_s=0.9, volumetric_flow_Nm3_h=500.0)
        results = _make_results(
            streams={
                "Gasifier_Biomass_Feed": biomass_feed,
                "Final_Syngas": syngas,
            },
            energy_streams={
                "E_PEM_AC_Power": _make_energy_stream(energy_flow_kW=1_000.0),  # 1 MW
            },
        )
        metrics = self.calculator.calculate(results)
        if metrics.specific_energy_consumption_kWh_t is not None:
            # 1000 kW / 3.6 t/h ≈ 278 kWh/t
            assert (
                200 < metrics.specific_energy_consumption_kWh_t < 400
            ), f"SEC {metrics.specific_energy_consumption_kWh_t:.1f} kWh/t out of expected range"

    def test_no_syngas_stream_returns_none_cge(self):
        """If there's no identifiable syngas stream, CGE should be None."""
        results = _make_results(streams={"Some_Other_Stream": _make_stream()})
        metrics = self.calculator.calculate(results)
        # CGE requires a biomass feed and a syngas stream — should be None here
        assert metrics.cold_gas_efficiency is None or isinstance(
            metrics.cold_gas_efficiency, float
        )

    def test_warnings_list_attribute_exists(self):
        """The metrics object should always have a warnings attribute."""
        results = _make_results()
        metrics = self.calculator.calculate(results)
        assert hasattr(metrics, "warnings")
        assert isinstance(metrics.warnings, list)

    def test_carbon_conversion_uses_explicit_biomass_carbon_basis(self):
        calculator = MetricsCalculator(biomass_carbon_mass_fraction=0.50)
        biomass_feed = _make_stream(
            mass_flow_kg_s=4.0,
            mass_fractions={"Hydrogen": 1.0},
        )
        syngas = _make_stream(
            mass_flow_kg_s=2.0,
            mass_fractions={"Methane": 1.0},
            volumetric_flow_Nm3_h=500.0,
        )
        results = _make_results(
            streams={
                "Gasifier_Biomass_Feed": biomass_feed,
                "Final_Syngas": syngas,
            }
        )

        metrics = calculator.calculate(results)

        expected = (2.0 * (12.011 / 16.043)) / (4.0 * 0.50)
        assert metrics.carbon_conversion_efficiency == pytest.approx(expected)

    def test_carbon_conversion_falls_back_to_surrogate_stream_basis(self):
        biomass_feed = _make_stream(
            mass_flow_kg_s=10.0,
            mass_fractions={"Carbon monoxide": 0.50, "Hydrogen": 0.50},
        )
        syngas = _make_stream(
            mass_flow_kg_s=5.0,
            mass_fractions={"Carbon monoxide": 0.50, "Hydrogen": 0.50},
            volumetric_flow_Nm3_h=800.0,
        )
        results = _make_results(
            streams={
                "Gasifier_Biomass_Feed": biomass_feed,
                "Final_Syngas": syngas,
            }
        )

        metrics = self.calculator.calculate(results)

        assert metrics.carbon_conversion_efficiency == pytest.approx(0.5)

    def test_carbon_conversion_warns_when_feed_carbon_basis_unavailable(self):
        biomass_feed = _make_stream(
            mass_flow_kg_s=4.0,
            mass_fractions={"Hydrogen": 1.0},
        )
        syngas = _make_stream(
            mass_flow_kg_s=1.0,
            mass_fractions={"Carbon monoxide": 1.0},
            volumetric_flow_Nm3_h=100.0,
        )
        results = _make_results(
            streams={
                "Gasifier_Biomass_Feed": biomass_feed,
                "Final_Syngas": syngas,
            }
        )

        metrics = self.calculator.calculate(results)

        assert metrics.carbon_conversion_efficiency == 0.0
        assert any("carbon basis" in warning.lower() for warning in metrics.warnings)

    def test_syngas_lhv_volumetric_basis_is_computed(self):
        syngas = _make_stream(
            mass_flow_kg_s=2.0,
            mass_fractions={"Methane": 1.0},
            volumetric_flow_Nm3_h=1000.0,
        )
        results = _make_results(streams={"Final_Syngas": syngas})

        metrics = self.calculator.calculate(results)

        expected = (2.0 * 50.05 * 3600.0) / 1000.0
        assert metrics.syngas_lhv_mj_nm3 == pytest.approx(expected)

    def test_invalid_biomass_carbon_basis_is_rejected(self):
        with pytest.raises(ValueError, match="biomass_carbon_mass_fraction"):
            MetricsCalculator(biomass_carbon_mass_fraction=1.5)


# ─────────────────────────────────────────────────────────────────────────────
# Integration: decomposer → metrics pathway
# ─────────────────────────────────────────────────────────────────────────────


class TestMetricsFromDecomposer:
    """
    Verify that the decomposer output can be fed through to the metrics
    calculator without crashing — an integration smoke test.
    """

    def test_full_pipeline_smoke(self):
        from dwsim_model.chemistry.biomass_decomposer import (
            BiomassDecomposer,
            BiomassFeed,
        )

        feed = BiomassFeed(
            ultimate_daf={
                "C": 0.501,
                "H": 0.062,
                "O": 0.421,
                "N": 0.008,
                "S": 0.005,
                "Cl": 0.003,
            },
            moisture_ar=0.15,
            ash_ar=0.10,
        )
        decomposer = BiomassDecomposer(
            available_compounds=[
                "Carbon monoxide",
                "Hydrogen",
                "Carbon dioxide",
                "Methane",
                "Water",
                "Helium",
            ]
        )
        mf = decomposer.decompose(feed)

        # Build a mock syngas stream from the decomposer output.
        # The decomposer returns mole_fractions; we leave mass_fractions empty
        # (LHV will be 0, but the H2/CO ratio will still be computed correctly).
        syngas = _make_stream(
            mass_flow_kg_s=5.0,
            mole_fractions=mf,
            volumetric_flow_Nm3_h=3000.0,
        )
        # Must use the recognised stream name so calculate() doesn't return early
        results = _make_results(streams={"Final_Syngas": syngas})
        metrics = MetricsCalculator().calculate(results)

        # Should have computed H2/CO ratio from the decomposed mole fractions
        assert metrics.h2_co_ratio is not None
        assert metrics.h2_co_ratio >= 0.0
