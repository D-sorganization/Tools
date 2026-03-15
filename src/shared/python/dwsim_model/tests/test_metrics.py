import pytest
from dwsim_model.results.extractor import (
    EnergyStreamResult,
    FlowsheetResults,
    StreamResult,
)
from dwsim_model.results.metrics import (
    LHV_MJ_KG,
    GasificationMetrics,
    MetricsCalculator,
)


def test_gasification_metrics_to_dict() -> None:
    m = GasificationMetrics(
        cold_gas_efficiency=0.75123,
        carbon_conversion_efficiency=0.98123,
        h2_co_ratio=1.234,
        specific_energy_consumption_kWh_t=850.5,
        tar_loading_mg_Nm3=45.67,
    )
    d = m.to_dict()
    assert d["cold_gas_efficiency"] == 0.7512
    assert d["carbon_conversion_efficiency"] == 0.9812
    assert d["h2_co_ratio"] == 1.234
    assert d["specific_energy_consumption_kWh_t"] == 850.5
    assert d["tar_loading_mg_Nm3"] == 45.67


def test_metrics_check_targets() -> None:
    m = GasificationMetrics(
        cold_gas_efficiency=0.8,
        carbon_conversion_efficiency=0.95,
        h2_co_ratio=2.0,
        tar_loading_mg_Nm3=5.0,
    )

    targets = {
        "cold_gas_efficiency_min": 0.75,
        "carbon_conversion_min": 0.90,
        "h2_co_ratio_target": 2.0,
        "tar_loading_mg_Nm3_max": 10.0,
    }

    failures = m.check_targets(targets)
    assert len(failures) == 0

    # Check failures
    targets_fail = {
        "cold_gas_efficiency_min": 0.85,
        "carbon_conversion_min": 0.99,
        "h2_co_ratio_target": 1.0,  # Deviates by 100%
        "tar_loading_mg_Nm3_max": 1.0,
    }
    failures_fail = m.check_targets(targets_fail)
    assert len(failures_fail) == 4


def test_metrics_calculator_init() -> None:
    with pytest.raises(ValueError):
        MetricsCalculator(biomass_carbon_mass_fraction=1.5)

    mc = MetricsCalculator(biomass_carbon_mass_fraction=0.5)
    assert mc.biomass_carbon_mass_fraction == 0.5


def test_metrics_calculator_syngas_lhv() -> None:
    mc = MetricsCalculator()
    mass_fracs = {"Hydrogen": 0.1, "Carbon monoxide": 0.9}
    lhv = mc._calc_syngas_lhv(mass_fracs)
    expected = 0.1 * LHV_MJ_KG["Hydrogen"] + 0.9 * LHV_MJ_KG["Carbon monoxide"]
    assert lhv == pytest.approx(expected)


def test_metrics_calculator_ratio() -> None:
    mc = MetricsCalculator()
    mol_fracs = {"Hydrogen": 0.6, "Carbon monoxide": 0.3}
    ratio = mc._calc_ratio(mol_fracs, "Hydrogen", "Carbon monoxide")
    assert ratio == pytest.approx(2.0)

    # Zero denominator
    assert mc._calc_ratio({"Hydrogen": 0.5}, "Hydrogen", "Carbon monoxide") == 0.0


def test_metrics_calculator_carbon_conversion() -> None:
    mc = MetricsCalculator(biomass_carbon_mass_fraction=0.5)
    res = FlowsheetResults()
    res.streams["Gasifier_Biomass_Feed"] = StreamResult(
        name="Gasifier_Biomass_Feed", mass_flow_kg_s=1.0
    )
    res.streams["Final_Syngas"] = StreamResult(
        name="Final_Syngas", mass_flow_kg_s=0.8, mass_fractions={"Carbon monoxide": 1.0}
    )

    # Carbon in feed = 1.0 * 0.5 = 0.5
    # Carbon in syngas = 0.8 * 1.0 * (12.011 / 28.010) = 0.34305
    cc = mc._calc_carbon_conversion(res, res.streams["Gasifier_Biomass_Feed"])
    assert cc == pytest.approx((0.8 * 1.0 * (12.011 / 28.010)) / 0.5)


def test_metrics_calculator_mass_balance() -> None:
    mc = MetricsCalculator()
    res = FlowsheetResults()
    res.streams["Gasifier_Biomass_Feed"] = StreamResult(
        name="Gasifier_Biomass_Feed", mass_flow_kg_s=2.0
    )  # INLET
    res.streams["Gasifier_Oxygen_Feed"] = StreamResult(
        name="Gasifier_Oxygen_Feed", mass_flow_kg_s=1.0
    )  # INLET
    res.streams["Final_Syngas"] = StreamResult(
        name="Final_Syngas", mass_flow_kg_s=2.8
    )  # OUTLET
    res.streams["Gasifier_Glass_Out"] = StreamResult(
        name="Gasifier_Glass_Out", mass_flow_kg_s=0.2
    )  # OUTLET

    mb = mc._calc_mass_balance(res)
    assert mb == pytest.approx(3.0 / 3.0)


def test_metrics_calculator_tar_loading() -> None:
    mc = MetricsCalculator()
    s = StreamResult(
        name="S",
        mass_flow_kg_s=0.1,
        volumetric_flow_Nm3_h=100.0,
        mass_fractions={"Toluene": 0.01},
    )

    # Toluene mass flow = 0.01 * 0.1 = 0.001 kg/s = 3600000 mg/h
    # vol = 100 Nm3/h
    # tar = 3600000 / 100 = 36000 mg/Nm3
    tl = mc._calc_tar_loading(s)
    assert tl == pytest.approx(36000.0)


def test_calculate_metrics_empty() -> None:
    mc = MetricsCalculator()
    res = FlowsheetResults()
    m = mc.calculate(res)
    assert "Final_Syngas stream not found" in m.warnings[0]


def test_calculate_metrics_full() -> None:
    mc = MetricsCalculator(biomass_carbon_mass_fraction=0.5, biomass_lhv_mj_kg=15.0)
    res = FlowsheetResults()

    res.streams["Gasifier_Biomass_Feed"] = StreamResult(
        name="Gasifier_Biomass_Feed", mass_flow_kg_s=1.0, specific_enthalpy_kJ_kg=1000.0
    )
    res.streams["Final_Syngas"] = StreamResult(
        name="Final_Syngas",
        mass_flow_kg_s=1.5,
        temperature_C=800.0,
        volumetric_flow_Nm3_h=2000.0,
        specific_enthalpy_kJ_kg=500.0,
        mass_fractions={"Hydrogen": 0.1, "Carbon monoxide": 0.4},
        mole_fractions={"Hydrogen": 0.5, "Carbon monoxide": 0.5},
    )

    res.energy_streams["E_PEM_AC_Power"] = EnergyStreamResult(
        name="E_PEM_AC_Power", energy_flow_kW=1000.0
    )

    m = mc.calculate(res)
    assert m.syngas_mass_flow_kg_s == 1.5
    assert m.syngas_temperature_C == 800.0
    assert m.syngas_volumetric_flow_Nm3_h == 2000.0
    assert m.feed_mass_flow_kg_s == 1.0

    # SEC = 1000 kW / (1 kg/s * 3.6 t/h) = 1000 / 3.6 = 277.78 kWh/t
    assert m.specific_energy_consumption_kWh_t == pytest.approx(277.777, rel=1e-3)


def test_metrics_calculator_energy_balance() -> None:
    mc = MetricsCalculator()
    res = FlowsheetResults()
    # Inlet = mass 1 kg/s * 1000 kJ/kg = 1000 kW + AC 500 kW = 1500 kW
    res.streams["Gasifier_Biomass_Feed"] = StreamResult(
        name="Gasifier_Biomass_Feed", mass_flow_kg_s=1.0, specific_enthalpy_kJ_kg=1000.0
    )
    res.energy_streams["E_PEM_AC_Power"] = EnergyStreamResult(
        name="E_PEM_AC_Power", energy_flow_kW=500.0
    )

    # Outlet = mass 0.5 kg/s * 2000 kJ/kg = 1000 kW + Loss 500 kW = 1500 kW
    res.streams["Final_Syngas"] = StreamResult(
        name="Final_Syngas", mass_flow_kg_s=0.5, specific_enthalpy_kJ_kg=2000.0
    )
    res.energy_streams["E_Gasifier_HeatLoss"] = EnergyStreamResult(
        name="E_Gasifier_HeatLoss", energy_flow_kW=-500.0
    )

    eb = mc._calc_energy_balance(res)
    assert eb == pytest.approx(1.0)
