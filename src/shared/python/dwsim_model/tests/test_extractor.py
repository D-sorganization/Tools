from unittest.mock import MagicMock

import pytest
from dwsim_model.results.extractor import (
    _PROP_ENERGY_FLOW,
    _PROP_ENTHALPY,
    _PROP_MASSFLOW,
    _PROP_PRESSURE,
    _PROP_TEMPERATURE,
    EnergyStreamResult,
    FlowsheetResults,
    ResultsExtractor,
    StreamResult,
)


class MockPropertyObject:
    def __init__(self, props):
        self._props = props

    def GetPropertyValue(self, prop_name):
        return self._props.get(prop_name)


def test_flowsheet_results_to_dict() -> None:
    res = FlowsheetResults()
    res.streams["S1"] = StreamResult(name="S1", temperature_C=100.0)
    res.energy_streams["E1"] = EnergyStreamResult(name="E1", energy_flow_kW=50.0)

    d = res.to_dict()
    assert d["streams"]["S1"]["temperature_C"] == 100.0
    assert d["energy_streams"]["E1"]["energy_flow_kW"] == 50.0
    assert d["converged"] is False


def test_results_extractor_calc_volumetric_flow() -> None:
    # 1 kg/s of Hydrogen (MW=2.016 g/mol)
    mf = {"Hydrogen": 1.0}

    # R = 8.314, T_NTP = 273.15, P_NTP = 101325
    # mw_mix_kg = 0.002016
    # V_dot = (1.0 / 0.002016) * 8.314 * 273.15 / 101325.0 = 11.121 m3/s = 40035 Nm3/h
    vol = ResultsExtractor._calc_volumetric_flow(1.0, mf, 101325.0)

    assert 39000 < vol < 41000

    vol_zero = ResultsExtractor._calc_volumetric_flow(0.0, mf, 101325.0)
    assert vol_zero == 0.0


def test_results_extractor_get_prop_no_value() -> None:
    assert (
        ResultsExtractor._get_prop(MockPropertyObject({}), "missing", default=123.0)
        == 123.0
    )


def test_results_extractor_get_prop() -> None:
    obj = MockPropertyObject({_PROP_TEMPERATURE: 400.0})
    val = ResultsExtractor._get_prop(obj, _PROP_TEMPERATURE)
    assert val == 400.0


def test_extract_material_stream() -> None:
    ext = ResultsExtractor(compound_names=["Hydrogen", "Oxygen"])
    obj = MockPropertyObject(
        {
            _PROP_TEMPERATURE: 373.15,  # 100 C
            _PROP_PRESSURE: 202650.0,  # 202.65 kPa
            _PROP_MASSFLOW: 2.0,
            _PROP_ENTHALPY: 50000.0,  # 50 kJ/kg
            "MoleFraction.Hydrogen": 0.8,
            "MoleFraction.Oxygen": 0.2,
            "MassFraction.Hydrogen": 0.1,
            "MassFraction.Oxygen": 0.9,
        }
    )

    res = ext._extract_material_stream("Stream1", obj)
    assert res.name == "Stream1"
    assert res.temperature_C == pytest.approx(100.0)
    assert res.pressure_kPa == pytest.approx(202.65)
    assert res.mass_flow_kg_s == 2.0
    assert res.specific_enthalpy_kJ_kg == 50.0
    assert res.mole_fractions["Hydrogen"] == 0.8
    assert res.mole_fractions["Oxygen"] == 0.2
    assert res.volumetric_flow_Nm3_h > 0


def test_extract_energy_stream() -> None:
    ext = ResultsExtractor()
    obj = MockPropertyObject({_PROP_ENERGY_FLOW: 150000.0})

    res = ext._extract_energy_stream("E1", obj)
    assert res.name == "E1"
    assert res.energy_flow_kW == 150.0


def test_extract_full() -> None:
    ext = ResultsExtractor(compound_names=["Hydrogen"])
    builder = MagicMock()
    builder.materials = {
        "S1": MockPropertyObject({_PROP_TEMPERATURE: 300.0}),
        "S2": MockPropertyObject({_PROP_TEMPERATURE: 400.0}),
    }
    builder.energy_streams = {
        "E1": MockPropertyObject({_PROP_ENERGY_FLOW: 10000.0}),
    }

    res = ext.extract(builder, converged=True)
    assert res.converged is True
    assert len(res.streams) == 2
    assert res.streams["S1"].temperature_C == pytest.approx(300.0 - 273.15)
    assert len(res.energy_streams) == 1
    assert res.energy_streams["E1"].energy_flow_kW == 10.0

    # Test specific key streams
    ext2 = ResultsExtractor(key_streams=["S1"])
    res2 = ext2.extract(builder)
    assert len(res2.streams) == 1
    assert "S1" in res2.streams
    assert "S2" not in res2.streams
