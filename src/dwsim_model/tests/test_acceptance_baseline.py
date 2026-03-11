from __future__ import annotations

from types import SimpleNamespace

import pytest

from dwsim_model.gasification import GasificationFlowsheet
from dwsim_model.results.extractor import ResultsExtractor
from dwsim_model.results.metrics import MetricsCalculator


class FakeGraphicObject:
    pass


class FakeStream:
    def __init__(self, name: str):
        self.Name = name
        self.GraphicObject = FakeGraphicObject()
        self._properties: dict[str, float] = {}

    def SetPropertyValue(self, prop_name: str, value: float) -> None:
        if prop_name == "PROP_ES_0":
            self._properties["EnergyFlow"] = float(value) * 1000.0
            self._properties[prop_name] = float(value)
            return
        self._properties[prop_name] = float(value)

    def GetPropertyValue(self, prop_name: str) -> float | None:
        return self._properties.get(prop_name)


class FakeOperation:
    def __init__(self, name: str):
        self.Name = name
        self.GraphicObject = FakeGraphicObject()


class FakeBuilder:
    def __init__(self):
        self.materials: dict[str, FakeStream] = {}
        self.energy_streams: dict[str, FakeStream] = {}
        self.operations: dict[str, FakeOperation] = {}
        self.sim = SimpleNamespace()
        self.connections: list[tuple[str, str, int, int]] = []

    def add_compound(self, _name: str) -> None:
        return None

    def add_property_package(self, _package_name: str) -> object:
        return object()

    def add_object(self, obj_type_name: str, name: str, _x: int = 0, _y: int = 0):
        if obj_type_name == "MaterialStream":
            stream = FakeStream(name)
            self.materials[name] = stream
            return stream
        if obj_type_name == "EnergyStream":
            stream = FakeStream(name)
            self.energy_streams[name] = stream
            return stream

        operation = FakeOperation(name)
        self.operations[name] = operation
        return operation

    def connect(
        self, source_obj, target_obj, source_port: int = 0, target_port: int = 0
    ) -> None:
        self.connections.append(
            (source_obj.Name, target_obj.Name, source_port, target_port)
        )

    def calculate(self) -> None:
        return None


def _set_stream_properties(
    stream: FakeStream,
    *,
    temperature_k: float,
    pressure_pa: float,
    mass_flow_kg_s: float,
    specific_enthalpy_kj_kg: float = 0.0,
    mole_fractions: dict[str, float] | None = None,
    mass_fractions: dict[str, float] | None = None,
) -> None:
    stream.SetPropertyValue("Temperature", temperature_k)
    stream.SetPropertyValue("Pressure", pressure_pa)
    stream.SetPropertyValue("MassFlow", mass_flow_kg_s)
    stream.SetPropertyValue("SpecificEnthalpy", specific_enthalpy_kj_kg * 1000.0)

    for compound, fraction in (mole_fractions or {}).items():
        stream.SetPropertyValue(f"MoleFraction.{compound}", fraction)

    for compound, fraction in (mass_fractions or {}).items():
        stream.SetPropertyValue(f"MassFraction.{compound}", fraction)


@pytest.mark.acceptance
def test_baseline_config_build_extract_and_kpis(monkeypatch) -> None:
    builder = FakeBuilder()
    flowsheet = GasificationFlowsheet(
        builder=builder,
        config_path="config/master_config.yaml",
    )

    monkeypatch.setattr(flowsheet, "_configure_reactors", lambda: None)

    flowsheet.build_flowsheet()

    assert "Gasifier_Biomass_Feed" in builder.materials
    assert "Final_Syngas" in builder.materials
    assert "E_PEM_AC_Power" in builder.energy_streams
    assert len(builder.connections) > 10

    _set_stream_properties(
        builder.materials["Gasifier_Biomass_Feed"],
        temperature_k=298.15,
        pressure_pa=101325.0,
        mass_flow_kg_s=10.0,
        specific_enthalpy_kj_kg=50.0,
        mole_fractions={
            "Carbon monoxide": 0.35,
            "Hydrogen": 0.25,
            "Carbon dioxide": 0.15,
            "Water": 0.15,
            "Methane": 0.10,
        },
        mass_fractions={
            "Carbon monoxide": 0.40,
            "Hydrogen": 0.03,
            "Carbon dioxide": 0.20,
            "Water": 0.17,
            "Methane": 0.20,
        },
    )
    _set_stream_properties(
        builder.materials["Final_Syngas"],
        temperature_k=973.15,
        pressure_pa=140000.0,
        mass_flow_kg_s=7.5,
        specific_enthalpy_kj_kg=900.0,
        mole_fractions={
            "Hydrogen": 0.40,
            "Carbon monoxide": 0.32,
            "Carbon dioxide": 0.12,
            "Methane": 0.08,
            "Water": 0.05,
            "Nitrogen": 0.03,
        },
        mass_fractions={
            "Hydrogen": 0.04,
            "Carbon monoxide": 0.42,
            "Carbon dioxide": 0.18,
            "Methane": 0.14,
            "Water": 0.16,
            "Nitrogen": 0.06,
        },
    )
    _set_stream_properties(
        builder.materials["Gasifier_Glass_Out"],
        temperature_k=923.15,
        pressure_pa=101325.0,
        mass_flow_kg_s=0.8,
        specific_enthalpy_kj_kg=100.0,
    )
    _set_stream_properties(
        builder.materials["PEM_Glass_Out"],
        temperature_k=923.15,
        pressure_pa=101325.0,
        mass_flow_kg_s=0.6,
        specific_enthalpy_kj_kg=100.0,
    )
    _set_stream_properties(
        builder.materials["Baghouse_Solids_Out"],
        temperature_k=373.15,
        pressure_pa=101325.0,
        mass_flow_kg_s=0.4,
        specific_enthalpy_kj_kg=20.0,
    )
    _set_stream_properties(
        builder.materials["Scrubber_Blowdown"],
        temperature_k=333.15,
        pressure_pa=101325.0,
        mass_flow_kg_s=0.5,
        specific_enthalpy_kj_kg=10.0,
    )
    _set_stream_properties(
        builder.materials["Gasifier_Cooling_Steam_Out"],
        temperature_k=473.15,
        pressure_pa=300000.0,
        mass_flow_kg_s=5.2,
        specific_enthalpy_kj_kg=2800.0,
        mole_fractions={"Water": 1.0},
        mass_fractions={"Water": 1.0},
    )

    extractor = ResultsExtractor(compound_names=list(flowsheet.compound_set))
    results = extractor.extract(builder, converged=True)

    metrics = MetricsCalculator(
        biomass_lhv_mj_kg=15.0,
        biomass_carbon_mass_fraction=0.40,
    ).calculate(results)

    assert results.converged is True
    assert metrics.cold_gas_efficiency > 0.0
    assert metrics.carbon_conversion_efficiency > 0.0
    assert metrics.h2_co_ratio > 1.0
    assert metrics.syngas_lhv_mj_nm3 > 0.0
    assert metrics.specific_energy_consumption_kWh_t > 0.0
