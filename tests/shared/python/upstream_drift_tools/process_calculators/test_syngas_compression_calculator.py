import pytest
from upstream_drift_tools.process_calculators.syngas_compression_calculator import (
    CompressionStage,
    SyngasCompressionEngine,
)


@pytest.fixture
def engine():
    return SyngasCompressionEngine()


def test_calculate_mixture_properties(engine):
    composition = {"H2": 50.0, "CO": 50.0}
    props = engine.calculate_mixture_properties(composition)
    assert props["molecular_weight"] > 0
    assert props["critical_temperature"] > 0
    assert props["critical_pressure"] > 0
    assert props["heat_capacity_ratio"] > 0
    assert "H2" in props["mole_fractions"]


def test_calculate_water_dropout(engine):
    # test water dropout logic
    res = engine.calculate_water_dropout(temperature=300.0, pressure=10.0, water_content=5.0)
    assert "water_vapor_pressure" in res
    assert "water_dropout" in res


def test_calculate_compression_work_isentropic(engine):
    composition = {"H2": 50.0, "CO": 50.0}
    props = engine.calculate_mixture_properties(composition)
    stage = CompressionStage(
        inlet_pressure=1.0,
        outlet_pressure=5.0,
        inlet_temperature=300.0,
        efficiency=0.8,
        compression_type="isentropic",
    )
    res = engine.calculate_compression_work(stage, flow_rate=100.0, mixture_props=props)
    assert res["work_actual"] > 0
    assert res["temp_out_actual"] > 300.0
    assert res["power_hp"] > 0


def test_calculate_compression_work_polytropic(engine):
    composition = {"H2": 50.0, "CO": 50.0}
    props = engine.calculate_mixture_properties(composition)
    stage = CompressionStage(
        inlet_pressure=1.0,
        outlet_pressure=5.0,
        inlet_temperature=300.0,
        efficiency=0.8,
        compression_type="polytropic",
    )
    res = engine.calculate_compression_work(stage, flow_rate=100.0, mixture_props=props)
    assert res["work_actual"] > 0
    assert res["power_hp"] > 0


def test_calculate_compression_work_isothermal(engine):
    composition = {"H2": 50.0, "CO": 50.0}
    props = engine.calculate_mixture_properties(composition)
    stage = CompressionStage(
        inlet_pressure=1.0,
        outlet_pressure=5.0,
        inlet_temperature=300.0,
        efficiency=0.8,
        compression_type="isothermal",
    )
    res = engine.calculate_compression_work(stage, flow_rate=100.0, mixture_props=props)
    assert res["work_actual"] > 0
    assert res["power_hp"] > 0


def test_calculate_multistage_compression(engine):
    composition = {"H2": 50.0, "CO": 50.0}
    stage1 = CompressionStage(
        inlet_pressure=1.0,
        outlet_pressure=3.0,
        inlet_temperature=300.0,
        efficiency=0.8,
        compression_type="isentropic",
    )
    stage2 = CompressionStage(
        inlet_pressure=3.0,
        outlet_pressure=9.0,
        inlet_temperature=300.0,
        efficiency=0.8,
        compression_type="isentropic",
    )
    res = engine.calculate_multistage_compression(
        [stage1, stage2], flow_rate=100.0, composition=composition
    )
    assert "stages" in res
    assert len(res["stages"]) == 2
    assert res["total_power_hp"] > 0


def test_analyze_process_conditions(engine):
    # mock a compression result
    res = {
        "final_temperature": 500.0,  # warning threshold is 473.15 K
        "final_pressure": 150.0,
        "total_power_hp": 10000.0,
        "stages": [
            {
                "water_dropout": {"water_dropout": 1.0},
                "work_isentropic": 100.0,
                "work_actual": 200.0,
            }
        ],
    }
    analysis = engine.analyze_process_conditions(res)
    assert len(analysis["concerns"]) > 0
    assert len(analysis["warnings"]) > 0


def test_invalid_pressure(engine):
    with pytest.raises(ValueError):
        engine.calculate_water_dropout(300.0, -1.0, 5.0)


def test_invalid_compression_type(engine):
    props = engine.calculate_mixture_properties({"H2": 100.0})
    stage = CompressionStage(
        inlet_pressure=1.0,
        outlet_pressure=5.0,
        inlet_temperature=300.0,
        efficiency=0.8,
        compression_type="magic",
    )
    with pytest.raises(ValueError):
        engine.calculate_compression_work(stage, 100.0, props)
