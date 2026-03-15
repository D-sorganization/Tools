from upstream_drift_tools.process_calculators.wgs_reactor_calculator import (
    WGSReactorEngine,
)


def test_wgs_reactor_engine_equilibrium_constant() -> None:
    engine = WGSReactorEngine()
    # At standard temp 298.15 the constant should be very large (shift to right)
    k_eq_298 = engine.calculate_equilibrium_constant(298.15)
    assert k_eq_298 > 1.0


def test_wgs_reactor_engine_equilibrium_composition() -> None:
    engine = WGSReactorEngine()

    inlet_comp = {
        "CO": 50.0,
        "H2O": 0.0,  # Steam will be added
        "CO2": 0.0,
        "H2": 50.0,
    }

    # 500 C = 773.15 K
    result = engine.calculate_equilibrium_composition(
        inlet_comp, 773.15, 25.0, steam_ratio=2.0
    )

    # Check that reaction shifted
    assert result["conversion"] > 0
    assert result["equilibrium_constant"] > 0
    assert result["composition"]["CO2"] > 0.0


def test_wgs_reactor_size() -> None:
    engine = WGSReactorEngine()
    sizing = engine.size_wgs_reactor(
        feed_rate=1000.0,
        conversion=80.0,
        temperature=773.15,
        catalyst_type="HTS",
    )

    assert sizing["reactor_volume"] > 0
    assert sizing["catalyst_volume"] > 0
    assert sizing["diameter"] > 0
    assert sizing["length"] > 0
    assert sizing["heat_duty"] > 0
