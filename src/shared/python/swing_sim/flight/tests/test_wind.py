"""Contract and physics tests for reproducible three-dimensional wind."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from shared.python.swing_sim.flight import (
    LaunchConditions,
    WaterlooPennerModel,
    WindGust,
    WindScenario,
)


def test_meteorological_from_bearing_maps_to_flight_frame_velocity() -> None:
    headwind = WindScenario.from_meteorological(10.0, 0.0)
    from_right = WindScenario.from_meteorological(10.0, 90.0, vertical_mps=2.0)

    assert headwind.velocity_at(0.0, (0.0, 0.0, 0.0)) == pytest.approx(
        (-10.0, 0.0, 0.0)
    )
    assert from_right.velocity_at(0.0, (0.0, 0.0, 0.0)) == pytest.approx(
        (0.0, 10.0, 2.0)
    )


def test_gust_has_smooth_zero_endpoints_and_declared_peak() -> None:
    gust = WindGust(1.0, 2.0, (4.0, -2.0, 1.0))

    assert gust.velocity_at(0.9) == (0.0, 0.0, 0.0)
    assert gust.velocity_at(1.0) == pytest.approx((0.0, 0.0, 0.0))
    assert gust.velocity_at(2.0) == pytest.approx((4.0, -2.0, 1.0))
    assert gust.velocity_at(3.0) == pytest.approx((0.0, 0.0, 0.0))


def test_shear_and_seeded_turbulence_are_pure_and_reproducible() -> None:
    scenario = WindScenario(
        base_velocity_mps=(8.0, 1.0, 0.0),
        shear_fraction_per_10m=0.1,
        turbulence_intensity_mps=0.8,
        seed=71,
    )

    first = scenario.velocity_at(1.25, (10.0, 0.0, 20.0))
    assert first == pytest.approx(scenario.velocity_at(1.25, (10.0, 0.0, 20.0)))
    assert first != pytest.approx(scenario.velocity_at(1.35, (10.0, 0.0, 20.0)))
    calm = WindScenario(base_velocity_mps=(8.0, 1.0, 0.0))
    assert calm.velocity_at(1.25, (10.0, 0.0, 20.0)) == pytest.approx((8.0, 1.0, 0.0))


def test_wind_scenario_and_legacy_wind_cannot_be_aliased() -> None:
    with pytest.raises(ValueError, match="either wind_scenario or legacy"):
        LaunchConditions(
            ball_speed=70.0,
            launch_angle=math.radians(12.0),
            wind_speed=2.0,
            wind_scenario=WindScenario(base_velocity_mps=(1.0, 0.0, 0.0)),
        )


def test_headwind_and_tailwind_change_the_actual_physics_path() -> None:
    model = WaterlooPennerModel()
    common = dict(ball_speed=65.0, launch_angle=math.radians(12.0), spin_rate=2600.0)
    calm = model.simulate(LaunchConditions(**common)).carry_distance
    headwind = model.simulate(
        LaunchConditions(
            **common,
            wind_scenario=WindScenario.from_meteorological(10.0, 0.0),
        )
    ).carry_distance
    tailwind = model.simulate(
        LaunchConditions(
            **common,
            wind_scenario=WindScenario.from_meteorological(10.0, 180.0),
        )
    ).carry_distance

    assert headwind < calm < tailwind


def test_python_matches_the_shared_cross_client_wind_fixture() -> None:
    fixture_path = (
        Path(__file__).resolve().parents[6]
        / "src/rate_of_closure/web/src/model/__fixtures__/wind_scenario_golden_v1.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    for case in fixture["cases"]:
        gusts = tuple(
            WindGust(
                item["start_time_s"],
                item["duration_s"],
                tuple(item["peak_velocity_mps"]),
            )
            for item in case["gusts"]
        )
        scenario = WindScenario(
            base_velocity_mps=tuple(case["base_velocity_mps"]),
            shear_fraction_per_10m=case["shear_fraction_per_10m"],
            gusts=gusts,
            turbulence_intensity_mps=case["turbulence_intensity_mps"],
            seed=case["seed"],
            provenance=f"golden:{case['name']}",
        )
        # Stated at 1e-12 m/s precision. The deterministic integer hash mixer
        # evaluates bit-for-bit identical parameters across platforms and runtimes,
        # restoring tight numerical cross-client parity.
        assert scenario.velocity_at(
            case["time_s"], case["position_m"]
        ) == pytest.approx(case["expected_velocity_mps"], rel=1e-12, abs=1e-12)
