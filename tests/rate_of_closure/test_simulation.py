"""Tests for the simulation session package (epic #4103 integration).

Covers: the app-frame swing sources (manual constant twist, double
pendulum adapter, new triple pendulum), the end-to-end session
(swing -> impact -> flight in a plausible driver band), the impact-time
scrubber math (tau shift -> clubhead-ball coincidence), the ISA adapter
sanity against ``twist_to_screw`` on a constant twist, and the CSV/JSON
export round-trip.
"""

from __future__ import annotations

import csv
import json
import math
import warnings

import numpy as np
import pytest

from rate_of_closure.club import get_club
from rate_of_closure.model import ImpactScenario
from rate_of_closure.simulation import (
    BALL_POSITION_M,
    SOURCE_KINDS,
    ManualSwingSource,
    SimulationConfig,
    SimulationRun,
    TriplePendulumParameters,
    TriplePendulumSwing,
    delivery_at,
    make_source,
    run_simulation,
    run_to_json_dict,
    screw_axis_samples,
    write_csv,
    write_json,
)
from shared.python.swing_sim.types import PlaneOrientation

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_SCENARIO = ImpactScenario(clubhead_speed_mph=113.0)
_DRIVER = get_club("Driver 10.5°")


# ── Sources ─────────────────────────────────────────────────────────


class TestManualSwingSource:
    def test_square_at_midpoint_with_reference_at_origin(self) -> None:
        source = ManualSwingSource(_SCENARIO, duration=0.06)
        sample = source.sample(0.03)
        assert np.allclose(sample.pose[:3, :3], np.eye(3), atol=1e-12)
        assert np.allclose(sample.pose[:3, 3], 0.0, atol=1e-12)

    def test_constant_twist_matches_scenario(self) -> None:
        source = ManualSwingSource(_SCENARIO)
        s0 = source.sample(0.0)
        s1 = source.sample(source.duration)
        assert np.allclose(s0.twist, s1.twist)
        speed_mps = 113.0 * 0.44704
        assert np.linalg.norm(s0.twist[3:]) == pytest.approx(speed_mps, rel=1e-9)

    def test_out_of_range_time_rejected(self) -> None:
        source = ManualSwingSource(_SCENARIO)
        with pytest.raises(Exception, match="within"):
            source.sample(source.duration + 1.0)


class TestPendulumSources:
    def test_make_source_covers_every_kind(self) -> None:
        for kind in SOURCE_KINDS:
            source = make_source(kind, _SCENARIO, duration=0.5)
            sample = source.sample(source.duration / 2.0)
            assert sample.pose.shape == (4, 4)
            assert sample.twist.shape == (6,)

    def test_unknown_kind_rejected(self) -> None:
        with pytest.raises(Exception, match="unknown swing source"):
            make_source("quadruple", _SCENARIO)

    def test_double_pendulum_app_frame_stays_on_tilted_plane(self) -> None:
        """With zero tilts the swing plane's normal is the app x axis...

        no — the swing frame's plane normal (y) maps to app -z, so a
        zero-tilt swing stays in the app x-y plane (z = const = 0).
        """
        source = make_source("double_pendulum", _SCENARIO, duration=0.5)
        for t in np.linspace(0.0, source.duration, 11):
            assert source.sample(float(t)).pose[2, 3] == pytest.approx(0.0, abs=1e-9)

    def test_triple_pendulum_conserves_energy_without_damping(self) -> None:
        params = TriplePendulumParameters.golf_default()
        undamped = TriplePendulumParameters(
            m=params.m,
            l=params.l,
            lc=params.lc,
            i_com=params.i_com,
            damping=(0.0, 0.0, 0.0),
        )
        swing = TriplePendulumSwing(parameters=undamped, duration=1.0, dt=5e-4)
        from rate_of_closure.simulation.sources import triple_total_energy

        e0 = triple_total_energy(undamped, swing._states[0], swing._g_inplane)
        e1 = triple_total_energy(undamped, swing._states[-1], swing._g_inplane)
        # e0 is ~0 J (all links horizontal defines the potential zero),
        # so compare absolutely against the ~100 J swing energy scale.
        assert e1 == pytest.approx(e0, abs=1e-6)

    def test_triple_pendulum_reach_bounded_by_total_length(self) -> None:
        swing = TriplePendulumSwing(duration=0.5)
        total = sum(swing.parameters.l)
        for t in np.linspace(0.0, swing.duration, 21):
            reach = float(np.linalg.norm(swing.sample(float(t)).pose[:3, 3]))
            assert reach <= total + 1e-9


# ── Session ─────────────────────────────────────────────────────────


class TestSession:
    @pytest.fixture(scope="class")
    def manual_run(self) -> SimulationRun:
        return run_simulation(SimulationConfig(scenario=_SCENARIO, club=_DRIVER))

    def test_manual_run_produces_plausible_driver_numbers(
        self, manual_run: SimulationRun
    ) -> None:
        launch = manual_run.launch
        # 113 mph clubhead, 10.5° driver: smash pushes ball speed well
        # above clubhead speed; typical fitting bands.
        assert 130.0 < launch["ball_speed_mph"] < 185.0
        assert 5.0 < launch["launch_angle_deg"] < 20.0
        assert 1000.0 < launch["spin_rpm"] < 5000.0
        assert 150.0 < launch["carry_m"] < 320.0
        assert 10.0 < launch["max_height_m"] < 60.0
        assert launch["flight_time_s"] > 3.0

    def test_pendulum_run_end_to_end(self) -> None:
        run = run_simulation(
            SimulationConfig(
                scenario=_SCENARIO,
                club=_DRIVER,
                source_kind="double_pendulum",
                plane=PlaneOrientation(side_tilt_deg=-30.0),
            )
        )
        # Auto impact time = maximum clubhead speed; a gravity-driven
        # double pendulum from horizontal reaches a modest but real
        # speed, and the ball must leave faster than the clubhead.
        speed = float(np.linalg.norm(run.delivery.clubhead_velocity))
        assert speed > 3.0
        assert run.launch["ball_speed_mph"] > speed * 2.23694
        assert run.launch["carry_m"] > 0.0
        assert run.total_duration_s > run.impact_time_s
        assert run.swing_joints.shape == (len(run.swing_times), 3, 3)
        np.testing.assert_allclose(
            run.swing_joints[:, -1], run.swing_positions, atol=1e-10
        )

    def test_triple_pendulum_run_end_to_end(self) -> None:
        run = run_simulation(
            SimulationConfig(
                scenario=_SCENARIO, club=_DRIVER, source_kind="triple_pendulum"
            )
        )
        assert run.launch["ball_speed_mph"] > 0.0
        assert len(run.flight_positions) > 2
        assert run.swing_joints.shape == (len(run.swing_times), 4, 3)
        np.testing.assert_allclose(
            run.swing_joints[:, -1], run.swing_positions, atol=1e-10
        )

    def test_scrubber_tau_shift_gives_clubhead_ball_coincidence(self) -> None:
        for tau in (0.010, 0.030, 0.045):
            run = run_simulation(
                SimulationConfig(scenario=_SCENARIO, club=_DRIVER, impact_time_s=tau)
            )
            assert run.impact_time_s == pytest.approx(tau)
            index = int(np.argmin(np.abs(run.swing_times - tau)))
            assert np.allclose(run.swing_positions[index], BALL_POSITION_M, atol=1e-6)

    def test_scrubber_delivery_updates_live(self) -> None:
        source = make_source("double_pendulum", _SCENARIO, duration=0.8)
        d1 = delivery_at(source, 0.3, _SCENARIO, _DRIVER)
        d2 = delivery_at(source, 0.6, _SCENARIO, _DRIVER)
        assert not np.allclose(d1.clubhead_velocity, d2.clubhead_velocity)

    def test_flight_starts_at_ball_position(self, manual_run: SimulationRun) -> None:
        assert np.allclose(manual_run.flight_positions[0], BALL_POSITION_M, atol=1e-9)

    def test_bad_flight_model_rejected(self) -> None:
        with pytest.raises(ValueError, match="not a valid"):
            SimulationConfig(
                scenario=_SCENARIO, club=_DRIVER, flight_model="warp_drive"
            )


# ── ISA adapter ─────────────────────────────────────────────────────


class TestIsaAdapter:
    def test_constant_twist_axis_and_rate_match_twist_to_screw(self) -> None:
        source = ManualSwingSource(_SCENARIO, duration=0.02)
        dt = 1e-3
        times = np.arange(0.0, source.duration + dt / 2.0, dt)
        poses = [source.sample(float(t)).pose for t in times]
        samples = screw_axis_samples(poses, dt)

        omega = source.sample(0.0).twist[:3]
        expected_axis = omega / np.linalg.norm(omega)
        expected_rate = math.degrees(float(np.linalg.norm(omega)))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from rotation_converter.twist_screw import twist_to_screw

        screw = twist_to_screw(source.sample(0.0).twist)
        assert np.allclose(np.asarray(screw["axis"]), expected_axis, atol=1e-9)

        for entry in samples:
            assert np.allclose(entry["axis"], expected_axis, atol=1e-6)
            assert entry["rate_dps"] == pytest.approx(expected_rate, rel=1e-6)
            assert math.isfinite(entry["r_isa_m"])

    def test_requires_two_poses_and_positive_dt(self) -> None:
        pose = np.eye(4)
        with pytest.raises(Exception, match="at least 2"):
            screw_axis_samples([pose], 1e-3)
        with pytest.raises(Exception, match="dt"):
            screw_axis_samples([pose, pose], 0.0)


# ── Export ──────────────────────────────────────────────────────────


class TestExport:
    @pytest.fixture(scope="class")
    def run(self) -> SimulationRun:
        return run_simulation(SimulationConfig(scenario=_SCENARIO, club=_DRIVER))

    def test_csv_round_trip(self, run: SimulationRun, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "run.csv"
        write_csv(run, path)
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        assert rows[0] == [
            "phase",
            "t_s",
            "x_m",
            "y_m",
            "z_m",
            "speed_mps",
            "is_fixed_ball_contact",
            "impact_occurred",
            "impact_time_s",
            "candidate_time_s",
            "closest_approach_m",
            "contact_margin_m",
        ]
        assert len(rows) - 1 == len(run.swing_times) + len(run.flight_times)
        phases = {row[0] for row in rows[1:]}
        assert phases == {"swing", "flight"}
        # Times must be numeric and non-decreasing within each phase.
        swing_ts = [float(r[1]) for r in rows[1:] if r[0] == "swing"]
        assert swing_ts == sorted(swing_ts)

    def test_json_round_trip(self, run: SimulationRun, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "run.json"
        write_json(run, path)
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded == run_to_json_dict(run)
        assert loaded["format"].startswith("rate_of_closure.simulation_run/")
        assert loaded["parameters"]["club"] == _DRIVER.name
        assert loaded["launch"]["ball_speed_mph"] == pytest.approx(
            run.launch["ball_speed_mph"]
        )
        assert len(loaded["series"]["rows"]) == len(run.swing_times) + len(
            run.flight_times
        )
