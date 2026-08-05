"""Swing kinetics tests (#4125 H2): inverse dynamics, energy, catalog.

Covers: the inverse-dynamics round trip (forced forward sim -> torque
profile recovered within differencing tolerance), energy consistency
(applied power integrates to the total-energy change for the undamped
forced pendulum; net joint power integrates to the kinetic-energy
change for the passive swing), reaction-force statics and geometry,
the catalog registration pin, and unsupported-source behavior.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rate_of_closure.club import get_club
from rate_of_closure.model import ImpactScenario
from rate_of_closure.plotting import CATALOG, extract, variables_by_category
from rate_of_closure.simulation import (
    KINETIC_JOINT_NAMES,
    SimulationConfig,
    compute_kinetics,
    inverse_dynamics,
    kinetics_for_run,
    run_simulation,
    simulate_forced,
    zero_torque_counterfactual,
)
from shared.python.swing_sim import reference
from shared.python.swing_sim.types import PendulumParameters, PendulumState

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_G_FLAT = (0.0, -9.80665)
_DT = 1e-3


def _params(d1: float = 0.4, d2: float = 0.25) -> PendulumParameters:
    base = PendulumParameters.golf_default()
    return PendulumParameters(
        m1=base.m1,
        l1=base.l1,
        lc1=base.lc1,
        i1=base.i1,
        m2=base.m2,
        l2=base.l2,
        lc2=base.lc2,
        i2=base.i2,
        d1=d1,
        d2=d2,
    )


@pytest.fixture(scope="module")
def pendulum_run():  # type: ignore[no-untyped-def]
    """One double-pendulum reference run shared by the module."""
    return run_simulation(
        SimulationConfig(
            scenario=ImpactScenario(clubhead_speed_mph=113.0),
            club=get_club("Driver 10.5°"),
            source_kind="double_pendulum",
        )
    )


class TestInverseDynamics:
    def test_round_trip_recovers_a_known_torque_profile(self) -> None:
        """Forward sim with known torques -> inverse dynamics recovers
        them (exact up to RK4 + central-difference error)."""
        p = _params()
        start = PendulumState(theta1=-math.pi / 2, theta2=0.1, omega1=0.0, omega2=0.0)

        def torque(t: float) -> tuple[float, float]:
            return 5.0 * math.sin(3.0 * t), 2.0 * math.cos(5.0 * t)

        states = simulate_forced(p, start, _G_FLAT, _DT, 1500, torque)
        result = inverse_dynamics(p, states, _G_FLAT, _DT)
        t = np.arange(states.shape[0]) * _DT
        expected = np.stack([5.0 * np.sin(3.0 * t), 2.0 * np.cos(5.0 * t)], axis=1)
        # Interior samples use central differences (O(dt^2)); the two
        # one-sided end samples are looser.
        np.testing.assert_allclose(result["applied"][2:-2], expected[2:-2], atol=1e-3)
        np.testing.assert_allclose(result["applied"], expected, atol=0.05)

    def test_passive_swing_has_near_zero_applied_torque(self) -> None:
        p = _params()
        start = PendulumState(theta1=-math.pi / 2, theta2=0.0, omega1=0.0, omega2=0.0)
        states = reference.simulate(p, start, _G_FLAT, _DT, 1500)
        result = inverse_dynamics(p, states, _G_FLAT, _DT)
        assert np.abs(result["applied"]).max() < 0.05

    def test_breakdown_identity_holds(self) -> None:
        p = _params()
        start = PendulumState(theta1=-1.0, theta2=0.3, omega1=0.0, omega2=0.0)
        states = reference.simulate(p, start, _G_FLAT, _DT, 500)
        r = inverse_dynamics(p, states, _G_FLAT, _DT)
        np.testing.assert_allclose(
            r["applied"], r["inertial"] - r["gravity"] - r["damping"], atol=1e-12
        )

    def test_rejects_bad_inputs(self) -> None:
        p = _params()
        with pytest.raises(Exception, match="N>=3"):
            inverse_dynamics(p, np.zeros((2, 4)), _G_FLAT, _DT)
        with pytest.raises(Exception, match="dt"):
            inverse_dynamics(p, np.zeros((10, 4)), _G_FLAT, 0.0)


class TestEnergyConsistency:
    def test_applied_power_integrates_to_energy_change_undamped(self) -> None:
        """∫ Σ τ_i·ω_i dt ≈ ΔE for the undamped forced pendulum."""
        p = _params(d1=0.0, d2=0.0)
        start = PendulumState(theta1=-1.2, theta2=0.2, omega1=0.0, omega2=0.0)

        def torque(t: float) -> tuple[float, float]:
            return 4.0 * math.sin(2.0 * t), 1.5 * math.sin(4.0 * t)

        states = simulate_forced(p, start, _G_FLAT, _DT, 2000, torque)
        t = np.arange(states.shape[0]) * _DT
        tau = np.stack([4.0 * np.sin(2.0 * t), 1.5 * np.sin(4.0 * t)], axis=1)
        power = np.sum(tau * states[:, 2:], axis=1)
        work = float(np.trapezoid(power, dx=_DT))

        def energy(row: np.ndarray) -> float:
            return reference.total_energy(
                p,
                PendulumState(
                    theta1=row[0], theta2=row[1], omega1=row[2], omega2=row[3]
                ),
                _G_FLAT,
            )

        delta_e = energy(states[-1]) - energy(states[0])
        assert work == pytest.approx(delta_e, abs=1e-3)

    def test_net_joint_power_integrates_to_kinetic_energy_change(
        self, pendulum_run
    ) -> None:  # type: ignore[no-untyped-def]
        """Σ power_w = d(KE)/dt for the passive swing (passivity of the
        Coriolis term), so its integral matches ΔKE."""
        series = compute_kinetics(pendulum_run)
        assert series is not None
        p = PendulumParameters.golf_default()

        # Rebuild joint rates from the source to evaluate KE directly.
        from rate_of_closure.simulation.sources import make_source

        source = make_source(
            "double_pendulum",
            pendulum_run.config.scenario,
            plane=pendulum_run.config.plane,
            duration=pendulum_run.config.swing_duration_s,
        )

        def kinetic(t: float) -> float:
            s = source.inner.state_at(min(t, source.inner.duration))  # type: ignore[attr-defined]
            m = reference.mass_matrix(p, s.theta2)
            qd = np.array([s.omega1, s.omega2])
            return float(0.5 * qd @ m @ qd)

        total_power = series.power_w.sum(axis=1)
        work = float(np.trapezoid(total_power, series.t))
        delta_ke = kinetic(float(series.t[-1])) - kinetic(float(series.t[0]))
        assert work == pytest.approx(delta_ke, abs=0.05)


class TestZeroTorqueCounterfactual:
    def test_unlocked_pointwise_identity_matches_passive_loads(self) -> None:
        p = _params()
        start = PendulumState(theta1=-1.1, theta2=0.25, omega1=0.8, omega2=-0.3)
        states = simulate_forced(
            p,
            start,
            _G_FLAT,
            _DT,
            50,
            lambda _t: (8.0, -3.0),
        )
        ztcf = zero_torque_counterfactual(p, states, _G_FLAT)
        inverse = inverse_dynamics(p, states, _G_FLAT, _DT)

        # With no constraints and zero commanded torque, the net inertial
        # torque equals the passive gravity+damping generalized torque at the
        # exact same state, independent of the torque that created that state.
        np.testing.assert_allclose(
            ztcf["inertial_torque"],
            inverse["gravity"] + inverse["damping"],
            atol=1e-12,
        )
        assert ztcf["acceleration"].shape == (51, 2)
        assert np.isfinite(ztcf["acceleration"]).all()

    def test_locked_coordinate_keeps_zero_counterfactual_acceleration(self) -> None:
        p = _params()
        states = np.tile((-1.0, 0.2, 0.0, -0.4), (4, 1))
        ztcf = zero_torque_counterfactual(
            p,
            states,
            _G_FLAT,
            locked=(True, False),
        )
        np.testing.assert_array_equal(ztcf["acceleration"][:, 0], 0.0)
        assert np.any(np.abs(ztcf["acceleration"][:, 1]) > 0.0)

    def test_rejects_nonfinite_or_malformed_states(self) -> None:
        p = _params()
        with pytest.raises(Exception, match="N>=1"):
            zero_torque_counterfactual(p, np.zeros((0, 4)), _G_FLAT)
        bad = np.zeros((1, 4))
        bad[0, 2] = np.nan
        with pytest.raises(Exception, match="finite"):
            zero_torque_counterfactual(p, bad, _G_FLAT)


class TestReactionForces:
    def test_static_hang_supports_the_weight(self) -> None:
        """At rest hanging straight down, the shoulder carries the full
        weight and the wrist the club's weight (sign convention pin)."""
        p = _params(d1=0.0, d2=0.0)
        rest = PendulumState(theta1=0.0, theta2=0.0, omega1=0.0, omega2=0.0)
        states = np.tile((rest.theta1, rest.theta2, rest.omega1, rest.omega2), (5, 1))
        from rate_of_closure.simulation.kinetics import _reaction_forces

        alpha = np.zeros((5, 2))
        f_shoulder, f_wrist, f_head = _reaction_forces(
            p, states[:, :2], states[:, 2:], alpha, _G_FLAT, 0.2
        )
        g = 9.80665
        # In-plane frame: local y up; supporting force is +y.
        np.testing.assert_allclose(f_shoulder[0], [0.0, (p.m1 + p.m2) * g], atol=1e-9)
        np.testing.assert_allclose(f_wrist[0], [0.0, p.m2 * g], atol=1e-9)
        np.testing.assert_allclose(f_head[0], [0.0, 0.2 * g], atol=1e-9)

    def test_geometry_is_consistent_with_the_run(self, pendulum_run) -> None:  # type: ignore[no-untyped-def]
        series = compute_kinetics(pendulum_run)
        assert series is not None
        p = PendulumParameters.golf_default()
        arm = np.linalg.norm(series.wrist_positions_m - series.pivot_position_m, axis=1)
        club = np.linalg.norm(
            series.clubhead_positions_m - series.wrist_positions_m, axis=1
        )
        np.testing.assert_allclose(arm, p.l1, atol=1e-9)
        np.testing.assert_allclose(club, p.l2, atol=1e-6)


class TestRunIntegration:
    def test_series_aligns_with_the_run_grid(self, pendulum_run) -> None:  # type: ignore[no-untyped-def]
        series = compute_kinetics(pendulum_run)
        assert series is not None
        assert series.joint_names == KINETIC_JOINT_NAMES
        n = pendulum_run.swing_times.shape[0]
        assert series.t.shape == (n,)
        assert series.torque_gravity_nm.shape == (n, 2)
        assert series.ztcf_acceleration_rad_s2.shape == (n, 2)
        assert series.ztcf_inertial_torque_nm.shape == (n, 2)
        assert series.ztcf_shoulder_force_n.shape == (n, 3)
        assert np.isfinite(series.ztcf_inertial_torque_nm).all()
        assert np.isfinite(series.power_w).all()
        assert series.impact_time_s == pendulum_run.impact_time_s

    def test_unsupported_sources_return_none(self) -> None:
        run = run_simulation(
            SimulationConfig(
                scenario=ImpactScenario(clubhead_speed_mph=113.0),
                club=get_club("Driver 10.5°"),
            )
        )
        assert compute_kinetics(run) is None
        assert kinetics_for_run(run) is None

    def test_cache_returns_the_same_object(self, pendulum_run) -> None:  # type: ignore[no-untyped-def]
        first = kinetics_for_run(pendulum_run)
        assert kinetics_for_run(pendulum_run) is first


class TestWebParityFixture:
    def test_checked_in_fixture_matches_the_python_kinetics(self) -> None:
        """The TS mirror is pinned against this fixture; regenerate it
        from this exact recipe when the kinetics change."""
        import json
        from pathlib import Path

        from rate_of_closure.simulation.kinetics import _reaction_forces

        fixture_path = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "rate_of_closure"
            / "web"
            / "src"
            / "model"
            / "__fixtures__"
            / "kinetics_parity.json"
        )
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        plan = payload["plan"]
        p = PendulumParameters.golf_default()
        g = reference.in_plane_gravity_from_tilts(
            math.radians(plan["planeYawDeg"]),
            math.radians(plan["planeSideTiltDeg"]),
            math.radians(plan["planeForwardTiltDeg"]),
            9.80665,
        )
        np.testing.assert_allclose(g, plan["gInplane"], rtol=1e-12)
        initial = plan["initialState"]
        states = reference.simulate(
            p,
            PendulumState(
                theta1=initial[0],
                theta2=initial[1],
                omega1=initial[2],
                omega2=initial[3],
            ),
            g,
            plan["dtS"],
            plan["nSteps"],
        )
        r = inverse_dynamics(p, states, g, plan["dtS"])
        fs, fw, fh = _reaction_forces(
            p, states[:, :2], states[:, 2:], r["alpha"], g, 0.2
        )
        for sample in payload["samples"]:
            i = sample["index"]
            expected = {
                "shoulderTorqueNm": r["inertial"][i, 0],
                "wristTorqueNm": r["inertial"][i, 1],
                "shoulderGravityTorqueNm": r["gravity"][i, 0],
                "wristGravityTorqueNm": r["gravity"][i, 1],
                "shoulderDampingTorqueNm": r["damping"][i, 0],
                "wristDampingTorqueNm": r["damping"][i, 1],
                "shoulderPowerW": r["inertial"][i, 0] * states[i, 2],
                "wristPowerW": r["inertial"][i, 1] * states[i, 3],
                "shoulderForceN": float(np.linalg.norm(fs[i])),
                "wristForceN": float(np.linalg.norm(fw[i])),
                "clubheadForceN": float(np.linalg.norm(fh[i])),
            }
            for key, value in expected.items():
                assert sample[key] == pytest.approx(value, rel=1e-12, abs=1e-12), (
                    key,
                    i,
                )


class TestCatalogRegistration:
    def test_kinetics_category_is_registered(self) -> None:
        keys = [spec.key for spec in variables_by_category("Kinetics")]
        assert keys == [
            "kinetics.shoulder_torque_nm",
            "kinetics.wrist_torque_nm",
            "kinetics.shoulder_gravity_torque_nm",
            "kinetics.wrist_gravity_torque_nm",
            "kinetics.shoulder_damping_torque_nm",
            "kinetics.wrist_damping_torque_nm",
            "kinetics.shoulder_ztcf_torque_nm",
            "kinetics.wrist_ztcf_torque_nm",
            "kinetics.shoulder_power_w",
            "kinetics.wrist_power_w",
            "kinetics.shoulder_force_n",
            "kinetics.wrist_force_n",
            "kinetics.clubhead_force_n",
            "kinetics.shoulder_ztcf_force_n",
            "kinetics.wrist_ztcf_force_n",
            "kinetics.clubhead_ztcf_force_n",
        ]
        for key in keys:
            assert CATALOG[key].is_series, key

    def test_units_follow_the_movement_optimizer_convention(self) -> None:
        """Middle-dot N·m, never 'Nm' (plot_renderer.py authority)."""
        for spec in variables_by_category("Kinetics"):
            assert spec.unit in ("N·m", "W", "N"), spec.key

    def test_extractors_are_finite_for_a_pendulum_run(self, pendulum_run) -> None:  # type: ignore[no-untyped-def]
        n = pendulum_run.swing_times.shape[0]
        for spec in variables_by_category("Kinetics"):
            values = extract(pendulum_run, spec.key)
            assert isinstance(values, np.ndarray)
            assert values.shape == (n,), spec.key
            assert np.isfinite(values).all(), spec.key
