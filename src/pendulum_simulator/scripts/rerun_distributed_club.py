"""Re-run the swing-objective results on the distributed club (Tools #5523).

Reproduces every number republished in ``docs/specs/SWING_ACTUATION_AND_REALISM.md``
and ``docs/specs/SWING_OBJECTIVE_COMPARISON.md`` after the canonical distributed
club convention (#5431) replaced the lumped tip mass.  Nothing here is tuned: each
section calls the package's own functions and prints the result.

Usage (from ``src/pendulum_simulator``)::

    PYTHONPATH=src python scripts/rerun_distributed_club.py SECTION [SECTION ...]

Sections: ``club``, ``bracket``, ``masses``, ``frontier``, ``actuation``,
``ranking``, ``comparison``, ``rollout``, ``all``.  Output is JSON lines on stdout.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace

import numpy as np
from scipy.integrate import solve_ivp

from double_pendulum_golf.physics import (
    JointLimits,
    PendulumParams,
    TorqueClamp,
    club_inertia,
    equations_of_motion,
)
from double_pendulum_golf.swing_objectives.actuation import (
    JointActuation,
    SwingActuation,
    tour_hub_actuation,
    tour_wrist_actuation,
)
from double_pendulum_golf.swing_objectives.club_equivalence import (
    DRIVER_SPEC,
    coupling_error,
    distributed_coupling_error,
    equivalent_tip_mass,
    wrist_inertia,
)
from double_pendulum_golf.swing_objectives.comparison import compare_objectives
from double_pendulum_golf.swing_objectives.downswing import DownswingConfig
from double_pendulum_golf.swing_objectives.impact_optimality import (
    energy_optimal_rates,
    impact_hand_speed_coefficient,
    optimal_hand_speed_sign,
)
from double_pendulum_golf.swing_objectives.model_adequacy import (
    _analyse,
    hand_speed_frontier,
)
from double_pendulum_golf.swing_objectives.objective_realism import (
    objective_realism_ranking,
)
from double_pendulum_golf.swing_objectives.presets import (
    DEFAULT_PRESET,
    build_config,
)

_L1, _L2 = 0.65, 1.10


def emit(section: str, **fields: object) -> None:
    """Print one JSON line."""
    print(json.dumps({"section": section, **fields}, default=float), flush=True)


def lumped(head_kg: float) -> PendulumParams:
    """Head-only (m2 -> 0) limit: the pre-#5431 lumped tip-mass club."""
    return PendulumParams(m1=5.0, m2=1e-15, L1=_L1, L2=_L2, mClub=head_kg)


def lumped_config(head_kg: float, **overrides: object) -> DownswingConfig:
    """Preset config with the club replaced by a lumped tip mass."""
    return replace(build_config(), params=lumped(head_kg), **overrides)


def legacy_config(**overrides: object) -> DownswingConfig:
    """The historical 0.36 s / 250 N m frontier configuration (lumped 0.50 kg)."""
    base = dict(
        params=lumped(0.50),
        node_count=17,
        duration_s=0.36,
        initial_state=np.array([2.618, 1.745, 0.0, 0.0]),
        impact_theta1_rad=0.0,
        torque_clamp=TorqueClamp(max_torque1=250.0, max_torque2=20.0),
        joint_limits=JointLimits(
            phi_min=-0.175, phi_max=2.094, theta1_min=-4.0, theta1_max=4.0
        ),
    )
    base.update(overrides)
    return DownswingConfig(**base)  # type: ignore[arg-type]


def section_club() -> None:
    """Inertia and coupling about the wrist: real, lumped and distributed."""
    params = DEFAULT_PRESET.to_params()
    shipped = club_inertia(params)
    emit(
        "club",
        real_inertia_wrist=wrist_inertia(DRIVER_SPEC),
        real_mu=DRIVER_SPEC.mass_kg * _L1 * DRIVER_SPEC.com_m,
        tip_031_inertia=0.310 * _L2**2,
        tip_031_mu=0.310 * _L1 * _L2,
        tip_050_inertia=0.500 * _L2**2,
        tip_050_mu=0.500 * _L1 * _L2,
        equiv_tip_mass=equivalent_tip_mass(DRIVER_SPEC, _L2),
        equiv_coupling_error=coupling_error(DRIVER_SPEC, _L1, _L2),
        shaft_mass=params.m2,
        head_mass=params.mClub,
        shipped_mass=shipped.mass_kg,
        shipped_com=shipped.com_m,
        shipped_icm=shipped.icm_kg_m2,
        shipped_inertia_wrist=shipped.icm_kg_m2 + shipped.mass_kg * shipped.com_m**2,
        shipped_mu=(params.m2 / 2 + params.mClub) * _L2 * _L1,
        distributed_coupling_error=distributed_coupling_error(DRIVER_SPEC, _L1, _L2),
    )


def section_bracket() -> None:
    """impact_optimality bracket with the true club COM and Icm."""
    cases = (
        ("lumped_0.50", lumped(0.50), {}),
        ("shipped_distributed", DEFAULT_PRESET.to_params(), {}),
        (
            "real_driver_spec_L2_1.143",
            PendulumParams(m1=5.0, m2=0.30, L1=_L1, L2=DRIVER_SPEC.length_m, mClub=0.01),
            dict(
                club_mass_kg=DRIVER_SPEC.mass_kg,
                club_com_m=DRIVER_SPEC.com_m,
                club_inertia_kgm2=DRIVER_SPEC.inertia_about_com_kgm2,
            ),
        ),
        (
            "legacy_doc_row_r2_0.89_I2_0.043",
            PendulumParams(m1=5.0, m2=0.30, L1=_L1, L2=1.143, mClub=0.01),
            dict(club_mass_kg=0.31, club_com_m=0.89, club_inertia_kgm2=0.043),
        ),
    )
    for label, params, kw in cases:
        coefficient = impact_hand_speed_coefficient(params, **kw)
        emit(
            "bracket",
            model=label,
            coefficient=coefficient,
            bracket=coefficient / params.L1,
            regime=optimal_hand_speed_sign(params, **kw),
        )
    for label, params in (
        ("lumped_0.50", lumped(0.50)),
        ("shipped_distributed", DEFAULT_PRESET.to_params()),
    ):
        rates = energy_optimal_rates(params, kinetic_energy_j=1000.0)
        emit(
            "bracket_rates",
            model=label,
            kinetic_energy_j=1000.0,
            arm_rate=rates.arm_rate_rad_s,
            uncock_rate=rates.uncock_rate_rad_s,
            hand_speed=rates.hand_speed_m_s,
            clubhead_speed=rates.clubhead_speed_m_s,
            hand_over_clubhead=rates.hand_speed_m_s / rates.clubhead_speed_m_s,
        )


def _point(label: str, config: DownswingConfig, floor: float = 0.0) -> None:
    point = _analyse(config, floor)
    emit("point", label=label, floor=floor, **asdict(point))


def section_masses() -> None:
    """Club sweep at the revised 0.28 s comparison duration."""
    for head in (0.500, 0.320, 0.238):
        _point(f"lumped_{head:.3f}kg_tip", lumped_config(head))
    _point("distributed_driver_preset", build_config())


def section_frontier() -> None:
    """hand_speed_frontier: legacy mis-specified club and the corrected preset."""
    legacy = hand_speed_frontier(legacy_config(), floors_ms=[0.0, 3.0, 5.0, 6.0])
    for point in legacy.points:
        emit("frontier_legacy_0.36s", **asdict(point))
    corrected = hand_speed_frontier(build_config(), floors_ms=[0.0, 6.0, 8.0])
    for point in corrected.points:
        emit("frontier_distributed_0.28s", **asdict(point))
    emit(
        "frontier_reaches_band",
        legacy=legacy.reaches_measured_hand_speed,
        distributed=corrected.reaches_measured_hand_speed,
    )


def section_actuation() -> None:
    """Actuation variants on the legacy lumped club and on the distributed preset."""
    hub, wrist = tour_hub_actuation(), tour_wrist_actuation()
    weak_brake = JointActuation(
        hub.peak_torque_nm, 1.0e6, 0.0, hub.brake_fraction, hub.eccentric_gain
    )
    hill_only = replace(hub, brake_fraction=1.0, eccentric_gain=1.0)
    variants = {
        "symmetric_clamp": None,
        "weak_brake_only": SwingActuation(hub=weak_brake, wrist=wrist),
        "hill_only": SwingActuation(hub=hill_only, wrist=wrist),
        "hill_plus_weak_brake": SwingActuation(hub=hub, wrist=wrist),
    }
    for name, actuation in variants.items():
        _point(f"actuation_legacy_lumped_0.36s_{name}", legacy_config(actuation=actuation))
        _point(
            f"actuation_distributed_0.28s_{name}", replace(build_config(), actuation=actuation)
        )


def _ranking(label: str, config: DownswingConfig) -> None:
    ranking = objective_realism_ranking(config)
    emit(
        "ranking",
        label=label,
        spread=ranking.deviation_spread,
        mean=ranking.mean_deviation,
        discriminating=ranking.is_discriminating,
        reaches=ranking.reaches_measured_behaviour,
        band_count=ranking.band_count,
        entries=[
            dict(
                key=e.key,
                feasible=e.feasible,
                deviation=e.total_deviation,
                inside=e.inside_count,
                worst=e.worst_observable,
                clubhead=e.clubhead_speed_ms,
                observables=e.observables,
            )
            for e in ranking.entries
        ],
    )


def section_ranking() -> None:
    """Realism ranking: legacy lumped club (0.36 s) and the distributed preset."""
    _ranking("legacy_unconstrained_0.36s", legacy_config())
    _ranking("legacy_hands_ge_3_0.36s", legacy_config(min_hand_speed_ms=3.0))
    _ranking("distributed_unconstrained_0.28s", build_config())
    _ranking("distributed_hands_ge_3_0.28s", replace(build_config(), min_hand_speed_ms=3.0))


def section_comparison() -> None:
    """Optimizer comparison matrix and cross-evaluation on the preset."""
    config = build_config()
    comparison = compare_objectives(config)
    emit(
        "comparison",
        keys=list(comparison.objective_keys),
        matrix=comparison.matrix.round(2).tolist(),
        raw=comparison.raw_values,
        degenerate=comparison.is_degenerate,
        max_swing_distance=comparison.max_swing_distance,
        t_min_s=config.minimum_sweep_duration_s,
        diagnostics=comparison.diagnostics,
    )
    legacy = compare_objectives(lumped_config(0.50, duration_s=0.36, node_count=17))
    emit(
        "comparison_legacy_lumped_0.36s",
        keys=list(legacy.objective_keys),
        matrix=legacy.matrix.round(2).tolist(),
        degenerate=legacy.is_degenerate,
        max_swing_distance=legacy.max_swing_distance,
    )


def _rollout(label: str, config: DownswingConfig) -> None:
    """Free rollout at full hub drive, wrist torque zero, no joint limits."""
    hub_nm = -config.torque_clamp.max_torque1

    def rhs(t: float, state: np.ndarray) -> np.ndarray:
        return equations_of_motion(state, t, config.params, lambda _t: np.array([hub_nm, 0.0]))

    sol = solve_ivp(rhs, (0.0, config.duration_s), config.initial_state, rtol=1e-9, atol=1e-11)
    cock_deg = np.degrees(sol.y[1])
    emit(
        "rollout",
        label=label,
        duration_s=config.duration_s,
        hub_torque_nm=hub_nm,
        cock_start_deg=cock_deg[0],
        cock_end_deg=cock_deg[-1],
        cock_max_deg=cock_deg.max(),
        released=bool(cock_deg.min() <= 0.0),
    )


def section_rollout() -> None:
    """Does full hub drive release the club? Lumped and distributed."""
    _rollout("legacy_lumped_0.50_0.36s", legacy_config())
    _rollout("legacy_lumped_0.50_0.28s", lumped_config(0.50))
    _rollout("distributed_preset_0.28s", build_config())


SECTIONS = {
    "club": section_club,
    "bracket": section_bracket,
    "masses": section_masses,
    "frontier": section_frontier,
    "actuation": section_actuation,
    "ranking": section_ranking,
    "comparison": section_comparison,
    "rollout": section_rollout,
}


def main(argv: list[str]) -> int:
    """Run the requested sections."""
    names = argv[1:] or ["all"]
    if "all" in names:
        names = list(SECTIONS)
    for name in names:
        SECTIONS[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
