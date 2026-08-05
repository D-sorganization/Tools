"""Simulation session for the Rate of Closure explorer (epic #4103).

Orchestrates a full swing -> impact -> flight run:

* :mod:`.sources` — app-frame swing sources (manual constant-twist,
  double pendulum via ``swing_sim``, and a planar triple pendulum).
* :mod:`.session` — the :class:`~rate_of_closure.simulation.session.
  SimulationRun` record and the orchestration entry points, including
  both legacy delivery inspection and fixed-ball contact outcomes.
* :mod:`.isa` — thin adapter over the rotation converter's screw-axis
  extraction (one file to touch when the Rust surface lands, #4108).
* :mod:`.kinetics` — inverse dynamics over the pendulum swing: joint
  torques, reaction forces, and powers per sample (#4125 H2).
* :mod:`.export` — CSV time-series and JSON summary export.
"""

from __future__ import annotations

from .contact import ContactMode, ImpactOutcome, ImpactStatus
from .export import (
    ball_setup_from_json_dict,
    run_to_json_dict,
    write_csv,
    write_json,
    write_torque_csv,
)
from .flight_explorer import (
    EXPLORER_METRIC_KEYS,
    FlightExploration,
    explore_flight,
    launch_from_delivery,
    launch_from_direct,
)
from .isa import screw_axis_samples
from .kinetics import (
    KINETIC_JOINT_NAMES,
    KineticsSeries,
    compute_kinetics,
    inverse_dynamics,
    kinetics_for_run,
    simulate_forced,
    zero_torque_counterfactual,
)
from .session import (
    BALL_POSITION_M,
    DEFAULT_DRIVER_TEE_HEIGHT_M,
    BallSetup,
    BallSupportMode,
    SimulationConfig,
    SimulationRun,
    delivery_at,
    run_simulation,
)
from .sources import (
    SOURCE_KINDS,
    AppFrameSwing,
    ManualSwingSource,
    TriplePendulumParameters,
    TriplePendulumSwing,
    make_source,
)
from .torque_history import fit_run_torque_profile

__all__ = [
    "BALL_POSITION_M",
    "DEFAULT_DRIVER_TEE_HEIGHT_M",
    "BallSetup",
    "BallSupportMode",
    "ball_setup_from_json_dict",
    "ContactMode",
    "EXPLORER_METRIC_KEYS",
    "KINETIC_JOINT_NAMES",
    "SOURCE_KINDS",
    "AppFrameSwing",
    "FlightExploration",
    "KineticsSeries",
    "ImpactOutcome",
    "ImpactStatus",
    "ManualSwingSource",
    "SimulationConfig",
    "SimulationRun",
    "TriplePendulumParameters",
    "TriplePendulumSwing",
    "compute_kinetics",
    "delivery_at",
    "explore_flight",
    "fit_run_torque_profile",
    "inverse_dynamics",
    "kinetics_for_run",
    "simulate_forced",
    "zero_torque_counterfactual",
    "launch_from_delivery",
    "launch_from_direct",
    "make_source",
    "run_simulation",
    "run_to_json_dict",
    "screw_axis_samples",
    "write_csv",
    "write_json",
    "write_torque_csv",
]
