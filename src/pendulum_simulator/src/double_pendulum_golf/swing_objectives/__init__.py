"""Mechanism-vs-outcome swing optimization for the double pendulum golf model.

The simulator can already optimize a swing for clubhead speed. This subpackage
adds the ability to optimize instead for the *mechanisms* golf coaching invokes —
centrifugal release, Coriolis kinetic-chain transfer, grip-force energy transfer,
grip-force impulse — under an identical torque budget, and to compare the
resulting swings against the speed-optimal one.

It is built entirely on the existing :mod:`double_pendulum_golf.physics` kernel
and the :mod:`double_pendulum_golf.transfer_strategy` contract; no equations of
motion are re-derived here.

See ``docs/specs/SWING_OBJECTIVE_COMPARISON.md``. Epic #4766.
"""

from double_pendulum_golf.swing_objectives.actuation import (
    JointActuation,
    SwingActuation,
    tour_hub_actuation,
    tour_wrist_actuation,
)
from double_pendulum_golf.swing_objectives.comparison import (
    COMPARISON_SCHEMA_VERSION,
    SwingComparison,
    compare_objectives,
    comparison_from_payload,
    comparison_to_payload,
    cross_evaluation_matrix,
)
from double_pendulum_golf.swing_objectives.downswing import (
    DownswingConfig,
    DownswingOptimizer,
    DownswingResult,
)
from double_pendulum_golf.swing_objectives.impact_optimality import (
    EnergyOptimalRates,
    energy_optimal_rates,
    impact_hand_speed_coefficient,
    optimal_hand_speed_sign,
)
from double_pendulum_golf.swing_objectives.model_adequacy import (
    FrontierPoint,
    HandSpeedFrontier,
    hand_speed_frontier,
    swing_observables,
)
from double_pendulum_golf.swing_objectives.objectives import (
    CENTRIFUGAL,
    CLUBHEAD_SPEED,
    CORIOLIS,
    ENERGY_TRANSFER,
    IMPULSE_TRANSFER,
    SWING_OBJECTIVES,
    SwingObjective,
    evaluate_all,
    get_objective,
)
from double_pendulum_golf.swing_objectives.reference_kinematics import (
    TOUR_DRIVER_BANDS,
    ObservableBand,
    RealismScore,
    score_against_reference,
)
from double_pendulum_golf.swing_objectives.presets import (
    DEFAULT_PRESET,
    GolferPreset,
    SwingBudget,
    build_config,
)
from double_pendulum_golf.swing_objectives.signals import (
    SwingSignals,
    build_swing_signals,
    generalized_accelerations,
)
from double_pendulum_golf.swing_objectives.velocity_terms import (
    VelocityTerms,
    centrifugal_vector,
    coriolis_only_vector,
    coupling_constant,
    decompose_velocity_terms,
)

__all__ = [
    # Velocity-term decomposition (#4767)
    "VelocityTerms",
    "centrifugal_vector",
    "coriolis_only_vector",
    "coupling_constant",
    "decompose_velocity_terms",
    # Trajectory signals (#4768)
    "SwingSignals",
    "build_swing_signals",
    "generalized_accelerations",
    # Objectives (#4768)
    "SwingObjective",
    "SWING_OBJECTIVES",
    "get_objective",
    "evaluate_all",
    "CLUBHEAD_SPEED",
    "CENTRIFUGAL",
    "CORIOLIS",
    "ENERGY_TRANSFER",
    "IMPULSE_TRANSFER",
    # Downswing optimization (#4769)
    "DownswingConfig",
    "DownswingOptimizer",
    "DownswingResult",
    # Objective comparison (#4770)
    "SwingComparison",
    "compare_objectives",
    "cross_evaluation_matrix",
    "comparison_to_payload",
    "comparison_from_payload",
    "COMPARISON_SCHEMA_VERSION",
    # Presets (#4771)
    "GolferPreset",
    "SwingBudget",
    "DEFAULT_PRESET",
    "build_config",
    # Impact optimality (#4776)
    "EnergyOptimalRates",
    "impact_hand_speed_coefficient",
    "optimal_hand_speed_sign",
    "energy_optimal_rates",
    # Actuation limits (#4777)
    "JointActuation",
    "SwingActuation",
    "tour_hub_actuation",
    "tour_wrist_actuation",
    # Measured reference kinematics (#4778)
    "ObservableBand",
    "RealismScore",
    "TOUR_DRIVER_BANDS",
    "score_against_reference",
    # Model adequacy (#4779)
    "FrontierPoint",
    "HandSpeedFrontier",
    "hand_speed_frontier",
    "swing_observables",
]
