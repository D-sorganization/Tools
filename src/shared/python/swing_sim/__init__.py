"""Shared swing simulation package (epic #4103, P0 foundation #4104).

Curated façade for the swing → impact → ball-flight platform's swing stage:
value types, the :class:`SwingSource` protocol, the double-pendulum swing
source, and backend discovery. Physics kernels live in the
``rust_core/swing-core`` Rust crate (Python wheel ``swing_core``, WASM NPM
package for the web app); :mod:`.reference` is the pure-Python parity
oracle and one-shot fallback.
"""

from __future__ import annotations

from ._rust_facade import rust_available
from .force_attribution import (
    ATTRIBUTION_SCHEMA_VERSION,
    AttributionComponent,
    AttributionProvider,
    ComponentHistory,
    ComponentMetrics,
    DoublePendulumAttributionProvider,
    StateAttribution,
    TrajectoryAttribution,
    attribute_state,
    attribute_trajectory,
    component_impulse_objective,
)
from .run_config import (
    DOUBLE_PENDULUM_JOINT_IDS,
    DOUBLE_PENDULUM_MODEL_ID,
    SHOULDER_JOINT_ID,
    WRIST_JOINT_ID,
    DoublePendulumRunConfig,
    JointLockConfig,
    LocalizedTorqueOffset,
    SwingRunMode,
)
from .swing_source import DoublePendulumSwing, SwingSource
from .torque_export import fit_torque_history_profile
from .torque_fitting import fit_torque_polynomial
from .torque_library import TorqueProfileLibrary
from .torque_profiles import (
    COEFFICIENT_ORDER,
    TORQUE_PROFILE_SCHEMA_VERSION,
    TORQUE_UNIT,
    FitMetadata,
    JointTorqueAssignment,
    PrescribedTorqueProfile,
    TorquePolynomial,
    TorqueProfileSource,
    evaluate_ascending_polynomial,
)
from .types import (
    DEFAULT_GRAVITY_M_S2,
    PendulumParameters,
    PendulumState,
    PlaneOrientation,
    SwingSample,
    SwingTrajectory,
)

__version__ = "0.2.0"

__all__ = [
    "DEFAULT_GRAVITY_M_S2",
    "ATTRIBUTION_SCHEMA_VERSION",
    "AttributionComponent",
    "AttributionProvider",
    "COEFFICIENT_ORDER",
    "DOUBLE_PENDULUM_JOINT_IDS",
    "DOUBLE_PENDULUM_MODEL_ID",
    "DoublePendulumSwing",
    "DoublePendulumAttributionProvider",
    "DoublePendulumRunConfig",
    "FitMetadata",
    "ComponentHistory",
    "ComponentMetrics",
    "JointTorqueAssignment",
    "JointLockConfig",
    "LocalizedTorqueOffset",
    "PendulumParameters",
    "PendulumState",
    "PlaneOrientation",
    "PrescribedTorqueProfile",
    "SwingSample",
    "SwingSource",
    "SwingTrajectory",
    "StateAttribution",
    "TrajectoryAttribution",
    "SwingRunMode",
    "SHOULDER_JOINT_ID",
    "TORQUE_UNIT",
    "TORQUE_PROFILE_SCHEMA_VERSION",
    "TorquePolynomial",
    "TorqueProfileLibrary",
    "TorqueProfileSource",
    "WRIST_JOINT_ID",
    "__version__",
    "attribute_state",
    "attribute_trajectory",
    "component_impulse_objective",
    "evaluate_ascending_polynomial",
    "fit_torque_polynomial",
    "fit_torque_history_profile",
    "rust_available",
]
