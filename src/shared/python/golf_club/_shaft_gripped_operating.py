"""Explicit constant-input prescriptions for local gripped-shaft linear models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._grip_contracts import finite_array
from ._rotating_body_contracts import RotatingFrameState
from ._shaft_affine_response import (
    AffineResponseAssessment,
    ForcedResponseControls,
    assess_affine_response,
)
from ._shaft_affine_transient import affine_state_at
from ._shaft_damped_spectrum import DampedPencil
from ._shaft_equilibrium import EquilibriumControls
from ._shaft_gripped_chain import GrippedSectionChain
from ._shaft_gripped_coordinates import scaled_gripped_operators
from ._shaft_gripped_dynamics import balanced_gripped_dynamics
from ._shaft_rotating_chain import RotatingSectionChain
from ._shaft_se3 import _rigid_pose
from ._shaft_spectrum import SpectrumScales
from ._validation import require_identifier

_Pose = tuple[tuple[float, ...], ...]


def _require_constant_frame(frame: RotatingFrameState) -> None:
    if not isinstance(frame, RotatingFrameState):
        raise TypeError("frame must be RotatingFrameState")
    if any(value != 0 for value in frame.angular_acceleration_rad_s2):
        raise ValueError("constant angular velocity requires zero angular acceleration")


def _owned_poses(poses: object, count: int) -> tuple[_Pose, ...]:
    current = finite_array(poses, (count, 4, 4), "reference poses")
    return tuple(
        tuple(tuple(float(value) for value in row) for row in _rigid_pose(pose))
        for pose in current
    )


def _source_ids(value: object) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError("grip sources must be an ordered sequence of identifiers")
    if not value:
        raise ValueError("at least one grip source is required")
    return tuple(require_identifier(item, "grip source") for item in value)


@dataclass(frozen=True)
class ConstantGrippedModel:
    """Owned local affine model under explicit constant operating prescriptions.

    Observer angular velocity and observer-resolved origin acceleration are
    prescribed constant; its orientation need not be. Grip anchor poses and
    observer-resolved spatial force/couple laws are held constant. This is a
    selected model, not inferred history or an actual player's impedance.
    Direct record construction validates shape/types, not assembly provenance.
    """

    pencil: DampedPencil
    scaled_residual: tuple[float, ...]
    scales: SpectrumScales
    frame: RotatingFrameState
    reference_poses: tuple[_Pose, ...]
    grip_source_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.pencil, DampedPencil) or not isinstance(
            self.scales, SpectrumScales
        ):
            raise TypeError("expected DampedPencil and SpectrumScales")
        _require_constant_frame(self.frame)
        size = np.asarray(self.pencil.mass).shape[0]
        if size % 6:
            raise ValueError("operating model requires complete six-axis nodes")
        residual = finite_array(self.scaled_residual, (size,), "scaled residual")
        object.__setattr__(self, "scaled_residual", tuple(float(x) for x in residual))
        object.__setattr__(
            self, "reference_poses", _owned_poses(self.reference_poses, size // 6)
        )
        object.__setattr__(self, "grip_source_ids", _source_ids(self.grip_source_ids))

    @property
    def input_scope(self) -> str:
        return "constant_observer_inputs_and_anchor_poses"

    @property
    def stability_status(self) -> str:
        """No nonlinear or physical stability follows from a local affine model."""
        return "unqualified"

    def assess_response(
        self, controls: ForcedResponseControls
    ) -> AffineResponseAssessment:
        """Use the same scaled residual, pencil and coordinate/error conventions."""
        return assess_affine_response(
            self.pencil, self.scaled_residual, self.scales, controls
        )

    def scaled_state_at(
        self, initial_state: object, time_s: object
    ) -> tuple[float, ...]:
        """Return x=(y,T*ydot) under this model's explicit constant prescriptions.

        This is local linear motion; departure from the recorded reference
        strain domain and nonlinear or physical stability remain unqualified.
        """
        state: tuple[float, ...] = affine_state_at(
            self.pencil, self.scaled_residual, self.scales, initial_state, time_s
        )
        return state


def constant_gripped_model(
    chain: GrippedSectionChain,
    poses: object,
    equilibrium: EquilibriumControls,
    scales: SpectrumScales,
) -> ConstantGrippedModel:
    """Prescribe constant inputs and recheck strain/balance without zeroing r.

    Calling this constructor selects a constant-input model; a zero-alpha
    snapshot alone does not establish its history. Nonzero angular acceleration
    is incompatible and refused exactly. For rotating R(t), constant observer
    acceleration a0 means physical origin acceleration R(t)a0. Anchor poses and
    existing observer-resolved forces/couples are prescribed for the interval.
    This does not bound nonlinear departure from the recorded reference shape.
    """
    if not isinstance(chain, GrippedSectionChain) or not isinstance(
        chain.shaft, RotatingSectionChain
    ):
        raise TypeError("constant model requires a gripped rotating chain")
    _require_constant_frame(chain.shaft.frame)
    operators = balanced_gripped_dynamics(chain, poses, equilibrium)
    pencil, _, residual = scaled_gripped_operators(operators, scales)
    return ConstantGrippedModel(
        pencil,
        tuple(float(x) for x in residual),
        scales,
        operators.frame,
        _owned_poses(poses, chain.node_count),
        tuple(item.grip.source_id for item in chain.grips),
    )


__all__ = ()
