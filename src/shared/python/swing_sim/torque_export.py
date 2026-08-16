"""Fit sampled per-joint torque histories into reusable profiles."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike

from shared.python.contracts import require

from ._torque_profile_validation import stable_id
from .torque_fitting import fit_torque_polynomial
from .torque_profiles import (
    JointTorqueAssignment,
    PrescribedTorqueProfile,
    TorqueProfileSource,
)


def _fit_assignments(
    times_s: ArrayLike,
    histories: Mapping[str, ArrayLike],
    degree: int,
    max_condition_number: float,
) -> tuple[JointTorqueAssignment, ...]:
    """Fit histories in deterministic stable-joint-ID order."""
    require(len(histories) > 0, "torque history mapping must not be empty")
    joint_ids = tuple(stable_id(joint_id, "joint_id") for joint_id in histories)
    assignments: list[JointTorqueAssignment] = []
    for joint_id in sorted(joint_ids):
        polynomial = fit_torque_polynomial(
            times_s,
            histories[joint_id],
            degree,
            max_condition_number=max_condition_number,
        )
        assignments.append(JointTorqueAssignment(joint_id, polynomial))
    return tuple(assignments)


def fit_torque_history_profile(
    *,
    profile_id: str,
    model_id: str,
    name: str,
    description: str,
    source_metadata: Mapping[str, str],
    created_at_utc: str,
    modified_at_utc: str,
    times_s: ArrayLike,
    torque_nm_by_joint: Mapping[str, ArrayLike],
    degree: int,
    max_condition_number: float = 1.0e8,
) -> PrescribedTorqueProfile:
    """Fit a sampled run into one portable ``fitted_run`` profile."""
    require(
        isinstance(torque_nm_by_joint, Mapping),
        "torque_nm_by_joint must be a mapping",
        torque_nm_by_joint,
    )
    assignments = _fit_assignments(
        times_s,
        torque_nm_by_joint,
        degree,
        max_condition_number,
    )
    normalized_times = np.asarray(times_s, dtype=np.float64)
    return PrescribedTorqueProfile(
        profile_id=profile_id,
        model_id=model_id,
        name=name,
        description=description,
        source=TorqueProfileSource.FITTED_RUN,
        source_metadata=source_metadata,
        created_at_utc=created_at_utc,
        modified_at_utc=modified_at_utc,
        time_domain_s=(float(normalized_times[0]), float(normalized_times[-1])),
        assignments=assignments,
    )


__all__ = ["fit_torque_history_profile"]
