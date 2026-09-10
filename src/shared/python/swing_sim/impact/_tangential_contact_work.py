"""Private elastic/Coulomb update with objective history and explicit work.

The caller supplies contact-frame transport and material slip. Contact spin is
a constitutive assumption; this port supplies no trajectory or calibration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ...golf_club._grip_contracts import finite_array
from ...golf_club._validation import (
    Vector3,
    require_finite_float,
    require_identifier,
    require_rotation,
)
from ._spatial_contact_kinematics import _unit_normal, _vector

_TANGENCY_TOLERANCE = 1e-12
_TRANSPORT_TOLERANCE = 1e-10


def _nonnegative(value: object, name: str) -> float:
    result = float(require_finite_float(value, name))
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _tangent(value: object, normal: Vector3, name: str) -> Vector3:
    vector = np.asarray(_vector(value, name))
    component = float(vector @ normal)
    if abs(component) > _TANGENCY_TOLERANCE * math.hypot(*vector):
        raise ValueError(f"{name} must be tangent to the contact normal")
    return _vector(vector - component * np.asarray(normal), name)


@dataclass(frozen=True)
class TangentialContactLaw:
    """Constant SI coefficients; no physical qualification is implied."""

    stiffness_n_per_m: float
    friction_coefficient: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stiffness_n_per_m",
            require_finite_float(
                self.stiffness_n_per_m, "tangential stiffness", positive=True
            ),
        )
        object.__setattr__(
            self,
            "friction_coefficient",
            _nonnegative(self.friction_coefficient, "friction coefficient"),
        )


def _elastic_energy(law: TangentialContactLaw, vector: object) -> float:
    array = np.asarray(vector)
    return float(
        require_finite_float(
            float(0.5 * law.stiffness_n_per_m * (array @ array)), "elastic energy"
        )
    )


@dataclass(frozen=True)
class TangentialContactState:
    """Owned elastic history and unit normal in one named observer frame."""

    law: TangentialContactLaw
    elastic_deflection_m: object
    normal: object
    observer_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.law, TangentialContactLaw):
            raise TypeError("law must be TangentialContactLaw")
        normal = _unit_normal(self.normal, "contact normal")
        object.__setattr__(self, "normal", normal)
        object.__setattr__(
            self,
            "elastic_deflection_m",
            _tangent(self.elastic_deflection_m, normal, "elastic deflection"),
        )
        object.__setattr__(
            self, "observer_id", require_identifier(self.observer_id, "observer_id")
        )
        require_finite_float(self.elastic_energy_j, "elastic energy")

    @property
    def elastic_energy_j(self) -> float:
        return _elastic_energy(self.law, self.elastic_deflection_m)


@dataclass(frozen=True)
class TangentialContactUpdate:
    """End-step work ledger; force on the ball is minus the resisting effort.

    Algorithmic loss includes storage removed when the normal-force cap
    collapses; it is not measured friction, material damping or sound. It must
    converge away in a resolved continuous contact history.
    """

    state: TangentialContactState
    resisting_force_n: Vector3
    delta_stored_energy_j: float
    input_work_j: float
    plastic_dissipation_j: float
    algorithmic_loss_j: float

    def __post_init__(self) -> None:
        if not isinstance(self.state, TangentialContactState):
            raise TypeError("state must be TangentialContactState")
        object.__setattr__(
            self,
            "resisting_force_n",
            _vector(self.resisting_force_n, "resisting force"),
        )
        for name in ("delta_stored_energy_j", "input_work_j"):
            object.__setattr__(
                self, name, require_finite_float(getattr(self, name), name)
            )
        for name in ("plastic_dissipation_j", "algorithmic_loss_j"):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        require_finite_float(self.work_residual_j, "tangential work residual")

    @property
    def work_residual_j(self) -> float:
        return (
            self.input_work_j
            - self.delta_stored_energy_j
            - self.plastic_dissipation_j
            - self.algorithmic_loss_j
        )


def advance_tangential_contact(
    state: TangentialContactState,
    slip_increment_m: object,
    transport_rotation: object,
    normal: object,
    normal_force_n: object,
) -> TangentialContactUpdate:
    """Project transported elastic history onto the current Coulomb disk.

    Rotation maps the previous contact frame into the current one in the same
    observer. Its spin is explicit; minimal-normal rotation alone cannot
    describe arbitrary contact twirl. No timestep or event is solved here.
    """
    if not isinstance(state, TangentialContactState):
        raise TypeError("state must be TangentialContactState")
    current_normal = _unit_normal(normal, "contact normal")
    rotation = np.asarray(
        require_rotation(finite_array(transport_rotation, (3, 3), "contact transport"))
    )
    if not np.allclose(
        rotation @ state.normal, current_normal, rtol=0, atol=_TRANSPORT_TOLERANCE
    ):
        raise ValueError("contact transport must map the previous normal")
    slip = np.asarray(_tangent(slip_increment_m, current_normal, "slip increment"))
    force = _nonnegative(normal_force_n, "normal force")
    law = state.law
    cap = require_finite_float(force * law.friction_coefficient, "Coulomb force cap")
    old = rotation @ state.elastic_deflection_m
    trial = old + slip
    trial_force = require_finite_float(
        law.stiffness_n_per_m * math.hypot(*trial), "trial tangential force"
    )
    elastic = trial * (min(1.0, cap / trial_force) if trial_force else 1.0)
    effort = law.stiffness_n_per_m * elastic
    updated = TangentialContactState(law, elastic, current_normal, state.observer_id)
    return TangentialContactUpdate(
        updated,
        _vector(effort, "resisting force"),
        updated.elastic_energy_j - state.elastic_energy_j,
        float(effort @ slip),
        float(effort @ (trial - elastic)),
        _elastic_energy(law, elastic - old),
    )


__all__ = ()
