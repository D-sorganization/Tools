"""Private instantaneous normal contact between a moving shaft and rigid ball.

The shaft already owns its distributed and attached head mass. This port adds
only a common-point contact load; no head inertia or gear-effect correction is
added. It does not integrate a trajectory or qualify contact coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np

from ...golf_club._grip_contracts import _node_index
from ...golf_club._shaft_chain import IndexedPointLoad
from ...golf_club._shaft_moving_chain import MovingChainResponse, moving_chain_response
from ...golf_club._shaft_moving_contracts import (
    InertialMovingChain,
    MovingChainControls,
    MovingChainState,
)
from ...golf_club._shaft_point_load import SpatialPointLoad
from ...golf_club._validation import require_finite_float, require_vector3
from ._normal_contact_work import NormalContactWork, normal_contact_work
from ._rigid_body_response import RigidBodyInertia, RigidBodyResponse
from ._spatial_contact_kinematics import (
    ContactBodyState,
    PlaneSphereGeometry,
    PlaneSphereKinematics,
)
from .contact import KelvinVoigtContactLaw


@dataclass(frozen=True)
class ContactEnergyBalance:
    """Complete instantaneous mechanical ledger, with disjoint loss channels.

    Positive anchor power is work delivered to the prescribed grip driver.
    Cutoff loss is removed contact storage, not acoustic energy. External
    power excludes internal ball/face contact work. No integration accuracy
    or physical qualification is inferred from a small power residual.
    """

    total_energy_j: float
    energy_rate_w: float
    external_power_w: float
    anchor_power_w: float
    grip_dissipated_power_w: float
    viscous_power_w: float
    cutoff_power_w: float

    def __post_init__(self) -> None:
        for item in fields(self):
            value = require_finite_float(getattr(self, item.name), item.name)
            if (
                item.name
                in (
                    "total_energy_j",
                    "grip_dissipated_power_w",
                    "viscous_power_w",
                    "cutoff_power_w",
                )
                and value < 0
            ):
                raise ValueError(f"{item.name} must be nonnegative")
            object.__setattr__(self, item.name, value)
        require_finite_float(self.power_residual_w, "contact power residual")

    @property
    def power_residual_w(self) -> float:
        return (
            self.energy_rate_w
            - self.external_power_w
            + self.anchor_power_w
            + self.grip_dissipated_power_w
            + self.viscous_power_w
            + self.cutoff_power_w
        )


@dataclass(frozen=True)
class NormalShaftContactResponse:
    """Owned kinematics, constitutive work, accelerations and combined ledger."""

    contact: PlaneSphereKinematics
    normal: NormalContactWork
    shaft: MovingChainResponse
    ball: RigidBodyResponse
    energy: ContactEnergyBalance


@dataclass(frozen=True)
class NormalShaftContact:
    """Normal-only sphere/plane composition with explicit moving grip ports.

    Face geometry is fixed in the selected node's material coordinates. Ball
    pose is centered at its geometric center; ball mass properties and input
    twist/wrench use the declared ball material frame. All supplied shaft
    loads and grip anchors are retained. Ball has no external load in this
    first composition. Nonlinear face modes and friction remain separate.
    """

    chain: InertialMovingChain
    face_node: int
    ball_inertia: RigidBodyInertia
    geometry: PlaneSphereGeometry
    law: KelvinVoigtContactLaw
    controls: MovingChainControls

    def __post_init__(self) -> None:
        for name, expected in (
            ("chain", InertialMovingChain),
            ("ball_inertia", RigidBodyInertia),
            ("geometry", PlaneSphereGeometry),
            ("law", KelvinVoigtContactLaw),
            ("controls", MovingChainControls),
        ):
            if not isinstance(getattr(self, name), expected):
                raise TypeError(f"{name} must be {expected.__name__}")
        node = _node_index(self.face_node)
        shaft = self.chain.shaft
        if node >= shaft.node_count:
            raise ValueError("face node is outside the shaft")
        object.__setattr__(self, "face_node", node)

    def _loaded_chain(
        self, contact: PlaneSphereKinematics, force: np.ndarray
    ) -> InertialMovingChain:
        shaft = self.chain.shaft
        elastic = shaft.elastic
        point_load = SpatialPointLoad(
            require_vector3(-force, "face reaction"), (0, 0, 0), contact.face_offset_m
        )
        loads = (*elastic.loads, IndexedPointLoad(self.face_node, point_load))
        return replace(
            self.chain, shaft=replace(shaft, elastic=replace(elastic, loads=loads))
        )

    def kinematics(
        self, shaft_state: MovingChainState, ball_state: ContactBodyState
    ) -> PlaneSphereKinematics:
        """Query canonical contact geometry without evaluating prescribed forces."""
        if not isinstance(shaft_state, MovingChainState):
            raise TypeError("shaft_state must be MovingChainState")
        shaft_model = self.chain.shaft
        if np.asarray(shaft_state.twists).shape[0] != shaft_model.node_count:
            raise ValueError("shaft state must contain every model node")
        face = ContactBodyState(
            np.asarray(shaft_state.poses)[self.face_node],
            np.asarray(shaft_state.twists)[self.face_node],
            shaft_state.observer_id,
        )
        return PlaneSphereKinematics(face, ball_state, self.geometry)

    def evaluate(
        self,
        shaft_state: MovingChainState,
        ball_state: ContactBodyState,
        ball_material_frame_id: str,
    ) -> NormalShaftContactResponse:
        """Recompute migrating contact and both responses from this stage state.

        The point-load tangent is not used as a migrating-contact Jacobian.
        Both load power maps use material-point velocity. Tangential return
        maps must not be inserted here as ordinary RK derivatives.
        """
        contact = self.kinematics(shaft_state, ball_state)
        normal = normal_contact_work(self.law, -contact.gap_m, -contact.gap_rate_mps)
        force = normal.force_n * np.asarray(contact.normal)
        pair = contact.load_pair(force)
        ball = self.ball_inertia.response(
            ball_state.twist,
            pair.ball_wrench,
            ball_material_frame_id,
            self.controls.scales,
        )
        shaft = moving_chain_response(
            self._loaded_chain(contact, force), shaft_state, self.controls
        )
        face_twist = np.asarray(shaft_state.twists)[self.face_node]
        face_power = float(np.asarray(pair.face_wrench) @ face_twist)
        energy = ContactEnergyBalance(
            shaft.total_energy_j + ball.kinetic_energy_j + normal.elastic_energy_j,
            shaft.energy_rate_w + ball.power_w + normal.elastic_power_w,
            shaft.applied_power_w - face_power,
            shaft.anchor_power_w,
            shaft.dissipated_power_w,
            normal.viscous_power_w,
            normal.cutoff_power_w,
        )
        return NormalShaftContactResponse(contact, normal, shaft, ball, energy)


__all__ = ()
