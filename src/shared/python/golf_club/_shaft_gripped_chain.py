"""Stationary finite-pose grip attachments on existing elastic/rotating chains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from ._grip_contracts import finite_array
from ._grip_finite_response import FinitePoseGrip
from ._grip_moving_kinematics import MaterialPointMotion
from ._grip_stationary import require_stationary_motion, stationary_grip_operators
from ._shaft_chain import ChainLinearization, SectionChain, _material_chart_connection
from ._shaft_rotating_chain import RotatingSectionChain
from ._shaft_section import SectionElement
from ._validation import require_finite_float


@dataclass(frozen=True)
class GripAttachment:
    """A grip attached to one material node; repeated nodes add distinct ports.

    The anchor is stationary in the declared observer for this balance slice.
    Its pose fixes the neutral position/orientation of this ideal law. Multiple
    ports do not identify physiological coupling between a player's hands.
    """

    node: int
    grip: FinitePoseGrip
    anchor: MaterialPointMotion

    def __post_init__(self) -> None:
        if isinstance(self.node, (bool, np.bool_)) or not isinstance(
            self.node, (int, np.integer)
        ):
            raise TypeError("grip node must be an integer")
        if self.node < 0:
            raise ValueError("grip node must be nonnegative")
        if not isinstance(self.grip, FinitePoseGrip):
            raise TypeError("grip must be FinitePoseGrip")
        require_stationary_motion(self.anchor)
        object.__setattr__(self, "node", int(self.node))

    def root_motion(self, pose: object) -> MaterialPointMotion:
        """Construct relative-rest state in this attachment's declared observer."""
        return MaterialPointMotion(
            pose, np.zeros(6), np.zeros(6), self.anchor.observer_id
        )


@dataclass(frozen=True)
class GrippedSectionChain:
    """Retain all shaft rows and add finite-grip force, energy and curvature.

    Anchors must share the shaft observer; existing head/body inertia samples
    remain in the rotating chain. No node is implicitly clamped. This is a
    stationary boundary composition, not moving-anchor time evolution.
    """

    shaft: SectionChain | RotatingSectionChain
    grips: tuple[GripAttachment, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.shaft, (SectionChain, RotatingSectionChain)):
            raise TypeError("shaft must be SectionChain or RotatingSectionChain")
        grips = tuple(self.grips)
        if not grips or any(not isinstance(item, GripAttachment) for item in grips):
            raise TypeError("grips must contain GripAttachment records")
        if any(item.node >= self.node_count for item in grips):
            raise ValueError("grip node is outside the shaft")
        observers = {item.anchor.observer_id for item in grips}
        if isinstance(self.shaft, RotatingSectionChain):
            observers.add(self.shaft.frame.frame_id)
        if len(observers) != 1:
            raise ValueError("grips and shaft must share one observer")
        object.__setattr__(self, "grips", grips)

    @property
    def node_count(self) -> int:
        return cast(int, self.shaft.node_count)

    @property
    def sections(self) -> tuple[SectionElement, ...]:
        return cast(tuple[SectionElement, ...], self.shaft.sections)

    def linearize(self, poses: object) -> ChainLinearization:
        """Return combined elastic storage and the full fixed-chart derivative.

        The grip left-side contribution is minus its physical shaft reaction.
        Its moving material derivative is converted to the same fixed-chart
        convention as the existing shaft; no symmetrization is applied.
        """
        current = finite_array(poses, (self.node_count, 4, 4), "gripped chain poses")
        base = self.shaft.linearize(current)
        residual, tangent, energy = base.residual, base.tangent, base.elastic_energy_j
        for item in self.grips:
            port = stationary_grip_operators(
                item.grip, item.root_motion(current[item.node]), item.anchor
            )
            rows = slice(6 * item.node, 6 * (item.node + 1))
            force = -np.asarray(port.response.root_wrench)
            residual[rows] += force
            tangent[rows, rows] += port.stiffness - _material_chart_connection(force)
            energy += port.response.storage.elastic_energy_j
        size = 6 * self.node_count
        return ChainLinearization(
            require_finite_float(energy, "combined elastic energy"),
            base.force_potential_j,
            finite_array(residual, (size,), "gripped residual"),
            finite_array(tangent, (size, size), "gripped tangent"),
        )


__all__ = ()
