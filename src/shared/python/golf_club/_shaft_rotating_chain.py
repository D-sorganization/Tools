"""Instantaneous loaded-chain balance in a prescribed accelerating observer.

This private composition retains deformed distributed inertia and existing
elastic/load work. A root is neither a stable operating point nor a trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._grip_contracts import finite_array
from ._rotating_body_contracts import RotatingFrameState
from ._shaft_chain import (
    ChainLinearization,
    SectionChain,
    _material_chart_connection,
)
from ._shaft_frame_inertia import rotating_section_inertia
from ._shaft_inertia import SectionInertia
from ._shaft_section import SectionElement

_NODE_DOF = 6


@dataclass(frozen=True)
class RotatingSectionChain:
    """Compose one explicit weighted inertia quadrature per elastic section.

    All poses and the prescribed frame motion use one common observer frame.
    Relative velocity and acceleration vanish only at this instantaneous
    balance. Samples carry reference material mass, not current length density.
    Inertia quadratures are copied into a tuple; material qualification remains
    explicit in the existing clamped root solver's strain controls.
    """

    elastic: SectionChain
    inertias: tuple[SectionInertia, ...]
    frame: RotatingFrameState

    def __post_init__(self) -> None:
        if not isinstance(self.elastic, SectionChain):
            raise TypeError("elastic must be a SectionChain")
        if not isinstance(self.frame, RotatingFrameState):
            raise TypeError("frame must be a RotatingFrameState")
        inertias = tuple(self.inertias)
        if any(not isinstance(item, SectionInertia) for item in inertias):
            raise TypeError("inertias must contain SectionInertia records")
        if len(inertias) != self.node_count - 1:
            raise ValueError("require exactly one inertia quadrature per section")
        object.__setattr__(self, "inertias", inertias)

    @property
    def node_count(self) -> int:
        """Delegate the common nodal topology to the elastic chain."""
        return self.elastic.node_count

    @property
    def sections(self) -> tuple[SectionElement, ...]:
        """Expose the same constitutive sections for material-domain checks."""
        return self.elastic.sections

    def linearize(self, poses: object) -> ChainLinearization:
        """Return full-node balance and fixed-chart tangent with fresh arrays.

        Inertial forces enter the left side of internal-external+inertia=0.
        Elastic energy and applied-force potential retain their original
        meanings; neither includes frame work or provides a total potential.
        Euler terms may be nonconservative and are never symmetrized.
        """
        current = finite_array(poses, (self.node_count, 4, 4), "chain poses")
        elastic = self.elastic.linearize(current)
        residual, tangent = elastic.residual, elastic.tangent
        for index, inertia in enumerate(self.inertias):
            result = rotating_section_inertia(
                inertia, current[index : index + 2], self.frame
            )
            rows = slice(_NODE_DOF * index, _NODE_DOF * (index + 2))
            residual[rows] += result.residual
            tangent[rows, rows] += result.moving_jacobian - _material_chart_connection(
                result.residual
            )
        size = _NODE_DOF * self.node_count
        return ChainLinearization(
            elastic.elastic_energy_j,
            elastic.force_potential_j,
            finite_array(residual, (size,), "rotating chain residual"),
            finite_array(tangent, (size, size), "rotating chain tangent"),
        )


__all__ = ()
