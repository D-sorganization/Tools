"""Assemble verified section and point-load work in common nodal charts.

Private dense residual building block; it supplies neither boundary constraints
nor an equilibrium, stability, dynamic or experimentally qualified shaft model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._grip_contracts import finite_array
from ._shaft_point_load import SpatialPointLoad
from ._shaft_se3 import _rigid_pose, twist_ad
from ._shaft_section import SectionElement
from ._validation import require_finite_float

_NODE_DOF = 6


def _material_chart_connection(residual: object) -> np.ndarray:
    """Return the nodal connection mapping fixed-chart to material derivatives.

    For linear-first right-increment coordinates its action on a is
    ad(a)^T residual/2. Subtract it for the reverse conversion. This is not
    an elastic stiffness and no symmetry may be imposed away from balance.
    """
    size = np.size(residual)
    if size == 0 or size % _NODE_DOF:
        raise ValueError("residual must contain complete six-axis nodes")
    force = finite_array(residual, (size,), "chain residual")
    connection = np.zeros((size, size))
    for start in range(0, size, _NODE_DOF):
        rows = slice(start, start + _NODE_DOF)
        connection[rows, rows] = np.column_stack(
            [0.5 * twist_ad(axis).T @ force[rows] for axis in np.eye(_NODE_DOF)]
        )
    return connection


@dataclass(frozen=True)
class IndexedPointLoad:
    """A physical point load assigned to one zero-based section node.

    Multiple loads may share a node; their distinct material offsets remain
    separate, rather than being replaced by a configuration-independent couple.
    """

    node: int
    load: SpatialPointLoad

    def __post_init__(self) -> None:
        if isinstance(self.node, (bool, np.bool_)) or not isinstance(
            self.node, (int, np.integer)
        ):
            raise TypeError("load node must be an integer")
        if self.node < 0:
            raise ValueError("load node must be nonnegative")
        if not isinstance(self.load, SpatialPointLoad):
            raise TypeError("load must be a SpatialPointLoad")
        object.__setattr__(self, "node", int(self.node))


@dataclass(frozen=True)
class ChainLinearization:
    """Fresh full-node residual and fixed-chart derivative at the supplied poses.

    Residual is internal minus external work, in local N and N m; coordinates
    are linear-first m and rad. Tangent is K_internal-K_external, with no
    symmetry enforcement. Force potential excludes couple work and therefore
    must not be summed with elastic energy as a general total potential.
    Unconstrained residual entries are retained for later support reactions.
    """

    elastic_energy_j: float
    force_potential_j: float
    residual: np.ndarray
    tangent: np.ndarray


@dataclass(frozen=True)
class SectionChain:
    """Consecutive uniform section elements with prescribed spatial point loads.

    Section i joins nodes i and i+1; all supplied poses share an observer frame.
    Elements and loads are copied into immutable tuples. Material strain and
    bandwidth qualification remain the caller's separate physical obligations.
    """

    sections: tuple[SectionElement, ...]
    loads: tuple[IndexedPointLoad, ...]

    def __post_init__(self) -> None:
        sections, loads = tuple(self.sections), tuple(self.loads)
        if not sections:
            raise ValueError("chain must contain at least one section")
        if any(not isinstance(item, SectionElement) for item in sections):
            raise TypeError("chain sections must be SectionElement records")
        if any(not isinstance(item, IndexedPointLoad) for item in loads):
            raise TypeError("chain loads must be IndexedPointLoad records")
        if any(item.node > len(sections) for item in loads):
            raise ValueError("load node is outside the section chain")
        object.__setattr__(self, "sections", sections)
        object.__setattr__(self, "loads", loads)

    @property
    def node_count(self) -> int:
        """Return the required number of poses, including both end nodes."""
        return len(self.sections) + 1

    def linearize(self, poses: object) -> ChainLinearization:
        """Sum element work without deleting constrained nodes or load curvature.

        For H_i(q_i)=H_i Exp(q_i), returns residual and its fixed-chart
        derivative at q=0. A derivative of the re-expressed moving material
        residual away from equilibrium is a different object; do not substitute
        it silently when selecting a nonlinear solution method.
        """
        current = finite_array(poses, (self.node_count, 4, 4), "chain poses")
        for pose in current:
            _rigid_pose(pose)
        size = _NODE_DOF * self.node_count
        residual, tangent = np.zeros(size), np.zeros((size, size))
        elastic, potential = 0.0, 0.0
        for index, section in enumerate(self.sections):
            response = section.linearize(current[index], current[index + 1])
            rows = slice(_NODE_DOF * index, _NODE_DOF * (index + 2))
            residual[rows] += response.gradient
            tangent[rows, rows] += response.tangent
            elastic += response.energy_j
        for item in self.loads:
            point_load = item.load
            load_response = point_load.linearize(current[item.node])
            rows = slice(_NODE_DOF * item.node, _NODE_DOF * (item.node + 1))
            residual[rows] -= load_response.wrench
            tangent[rows, rows] -= load_response.tangent
            potential += point_load.force_potential(current[item.node])
        return ChainLinearization(
            float(require_finite_float(elastic, "chain elastic energy")),
            float(require_finite_float(potential, "chain force potential")),
            finite_array(residual, (size,), "chain residual"),
            finite_array(tangent, (size, size), "chain tangent"),
        )


__all__ = ()
