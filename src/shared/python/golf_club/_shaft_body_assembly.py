"""Rigid component attachment through existing loaded shaft inertia quadrature."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ._shaft_inertia import InertiaSample, SectionInertia
from ._shaft_rotating_chain import RotatingSectionChain
from .types import ComponentMassProperties


def attach_nodal_body(
    chain: RotatingSectionChain, node: int, body: ComponentMassProperties
) -> RotatingSectionChain:
    """Return a new chain with one additional rigid body fixed to a chosen node.

    Body COM offset and full COM inertia must already use the section's material
    axes and nodal origin. A frame mismatch is refused, never transformed by
    assumption. Body mass/inertia are physical integrated properties, not a
    density: the endpoint sample receives no quadrature-length multiplier.

    The root uses the first section's left endpoint; every other node uses
    only its preceding section's right endpoint, avoiding shared-node double
    counting. Existing section samples, loads, geometry and frame are retained.
    This is additive composition: callers must exclude this physical body from
    the original distributed mass. It supplies no flexible head, grip impedance,
    contact law, calibration or stability qualification.
    """
    if not isinstance(chain, RotatingSectionChain):
        raise TypeError("chain must be RotatingSectionChain")
    if not isinstance(body, ComponentMassProperties):
        raise TypeError("body must be ComponentMassProperties")
    if isinstance(node, (bool, np.bool_)) or not isinstance(node, (int, np.integer)):
        raise TypeError("node must be an integer")
    if not 0 <= node < chain.node_count:
        raise ValueError("node must identify an existing chain node")
    index = max(0, int(node) - 1)
    fraction = 0.0 if node == 0 else 1.0
    inertias = list(chain.inertias)
    section = inertias[index]
    inertias[index] = SectionInertia((*section.samples, InertiaSample(fraction, body)))
    return replace(chain, inertias=tuple(inertias))


__all__ = ()
