"""Compose declared portable coefficients into existing private shaft kernels."""

from __future__ import annotations

from ._rotating_body_contracts import RotatingFrameState
from ._shaft_chain import IndexedPointLoad, SectionChain
from ._shaft_model_contracts import DistributedShaftModel
from ._shaft_rotating_chain import RotatingSectionChain


def compile_shaft_model(
    model: DistributedShaftModel,
    frame: RotatingFrameState,
    loads: tuple[IndexedPointLoad, ...] = (),
) -> RotatingSectionChain:
    """Reuse exact coefficients, quadrature and existing frame/load contracts.

    No resampling, profile inference, extra length/weight multiplication,
    physical qualification or change to the constitutive law occurs. Frame
    and loads are separate operating inputs, not part of the coefficient hash.
    """
    if not isinstance(model, DistributedShaftModel):
        raise TypeError("model must be DistributedShaftModel")
    elastic = SectionChain(tuple(record.elastic for record in model.sections), loads)
    inertias = tuple(record.inertia for record in model.sections)
    return RotatingSectionChain(elastic, inertias, frame)


__all__ = ()
