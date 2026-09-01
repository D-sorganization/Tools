"""Tools-side exporter for the ball-flight wire (ADR-0047 H1, UD #9350).

Turns what :mod:`shared.python.swing_sim.flight` actually produces —
a :class:`~shared.python.swing_sim.flight.types.FlightResult` whose
``trajectory`` is the retained, time-ordered
:class:`~shared.python.swing_sim.flight.types.TrajectoryPoint` series
that the P8 playback transport already replays — into the neutral
``swing_sim.ball_flight_trajectory/1`` record.

Direction of the seam
---------------------
This module is the *export* half only. The import half is
:func:`~.trajectory.ball_flight_trajectory_from_json`, which is
family-neutral by construction: a record produced by UpstreamDrift's
``physics/flight_models.py`` parses here exactly as one produced by
this repo does, because neither side's runtime is involved. The
UpstreamDrift export half is written against the *documented* wire in
:mod:`.trajectory` and lives in that repository, the same posture as
:mod:`shared.python.swing_sim.putting.ud_adapter`.

Frame and channels
------------------
``swing_sim.flight`` integrates in the UpstreamDrift flight frame (x
forward, y left, z up), so exports declare
:data:`~.trajectory.FLIGHT_FRAME_ID` and no conversion happens here;
callers wanting the app frame convert with
:func:`shared.python.swing_sim.flight.frames.from_flight_frame` before
building their own record. Every retained point carries a position and
a velocity, so exports declare the ``velocity_mps`` channel. Spin is
**not** exported: neither family retains a per-sample spin vector (both
decay a scalar spin analytically inside the derivative function), and
reconstructing one after the fact would put a number on the wire that
no integrator ever held.

Parameter provenance
--------------------
:func:`flight_model_parameters` names the coefficient set each model
integrates with, and :func:`trajectory_from_flight_result` digests it
into the record. A model class this module does not know is **refused**
rather than exported with an empty or guessed parameter set — an
unattributable trajectory is precisely what the wire exists to
prevent. The long-term home for these names is a ``coefficients``
property on the model family itself (UpstreamDrift's twin already has
one, its issue #8978); until that lands here, the adapter names them
and this function is the single place to update.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from shared.python.contracts import require

from ..flight.models import (
    BallFlightModel,
    ConstantCoefficientModel,
    MacDonaldHanzelyModel,
    WaterlooPennerModel,
)
from ..flight.types import FlightResult
from .trajectory import (
    FLIGHT_FRAME_ID,
    BallFlightTrajectory,
    TrajectoryProvenance,
    from_samples,
    parameter_digest,
)

TOOLS_FLIGHT_FAMILY = "swing_sim.flight"
"""The ``model_family`` every record exported from this repo declares."""

_WATERLOO_PENNER_KEYS = ("cd0", "cd1", "cd2", "cl0", "cl1", "cl2", "cl_max")

__all__ = [
    "TOOLS_FLIGHT_FAMILY",
    "flight_model_parameters",
    "trajectory_from_flight_result",
]


def _triplet(vector: Iterable[Any]) -> tuple[float, float, float]:
    """Return a NumPy 3-vector as a plain float triple for the wire."""
    values = [float(component) for component in vector]
    require(len(values) == 3, "flight samples must carry 3-vectors", len(values))
    return (values[0], values[1], values[2])


def flight_model_parameters(model: BallFlightModel) -> dict[str, float]:
    """Return the named coefficient set ``model`` integrates with.

    Args:
        model: A concrete flight model from
            :mod:`shared.python.swing_sim.flight.models`.

    Returns:
        A flat mapping of coefficient name to value, suitable for
        :func:`~.trajectory.parameter_digest`. Waterloo/Penner reports
        its seven quadratic-drag / Penner-lift coefficients; the
        spin-decay models report ``cd``, ``cl``, and ``spin_decay``.

    Raises:
        ContractViolationError: If ``model`` is not a known flight
            model. Refusing is deliberate: a record whose parameters
            cannot be named cannot carry honest provenance.
    """
    if isinstance(model, WaterlooPennerModel):
        return dict(zip(_WATERLOO_PENNER_KEYS, model.params, strict=True))
    if isinstance(model, MacDonaldHanzelyModel):
        return {"cd": model.cd, "cl": model.cl, "spin_decay": model.decay}
    if isinstance(model, ConstantCoefficientModel):
        spec = model._spec  # noqa: SLF001 - same-family exporter; see module docstring
        return {"cd": spec.cd, "cl": spec.cl, "spin_decay": spec.spin_decay}
    require(
        False,
        "unknown flight model: name its coefficients in "
        "flight_interchange.adapters.flight_model_parameters before exporting",
        type(model).__name__,
    )
    raise AssertionError("unreachable")  # pragma: no cover - require() raises


def trajectory_from_flight_result(
    result: FlightResult,
    model: BallFlightModel,
    *,
    source_id: str | None = None,
) -> BallFlightTrajectory:
    """Export one ``swing_sim.flight`` result as a v1 trajectory record.

    Args:
        result: The flight to export. Its ``trajectory`` is the
            retained integrator sample series — the same samples P8
            playback replays — and is never re-simulated or resampled
            here.
        model: The model that produced ``result``; its identity and
            coefficients become the record's provenance.
        source_id: Optional run identifier. Defaults to
            ``"swing_sim.flight:<model name>"``.

    Returns:
        A :class:`~.trajectory.BallFlightTrajectory` in the flight
        frame carrying the ``velocity_mps`` channel.

    Raises:
        ContractViolationError: If ``result`` and ``model`` disagree on
            the model name (exporting one model's samples under
            another's coefficients is a provenance lie), if the result
            retains fewer than two samples, or if the model's
            coefficients cannot be named.
    """
    require(isinstance(result, FlightResult), "result must be FlightResult")
    require(isinstance(model, BallFlightModel), "model must be a BallFlightModel")
    require(
        result.model_name == model.name,
        "result and model must name the same model",
        (result.model_name, model.name),
    )
    require(
        len(result.trajectory) >= 2,
        "a flight with fewer than two retained samples is not a trajectory",
        len(result.trajectory),
    )
    provenance = TrajectoryProvenance(
        model_family=TOOLS_FLIGHT_FAMILY,
        model_name=model.name,
        parameter_digest=parameter_digest(flight_model_parameters(model)),
    )
    return from_samples(
        source_id=source_id or f"{TOOLS_FLIGHT_FAMILY}:{model.name}",
        frame_id=FLIGHT_FRAME_ID,
        provenance=provenance,
        times_s=[float(point.time) for point in result.trajectory],
        positions_m=[_triplet(point.position) for point in result.trajectory],
        velocities_mps=[_triplet(point.velocity) for point in result.trajectory],
    )
