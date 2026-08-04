"""Simulation session for the Rate of Closure explorer (epic #4103).

Orchestrates a full swing -> impact -> flight run:

* :mod:`.sources` — app-frame swing sources (manual constant-twist,
  double pendulum via ``swing_sim``, and a planar triple pendulum).
* :mod:`.session` — the :class:`~rate_of_closure.simulation.session.
  SimulationRun` record and the orchestration entry points, including
  the impact-time scrubber math (the swing translates so the clubhead
  at time tau meets the fixed ball).
* :mod:`.isa` — thin adapter over the rotation converter's screw-axis
  extraction (one file to touch when the Rust surface lands, #4108).
* :mod:`.export` — CSV time-series and JSON summary export.
"""

from __future__ import annotations

from .export import run_to_json_dict, write_csv, write_json
from .isa import screw_axis_samples
from .session import (
    BALL_POSITION_M,
    SimulationConfig,
    SimulationRun,
    delivery_at,
    run_simulation,
)
from .sources import (
    SOURCE_KINDS,
    AppFrameSwing,
    ManualSwingSource,
    TriplePendulumParameters,
    TriplePendulumSwing,
    make_source,
)

__all__ = [
    "BALL_POSITION_M",
    "SOURCE_KINDS",
    "AppFrameSwing",
    "ManualSwingSource",
    "SimulationConfig",
    "SimulationRun",
    "TriplePendulumParameters",
    "TriplePendulumSwing",
    "delivery_at",
    "make_source",
    "run_simulation",
    "run_to_json_dict",
    "screw_axis_samples",
    "write_csv",
    "write_json",
]
