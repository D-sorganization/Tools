"""Putting vertical for swing_sim (epic #4125, H3).

Self-façaded: downstream code imports from
``shared.python.swing_sim.putting`` only. The parent
``swing_sim/__init__.py`` façade is wired up during epic integration;
do not add putting exports there from this subpackage (same policy as
``swing_sim.impact`` and ``swing_sim.variation``).

The package covers the full putt: putter-ball impact
(:mod:`.impact` — low-speed impulse with loft, the 2/7 rolling-cap
tangential transfer, and putter-face COR), the skid → pure-roll model
with stimpmeter-derived green-speed parameterization (:mod:`.roll`),
and a planar sloped green with trajectory integration, break, and a
geometric lip-capture condition (:mod:`.green`).

Overlap review (surveyed before building, per the epic): the ideas
adopted from UpstreamDrift's putting assets are credited inline —
the stimp-as-friction parameterization concept and the putter COR
0.78 default follow ``src/engines/physics_engines/putting_green``
and ``rust_core/upstream-physics/src/contact.rs``; the explicit
SLIDING/ROLLING mode machine follows ``ball_roll_physics.py``. All
derivations here are re-done from first principles in the module
docstrings (this package shares no code with UpstreamDrift).
:mod:`.ud_adapter` (#4800 P9) is the runtime-free interchange seam
with UpstreamDrift's ``putting_green`` topography files, and
:mod:`.result_wire` (#4800 P5) is the versioned fail-closed
``swing_sim.putting_result/2`` record of one integrated putt.

Two P5 modules are deliberately **not** re-exported here:
:mod:`.dispersion` and :mod:`.variation`. Importing either pulls the
shared ``swing_sim.variation`` engine (and therefore SciPy), which the
rest of this package does not need — so they are imported directly,
the same policy ``swing_sim.variation`` itself declares.

Putter specs: :data:`~.impact.MINIMAL_PUTTERS` are deliberately
minimal H3-local specs. The reconciliation with the H1 club-library
putters landed in ``shared.python.golf_club.putter_head`` (#4800 P3):
build heads there (mesh or library fallback) and fall back to these
only when neither source is available.
"""

from __future__ import annotations

from .capture import (
    HOLE_RADIUS_M,
    capture_speed_mps,
    effective_hole_radius_m,
)
from .green import (
    CaptureModel,
    GreenConditions,
    PuttResult,
    simulate_putt,
    simulate_putt_on_surface,
)
from .impact import (
    DEFAULT_PUTTER_COR,
    DEFAULT_PUTTER_MOI_KG_M2,
    MINIMAL_PUTTERS,
    PutterSpec,
    PuttLaunch,
    clubhead_speed_from_backstroke,
    strike,
)
from .result_wire import (
    PUTTING_RESULT_FORMAT,
    PUTTING_RESULT_FORMAT_V1,
    PUTTING_RESULT_KERNEL,
    PuttingResultDocument,
    PuttingResultProvenance,
    PuttingResultV1Archive,
    putting_result_document,
    putting_result_from_json,
    putting_result_to_json,
    putting_result_v1_archive_from_json,
)
from .roll import (
    DEFAULT_SLIDING_MU,
    STIMP_RELEASE_SPEED_MPS,
    SkidSolution,
    roll_out_distance,
    roll_time_s,
    rolling_mu_to_stimp,
    solve_skid,
    stimp_to_rolling_mu,
)
from .surface import (
    GREEN_SURFACE_FORMAT,
    GreenSurface,
    GridGreenSurface,
    PlanarGreenSurface,
    green_surface_from_json,
    green_surface_to_json,
)
from .ud_adapter import (
    UdGreenTopography,
    green_surface_from_ud_json,
    green_surface_to_ud_json,
)

__all__ = [
    "DEFAULT_PUTTER_COR",
    "DEFAULT_PUTTER_MOI_KG_M2",
    "DEFAULT_SLIDING_MU",
    "GREEN_SURFACE_FORMAT",
    "HOLE_RADIUS_M",
    "MINIMAL_PUTTERS",
    "PUTTING_RESULT_FORMAT",
    "PUTTING_RESULT_FORMAT_V1",
    "PUTTING_RESULT_KERNEL",
    "STIMP_RELEASE_SPEED_MPS",
    "CaptureModel",
    "GreenConditions",
    "GreenSurface",
    "GridGreenSurface",
    "PlanarGreenSurface",
    "PuttLaunch",
    "PuttResult",
    "PutterSpec",
    "PuttingResultDocument",
    "PuttingResultProvenance",
    "PuttingResultV1Archive",
    "SkidSolution",
    "UdGreenTopography",
    "capture_speed_mps",
    "clubhead_speed_from_backstroke",
    "effective_hole_radius_m",
    "green_surface_from_json",
    "green_surface_from_ud_json",
    "green_surface_to_json",
    "green_surface_to_ud_json",
    "putting_result_document",
    "putting_result_from_json",
    "putting_result_to_json",
    "putting_result_v1_archive_from_json",
    "roll_out_distance",
    "roll_time_s",
    "rolling_mu_to_stimp",
    "simulate_putt",
    "simulate_putt_on_surface",
    "solve_skid",
    "stimp_to_rolling_mu",
]
