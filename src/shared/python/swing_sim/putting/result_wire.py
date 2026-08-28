"""Putt result wire ``swing_sim.putting_result/2`` (epic #4800, P5).

What v2 is
----------
One versioned, fail-closed, byte-deterministic record of a single
integrated putt: the v1 scalar summary (roll-out, skid, time, break,
capture) **plus** the 2-D quantities the #4800 P1/P2 work produced —
start azimuth and sidespin off the face, a break-trajectory summary,
and the capture margin under the published Holmes/Penner effective-
radius model — together with the provenance that says which putter
and which stroke produced it.

Wire posture follows the package idiom (``surface.py``,
``golf_club.putter_head/1``): sorted keys, compact separators,
``allow_nan=False``, unknown fields refused, missing fields refused,
non-finite values refused, byte-identical round-trips within a
runtime. Float formatting is runtime-local (Python ``repr`` vs JS
shortest-round-trip differ on integral floats), so cross-runtime
interchange is by JSON *value*.

v2 supersedes v1 — no silent migration
--------------------------------------
This follows the ``golf_club.wedge_export`` posture verbatim: a
version is a *contract*, and a reader either understands the exact
declared format or refuses. v1 was the unversioned in-process summary
authority ``rate_of_closure.putting_result_contract`` — scalars only,
no start line, no sidespin, no effective-radius capture — so a v1
record simply cannot answer the questions v2 exists to answer.

Consequently:

* :func:`putting_result_from_json` refuses a v1 payload. It does not
  upgrade it, does not default the missing 2-D fields, and does not
  guess: fabricating ``start_azimuth_deg = 0`` for a record that never
  measured one would silently turn "unknown" into "square".
* :func:`putting_result_v1_archive_from_json` reads v1 **as archive
  evidence only**, returning a distinct :class:`PuttingResultV1Archive`
  that is deliberately *not* a :class:`PuttingResultDocument` and has
  no upgrade path. Retained v1 evidence stays readable; it never
  re-enters the v2 pipeline.
* Writers only ever emit v2.

Break-trajectory summary
------------------------
Derived from the exact retained samples of the integrated
:class:`~.green.PuttResult` — never re-simulated:

* ``apex_break_m``: the signed lateral offset of largest magnitude
  along the path (left positive), i.e. how far the ball got from the
  target line at the top of the break; ``apex_break_at_m`` is the
  along-line station where that happened.
* ``final_break_m``: the lateral offset at rest or capture (the v1
  ``break_m``).
* ``entry_azimuth_deg``: the direction of travel at the closest
  approach to the hole, off the target line, ``+`` = right — the same
  sign convention as ``start_azimuth_deg``.

Capture margin (Holmes/Penner)
------------------------------
:mod:`.capture` gives the effective hole radius
``R_eff(v) = R sqrt(1 - (v / v_capture)^2)``. The v2 margin is the
geometric one that model implies::

    capture_margin_m = R_eff(v_closest) - closest_approach_m

positive iff the ball passed inside the effective hole — so a holed
putt has a non-negative margin and a lipped-out putt reports how much
hole it needed. The v1 speed margin (``margin_mps``) is retained
beside it unchanged, because the two answer different questions
(how much *speed* was to spare vs how much *hole*).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from shared.python.contracts import require, require_finite

from .capture import effective_hole_radius_m
from .green import PuttResult
from .impact import PuttLaunch

PUTTING_RESULT_FORMAT = "swing_sim.putting_result/2"
PUTTING_RESULT_FORMAT_V1 = "swing_sim.putting_result/1"

#: Integration kernel identity pinned into every v2 record (the
#: fixed-step RK4 the P2 surface integrator runs; see ``green.py``).
PUTTING_RESULT_KERNEL = "RK4-2ms-v1"

__all__ = [
    "PUTTING_RESULT_FORMAT",
    "PUTTING_RESULT_FORMAT_V1",
    "PUTTING_RESULT_KERNEL",
    "PuttingResultDocument",
    "PuttingResultProvenance",
    "PuttingResultV1Archive",
    "putting_result_document",
    "putting_result_from_json",
    "putting_result_to_json",
    "putting_result_v1_archive_from_json",
]

#: Where the putter came from. ``"mesh"``/``"library"`` mirror
#: ``golf_club.putter_head/1`` provenance kinds (P3); ``"minimal"`` is
#: the ``swing_sim.putting.impact.MINIMAL_PUTTERS`` last-resort spec.
PUTTER_SOURCES = ("mesh", "library", "minimal")

#: Where the stroke came from: declared parameters, or a recorded
#: stroke imported through the P4 interchange seam.
STROKE_SOURCES = ("declared", "interchange")

_PROVENANCE_FIELDS = frozenset(
    {"putter_source", "putter_name", "putter_mesh_sha256", "putter_library_name"}
    | {"stroke_source", "stroke_source_id", "capture_model", "kernel"}
)

_LAUNCH_FIELDS = (
    "ball_speed_mps",
    "launch_angle_deg",
    "horizontal_speed_mps",
    "spin_rad_s",
    "sidespin_rad_s",
    "start_azimuth_deg",
    "effective_loft_deg",
)

_ROLL_FIELDS = (
    "skid_distance_m",
    "total_distance_m",
    "time_s",
)

_BREAK_FIELDS = (
    "apex_break_m",
    "apex_break_at_m",
    "final_break_m",
    "entry_azimuth_deg",
)

_CAPTURE_REQUIRED = (
    "hole_distance_m",
    "closest_approach_m",
    "speed_at_closest_mps",
    "effective_hole_radius_m",
    "capture_margin_m",
)

_CAPTURE_OPTIONAL = ("speed_at_hole_mps", "margin_mps", "miss_distance_m")

_DOCUMENT_FIELDS = frozenset({"format", "provenance", "launch", "roll"} | {"capture"})

_V1_SUMMARY_FIELDS = frozenset(
    {"skid_distance_m", "total_distance_m", "time_s", "break_m", "holed"}
    | {"speed_at_hole_mps", "margin_mps", "miss_distance_m"}
)
_V1_FIELDS = frozenset({"format", "summary"})


def _finite_number(value: object, name: str) -> float:
    """A strict JSON number: int or float, never bool, always finite."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    require(math.isfinite(result), f"{name} must be finite", value)
    return result


def _optional_number(data: dict[str, Any], name: str) -> float | None:
    value = data.get(name)
    return None if value is None else _finite_number(value, name)


def _identifier(value: object, name: str) -> str:
    require(isinstance(value, str) and bool(value.strip()), f"{name} must be a name")
    return str(value)


@dataclass(frozen=True)
class PuttingResultProvenance:
    """Which putter and which stroke produced the record.

    Fail-closed per source kind, mirroring
    :class:`~shared.python.golf_club.putter_head.PutterHeadProvenance`
    (P3): ``"mesh"`` requires the source-mesh SHA-256 and forbids a
    library name, ``"library"`` requires the library name and forbids
    a digest, and ``"minimal"`` carries neither. ``stroke_source_id``
    is required for an imported stroke (P4) and forbidden for a
    declared one, so a record can never claim an interchange origin it
    cannot name.
    """

    putter_source: str
    putter_name: str
    stroke_source: str
    capture_model: str
    putter_mesh_sha256: str | None = None
    putter_library_name: str | None = None
    stroke_source_id: str | None = None
    kernel: str = PUTTING_RESULT_KERNEL

    def __post_init__(self) -> None:
        require(
            self.putter_source in PUTTER_SOURCES,
            f"putter_source must be one of {PUTTER_SOURCES}",
            self.putter_source,
        )
        require(
            self.stroke_source in STROKE_SOURCES,
            f"stroke_source must be one of {STROKE_SOURCES}",
            self.stroke_source,
        )
        _identifier(self.putter_name, "putter_name")
        _identifier(self.capture_model, "capture_model")
        _identifier(self.kernel, "kernel")
        self._require_putter_source()
        if self.stroke_source == "interchange":
            _identifier(self.stroke_source_id, "stroke_source_id")
        elif self.stroke_source_id is not None:
            raise ValueError("declared strokes must not carry stroke_source_id")

    def _require_putter_source(self) -> None:
        if self.putter_source == "mesh":
            digest = self.putter_mesh_sha256
            require(
                isinstance(digest, str) and len(digest) == 64,
                "mesh putters must carry a 64-character mesh SHA-256",
            )
            require(
                self.putter_library_name is None,
                "mesh putters must not carry putter_library_name",
            )
            return
        require(
            self.putter_mesh_sha256 is None,
            "only mesh putters may carry putter_mesh_sha256",
        )
        if self.putter_source == "library":
            _identifier(self.putter_library_name, "putter_library_name")
        elif self.putter_library_name is not None:
            raise ValueError("minimal putters must not carry putter_library_name")


@dataclass(frozen=True)
class PuttingResultDocument:
    """One putt, v2: the v1 summary plus the 2-D and capture fields.

    Every value is measured from the retained integration samples;
    nothing here is re-simulated or defaulted. ``holed`` is the
    integrator's own capture decision, and the optional v1 capture
    scalars keep exactly their v1 meaning (``None`` when the ball
    never crossed the hole mouth / never holed).
    """

    provenance: PuttingResultProvenance
    ball_speed_mps: float
    launch_angle_deg: float
    horizontal_speed_mps: float
    spin_rad_s: float
    sidespin_rad_s: float
    start_azimuth_deg: float
    effective_loft_deg: float
    skid_distance_m: float
    total_distance_m: float
    time_s: float
    apex_break_m: float
    apex_break_at_m: float
    final_break_m: float
    entry_azimuth_deg: float
    hole_distance_m: float
    closest_approach_m: float
    speed_at_closest_mps: float
    effective_hole_radius_m: float
    capture_margin_m: float
    holed: bool
    speed_at_hole_mps: float | None = None
    margin_mps: float | None = None
    miss_distance_m: float | None = None

    def __post_init__(self) -> None:
        require(
            isinstance(self.provenance, PuttingResultProvenance),
            "provenance must be PuttingResultProvenance",
        )
        require(isinstance(self.holed, bool), "holed must be boolean")
        for name in _LAUNCH_FIELDS + _ROLL_FIELDS + _BREAK_FIELDS + _CAPTURE_REQUIRED:
            require_finite(getattr(self, name), name)
        for name in _CAPTURE_OPTIONAL:
            value = getattr(self, name)
            if value is not None:
                require_finite(value, name)
                require(value >= 0.0, f"{name} must be non-negative", value)
        for name in _ROLL_FIELDS:
            require(getattr(self, name) >= 0.0, f"{name} must be non-negative", name)
        require(
            self.skid_distance_m <= self.total_distance_m + 1e-9,
            "skid cannot exceed the total roll",
            (self.skid_distance_m, self.total_distance_m),
        )
        require(
            self.closest_approach_m >= 0.0,
            "closest_approach_m must be non-negative",
            self.closest_approach_m,
        )
        require(
            self.effective_hole_radius_m >= 0.0,
            "effective_hole_radius_m must be non-negative",
            self.effective_hole_radius_m,
        )
        require(
            abs(self.apex_break_m) >= abs(self.final_break_m) - 1e-12,
            "apex break must be the largest lateral excursion",
            (self.apex_break_m, self.final_break_m),
        )
        coherent = (
            self.speed_at_hole_mps is not None
            and self.margin_mps is not None
            and self.miss_distance_m is None
            if self.holed
            else self.margin_mps is None and self.miss_distance_m is not None
        )
        require(coherent, "capture summaries are internally inconsistent")


@dataclass(frozen=True)
class PuttingResultV1Archive:
    """A retained v1 summary, readable as evidence and nothing else.

    Deliberately not a :class:`PuttingResultDocument` and deliberately
    without an upgrade constructor: v1 never measured a start line, a
    sidespin, or an effective-radius capture, so there is no honest
    way to present it as a v2 record.
    """

    skid_distance_m: float
    total_distance_m: float
    time_s: float
    break_m: float
    holed: bool
    speed_at_hole_mps: float | None
    margin_mps: float | None
    miss_distance_m: float | None


def _break_summary(
    result: PuttResult, hole_distance_m: float
) -> tuple[float, float, float, float, float]:
    """Apex break, its station, entry azimuth, closest approach, speed."""
    apex_index = max(range(len(result.path_y_m)), key=lambda i: abs(result.path_y_m[i]))
    closest_index = min(
        range(len(result.path_x_m)),
        key=lambda i: math.hypot(
            result.path_x_m[i] - hole_distance_m, result.path_y_m[i]
        ),
    )
    entry_index = max(closest_index, 1)
    dx = result.path_x_m[entry_index] - result.path_x_m[entry_index - 1]
    dy = result.path_y_m[entry_index] - result.path_y_m[entry_index - 1]
    entry_azimuth_deg = (
        0.0 if dx == 0.0 and dy == 0.0 else math.degrees(math.atan2(-dy, dx))
    )
    closest = math.hypot(
        result.path_x_m[closest_index] - hole_distance_m,
        result.path_y_m[closest_index],
    )
    return (
        result.path_y_m[apex_index],
        result.path_x_m[apex_index],
        entry_azimuth_deg,
        closest,
        result.speeds_mps[closest_index],
    )


def putting_result_document(
    launch: PuttLaunch,
    result: PuttResult,
    provenance: PuttingResultProvenance,
    *,
    hole_distance_m: float,
) -> PuttingResultDocument:
    """Build the v2 record from one launch and its integrated result.

    Args:
        launch: The post-impact state from :func:`~.impact.strike`.
        result: The integrated putt (:func:`~.green.simulate_putt` or
            :func:`~.green.simulate_putt_on_surface`).
        provenance: Putter and stroke origin for the record.
        hole_distance_m: Hole distance the putt was integrated
            against [m]; the record is meaningless without it.

    Returns:
        The v2 :class:`PuttingResultDocument`.

    Raises:
        TypeError: If ``launch`` or ``result`` is the wrong type.
        ValueError / ContractViolationError: If any value is invalid.
    """
    if not isinstance(launch, PuttLaunch):
        raise TypeError("launch must be PuttLaunch")
    if not isinstance(result, PuttResult):
        raise TypeError("result must be PuttResult")
    require_finite(hole_distance_m, "hole_distance_m")
    apex, apex_at, entry_azimuth, closest, speed_at_closest = _break_summary(
        result, hole_distance_m
    )
    radius = effective_hole_radius_m(speed_at_closest)
    return PuttingResultDocument(
        provenance=provenance,
        ball_speed_mps=launch.ball_speed_mps,
        launch_angle_deg=launch.launch_angle_deg,
        horizontal_speed_mps=launch.horizontal_speed_mps,
        spin_rad_s=launch.spin_rad_s,
        sidespin_rad_s=launch.sidespin_rad_s,
        start_azimuth_deg=launch.start_azimuth_deg,
        effective_loft_deg=launch.effective_loft_deg,
        skid_distance_m=result.skid_distance_m,
        total_distance_m=result.total_distance_m,
        time_s=result.time_s,
        apex_break_m=apex,
        apex_break_at_m=apex_at,
        final_break_m=result.break_m,
        entry_azimuth_deg=entry_azimuth,
        hole_distance_m=hole_distance_m,
        closest_approach_m=closest,
        speed_at_closest_mps=speed_at_closest,
        effective_hole_radius_m=radius,
        capture_margin_m=radius - closest,
        holed=result.holed,
        speed_at_hole_mps=result.speed_at_hole_mps,
        margin_mps=result.margin_mps,
        miss_distance_m=result.miss_distance_m,
    )


def _provenance_payload(provenance: PuttingResultProvenance) -> dict[str, Any]:
    return {
        "putter_source": provenance.putter_source,
        "putter_name": provenance.putter_name,
        "putter_mesh_sha256": provenance.putter_mesh_sha256,
        "putter_library_name": provenance.putter_library_name,
        "stroke_source": provenance.stroke_source,
        "stroke_source_id": provenance.stroke_source_id,
        "capture_model": provenance.capture_model,
        "kernel": provenance.kernel,
    }


def putting_result_to_json(document: PuttingResultDocument) -> str:
    """Serialize v2 deterministically; identical runs are byte-identical."""
    if not isinstance(document, PuttingResultDocument):
        raise TypeError("document must be PuttingResultDocument")
    payload: dict[str, Any] = {
        "format": PUTTING_RESULT_FORMAT,
        "provenance": _provenance_payload(document.provenance),
        "launch": {name: getattr(document, name) for name in _LAUNCH_FIELDS},
        "roll": {name: getattr(document, name) for name in _ROLL_FIELDS}
        | {name: getattr(document, name) for name in _BREAK_FIELDS},
        "capture": {name: getattr(document, name) for name in _CAPTURE_REQUIRED}
        | {"holed": document.holed}
        | {name: getattr(document, name) for name in _CAPTURE_OPTIONAL},
    }
    return json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _require_exact_keys(
    data: object, expected: frozenset[str], what: str
) -> dict[str, Any]:
    """Refuse anything but an object carrying exactly ``expected``."""
    if not isinstance(data, dict):
        raise TypeError(f"{what} must be an object")
    section: dict[str, Any] = data
    require(
        set(section) == expected,
        f"{what} fields must be exactly {sorted(expected)}",
    )
    return section


def putting_result_from_json(text: str) -> PuttingResultDocument:
    """Parse a **v2** record; v1, unknown fields, and drift are refused.

    A v1 payload is refused by design (module docstring): read it with
    :func:`putting_result_v1_archive_from_json` instead.
    """
    require(isinstance(text, str), "text must be str")
    data = json.loads(text)
    require(isinstance(data, dict), "putting result must be an object")
    declared = data.get("format")
    require(
        declared != PUTTING_RESULT_FORMAT_V1,
        "putting_result/1 is superseded and is not migrated; read it as archive "
        "evidence with putting_result_v1_archive_from_json",
    )
    require(
        declared == PUTTING_RESULT_FORMAT, f"format must be {PUTTING_RESULT_FORMAT!r}"
    )
    _require_exact_keys(data, _DOCUMENT_FIELDS, "putting result")
    launch = _require_exact_keys(data["launch"], frozenset(_LAUNCH_FIELDS), "launch")
    roll = _require_exact_keys(
        data["roll"], frozenset(_ROLL_FIELDS + _BREAK_FIELDS), "roll"
    )
    capture = _require_exact_keys(
        data["capture"],
        frozenset(_CAPTURE_REQUIRED + _CAPTURE_OPTIONAL + ("holed",)),
        "capture",
    )
    holed = capture["holed"]
    require(isinstance(holed, bool), "holed must be boolean")
    values: dict[str, Any] = {
        name: _finite_number(launch[name], name) for name in _LAUNCH_FIELDS
    }
    values.update(
        {
            name: _finite_number(roll[name], name)
            for name in _ROLL_FIELDS + _BREAK_FIELDS
        }
    )
    values.update(
        {name: _finite_number(capture[name], name) for name in _CAPTURE_REQUIRED}
    )
    values.update({name: _optional_number(capture, name) for name in _CAPTURE_OPTIONAL})
    return PuttingResultDocument(
        provenance=_provenance_from_json(data["provenance"]),
        holed=holed,
        **values,
    )


def _provenance_from_json(data: object) -> PuttingResultProvenance:
    section = _require_exact_keys(data, _PROVENANCE_FIELDS, "provenance")
    optional: dict[str, str | None] = {}
    for name in ("putter_mesh_sha256", "putter_library_name", "stroke_source_id"):
        value = section[name]
        require(value is None or isinstance(value, str), f"{name} must be a string")
        optional[name] = value
    return PuttingResultProvenance(
        putter_source=_identifier(section["putter_source"], "putter_source"),
        putter_name=_identifier(section["putter_name"], "putter_name"),
        stroke_source=_identifier(section["stroke_source"], "stroke_source"),
        capture_model=_identifier(section["capture_model"], "capture_model"),
        kernel=_identifier(section["kernel"], "kernel"),
        **optional,
    )


def putting_result_v1_archive_from_json(text: str) -> PuttingResultV1Archive:
    """Read a retained **v1** summary as archive evidence only.

    Refuses a v2 payload: v2 is read by
    :func:`putting_result_from_json`, and silently narrowing a v2
    record to the v1 fields would discard exactly the evidence v2 was
    introduced to carry.
    """
    require(isinstance(text, str), "text must be str")
    data = json.loads(text)
    require(isinstance(data, dict), "putting result must be an object")
    require(
        data.get("format") != PUTTING_RESULT_FORMAT,
        "this is a putting_result/2 record; read it with putting_result_from_json",
    )
    require(
        data.get("format") == PUTTING_RESULT_FORMAT_V1,
        f"format must be {PUTTING_RESULT_FORMAT_V1!r}",
    )
    _require_exact_keys(data, _V1_FIELDS, "putting result")
    summary = _require_exact_keys(data["summary"], _V1_SUMMARY_FIELDS, "summary")
    holed = summary["holed"]
    require(isinstance(holed, bool), "holed must be boolean")
    return PuttingResultV1Archive(
        skid_distance_m=_finite_number(summary["skid_distance_m"], "skid_distance_m"),
        total_distance_m=_finite_number(
            summary["total_distance_m"], "total_distance_m"
        ),
        time_s=_finite_number(summary["time_s"], "time_s"),
        break_m=_finite_number(summary["break_m"], "break_m"),
        holed=holed,
        speed_at_hole_mps=_optional_number(summary, "speed_at_hole_mps"),
        margin_mps=_optional_number(summary, "margin_mps"),
        miss_distance_m=_optional_number(summary, "miss_distance_m"),
    )
