"""Unit systems for the rate-of-closure impact explorer.

The model is always canonical — mph, deg/s, mm, microseconds — and the
UIs convert at the edge, the same pattern the UpstreamDrift apps use
for their unit drop-downs. Each unit maps to the factor that converts
one displayed unit into the canonical unit, so

    canonical = displayed * factor        displayed = canonical / factor

Also owns the per-field hover guidance: a suggested range for the golf
swing plus where the suggestion comes from (kept vendor-neutral; the
primary citations live in the AffineDrift closure-rate dossier).
"""

from __future__ import annotations

from ._contracts import require

__all__ = [
    "DISTANCE_UNITS",
    "FIELD_GUIDANCE",
    "LENGTH_UNITS",
    "QUANTITY_UNITS",
    "ROTATION_UNITS",
    "SPEED_UNITS",
    "convert_from_canonical",
    "convert_to_canonical",
    "display_distance_unit",
    "format_distance_m",
    "set_display_distance_unit",
]

#: Display unit -> factor to the canonical unit (mph).
SPEED_UNITS: dict[str, float] = {
    "mph": 1.0,
    "m/s": 2.236936292054402,
    "km/h": 0.621371192237334,
    "ft/s": 0.681818181818182,
}

#: Display unit -> factor to the canonical unit (deg/s).
ROTATION_UNITS: dict[str, float] = {
    "deg/s": 1.0,
    "rad/s": 57.29577951308232,
    "rpm": 6.0,
}

#: Display unit -> factor to the canonical unit (mm).
LENGTH_UNITS: dict[str, float] = {
    "mm": 1.0,
    "cm": 10.0,
    "in": 25.4,
}

#: Ball-flight distances (#4125 H6): display unit -> factor to the
#: canonical unit (metres — internal physics stays SI). Listed with
#: yards FIRST: the drop-down convention takes the first entry as the
#: default, so golf distances read in yards out of the box.
DISTANCE_UNITS: dict[str, float] = {
    "yd": 0.9144,
    "m": 1.0,
}

#: Quantity name -> its unit table. Angles and times stay fixed
#: (degrees / microseconds) — they have no common alternates in the
#: golf-delivery literature.
QUANTITY_UNITS: dict[str, dict[str, float]] = {
    "speed": SPEED_UNITS,
    "rotation": ROTATION_UNITS,
    "length": LENGTH_UNITS,
    "distance": DISTANCE_UNITS,
}

#: The session's selected ball-flight distance display unit. Yards by
#: default (user direction, #4125 H6); the Units drop-downs of both
#: UIs switch it. Internal canonical values remain SI metres always —
#: this only affects presentation (result rows, view axes, plot
#: variables, variation outputs, target entries).
_DISPLAY_DISTANCE_UNIT: list[str] = ["yd"]


def display_distance_unit() -> str:
    """The selected ball-flight distance display unit (``yd`` default)."""
    return _DISPLAY_DISTANCE_UNIT[0]


def set_display_distance_unit(unit: str) -> None:
    """Select the ball-flight distance display unit (``yd`` or ``m``)."""
    require(unit in DISTANCE_UNITS, f"unknown distance unit {unit!r}")
    _DISPLAY_DISTANCE_UNIT[0] = unit


def format_distance_m(value_m: float, decimals: int = 1) -> str:
    """A canonical-metres distance formatted in the display unit.

    >>> set_display_distance_unit("yd"); format_distance_m(91.44)
    '100.0 yd'
    """
    unit = display_distance_unit()
    return f"{value_m / DISTANCE_UNITS[unit]:.{decimals}f} {unit}"


def convert_to_canonical(quantity: str, unit: str, value: float) -> float:
    """Convert a displayed value into the model's canonical unit."""
    table = QUANTITY_UNITS.get(quantity)
    require(table is not None, f"unknown quantity {quantity!r}")
    assert table is not None  # for the type checker; require() raised otherwise
    require(unit in table, f"unknown {quantity} unit {unit!r}")
    return value * table[unit]


def convert_from_canonical(quantity: str, unit: str, value: float) -> float:
    """Convert a canonical model value into the displayed unit."""
    table = QUANTITY_UNITS.get(quantity)
    require(table is not None, f"unknown quantity {quantity!r}")
    assert table is not None
    require(unit in table, f"unknown {quantity} unit {unit!r}")
    return value / table[unit]


#: Hover guidance per scenario field: suggested golf-swing range and the
#: source of the suggestion. Shown as tooltips in both UIs.
FIELD_GUIDANCE: dict[str, str] = {
    "clubhead_speed_mph": (
        "Suggested range: 80-130 mph driver clubhead speed (tour average "
        "near 113 mph; strong amateurs 90-105). Source: openly published "
        "tour launch-monitor averages. Reference frame: speed magnitude "
        "of the clubhead reference point; +x is down the target line."
    ),
    "omega_plane_dps": (
        "Suggested range: 1,800-2,400 deg/s swing-plane rotation at "
        "impact for skilled players. Source: 3-D motion-capture studies "
        "collected in the AffineDrift closure-rate dossier. Reference frame: "
        "right-hand rotation about the oriented swing-plane normal (SPV)."
    ),
    "omega_shaft_dps": (
        "Suggested range: 652-2,432 deg/s about the shaft (tour driver "
        "mean 1,307 +/- 304, n = 94). Source: Cheetham 2014, via the "
        "AffineDrift closure-rate dossier. Reference frame: right-hand "
        "rotation about the shaft axis from grip toward clubhead (HTV)."
    ),
    "lie_angle_deg": (
        "Suggested range: 55-62 deg for a driver delivered near its "
        "static lie; 90 deg makes the shaft vertical to isolate pure "
        "horizontal closure. Source: published driver spec sheets."
    ),
    "com_to_face_mm": (
        "Suggested range: 25-50 mm from the geometric center forward to "
        "the face center for modern drivers; 40 mm is the AffineDrift "
        "worked-example value. Source: openly published launch-monitor "
        "material."
    ),
    "impact_offset_toe_mm": (
        "Suggested range: within +/-15 mm of face center for reasonable "
        "strikes; gear-effect studies use up to +/-20 mm. Source: "
        "published robot-test impact maps."
    ),
    "impact_offset_high_mm": (
        "Suggested range: within +/-10 mm of face center vertically. "
        "Source: published robot-test impact maps."
    ),
    "contact_duration_us": (
        "Suggested range: 400-500 microseconds of ball-face contact for "
        "a driver. Source: openly published high-speed impact studies."
    ),
    "club_selection": (
        "Suggested range: pick the club closest to yours; selecting one "
        "drives GC-to-face and lie from its spec (your overrides are "
        "kept). Source: typical published manufacturer specs, normalized "
        "to SI in the club library."
    ),
    "club_loft_deg": (
        "Suggested range: 8-13 deg drivers, 15-19 deg fairway woods and "
        "hybrids, 21-45 deg irons, 46-64 deg wedges. Source: typical "
        "published manufacturer spec sheets."
    ),
    "face_curvature_enabled": (
        "Suggested range: on for drivers, fairway woods, and hybrids "
        "(curved faces); off for irons, wedges, and putters (flat "
        "faces). Source: typical published fitting references."
    ),
    "face_bulge_radius_mm": (
        "Suggested range: 250-330 mm horizontal (heel-toe) face radius "
        "for a modern driver (10-13 in). Source: typical published "
        "fitting values."
    ),
    "face_roll_radius_mm": (
        "Suggested range: 250-330 mm vertical (crown-sole) face radius, "
        "usually similar to bulge on drivers. Source: typical published "
        "fitting values."
    ),
    "swing_source": (
        "Suggested range: Manual Scenario replays the explorer's "
        "constant-twist delivery; Double and Triple Pendulum generate a "
        "gravity-driven swing on the oriented plane. Source: classic "
        "double-pendulum golf models (Cochran & Stobbs; Jorgensen, The "
        "Physics of Golf)."
    ),
    "plane_yaw_deg": (
        "Suggested range: -20 to +20 deg rotation of the swing plane "
        "about the vertical (aim left/right of the target line). "
        "Source: 3-D swing-plane studies collected in the AffineDrift "
        "closure-rate dossier. Reference frame: rotate the plane about "
        "world +y (up); +x is the target line and +z is right of target."
    ),
    "plane_side_tilt_deg": (
        "Suggested range: -60 to -35 deg side tilt for a driver (a "
        "vertical plane is 0; tour driver swing planes lean roughly "
        "45-55 deg from vertical). Source: published 3-D swing-plane "
        "measurements. Reference frame: roll about the plane's downrange "
        "axis after yaw; negative leans a right-handed driver's plane "
        "toward the golfer."
    ),
    "plane_forward_tilt_deg": (
        "Suggested range: -10 to +10 deg forward/back tilt of the "
        "in-plane upright axis. Source: published 3-D swing-plane "
        "measurements. Reference frame: pitch about the yawed-and-side-tilted "
        "plane's local lateral axis; positive tips the upright axis downrange."
    ),
    "impact_time_scrub": (
        "Suggested range: anywhere inside the swing; the default is the "
        "instant of maximum clubhead speed. Scrubbing moves the swing "
        "relative to the fixed ball so the clubhead meets it at the "
        "chosen instant. Source: launch-monitor impact-timing "
        "convention (maximum-compression reference)."
    ),
    "flight_model": (
        "Suggested range: Waterloo/Penner for driver work; the other "
        "entries are literature models useful for cross-checks. "
        "Source: Penner (2003) and the citations carried on each model "
        "in the shared flight package."
    ),
    "ball_visible": (
        "Suggested range: on to show the ball at its fixed impact "
        "position. Source: launch-monitor convention of a fixed ball "
        "and a swing delivered to it."
    ),
    "ground_visible": (
        "Suggested range: on to show the ground plane for spatial "
        "reference. Source: standard 3-D golf-scene convention."
    ),
    "course_visible": (
        "Suggested range: on to render the course furniture — fairway "
        "strip along the target line, putting green with hole and flag "
        "at the configurable green distance, tee marker at the origin. "
        "Source: standard golf-course presentation; tones derived from "
        "the active theme palette."
    ),
    "screw_axis_visible": (
        "Suggested range: on to overlay the clubhead's instantaneous "
        "screw axis near the playback instant. Source: the AffineDrift "
        "closure-rate derivation (omega/v = 1/R_ISA)."
    ),
    "kinetics_visible": (
        "Suggested range: on (double-pendulum source only) to overlay "
        "per-joint torque arcs (radius grows with torque magnitude, "
        "sweep direction follows its sign) and reaction-force arrows "
        "at the shoulder and wrist during playback. Source: inverse "
        "dynamics over the pendulum EOM (swing_sim.reference), "
        "presented per the movement-optimizer overlay conventions."
    ),
    "swing_flight_toggle": (
        "Off by default: the flight envelope (100+ m) dwarfs the "
        "swing envelope (~3 m), collapsing the swing to a dot when both "
        "share one scale. Turn on only to see the full trajectory in "
        "context. Source: typical driver carry (Penner 2003) vs. club "
        "length (published manufacturer specs)."
    ),
    "strike_curvature_visible": (
        "Suggested range: on for curved-face clubs to contour the "
        "bulge/roll set-back across the face (sagitta, mm). Source: "
        "typical published fitting references (10-13 in driver bulge)."
    ),
    "strike_vectors_visible": (
        "Suggested range: on to draw the delivered club path, face "
        "normal, and attack-angle directions projected into the face "
        "plane. Source: standard launch-monitor D-plane presentation "
        "(TrackMan literature; Jorgensen, The Physics of Golf)."
    ),
    "strike_history_visible": (
        "Suggested range: on to keep a scatter of previous impact "
        "locations for dispersion context. Source: published robot-test "
        "impact maps (strike dispersion within +/-15 mm of center)."
    ),
    "strike_club_info_visible": (
        "Suggested range: on to annotate the selected club's name, "
        "loft, and face curvature radii. Source: typical published "
        "manufacturer spec sheets."
    ),
    "show_cg_marker": (
        "Suggested range: on to mark the head's center of gravity — "
        "the geometric centroid of the generated head computed from "
        "its closed mesh by the divergence theorem (falls back to the "
        "spec CG for loaded STLs that are not watertight). Source: "
        "divergence-theorem solid centroid (standard vector calculus); "
        "typical published CG specs for the per-type bands."
    ),
    "flight_side_visible": (
        "Suggested range: on for the side profile (height vs carry) — "
        "the classic trajectory presentation. Source: launch-monitor "
        "trajectory displays; Penner (2003)."
    ),
    "flight_top_visible": (
        "Suggested range: on for the top-down view (lateral vs carry) "
        "showing curvature and dispersion. Source: launch-monitor "
        "dispersion displays; TrackMan D-plane literature."
    ),
    "flight_3d_visible": (
        "Suggested range: on for the 3-D trajectory polyline at flight "
        "scale. Source: standard 3-D golf-scene convention."
    ),
    "flight_landing_visible": (
        "Suggested range: on to annotate the landing point with carry "
        "and lateral numbers. Source: launch-monitor carry/offline "
        "reporting convention."
    ),
    "flight_apex_visible": (
        "Suggested range: on to mark the apex (maximum height). "
        "Source: launch-monitor apex reporting convention."
    ),
    "fx_mode": (
        "Suggested range: Direct Launch to type launch-monitor ball "
        "numbers; Impact Delivery to type club delivery numbers and run "
        "them through the impact model first. Source: launch-monitor "
        "convention of ball data vs. club data (TrackMan literature)."
    ),
    "fx_ball_speed": (
        "Suggested range: 120-190 mph ball speed (tour driver average "
        "near 167 mph; strong amateurs 140-160). Source: openly "
        "published tour launch-monitor averages."
    ),
    "fx_launch_angle": (
        "Suggested range: 8-16 deg launch for drivers (tour average "
        "near 10.9 deg); higher for irons and wedges. Source: openly "
        "published tour launch-monitor averages."
    ),
    "fx_azimuth": (
        "Suggested range: within +/-10 deg of the target line; + = "
        "right of target. Source: standard launch-monitor sign "
        "convention (launch direction)."
    ),
    "fx_spin_rpm": (
        "Suggested range: 2,000-3,500 rpm total spin for drivers (tour "
        "average near 2,686 rpm); 4,000-10,000+ for irons and wedges. "
        "Source: openly published tour launch-monitor averages."
    ),
    "fx_spin_axis_tilt": (
        "Suggested range: within +/-20 deg spin-axis tilt; + = "
        "fade/slice side (curves right for a right-handed player), - = "
        "draw/hook side. Source: TrackMan D-plane literature."
    ),
    "fx_speed_unit": (
        "Suggested range: mph for launch-monitor style entry, m/s for "
        "SI work; the model always computes in SI. Source: launch-"
        "monitor display convention."
    ),
    "fx_club_path": (
        "Suggested range: within +/-8 deg club path; + = in-to-out. "
        "Source: openly published tour launch-monitor averages."
    ),
    "fx_face_angle": (
        "Suggested range: within +/-5 deg face angle; + = open (right "
        "of target). Source: openly published tour launch-monitor "
        "averages."
    ),
    "fx_attack_angle": (
        "Suggested range: -5 to +5 deg attack angle for drivers (+ = "
        "hitting up); negative for irons. Source: openly published tour "
        "launch-monitor averages."
    ),
    "fx_dynamic_loft": (
        "Suggested range: 8-16 deg dynamic loft for drivers, up to "
        "45+ deg for wedges. Source: openly published tour launch-"
        "monitor averages."
    ),
}
