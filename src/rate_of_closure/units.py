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
    "FIELD_GUIDANCE",
    "LENGTH_UNITS",
    "QUANTITY_UNITS",
    "ROTATION_UNITS",
    "SPEED_UNITS",
    "convert_from_canonical",
    "convert_to_canonical",
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

#: Quantity name -> its unit table. Angles and times stay fixed
#: (degrees / microseconds) — they have no common alternates in the
#: golf-delivery literature.
QUANTITY_UNITS: dict[str, dict[str, float]] = {
    "speed": SPEED_UNITS,
    "rotation": ROTATION_UNITS,
    "length": LENGTH_UNITS,
}


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
        "tour launch-monitor averages."
    ),
    "omega_plane_dps": (
        "Suggested range: 1,800-2,400 deg/s swing-plane rotation at "
        "impact for skilled players. Source: 3-D motion-capture studies "
        "collected in the AffineDrift closure-rate dossier."
    ),
    "omega_shaft_dps": (
        "Suggested range: 652-2,432 deg/s about the shaft (tour driver "
        "mean 1,307 +/- 304, n = 94). Source: Cheetham 2014, via the "
        "AffineDrift closure-rate dossier."
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
}
