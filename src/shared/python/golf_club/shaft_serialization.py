"""Deterministic JSON and tabular CSV persistence for shaft profiles."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Mapping
from typing import Any

from ._validation import reject_unknown_fields, require_mapping
from .shaft_profile import ShaftProfile, ShaftProfileProvenance, ShaftStation

SHAFT_PROFILE_FORMAT = "golf_club.shaft_profile/1"

_PROFILE_FIELDS = frozenset(
    {
        "format",
        "shaft_id",
        "frame_id",
        "raw_length_m",
        "cut_length_m",
        "tip_trim_m",
        "butt_trim_m",
        "insertion_depth_m",
        "stations",
        "provenance",
    }
)
_PROVENANCE_FIELDS = frozenset(
    {
        "source_name",
        "measurement_method",
        "uncertainty_note",
        "source_uri",
        "data_license",
    }
)
_STATION_FIELDS = (
    "position_m",
    "outer_diameter_m",
    "inner_diameter_m",
    "linear_density_kg_m",
    "ei_about_x_n_m2",
    "ei_about_y_n_m2",
    "gj_n_m2",
    "damping_ratio",
    "spine_angle_rad",
)
_CSV_METADATA_FIELDS = (
    "format",
    "shaft_id",
    "frame_id",
    "raw_length_m",
    "cut_length_m",
    "tip_trim_m",
    "butt_trim_m",
    "insertion_depth_m",
    "source_name",
    "measurement_method",
    "uncertainty_note",
    "source_uri",
    "data_license",
)
_CSV_FIELDS = _CSV_METADATA_FIELDS + _STATION_FIELDS


def shaft_profile_to_json(profile: ShaftProfile) -> str:
    """Serialize one profile with deterministic key ordering."""
    return json.dumps(
        shaft_profile_to_json_dict(profile),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def shaft_profile_to_json_dict(profile: ShaftProfile) -> dict[str, Any]:
    """Return the canonical version-one profile mapping."""
    if not isinstance(profile, ShaftProfile):
        raise TypeError("profile must be a ShaftProfile")
    provenance = profile.provenance
    return {
        "format": SHAFT_PROFILE_FORMAT,
        "shaft_id": profile.shaft_id,
        "frame_id": profile.frame_id,
        "raw_length_m": profile.raw_length_m,
        "cut_length_m": profile.cut_length_m,
        "tip_trim_m": profile.tip_trim_m,
        "butt_trim_m": profile.butt_trim_m,
        "insertion_depth_m": profile.insertion_depth_m,
        "stations": [_station_to_dict(station) for station in profile.stations],
        "provenance": {
            "source_name": provenance.source_name,
            "measurement_method": provenance.measurement_method,
            "uncertainty_note": provenance.uncertainty_note,
            "source_uri": provenance.source_uri,
            "data_license": provenance.data_license,
        },
    }


def shaft_profile_from_json(text: str) -> ShaftProfile:
    """Parse one strict profile JSON document."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    try:
        value = json.loads(text, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as error:
        raise ValueError("text must contain valid JSON") from error
    return shaft_profile_from_json_dict(require_mapping(value, "shaft profile JSON"))


def shaft_profile_from_json_dict(data: Mapping[str, Any]) -> ShaftProfile:
    """Validate and load a supported shaft-profile mapping."""
    source = require_mapping(data, "shaft profile JSON")
    format_name = source.get("format")
    if not isinstance(format_name, str):
        raise TypeError("format must be a string")
    if format_name != SHAFT_PROFILE_FORMAT:
        raise ValueError(f"unsupported shaft-profile format {format_name!r}")
    reject_unknown_fields(source, _PROFILE_FIELDS, "shaft profile JSON")
    provenance_data = require_mapping(source.get("provenance"), "provenance")
    reject_unknown_fields(provenance_data, _PROVENANCE_FIELDS, "provenance")
    station_values = source.get("stations")
    if not isinstance(station_values, list):
        raise TypeError("stations must be a JSON array")
    return ShaftProfile(
        shaft_id=source.get("shaft_id"),  # type: ignore[arg-type]
        frame_id=source.get("frame_id"),  # type: ignore[arg-type]
        raw_length_m=source.get("raw_length_m"),  # type: ignore[arg-type]
        cut_length_m=source.get("cut_length_m"),  # type: ignore[arg-type]
        tip_trim_m=source.get("tip_trim_m"),  # type: ignore[arg-type]
        butt_trim_m=source.get("butt_trim_m"),  # type: ignore[arg-type]
        insertion_depth_m=source.get("insertion_depth_m"),  # type: ignore[arg-type]
        stations=tuple(_station_from_dict(value) for value in station_values),
        provenance=_provenance_from_dict(provenance_data),
    )


def shaft_profile_to_csv(profile: ShaftProfile) -> str:
    """Export a self-contained station table with explicit SI headers."""
    payload = shaft_profile_to_json_dict(profile)
    provenance = payload["provenance"]
    assert isinstance(provenance, dict)  # construction invariant
    metadata = {
        key: payload[key]
        for key in _CSV_METADATA_FIELDS
        if key not in _PROVENANCE_FIELDS
    }
    metadata.update({key: provenance[key] for key in _PROVENANCE_FIELDS})
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=_CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for station in profile.stations:
        writer.writerow(metadata | _station_to_dict(station))
    return stream.getvalue()


def shaft_profile_from_csv(text: str) -> ShaftProfile:
    """Load a self-contained canonical station CSV document."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames != list(_CSV_FIELDS):
        raise ValueError("CSV headers must exactly match the canonical shaft schema")
    rows = list(reader)
    if len(rows) < 2:
        raise ValueError("CSV must contain at least two station rows")
    first_metadata = tuple(rows[0][field] for field in _CSV_METADATA_FIELDS)
    if any(
        tuple(row[field] for field in _CSV_METADATA_FIELDS) != first_metadata
        for row in rows[1:]
    ):
        raise ValueError("CSV profile metadata must be identical on every row")
    first = rows[0]
    payload: dict[str, Any] = {
        "format": first["format"],
        "shaft_id": first["shaft_id"],
        "frame_id": first["frame_id"],
        "raw_length_m": _csv_float(first, "raw_length_m"),
        "cut_length_m": _csv_float(first, "cut_length_m"),
        "tip_trim_m": _csv_float(first, "tip_trim_m"),
        "butt_trim_m": _csv_float(first, "butt_trim_m"),
        "insertion_depth_m": _csv_float(first, "insertion_depth_m"),
        "provenance": {
            "source_name": first["source_name"],
            "measurement_method": first["measurement_method"],
            "uncertainty_note": first["uncertainty_note"],
            "source_uri": first["source_uri"] or None,
            "data_license": first["data_license"] or None,
        },
        "stations": [
            {field: _csv_float(row, field) for field in _STATION_FIELDS} for row in rows
        ],
    }
    return shaft_profile_from_json_dict(payload)


def _station_to_dict(station: ShaftStation) -> dict[str, float]:
    return {field: float(getattr(station, field)) for field in _STATION_FIELDS}


def _station_from_dict(value: object) -> ShaftStation:
    data = require_mapping(value, "shaft station")
    reject_unknown_fields(data, frozenset(_STATION_FIELDS), "shaft station")
    return ShaftStation(**{field: data.get(field) for field in _STATION_FIELDS})  # type: ignore[arg-type]


def _provenance_from_dict(data: Mapping[str, Any]) -> ShaftProfileProvenance:
    return ShaftProfileProvenance(
        source_name=data.get("source_name"),  # type: ignore[arg-type]
        measurement_method=data.get("measurement_method"),  # type: ignore[arg-type]
        uncertainty_note=data.get("uncertainty_note"),  # type: ignore[arg-type]
        source_uri=data.get("source_uri"),
        data_license=data.get("data_license"),
    )


def _csv_float(row: Mapping[str, str], field: str) -> float:
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"CSV field {field!r} must contain a real number") from error


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object contains duplicate field {key!r}")
        result[key] = value
    return result


__all__ = [
    "SHAFT_PROFILE_FORMAT",
    "shaft_profile_from_csv",
    "shaft_profile_from_json",
    "shaft_profile_from_json_dict",
    "shaft_profile_to_csv",
    "shaft_profile_to_json",
    "shaft_profile_to_json_dict",
]
