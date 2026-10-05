"""Strict versioned JSON persistence for iron-family parameters."""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from typing import Any

from ._validation import reject_unknown_fields, require_mapping
from .iron_parameters import IronHeadParameters
from .serialization import _unique_object
from .wedge_parameters import Handedness, WedgeGeometryProvenance

IRON_PARAMETERS_FORMAT = "golf_club.iron_parameters/1"
_DOCUMENT_FIELDS = frozenset({"format", "parameters"})
_PARAMETER_FIELDS = frozenset(item.name for item in fields(IronHeadParameters))
_PROVENANCE_FIELDS = frozenset(item.name for item in fields(WedgeGeometryProvenance))


def iron_parameters_to_json(parameters: IronHeadParameters) -> str:
    """Serialize one iron parameter set deterministically (sorted keys)."""
    if not isinstance(parameters, IronHeadParameters):
        raise TypeError("parameters must be IronHeadParameters")
    values: dict[str, Any] = {
        name: getattr(parameters, name) for name in _PARAMETER_FIELDS
    }
    values["handedness"] = parameters.handedness.value
    values["provenance"] = asdict(parameters.provenance)
    return json.dumps(
        {"format": IRON_PARAMETERS_FORMAT, "parameters": values},
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def iron_parameters_from_json(text: str) -> IronHeadParameters:
    """Parse a strict current-version iron parameter document.

    Unknown or duplicate fields, a foreign ``format``, and any value outside
    the schema domain are rejected; the schema constructor is the authority.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    try:
        value = json.loads(text, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as error:
        raise ValueError("text must contain valid JSON") from error
    document = require_mapping(value, "iron parameter JSON")
    reject_unknown_fields(document, _DOCUMENT_FIELDS, "iron parameter JSON")
    format_name = document.get("format")
    if format_name != IRON_PARAMETERS_FORMAT:
        raise ValueError(f"unsupported iron-parameter format {format_name!r}")
    return _parameters_from_dict(document.get("parameters"))


def _parameters_from_dict(value: object) -> IronHeadParameters:
    data = require_mapping(value, "iron parameters")
    reject_unknown_fields(data, _PARAMETER_FIELDS, "iron parameters")
    handedness_value = data.get("handedness")
    try:
        handedness = Handedness(handedness_value)
    except ValueError as error:
        raise ValueError(f"unknown handedness {handedness_value!r}") from error
    provenance = require_mapping(data.get("provenance"), "iron provenance")
    reject_unknown_fields(provenance, _PROVENANCE_FIELDS, "iron provenance")
    values: dict[str, Any] = {name: data.get(name) for name in _PARAMETER_FIELDS}
    values["handedness"] = handedness
    values["provenance"] = WedgeGeometryProvenance(**provenance)
    return IronHeadParameters(**values)


__all__ = [
    "IRON_PARAMETERS_FORMAT",
    "iron_parameters_from_json",
    "iron_parameters_to_json",
]
