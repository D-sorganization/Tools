"""Versioned explicit distributed-shaft coefficients and source byte identity.

The portable boundary is strict JSON. Kernel composition remains a Tools
implementation detail; this format does not qualify a trajectory or FRF.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, fields
from typing import Any

from ._grip_contracts import finite_array
from ._shaft_inertia import InertiaSample, SectionInertia
from ._shaft_model_contracts import (
    DistributedShaftModel,
    ShaftModelSection,
    ShaftModelSource,
)
from ._shaft_section import SectionElement
from ._validation import reject_unknown_fields, require_mapping
from .serialization import _mass_from_dict, _unique_object

DISTRIBUTED_SHAFT_FORMAT = "golf_club.distributed_shaft/1"
_MODEL_FIELDS = frozenset(
    {"format", "model_id", "material_frame_id", "sources", "sections"}
)
_SECTION_FIELDS = frozenset(
    {
        "length_m",
        "reference_twist",
        "stiffness",
        "inertia_samples",
        "coefficient_source_id",
        "inertia_source_id",
    }
)
_SOURCE_FIELDS = frozenset(item.name for item in fields(ShaftModelSource))
_SAMPLE_FIELDS = frozenset({"fraction", "body"})


def _record(value: object, expected: frozenset[str], name: str) -> Mapping[str, Any]:
    result: Mapping[str, Any] = require_mapping(value, name)
    reject_unknown_fields(result, expected, name)
    missing = expected - result.keys()
    if missing:
        raise ValueError(f"{name} has missing fields: {sorted(missing)}")
    return result


def _array(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON array")
    return value


def _sample(value: object) -> InertiaSample:
    record = _record(value, _SAMPLE_FIELDS, "inertia sample")
    body = require_mapping(record["body"], "sample body")
    # Legacy mass serialization permits scalar coercion; this new wire does not.
    finite_array(body.get("center_of_mass_m"), (3,), "center_of_mass_m")
    finite_array(body.get("inertia_at_com_kg_m2"), (3, 3), "inertia_at_com_kg_m2")
    return InertiaSample(record["fraction"], _mass_from_dict(body))


def _section(value: object) -> ShaftModelSection:
    record = _record(value, _SECTION_FIELDS, "section")
    elastic = SectionElement(
        record["length_m"], record["reference_twist"], record["stiffness"]
    )
    inertia = SectionInertia(
        tuple(
            _sample(item)
            for item in _array(record["inertia_samples"], "inertia_samples")
        )
    )
    return ShaftModelSection(
        elastic, inertia, record["coefficient_source_id"], record["inertia_source_id"]
    )


def shaft_model_from_json(document: str) -> DistributedShaftModel:
    """Load an explicit model; refuse unsupported, ambiguous or incomplete data.

    Every schema field is mandatory, including explicit null calibration for
    uncalibrated sources. Existing mass and section physics contracts apply.
    No filesystem/network access, material inference or data promotion occurs.
    """
    if not isinstance(document, str):
        raise TypeError("document must be a string")
    value = json.loads(document, object_pairs_hook=_unique_object)
    record = _record(value, _MODEL_FIELDS, "distributed shaft")
    if record["format"] != DISTRIBUTED_SHAFT_FORMAT:
        raise ValueError("unsupported distributed shaft format")
    sources = tuple(
        ShaftModelSource(**_record(item, _SOURCE_FIELDS, "source"))
        for item in _array(record["sources"], "sources")
    )
    sections = tuple(_section(item) for item in _array(record["sections"], "sections"))
    return DistributedShaftModel(
        record["model_id"], record["material_frame_id"], sections, sources
    )


def _section_payload(record: ShaftModelSection) -> dict[str, Any]:
    result = asdict(record.elastic)
    samples = []
    for sample in record.inertia.samples:
        body = sample.body
        mass = asdict(body)
        mass["role"] = body.role.value
        samples.append({"fraction": sample.fraction, "body": mass})
    return result | {
        "inertia_samples": samples,
        "coefficient_source_id": record.coefficient_source_id,
        "inertia_source_id": record.inertia_source_id,
    }


def shaft_model_to_json(model: DistributedShaftModel) -> str:
    """Return deterministic UTF-8-ready JSON with explicit integrated SI data.

    Coefficient blocks use N, N m and N m²; reference translations/rotations
    use m/rad. Masses and COM inertias already include quadrature weights.
    JSON keys and source IDs are sorted; physical section/sample order stays.
    """
    if not isinstance(model, DistributedShaftModel):
        raise TypeError("model must be DistributedShaftModel")
    payload = {
        "format": DISTRIBUTED_SHAFT_FORMAT,
        "model_id": model.model_id,
        "material_frame_id": model.material_frame_id,
        "sources": [asdict(source) for source in model.sources],
        "sections": [_section_payload(section) for section in model.sections],
    }
    return json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))


def shaft_model_digest(model: DistributedShaftModel) -> str:
    """Hash canonical model JSON, including every coefficient and declaration.

    This identifies the interpreted input model, not the original JSON bytes,
    authenticated authorship or numerical/physical qualification evidence.
    """
    return hashlib.sha256(shaft_model_to_json(model).encode("utf-8")).hexdigest()


def verify_shaft_source_bytes(
    model: DistributedShaftModel, blobs: Mapping[str, bytes]
) -> tuple[str, ...]:
    """Check exactly the declared artifact/calibration bytes and return digests.

    Preconditions: mapping keys are the referenced SHA256 values; values are
    exact bytes. Missing, extra and mismatched artifacts refuse atomically.
    Postcondition: sorted matched digests; the model remains unqualified.
    """
    if not isinstance(model, DistributedShaftModel):
        raise TypeError("model must be DistributedShaftModel")
    mapping = require_mapping(blobs, "source blobs")
    required = {
        value
        for source in model.sources
        for value in (source.artifact_sha256, source.calibration_sha256)
        if value is not None
    }
    if set(mapping) != required:
        raise ValueError("source blobs must exactly match referenced artifact digests")
    for digest, content in mapping.items():
        if not isinstance(content, bytes):
            raise TypeError("source blob values must be bytes")
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(f"source artifact digest mismatch: {digest}")
    return tuple(sorted(required))


__all__ = [
    "DISTRIBUTED_SHAFT_FORMAT",
    "DistributedShaftModel",
    "ShaftModelSection",
    "ShaftModelSource",
    "shaft_model_from_json",
    "shaft_model_to_json",
    "shaft_model_digest",
    "verify_shaft_source_bytes",
]
