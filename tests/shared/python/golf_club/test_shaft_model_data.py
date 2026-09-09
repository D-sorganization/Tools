"""Strict portable coefficients: identity, refusal and existing-kernel reuse."""

import copy
import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from typing import Any

import numpy as np
import pytest

from shared.python.golf_club.shaft_model_data import (
    DistributedShaftModel,
    shaft_model_digest,
    shaft_model_from_json,
    shaft_model_to_json,
    verify_shaft_source_bytes,
)


def _payload() -> dict:
    stiffness = np.diag([3000.0, 4000.0, 5000.0, 20.0, 30.0, 40.0])
    stiffness[0, 4] = stiffness[4, 0] = 2.0
    source = {
        "source_id": "fixture",
        "kind": "synthetic",
        "artifact_sha256": hashlib.sha256(b"fixture").hexdigest(),
        "calibration_sha256": None,
        "method": "declared coefficients",
        "uncertainty_note": "unquantified",
        "data_license": "CC0-1.0",
    }
    body = {
        "component_id": "section",
        "role": "shaft",
        "frame_id": "material",
        "mass_kg": 0.25,
        "center_of_mass_m": [0.01, -0.02, 0.0],
        "inertia_at_com_kg_m2": [[0.003, 0.0001, 0], [0.0001, 0.004, 0], [0, 0, 0.005]],
    }
    section = {
        "length_m": 0.4,
        "reference_twist": [0, 0, 0.4, 0.01, 0, 0],
        "stiffness": stiffness.tolist(),
        "coefficient_source_id": "fixture",
        "inertia_source_id": "fixture",
        "inertia_samples": [
            {"fraction": 0.25, "body": body},
            {"fraction": 0.75, "body": copy.deepcopy(body)},
        ],
    }
    return {
        "format": "golf_club.distributed_shaft/1",
        "model_id": "test",
        "material_frame_id": "material",
        "sources": [source],
        "sections": [section],
    }


def _load(payload: dict) -> DistributedShaftModel:
    result: DistributedShaftModel = shaft_model_from_json(json.dumps(payload))
    return result


def test_roundtrip_owns_inputs_and_preserves_coupling_and_integrated_mass() -> None:
    payload = _payload()
    model = _load(payload)
    wire = shaft_model_to_json(model)
    assert shaft_model_to_json(shaft_model_from_json(wire)) == wire
    record = model.sections[0]
    assert record.elastic.stiffness[0][4] == 2.0
    assert sum(sample.body.mass_kg for sample in record.inertia.samples) == 0.5
    payload["sections"][0]["stiffness"][0][4] = 100
    assert shaft_model_to_json(model) == wire
    assert model.validation_status == "unqualified"
    with pytest.raises(FrozenInstanceError):
        model.model_id = "changed"


@pytest.mark.parametrize(
    "path",
    [
        (),
        ("sources", 0),
        ("sections", 0),
        ("sections", 0, "inertia_samples", 0),
        ("sections", 0, "inertia_samples", 0, "body"),
    ],
)
def test_unknown_fields_refuse_at_every_level(path: tuple) -> None:
    payload = _payload()
    item = payload
    for key in path:
        item = item[key]
    item["unreviewed_extension"] = True
    with pytest.raises(ValueError, match="unknown"):
        _load(payload)


@pytest.mark.parametrize("field", list(_payload()))
def test_missing_top_level_fields_refuse(field: str) -> None:
    payload = _payload()
    del payload[field]
    with pytest.raises((TypeError, ValueError)):
        _load(payload)


@pytest.mark.parametrize("value", ["golf_club.distributed_shaft/2", None, True])
def test_unsupported_format_refuses(value: object) -> None:
    payload = _payload()
    payload["format"] = value
    with pytest.raises((TypeError, ValueError), match="format"):
        _load(payload)


def test_duplicate_json_key_refuses_before_ambiguous_interpretation() -> None:
    text = json.dumps(_payload()).replace(
        '"mass_kg": 0.25', '"mass_kg": 9, "mass_kg": 0.25'
    )
    with pytest.raises(ValueError, match="duplicate"):
        shaft_model_from_json(text)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "unused",
        "frame",
        "empty_sections",
        "empty_inertia",
        "active_stiffness",
        "bool_stiffness",
    ],
)
def test_references_frames_and_physical_domains_refuse(mutation: str) -> None:
    payload = _payload()
    section = payload["sections"][0]
    if mutation == "missing":
        section["inertia_source_id"] = "absent"
    elif mutation == "duplicate":
        payload["sources"].append(copy.deepcopy(payload["sources"][0]))
    elif mutation == "unused":
        payload["sources"].append(dict(payload["sources"][0], source_id="unused"))
    elif mutation == "frame":
        payload["material_frame_id"] = "other"
    elif mutation == "empty_sections":
        payload["sections"] = []
    elif mutation == "empty_inertia":
        section["inertia_samples"] = []
    elif mutation == "active_stiffness":
        section["stiffness"][0][0] = -1
    else:
        section["stiffness"][0][0] = True
    with pytest.raises((TypeError, ValueError)):
        _load(payload)


@pytest.mark.parametrize("field", ["kind", "artifact_sha256", "calibration_sha256"])
def test_source_declarations_refuse_invalid_values(field: str) -> None:
    payload = _payload()
    payload["sources"][0][field] = "measured_validated"
    with pytest.raises(ValueError):
        _load(payload)


def test_measurement_declaration_needs_calibration_but_does_not_validate_model() -> (
    None
):
    payload = _payload()
    source = payload["sources"][0]
    source["kind"] = "measurement-derived"
    with pytest.raises(ValueError, match="calibration"):
        _load(payload)
    source["calibration_sha256"] = hashlib.sha256(b"calibration").hexdigest()
    model = _load(payload)
    blobs = {
        source["artifact_sha256"]: b"fixture",
        source["calibration_sha256"]: b"calibration",
    }
    assert verify_shaft_source_bytes(model, blobs) == tuple(sorted(blobs))
    assert model.validation_status == "unqualified"


@pytest.mark.parametrize("mutation", ["missing", "extra", "changed", "not_bytes"])
def test_blob_verification_refuses_incomplete_or_mismatched_artifacts(
    mutation: str,
) -> None:
    model = _load(_payload())
    key = model.sources[0].artifact_sha256
    blobs: dict[str, Any] = {key: b"fixture"}
    if mutation == "missing":
        blobs.clear()
    elif mutation == "extra":
        blobs["a" * 64] = b"extra"
    elif mutation == "changed":
        blobs[key] = b"changed"
    else:
        blobs[key] = "fixture"
    with pytest.raises((TypeError, ValueError)):
        verify_shaft_source_bytes(model, blobs)


def test_digest_binds_coefficients_mass_and_provenance_without_json_whitespace() -> (
    None
):
    payload = _payload()
    model = _load(payload)
    digest = shaft_model_digest(model)
    assert digest == hashlib.sha256(shaft_model_to_json(model).encode()).hexdigest()
    assert (
        shaft_model_digest(shaft_model_from_json(json.dumps(payload, indent=2)))
        == digest
    )
    for mutation in ("stiffness", "mass", "provenance"):
        changed = copy.deepcopy(payload)
        if mutation == "stiffness":
            changed["sections"][0]["stiffness"][0][0] += 1
        elif mutation == "mass":
            changed["sections"][0]["inertia_samples"][0]["body"]["mass_kg"] += 0.01
        else:
            changed["sources"][0]["uncertainty_note"] = "a different declaration"
        assert shaft_model_digest(_load(changed)) != digest


def test_direct_replacement_cannot_detach_source_graph() -> None:
    model = _load(_payload())
    with pytest.raises(ValueError, match="source"):
        replace(model, sources=())


@pytest.mark.parametrize(
    "field,index", [("center_of_mass_m", (0,)), ("inertia_at_com_kg_m2", (0, 0))]
)
@pytest.mark.parametrize("value", [True, "0.003"])
def test_wire_mass_arrays_refuse_scalar_coercion(
    field: str, index: tuple, value: object
) -> None:
    payload = _payload()
    target = payload["sections"][0]["inertia_samples"][0]["body"][field]
    for key in index[:-1]:
        target = target[key]
    target[index[-1]] = value
    with pytest.raises(TypeError, match="boolean|strings|real"):
        _load(payload)
