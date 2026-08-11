"""Strict request wire and request/result pairing for repeated bounce."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from shared.python.swing_sim.ground import (
    MAX_REPEATED_BOUNCE_REQUEST_WIRE_BYTES,
    REPEATED_BOUNCE_REQUEST_SCHEMA_VERSION,
    RepeatedBounceRequest,
    RepeatedBounceRequestResultPair,
    repeated_bounce_request_from_json,
)

from ._support import _settled_prefix, _surface_run_request

_FIXTURES = Path(__file__).parents[5] / "rate_of_closure/web/src/model/__fixtures__"


def _fixture(name: str) -> dict[str, Any]:
    """Load one shared cross-runtime ground fixture."""
    return cast(
        dict[str, Any],
        json.loads((_FIXTURES / name).read_text(encoding="utf-8")),
    )


def _request() -> RepeatedBounceRequest:
    """Build the deterministic analytic request used by the golden fixture."""
    return RepeatedBounceRequest(_surface_run_request(), capture_speed_m_s=0.05)


def test_request_wire_matches_canonical_golden_and_sha256() -> None:
    """The request must round-trip to the exact shared canonical document."""
    fixture = _fixture("ground_repeated_bounce_request_wire_golden_v1.json")
    request = _request()
    text = request.to_json()

    assert json.loads(text) == fixture["request"]
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == fixture["sha256"]
    assert request.schema_version == REPEATED_BOUNCE_REQUEST_SCHEMA_VERSION
    assert request.unit_system == "SI"
    assert repeated_bounce_request_from_json(text) == request
    assert repeated_bounce_request_from_json(text).to_json() == text
    assert request.settings.capture_speed_m_s == pytest.approx(0.05)


def test_request_pairs_with_exact_result_identity_and_ground_fingerprint() -> None:
    """A result from the embedded request must form one immutable pair."""
    request = _request()
    result = _settled_prefix(request.ground_request)
    pair = RepeatedBounceRequestResultPair(request, result)

    assert pair.request is request
    assert pair.result is result
    assert pair.execution_input_sha256 == request.execution_input_sha256
    assert result.request_fingerprint_sha256 == request.ground_request_sha256


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"extra": True}, "fields"),
        ({"schema_version": "future"}, "schema_version"),
        ({"unit_system": "imperial"}, "unit_system"),
        ({"frame": "world"}, "frame"),
        ({"request_id": "wrong"}, "request_id"),
        ({"surface_id": "wrong"}, "surface_id"),
        ({"model_id": "wrong"}, "model_id"),
        ({"model_version": "wrong"}, "model_version"),
        ({"ground_request_sha256": "0" * 64}, "ground_request_sha256"),
        ({"execution_input_sha256": "0" * 64}, "execution_input_sha256"),
        ({"capture_speed_m_s": float("inf")}, "capture_speed_m_s"),
    ],
)
def test_request_wire_rejects_tampered_identity_and_digest(
    change: dict[str, object], message: str
) -> None:
    """Every redundant identity and digest must fail closed on drift."""
    payload = _request().to_dict()
    payload.update(change)

    with pytest.raises((TypeError, ValueError), match=message):
        RepeatedBounceRequest.from_dict(payload)


def test_request_wire_rejects_nested_tampering_duplicates_and_size() -> None:
    """Nested schema, duplicate keys, and UTF-8 size remain bounded."""
    payload = _request().to_dict()
    cast(dict[str, Any], payload["ground_request"])["extra"] = True
    with pytest.raises(ValueError, match="fields"):
        RepeatedBounceRequest.from_dict(payload)

    text = _request().to_json()
    duplicate = text.replace(
        '"request_id":"surface-run-analytic"',
        '"request_id":"duplicate","request_id":"surface-run-analytic"',
        1,
    )
    with pytest.raises(ValueError, match="duplicate"):
        repeated_bounce_request_from_json(duplicate)

    oversized = "é" * (MAX_REPEATED_BOUNCE_REQUEST_WIRE_BYTES // 2 + 1)
    with pytest.raises(ValueError, match="maximum wire size"):
        repeated_bounce_request_from_json(oversized)


def test_request_wire_rejects_finite_capture_speed_digest_drift() -> None:
    """A finite capture-threshold change must invalidate the joint digest."""
    payload = _request().to_dict()
    payload["capture_speed_m_s"] = 0.06

    with pytest.raises(ValueError, match="execution_input_sha256"):
        RepeatedBounceRequest.from_dict(payload)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"request_id": "wrong"}, "request identity"),
        ({"surface_id": "wrong"}, "surface identity"),
        ({"model_id": "wrong"}, "model identity"),
        ({"model_version": "wrong"}, "model identity"),
        ({"request_fingerprint_sha256": "0" * 64}, "fingerprint"),
    ],
)
def test_pairing_rejects_mismatched_result_evidence(
    change: dict[str, object], message: str
) -> None:
    """Pairing must reject mismatched result authority or fingerprints."""
    request = _request()
    result = replace(_settled_prefix(request.ground_request), **change)

    with pytest.raises(ValueError, match=message):
        RepeatedBounceRequestResultPair(request, result)


def test_pairing_requires_exact_records() -> None:
    """Pairing accepts no structural lookalikes or subclasses."""
    request = _request()
    result = _settled_prefix(request.ground_request)

    with pytest.raises(ValueError, match="exact request"):
        RepeatedBounceRequestResultPair(cast(RepeatedBounceRequest, object()), result)
    with pytest.raises(ValueError, match="exact result"):
        RepeatedBounceRequestResultPair(request, cast(Any, object()))
