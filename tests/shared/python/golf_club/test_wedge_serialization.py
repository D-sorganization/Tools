"""Versioned persistence tests for wedge-family parameters."""

from __future__ import annotations

import json

import pytest

from shared.python.golf_club import (
    WedgePreset,
    wedge_parameters_from_json,
    wedge_parameters_to_json,
    wedge_preset,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def test_json_round_trip_is_versioned_deterministic_and_lossless() -> None:
    parameters = wedge_preset(WedgePreset.HIGH_BOUNCE)

    first = wedge_parameters_to_json(parameters)
    second = wedge_parameters_to_json(parameters)

    assert first == second
    assert json.loads(first)["format"] == "golf_club.wedge_parameters/1"
    assert wedge_parameters_from_json(first) == parameters


@pytest.mark.parametrize(
    ("payload", "error_type", "message"),
    [
        ("not-json", ValueError, "valid JSON"),
        (
            '{"format":"golf_club.wedge_parameters/99"}',
            ValueError,
            "unsupported",
        ),
        (
            '{"format":"golf_club.wedge_parameters/1","format":"duplicate"}',
            ValueError,
            "duplicate field",
        ),
    ],
)
def test_corrupt_unknown_or_ambiguous_documents_are_rejected(
    payload: str, error_type: type[Exception], message: str
) -> None:
    with pytest.raises(error_type, match=message):
        wedge_parameters_from_json(payload)


def test_unknown_fields_are_rejected() -> None:
    payload = json.loads(wedge_parameters_to_json(wedge_preset(WedgePreset.MID_BOUNCE)))
    payload["parameters"]["unknown"] = True

    with pytest.raises(ValueError, match="unknown fields"):
        wedge_parameters_from_json(json.dumps(payload))
