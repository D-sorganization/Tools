"""Strict versioned persistence for compliant-turf profiles."""

from __future__ import annotations

import json

import pytest

from shared.python.golf_club import (
    TURF_PROFILE_FORMAT,
    TurfPreset,
    turf_profile_from_json,
    turf_profile_preset,
    turf_profile_to_json,
)


def test_profile_json_round_trip_is_deterministic_and_complete() -> None:
    profile = turf_profile_preset(TurfPreset.SOFT_TURF)

    first = turf_profile_to_json(profile)
    restored = turf_profile_from_json(first)

    assert restored == profile
    assert turf_profile_to_json(restored) == first
    assert json.loads(first)["format"] == TURF_PROFILE_FORMAT
    assert "provenance" in json.loads(first)["profile"]


def test_profile_json_rejects_unknown_fields() -> None:
    payload = json.loads(
        turf_profile_to_json(turf_profile_preset(TurfPreset.SAND_LIKE))
    )
    payload["profile"]["unsupported_claim"] = True

    with pytest.raises(ValueError, match="unknown fields"):
        turf_profile_from_json(json.dumps(payload))


def test_profile_json_rejects_wrong_format() -> None:
    payload = json.loads(
        turf_profile_to_json(turf_profile_preset(TurfPreset.FIRM_FAIRWAY))
    )
    payload["format"] = "golf-club.turf-profile/v999"

    with pytest.raises(ValueError, match="unsupported turf profile format"):
        turf_profile_from_json(json.dumps(payload))
