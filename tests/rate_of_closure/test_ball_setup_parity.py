"""Golden parity coverage for Python and React ball-support semantics."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from rate_of_closure.club import get_club
from rate_of_closure.model import ImpactScenario
from rate_of_closure.simulation import (
    BallSetup,
    BallSupportMode,
    SimulationConfig,
    ball_setup_from_json_dict,
)
from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.ball_setup import HEIGHT_REFERENCE
from shared.python.swing_sim.impact import GOLF_BALL_RADIUS_M

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_FIXTURE_PATH = (
    Path(__file__).parents[2]
    / "src"
    / "rate_of_closure"
    / "web"
    / "src"
    / "model"
    / "__fixtures__"
    / "ball_setup_golden_v1.json"
)
_SCENARIO = ImpactScenario(clubhead_speed_mph=113.0)


def _fixture() -> dict[str, Any]:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


def _setup(document: dict[str, Any]) -> BallSetup:
    return BallSetup.from_json_dict(document)


def _height(value: float | str) -> float:
    if value == "nan":
        return math.nan
    if value == "positive_infinity":
        return math.inf
    if value == "negative_infinity":
        return -math.inf
    return float(value)


def _assert_matches_golden(actual: BallSetup, expected: dict[str, Any]) -> None:
    assert set(expected) == {
        "support_mode",
        "tee_height_m",
        "height_reference",
        "ball_center_m",
    }
    serialized = actual.to_json_dict()
    assert serialized["support_mode"] == expected["support_mode"]
    assert serialized["tee_height_m"] == pytest.approx(expected["tee_height_m"])
    assert serialized["height_reference"] == expected["height_reference"]
    assert serialized["ball_center_m"] == pytest.approx(expected["ball_center_m"])


def test_golden_fixture_has_strict_versioned_si_contract() -> None:
    fixture = _fixture()

    assert set(fixture) == {
        "schema",
        "schema_version",
        "units",
        "height_reference",
        "ball_radius_m",
        "default_cases",
        "override_cases",
        "geometry_cases",
        "invalid_cases",
        "legacy_cases",
    }
    assert fixture["schema"] == "rate_of_closure.ball_setup_golden"
    assert fixture["schema_version"] == 1
    assert fixture["units"] == {"length": "m"}
    assert fixture["height_reference"] == HEIGHT_REFERENCE
    assert fixture["ball_radius_m"] == pytest.approx(GOLF_BALL_RADIUS_M)


def test_golden_defaults_and_explicit_overrides_match_python() -> None:
    fixture = _fixture()

    for case in fixture["default_cases"]:
        actual = SimulationConfig(
            scenario=_SCENARIO,
            club=get_club(case["club_name"]),
        ).ball_setup
        _assert_matches_golden(actual, case["expected"])

    for case in fixture["override_cases"]:
        requested = _setup(case["input"])
        actual = SimulationConfig(
            scenario=_SCENARIO,
            club=get_club(case["club_name"]),
            ball_setup=requested,
        ).ball_setup
        _assert_matches_golden(actual, case["expected"])


def test_golden_geometry_and_ground_effective_height_match_python() -> None:
    for case in _fixture()["geometry_cases"]:
        setup = _setup(case["input"])

        assert setup.tee_height_m == pytest.approx(case["effective_tee_height_m"])
        assert setup.ball_center_m == pytest.approx(case["ball_center_m"])
        _assert_matches_golden(setup, case["serialized"])


def test_golden_invalid_heights_fail_closed_in_python() -> None:
    for case in _fixture()["invalid_cases"]:
        with pytest.raises(ContractViolationError, match=case["error_pattern"]):
            BallSetup(
                BallSupportMode(case["support_mode"]),
                _height(case["tee_height"]),
            )


def test_golden_legacy_documents_migrate_to_ground_in_python() -> None:
    for case in _fixture()["legacy_cases"]:
        actual = ball_setup_from_json_dict(case["document"])

        _assert_matches_golden(actual, case["expected"])
