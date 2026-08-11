"""Strict cross-runtime wire-contract tests for repeated-bounce evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from shared.python.swing_sim.ground import (
    MAX_REPEATED_BOUNCE_WIRE_BYTES,
    REPEATED_BOUNCE_SCHEMA_VERSION,
    RepeatedBounceResult,
    repeated_bounce_result_from_json,
    repeated_bounce_result_to_json,
)

from ._support import _settled_prefix, _surface_run_request

_FIXTURE = (
    Path(__file__).parents[5]
    / "rate_of_closure/web/src/model/__fixtures__"
    / "ground_repeated_bounce_wire_golden_v1.json"
)


def _result() -> RepeatedBounceResult:
    return _settled_prefix(_surface_run_request())


def _fixture() -> dict[str, object]:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def test_repeated_bounce_wire_matches_canonical_golden_and_sha256() -> None:
    fixture = _fixture()
    text = repeated_bounce_result_to_json(_result())

    assert json.loads(text) == fixture["result"]
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == fixture["sha256"]
    parsed = repeated_bounce_result_from_json(text)
    assert parsed.termination.elapsed_time_s == pytest.approx(0.02)
    assert repeated_bounce_result_to_json(parsed) == text
    assert parsed.to_json() == text
    assert parsed.to_dict() == json.loads(text)
    assert json.loads(text)["schema_version"] == REPEATED_BOUNCE_SCHEMA_VERSION
    assert json.loads(text)["unit_system"] == "SI"


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda value: value.update(extra=True), "fields do not match"),
        (lambda value: value.pop("events"), "fields do not match"),
        (lambda value: value.update(schema_version="future"), "schema_version"),
        (lambda value: value.update(unit_system="imperial"), "unit_system"),
        (lambda value: value.update(frame="world"), "[Ff]rame"),
        (lambda value: value.update(request_fingerprint_sha256="0"), "fingerprint"),
        (
            lambda value: value["trajectory"][0].update(extra=True),
            "fields do not match",
        ),
        (
            lambda value: value["impacts"][0]["energy"].update(extra=True),
            "fields do not match",
        ),
        (
            lambda value: value["airborne_segments"][0].update(end_time_s=float("inf")),
            "finite",
        ),
        (
            lambda value: value.update(handoff_state=None),
            "settled bounce prefix requires a handoff state",
        ),
        (
            lambda value: value["termination"].update(time_s=999.0),
            "termination time must match",
        ),
        (
            lambda value: value["termination"].update(elapsed_time_s=999.0),
            "elapsed time must match",
        ),
        (
            lambda value: value["impacts"][0]["energy"].update(dissipation_j=999.0),
            "energy balance",
        ),
        (
            lambda value: (
                value["impacts"][0]["state_before"].update(time_s=1.0050000005),
                value["impacts"][0]["state_after"].update(time_s=1.0050000005),
            ),
            "event states must match",
        ),
        (
            lambda value: (
                value["trajectory"][-1]["position_m"].__setitem__(0, 1.14),
                value["handoff_state"]["position_m"].__setitem__(0, 1.14),
            ),
            "matching post-impact trajectory point",
        ),
        (
            lambda value: value.update(trajectory=[]),
            "bounce events require trajectory evidence",
        ),
    ],
)
def test_repeated_bounce_wire_rejects_malformed_or_ambiguous_records(
    mutate, match: str
) -> None:  # type: ignore[no-untyped-def]
    payload = _fixture()["result"]
    assert isinstance(payload, dict)
    mutate(payload)

    with pytest.raises((TypeError, ValueError), match=match):
        RepeatedBounceResult.from_dict(payload)


def test_repeated_bounce_wire_rejects_duplicate_keys_and_oversize_utf8() -> None:
    text = repeated_bounce_result_to_json(_result())
    duplicated = text.replace(
        '"request_id":"surface-run-analytic"',
        '"request_id":"duplicate","request_id":"surface-run-analytic"',
        1,
    )
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        repeated_bounce_result_from_json(duplicated)

    oversized = "é" * (MAX_REPEATED_BOUNCE_WIRE_BYTES // 2 + 1)
    with pytest.raises(ValueError, match="maximum wire size"):
        repeated_bounce_result_from_json(oversized)


def test_parser_reuses_record_invariants_for_cross_record_evidence() -> None:
    payload = _fixture()["result"]
    assert isinstance(payload, dict)
    payload["handoff_state"]["position_m"][0] += 1.0

    with pytest.raises(ValueError, match="handoff state must match"):
        RepeatedBounceResult.from_dict(payload)

    result = _result()
    with pytest.raises(ValueError, match="canonical evidence"):
        repeated_bounce_result_to_json(
            replace(result, warnings=(" valid but not canonical ",))
        )
