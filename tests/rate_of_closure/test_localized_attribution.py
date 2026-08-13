"""Strict localized source-to-target attribution authority tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rate_of_closure.variation._localized_attribution_contract import response_matches
from rate_of_closure.variation._localized_attribution_csv import (
    canonical_binary64_csv_text,
)
from rate_of_closure.variation.localized_attribution import (
    AttributionViewDefinition,
    attribution_authority_from_dict,
    attribution_authority_to_dict,
    attribution_observations_to_csv,
    attribution_observations_to_rows,
    attribution_view_from_json,
    attribution_view_to_json,
    build_attribution_view,
)
from shared.python.contracts import ContractViolationError

FIXTURE = Path(__file__).parent / "fixtures" / "localized_attribution_authority_v1.json"
WEB_FIXTURE = (
    Path(__file__).parents[2]
    / "src"
    / "rate_of_closure"
    / "web"
    / "src"
    / "model"
    / "__fixtures__"
    / "localized_attribution_authority_v1.json"
)
CSV_ROWS_FIXTURE = (
    Path(__file__).parent / "fixtures" / "localized_attribution_csv_rows_v1.json"
)
FLOAT_TEXT_FIXTURE = (
    Path(__file__).parent / "fixtures" / "localized_attribution_float_text_v1.json"
)
WEB_FLOAT_TEXT_FIXTURE = (
    Path(__file__).parents[2]
    / "src/rate_of_closure/web/src/model/__fixtures__"
    / "localized_attribution_float_text_v1.json"
)
EXTREME_CSV_AUTHORITY = (
    Path(__file__).parent / "fixtures/localized_attribution_extreme_csv_v1.json"
)
EXTREME_CSV_BYTES = (
    Path(__file__).parent / "fixtures/localized_attribution_extreme_csv_v1.csv"
)
WEB_EXTREME_CSV_BYTES = (
    Path(__file__).parents[2]
    / "src/rate_of_closure/web/src/model/__fixtures__"
    / "localized_attribution_extreme_csv_v1.csv"
)


def _payload() -> dict[str, object]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_cross_runtime_fixture_is_exact_and_strictly_round_trips() -> None:
    assert json.loads(WEB_FIXTURE.read_text(encoding="utf-8")) == _payload()
    authority = attribution_authority_from_dict(_payload())

    assert attribution_authority_to_dict(authority) == _payload()
    assert authority.interpretation == "paired-planted-intervention-noncausal"
    assert authority.sources[0].joint_id == "joint.shoulder"
    assert authority.targets[0].point_id == "swing.clubhead.reference"


def test_empty_authority_fails_closed() -> None:
    payload = _payload()
    payload["observations"] = []

    with pytest.raises(ContractViolationError, match="observations"):
        attribution_authority_from_dict(payload)


def test_selected_pair_and_denominator_keep_misses_and_failures_typed() -> None:
    authority = attribution_authority_from_dict(_payload())
    definition = AttributionViewDefinition(
        authority_id=authority.authority_id,
        source_spec_id="fixture.shoulder",
        target_id="state.clubhead.x.0_002",
        baseline_trial_index=0,
        perturbed_trial_index=2,
    )

    view = build_attribution_view(authority, definition)

    assert view.selected.response == pytest.approx(0.1)
    assert view.selected.perturbed_status.value == "evaluated_no_impact"
    assert view.denominator.total_pairs == 3
    assert view.denominator.available_pairs == 2
    assert view.denominator.typed_no_impact_pairs == 1
    assert view.denominator.unavailable_no_impact_pairs == 0
    assert view.denominator.failed_pairs == 1


def test_impact_target_reports_no_impact_as_unavailable_not_zero() -> None:
    authority = attribution_authority_from_dict(_payload())
    definition = AttributionViewDefinition(
        authority_id=authority.authority_id,
        source_spec_id="fixture.shoulder",
        target_id="impact.clubhead_speed",
        baseline_trial_index=0,
        perturbed_trial_index=2,
    )

    view = build_attribution_view(authority, definition)

    assert view.selected.perturbed_target_value is None
    assert view.selected.response is None
    assert view.selected.availability.value == "no_impact_unavailable"
    assert view.denominator.available_pairs == 1
    assert view.denominator.unavailable_no_impact_pairs == 1
    assert view.denominator.failed_pairs == 1


def test_view_definition_is_exact_versioned_and_rejects_coercion() -> None:
    definition = AttributionViewDefinition(
        authority_id="fixture.localized-attribution.v1",
        source_spec_id="fixture.shoulder",
        target_id="shot.carry",
        baseline_trial_index=0,
        perturbed_trial_index=1,
    )
    encoded = attribution_view_to_json(definition)

    assert attribution_view_from_json(encoded) == definition
    payload = json.loads(encoded)
    payload["baseline_trial_index"] = "0"
    with pytest.raises(ContractViolationError, match="baseline_trial_index"):
        attribution_view_from_json(json.dumps(payload))
    payload = json.loads(encoded)
    payload["extra"] = True
    with pytest.raises(ContractViolationError, match="fields"):
        attribution_view_from_json(json.dumps(payload))


def test_raw_export_includes_provenance_caveat_and_typed_unavailability() -> None:
    authority = attribution_authority_from_dict(_payload())

    csv_text = attribution_observations_to_csv(authority)

    assert "schema_id,schema_version,authority_id,interpretation" in csv_text
    assert "source_variable,source_unit" in csv_text
    assert "target_unit,target_frame,target_convention" in csv_text
    assert "paired-planted-intervention-noncausal" in csv_text
    assert "no_impact_unavailable" in csv_text
    assert "numerical_failure" in csv_text
    assert ",'-2.0," not in csv_text
    assert ",-2.0," in csv_text
    assert attribution_observations_to_rows(authority) == json.loads(
        CSV_ROWS_FIXTURE.read_text(encoding="utf-8")
    )


def test_pair_roster_matrix_and_cross_target_identity_fail_closed() -> None:
    for mutate, message in (
        (lambda raw: raw["observations"].pop(), "matrix"),
        (lambda raw: raw["pairs"].pop(), "matrix"),
        (
            lambda raw: raw["observations"][3].__setitem__(
                "perturbed_status", "evaluated_no_impact"
            ),
            "pair roster",
        ),
        (
            lambda raw: raw["observations"][3].__setitem__(
                "perturbed_source_value", 99.0
            ),
            "pair roster",
        ),
    ):
        payload = _payload()
        mutate(payload)
        with pytest.raises(ContractViolationError, match=message):
            attribution_authority_from_dict(payload)


def test_orphans_caps_safe_indices_and_target_registry_fail_closed() -> None:
    cases = (
        (
            lambda raw: raw["sources"].append(
                {**raw["sources"][0], "spec_id": "orphan"}
            ),
            "matrix",
        ),
        (lambda raw: raw["targets"][0].__setitem__("unit", "ft"), "target registry"),
        (
            lambda raw: raw["targets"][0].__setitem__("convention", "forged"),
            "target registry",
        ),
        (
            lambda raw: raw["pairs"][0].__setitem__(
                "baseline_trial_index", 9007199254740992
            ),
            "safe integer",
        ),
        (lambda raw: raw.__setitem__("authority_id", "x" * 257), "length"),
    )
    for mutate, message in cases:
        payload = _payload()
        mutate(payload)
        with pytest.raises(ContractViolationError, match=message):
            attribution_authority_from_dict(payload)


def test_response_uses_shared_four_scaled_ulp_policy() -> None:
    payload = _payload()
    expected = (
        payload["observations"][0]["perturbed_target_value"]
        - payload["observations"][0]["baseline_target_value"]
    )
    tolerance = 4.0 * 2.220446049250313e-16
    payload["observations"][0]["response"] = expected + tolerance
    attribution_authority_from_dict(payload)
    payload["observations"][0]["response"] = expected + 2.0 * tolerance
    with pytest.raises(ContractViolationError, match="response"):
        attribution_authority_from_dict(payload)


def test_response_rejects_overflowed_expected_value_and_zero_intervention() -> None:
    assert not response_matches(0.0, float("inf"))
    assert not response_matches(0.0, float("-inf"))
    overflow = _payload()
    overflow["observations"][0]["baseline_target_value"] = -float.fromhex(
        "0x1.fffffffffffffp+1023"
    )
    overflow["observations"][0]["perturbed_target_value"] = float.fromhex(
        "0x1.fffffffffffffp+1023"
    )
    overflow["observations"][0]["response"] = 0.0
    with pytest.raises(ContractViolationError, match="response"):
        attribution_authority_from_dict(overflow)
    payload = _payload()
    payload["pairs"][0]["perturbed_source_value"] = 0.0
    for row in payload["observations"]:
        if (
            row["source_spec_id"] == "fixture.shoulder"
            and row["perturbed_trial_index"] == 1
        ):
            row["perturbed_source_value"] = 0.0
    with pytest.raises(ContractViolationError, match="nonzero"):
        attribution_authority_from_dict(payload)


def test_binary64_csv_text_is_cross_runtime_canonical() -> None:
    golden = json.loads(FLOAT_TEXT_FIXTURE.read_text(encoding="utf-8"))
    assert golden == json.loads(WEB_FLOAT_TEXT_FIXTURE.read_text(encoding="utf-8"))
    assert [canonical_binary64_csv_text(row["value"]) for row in golden] == [
        row["text"] for row in golden
    ]


def test_extreme_binary64_csv_writer_matches_cross_runtime_bytes() -> None:
    authority = attribution_authority_from_dict(
        json.loads(EXTREME_CSV_AUTHORITY.read_text(encoding="utf-8"))
    )
    expected = EXTREME_CSV_BYTES.read_text(encoding="utf-8")
    assert expected == WEB_EXTREME_CSV_BYTES.read_text(encoding="utf-8")
    assert attribution_observations_to_csv(authority) == expected
    assert ",-0.0,1e20,-0.0,1e-5,1e-5," in expected


def test_resource_caps_apply_before_element_construction() -> None:
    payload = _payload()
    payload["sources"] = [None] * 33
    with pytest.raises(ContractViolationError, match="resource cap"):
        attribution_authority_from_dict(payload)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("schema_version",), "1", "schema_version"),
        (("schema_version",), 1.0, "schema_version"),
        (("sources", 0, "time_window_s", 0), "0.001", "window"),
        (("sources", 0, "joint_id"), "swing.wrist", "joint"),
        (("observations", 0, "response"), 99.0, "response"),
        (("observations", 4, "perturbed_target_value"), 0.0, "unavailable"),
        (("observations", 0, "availability"), "forged", "availability"),
    ],
)
def test_authority_rejects_wire_bypasses(
    path: tuple[object, ...], value: object, message: str
) -> None:
    root = _payload()
    payload: object = root
    for key in path[:-1]:
        payload = payload[key]  # type: ignore[index]
    payload[path[-1]] = value  # type: ignore[index]

    with pytest.raises(ContractViolationError, match=message):
        attribution_authority_from_dict(root)
