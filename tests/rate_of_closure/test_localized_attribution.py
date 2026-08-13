"""Strict localized source-to-target attribution authority tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rate_of_closure.variation.localized_attribution import (
    AttributionViewDefinition,
    attribution_authority_from_dict,
    attribution_authority_to_dict,
    attribution_observations_to_csv,
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

    assert "interpretation,source_spec_id,joint_id,window_start_s" in csv_text
    assert "paired-planted-intervention-noncausal" in csv_text
    assert "no_impact_unavailable" in csv_text
    assert "numerical_failure" in csv_text


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
