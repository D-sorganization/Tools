"""Adversarial coverage for the lossless Morris workspace contract."""

from __future__ import annotations

import csv
import io
import json
from copy import deepcopy
from pathlib import Path

import pytest

from rate_of_closure.application.morris.workspace import (
    MORRIS_WORKSPACE_SCHEMA_ID,
    dumps_morris_workspace,
    loads_morris_workspace,
    morris_report_csv,
    parse_morris_workspace,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

FIXTURE = Path(__file__).parent / "fixtures" / "morris_workspace_v1.json"


def _document() -> dict[str, object]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_v1_fixture_round_trips_losslessly_and_is_deeply_immutable() -> None:
    document = _document()
    workspace = parse_morris_workspace(document)

    assert workspace.schema_id == MORRIS_WORKSPACE_SCHEMA_ID
    assert len(workspace.setup.factor_drafts) == 10
    assert workspace.setup.factor_drafts[-1].enabled
    assert json.loads(dumps_morris_workspace(workspace)) == document
    with pytest.raises(TypeError):
        workspace.setup.base["club_name"] = "mutated"  # type: ignore[index]
    with pytest.raises(TypeError):
        workspace.completed_evidence.request.base.values["club_name"] = "mutated"  # type: ignore[index,union-attr]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda item: item.update(extra=True), "fields"),
        (lambda item: item.update(schema_version=2), "version"),
        (
            lambda item: item["setup"]["factor_drafts"].pop(),
            "canonical",
        ),
        (
            lambda item: item["setup"]["factor_drafts"][1].update(
                variable_key=item["setup"]["factor_drafts"][0]["variable_key"]
            ),
            "canonical",
        ),
        (
            lambda item: item["completed_evidence"]["request"]["base"].update(
                plane_yaw_deg=1.0
            ),
            "base",
        ),
        (
            lambda item: item["completed_evidence"]["request"].update(seed=999),
            "design",
        ),
        (
            lambda item: item["completed_evidence"]["job"].update(
                request_id="crossed-request"
            ),
            "request_id",
        ),
        (
            lambda item: item["completed_evidence"]["job"]["report"]["estimates"][0][
                "source"
            ].update(spec_id="crossed-source"),
            "sources",
        ),
        (
            lambda item: item["completed_evidence"]["job"].update(status="running"),
            "completed",
        ),
    ],
)
def test_workspace_rejects_unknown_fields_and_crossed_evidence(
    mutate: object, message: str
) -> None:
    document = deepcopy(_document())
    mutate(document)  # type: ignore[operator]

    with pytest.raises((TypeError, ValueError), match=message):
        parse_morris_workspace(document)


def test_workspace_without_completed_evidence_is_valid_but_not_exportable() -> None:
    document = _document()
    document["completed_evidence"] = None
    workspace = parse_morris_workspace(document)

    assert workspace.completed_evidence is None
    with pytest.raises(ValueError, match="completed evidence"):
        morris_report_csv(workspace)


def test_disabled_invalid_draft_text_and_validation_state_round_trip() -> None:
    document = _document()
    document["completed_evidence"] = None
    draft = document["setup"]["factor_drafts"][0]
    draft.update(
        enabled=False,
        lower="not-a-number",
        validation_error=("Bounds must be finite numbers with lower < upper."),
    )

    workspace = parse_morris_workspace(document)

    assert workspace.setup.factor_drafts[0].lower == "not-a-number"
    assert json.loads(dumps_morris_workspace(workspace)) == document


@pytest.mark.parametrize("control", ("\x00", "\x1f", "\x7f", "\x9f"))
def test_raw_bounds_reject_c0_and_c1_control_characters(control: str) -> None:
    document = _document()
    document["completed_evidence"] = None
    document["setup"]["factor_drafts"][0]["lower"] = f"1{control}2"

    with pytest.raises(ValueError, match="bounded text"):
        parse_morris_workspace(document)


def test_raw_bound_limit_counts_unicode_code_points() -> None:
    document = _document()
    document["completed_evidence"] = None
    draft = document["setup"]["factor_drafts"][0]
    draft.update(
        enabled=False,
        lower="😀" * 128,
        validation_error="Bounds must be finite numbers with lower < upper.",
    )
    assert len(parse_morris_workspace(document).setup.factor_drafts[0].lower) == 128

    draft["lower"] += "😀"
    with pytest.raises(ValueError, match="bounded text"):
        parse_morris_workspace(document)


def test_factor_bound_outside_pyqt_range_is_canonically_invalid() -> None:
    document = _document()
    document["completed_evidence"] = None
    draft = document["setup"]["factor_drafts"][0]
    draft.update(
        enabled=False,
        lower="-1000000001",
        validation_error=("Bounds must be finite numbers with lower < upper."),
    )

    assert parse_morris_workspace(document).setup.factor_drafts[0].validation_error


@pytest.mark.parametrize("value", ("0", "-1", "+1.25", ".5", "5.", "1e3", "-2.5E-4"))
def test_bound_numeric_lexemes_are_cross_runtime_portable(value: str) -> None:
    document = _document()
    document["completed_evidence"] = None
    draft = document["setup"]["factor_drafts"][0]
    draft.update(enabled=False, lower=value, upper="1000000000", validation_error=None)

    assert parse_morris_workspace(document).setup.factor_drafts[0].lower == value


@pytest.mark.parametrize("value", (" 1", "1 ", "1_0", "0x10", "Infinity", "NaN", "+."))
def test_divergent_numeric_lexemes_require_exact_invalid_state(value: str) -> None:
    document = _document()
    document["completed_evidence"] = None
    draft = document["setup"]["factor_drafts"][0]
    draft.update(
        enabled=False,
        lower=value,
        validation_error="Bounds must be finite numbers with lower < upper.",
    )

    assert parse_morris_workspace(document).setup.factor_drafts[0].validation_error


def test_ground_valid_disabled_tee_has_no_validation_error() -> None:
    document = _document()
    document["completed_evidence"] = None
    document["setup"]["base"].update(support_mode="ground", tee_height_m=0.0)
    tee = document["setup"]["factor_drafts"][-1]
    tee.update(enabled=False, validation_error=None)

    assert (
        parse_morris_workspace(document).setup.factor_drafts[-1].validation_error
        is None
    )


def test_completed_report_assumption_count_is_bounded_before_deep_parse() -> None:
    document = _document()
    document["completed_evidence"]["job"]["report"]["assumptions"] = [
        f"assumption-{index}" for index in range(65)
    ]

    with pytest.raises(ValueError, match="assumption count"):
        parse_morris_workspace(document)


def test_setup_only_preserves_an_all_disabled_incomplete_design() -> None:
    document = _document()
    document["completed_evidence"] = None
    for draft in document["setup"]["factor_drafts"]:
        draft["enabled"] = False

    workspace = parse_morris_workspace(document)

    assert not any(draft.enabled for draft in workspace.setup.factor_drafts)
    assert workspace.base_config().club.name == "Driver 10.5°"


def test_csv_is_deterministic_aggregate_only_and_retains_provenance() -> None:
    workspace = parse_morris_workspace(_document())
    first = morris_report_csv(workspace)
    second = morris_report_csv(workspace)
    rows = list(csv.DictReader(io.StringIO(first)))

    assert first == second and first.endswith("\n")
    assert len(rows) == 10
    assert set(("mu", "mu_star", "mu_star_standard_error", "sigma")) <= set(rows[0])
    assert (
        rows[0]["source_variable_key"] == workspace.setup.factor_drafts[0].variable_key
    )
    assert rows[0]["target_coordinate_frame"]
    assert rows[0]["design_trajectories"] == "12"
    assert rows[0]["denominator_total_pairs"] == "12"
    assert "raw_sample" not in first


def test_csv_neutralizes_formula_text_but_keeps_negative_numbers_numeric() -> None:
    document = _document()
    estimates = document["completed_evidence"]["job"]["report"]["estimates"]
    for estimate in estimates:
        estimate["target"]["coordinate_frame"] = "=malicious-formula"
    rows = list(
        csv.DictReader(io.StringIO(morris_report_csv(parse_morris_workspace(document))))
    )

    assert rows[0]["target_coordinate_frame"] == "'=malicious-formula"
    assert rows[0]["source_lower"] == "-3.0"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('{"schema_id":"a","schema_id":"b"}', "duplicate"),
        ("[" * 34 + "]" * 34, "depth"),
        ('{"value":NaN}', "non-finite"),
        (" " * 2_000_001, "payload"),
        ("[" + ",".join("0" for _ in range(25_001)) + "]", "node"),
    ],
    ids=("duplicate", "depth", "nonfinite", "payload", "nodes"),
)
def test_json_decoder_rejects_duplicate_nonfinite_deep_and_large_payloads(
    text: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        loads_morris_workspace(text)


@pytest.mark.parametrize(
    "forbidden",
    ("token", "url", "headers", "environment", "process", "path", "client"),
)
def test_workspace_allowlist_rejects_ambient_or_transport_fields(
    forbidden: str,
) -> None:
    document = _document()
    document[forbidden] = "must never persist"

    with pytest.raises(ValueError, match="fields"):
        parse_morris_workspace(document)
