"""Provenance-complete, non-recomputing Morris result selection contracts."""

from __future__ import annotations

import ast
import json
from copy import deepcopy
from pathlib import Path

import pytest

from rate_of_closure.application.morris.presentation import present_morris_report
from rate_of_closure.application.morris.response_contract import parse_morris_report
from rate_of_closure.application.morris.selector import (
    MorrisReportSelection,
    list_morris_source_options,
    list_morris_target_options,
    select_morris_report,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe, pytest.mark.contract]


def _estimate(
    source_id: str,
    target: dict[str, object],
    mu_star: float,
) -> dict[str, object]:
    return {
        "source": {
            "spec_id": source_id,
            "variable_key": f"swing_sim.swing.{source_id}_deg",
            "unit": "deg",
            "bounds": [-1.0, 1.0],
            "time_window_s": None,
            "point_ids": [],
        },
        "target": target,
        "effects": {
            "mu": mu_star,
            "mu_star": mu_star,
            "mu_star_standard_error": 0.0,
            "sigma": 0.0,
        },
        "availability": "available",
        "sample_adequacy": "limited",
        "denominator": {
            "total_pairs": 4,
            "valid_pairs": 4,
            "typed_no_impact_pairs": 0,
            "no_impact_unavailable_pairs": 0,
            "failed_pairs": 0,
            "nonfinite_pairs": 0,
        },
    }


def _report_document() -> dict[str, object]:
    targets = (
        {
            "name": "clubhead_x_m",
            "unit": "m",
            "kind": "state-point",
            "time_s": 0.10,
            "point_id": "clubhead",
            "coordinate_frame": "app_frame:x_target,y_up,z_right",
        },
        {
            "name": "clubhead_x_m",
            "unit": "m",
            "kind": "state-point",
            "time_s": 0.20,
            "point_id": "grip",
            "coordinate_frame": "app_frame:x_target,y_up,z_right",
        },
        {
            "name": "clubhead_speed_mps",
            "unit": "m/s",
            "kind": "impact",
            "time_s": None,
            "point_id": None,
            "coordinate_frame": "app_frame:x_target,y_up,z_right",
        },
        {
            "name": "carry_m",
            "unit": "m",
            "kind": "shot-outcome",
            "time_s": None,
            "point_id": None,
            "coordinate_frame": "app_frame:x_target,y_up,z_right",
        },
    )
    sources = ("yaw", "forward")
    estimates = [
        _estimate(source_id, target, 10.0 - index)
        for index, target in enumerate(targets)
        for source_id in sources
    ]
    return {
        "schema_id": "swing-sim/morris-global-sensitivity-report",
        "schema_version": 1,
        "method": "morris-elementary-effects",
        "design": {
            "trajectories": 4,
            "levels": 4,
            "seed": 7,
            "total_samples": 12,
            "normalized_step": 2 / 3,
        },
        "assumptions": ["model scenario"],
        "interaction_caveat": "screening only",
        "estimates": estimates,
    }


def test_selector_keeps_point_phase_impact_and_shot_targets_distinct() -> None:
    report = parse_morris_report(_report_document())

    options = list_morris_target_options(report)

    assert len(options) == 4
    state_options = [
        option for option in options if option.identity.kind == "state-point"
    ]
    assert [
        (item.identity.point_id, item.identity.time_s) for item in state_options
    ] == [
        ("clubhead", 0.10),
        ("grip", 0.20),
    ]
    assert {option.identity.kind for option in options} == {
        "state-point",
        "impact",
        "shot-outcome",
    }
    assert all(option.identity.schema_version == 1 for option in options)


def test_python_matches_shared_cross_runtime_selector_fixture() -> None:
    fixture_path = (
        Path(__file__).parents[2]
        / "src/rate_of_closure/web/src/model/__fixtures__"
        / "morris_selector_parity_v1.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    document = _report_document()
    document["estimates"] = [
        _estimate(source_id, target, 4.0 - source_index)
        for target in fixture["targets"]
        for source_index, source_id in enumerate(("yaw", "forward"))
    ]
    report = parse_morris_report(document)
    options = list_morris_target_options(report)

    assert [option.label for option in options] == fixture["expected_labels"]
    assert [
        [option.identity.point_id, option.identity.time_s]
        for option in options
        if option.identity.kind == "state-point"
    ] == fixture["expected_state_points"]


def test_selection_filters_one_source_without_mutating_or_reanalyzing_report() -> None:
    report = parse_morris_report(_report_document())
    original = deepcopy(report)
    target = list_morris_target_options(report)[0].identity
    sources = list_morris_source_options(report, target)

    global_view = select_morris_report(report, MorrisReportSelection(target, None))
    selected_view = select_morris_report(
        report,
        MorrisReportSelection(target, sources[1].spec_id),
    )

    assert len(global_view.rows) == 2
    assert len(selected_view.rows) == 1
    assert selected_view.rows[0].spec_id == sources[1].spec_id
    assert selected_view.rows[0].rank == next(
        row.rank for row in global_view.rows if row.spec_id == sources[1].spec_id
    )
    assert report == original


def test_legacy_name_selector_fails_closed_when_target_name_is_ambiguous() -> None:
    report = parse_morris_report(_report_document())

    with pytest.raises(ValueError, match="ambiguous"):
        present_morris_report(report, "clubhead_x_m")


def test_selector_rejects_unknown_target_and_cross_target_source() -> None:
    report = parse_morris_report(_report_document())
    options = list_morris_target_options(report)
    invalid_target = options[0].identity.__class__(
        **{**options[0].identity.__dict__, "point_id": "missing"}
    )
    with pytest.raises(ValueError, match="target"):
        select_morris_report(report, MorrisReportSelection(invalid_target, None))
    with pytest.raises(ValueError, match="source"):
        select_morris_report(
            report,
            MorrisReportSelection(options[0].identity, "missing-source"),
        )


def test_selector_module_has_no_simulation_or_analysis_dependency() -> None:
    source_path = (
        Path(__file__).parents[2] / "src/rate_of_closure/application/morris/selector.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }

    assert not any(
        forbidden in module
        for module in imported
        for forbidden in ("simulation", "global_sensitivity", "engine", "service")
    )
