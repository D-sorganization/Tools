"""Source-backed strokes-gained baseline and calculation contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from rate_of_closure.launch_monitor_strokes_gained import (
    EXCLUSION_REASON_CODES,
    SourceBackedStrokesGainedRequest,
    StrokesGainedExclusionSummary,
    TrustedSummaryRequest,
    baseline_table_hash,
    build_source_backed_strokes_gained_payload,
    calculate_source_backed_strokes_gained,
    load_strokes_gained_baseline,
)
from rate_of_closure.launch_monitor_strokes_gained_baseline import (
    StrokesGainedBaseline as BaselineArtifact,
)
from rate_of_closure.launch_monitor_strokes_gained_baseline import (
    load_strokes_gained_baseline as load_baseline_artifact,
)


def _baseline(path: Path) -> Path:
    states = [
        {
            "lie": "fairway",
            "context": "standard",
            "target": "hole-1",
            "distance_yards": 100.0,
            "expected_strokes": 2.8,
            "standard_error": 0.1,
        },
        {
            "lie": "fairway",
            "context": "standard",
            "target": "hole-1",
            "distance_yards": 200.0,
            "expected_strokes": 3.8,
            "standard_error": 0.14,
        },
        {
            "lie": "green",
            "context": "standard",
            "target": "hole-1",
            "distance_yards": 0.0,
            "expected_strokes": 0.0,
            "standard_error": 0.0,
        },
        {
            "lie": "green",
            "context": "standard",
            "target": "hole-1",
            "distance_yards": 20.0,
            "expected_strokes": 1.5,
            "standard_error": 0.08,
        },
    ]
    payload = {
        "contract_version": "launch-monitor-strokes-gained-baseline/2.0.0",
        "baseline_id": "licensed-test-baseline",
        "version": "2026.1",
        "source_url": "https://example.org/methodology",
        "license": "test-only",
        "table_sha256": baseline_table_hash(states),
        "states": states,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_source_backed_sg_verifies_hash_and_interpolates_course_state(
    tmp_path: Path,
) -> None:
    baseline = load_strokes_gained_baseline(_baseline(tmp_path / "baseline.json"))
    assert baseline.table_sha256 == (
        "5250552cc6ec58da60dfe8ebf50f7238534d28016b0725bf42d8098054404428"  # noqa: E501  # pragma: allowlist secret
    )
    result = calculate_source_backed_strokes_gained(
        pd.DataFrame(
            {
                "before_lie": ["fairway", "fairway"],
                "before_context": ["standard", "standard"],
                "target": ["hole-1", "hole-1"],
                "before_distance": [150.0, 200.0],
                "after_lie": ["green", "green"],
                "after_context": ["standard", "standard"],
                "after_distance": [20.0, 0.0],
            }
        ),
        baseline,
        SourceBackedStrokesGainedRequest(
            "before_lie",
            "before_context",
            "target",
            "before_distance",
            "after_lie",
            "after_context",
            "target",
            "after_distance",
            "yd",
            "yd",
        ),
    )

    assert result.values == pytest.approx((0.8, 2.8))
    assert result.mean == pytest.approx(1.8)
    assert result.baseline_id == "licensed-test-baseline"
    assert result.baseline_version == "2026.1"
    assert result.table_sha256 == baseline.table_sha256
    assert result.backing_rows[0].expected_before == pytest.approx(3.3)
    assert result.status == "available"
    assert result.excluded_rows == ()
    assert result.exclusions == StrokesGainedExclusionSummary(2, 2, 0, {})


def test_baseline_artifact_seam_is_reexported_by_sg_facade(tmp_path: Path) -> None:
    """The artifact authority remains reusable without the calculation façade."""

    direct = load_baseline_artifact(_baseline(tmp_path / "baseline.json"))
    facade = load_strokes_gained_baseline(tmp_path / "baseline.json")

    assert isinstance(direct, BaselineArtifact)
    assert direct == facade


def _request() -> SourceBackedStrokesGainedRequest:
    return SourceBackedStrokesGainedRequest(
        "before_lie",
        "before_context",
        "target",
        "before_distance",
        "after_lie",
        "after_context",
        "target",
        "after_distance",
        "yd",
        "yd",
    )


def _clean_row() -> dict[str, object]:
    return {
        "before_lie": "fairway",
        "before_context": "standard",
        "target": "hole-1",
        "before_distance": 150.0,
        "after_lie": "green",
        "after_context": "standard",
        "after_distance": 20.0,
    }


def test_source_backed_sg_baseline_tamper_still_fails_closed(tmp_path: Path) -> None:
    """Artifact integrity is a request-level defect and still raises."""

    path = _baseline(tmp_path / "baseline.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["states"][0]["expected_strokes"] = 9.9
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256"):
        load_strokes_gained_baseline(path)


def test_source_backed_sg_request_defects_still_raise(tmp_path: Path) -> None:
    """G1-D3 relaxes *row* handling only; the request contract stays fatal."""

    baseline = load_strokes_gained_baseline(_baseline(tmp_path / "baseline.json"))
    frame = pd.DataFrame([_clean_row()])

    with pytest.raises(ValueError, match="columns are unavailable"):
        calculate_source_backed_strokes_gained(
            frame.drop(columns=["after_distance"]), baseline, _request()
        )
    with pytest.raises(ValueError, match="distance unit must be"):
        calculate_source_backed_strokes_gained(
            frame,
            baseline,
            SourceBackedStrokesGainedRequest(
                "before_lie",
                "before_context",
                "target",
                "before_distance",
                "after_lie",
                "after_context",
                "target",
                "after_distance",
                "furlong",
                "yd",
            ),
        )


@pytest.mark.parametrize(
    ("overrides", "reason_code"),
    [
        ({"before_lie": "rough"}, "outside_baseline"),
        ({"before_context": "windy"}, "outside_baseline"),
        ({"before_distance": 400.0}, "outside_baseline"),
        ({"before_lie": "   "}, "missing_course_state"),
        ({"before_lie": None}, "missing_course_state"),
        ({"after_distance": None}, "missing_course_state"),
        ({"after_distance": "not-a-number"}, "missing_course_state"),
        ({"after_distance": -1.5}, "invalid_distance"),
        ({"after_distance": float("inf")}, "invalid_distance"),
    ],
)
def test_source_backed_sg_excludes_and_audits_each_malformed_row(
    tmp_path: Path, overrides: dict[str, object], reason_code: str
) -> None:
    """ADR-0048 G1-D3: one bad row is excluded and audited, never fatal."""

    baseline = load_strokes_gained_baseline(_baseline(tmp_path / "baseline.json"))
    bad = {**_clean_row(), **overrides}
    frame = pd.DataFrame([_clean_row(), bad, _clean_row()])

    result = calculate_source_backed_strokes_gained(frame, baseline, _request())

    assert result.status == "partial"
    assert result.mean == pytest.approx(0.8)
    assert len(result.values) == 2
    assert result.exclusions == StrokesGainedExclusionSummary(3, 2, 1, {reason_code: 1})
    assert [row.source_index for row in result.excluded_rows] == [1]
    assert result.excluded_rows[0].reason_code == reason_code
    assert result.excluded_rows[0].message


def test_source_backed_sg_reports_unavailable_when_no_row_survives(
    tmp_path: Path,
) -> None:
    """Zero scorable rows is an audited ``unavailable`` result, not an exception."""

    baseline = load_strokes_gained_baseline(_baseline(tmp_path / "baseline.json"))
    frame = pd.DataFrame(
        [
            {**_clean_row(), "before_lie": "rough"},
            {**_clean_row(), "before_lie": "   "},
        ]
    )

    result = calculate_source_backed_strokes_gained(frame, baseline, _request())

    assert result.status == "unavailable"
    assert result.mean is None
    assert result.values == ()
    assert result.exclusions == StrokesGainedExclusionSummary(
        2, 0, 2, {"outside_baseline": 1, "missing_course_state": 1}
    )


def test_source_backed_sg_accounts_for_every_supplied_row(tmp_path: Path) -> None:
    """No row is dropped in silence: included + excluded == input, always."""

    baseline = load_strokes_gained_baseline(_baseline(tmp_path / "baseline.json"))
    frame = pd.DataFrame(
        [
            _clean_row(),
            {**_clean_row(), "before_lie": "rough"},
            {**_clean_row(), "after_context": ""},
            {**_clean_row(), "before_distance": -2.0},
        ]
    )

    result = calculate_source_backed_strokes_gained(frame, baseline, _request())
    summary = result.exclusions

    assert summary.input_row_count == len(frame)
    assert summary.included_row_count + summary.total_excluded == len(frame)
    assert summary.included_row_count == len(result.backing_rows) == len(result.values)
    assert summary.total_excluded == len(result.excluded_rows)
    assert sum(summary.by_reason.values()) == summary.total_excluded
    assert set(summary.by_reason) <= set(EXCLUSION_REASON_CODES)
    assert summary.by_reason == {
        "outside_baseline": 1,
        "missing_course_state": 1,
        "invalid_distance": 1,
    }


def test_canonical_payload_only_groups_explicitly_attested_identities(
    tmp_path: Path,
) -> None:
    baseline = load_strokes_gained_baseline(_baseline(tmp_path / "baseline.json"))
    frame = pd.DataFrame({"player": ["p1"], "order": [1]})
    request = SourceBackedStrokesGainedRequest(
        "start_lie",
        "start_context",
        "target",
        "start_distance",
        "finish_lie",
        "finish_context",
        "target",
        "finish_distance",
        "yd",
        "yd",
        TrustedSummaryRequest(player_column="player", order_column="order"),
    )

    payload = build_source_backed_strokes_gained_payload(frame, baseline, request)
    wire = payload["request"]
    assert isinstance(wire, dict)
    assert wire["summaries"][0]["trust_level"] == "explicit_user_attested"
    assert wire["longitudinal"]["group_dimension"] == "player"


def test_pyqt_source_backed_sg_requires_verified_baseline_and_course_state(
    tmp_path: Path, qtbot
) -> None:  # type: ignore[no-untyped-def]
    from rate_of_closure.ui.pyqt6.launch_monitor_source_backed_sg import (
        LaunchMonitorSourceBackedStrokesGainedWidget,
    )

    widget = LaunchMonitorSourceBackedStrokesGainedWidget()
    qtbot.addWidget(widget)
    widget.set_dataset(
        pd.DataFrame(
            {
                "before_lie": ["fairway"],
                "before_context": ["standard"],
                "target": ["hole-1"],
                "before_distance": [150.0],
                "after_lie": ["green"],
                "after_context": ["standard"],
                "after_distance": [20.0],
            }
        )
    )
    assert not widget.calculate_button.isEnabled()
    widget.load_path(_baseline(tmp_path / "baseline.json"))
    widget.before_lie.setCurrentText("before_lie")
    widget.before_context.setCurrentText("before_context")
    widget.before_target.setCurrentText("target")
    widget.before_distance.setCurrentText("before_distance")
    widget.after_lie.setCurrentText("after_lie")
    widget.after_context.setCurrentText("after_context")
    widget.after_target.setCurrentText("target")
    widget.after_distance.setCurrentText("after_distance")
    assert widget.calculate_button.isEnabled()
    result = widget.calculate()
    assert result.mean == pytest.approx(0.8)
    assert "licensed-test-baseline" in widget.status.text()
