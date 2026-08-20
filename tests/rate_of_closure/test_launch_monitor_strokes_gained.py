"""Source-backed strokes-gained baseline and calculation contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from rate_of_closure.launch_monitor_strokes_gained import (
    SourceBackedStrokesGainedRequest,
    baseline_table_hash,
    calculate_source_backed_strokes_gained,
    load_strokes_gained_baseline,
)


def _baseline(path: Path) -> Path:
    states = [
        {"lie": "fairway", "distance_yards": 100.0, "expected_strokes": 2.8},
        {"lie": "fairway", "distance_yards": 200.0, "expected_strokes": 3.8},
        {"lie": "green", "distance_yards": 0.0, "expected_strokes": 0.0},
        {"lie": "green", "distance_yards": 20.0, "expected_strokes": 1.5},
    ]
    payload = {
        "contract_version": "launch-monitor-strokes-gained-baseline/1.0.0",
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
    result = calculate_source_backed_strokes_gained(
        pd.DataFrame(
            {
                "before_lie": ["fairway", "fairway"],
                "before_distance": [150.0, 200.0],
                "after_lie": ["green", "green"],
                "after_distance": [20.0, 0.0],
            }
        ),
        baseline,
        SourceBackedStrokesGainedRequest(
            "before_lie",
            "before_distance",
            "after_lie",
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


def test_source_backed_sg_fails_closed_for_tamper_and_out_of_range(
    tmp_path: Path,
) -> None:
    path = _baseline(tmp_path / "baseline.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["states"][0]["expected_strokes"] = 9.9
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        load_strokes_gained_baseline(path)

    baseline = load_strokes_gained_baseline(_baseline(path))
    frame = pd.DataFrame(
        {
            "before_lie": ["rough"],
            "before_distance": [150.0],
            "after_lie": ["green"],
            "after_distance": [10.0],
        }
    )
    request = SourceBackedStrokesGainedRequest(
        "before_lie", "before_distance", "after_lie", "after_distance", "yd", "yd"
    )
    with pytest.raises(ValueError, match="outside the baseline"):
        calculate_source_backed_strokes_gained(frame, baseline, request)


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
                "before_distance": [150.0],
                "after_lie": ["green"],
                "after_distance": [20.0],
            }
        )
    )
    assert not widget.calculate_button.isEnabled()
    widget.load_path(_baseline(tmp_path / "baseline.json"))
    widget.before_lie.setCurrentText("before_lie")
    widget.before_distance.setCurrentText("before_distance")
    widget.after_lie.setCurrentText("after_lie")
    widget.after_distance.setCurrentText("after_distance")
    assert widget.calculate_button.isEnabled()
    result = widget.calculate()
    assert result.mean == pytest.approx(0.8)
    assert "licensed-test-baseline" in widget.status.text()
