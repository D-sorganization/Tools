"""Tests for the advisory asset-health detectors (SCADA F10, issue #5531).

Pure stdlib module, so these run in CI with no PLC, DB or FastAPI. Covers the
flatline and command/feedback-mismatch detectors over synthetic series, the
NaN/None bad-quality handling, DbC parameter validation, and the safety
property that findings are advisory-only with no import path to PLC, alarm or
trip state.
"""

from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import asset_health  # noqa: E402
from asset_health import (  # noqa: E402
    ADVISORY,
    FindingKind,
    HealthFinding,
    TimedSample,
    detect_command_feedback_mismatch,
    detect_flatline,
)


def _series(values, dt=1.0):
    return [TimedSample(i * dt, v) for i, v in enumerate(values)]


# --- flatline -------------------------------------------------------------


def test_constant_signal_beyond_min_duration_flags_flatline():
    findings = detect_flatline(_series([5.0] * 12), min_duration=10.0, tolerance=0.01)
    assert len(findings) == 1
    f = findings[0]
    assert f.kind is FindingKind.FLATLINE
    assert (f.start_s, f.end_s) == (0.0, 11.0)


def test_constant_signal_shorter_than_min_duration_is_not_flagged():
    assert detect_flatline(_series([5.0] * 5), min_duration=10.0, tolerance=0.01) == []


def test_noisy_live_signal_is_not_flagged():
    values = [5.0 + 0.5 * math.sin(i) for i in range(60)]
    assert detect_flatline(_series(values), min_duration=10.0, tolerance=0.01) == []


def test_step_change_splits_flat_runs():
    values = [1.0] * 6 + [9.0] * 6
    findings = detect_flatline(_series(values), min_duration=4.0, tolerance=0.01)
    assert [(f.start_s, f.end_s) for f in findings] == [(0.0, 5.0), (6.0, 11.0)]


def test_step_change_inside_window_prevents_flatline():
    values = [1.0] * 5 + [9.0] * 5
    assert detect_flatline(_series(values), min_duration=8.0, tolerance=0.01) == []


@pytest.mark.parametrize("bad", [None, math.nan, math.inf, -math.inf])
def test_bad_samples_break_run_and_are_never_counted_flat(bad):
    values = [5.0] * 5 + [bad] * 10 + [5.0] * 5
    assert detect_flatline(_series(values), min_duration=8.0, tolerance=0.01) == []


def test_all_bad_samples_yield_no_finding():
    assert detect_flatline(_series([None] * 30), min_duration=5.0, tolerance=0.01) == []


def test_flatline_within_tolerance_band():
    values = [5.0, 5.004, 4.998, 5.002, 5.0, 5.001]
    assert len(detect_flatline(_series(values), min_duration=4.0, tolerance=0.01)) == 1


def test_empty_series_yields_nothing():
    assert detect_flatline([], min_duration=1.0, tolerance=0.01) == []


# --- command / feedback mismatch ------------------------------------------


def test_sustained_mismatch_beyond_dwell_flags():
    cmd = _series([50.0] * 10)
    fb = _series([10.0] * 10)
    findings = detect_command_feedback_mismatch(cmd, fb, tolerance=2.0, window=5.0)
    assert len(findings) == 1
    assert findings[0].kind is FindingKind.COMMAND_FEEDBACK_MISMATCH
    assert (findings[0].start_s, findings[0].end_s) == (0.0, 9.0)


def test_transient_gap_shorter_than_window_is_not_flagged():
    cmd = _series([50.0] * 10)
    fb = _series([50.0, 50.0, 10.0, 10.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0])
    assert detect_command_feedback_mismatch(cmd, fb, tolerance=2.0, window=5.0) == []


def test_feedback_within_tolerance_is_not_flagged():
    cmd = _series([50.0] * 10)
    fb = _series([51.5] * 10)
    assert detect_command_feedback_mismatch(cmd, fb, tolerance=2.0, window=3.0) == []


def test_tracking_a_ramping_command_is_not_flagged():
    cmd = _series([float(i) for i in range(20)])
    fb = _series([i - 0.5 for i in range(20)])
    assert detect_command_feedback_mismatch(cmd, fb, tolerance=1.0, window=3.0) == []


@pytest.mark.parametrize("bad", [None, math.nan])
def test_bad_feedback_is_not_evidence_of_mismatch(bad):
    cmd = _series([50.0] * 10)
    fb = _series([bad] * 10)
    assert detect_command_feedback_mismatch(cmd, fb, tolerance=2.0, window=3.0) == []


def test_bad_sample_breaks_a_mismatch_run():
    cmd = _series([50.0] * 10)
    fb = _series([10.0, 10.0, 10.0, None, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    findings = detect_command_feedback_mismatch(cmd, fb, tolerance=2.0, window=5.0)
    assert [(f.start_s, f.end_s) for f in findings] == [(4.0, 9.0)]


# --- DbC ------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_flatline_rejects_non_positive_or_non_finite_parameters(bad):
    with pytest.raises(ValueError):
        detect_flatline(_series([1.0, 1.0]), min_duration=bad, tolerance=0.01)
    with pytest.raises(ValueError):
        detect_flatline(_series([1.0, 1.0]), min_duration=1.0, tolerance=bad)


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_mismatch_rejects_non_positive_or_non_finite_parameters(bad):
    s = _series([1.0, 1.0])
    with pytest.raises(ValueError):
        detect_command_feedback_mismatch(s, s, tolerance=bad, window=1.0)
    with pytest.raises(ValueError):
        detect_command_feedback_mismatch(s, s, tolerance=1.0, window=bad)


def test_parameters_have_no_defaults():
    with pytest.raises(TypeError):
        detect_flatline(_series([1.0]))  # type: ignore[call-arg]


def test_rejects_bool_parameters():
    with pytest.raises(TypeError):
        detect_flatline(_series([1.0]), min_duration=True, tolerance=0.01)


def test_rejects_non_monotonic_timestamps():
    bad = [TimedSample(0.0, 1.0), TimedSample(0.0, 1.0)]
    with pytest.raises(ValueError):
        detect_flatline(bad, min_duration=1.0, tolerance=0.01)


def test_rejects_misaligned_command_and_feedback():
    with pytest.raises(ValueError):
        detect_command_feedback_mismatch(
            _series([1.0] * 3), _series([1.0] * 4), tolerance=1.0, window=1.0
        )
    with pytest.raises(ValueError):
        detect_command_feedback_mismatch(
            _series([1.0] * 3), _series([1.0] * 3, dt=2.0), tolerance=1.0, window=1.0
        )


def test_rejects_non_numeric_sample_value():
    with pytest.raises(TypeError):
        TimedSample(0.0, "5")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        TimedSample(0.0, True)  # type: ignore[arg-type]


# --- advisory-only safety property ----------------------------------------


def test_findings_are_tagged_advisory_and_immutable():
    f = detect_flatline(_series([5.0] * 12), min_duration=10.0, tolerance=0.01)[0]
    assert isinstance(f, HealthFinding)
    assert f.finding_type == ADVISORY == "advisory"
    with pytest.raises(AttributeError):
        f.finding_type = "alarm"  # type: ignore[misc]


def test_finding_type_cannot_be_overridden_at_construction():
    with pytest.raises(TypeError):
        HealthFinding(  # type: ignore[call-arg]
            kind=FindingKind.FLATLINE,
            start_s=0.0,
            end_s=1.0,
            message="x",
            finding_type="alarm",
        )


def test_finding_has_no_trip_or_priority_field():
    names = set(HealthFinding.__dataclass_fields__)
    assert not {n for n in names if "trip" in n or "priority" in n or "alarm" in n}


_FORBIDDEN_FRAGMENTS = (
    "plc",
    "alarm",
    "trip",
    "interlock",
    "setpoint",
    "fastapi",
    "driver",
    "config_store",
)


def _imported_names() -> list[str]:
    tree = ast.parse(Path(asset_health.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
            imported += [a.name for a in node.names]
    return imported


def test_module_has_no_import_path_to_plc_or_alarm_state():
    imported = _imported_names()
    offenders = [
        m for m in imported for frag in _FORBIDDEN_FRAGMENTS if frag in m.lower()
    ]
    assert offenders == []


def test_module_imports_only_stdlib_and_pure_signal_stats():
    tree = ast.parse(Path(asset_health.__file__).read_text(encoding="utf-8"))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules += [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules.append((node.module or "").split(".")[0])
    # ``shared`` supplies only the StrEnum compatibility shim.
    allowed = sys.stdlib_module_names | {"signal_stats", "shared"}
    assert all(m in allowed for m in modules if m), modules


def test_detectors_only_read_the_data_passed_in():
    # Read-only collaborators: immutable tuples work, nothing is mutated.
    cmd = tuple(_series([50.0] * 10))
    fb = tuple(_series([10.0] * 10))
    before = (cmd, fb)
    detect_command_feedback_mismatch(cmd, fb, tolerance=2.0, window=5.0)
    assert (cmd, fb) == before
