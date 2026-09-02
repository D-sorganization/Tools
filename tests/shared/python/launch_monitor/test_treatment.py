"""Canonical data-treatment tests (ADR-0046 G1 step P6).

The first two cases are the two treatment cases from UpstreamDrift's
``tests/unit/launch_monitor/test_analysis.py`` —
``test_treatment_flags_duplicates_missing_and_robust_outliers`` and
``test_treatment_filters_and_labels_derived_metrics`` — travelling verbatim
with the module they exercise. The remaining cases pin the structural refusals
and the audit guarantees the module's docstring documents, which
``CLAUDE.md``'s design-by-contract rule asks of every ported public entry
point.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from shared.python.launch_monitor.treatment import (
    FilterRule,
    TreatmentConfig,
    apply_treatment,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_treatment_flags_duplicates_missing_and_robust_outliers(
    shots: Callable[..., pd.DataFrame],
) -> None:
    """Ported verbatim from UpstreamDrift's ``test_analysis.py``."""
    frame = shots(20)
    frame.loc[2, "ball_speed"] = np.nan
    frame.loc[3, "club_speed"] = 999.0
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    result = apply_treatment(
        frame,
        TreatmentConfig(
            required_metrics=("club_speed", "ball_speed"),
            duplicate_columns=("shot_id",),
            outlier_metrics=("club_speed",),
            robust_z_threshold=3.5,
            exclude_flagged=True,
        ),
    )
    assert {"missing_required", "duplicate", "robust_outlier"} <= set(
        result.flags["flag_type"]
    )
    assert len(result.data) == len(frame) - 3
    assert len(result.audit_log) >= 3


def test_treatment_filters_and_labels_derived_metrics(
    shots: Callable[..., pd.DataFrame],
) -> None:
    """Ported verbatim from UpstreamDrift's ``test_analysis.py``."""
    frame = shots(20).drop(columns=["smash_factor"])
    result = apply_treatment(
        frame,
        TreatmentConfig(
            filters=(FilterRule("monitor_vendor", "eq", "TrackMan"),),
        ),
    )
    assert set(result.data["monitor_vendor"]) == {"TrackMan"}
    assert result.data["smash_factor"].notna().all()
    assert set(result.data["status::smash_factor"]) == {"derived"}
    assert any(item["action"] == "filter" for item in result.audit_log)
    assert any(item["action"] == "derive_metric" for item in result.audit_log)


def test_flagging_is_non_destructive_unless_exclusion_is_requested(
    shots: Callable[..., pd.DataFrame],
) -> None:
    """Flags are recorded; rows survive until ``exclude_flagged`` says otherwise."""
    frame = shots(20)
    frame.loc[2, "ball_speed"] = np.nan
    config = TreatmentConfig(required_metrics=("club_speed", "ball_speed"))
    kept = apply_treatment(frame, config)
    assert len(kept.data) == 20
    assert list(kept.flags["flag_type"]) == ["missing_required"]

    dropped = apply_treatment(
        frame,
        TreatmentConfig(
            required_metrics=("club_speed", "ball_speed"), exclude_flagged=True
        ),
    )
    assert len(dropped.data) == 19
    exclusions = [
        item for item in dropped.audit_log if item["action"] == "exclude_flagged"
    ]
    assert exclusions == [{"action": "exclude_flagged", "row_count": 1}]


def test_filter_audit_records_rows_before_and_after(
    shots: Callable[..., pd.DataFrame],
) -> None:
    """A subset must be reconstructible from the log, not merely asserted."""
    result = apply_treatment(
        shots(20),
        TreatmentConfig(filters=(FilterRule("monitor_vendor", "eq", "TrackMan"),)),
    )
    record = next(item for item in result.audit_log if item["action"] == "filter")
    assert record["rows_before"] == 20
    assert record["rows_after"] == len(result.data)
    assert record["column"] == "monitor_vendor"
    assert record["operator"] == "eq"


def test_derivation_stamps_status_and_never_overwrites_a_measured_value(
    shots: Callable[..., pd.DataFrame],
) -> None:
    """Derivation fills gaps only; a reported value is left exactly as reported."""
    frame = shots(20)
    frame.loc[0, "smash_factor"] = np.nan
    original = float(frame.loc[1, "smash_factor"])
    result = apply_treatment(frame, TreatmentConfig())
    assert result.data.loc[0, "status::smash_factor"] == "derived"
    assert result.data.loc[1, "status::smash_factor"] == "unknown"
    assert result.data.loc[1, "smash_factor"] == pytest.approx(original)
    action = next(
        item for item in result.audit_log if item["action"] == "derive_metric"
    )
    assert action["metric"] == "smash_factor"
    assert action["inputs"] == ["ball_speed", "club_speed"]
    assert action["row_count"] == 1


def test_treatment_refuses_an_empty_frame() -> None:
    """There is nothing to treat, and an empty result would look like a clean one."""
    with pytest.raises(ValueError, match=r"at least one shot"):
        apply_treatment(pd.DataFrame(), TreatmentConfig())


def test_treatment_refuses_a_missing_required_or_outlier_metric(
    shots: Callable[..., pd.DataFrame],
) -> None:
    """A malformed *request* raises; only a malformed *row* is flagged."""
    frame = shots(20)
    with pytest.raises(ValueError, match=r"Required metrics not present"):
        apply_treatment(frame, TreatmentConfig(required_metrics=("not_a_metric",)))
    with pytest.raises(ValueError, match=r"Outlier metric not present"):
        apply_treatment(frame, TreatmentConfig(outlier_metrics=("not_a_metric",)))


def test_filter_rule_refuses_an_unsupported_operator_or_empty_column() -> None:
    """The operator vocabulary is closed, and a rule needs a column."""
    with pytest.raises(ValueError, match=r"Unsupported filter operator: like"):
        FilterRule("club_speed", "like", "Trackman")
    with pytest.raises(ValueError, match=r"filter column must be non-empty"):
        FilterRule("  ", "eq", "TrackMan")


def test_filter_refuses_an_absent_column_or_a_non_numeric_comparison(
    shots: Callable[..., pd.DataFrame],
) -> None:
    """Numeric operators need a numeric bound; a filter cannot invent a column."""
    frame = shots(20)
    with pytest.raises(ValueError, match=r"Filter column not present: nope"):
        apply_treatment(frame, TreatmentConfig(filters=(FilterRule("nope", "eq", 1),)))
    with pytest.raises(ValueError, match=r"must be numeric"):
        apply_treatment(
            frame,
            TreatmentConfig(filters=(FilterRule("club_speed", "gt", "fast"),)),
        )


def test_treatment_config_refuses_a_non_positive_threshold() -> None:
    """A zero or non-finite robust-z threshold flags everything or nothing."""
    for threshold in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match=r"robust_z_threshold must be"):
            TreatmentConfig(robust_z_threshold=threshold)
