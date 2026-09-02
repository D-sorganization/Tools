"""Canonical schema tests (ADR-0046 G1 step P5).

The port plan's *Two structural facts* note says UpstreamDrift's
``tests/unit/launch_monitor/test_importer.py`` covers four modules at once —
``importer``, ``profiles``, ``schema``, and the ``app-local`` ``project`` — and
therefore has to be split so tests travel with the module they exercise. This
file is the **schema half** of that split: the mapping *contract* the plan's
P5 row calls "``test_importer.py`` mapping cases", pinned directly on
:class:`~shared.python.launch_monitor.schema.ColumnMapping` rather than through
an importer that does not arrive until P9. The end-to-end mapping round-trips
that consume these mappings travel with
:mod:`shared.python.launch_monitor.importer` in that step.

The last case is the containment pin: the ``rate_of_closure`` counterpart the
port plan names is a bare name allow-list, this module is a definition table,
and the two are compared here without either package re-exporting the other.
"""

from __future__ import annotations

import pandas as pd
import pytest

from shared.python.launch_monitor.schema import (
    IDENTITY_COLUMNS,
    METRICS,
    ColumnMapping,
    ImportedSession,
    ImportManifest,
    ImportOptions,
    numeric_metric_columns,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _manifest() -> ImportManifest:
    return ImportManifest(
        source_path="shots.csv",
        file_sha256="0" * 64,
        profile_id="generic",
        vendor="Generic",
        imported_at="2026-09-02T00:00:00+00:00",
        row_count=1,
        source_columns=("speed",),
        metric_sources={"ball_speed": "speed"},
        source_units={"ball_speed": "mph"},
        unit_evidence={"ball_speed": "mapping"},
    )


def test_mapping_accepts_metric_identity_and_timestamp_targets() -> None:
    """The three legal target families, as the importer's mapping cases use them."""
    metric = ColumnMapping("Ball Speed (mph)", "ball_speed", "mph")
    identity = ColumnMapping("Player Name", "player")
    date_part = ColumnMapping("Shot Date", "date")
    time_part = ColumnMapping("Shot Time", "time")
    assert metric.target_column in METRICS
    assert identity.target_column in IDENTITY_COLUMNS
    assert (date_part.target_column, time_part.target_column) == ("date", "time")
    assert metric.multiplier == 1.0
    assert metric.measurement_status == "reported"


def test_mapping_refuses_an_unknown_target_column() -> None:
    """A mapping may only target a canonical metric, an identity, or date/time."""
    with pytest.raises(ValueError, match=r"Unknown target column: not_a_metric"):
        ColumnMapping("Whatever", "not_a_metric")


def test_mapping_refuses_an_empty_source_column() -> None:
    """A mapping with no source is not a mapping."""
    with pytest.raises(ValueError, match=r"source_column must be non-empty"):
        ColumnMapping("   ", "ball_speed")


@pytest.mark.parametrize("multiplier", [0.0, float("nan")])
def test_mapping_refuses_a_zero_or_non_finite_multiplier(multiplier: float) -> None:
    """A zero multiplier silently erases a column; NaN silently voids it."""
    with pytest.raises(ValueError, match=r"multiplier must be finite and non-zero"):
        ColumnMapping("Ball Speed", "ball_speed", "mph", multiplier)


def test_mapping_refuses_an_unknown_measurement_status() -> None:
    """Measurement status is a closed vocabulary: it feeds the audit trail."""
    with pytest.raises(ValueError, match=r"measurement_status must be"):
        ColumnMapping("Ball Speed", "ball_speed", "mph", 1.0, "guessed")


def test_metric_definitions_carry_units_and_derivation_edges() -> None:
    """Every metric declares a canonical unit; derived metrics name their inputs.

    This is the whole reason the canonical layer needs a definition table and
    not a name list: ``derived_from`` is what ``relationships`` flags and what
    ``modeling`` refuses as identity leakage.
    """
    assert METRICS["spin_rate"].canonical_unit == "rad/s"
    assert METRICS["carry_distance"].canonical_unit == "m"
    assert METRICS["attack_angle"].canonical_unit == "rad"
    assert METRICS["smash_factor"].derived_from == ("ball_speed", "club_speed")
    assert METRICS["face_to_path"].derived_from == ("face_angle", "club_path")
    assert all(
        definition.canonical_unit in {"m/s", "m", "rad", "rad/s", "s", "1"}
        for definition in METRICS.values()
    )
    assert all(name == definition.name for name, definition in METRICS.items())


def test_numeric_metric_columns_selects_only_canonical_numeric_columns() -> None:
    """Non-canonical and non-numeric columns are not analysis inputs."""
    frame = pd.DataFrame(
        {
            "ball_speed": [60.0, 61.0],
            "club_speed": ["fast", "faster"],
            "not_a_metric": [1.0, 2.0],
            "shot_id": ["a", "b"],
        }
    )
    assert numeric_metric_columns(frame) == ["ball_speed"]


def test_imported_session_refuses_an_empty_or_incomplete_frame() -> None:
    """A session must be identifiable, non-empty, and carry its identity columns."""
    complete = pd.DataFrame(
        {
            "shot_id": ["s0"],
            "session_id": ["generic-0"],
            "source_row": [2],
            "monitor_vendor": ["Generic"],
        }
    )
    session = ImportedSession("generic-0", "Shots", complete, _manifest())
    assert session.metadata == {}

    with pytest.raises(ValueError, match=r"session_id must be non-empty"):
        ImportedSession("", "Shots", complete, _manifest())
    with pytest.raises(ValueError, match=r"at least one shot"):
        ImportedSession("generic-0", "Shots", complete.iloc[0:0], _manifest())
    with pytest.raises(ValueError, match=r"shots missing required columns"):
        ImportedSession(
            "generic-0",
            "Shots",
            complete.drop(columns=["monitor_vendor"]),
            _manifest(),
        )


def test_import_options_default_to_detection_with_no_overrides() -> None:
    """An unconfigured import asks the profile layer, not the caller."""
    options = ImportOptions()
    assert options.profile_id is None
    assert options.mappings == ()
    assert options.tags == ()


def test_canonical_metrics_are_a_strict_superset_of_the_tools_wire_allow_list() -> None:
    """Containment pin: the ``rate_of_closure`` counterpart is a name list.

    The port plan records the Tools counterpart of this module as
    ``launch_monitor_canonical_v2.CANONICAL_DATASET_METRICS`` — 17 metric
    *names* validating a wire payload, with no unit, label, category, display
    unit, or derivation record. Every one of them has a definition here and the
    reverse does not hold, which is why this module is the port and not a
    duplicate. The relationship is asserted here rather than expressed as a
    re-export: an alias between the two packages is exactly the silent-merge
    hazard the separate package exists to prevent.
    """
    from rate_of_closure import launch_monitor_canonical_v2 as wire

    assert set(wire.CANONICAL_DATASET_METRICS) <= set(METRICS)
    assert set(METRICS) - set(wire.CANONICAL_DATASET_METRICS)
    assert not hasattr(wire, "METRICS")
    assert not hasattr(wire, "MetricDefinition")
