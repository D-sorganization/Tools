"""Canonical importer tests (ADR-0046 G1 step P9, importer half).

The **importer half** of the split described in ``test_profiles.py``: the four
import round-trips from UpstreamDrift's
``tests/unit/launch_monitor/test_importer.py`` travel here verbatim. The added
cases pin the refusals, the provenance guarantees, and the exclude-and-audit
posture the module's docstring documents — an unconvertible unit is a manifest
warning, not an aborted import.

UpstreamDrift's seventh case, ``test_project_aggregates_sessions_and_round_trips``,
does **not** travel: the port plan classifies ``project.py`` as ``app-local``
on evidence (its ``LaunchMonitorProject`` persists shot rows, while the
identically named ``rate_of_closure`` class is deliberately row-free), so it
stays in UpstreamDrift.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shared.python.launch_monitor.importer import import_session
from shared.python.launch_monitor.schema import ColumnMapping, ImportOptions

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_trackman_import_converts_units_and_preserves_provenance(
    fixtures_dir: Path,
) -> None:
    """Ported verbatim from UpstreamDrift's ``test_importer.py``."""
    session = import_session(fixtures_dir / "trackman.csv")
    shots = session.shots
    assert session.manifest.profile_id == "trackman"
    assert session.manifest.file_sha256
    assert session.manifest.row_count == 2
    assert shots.loc[0, "club_speed"] == pytest.approx(88 * 0.44704)
    assert shots.loc[0, "carry_distance"] == pytest.approx(170 * 0.9144)
    assert shots.loc[0, "attack_angle"] == pytest.approx(np.deg2rad(-3.5))
    assert shots.loc[0, "spin_rate"] == pytest.approx(6100 * 2 * np.pi / 60)
    assert shots.loc[0, "source_row"] == 2
    assert "source::Club Speed (mph)" in shots.columns
    assert shots.loc[0, "source::Club Speed (mph)"] == 88
    assert session.manifest.metric_sources["club_speed"] == "Club Speed (mph)"
    assert set(shots["status::club_speed"]) == {"reported"}


def test_generic_json_import_uses_explicit_mapping(tmp_path: Path) -> None:
    """Ported verbatim from UpstreamDrift's ``test_importer.py``."""
    source = tmp_path / "shots.json"
    source.write_text(
        json.dumps([{"speed": 100.0, "launch": 12.0, "note": "fit"}]),
        encoding="utf-8",
    )
    options = ImportOptions(
        profile_id="generic",
        mappings=(
            ColumnMapping("speed", "ball_speed", "mph"),
            ColumnMapping("launch", "launch_angle", "deg"),
        ),
        session_name="Mapped JSON",
    )
    session = import_session(source, options)
    assert session.shots.loc[0, "ball_speed"] == pytest.approx(44.704)
    assert session.shots.loc[0, "launch_angle"] == pytest.approx(np.deg2rad(12))
    assert session.shots.loc[0, "source::note"] == "fit"


def test_gspro_open_connect_nested_json_is_flattened(tmp_path: Path) -> None:
    """Ported verbatim from UpstreamDrift's ``test_importer.py``."""
    source = tmp_path / "gspro.json"
    source.write_text(
        json.dumps(
            {
                "DeviceID": "Test Device",
                "Units": "Yards",
                "BallData": {
                    "Speed": 150.0,
                    "HLA": 1.0,
                    "VLA": 12.0,
                    "TotalSpin": 2500.0,
                    "BackSpin": 2450.0,
                },
                "ClubData": {"Speed": 101.0, "AngleOfAttack": -1.5},
            }
        ),
        encoding="utf-8",
    )
    session = import_session(source)
    assert session.manifest.profile_id == "gspro"
    assert session.shots.loc[0, "ball_speed"] == pytest.approx(150 * 0.44704)
    assert session.shots.loc[0, "club_speed"] == pytest.approx(101 * 0.44704)
    assert "source::BallData.Speed" in session.shots


@pytest.mark.parametrize("suffix", [".csv", ".tsv", ".xlsx"])
def test_generic_tabular_formats_use_same_mapping(tmp_path: Path, suffix: str) -> None:
    """Ported verbatim from UpstreamDrift's ``test_importer.py``."""
    pytest.importorskip("openpyxl")
    source = tmp_path / f"shots{suffix}"
    frame = pd.DataFrame({"speed": [90.0, 91.0], "distance": [150.0, 152.0]})
    if suffix == ".xlsx":
        frame.to_excel(source, index=False)
    else:
        frame.to_csv(source, index=False, sep="\t" if suffix == ".tsv" else ",")
    session = import_session(
        source,
        ImportOptions(
            profile_id="generic",
            mappings=(
                ColumnMapping("speed", "club_speed", "mph"),
                ColumnMapping("distance", "carry_distance", "yd"),
            ),
        ),
    )
    assert len(session.shots) == 2
    assert session.shots.loc[0, "club_speed"] == pytest.approx(90 * 0.44704)


@pytest.mark.parametrize("suffix", [".csv", ".tsv"])
def test_delimited_text_formats_import_without_the_excel_extra(
    tmp_path: Path, suffix: str
) -> None:
    """The CSV and TSV halves of the case above, free of the ``openpyxl`` skip.

    UpstreamDrift's parametrisation guards all three formats behind one
    ``importorskip("openpyxl")``, so in an environment without the Excel extra
    the delimited-text readers go untested. That parametrisation travels
    verbatim above; this case keeps the coverage.
    """
    source = tmp_path / f"shots{suffix}"
    frame = pd.DataFrame({"speed": [90.0, 91.0], "distance": [150.0, 152.0]})
    frame.to_csv(source, index=False, sep="\t" if suffix == ".tsv" else ",")
    session = import_session(
        source,
        ImportOptions(
            profile_id="generic",
            mappings=(
                ColumnMapping("speed", "club_speed", "mph"),
                ColumnMapping("distance", "carry_distance", "yd"),
            ),
        ),
    )
    assert len(session.shots) == 2
    assert session.shots.loc[0, "club_speed"] == pytest.approx(90 * 0.44704)
    assert session.shots.loc[1, "carry_distance"] == pytest.approx(152 * 0.9144)


def test_manifest_records_how_each_unit_was_established(fixtures_dir: Path) -> None:
    """Unit evidence is ranked mapping > header > profile default, and recorded."""
    session = import_session(fixtures_dir / "trackman.csv")
    assert session.manifest.unit_evidence["club_speed"] == "header"
    assert session.manifest.source_units["club_speed"] == "mph"

    override = import_session(
        fixtures_dir / "trackman.csv",
        ImportOptions(
            mappings=(ColumnMapping("Club Speed (mph)", "club_speed", "m/s"),)
        ),
    )
    assert override.manifest.unit_evidence["club_speed"] == "mapping"
    assert override.manifest.source_units["club_speed"] == "m/s"
    assert override.shots.loc[0, "club_speed"] == pytest.approx(88.0)


def test_every_source_column_is_retained_verbatim(fixtures_dir: Path) -> None:
    """Conversion stays reversible because nothing is discarded on the way in."""
    session = import_session(fixtures_dir / "garmin.csv")
    raw = pd.read_csv(fixtures_dir / "garmin.csv")
    for column in raw.columns:
        assert f"source::{column}" in session.shots.columns
    assert tuple(raw.columns) == session.manifest.source_columns
    assert session.shots["source_row"].tolist() == [2, 3]
    assert session.shots["shot_id"].tolist() == ["1", "2"]


def test_unconvertible_unit_is_a_warning_not_an_aborted_import(
    tmp_path: Path,
) -> None:
    """Exclude-and-audit: one bad unit costs one metric, not the whole session."""
    source = tmp_path / "shots.csv"
    source.write_text("speed,distance\n90,150\n91,152\n", encoding="utf-8")
    session = import_session(
        source,
        ImportOptions(
            profile_id="generic",
            mappings=(
                ColumnMapping("speed", "club_speed", "furlongs/fortnight"),
                ColumnMapping("distance", "carry_distance", "yd"),
            ),
        ),
    )
    assert "club_speed" not in session.shots.columns
    assert "club_speed" not in session.manifest.metric_sources
    assert any("club_speed" in warning for warning in session.manifest.warnings)
    assert session.shots.loc[0, "carry_distance"] == pytest.approx(150 * 0.9144)


def test_a_mapping_naming_an_absent_column_is_warned_not_raised(
    tmp_path: Path,
) -> None:
    """The same posture for a mapping the file does not satisfy."""
    source = tmp_path / "shots.csv"
    source.write_text("speed\n90\n91\n", encoding="utf-8")
    session = import_session(
        source,
        ImportOptions(
            profile_id="generic",
            mappings=(
                ColumnMapping("speed", "club_speed", "mph"),
                ColumnMapping("absent", "carry_distance", "yd"),
            ),
        ),
    )
    assert any(
        "Mapped source column not found: absent" in warning
        for warning in session.manifest.warnings
    )
    assert "carry_distance" not in session.shots.columns


def test_importer_refuses_absent_empty_and_unsupported_sources(
    tmp_path: Path,
) -> None:
    """A source that cannot yield shots is refused by name."""
    with pytest.raises(ValueError, match=r"source does not exist"):
        import_session(tmp_path / "nope.csv")

    empty = tmp_path / "empty.csv"
    empty.write_text("speed,distance\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"contains no rows"):
        import_session(empty)

    unsupported = tmp_path / "shots.parquet"
    unsupported.write_bytes(b"not a parquet file")
    with pytest.raises(ValueError, match=r"Unsupported launch-monitor file type"):
        import_session(unsupported)


def test_importer_refuses_an_unknown_profile_id(fixtures_dir: Path) -> None:
    """An explicit profile must exist, or its unit defaults would be invented."""
    with pytest.raises(ValueError, match=r"Unknown import profile: not_a_vendor"):
        import_session(
            fixtures_dir / "trackman.csv", ImportOptions(profile_id="not_a_vendor")
        )


def test_json_scalar_payload_is_refused(tmp_path: Path) -> None:
    """A JSON number is neither a shot record nor a list of them."""
    source = tmp_path / "shots.json"
    source.write_text("42", encoding="utf-8")
    with pytest.raises(ValueError, match=r"object or list of shot records"):
        import_session(source)
