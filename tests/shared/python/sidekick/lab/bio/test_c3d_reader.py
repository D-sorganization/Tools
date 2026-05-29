"""Unit tests for the Qt-free / file-free parts of ``lab.bio.c3d_reader``.

The full ``C3DDataReader`` needs a real C3D capture file, but its value objects
(``C3DEvent``, ``C3DMetadata``) and static helpers (CSV sanitization, unit
scaling, export-path validation) are pure logic and tested directly here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pandas")

from sidekick.lab.bio.c3d_reader import (  # noqa: E402
    C3DDataReader,
    C3DEvent,
    C3DMetadata,
)

# ---------------------------------------------------------------------------
# C3DEvent
# ---------------------------------------------------------------------------


def test_event_valid() -> None:
    event = C3DEvent(label="heel_strike", time=1.25)
    assert event.label == "heel_strike"
    assert event.time == 1.25


def test_event_empty_label_raises() -> None:
    with pytest.raises(ValueError, match="label cannot be empty"):
        C3DEvent(label="", time=0.0)


def test_event_allows_negative_time() -> None:
    # Pre-trigger events have negative time per the C3D spec.
    assert C3DEvent(label="pre", time=-0.5).time == -0.5


# ---------------------------------------------------------------------------
# C3DMetadata
# ---------------------------------------------------------------------------


def _metadata(**overrides) -> C3DMetadata:
    kwargs = {
        "marker_labels": ["L_HEEL", "R_HEEL"],
        "frame_count": 200,
        "frame_rate": 100.0,
        "units": "mm",
        "analog_labels": ["FX", "FY"],
        "analog_units": ["N", "N"],
        "analog_rate": 1000.0,
        "events": [],
    }
    kwargs.update(overrides)
    return C3DMetadata(**kwargs)


def test_metadata_properties() -> None:
    meta = _metadata()
    assert meta.marker_count == 2
    assert meta.analog_count == 2
    assert meta.duration == pytest.approx(2.0)  # 200 frames / 100 Hz


def test_metadata_duration_zero_rate() -> None:
    assert _metadata(frame_rate=0.0).duration == 0.0


def test_metadata_negative_frame_count_raises() -> None:
    with pytest.raises(ValueError, match="Frame count cannot be negative"):
        _metadata(frame_count=-1)


def test_metadata_negative_frame_rate_raises() -> None:
    with pytest.raises(ValueError, match="Frame rate cannot be negative"):
        _metadata(frame_rate=-1.0)


def test_metadata_negative_analog_rate_raises() -> None:
    with pytest.raises(ValueError, match="Analog rate cannot be negative"):
        _metadata(analog_rate=-1.0)


def test_metadata_analog_label_unit_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        _metadata(analog_labels=["FX", "FY"], analog_units=["N"])


# ---------------------------------------------------------------------------
# static helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prefix", ["=", "+", "-", "@"])
def test_sanitize_for_csv_prefixes_formula_chars(prefix: str) -> None:
    out = C3DDataReader._sanitize_for_csv(f"{prefix}cmd")
    assert out == f"'{prefix}cmd"


def test_sanitize_for_csv_passes_through_plain_values() -> None:
    assert C3DDataReader._sanitize_for_csv("safe") == "safe"
    assert C3DDataReader._sanitize_for_csv(42) == 42


@pytest.mark.parametrize(
    ("current", "target", "expected"),
    [
        ("mm", None, 1.0),
        ("m", "m", 1.0),
        ("mm", "m", 0.001),
        ("m", "mm", 1000.0),
        ("cm", "m", 0.01),
        ("in", "m", 0.0254),
    ],
)
def test_unit_scale(current: str, target: str | None, expected: float) -> None:
    assert C3DDataReader._unit_scale(current, target) == pytest.approx(expected)


def test_unit_scale_unsupported_source_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported source unit"):
        C3DDataReader._unit_scale("furlong", "m")


def test_unit_scale_unsupported_target_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported target unit"):
        C3DDataReader._unit_scale("m", "furlong")


def test_validate_export_path_allows_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("C3D_ALLOW_ANY_EXPORT_PATH", "1")
    # Should not raise even though tmp_path is outside the project root.
    C3DDataReader._validate_export_path(tmp_path / "out.csv")


def test_validate_export_path_rejects_outside_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("C3D_ALLOW_ANY_EXPORT_PATH", raising=False)
    with pytest.raises(ValueError, match="outside project root"):
        C3DDataReader._validate_export_path(tmp_path / "out.csv")


def test_validate_export_path_rejects_bad_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("C3D_ALLOW_ANY_EXPORT_PATH", raising=False)
    # A path under cwd but with an unsupported extension.
    target = Path.cwd() / "scratch_export.exe"
    with pytest.raises(ValueError, match="Unsupported export format"):
        C3DDataReader._validate_export_path(target)
