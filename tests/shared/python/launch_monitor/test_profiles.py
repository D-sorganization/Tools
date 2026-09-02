"""Canonical import-profile tests (ADR-0046 G1 step P9, profiles half).

UpstreamDrift's ``tests/unit/launch_monitor/test_importer.py`` covers four
modules — ``importer``, ``profiles``, ``schema``, and the ``app-local``
``project`` — so the port plan's structural note requires it to be split. This
file is the **profiles half**: its two detection cases travel here verbatim,
the import round-trips travel to ``test_importer.py``, the mapping contract
travelled to ``test_schema.py`` in P5, and the project round-trip does not
travel at all because ``project.py`` stays in UpstreamDrift.

The added cases pin the refusal and the "I do not know" outcome that make
detection safe to trust, since the detected profile also supplies the unit
defaults the importer converts with.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from shared.python.launch_monitor.profiles import (
    PROFILES,
    detect_profile,
    normalize_header,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.mark.parametrize(
    ("filename", "profile"),
    [
        ("trackman.csv", "trackman"),
        ("foresight.csv", "foresight"),
        ("flightscope.csv", "flightscope"),
        ("garmin.csv", "garmin"),
        ("skytrak.csv", "skytrak"),
        ("uneekor.csv", "uneekor"),
    ],
)
def test_detects_vendor_profiles(
    filename: str, profile: str, fixtures_dir: Path
) -> None:
    """Ported verbatim from UpstreamDrift's ``test_importer.py``."""
    headers = pd.read_csv(fixtures_dir / filename, nrows=0).columns.tolist()
    result = detect_profile(headers)
    assert result.profile_id == profile
    assert result.confidence >= 0.5


@pytest.mark.parametrize(
    ("headers", "profile"),
    [
        (
            [
                "Club Speed (mph)",
                "Ball Speed (mph)",
                "Face to Path (deg)",
                "Carry Distance (yd)",
            ],
            "full_swing",
        ),
        (
            [
                "Smash Factor",
                "Launch Direction (deg)",
                "Shot Type",
                "Carry Distance (yd)",
            ],
            "rapsodo",
        ),
    ],
)
def test_detects_additional_common_profiles(headers: list[str], profile: str) -> None:
    """Ported verbatim from UpstreamDrift's ``test_importer.py``."""
    assert detect_profile(headers).profile_id == profile


def test_unrecognised_headers_fall_back_to_generic_at_zero_confidence() -> None:
    """Below half a profile's signatures the answer is "I do not know".

    That matters more than it looks: the detected profile also supplies the
    unit defaults the importer converts with, so guessing a vendor would
    silently pick a unit system.
    """
    result = detect_profile(["alpha", "beta", "gamma"])
    assert result.profile_id == "generic"
    assert result.confidence == 0.0
    assert result.matched_signatures == ()
    assert result.alternatives


def test_detection_refuses_an_empty_header_list() -> None:
    """No headers is not a fingerprint."""
    with pytest.raises(ValueError, match=r"headers must contain at least one column"):
        detect_profile([])


def test_normalisation_strips_units_and_splits_camel_case() -> None:
    """Alias matching compares meaning, not spelling."""
    assert normalize_header("Club Speed (mph)") == "club speed"
    assert normalize_header("ClubHeadSpeed") == "club head speed"
    assert normalize_header("Carry Distance yd") == "carry distance"
    assert normalize_header("Spin Rate [rpm]") == "spin rate"


def test_mappings_claim_each_source_column_at_most_once() -> None:
    """An ambiguous header cannot be mapped to two canonical targets."""
    headers = [
        "Shot",
        "Club Speed (mph)",
        "Ball Speed (mph)",
        "Carry (yd)",
        "Total (yd)",
    ]
    mappings = PROFILES["trackman"].mappings_for(headers)
    sources = [mapping.source_column for mapping in mappings]
    targets = [mapping.target_column for mapping in mappings]
    assert len(sources) == len(set(sources))
    assert len(targets) == len(set(targets))
    assert set(sources) <= set(headers)


def test_every_profile_supplies_a_display_unit_for_every_metric() -> None:
    """A profile that cannot name a unit for a metric cannot convert it."""
    from shared.python.launch_monitor.schema import METRICS

    for profile in PROFILES.values():
        assert set(profile.default_units) == set(METRICS)
        assert all(unit for unit in profile.default_units.values())
