"""Contract tests for the measured-golfer reference bands and realism score.

The bands are the only thing standing between "this swing looks plausible" and
"this swing matches published measurements", so they are treated as data with
provenance: every band must carry a source and a resolvable link.

Closes #4778.
"""

from __future__ import annotations

import numpy as np
import pytest

from double_pendulum_golf.swing_objectives.reference_kinematics import (
    TOUR_DRIVER_BANDS,
    ObservableBand,
    score_against_reference,
)

_GOLF_LIKE = {
    "clubhead_speed_ms": 50.0,
    "hand_speed_ms": 7.0,
    "downswing_time_s": 0.27,
    "club_arm_rate_ratio": 3.2,
    "wrist_cock_impact_deg": 5.0,
    "release_fraction": 0.68,
}


def test_every_band_carries_a_source_and_a_link() -> None:
    """A band without provenance is an invented target, not a measurement."""
    for band in TOUR_DRIVER_BANDS:
        assert band.source.strip(), f"{band.key} has no source"
        assert band.url.startswith("https://"), f"{band.key} has no resolvable link"
        assert band.units.strip()
        assert band.label.strip()


def test_band_keys_are_unique() -> None:
    """Duplicate keys would silently drop an observable from the score."""
    keys = [band.key for band in TOUR_DRIVER_BANDS]
    assert len(keys) == len(set(keys))


def test_bands_reject_inverted_or_non_finite_intervals() -> None:
    """Contract: a band must be a usable interval."""
    with pytest.raises(ValueError, match="low must be below high"):
        ObservableBand("k", "L", "u", 5.0, 1.0, "s", "https://example.org")
    with pytest.raises(ValueError, match="finite"):
        ObservableBand("k", "L", "u", np.nan, 1.0, "s", "https://example.org")


def test_deviation_is_zero_inside_and_grows_outside() -> None:
    """Distance is measured in half-widths so units cannot skew the total."""
    band = ObservableBand("k", "L", "m/s", 4.0, 8.0, "s", "https://example.org")
    assert band.deviation(6.0) == 0.0
    assert band.deviation(4.0) == 0.0
    assert band.deviation(8.0) == 0.0
    assert band.deviation(10.0) == pytest.approx(1.0)  # one half-width above
    assert band.deviation(2.0) == pytest.approx(1.0)  # one half-width below


def test_a_golf_like_swing_scores_zero() -> None:
    """Values drawn from inside every published band must score perfectly."""
    score = score_against_reference(_GOLF_LIKE)
    assert score.total_deviation == pytest.approx(0.0)
    assert score.inside_count == len(TOUR_DRIVER_BANDS)
    assert score.missing == ()


def test_the_shipped_optimum_scores_badly_on_hand_speed() -> None:
    """The measured failure from epic #4775 must show up as the worst observable.

    These are the numbers the symmetric-clamp optimizer actually produced.
    """
    score = score_against_reference(
        {**_GOLF_LIKE, "hand_speed_ms": 0.36, "club_arm_rate_ratio": 59.2}
    )
    assert score.total_deviation > 0.0
    assert not score.inside["hand_speed_ms"]
    assert not score.inside["club_arm_rate_ratio"]
    # The rate ratio is off by far more half-widths than anything else.
    assert score.worst[0] == "club_arm_rate_ratio"


def test_missing_observables_are_reported_not_scored_as_zero() -> None:
    """A model that cannot produce an observable must not look perfect on it."""
    score = score_against_reference({"clubhead_speed_ms": 50.0})
    assert "hand_speed_ms" in score.missing
    assert "hand_speed_ms" not in score.deviations
    assert score.total_deviation == pytest.approx(0.0)
    assert score.inside_count == 1


def test_unknown_measurement_keys_are_ignored() -> None:
    """Extra diagnostics from a caller must not break scoring."""
    score = score_against_reference({**_GOLF_LIKE, "not_an_observable": 1.0})
    assert "not_an_observable" not in score.deviations
    assert score.total_deviation == pytest.approx(0.0)


def test_rejects_non_finite_measurements() -> None:
    """Contract: a NaN must fail loudly rather than score as inside the band."""
    with pytest.raises(ValueError, match="must be finite"):
        score_against_reference({"clubhead_speed_ms": np.nan})


def test_score_is_immutable() -> None:
    """Reversibility: a score handed to a report cannot be edited in place."""
    score = score_against_reference(_GOLF_LIKE)
    with pytest.raises((AttributeError, TypeError)):
        score.deviations = {}  # type: ignore[misc]
