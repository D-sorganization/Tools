"""Confidence ellipsoid and selectable quiet-zone metric contracts."""

from __future__ import annotations

import math

import numpy as np
import pytest

from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.variation import (
    ELLIPSOID_VOLUME,
    LARGEST_PRINCIPAL_SIGMA,
    RMS_RADIUS,
    LowVariabilityMetricCriteria,
    PositionDispersion,
    build_confidence_ellipsoids,
    build_dispersion_metric_series,
    find_ranked_low_variability_intervals,
)


def _dispersion() -> PositionDispersion:
    times = np.array([0.0, 1.0, 2.0, 3.0])
    eigenvalues = np.array(
        [
            [[4.0, 1.0, 0.25], [1.0, 0.25, 0.04]],
            [[1.0, 0.25, 0.0], [1.0, 0.25, 0.04]],
            [[np.nan, np.nan, np.nan], [1.0, 0.25, 0.04]],
            [[0.25, 0.04, 0.01], [1.0, 0.25, 0.04]],
        ]
    )
    covariance = np.full((4, 2, 3, 3), np.nan)
    axes = np.full((4, 2, 3, 3), np.nan)
    for sample_index in (0, 1, 3):
        covariance[sample_index, 0] = np.diag(eigenvalues[sample_index, 0])
        axes[sample_index, 0] = np.eye(3)
    for sample_index in range(4):
        covariance[sample_index, 1] = np.diag(eigenvalues[sample_index, 1])
        axes[sample_index, 1] = np.eye(3)
    return PositionDispersion(
        sample_times_s=times,
        coordinate_frame="swing.world",
        point_ids=("clubhead", "hands"),
        count=np.array([[8, 8], [3, 8], [1, 8], [8, 8]]),
        mean_positions_m=np.zeros((4, 2, 3)),
        covariance_m2=covariance,
        eigenvalues_m2=eigenvalues,
        principal_axes=axes,
        rms_radius_m=np.array([[0.10, 0.10], [0.10, 0.10], [0.10, 1.00], [0.20, 0.30]]),
    )


def test_confidence_ellipsoid_uses_exact_three_dimensional_chi_square_scale() -> None:
    result = build_confidence_ellipsoids(_dispersion(), "clubhead", 0.95)

    assert result.confidence_level == 0.95
    assert result.interpretation == "gaussian-position-content-region"
    assert result.degrees_of_freedom == 3
    assert result.chi_square_quantile == pytest.approx(7.814727903251179)
    assert result.radius_scale == pytest.approx(math.sqrt(7.814727903251179))
    np.testing.assert_allclose(
        result.semi_axis_lengths_m[0],
        result.radius_scale * np.array([2.0, 1.0, 0.5]),
    )
    expected_volume = 4.0 * math.pi / 3.0 * np.prod(result.semi_axis_lengths_m[0])
    assert result.volume_m3[0] == pytest.approx(expected_volume)
    assert result.adequacy == (
        "estimable",
        "rank-deficient",
        "insufficient-samples",
        "estimable",
    )
    assert result.minimum_samples_for_full_rank == 4
    assert result.semi_axis_lengths_m.flags.writeable is False


@pytest.mark.parametrize("confidence_level", [0.0, 1.0, -0.1, np.nan, True])
def test_confidence_ellipsoid_rejects_invalid_confidence_levels(
    confidence_level: object,
) -> None:
    with pytest.raises(ContractViolationError, match="confidence_level"):
        build_confidence_ellipsoids(
            _dispersion(),
            "clubhead",
            confidence_level,  # type: ignore[arg-type]
        )


def test_selectable_metric_series_declares_units_and_sample_adequacy() -> None:
    dispersion = _dispersion()

    rms = build_dispersion_metric_series(dispersion, "clubhead", RMS_RADIUS)
    sigma = build_dispersion_metric_series(
        dispersion, "clubhead", LARGEST_PRINCIPAL_SIGMA
    )
    volume = build_dispersion_metric_series(
        dispersion,
        "clubhead",
        ELLIPSOID_VOLUME,
        confidence_level=0.95,
    )

    assert rms.unit == "m"
    np.testing.assert_allclose(rms.values, [0.1, 0.1, 0.1, 0.2])
    assert sigma.unit == "m"
    np.testing.assert_allclose(sigma.values[[0, 1, 3]], [2.0, 1.0, 0.5])
    assert np.isnan(sigma.values[2])
    assert volume.unit == "m^3"
    assert np.isfinite(volume.values[[0, 3]]).all()
    assert np.isnan(volume.values[[1, 2]]).all()
    assert volume.adequacy == (
        "estimable",
        "rank-deficient",
        "insufficient-samples",
        "estimable",
    )


def test_ranked_intervals_use_declared_metric_score_and_dense_ties() -> None:
    criteria = LowVariabilityMetricCriteria(
        metric=RMS_RADIUS,
        max_value=0.2,
        min_samples=2,
    )

    intervals = find_ranked_low_variability_intervals(_dispersion(), criteria)

    assert [
        (item.point_id, item.start_index, item.end_index) for item in intervals
    ] == [
        ("clubhead", 0, 1),
        ("hands", 0, 1),
    ]
    assert intervals[0].score == pytest.approx(0.5)
    assert intervals[1].score == pytest.approx(0.5)
    assert [item.rank for item in intervals] == [1, 1]
    assert all(item.metric == RMS_RADIUS and item.unit == "m" for item in intervals)


def test_rank_order_is_score_then_stable_point_and_time_with_explicit_ties() -> None:
    criteria = LowVariabilityMetricCriteria(
        metric=RMS_RADIUS,
        max_value=0.3,
        min_samples=1,
        point_ids=("hands", "clubhead"),
    )

    first = find_ranked_low_variability_intervals(_dispersion(), criteria)
    second = find_ranked_low_variability_intervals(_dispersion(), criteria)

    assert first == second
    assert [(item.point_id, item.start_index, item.rank) for item in first] == [
        ("clubhead", 0, 1),
        ("hands", 0, 1),
        ("clubhead", 3, 2),
        ("hands", 3, 3),
    ]
    assert first[0].score == first[1].score
    assert first[2].score == pytest.approx(2.0 / 3.0)
    assert first[3].score == pytest.approx(1.0)


def test_volume_metric_excludes_non_estimable_samples_from_quiet_intervals() -> None:
    metric = build_dispersion_metric_series(
        _dispersion(), "clubhead", ELLIPSOID_VOLUME, confidence_level=0.95
    )
    criteria = LowVariabilityMetricCriteria(
        metric=ELLIPSOID_VOLUME,
        max_value=float((metric.values[0] + metric.values[3]) / 2.0),
        confidence_level=0.95,
        point_ids=("clubhead",),
    )

    intervals = find_ranked_low_variability_intervals(_dispersion(), criteria)

    assert [(item.start_index, item.end_index) for item in intervals] == [(3, 3)]
    assert intervals[0].confidence_level == 0.95


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"metric": "unknown", "max_value": 1.0}, "metric"),
        ({"metric": RMS_RADIUS, "max_value": 0.0}, "max_value"),
        ({"metric": RMS_RADIUS, "max_value": 1.0, "min_samples": True}, "min_samples"),
        (
            {"metric": RMS_RADIUS, "max_value": 1.0, "confidence_level": 1.0},
            "confidence_level",
        ),
    ],
)
def test_metric_criteria_fail_closed(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ContractViolationError, match=message):
        LowVariabilityMetricCriteria(**kwargs)  # type: ignore[arg-type]
