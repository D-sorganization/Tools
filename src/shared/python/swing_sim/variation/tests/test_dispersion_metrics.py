"""Confidence ellipsoid and selectable quiet-zone metric contracts."""

from __future__ import annotations

import math

import numpy as np
import pytest

from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.variation import (
    ELLIPSOID_VOLUME,
    LARGEST_PRINCIPAL_SIGMA,
    MIN_CONFIDENCE_LEVEL,
    RMS_RADIUS,
    LowVariabilityMetricCriteria,
    PositionDispersion,
    build_confidence_ellipsoids,
    build_dispersion_metric_series,
    find_ranked_low_variability_intervals,
)
from shared.python.swing_sim.variation.dispersion_metric_types import (
    ConfidenceEllipsoidSeries,
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


def _single_sample_dispersion(
    *,
    eigenvalues: np.ndarray | None = None,
    mean_positions_m: np.ndarray | None = None,
    covariance_m2: np.ndarray | None = None,
    principal_axes: np.ndarray | None = None,
) -> PositionDispersion:
    values = (
        np.array([4.0, 1.0, 0.25])
        if eigenvalues is None
        else np.asarray(eigenvalues, dtype=float)
    )
    return PositionDispersion(
        sample_times_s=np.array([0.0]),
        coordinate_frame="swing.world",
        point_ids=("clubhead",),
        count=np.array([[8]]),
        mean_positions_m=(
            np.zeros((1, 1, 3)) if mean_positions_m is None else mean_positions_m
        ),
        covariance_m2=(
            np.array([[np.diag(values)]]) if covariance_m2 is None else covariance_m2
        ),
        eigenvalues_m2=values[np.newaxis, np.newaxis],
        principal_axes=(
            np.array([[np.eye(3)]]) if principal_axes is None else principal_axes
        ),
        rms_radius_m=np.array([[0.1]]),
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


@pytest.mark.parametrize(
    "confidence_level",
    [0.0, MIN_CONFIDENCE_LEVEL / 2.0, 1.0, -0.1, np.nan, True],
)
def test_confidence_ellipsoid_rejects_invalid_confidence_levels(
    confidence_level: object,
) -> None:
    with pytest.raises(ContractViolationError, match="confidence_level"):
        build_confidence_ellipsoids(
            _dispersion(),
            "clubhead",
            confidence_level,  # type: ignore[arg-type]
        )


def test_confidence_ellipsoid_is_accurate_in_the_representable_upper_tail() -> None:
    confidence = math.nextafter(1.0, 0.0)

    result = build_confidence_ellipsoids(_dispersion(), "clubhead", confidence)

    assert result.chi_square_quantile == pytest.approx(77.39631549062088, rel=2e-14)


@pytest.mark.parametrize(
    "eigenvalues",
    [
        np.array([-1.0, -2.0, -3.0]),
        np.array([0.1, 1.0, 0.5]),
    ],
)
def test_invalid_eigensystems_cannot_qualify_as_quiet(
    eigenvalues: np.ndarray,
) -> None:
    dispersion = _single_sample_dispersion(eigenvalues=eigenvalues)

    ellipsoid = build_confidence_ellipsoids(dispersion, "clubhead")
    sigma = build_dispersion_metric_series(
        dispersion, "clubhead", LARGEST_PRINCIPAL_SIGMA
    )
    intervals = find_ranked_low_variability_intervals(
        dispersion,
        LowVariabilityMetricCriteria(
            metric=LARGEST_PRINCIPAL_SIGMA,
            max_value=10.0,
        ),
    )

    assert ellipsoid.adequacy == ("invalid-covariance",)
    assert np.isnan(ellipsoid.semi_axis_lengths_m[0]).all()
    assert np.isnan(sigma.values[0])
    assert intervals == ()


def test_roundoff_scale_negative_eigenvalue_is_treated_as_zero_rank() -> None:
    eigenvalues = np.array([1.0, 0.25, -4.0 * np.finfo(float).eps])
    dispersion = _single_sample_dispersion(eigenvalues=eigenvalues)

    result = build_confidence_ellipsoids(dispersion, "clubhead")

    assert result.adequacy == ("rank-deficient",)
    assert result.semi_axis_lengths_m[0, 2] == 0.0


@pytest.mark.parametrize("invalid_field", ["center", "axes", "covariance"])
def test_invalid_plot_geometry_is_never_marked_estimable(invalid_field: str) -> None:
    overrides: dict[str, np.ndarray] = {}
    if invalid_field == "center":
        overrides["mean_positions_m"] = np.full((1, 1, 3), np.nan)
    elif invalid_field == "axes":
        axes = np.eye(3)
        axes[0, 0] = 2.0
        overrides["principal_axes"] = axes[np.newaxis, np.newaxis]
    else:
        overrides["covariance_m2"] = np.array([[np.diag([4.0, 1.0, 0.5])]])
    dispersion = _single_sample_dispersion(**overrides)

    result = build_confidence_ellipsoids(dispersion, "clubhead")

    assert result.adequacy == ("invalid-covariance",)
    assert np.isnan(result.volume_m3[0])


def test_plot_ready_result_contract_rejects_nonorthonormal_estimable_axes() -> None:
    with pytest.raises(ContractViolationError, match="orthonormal"):
        ConfidenceEllipsoidSeries(
            point_id="clubhead",
            coordinate_frame="swing.world",
            interpretation="gaussian-position-content-region",
            confidence_level=0.95,
            degrees_of_freedom=3,
            chi_square_quantile=7.814727903251179,
            radius_scale=math.sqrt(7.814727903251179),
            minimum_samples_for_full_rank=4,
            sample_times_s=np.array([0.0]),
            valid_trial_count=np.array([8]),
            centers_m=np.zeros((1, 3)),
            principal_axes=np.array([np.diag([2.0, 1.0, 1.0])]),
            semi_axis_lengths_m=np.ones((1, 3)),
            volume_m3=np.array([4.0 * math.pi / 3.0]),
            adequacy=("estimable",),
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


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"metric": RMS_RADIUS, "max_value": "1.0"}, "max_value"),
        (
            {"metric": RMS_RADIUS, "max_value": 1.0, "min_duration_s": "0.0"},
            "min_duration_s",
        ),
        (
            {"metric": RMS_RADIUS, "max_value": 1.0, "point_ids": (1,)},
            "point_ids",
        ),
    ],
)
def test_metric_criteria_rejects_coercible_or_malformed_values(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ContractViolationError, match=message):
        LowVariabilityMetricCriteria(**kwargs)  # type: ignore[arg-type]


def test_metric_criteria_normalizes_real_scalars() -> None:
    criteria = LowVariabilityMetricCriteria(
        metric=RMS_RADIUS,
        max_value=np.float32(0.2),
        confidence_level=np.float32(0.95),
        min_duration_s=np.float32(0.1),
    )

    assert type(criteria.max_value) is float
    assert type(criteria.confidence_level) is float
    assert type(criteria.min_duration_s) is float
