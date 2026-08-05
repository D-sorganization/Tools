"""Pure-data ensemble swing-geometry contracts and analysis."""

from __future__ import annotations

import numpy as np
import pytest

from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.variation import (
    CATEGORY_LAUNCH,
    EnsemblePositionTraces,
    LowVariabilityCriteria,
    NoiseSpec,
    VariationDataset,
    VariationPlan,
    compute_position_dispersion,
    find_low_variability_intervals,
    outputs_for_mode,
)

pytestmark = pytest.mark.physics

_BALL_SPEED = f"{CATEGORY_LAUNCH}.ball_speed_mph"
_POINT_IDS = ("swing.pivot", "swing.clubhead")


def _variation_dataset(
    n_trials: int, success: np.ndarray | None = None
) -> VariationDataset:
    plan = VariationPlan(
        mode="launch",
        noise=(NoiseSpec(_BALL_SPEED, scale=1.0),),
        n_runs=n_trials,
        seed=7,
    )
    output_names = outputs_for_mode("launch")
    success_flags = np.ones(n_trials, dtype=bool) if success is None else success
    outputs = np.zeros((n_trials, len(output_names)))
    outputs[~success_flags] = np.nan
    return VariationDataset(
        plan=plan,
        input_names=(_BALL_SPEED,),
        inputs=np.arange(n_trials, dtype=float).reshape(-1, 1),
        output_names=output_names,
        outputs=outputs,
        success=success_flags,
    )


def _ensemble(scales: tuple[float, ...] = (1.0, 0.2, 0.1)) -> EnsemblePositionTraces:
    n_trials = 3
    sample_times_s = np.arange(len(scales), dtype=float)
    positions = np.zeros((n_trials, len(scales), len(_POINT_IDS), 3))
    offsets = np.array([-1.0, 0.0, 1.0])
    for sample_index, scale in enumerate(scales):
        positions[:, sample_index, 1, 0] = offsets * scale
    return EnsemblePositionTraces(
        variation=_variation_dataset(n_trials),
        sample_times_s=sample_times_s,
        coordinate_frame="swing.world",
        point_ids=_POINT_IDS,
        positions_m=positions,
        sample_valid=np.ones((n_trials, len(scales)), dtype=bool),
        impact_sample_indices=np.array([len(scales) - 1, -1, len(scales) - 1]),
    )


class TestEnsemblePositionTraces:
    def test_retains_no_impact_trial_and_stable_point_lookup(self) -> None:
        ensemble = _ensemble()

        assert ensemble.n_trials == 3
        assert ensemble.n_no_impact == 1
        assert ensemble.coordinate_frame == "swing.world"
        np.testing.assert_array_equal(
            ensemble.impact_occurred, np.array([True, False, True])
        )
        assert ensemble.point_index("swing.clubhead") == 1

    def test_no_impact_or_failed_shot_row_still_contributes_geometry(self) -> None:
        original = _ensemble((1.0,))
        failed_shot_dataset = _variation_dataset(
            3, success=np.array([True, False, True])
        )
        ensemble = EnsemblePositionTraces(
            variation=failed_shot_dataset,
            sample_times_s=original.sample_times_s,
            coordinate_frame=original.coordinate_frame,
            point_ids=original.point_ids,
            positions_m=original.positions_m,
            sample_valid=original.sample_valid,
            impact_sample_indices=original.impact_sample_indices,
        )

        dispersion = compute_position_dispersion(ensemble)

        assert dispersion.count[0, 1] == 3
        assert dispersion.rms_radius_m[0, 1] == pytest.approx(np.sqrt(2.0 / 3.0))

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("sample_times_s", np.array([0.0, 0.0, 1.0]), "strictly increasing"),
            ("point_ids", ("swing.pivot", "swing.pivot"), "unique"),
            (
                "positions_m",
                np.zeros((3, 3, 2, 2)),
                "three Cartesian coordinates",
            ),
            ("coordinate_frame", "  ", "coordinate_frame"),
            ("impact_sample_indices", np.array([3, -1, 2]), "impact sample"),
        ],
    )
    def test_rejects_invalid_trace_contracts(
        self, field: str, value: object, message: str
    ) -> None:
        valid = _ensemble()
        kwargs = {
            "variation": valid.variation,
            "sample_times_s": valid.sample_times_s,
            "coordinate_frame": valid.coordinate_frame,
            "point_ids": valid.point_ids,
            "positions_m": valid.positions_m,
            "sample_valid": valid.sample_valid,
            "impact_sample_indices": valid.impact_sample_indices,
        }
        kwargs[field] = value

        with pytest.raises(ContractViolationError, match=message):
            EnsemblePositionTraces(**kwargs)

    def test_rejects_impact_marker_on_an_invalid_sample(self) -> None:
        valid = _ensemble()
        positions = valid.positions_m.copy()
        sample_valid = valid.sample_valid.copy()
        positions[0, 1] = np.nan
        sample_valid[0, 1] = False

        with pytest.raises(ContractViolationError, match="refer to a valid"):
            EnsemblePositionTraces(
                variation=valid.variation,
                sample_times_s=valid.sample_times_s,
                coordinate_frame=valid.coordinate_frame,
                point_ids=valid.point_ids,
                positions_m=positions,
                sample_valid=sample_valid,
                impact_sample_indices=np.array([1, -1, 2]),
            )

    def test_rejects_nonfinite_valid_positions_and_finite_invalid_positions(
        self,
    ) -> None:
        valid = _ensemble()
        positions = valid.positions_m.copy()
        positions[0, 0, 0, 0] = np.nan
        with pytest.raises(ContractViolationError, match="valid samples.*finite"):
            EnsemblePositionTraces(
                variation=valid.variation,
                sample_times_s=valid.sample_times_s,
                coordinate_frame=valid.coordinate_frame,
                point_ids=valid.point_ids,
                positions_m=positions,
                sample_valid=valid.sample_valid,
                impact_sample_indices=valid.impact_sample_indices,
            )

        positions = valid.positions_m.copy()
        sample_valid = valid.sample_valid.copy()
        sample_valid[0, 0] = False
        with pytest.raises(ContractViolationError, match="invalid samples.*NaN"):
            EnsemblePositionTraces(
                variation=valid.variation,
                sample_times_s=valid.sample_times_s,
                coordinate_frame=valid.coordinate_frame,
                point_ids=valid.point_ids,
                positions_m=positions,
                sample_valid=sample_valid,
                impact_sample_indices=np.array([-1, -1, 2]),
            )


class TestPositionDispersion:
    def test_covariance_eigenvalues_and_rms_radius_are_per_sample_and_point(
        self,
    ) -> None:
        dispersion = compute_position_dispersion(_ensemble())

        assert dispersion.point_ids == _POINT_IDS
        assert dispersion.coordinate_frame == "swing.world"
        assert dispersion.covariance_m2.shape == (3, 2, 3, 3)
        np.testing.assert_allclose(
            dispersion.covariance_m2[0, 1], np.diag([1.0, 0.0, 0.0])
        )
        np.testing.assert_allclose(dispersion.eigenvalues_m2[0, 1], [1.0, 0.0, 0.0])
        assert dispersion.rms_radius_m[0, 1] == pytest.approx(np.sqrt(2.0 / 3.0))
        np.testing.assert_allclose(dispersion.mean_positions_m[:, 1], 0.0)

    def test_single_valid_trial_reports_mean_but_not_sample_covariance(self) -> None:
        ensemble = _ensemble((1.0,))
        sample_valid = np.array([[True], [False], [False]])
        positions = ensemble.positions_m.copy()
        positions[~sample_valid] = np.nan
        ensemble = EnsemblePositionTraces(
            variation=ensemble.variation,
            sample_times_s=ensemble.sample_times_s,
            coordinate_frame=ensemble.coordinate_frame,
            point_ids=ensemble.point_ids,
            positions_m=positions,
            sample_valid=sample_valid,
            impact_sample_indices=np.array([0, -1, -1]),
        )

        dispersion = compute_position_dispersion(ensemble)

        assert dispersion.count[0, 1] == 1
        assert np.all(np.isfinite(dispersion.mean_positions_m[0, 1]))
        assert np.all(np.isnan(dispersion.covariance_m2[0, 1]))
        assert np.all(np.isnan(dispersion.eigenvalues_m2[0, 1]))
        assert dispersion.rms_radius_m[0, 1] == pytest.approx(0.0)

    def test_result_is_deterministic_and_does_not_mutate_inputs(self) -> None:
        ensemble = _ensemble()
        original = ensemble.positions_m.copy()

        first = compute_position_dispersion(ensemble)
        second = compute_position_dispersion(ensemble)

        np.testing.assert_array_equal(first.count, second.count)
        np.testing.assert_array_equal(first.covariance_m2, second.covariance_m2)
        np.testing.assert_array_equal(ensemble.positions_m, original)

    def test_public_arrays_are_defensive_read_only_copies(self) -> None:
        source = np.zeros((3, 1, len(_POINT_IDS), 3))
        ensemble = EnsemblePositionTraces(
            variation=_variation_dataset(3),
            sample_times_s=np.array([0.0]),
            coordinate_frame="swing.world",
            point_ids=_POINT_IDS,
            positions_m=source,
            sample_valid=np.ones((3, 1), dtype=bool),
            impact_sample_indices=np.array([0, -1, 0]),
        )
        source[0, 0, 0, 0] = 99.0

        assert ensemble.positions_m[0, 0, 0, 0] == 0.0
        with pytest.raises(ValueError, match="read-only"):
            ensemble.positions_m[0, 0, 0, 0] = 5.0
        dispersion = compute_position_dispersion(ensemble)
        with pytest.raises(ValueError, match="read-only"):
            dispersion.rms_radius_m[0, 0] = 5.0

    def test_principal_axis_signs_are_canonical_for_export_parity(self) -> None:
        covariance = np.array([[1.0, 0.2, 0.0], [0.2, 2.0, 0.1], [0.0, 0.1, 0.5]])
        basis = np.linalg.cholesky(covariance) * np.sqrt(5.0 / 2.0)
        samples = np.concatenate([basis.T, -basis.T], axis=0)
        ensemble = EnsemblePositionTraces(
            variation=_variation_dataset(6),
            sample_times_s=np.array([0.0]),
            coordinate_frame="swing.world",
            point_ids=("swing.clubhead",),
            positions_m=samples[:, np.newaxis, np.newaxis, :],
            sample_valid=np.ones((6, 1), dtype=bool),
            impact_sample_indices=np.full(6, -1),
        )

        axes = compute_position_dispersion(ensemble).principal_axes[0, 0]
        largest_rows = np.argmax(np.abs(axes), axis=0)
        signed_components = axes[largest_rows, np.arange(3)]

        assert np.all(signed_components >= 0.0)


class TestLowVariabilityIntervals:
    def test_finds_contiguous_intervals_for_each_requested_point(self) -> None:
        dispersion = compute_position_dispersion(_ensemble((0.1, 0.2, 1.0, 0.1, 0.1)))
        criteria = LowVariabilityCriteria(
            max_rms_radius_m=0.2,
            min_samples=2,
            point_ids=("swing.clubhead",),
        )

        intervals = find_low_variability_intervals(dispersion, criteria)

        assert [(item.start_index, item.end_index) for item in intervals] == [
            (0, 1),
            (3, 4),
        ]
        assert all(item.point_id == "swing.clubhead" for item in intervals)
        assert intervals[0].max_rms_radius_m == pytest.approx(0.2 * np.sqrt(2.0 / 3.0))

    def test_duration_and_sample_count_filters_are_explicit(self) -> None:
        dispersion = compute_position_dispersion(_ensemble((0.1, 0.1, 1.0)))

        intervals = find_low_variability_intervals(
            dispersion,
            LowVariabilityCriteria(
                max_rms_radius_m=0.2,
                min_duration_s=1.1,
                point_ids=("swing.clubhead",),
            ),
        )

        assert intervals == ()

    def test_rejects_unknown_point_and_invalid_threshold(self) -> None:
        dispersion = compute_position_dispersion(_ensemble())
        with pytest.raises(ContractViolationError, match="max_rms_radius_m"):
            LowVariabilityCriteria(max_rms_radius_m=0.0)
        with pytest.raises(ContractViolationError, match="unknown point_id"):
            find_low_variability_intervals(
                dispersion,
                LowVariabilityCriteria(
                    max_rms_radius_m=1.0, point_ids=("swing.unknown",)
                ),
            )
