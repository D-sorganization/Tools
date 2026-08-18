"""Paired source-to-downstream geometric propagation contracts."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.variation import (
    CATEGORY_LAUNCH,
    CommonReferenceTrace,
    EnsemblePositionTraces,
    NoiseSpec,
    PairedIntervention,
    VariationDataset,
    VariationPlan,
    analyze_paired_intervention,
    outputs_for_mode,
)

pytestmark = pytest.mark.physics

_BALL_SPEED = f"{CATEGORY_LAUNCH}.ball_speed_mph"
_POINT_IDS = ("swing.wrist", "swing.clubhead")
_TIMES = np.arange(5, dtype=float)


def _dataset(n_trials: int) -> VariationDataset:
    plan = VariationPlan(
        mode="launch",
        noise=(NoiseSpec(_BALL_SPEED, scale=1.0, spec_id="source-speed"),),
        n_runs=n_trials,
        seed=3,
    )
    names = outputs_for_mode("launch")
    success = np.ones(n_trials, dtype=bool)
    if n_trials > 1:
        success[1] = False
    outputs = np.zeros((n_trials, len(names)))
    outputs[~success] = np.nan
    return VariationDataset(
        plan=plan,
        input_names=(_BALL_SPEED,),
        inputs=np.arange(n_trials, dtype=float).reshape(-1, 1),
        output_names=names,
        outputs=outputs,
        success=success,
    )


def _ensemble(
    positions: np.ndarray,
    *,
    times: np.ndarray = _TIMES,
    frame: str = "swing.world",
    point_ids: tuple[str, ...] = _POINT_IDS,
) -> EnsemblePositionTraces:
    n_trials, n_samples = positions.shape[:2]
    impacts = np.full(n_trials, n_samples - 1)
    if n_trials > 1:
        impacts[1] = -1
    return EnsemblePositionTraces(
        variation=_dataset(n_trials),
        sample_times_s=times,
        coordinate_frame=frame,
        point_ids=point_ids,
        positions_m=positions,
        sample_valid=np.ones((n_trials, n_samples), dtype=bool),
        impact_sample_indices=impacts,
    )


def _planted_intervention() -> PairedIntervention:
    baseline_positions = np.zeros((3, len(_TIMES), len(_POINT_IDS), 3))
    perturbed_positions = baseline_positions.copy()
    amplitudes = np.array([0.8, 1.0, 1.2])
    perturbed_positions[:, 2:, 0, 1] = amplitudes[:, np.newaxis] * np.array(
        [0.02, 0.04, 0.06]
    )
    perturbed_positions[:, 2:, 1, 0] = amplitudes[:, np.newaxis] * np.array(
        [0.1, 0.25, 0.5]
    )
    spec = NoiseSpec(
        _BALL_SPEED,
        scale=1.0,
        spec_id="source-speed",
        time_window_s=(2.0, 4.0),
        point_ids=("swing.wrist",),
    )
    return PairedIntervention(
        spec=spec,
        baseline=_ensemble(baseline_positions),
        perturbed=_ensemble(perturbed_positions),
    )


class TestPairedPropagation:
    def test_planted_effect_has_no_pre_window_response_and_dominates_downstream(
        self,
    ) -> None:
        result = analyze_paired_intervention(_planted_intervention())

        np.testing.assert_array_equal(result.mean_displacement_m[:2], 0.0)
        np.testing.assert_array_equal(result.rms_induced_displacement_m[:2], 0.0)
        peak_index = np.unravel_index(
            np.argmax(result.rms_induced_displacement_m),
            result.rms_induced_displacement_m.shape,
        )
        assert peak_index == (4, 1)
        assert result.point_ids[peak_index[1]] == "swing.clubhead"
        assert result.sample_times_s[peak_index[0]] == 4.0
        np.testing.assert_allclose(result.mean_displacement_m[4, 1], [0.5, 0.0, 0.0])

    def test_result_retains_source_identity_and_locus_metadata(self) -> None:
        result = analyze_paired_intervention(_planted_intervention())

        assert result.spec_id == "source-speed"
        assert result.variable_key == _BALL_SPEED
        assert result.source_time_window_s == (2.0, 4.0)
        assert result.source_point_ids == ("swing.wrist",)
        assert result.coordinate_frame == "swing.world"

    def test_no_impact_and_failed_outcome_row_remains_in_geometry(self) -> None:
        result = analyze_paired_intervention(_planted_intervention())

        assert result.paired_count[4, 1] == 3
        expected = 0.5 * np.sqrt(np.mean(np.square([0.8, 1.0, 1.2])))
        assert result.rms_induced_displacement_m[4, 1] == pytest.approx(expected)

    def test_outputs_are_defensive_read_only_copies(self) -> None:
        result = analyze_paired_intervention(_planted_intervention())

        with pytest.raises(ValueError, match="read-only"):
            result.mean_displacement_m[0, 0, 0] = 1.0
        with pytest.raises(ValueError, match="read-only"):
            result.paired_count[0, 0] = 0

    def test_result_contract_rejects_invalid_identity_and_statistics(self) -> None:
        result = analyze_paired_intervention(_planted_intervention())
        with pytest.raises(ContractViolationError, match="spec_id"):
            replace(result, spec_id=" source-speed")

        negative_rms = result.rms_induced_displacement_m.copy()
        negative_rms[0, 0] = -1.0
        with pytest.raises(ContractViolationError, match="RMS non-negative"):
            replace(result, rms_induced_displacement_m=negative_rms)

        missing_count = result.paired_count.copy()
        missing_count[0, 0] = 0
        with pytest.raises(ContractViolationError, match="unobserved"):
            replace(result, paired_count=missing_count)

    @pytest.mark.parametrize("mismatch", ["frame", "time", "point", "trials"])
    def test_rejects_unpaired_or_incompatible_layouts(self, mismatch: str) -> None:
        intervention = _planted_intervention()
        perturbed = intervention.perturbed
        positions = perturbed.positions_m.copy()
        times = _TIMES
        frame = "swing.world"
        point_ids = _POINT_IDS
        if mismatch == "frame":
            frame = "launch.world"
        elif mismatch == "time":
            times = _TIMES + 0.1
        elif mismatch == "point":
            point_ids = ("swing.pivot", "swing.clubhead")
        else:
            positions = positions[:2]
        incompatible = _ensemble(
            positions, times=times, frame=frame, point_ids=point_ids
        )

        with pytest.raises(ContractViolationError, match=mismatch.rstrip("s")):
            PairedIntervention(
                spec=intervention.spec,
                baseline=intervention.baseline,
                perturbed=incompatible,
            )


def test_common_reference_trace_is_broadcast_across_perturbed_trials() -> None:
    intervention = _planted_intervention()
    reference = CommonReferenceTrace(
        sample_times_s=intervention.baseline.sample_times_s,
        coordinate_frame=intervention.baseline.coordinate_frame,
        point_ids=intervention.baseline.point_ids,
        positions_m=intervention.baseline.positions_m[0],
        sample_valid=intervention.baseline.sample_valid[0],
    )

    result = analyze_paired_intervention(
        PairedIntervention(
            spec=intervention.spec,
            baseline=reference,
            perturbed=intervention.perturbed,
        )
    )

    assert result.reference_kind == "common"
    assert np.all(result.paired_count == 3)
