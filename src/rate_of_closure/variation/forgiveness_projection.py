"""Project chip decision metrics onto the shared variation plot schema."""

from __future__ import annotations

import math

import numpy as np

from shared.python.swing_sim.variation.engine import VariationDataset

from .chip_forgiveness import ChipTrialCohort
from .forgiveness_runner import ChipForgivenessStudy


def forgiveness_variation_dataset(
    study: ChipForgivenessStudy,
) -> VariationDataset:
    """Return plot-ready loss, constraints, and availability-aware metrics."""
    if not isinstance(study, ChipForgivenessStudy):
        raise TypeError("study must be ChipForgivenessStudy")
    metric_names = sorted({name for record in study.records for name in record.metrics})
    output_names = ("loss", "constraint_violated", *metric_names)
    outputs = np.full((len(study.records), len(output_names)), np.nan)
    success: np.ndarray = np.ones(len(study.records), dtype=bool)
    for record in study.records:
        values = (
            record.loss,
            float(record.constraint_violated),
            *(record.metrics.get(name) for name in metric_names),
        )
        outputs[record.trial_index] = [
            math.nan if value is None else float(value) for value in values
        ]
        success[record.trial_index] = (
            record.cohort is not ChipTrialCohort.NUMERICAL_FAILURE
        )
    return VariationDataset(
        plan=study.plan,
        input_names=study.input_names,
        inputs=study.sampled_inputs,
        output_names=output_names,
        outputs=outputs,
        success=success,
        elapsed_s=0.0,
    )


__all__ = ["forgiveness_variation_dataset"]
