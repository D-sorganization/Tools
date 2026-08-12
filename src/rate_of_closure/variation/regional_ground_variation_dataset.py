"""Scalar-ensemble projection for seeded regional-ground variation."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from rate_of_closure.variation.scalar_ensemble_contract import (
    ScalarEnsembleDataset,
    ScalarEnsembleProvenance,
    ScalarEnsembleRow,
    ScalarEnsembleStage,
    ScalarVariableCategory,
    ScalarVariableDefinition,
)
from shared.python.swing_sim.flight import (
    FLIGHT_REGIONAL_GROUND_PIPELINE_CONTRACT_VERSION,
)
from shared.python.swing_sim.ground.regional_plan_records import (
    regional_plan_request_sha256,
)
from shared.python.swing_sim.variation.registry import CATEGORY_LAUNCH, VariableDef
from shared.python.swing_sim.variation.spec import SCHEMA_VERSION, VariationPlan

if TYPE_CHECKING:
    from .regional_ground_variation import (
        GroundRegionalVariationRequest,
        GroundRegionalVariationTrial,
    )

GROUND_NORMAL_RESTITUTION_KEY = f"{CATEGORY_LAUNCH}.ground_normal_restitution"
GROUND_ROLLING_RESISTANCE_KEY = f"{CATEGORY_LAUNCH}.ground_rolling_resistance"
INPUT_NORMAL_RESTITUTION_KEY = "input.ground.base.normal_restitution"
INPUT_ROLLING_RESISTANCE_KEY = "input.ground.base.rolling_resistance"

REGIONAL_GROUND_VARIATION_ADAPTER_ID = "regional-ground-variation/scalar-ensemble/v1"
REGIONAL_GROUND_VARIATION_PRODUCER_VERSION = "1.0.0"
_INPUT_STAGE = ScalarEnsembleStage("ground_input", "Ground Material Input")
_INPUT_CATEGORY = ScalarVariableCategory("ground_parameter", "Ground Parameter")
_OUTPUT_KEYS = {
    GROUND_NORMAL_RESTITUTION_KEY: INPUT_NORMAL_RESTITUTION_KEY,
    GROUND_ROLLING_RESISTANCE_KEY: INPUT_ROLLING_RESISTANCE_KEY,
}
VARIABLE_DEFINITIONS = (
    VariableDef(
        GROUND_NORMAL_RESTITUTION_KEY,
        "Restitution",
        "1",
        0.4,
        0.05,
        "Base-surface normal restitution.",
    ),
    VariableDef(
        GROUND_ROLLING_RESISTANCE_KEY,
        "Rolling Resistance",
        "1",
        0.04,
        0.01,
        "Base-surface rolling resistance.",
    ),
)


def _input_variables(plan: VariationPlan) -> tuple[ScalarVariableDefinition, ...]:
    definitions = {item.key: item for item in VARIABLE_DEFINITIONS}
    return tuple(
        ScalarVariableDefinition(
            _OUTPUT_KEYS[spec.variable_key],
            definitions[spec.variable_key].label,
            definitions[spec.variable_key].unit,
            _INPUT_STAGE.key,
            _INPUT_CATEGORY.key,
        )
        for spec in plan.noise
    )


def _augmented_rows(
    dataset: ScalarEnsembleDataset,
    trials: tuple[GroundRegionalVariationTrial, ...],
    plan: VariationPlan,
) -> tuple[ScalarEnsembleRow, ...]:
    rows = []
    for row, trial in zip(dataset.rows, trials, strict=True):
        values = dict(row.values)
        for spec in plan.noise:
            values[_OUTPUT_KEYS[spec.variable_key]] = trial.sampled_values[
                spec.variable_key
            ]
        attributes = {} if row.attributes is None else dict(row.attributes)
        attributes.update(
            {
                "variation_seed": str(plan.seed),
                "variation_trial_index": str(trial.trial_index),
                "variation_input_sha256": trial.input_sha256,
                "variation_regional_plan_sha256": regional_plan_request_sha256(
                    trial.regional_plan
                ),
            }
        )
        rows.append(replace(row, values=values, attributes=attributes))
    return tuple(rows)


def augment_regional_ground_variation_dataset(
    dataset: ScalarEnsembleDataset,
    request: GroundRegionalVariationRequest,
    trials: tuple[GroundRegionalVariationTrial, ...],
) -> ScalarEnsembleDataset:
    """Add sampled inputs and immutable per-trial provenance to a dataset."""
    source_schema = (
        f"variation-plan/v{SCHEMA_VERSION}+"
        f"{FLIGHT_REGIONAL_GROUND_PIPELINE_CONTRACT_VERSION}"
    )
    return ScalarEnsembleDataset(
        dataset.schema_version,
        dataset.result_id,
        ScalarEnsembleProvenance(
            REGIONAL_GROUND_VARIATION_ADAPTER_ID,
            source_schema,
            request.source_provenance,
        ),
        (_INPUT_STAGE, *dataset.stages),
        (_INPUT_CATEGORY, *dataset.categories),
        (*_input_variables(request.plan), *dataset.variables),
        dataset.cohorts,
        _augmented_rows(dataset, trials, request.plan),
    )


__all__ = [
    "GROUND_NORMAL_RESTITUTION_KEY",
    "GROUND_ROLLING_RESISTANCE_KEY",
    "INPUT_NORMAL_RESTITUTION_KEY",
    "INPUT_ROLLING_RESISTANCE_KEY",
    "REGIONAL_GROUND_VARIATION_ADAPTER_ID",
    "REGIONAL_GROUND_VARIATION_PRODUCER_VERSION",
    "VARIABLE_DEFINITIONS",
    "augment_regional_ground_variation_dataset",
]
