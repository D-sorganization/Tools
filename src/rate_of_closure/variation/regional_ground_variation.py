"""Run bounded seeded variation over qualified regional-ground materials."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import cast

from rate_of_closure.variation.regional_ground_study_adapter import (
    MAX_REGIONAL_GROUND_STUDY_ROWS,
    RegionalGroundStudyOutcome,
    build_regional_ground_study_ensemble,
)
from rate_of_closure.variation.scalar_ensemble_contract import ScalarEnsembleDataset
from shared.python.contracts import require
from shared.python.swing_sim.flight import (
    FlightGroundTransferError,
    FlightRegionalGroundPipelineResult,
)
from shared.python.swing_sim.ground import (
    GroundProvenance,
    GroundRegionalMaterialPlanRequest,
)
from shared.python.swing_sim.ground.regional_plan_records import (
    regional_plan_request_sha256,
)
from shared.python.swing_sim.variation.engine import sample_inputs
from shared.python.swing_sim.variation.registry import (
    register_variable,
    variable_registry,
)
from shared.python.swing_sim.variation.spec import (
    NoiseSpec,
    VariationPlan,
)

from .regional_ground_variation_control import (
    GroundRegionalVariationCancelled as GroundRegionalVariationCancelled,
)
from .regional_ground_variation_control import (
    GroundRegionalVariationFailed as GroundRegionalVariationFailed,
)
from .regional_ground_variation_control import (
    GroundRegionalVariationFailureStage as GroundRegionalVariationFailureStage,
)
from .regional_ground_variation_control import (
    GroundRegionalVariationHooks,
)
from .regional_ground_variation_control import (
    GroundRegionalVariationProgress as GroundRegionalVariationProgress,
)
from .regional_ground_variation_control import (
    GroundRegionalVariationTerminalError as GroundRegionalVariationTerminalError,
)
from .regional_ground_variation_dataset import (
    GROUND_NORMAL_RESTITUTION_KEY as GROUND_NORMAL_RESTITUTION_KEY,
)
from .regional_ground_variation_dataset import (
    GROUND_ROLLING_RESISTANCE_KEY as GROUND_ROLLING_RESISTANCE_KEY,
)
from .regional_ground_variation_dataset import (
    INPUT_NORMAL_RESTITUTION_KEY as INPUT_NORMAL_RESTITUTION_KEY,
)
from .regional_ground_variation_dataset import (
    INPUT_ROLLING_RESISTANCE_KEY as INPUT_ROLLING_RESISTANCE_KEY,
)
from .regional_ground_variation_dataset import (
    REGIONAL_GROUND_VARIATION_ADAPTER_ID,
    REGIONAL_GROUND_VARIATION_PRODUCER_VERSION,
    VARIABLE_DEFINITIONS,
    augment_regional_ground_variation_dataset,
)
from .regional_ground_variation_execution import (
    CompleteBatchExecution,
    execute_complete_batch,
)

_SUPPORTED_KEYS = frozenset(
    (GROUND_NORMAL_RESTITUTION_KEY, GROUND_ROLLING_RESISTANCE_KEY)
)
_FIELD_NAMES = {
    GROUND_NORMAL_RESTITUTION_KEY: "normal_restitution",
    GROUND_ROLLING_RESISTANCE_KEY: "rolling_resistance",
}


def register_ground_variation_variables() -> None:
    """Register the adapter inputs through the shared extension seam."""
    registry = variable_registry()
    for definition in VARIABLE_DEFINITIONS:
        existing = registry.get(definition.key)
        if existing is None:
            register_variable(definition)
        else:
            require(existing == definition, "ground variation registry conflict")


@dataclass(frozen=True)
class GroundRegionalVariationRequest:
    """Immutable inputs and bounds for one seeded regional-ground study."""

    plan: VariationPlan
    regional_plan: GroundRegionalMaterialPlanRequest
    result_id: str
    source_provenance: str
    max_rows: int
    series_id: str | None = None

    def __post_init__(self) -> None:
        _validate_request_types(self)
        _validate_plan(self.plan, self.regional_plan)
        require(self.plan.n_runs <= self.max_rows, "plan n_runs exceeds max_rows")


@dataclass(frozen=True)
class GroundRegionalVariationTrial:
    """One immutable sampled plan passed to the injected pipeline executor."""

    trial_index: int
    sampled_values: Mapping[str, float]
    regional_plan: GroundRegionalMaterialPlanRequest
    input_sha256: str

    def __post_init__(self) -> None:
        require(
            type(self.trial_index) is int and self.trial_index >= 0,
            "trial_index must be a nonnegative integer",
        )
        values = dict(self.sampled_values)
        require(set(values) == _SUPPORTED_KEYS, "trial sampled keys are invalid")
        require(
            all(math.isfinite(value) for value in values.values()), "nonfinite trial"
        )
        require(
            type(self.regional_plan) is GroundRegionalMaterialPlanRequest,
            "trial regional_plan must be exact",
        )
        require(
            len(self.input_sha256) == 64
            and all(character in "0123456789abcdef" for character in self.input_sha256),
            "trial input_sha256 is invalid",
        )
        require(
            self.regional_plan.provenance.input_sha256 == self.input_sha256,
            "trial provenance must match input_sha256",
        )
        object.__setattr__(self, "sampled_values", MappingProxyType(values))


def _validate_request_types(request: GroundRegionalVariationRequest) -> None:
    require(type(request.plan) is VariationPlan, "plan must be an exact VariationPlan")
    require(
        type(request.regional_plan) is GroundRegionalMaterialPlanRequest,
        "regional_plan must be exact",
    )
    require(bool(request.result_id.strip()), "result_id must be nonempty")
    require(
        bool(request.source_provenance.strip()), "source_provenance must be nonempty"
    )
    require(
        type(request.max_rows) is int
        and 1 <= request.max_rows <= MAX_REGIONAL_GROUND_STUDY_ROWS,
        "max_rows is outside the supported range",
    )
    require(
        request.series_id is None or bool(request.series_id.strip()),
        "series_id must be nonempty when supplied",
    )


def _validate_plan(
    plan: VariationPlan, regional_plan: GroundRegionalMaterialPlanRequest
) -> None:
    require(plan.mode == "launch", "ground material variation requires launch mode")
    require(set(plan.base_variables) == _SUPPORTED_KEYS, "unsupported ground base key")
    require(
        all(spec.variable_key in _SUPPORTED_KEYS for spec in plan.noise),
        "unsupported ground variation key",
    )
    for spec in plan.noise:
        _validate_noise_spec(spec, float(plan.base_variables[spec.variable_key]))
    surface = regional_plan.base_surface
    require(
        plan.base_variables[GROUND_NORMAL_RESTITUTION_KEY]
        == surface.normal_restitution,
        "normal restitution base does not match regional plan",
    )
    require(
        plan.base_variables[GROUND_ROLLING_RESISTANCE_KEY]
        == surface.rolling_resistance,
        "rolling resistance base does not match regional plan",
    )


def _validate_noise_spec(spec: NoiseSpec, base_value: float) -> None:
    require(spec.is_global, "ground material variation must be global")
    require(
        type(spec.scale) is float and math.isfinite(spec.scale),
        "noise scale must be a finite float",
    )
    require(
        type(spec.lower) is float and type(spec.upper) is float,
        "ground material bounds must be explicit finite floats",
    )
    lower = cast(float, spec.lower)
    upper = cast(float, spec.upper)
    require(0.0 <= lower < upper <= 1.0, "ground material bounds must be in [0, 1]")
    require(lower <= base_value <= upper, "ground base must lie within bounds")


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(
        payload, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _trial_input_digest(
    request: GroundRegionalVariationRequest,
    trial_index: int,
    values: Mapping[str, float],
) -> str:
    return _canonical_digest(
        {
            "base_plan_sha256": regional_plan_request_sha256(request.regional_plan),
            "sampled_values": dict(values),
            "trial_index": trial_index,
            "variation_plan": request.plan.to_json_dict(),
        }
    )


def _sampled_plan(
    request: GroundRegionalVariationRequest,
    trial_index: int,
    values: Mapping[str, float],
) -> GroundRegionalVariationTrial:
    digest = _trial_input_digest(request, trial_index, values)
    surface_values = {_FIELD_NAMES[key]: value for key, value in values.items()}
    base_surface = replace(request.regional_plan.base_surface, **surface_values)
    provenance = GroundProvenance(
        REGIONAL_GROUND_VARIATION_ADAPTER_ID,
        REGIONAL_GROUND_VARIATION_PRODUCER_VERSION,
        request.regional_plan.provenance.source_revision,
        digest,
    )
    sampled_plan = replace(
        request.regional_plan,
        request_id=f"{request.regional_plan.request_id}/seed-{request.plan.seed}/trial-{trial_index}",
        base_surface=base_surface,
        provenance=provenance,
    )
    return GroundRegionalVariationTrial(trial_index, values, sampled_plan, digest)


def _trials(
    request: GroundRegionalVariationRequest,
) -> tuple[GroundRegionalVariationTrial, ...]:
    sampled = sample_inputs(request.plan)
    require(
        sampled.shape == (request.plan.n_runs, len(request.plan.noise)), "sample shape"
    )
    base = {key: float(request.plan.base_variables[key]) for key in _SUPPORTED_KEYS}
    trials = []
    for trial_index, sample in enumerate(sampled):
        values = dict(base)
        values.update(
            {
                spec.variable_key: float(sample[index])
                for index, spec in enumerate(request.plan.noise)
            }
        )
        require(
            all(math.isfinite(value) for value in values.values()), "nonfinite sample"
        )
        trials.append(_sampled_plan(request, trial_index, values))
    return tuple(trials)


def _validate_outcome(
    trial: GroundRegionalVariationTrial, outcome: object
) -> RegionalGroundStudyOutcome:
    require(
        type(outcome)
        in (FlightRegionalGroundPipelineResult, FlightGroundTransferError),
        "executor must return an exact pipeline result or transfer failure",
    )
    if type(outcome) is FlightRegionalGroundPipelineResult:
        pipeline = cast(FlightRegionalGroundPipelineResult, outcome)
        require(
            pipeline.regional_plan == trial.regional_plan,
            "pipeline result must retain the sampled regional plan",
        )
        require(
            pipeline.regional_plan_sha256
            == regional_plan_request_sha256(trial.regional_plan),
            "pipeline result digest must match the sampled regional plan",
        )
    return outcome


def _publish_complete_dataset(
    request: GroundRegionalVariationRequest,
    trials: tuple[GroundRegionalVariationTrial, ...],
    outcomes: tuple[RegionalGroundStudyOutcome, ...],
) -> ScalarEnsembleDataset:
    dataset = build_regional_ground_study_ensemble(
        outcomes,
        request.result_id,
        request.source_provenance,
        request.max_rows,
        series_id=request.series_id,
    )
    return augment_regional_ground_variation_dataset(dataset, request, trials)


def run_regional_ground_variation(
    request: GroundRegionalVariationRequest,
    executor: Callable[[GroundRegionalVariationTrial], RegionalGroundStudyOutcome],
    hooks: GroundRegionalVariationHooks | None = None,
) -> ScalarEnsembleDataset:
    """Publish one complete deterministic dataset or typed terminal error."""
    require(
        type(request) is GroundRegionalVariationRequest,
        "request must be an exact GroundRegionalVariationRequest",
    )
    require(callable(executor), "executor must be callable")
    trials = _trials(request)
    job = CompleteBatchExecution(
        trials,
        executor,
        _validate_outcome,
        lambda outcomes: _publish_complete_dataset(request, trials, outcomes),
    )
    return execute_complete_batch(job, hooks)
