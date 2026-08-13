"""Deterministic production of genuine paired localized attribution authority."""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import cast

import numpy as np

from rate_of_closure.simulation import SimulationConfig, run_simulation
from rate_of_closure.simulation.pipeline import configured_swing_sample_times
from rate_of_closure.variation._localized_attribution_contract import (
    require_authority_shape,
)
from rate_of_closure.variation._localized_attribution_provenance import (
    canonical_design_identity,
    finite_value,
    require_result_matches_request,
    stable_id,
)
from rate_of_closure.variation.ensemble_request_identity import (
    request_identity_sha256,
)
from rate_of_closure.variation.localized_attribution import (
    AttributionAuthority,
    AttributionObservation,
    AttributionPair,
    AttributionSource,
    AttributionTarget,
    Availability,
    TrialStatus,
)
from rate_of_closure.variation.request_builder import (
    LOCALIZED_TORQUE_VARIABLE_JOINTS,
    build_simulation_ensemble_request_from_samples,
)
from rate_of_closure.variation.simulation_adapter import (
    run_simulation_ensemble,
    spatial_point_ids_for_source,
)
from rate_of_closure.variation.simulation_types import (
    NUMERICAL_FAILURE,
    SimulationEnsembleRequest,
    SimulationEnsembleResult,
    SimulationTrialOutcome,
)
from rate_of_closure.variation.trial_projection import SimulationExecutor
from shared.python.contracts import require
from shared.python.swing_sim.solver.solve import ProgressCallback
from shared.python.swing_sim.variation import NoiseSpec, VariationPlan


@dataclass(frozen=True)
class LocalizedAttributionDesign:
    """One explicit one-at-a-time localized intervention experiment."""

    design_id: str
    source_plan: VariationPlan
    base_config: SimulationConfig
    targets: tuple[AttributionTarget, ...]
    intervention_deltas_nm: Mapping[str, float]

    def __post_init__(self) -> None:
        design_id = stable_id(self.design_id, "design_id")
        require(isinstance(self.source_plan, VariationPlan), "invalid source_plan")
        require(isinstance(self.base_config, SimulationConfig), "invalid base_config")
        require(self.source_plan.mode == "swing", "source_plan must use swing mode")
        require(
            self.base_config.source_kind == "double_pendulum",
            "localized attribution requires the double_pendulum source",
        )
        require(
            not self.base_config.swing_run_config.commanded_torque_offsets,
            "base_config cannot contain pre-existing localized torque offsets",
        )
        require(not self.source_plan.groups, "paired design cannot retain noise groups")
        specs = self.source_plan.noise
        require(
            all(
                spec.variable_key in LOCALIZED_TORQUE_VARIABLE_JOINTS for spec in specs
            ),
            "paired attribution supports only localized torque sources",
        )
        deltas = {
            stable_id(key, "intervention spec ID"): finite_value(
                value, "intervention delta"
            )
            for key, value in self.intervention_deltas_nm.items()
        }
        spec_ids = {cast(str, spec.spec_id) for spec in specs}
        require(set(deltas) == spec_ids, "intervention roster must match source specs")
        require(
            all(value != 0.0 for value in deltas.values()),
            "intervention delta must be nonzero",
        )
        targets = tuple(self.targets)
        require(bool(targets), "paired attribution requires at least one target")
        require(
            all(isinstance(target, AttributionTarget) for target in targets),
            "targets must contain AttributionTarget values",
        )
        require(
            len({target.target_id for target in targets}) == len(targets),
            "target IDs must be unique",
        )
        require_authority_shape(
            len(specs), len(targets), len(specs), len(specs) * len(targets)
        )
        _validate_state_targets(self.base_config, targets)
        object.__setattr__(self, "design_id", design_id)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "intervention_deltas_nm", MappingProxyType(deltas))


@dataclass(frozen=True)
class LocalizedAttributionProduction:
    """Authority plus immutable design/request provenance."""

    authority: AttributionAuthority
    design: LocalizedAttributionDesign
    request: SimulationEnsembleRequest = field(repr=False)
    result: SimulationEnsembleResult = field(repr=False)
    design_identity: str
    request_identity: str

    def __post_init__(self) -> None:
        require(isinstance(self.authority, AttributionAuthority), "invalid authority")
        require(isinstance(self.design, LocalizedAttributionDesign), "invalid design")
        require(isinstance(self.request, SimulationEnsembleRequest), "invalid request")
        require(isinstance(self.result, SimulationEnsembleResult), "invalid result")
        for value, label in (
            (self.design_identity, "design_identity"),
            (self.request_identity, "request_identity"),
        ):
            require(
                isinstance(value, str)
                and len(value) == 64
                and all(char in "0123456789abcdef" for char in value),
                f"{label} must be a lowercase SHA-256",
            )
        actual_request_identity = request_identity_sha256(self.request)
        require(
            self.request_identity == actual_request_identity,
            "request identity must match the retained exact request",
        )
        expected_request, _ = _pair_request(self.design)
        require(
            actual_request_identity == request_identity_sha256(expected_request),
            "request must match the exact paired design",
        )
        require_result_matches_request(self.result, self.request)
        expected_design_identity = canonical_design_identity(
            self.design.design_id,
            self.design.base_config,
            self.design.source_plan,
            self.design.targets,
            self.design.intervention_deltas_nm,
            actual_request_identity,
        )
        require(
            self.design_identity == expected_design_identity,
            "design identity must match the canonical retained design",
        )
        require(
            self.authority.authority_id == f"paired-attribution.{self.design_identity}",
            "authority ID must bind the design identity",
        )
        expected_authority = _build_authority(
            self.design,
            self.result,
            self.request.sampled_inputs,
            expected_design_identity,
        )
        require(
            self.authority == expected_authority,
            "authority payload must match its design, request, and result",
        )


def _validate_state_targets(
    config: SimulationConfig, targets: tuple[AttributionTarget, ...]
) -> None:
    times = configured_swing_sample_times(config)
    spatial_points = spatial_point_ids_for_source(config.source_kind)
    for target in targets:
        if target.kind != "state":
            continue
        require(
            target.point_id in spatial_points,
            "unsupported state point",
            target.point_id,
        )
        require(
            target.time_s is not None and bool(np.any(times == target.time_s)),
            "state target time must lie exactly on the configured swing sample grid",
            target.time_s,
        )


def _pair_request(
    design: LocalizedAttributionDesign,
) -> tuple[SimulationEnsembleRequest, np.ndarray]:
    specs = design.source_plan.noise
    pair_plan = replace(design.source_plan, n_runs=2 * len(specs), groups=())
    base = design.source_plan.resolved_base()
    baseline = np.array([base[spec.variable_key] for spec in specs], dtype=float)
    samples = np.repeat(baseline[np.newaxis, :], pair_plan.n_runs, axis=0)
    for source_index, spec in enumerate(specs):
        samples[2 * source_index + 1, source_index] += design.intervention_deltas_nm[
            cast(str, spec.spec_id)
        ]
    request = build_simulation_ensemble_request_from_samples(
        pair_plan, design.base_config, samples
    )
    return request, samples


def produce_localized_attribution(
    design: LocalizedAttributionDesign,
    *,
    executor: SimulationExecutor = run_simulation,
    progress_cb: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> LocalizedAttributionProduction:
    """Execute exact baseline/one-source perturbation pairs in stable order."""
    require(isinstance(design, LocalizedAttributionDesign), "invalid design")
    request, samples = _pair_request(design)
    request_id = request_identity_sha256(request)
    design_id = canonical_design_identity(
        design.design_id,
        design.base_config,
        design.source_plan,
        design.targets,
        design.intervention_deltas_nm,
        request_id,
    )
    result = run_simulation_ensemble(
        request,
        executor=executor,
        progress_cb=progress_cb,
        cancel_event=cancel_event,
    )
    authority = _build_authority(design, result, samples, design_id)
    return LocalizedAttributionProduction(
        authority,
        design,
        request,
        result,
        design_id,
        request_id,
    )


def _build_authority(
    design: LocalizedAttributionDesign,
    result: SimulationEnsembleResult,
    samples: np.ndarray,
    design_identity: str,
) -> AttributionAuthority:
    sources = tuple(_source(spec) for spec in design.source_plan.noise)
    pairs: list[AttributionPair] = []
    observations: list[AttributionObservation] = []
    for source_index, source in enumerate(sources):
        baseline_index, perturbed_index = 2 * source_index, 2 * source_index + 1
        baseline = result.outcomes[baseline_index]
        perturbed = result.outcomes[perturbed_index]
        pair = AttributionPair(
            source.spec_id,
            baseline_index,
            perturbed_index,
            _status(baseline),
            _status(perturbed),
            float(samples[baseline_index, source_index]),
            float(samples[perturbed_index, source_index]),
        )
        pairs.append(pair)
        observations.extend(
            _observation(pair, target, result) for target in design.targets
        )
    return AttributionAuthority(
        f"paired-attribution.{design_identity}",
        sources,
        design.targets,
        tuple(pairs),
        tuple(observations),
    )


def _source(spec: NoiseSpec) -> AttributionSource:
    variable_key = spec.variable_key
    window = cast(tuple[float, float], spec.time_window_s)
    return AttributionSource(
        cast(str, spec.spec_id),
        variable_key,
        LOCALIZED_TORQUE_VARIABLE_JOINTS[variable_key],
        window,
        "N·m",
    )


def _status(outcome: SimulationTrialOutcome) -> TrialStatus:
    return TrialStatus(outcome.status.value)


def _observation(
    pair: AttributionPair,
    target: AttributionTarget,
    result: SimulationEnsembleResult,
) -> AttributionObservation:
    baseline = result.outcomes[pair.baseline_trial_index]
    perturbed = result.outcomes[pair.perturbed_trial_index]
    baseline_value = _target_value(result, baseline, target)
    perturbed_value = _target_value(result, perturbed, target)
    availability = _availability(
        target, baseline, perturbed, baseline_value, perturbed_value
    )
    response = (
        perturbed_value - baseline_value
        if availability is Availability.AVAILABLE
        and baseline_value is not None
        and perturbed_value is not None
        else None
    )
    return AttributionObservation(
        pair.source_spec_id,
        target.target_id,
        pair.baseline_trial_index,
        pair.perturbed_trial_index,
        pair.baseline_status,
        pair.perturbed_status,
        pair.baseline_source_value,
        pair.perturbed_source_value,
        baseline_value,
        perturbed_value,
        response,
        availability,
    )


def _target_value(
    result: SimulationEnsembleResult,
    outcome: SimulationTrialOutcome,
    target: AttributionTarget,
) -> float | None:
    if outcome.status is NUMERICAL_FAILURE:
        return None
    if target.kind != "state":
        raw_value: object = outcome.value(target.name)
        return None if raw_value is None else finite_value(raw_value, target.name)
    sample_index = int(np.flatnonzero(result.traces.sample_times_s == target.time_s)[0])
    if not result.traces.sample_valid[outcome.trial_index, sample_index]:
        return None
    point_index = result.traces.point_index(cast(str, target.point_id))
    axis = {"position_x_m": 0, "position_y_m": 1, "position_z_m": 2}[target.name]
    value = float(
        result.traces.positions_m[outcome.trial_index, sample_index, point_index, axis]
    )
    return value if math.isfinite(value) else None


def _availability(
    target: AttributionTarget,
    baseline: SimulationTrialOutcome,
    perturbed: SimulationTrialOutcome,
    baseline_value: float | None,
    perturbed_value: float | None,
) -> Availability:
    if NUMERICAL_FAILURE in (baseline.status, perturbed.status):
        return Availability.NUMERICAL_FAILURE
    if target.kind != "state" and (baseline_value is None or perturbed_value is None):
        return Availability.NO_IMPACT_UNAVAILABLE
    if baseline_value is None or perturbed_value is None:
        return Availability.NONFINITE_UNAVAILABLE
    return Availability.AVAILABLE


__all__ = [
    "LocalizedAttributionDesign",
    "LocalizedAttributionProduction",
    "produce_localized_attribution",
]
