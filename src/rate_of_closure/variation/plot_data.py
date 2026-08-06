"""Plot-ready facade for complete, typed Rate simulation ensembles.

The facade owns selection, availability, units, and deterministic rendering
budgets. Renderers receive immutable arrays and must not recompute physics or
silently discard misses and numerical failures.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

import numpy as np

from rate_of_closure.variation.geometric_plot_data import (
    GeometricVariabilityData,
    build_geometric_variability,
)
from rate_of_closure.variation.plot_labels import OUTPUT_LABELS, OUTPUT_UNITS
from rate_of_closure.variation.simulation_types import (
    CONTACT_OUTPUT_NAMES,
    IMPACT_OUTPUT_NAMES,
    SHOT_OUTPUT_NAMES,
    SimulationEnsembleResult,
    TrialEvaluationStatus,
)
from shared.python.contracts import require
from shared.python.swing_sim.variation import (
    SCHEMA_VERSION,
    LowVariabilityCriteria,
    PositionDispersion,
    VariationDataset,
    compute_position_dispersion,
    variable_registry,
)
from shared.python.swing_sim.variation.ensemble_types import immutable_array

DEFAULT_MAX_ARC_VERTICES = 250_000


class ScalarVariableKind(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    """Source/availability class of one selectable scalar plot variable."""

    INPUT = "input"
    CONTACT = "contact"
    IMPACT = "impact"
    SHOT = "shot"


@dataclass(frozen=True)
class ScalarPlotVariable:
    """Stable scalar axis descriptor with an explicit unit and data class."""

    key: str
    label: str
    unit: str
    kind: ScalarVariableKind

    def __post_init__(self) -> None:
        require(bool(self.key) and self.key == self.key.strip(), "key must be trimmed")
        require(
            bool(self.label) and self.label == self.label.strip(),
            "label must be trimmed",
        )
        require(self.unit == self.unit.strip(), "unit must be trimmed")


@dataclass(frozen=True)
class CohortAvailability:
    """Counts for one typed cohort in a particular paired scatter view."""

    cohort: TrialEvaluationStatus
    total: int
    plotted: int
    unavailable: int

    def __post_init__(self) -> None:
        require(self.total >= 0, "total must be non-negative", self.total)
        require(self.plotted >= 0, "plotted must be non-negative", self.plotted)
        require(
            self.unavailable == self.total - self.plotted,
            "unavailable must equal total minus plotted",
        )


@dataclass(frozen=True)
class ScatterPlotData:
    """Finite paired scalar values plus an honest cohort-availability ledger."""

    x_variable: ScalarPlotVariable
    y_variable: ScalarPlotVariable
    trial_indices: np.ndarray = field(repr=False)
    x: np.ndarray = field(repr=False)
    y: np.ndarray = field(repr=False)
    cohorts: tuple[TrialEvaluationStatus, ...]
    cohort_summaries: Mapping[TrialEvaluationStatus, CohortAvailability]

    def __post_init__(self) -> None:
        trial_indices = np.asarray(self.trial_indices, dtype=int)
        x_values = np.asarray(self.x, dtype=float)
        y_values = np.asarray(self.y, dtype=float)
        expected = (trial_indices.size,)
        require(x_values.shape == expected, "x must align with trial_indices")
        require(y_values.shape == expected, "y must align with trial_indices")
        require(len(self.cohorts) == trial_indices.size, "cohorts must align")
        require(np.all(np.isfinite(x_values)), "plotted x values must be finite")
        require(np.all(np.isfinite(y_values)), "plotted y values must be finite")
        object.__setattr__(self, "trial_indices", immutable_array(trial_indices, int))
        object.__setattr__(self, "x", immutable_array(x_values, float))
        object.__setattr__(self, "y", immutable_array(y_values, float))
        object.__setattr__(
            self, "cohort_summaries", MappingProxyType(dict(self.cohort_summaries))
        )

    def summary(self, cohort: TrialEvaluationStatus) -> CohortAvailability:
        """Return availability counts for ``cohort`` in this scatter."""
        require(cohort in self.cohort_summaries, "unknown cohort", cohort)
        return self.cohort_summaries[cohort]


@dataclass(frozen=True)
class PlotBudget:
    """Deterministic upper bound for vertices handed to an arc renderer."""

    max_arc_vertices: int = DEFAULT_MAX_ARC_VERTICES

    def __post_init__(self) -> None:
        require(
            isinstance(self.max_arc_vertices, int) and self.max_arc_vertices >= 1,
            "max_arc_vertices must be an integer >= 1",
            self.max_arc_vertices,
        )


DEFAULT_PLOT_BUDGET = PlotBudget()


@dataclass(frozen=True)
class ArcOverlayData:
    """All trial arc rows for one modeled point on a deterministic time grid."""

    point_id: str
    coordinate_frame: str
    position_unit: str
    sample_indices: np.ndarray = field(repr=False)
    sample_times_s: np.ndarray = field(repr=False)
    positions_m: np.ndarray = field(repr=False)
    sample_valid: np.ndarray = field(repr=False)
    reference_positions_m: np.ndarray = field(repr=False)
    cohorts: tuple[TrialEvaluationStatus, ...]
    dispersion: PositionDispersion
    raw_vertex_count: int
    rendered_vertex_count: int

    def __post_init__(self) -> None:
        indices = np.asarray(self.sample_indices, dtype=int)
        times = np.asarray(self.sample_times_s, dtype=float)
        positions = np.asarray(self.positions_m, dtype=float)
        valid = np.asarray(self.sample_valid, dtype=bool)
        reference = np.asarray(self.reference_positions_m, dtype=float)
        expected = (len(self.cohorts), indices.size)
        require(times.shape == (indices.size,), "sample_times_s must align")
        require(positions.shape == expected + (3,), "positions_m has invalid shape")
        require(valid.shape == expected, "sample_valid has invalid shape")
        require(reference.shape == (indices.size, 3), "reference has invalid shape")
        require(self.rendered_vertex_count <= self.raw_vertex_count, "invalid budget")
        object.__setattr__(self, "sample_indices", immutable_array(indices, int))
        object.__setattr__(self, "sample_times_s", immutable_array(times, float))
        object.__setattr__(self, "positions_m", immutable_array(positions, float))
        object.__setattr__(self, "sample_valid", immutable_array(valid, bool))
        object.__setattr__(
            self, "reference_positions_m", immutable_array(reference, float)
        )


@dataclass(frozen=True)
class EnsemblePlotDataset:
    """Single plot-data entrypoint for a complete simulation ensemble."""

    result_id: str
    result: SimulationEnsembleResult = field(repr=False)
    variables: tuple[ScalarPlotVariable, ...]
    dispersion: PositionDispersion

    def __post_init__(self) -> None:
        require(
            bool(self.result_id) and self.result_id == self.result_id.strip(),
            "result_id must be a non-empty trimmed stable ID",
        )
        keys = tuple(variable.key for variable in self.variables)
        require(len(set(keys)) == len(keys), "scalar variable keys must be unique")

    @property
    def coordinate_frame(self) -> str:
        """Coordinate frame used by every spatial trace and arc view."""
        return str(self.result.traces.coordinate_frame)

    @property
    def cohorts(self) -> tuple[TrialEvaluationStatus, ...]:
        """Typed cohort for every trial row in stable trial order."""
        return tuple(outcome.status for outcome in self.result.outcomes)

    def geometric_variability(
        self, point_id: str, criteria: LowVariabilityCriteria
    ) -> GeometricVariabilityData:
        """Return one point's RMS envelope, principal spread, and quiet zones."""
        return build_geometric_variability(self.dispersion, point_id, criteria)

    def variable(self, key: str) -> ScalarPlotVariable:
        """Return a scalar descriptor by stable prefixed key."""
        match = next((item for item in self.variables if item.key == key), None)
        require(match is not None, "unknown scalar variable", key)
        assert match is not None
        return match

    def scatter(self, x_key: str, y_key: str) -> ScatterPlotData:
        """Prepare finite paired values while accounting for unavailable rows."""
        x_variable = self.variable(x_key)
        y_variable = self.variable(y_key)
        x_values = self._scalar_values(x_variable)
        y_values = self._scalar_values(y_variable)
        available = np.isfinite(x_values) & np.isfinite(y_values)
        indices = np.flatnonzero(available)
        plotted_cohorts = tuple(self.cohorts[index] for index in indices)
        summaries = {
            cohort: _cohort_availability(self.cohorts, available, cohort)
            for cohort in TrialEvaluationStatus
        }
        return ScatterPlotData(
            x_variable=x_variable,
            y_variable=y_variable,
            trial_indices=indices,
            x=x_values[available],
            y=y_values[available],
            cohorts=plotted_cohorts,
            cohort_summaries=summaries,
        )

    def arc_overlay(
        self,
        point_id: str,
        budget: PlotBudget | None = None,
    ) -> ArcOverlayData:
        """Prepare every trial arc with deterministic sample-axis decimation."""
        traces = self.result.traces
        require(point_id in traces.point_ids, "unknown point_id", point_id)
        point_index = traces.point_index(point_id)
        selected_budget = budget or DEFAULT_PLOT_BUDGET
        sample_indices = _sample_indices(
            traces.n_trials,
            traces.sample_times_s.size,
            selected_budget.max_arc_vertices,
        )
        positions = traces.positions_m[:, sample_indices, point_index, :]
        valid = traces.sample_valid[:, sample_indices]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            reference = np.nanmedian(positions, axis=0)
        return ArcOverlayData(
            point_id=point_id,
            coordinate_frame=traces.coordinate_frame,
            position_unit="m",
            sample_indices=sample_indices,
            sample_times_s=traces.sample_times_s[sample_indices],
            positions_m=positions,
            sample_valid=valid,
            reference_positions_m=reference,
            cohorts=self.cohorts,
            dispersion=self.dispersion,
            raw_vertex_count=traces.n_trials * traces.sample_times_s.size,
            rendered_vertex_count=traces.n_trials * sample_indices.size,
        )

    def _scalar_values(self, variable: ScalarPlotVariable) -> np.ndarray:
        """Return one immutable scalar column aligned to all trial rows."""
        source, name = variable.key.split(":", 1)
        if source == "input":
            column = self.result.variation.input_names.index(name)
            return np.asarray(self.result.variation.inputs[:, column], dtype=float)
        column = self.result.variation.output_names.index(name)
        return np.asarray(self.result.variation.outputs[:, column], dtype=float)


def _cohort_availability(
    cohorts: tuple[TrialEvaluationStatus, ...],
    available: np.ndarray,
    cohort: TrialEvaluationStatus,
) -> CohortAvailability:
    """Count total and paired-finite rows for one cohort."""
    cohort_mask = np.fromiter((item is cohort for item in cohorts), dtype=bool)
    total = int(np.count_nonzero(cohort_mask))
    plotted = int(np.count_nonzero(cohort_mask & available))
    return CohortAvailability(cohort, total, plotted, total - plotted)


def _sample_indices(n_trials: int, n_samples: int, max_vertices: int) -> np.ndarray:
    """Return deterministic monotonic sample indices, preserving endpoints."""
    allowed_samples = max(1, max_vertices // max(n_trials, 1))
    if allowed_samples >= n_samples:
        return np.arange(n_samples, dtype=int)
    indices = np.linspace(0, n_samples - 1, allowed_samples, dtype=int)
    return np.unique(indices)


def _kind_for_output(name: str) -> ScalarVariableKind:
    """Return the canonical availability class for one output name."""
    if name in CONTACT_OUTPUT_NAMES:
        return ScalarVariableKind.CONTACT
    if name in IMPACT_OUTPUT_NAMES:
        return ScalarVariableKind.IMPACT
    require(name in SHOT_OUTPUT_NAMES, "unknown output variable", name)
    return ScalarVariableKind.SHOT


def _kind_for_any_output(name: str) -> ScalarVariableKind:
    """Classify complete-simulation and scalar-pipeline output vocabularies."""
    if name in CONTACT_OUTPUT_NAMES + IMPACT_OUTPUT_NAMES + SHOT_OUTPUT_NAMES:
        return _kind_for_output(name)
    if name in {
        "club_path_deg",
        "face_angle_deg",
        "attack_angle_deg",
        "dynamic_loft_deg",
    }:
        return ScalarVariableKind.IMPACT
    return ScalarVariableKind.SHOT


def _title(name: str) -> str:
    """Convert a stable snake-case identifier into a compact display label."""
    return str(OUTPUT_LABELS.get(name, name.replace("_", " ").title()))


def scalar_plot_variables(
    dataset: VariationDataset,
) -> tuple[ScalarPlotVariable, ...]:
    """Build unit-bearing scalar descriptors for any variation dataset."""
    registry = variable_registry()
    inputs = tuple(
        ScalarPlotVariable(
            key=f"input:{name}",
            label=registry[name].label,
            unit=registry[name].unit,
            kind=ScalarVariableKind.INPUT,
        )
        for name in dataset.input_names
    )
    outputs = tuple(
        ScalarPlotVariable(
            key=f"output:{name}",
            label=_title(name),
            unit=OUTPUT_UNITS[name],
            kind=_kind_for_any_output(name),
        )
        for name in dataset.output_names
    )
    return inputs + outputs


def build_ensemble_plot_dataset(
    result: SimulationEnsembleResult,
    result_id: str | None = None,
) -> EnsemblePlotDataset:
    """Build the canonical immutable plot-data facade for ``result``."""
    require(
        isinstance(result, SimulationEnsembleResult),
        "result must be a SimulationEnsembleResult",
    )
    stable_id = result_id or (
        f"variation-v{SCHEMA_VERSION}:"
        f"seed-{result.variation.plan.seed}:runs-{result.variation.plan.n_runs}"
    )
    return EnsemblePlotDataset(
        result_id=stable_id,
        result=result,
        variables=scalar_plot_variables(result.variation),
        dispersion=compute_position_dispersion(result.traces),
    )


__all__ = [
    "ArcOverlayData",
    "CohortAvailability",
    "DEFAULT_MAX_ARC_VERTICES",
    "DEFAULT_PLOT_BUDGET",
    "EnsemblePlotDataset",
    "PlotBudget",
    "ScalarPlotVariable",
    "ScalarVariableKind",
    "ScatterPlotData",
    "build_ensemble_plot_dataset",
    "scalar_plot_variables",
]
