"""One-at-a-time sensitivity from complete verified durable substudies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from rate_of_closure.variation.durable_ensemble_chunks import DurableEnsembleArchive
from rate_of_closure.variation.ensemble_source import SimulationEnsembleSource
from rate_of_closure.variation.streaming_ensemble_analysis import (
    analyze_durable_ensemble,
)
from shared.python.contracts import require
from shared.python.swing_sim.variation import (
    NoiseSpec,
    SensitivityResult,
    VariationPlan,
    sensitivity_from_standard_deviations,
)


@dataclass(frozen=True, slots=True)
class DurableSensitivityStudy:
    """One exact single-factor source and its durable archive directory."""

    source: SimulationEnsembleSource
    directory: Path

    def __post_init__(self) -> None:
        require(
            isinstance(self.source, SimulationEnsembleSource),
            "source must be a SimulationEnsembleSource",
        )
        directory = Path(self.directory).resolve()
        require(directory.is_dir(), "sensitivity archive directory must exist")
        object.__setattr__(self, "directory", directory)


@dataclass(frozen=True, slots=True)
class DurableOatSensitivity:
    """Canonical OAT result and ordered durable evidence authorities."""

    result: SensitivityResult
    archives: tuple[DurableEnsembleArchive, ...]

    def __post_init__(self) -> None:
        require(
            len(self.archives) == len(self.result.input_keys),
            "sensitivity archives must match result rows",
        )
        require(
            all(archive.status == "complete" for archive in self.archives),
            "sensitivity cannot promote incomplete archives",
        )


def _require_subplan(
    plan: VariationPlan, spec: NoiseSpec, source: SimulationEnsembleSource
) -> None:
    expected = replace(plan, noise=(spec,), groups=())
    require(
        source.plan.to_json_dict() == expected.to_json_dict(),
        "sensitivity source does not match its registered single-factor plan",
        spec.variable_key,
    )


def analyze_durable_oat_sensitivity(
    plan: VariationPlan, studies: Mapping[str, DurableSensitivityStudy]
) -> DurableOatSensitivity:
    """Build a complete OAT matrix without materializing any substudy."""
    require(isinstance(plan, VariationPlan), "plan must be a VariationPlan")
    expected_keys = tuple(spec.variable_key for spec in plan.noise)
    require(set(studies) == set(expected_keys), "sensitivity studies do not match plan")
    rows: list[np.ndarray] = []
    archives: list[DurableEnsembleArchive] = []
    output_names: tuple[str, ...] | None = None
    for spec in plan.noise:
        study = studies[spec.variable_key]
        require(isinstance(study, DurableSensitivityStudy), "invalid study record")
        _require_subplan(plan, spec, study.source)
        summary = analyze_durable_ensemble(study.source, study.directory)
        require(summary.archive.status == "complete", "sensitivity archive is partial")
        names = tuple(item.name for item in summary.output_moments)
        require(output_names in {None, names}, "sensitivity output axes do not match")
        output_names = names
        rows.append(
            np.asarray(
                [
                    np.nan if item.sample_std is None else item.sample_std
                    for item in summary.output_moments
                ],
                dtype=float,
            )
        )
        archives.append(summary.archive)
    assert output_names is not None
    result = sensitivity_from_standard_deviations(
        expected_keys, output_names, np.vstack(rows)
    )
    return DurableOatSensitivity(result, tuple(archives))


__all__ = [
    "DurableOatSensitivity",
    "DurableSensitivityStudy",
    "analyze_durable_oat_sensitivity",
]
